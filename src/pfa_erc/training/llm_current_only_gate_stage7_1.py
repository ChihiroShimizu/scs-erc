"""Stage 7.1: current-only, label-supervised conditional context routing.

The production path performs one Qwen forward pass over the current utterance.
The same pass yields emotion logits and a tiny gate computed from the LLM's
last hidden state.  Only READ or UNCERTAIN decisions trigger a second forward
pass over the causal recent-three window.  No history text, teacher label,
gold emotion, or future turn is available to the gate.

This module intentionally does not expose a MELD-test command.  Stage 7.1 is a
new validation-only experiment whose preregistered accuracy, mechanism, and
latency gates must all pass before a separate frozen test package is created.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from sklearn.metrics import average_precision_score, f1_score

from pfa_erc.data import read_jsonl
from pfa_erc.training.llm_evidence_supervision_stage7 import (
    LABELS,
    Target,
    VERSION as STAGE7_PARENT_VERSION,
    _annotation_object,
    _apply_chat_template,
    _bootstrap_advantage,
    _fit_temperature,
    _indexed_history,
    _metrics,
    _project_root,
    _read_jsonl,
    _read_yaml,
    _resolve,
    _sha256,
    _softmax,
    _split_dialogues,
    _target_from_serialized,
    _target_map,
    _targets,
    _write_json,
    _write_jsonl,
    validate_teacher_annotation,
)


VERSION = "2.1.1"
EXPERIMENT = "llm_current_only_gate_stage7_1"

GATE_LABELS = ("SKIP", "READ", "UNCERTAIN")
GATE_TO_ID = {label: index for index, label in enumerate(GATE_LABELS)}
TRAIN_VARIANTS = ("emotion_base", "label_gate", "shuffled_gate")
REPORT_VARIANTS = (
    "base_current",
    "base_recent3",
    "entropy_adaptive",
    "label_current",
    "label_recent3",
    "label_adaptive",
    "shuffled_adaptive",
)
VARIANT_LABELS = {
    "base_current": "7.1A shared-base current-only",
    "base_recent3": "7.1B shared-base always recent-3",
    "entropy_adaptive": "7.1C entropy adaptive baseline",
    "label_current": "7.1D label-adapter current-only",
    "label_recent3": "7.1E label-adapter always recent-3",
    "label_adaptive": "7.1F current-only learned gate",
    "shuffled_adaptive": "7.1G shuffled-gate control",
}

CURRENT_SYSTEM = (
    "Classify the target speaker's expressed emotion using only the current "
    "utterance. Return exactly one emotion label."
)
RECENT3_SYSTEM = (
    "Classify the target speaker's expressed emotion using the supplied causal "
    "recent context and current utterance. Return exactly one emotion label."
)


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("llm_current_only_gate_stage7_1")
    if not isinstance(section, dict):
        raise KeyError("Missing llm_current_only_gate_stage7_1 configuration section")
    return section


def _paths(
    root: Path, section: dict[str, Any], run_name: str = "validation"
) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get("artifact_dir", "artifacts/llm_current_only_gate_stage7_1/meld"),
    )
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "data": base / "prepared",
        "adapters": base / "adapters",
        "scores": base / "scores",
        "reports": base / "reports",
        "preflight": base / "reports" / "preflight.json",
        "data_audit": base / "reports" / "training_data_audit.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_validation_decision.json",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
        "manifest": base / "run_manifest.json",
    }


def _adapter_dir(paths: dict[str, Path], seed: int, variant: str) -> Path:
    return paths["adapters"] / f"seed{seed}" / variant


def _render_history(target: Target) -> str:
    history = _indexed_history(target)
    if not history:
        return "[no earlier turns]"
    return "\n".join(
        f"[{turn_id}] {turn.speaker}: {turn.text}" for turn_id, turn in history
    )


def build_current_prompt(target: Target, labels: Sequence[str] = LABELS) -> str:
    """The gate-bearing prompt.  History is deliberately impossible to insert."""
    return (
        f"Target speaker: {target.speaker}\n"
        f"Current utterance: {target.text}\n"
        f"Emotion labels: {', '.join(labels)}\n"
        "Emotion:"
    )


def build_recent3_prompt(target: Target, labels: Sequence[str] = LABELS) -> str:
    return (
        f"Earlier causal turns:\n{_render_history(target)}\n\n"
        f"Target speaker: {target.speaker}\n"
        f"Current utterance: {target.text}\n"
        f"Emotion labels: {', '.join(labels)}\n"
        "Emotion:"
    )


def context_need_to_gate(value: str) -> str:
    if value == "current_sufficient":
        return "SKIP"
    if value in {"history_helpful", "history_required"}:
        return "READ"
    if value == "ambiguous":
        return "UNCERTAIN"
    raise ValueError(f"Unknown context_need label: {value}")


def deployment_reads(gate_label: str, history_available: bool) -> bool:
    """UNCERTAIN conservatively reads history; empty histories always short-circuit."""
    return bool(history_available and gate_label != "SKIP")


def _serialize_target(target: Target) -> dict[str, Any]:
    return {
        "key": target.key,
        "dialogue_id": target.dialogue_id,
        "turn_index": target.turn_index,
        "utterance_id": target.utterance_id,
        "speaker": target.speaker,
        "text": target.text,
        "gold": target.gold,
        "history": [
            {
                "turn_index": turn_id,
                "utterance_id": turn.utterance_id,
                "speaker": turn.speaker,
                "text": turn.text,
            }
            for turn_id, turn in _indexed_history(target)
        ],
    }


def _lm_record(
    system: str, prompt: str, completion: str, **meta: Any
) -> dict[str, Any]:
    return {
        "task": "lm",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "completion": completion,
        "meta": meta,
    }


def _gate_record(target: Target, gate_label: str, **meta: Any) -> dict[str, Any]:
    prompt = build_current_prompt(target)
    return {
        "task": "gate",
        "messages": [
            {"role": "system", "content": CURRENT_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "gate_label": gate_label,
        "meta": {"key": target.key, "dialogue_id": target.dialogue_id, **meta},
    }


def _deranged_gate_labels(
    records: Sequence[dict[str, Any]], seed: int
) -> tuple[list[str], float]:
    """Permute labels across examples while preserving the exact marginal counts."""
    original = [str(item["gate_label"]) for item in records]
    if len(original) < 2:
        return list(original), 0.0
    rng = random.Random(seed)
    best = list(original)
    best_changed = -1
    for _ in range(1000):
        candidate = list(original)
        rng.shuffle(candidate)
        changed = sum(left != right for left, right in zip(original, candidate))
        if changed > best_changed:
            best = candidate
            best_changed = changed
        if changed == len(original):
            break
    return best, best_changed / len(original)


def preflight(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if str(section.get("dataset", "meld")).lower() != "meld":
        raise ValueError("Stage 7.1 accepts MELD only")
    if STAGE7_PARENT_VERSION != "2.0.0":
        raise ValueError(
            f"Stage 7.1 requires Stage 7 v2.0.0, found {STAGE7_PARENT_VERSION}"
        )
    if section.get("test_locked") is not True:
        raise ValueError("Stage 7.1 requires test_locked: true")
    processed = _resolve(root, section["processed_dir"])
    train_path = processed / "train.jsonl"
    validation_path = processed / "validation.jsonl"
    teacher_path = _resolve(root, section["teacher_annotation_file"])
    missing = [
        str(path)
        for path in (train_path, validation_path, teacher_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing Stage 7.1 input(s): " + ", ".join(missing))
    if "test" in str(teacher_path).lower():
        raise ValueError("Teacher path must not reference MELD test")
    raw_teacher = _read_jsonl(teacher_path)
    expected_count = int(section.get("expected_teacher_records", 500))
    if len(raw_teacher) != expected_count:
        raise ValueError(
            f"Expected {expected_count} teacher records, found {len(raw_teacher)}"
        )
    actual_hash = _sha256(teacher_path)
    expected_hash = str(section.get("expected_teacher_sha256", "")).strip().lower()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Teacher SHA256 mismatch: expected {expected_hash}, found {actual_hash}"
        )
    train_dialogues = read_jsonl(train_path)
    validation_dialogues = read_jsonl(validation_path)
    mapping = _target_map(train_dialogues, max_history=3)
    annotations: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for raw in raw_teacher:
        annotation = _annotation_object(raw)
        key = f"{annotation.get('dialogue_id')}::{int(annotation.get('target_turn_id', -1))}"
        if key in annotations:
            issues.append(f"duplicate teacher key {key}")
            continue
        annotations[key] = annotation
        target = mapping.get(key)
        if target is None:
            issues.append(f"teacher key absent from MELD train: {key}")
        else:
            issues.extend(validate_teacher_annotation(annotation, target))
    if issues:
        raise ValueError(
            f"Teacher annotation audit failed ({len(issues)} issues):\n - "
            + "\n - ".join(issues[:20])
        )
    result = {
        "status": "ready",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "model_family": "causal_llm_with_internal_gate_head",
        "model_name": section["model_name"],
        "roberta_or_bert_used": False,
        "gate_input": "current_utterance_only",
        "history_visible_to_gate": False,
        "gate_uses_environment_features": False,
        "teacher_source": "GPT-5.6 Sol generated and self-audited",
        "teacher_records": len(annotations),
        "teacher_sha256": actual_hash,
        "train_dialogues": len(train_dialogues),
        "validation_dialogues": len(validation_dialogues),
        "context_fields_used_in_stage7_1": ["context_need"],
        "candidate_and_impact_fields_reserved_for_later_stage": True,
        "test_locked": True,
    }
    _write_json(paths["preflight"], result)
    _write_json(
        paths["manifest"],
        {
            "experiment": EXPERIMENT,
            "version": VERSION,
            "config": str(config_path.expanduser().resolve()),
            "config_sha256": _sha256(config_path.expanduser().resolve()),
            "implementation": str(Path(__file__).resolve()),
            "implementation_sha256": _sha256(Path(__file__).resolve()),
            "teacher_sha256": actual_hash,
            "seeds": list(section.get("seeds", [42, 43, 44])),
            "test_locked": True,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def prepare_training_data(
    config_path: Path, run_name: str = "validation"
) -> dict[str, Any]:
    preflight(config_path, run_name)
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    processed = _resolve(root, section["processed_dir"])
    train_dialogues = read_jsonl(processed / "train.jsonl")
    target_map = _target_map(train_dialogues, max_history=3)
    annotations = [
        _annotation_object(item)
        for item in _read_jsonl(_resolve(root, section["teacher_annotation_file"]))
    ]
    teacher_by_key = {
        f"{item['dialogue_id']}::{int(item['target_turn_id'])}": item
        for item in annotations
    }
    train_ids, calibration_ids = _split_dialogues(
        [dialogue.dialogue_id for dialogue in train_dialogues],
        float(section.get("calibration_dialogue_fraction", 0.20)),
        int(section.get("split_seed", 2171)),
    )
    labels = list(section.get("labels", LABELS))
    base_examples: list[dict[str, Any]] = []
    calibration_targets: list[dict[str, Any]] = []
    for target in target_map.values():
        if target.dialogue_id in train_ids:
            base_examples.append(
                _lm_record(
                    CURRENT_SYSTEM,
                    build_current_prompt(target, labels),
                    target.gold,
                    prompt_mode="current",
                    key=target.key,
                    dialogue_id=target.dialogue_id,
                )
            )
            if target.history:
                base_examples.append(
                    _lm_record(
                        RECENT3_SYSTEM,
                        build_recent3_prompt(target, labels),
                        target.gold,
                        prompt_mode="recent3",
                        key=target.key,
                        dialogue_id=target.dialogue_id,
                    )
                )
        else:
            calibration_targets.append(_serialize_target(target))

    teacher_train = [
        annotation
        for key, annotation in teacher_by_key.items()
        if target_map[key].dialogue_id in train_ids
    ]
    teacher_calibration = [
        annotation
        for key, annotation in teacher_by_key.items()
        if target_map[key].dialogue_id in calibration_ids
    ]
    correct_gate_records = [
        _gate_record(
            target_map[
                f"{annotation['dialogue_id']}::{int(annotation['target_turn_id'])}"
            ],
            context_need_to_gate(str(annotation["context_need"])),
            context_need=annotation["context_need"],
        )
        for annotation in teacher_train
    ]
    shuffled_labels, changed_rate = _deranged_gate_labels(
        correct_gate_records, int(section.get("shuffle_control_seed", 2277))
    )
    shuffled_gate_records = []
    for record, shuffled_label in zip(correct_gate_records, shuffled_labels):
        copy = json.loads(json.dumps(record))
        copy["gate_label"] = shuffled_label
        copy["meta"]["shuffled_control"] = True
        shuffled_gate_records.append(copy)

    gate_repeats = int(section.get("gate_repeats", 4))
    correct_repeated = [
        json.loads(json.dumps(item))
        for item in correct_gate_records
        for _ in range(gate_repeats)
    ]
    shuffled_repeated = [
        json.loads(json.dumps(item))
        for item in shuffled_gate_records
        for _ in range(gate_repeats)
    ]
    replay_count = len(correct_repeated) * int(
        section.get("emotion_replay_per_gate", 1)
    )
    replay_pool = [json.loads(json.dumps(item)) for item in base_examples]
    random.Random(int(section.get("split_seed", 2171)) + 91).shuffle(replay_pool)
    replay = [
        json.loads(json.dumps(replay_pool[index % len(replay_pool)]))
        for index in range(replay_count)
    ]
    for item in replay:
        item["meta"]["branch_replay"] = True
    label_branch = correct_repeated + [json.loads(json.dumps(item)) for item in replay]
    shuffled_branch = shuffled_repeated + [
        json.loads(json.dumps(item)) for item in replay
    ]
    random.Random(int(section.get("split_seed", 2171)) + 101).shuffle(label_branch)
    random.Random(int(section.get("split_seed", 2171)) + 101).shuffle(shuffled_branch)

    _write_jsonl(paths["data"] / "emotion_base_train.jsonl", base_examples)
    _write_jsonl(paths["data"] / "label_gate_train.jsonl", label_branch)
    _write_jsonl(paths["data"] / "shuffled_gate_train.jsonl", shuffled_branch)
    _write_jsonl(paths["data"] / "calibration_targets.jsonl", calibration_targets)
    _write_jsonl(paths["data"] / "teacher_train_annotations.jsonl", teacher_train)
    _write_jsonl(
        paths["data"] / "teacher_calibration_annotations.jsonl", teacher_calibration
    )

    gate_distribution = Counter(item["gate_label"] for item in correct_gate_records)
    context_distribution = Counter(str(item["context_need"]) for item in teacher_train)
    audit = {
        "status": "passed",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "split_policy": "fixed dialogue-disjoint MELD-train/model-calibration split",
        "split_seed": int(section.get("split_seed", 2171)),
        "calibration_dialogue_fraction": float(
            section.get("calibration_dialogue_fraction", 0.20)
        ),
        "model_train_dialogues": len(train_ids),
        "calibration_dialogues": len(calibration_ids),
        "base_emotion_examples": len(base_examples),
        "calibration_targets": len(calibration_targets),
        "teacher_train_records": len(teacher_train),
        "teacher_calibration_records": len(teacher_calibration),
        "gate_distribution_train": dict(gate_distribution),
        "context_need_distribution_train": dict(context_distribution),
        "gate_repeats": gate_repeats,
        "branch_examples": {
            "label_gate": len(label_branch),
            "shuffled_gate": len(shuffled_branch),
        },
        "shuffle_changed_label_rate": changed_rate,
        "shuffle_preserves_marginal_distribution": Counter(shuffled_labels)
        == Counter(item["gate_label"] for item in correct_gate_records),
        "gate_prompt_contains_history": False,
        "gate_prompt_contains_gold_emotion": False,
        "teacher_labels_hidden_at_validation_inference": True,
        "test_locked": True,
    }
    _write_json(paths["data_audit"], audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


class CurrentOnlyGateHead(nn.Module):
    """A negligible-cost head whose gradients also update the Qwen LoRA."""

    def __init__(self, hidden_size: int, dropout: float = 0.10):
        super().__init__()
        self.dropout = nn.Dropout(float(dropout))
        self.projection = nn.Linear(int(hidden_size), len(GATE_LABELS))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.dropout(hidden))


class AdaptiveTrainingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        enable_thinking: bool = False,
    ):
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.enable_thinking = bool(enable_thinking)

    def __len__(self) -> int:
        return len(self.records)

    def _prompt_ids(self, messages: Sequence[dict[str, str]]) -> list[int]:
        if hasattr(self.tokenizer, "apply_chat_template"):
            return list(
                _apply_chat_template(
                    self.tokenizer,
                    list(messages),
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )
        text = "\n".join(f"{item['role']}: {item['content']}" for item in messages)
        return list(
            self.tokenizer.encode(text + "\nassistant:", add_special_tokens=True)
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        prompt_ids = self._prompt_ids(record["messages"])
        gate_label = -1
        if record["task"] == "gate":
            prompt_ids = prompt_ids[-self.max_length :]
            input_ids = prompt_ids
            labels = [-100] * len(input_ids)
            gate_label = GATE_TO_ID[str(record["gate_label"])]
        elif record["task"] == "lm":
            completion_ids = list(
                self.tokenizer.encode(
                    str(record["completion"]), add_special_tokens=False
                )
            )
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                completion_ids.append(int(eos_id))
            if len(completion_ids) >= self.max_length:
                raise ValueError("Completion exceeds max sequence length")
            prompt_ids = prompt_ids[-(self.max_length - len(completion_ids)) :]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids
        else:
            raise ValueError(f"Unknown Stage 7.1 training task: {record['task']}")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "gate_label": torch.tensor(gate_label, dtype=torch.long),
        }


def _collate_adaptive(
    batch: Sequence[dict[str, torch.Tensor]], pad_id: int
) -> dict[str, torch.Tensor]:
    length = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), length), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(batch), length), dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    gate_labels = torch.full((len(batch),), -1, dtype=torch.long)
    for row, item in enumerate(batch):
        size = int(item["input_ids"].numel())
        input_ids[row, :size] = item["input_ids"]
        attention[row, :size] = item["attention_mask"]
        labels[row, :size] = item["labels"]
        gate_labels[row] = item["gate_label"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "labels": labels,
        "gate_labels": gate_labels,
    }


def _load_tokenizer(section: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        section["model_name"],
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        local_files_only=bool(section.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    return tokenizer


def _load_trainable_model(
    section: dict[str, Any], parent_adapter: Path | None
) -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError(
            "Stage 7.1 requires transformers, accelerate and peft; run the setup script"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7.1 training requires CUDA")
    tokenizer = _load_tokenizer(section)
    dtype = (
        torch.bfloat16
        if str(section.get("dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        section["model_name"],
        dtype=dtype,
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        local_files_only=bool(section.get("local_files_only", False)),
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    if bool(section.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if parent_adapter is None:
        lora = LoraConfig(
            r=int(section.get("lora_r", 16)),
            lora_alpha=int(section.get("lora_alpha", 32)),
            lora_dropout=float(section.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=list(
                section.get(
                    "lora_target_modules",
                    [
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                )
            ),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    else:
        if not (parent_adapter / "training_complete.json").exists():
            raise FileNotFoundError(f"Parent adapter is incomplete: {parent_adapter}")
        model = PeftModel.from_pretrained(model, str(parent_adapter), is_trainable=True)
    model.to("cuda")
    return model, tokenizer


def _causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    if not bool(valid.any()):
        return None
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _gate_class_weights(records: Sequence[dict[str, Any]]) -> torch.Tensor:
    counts = Counter(
        str(item["gate_label"]) for item in records if item.get("task") == "gate"
    )
    raw = np.asarray(
        [1.0 / math.sqrt(max(counts.get(label, 0), 1)) for label in GATE_LABELS],
        dtype=np.float64,
    )
    raw /= raw.mean()
    return torch.tensor(raw, dtype=torch.float32, device="cuda")


def train_variant(
    config_path: Path,
    variant: str,
    seed: int,
    run_name: str = "validation",
    *,
    max_steps_override: int | None = None,
    max_examples: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if variant not in TRAIN_VARIANTS:
        raise ValueError(f"Unknown variant {variant}; choose from {TRAIN_VARIANTS}")
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if not paths["data_audit"].exists():
        prepare_training_data(config_path, run_name)
    output_dir = _adapter_dir(paths, seed, variant)
    done = output_dir / "training_complete.json"
    if done.exists() and not overwrite:
        result = json.loads(done.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if output_dir.exists() and overwrite:
        backup = output_dir.with_name(
            output_dir.name + ".backup_" + time.strftime("%Y%m%dT%H%M%S")
        )
        shutil.move(str(output_dir), str(backup))
    records = _read_jsonl(paths["data"] / f"{variant}_train.jsonl")
    if max_examples is not None:
        records = records[:max_examples]
    if not records:
        raise ValueError(f"No training examples for {variant}")
    parent = (
        None if variant == "emotion_base" else _adapter_dir(paths, seed, "emotion_base")
    )
    if parent is not None and not (parent / "training_complete.json").exists():
        raise FileNotFoundError(f"Train emotion_base seed={seed} before {variant}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, tokenizer = _load_trainable_model(section, parent)
    gate_head: CurrentOnlyGateHead | None = None
    if variant != "emotion_base":
        gate_head = CurrentOnlyGateHead(
            int(model.config.hidden_size), float(section.get("gate_dropout", 0.10))
        ).to("cuda")
    dataset = AdaptiveTrainingDataset(
        records,
        tokenizer,
        int(section.get("max_sequence_tokens", 768)),
        enable_thinking=bool(section.get("enable_thinking", False)),
    )
    micro_batch = int(section.get("micro_batch_size", 1))
    pad_id = int(tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: _collate_adaptive(batch, pad_id),
    )
    if variant == "emotion_base":
        default_steps = int(section.get("emotion_base_max_steps", 700))
        learning_rate = float(section.get("emotion_base_learning_rate", 2.0e-5))
    else:
        default_steps = int(section.get("gate_branch_max_steps", 240))
        learning_rate = float(section.get("gate_branch_learning_rate", 1.0e-5))
    max_steps = int(max_steps_override or default_steps)
    accumulation = int(section.get("gradient_accumulation_steps", 16))
    warmup = min(int(section.get("warmup_steps", 20)), max(0, max_steps - 1))
    from transformers import get_linear_schedule_with_warmup

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if gate_head is not None:
        parameters.extend(
            parameter for parameter in gate_head.parameters() if parameter.requires_grad
        )
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(section.get("weight_decay", 0.01)),
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max_steps)
    class_weights = _gate_class_weights(records) if gate_head is not None else None
    gate_loss_weight = float(section.get("gate_loss_weight", 1.0))
    model.train()
    if gate_head is not None:
        gate_head.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    losses: list[float] = []
    lm_losses: list[float] = []
    gate_losses: list[float] = []
    started = time.time()
    while step < max_steps:
        for batch in loader:
            batch = {
                key: value.to("cuda", non_blocking=True) for key, value in batch.items()
            }
            gate_mask = batch["gate_labels"].ge(0)
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=bool(gate_mask.any()),
                use_cache=False,
            )
            lm_loss = _causal_lm_loss(output.logits, batch["labels"])
            gate_loss: torch.Tensor | None = None
            if bool(gate_mask.any()):
                if gate_head is None:
                    raise RuntimeError("Gate examples reached emotion_base training")
                last_positions = batch["attention_mask"].sum(dim=1) - 1
                rows = torch.arange(len(last_positions), device="cuda")
                last_hidden = output.hidden_states[-1][rows, last_positions]
                gate_logits = gate_head(last_hidden[gate_mask].float())
                gate_loss = F.cross_entropy(
                    gate_logits,
                    batch["gate_labels"][gate_mask],
                    weight=class_weights,
                )
            components = []
            if lm_loss is not None:
                components.append(lm_loss)
                lm_losses.append(float(lm_loss.detach().cpu()))
            if gate_loss is not None:
                components.append(gate_loss_weight * gate_loss)
                gate_losses.append(float(gate_loss.detach().cpu()))
            if not components:
                raise RuntimeError("Batch contained neither LM nor gate supervision")
            loss = sum(components) / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Stage 7.1 loss variant={variant} seed={seed} step={step}"
                )
            loss.backward()
            losses.append(float(loss.detach().cpu()) * accumulation)
            micro_step += 1
            if micro_step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(section.get("max_grad_norm", 1.0))
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % int(section.get("log_every_steps", 10)) == 0:
                    recent = float(
                        np.mean(losses[-min(len(losses), accumulation * 10) :])
                    )
                    print(
                        f"[Stage7.1] {variant} seed={seed} step={step}/{max_steps} "
                        f"loss={recent:.5f} lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if step >= max_steps:
                    break
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    gate_trainable = 0
    if gate_head is not None:
        state = {
            key: value.detach().cpu() for key, value in gate_head.state_dict().items()
        }
        save_file(state, str(output_dir / "gate_head.safetensors"))
        gate_trainable = sum(value.numel() for value in gate_head.parameters())
        _write_json(
            output_dir / "gate_head_config.json",
            {
                "hidden_size": int(model.config.hidden_size),
                "classes": list(GATE_LABELS),
                "dropout": float(section.get("gate_dropout", 0.10)),
                "deployment_policy": "READ and UNCERTAIN load recent-3; SKIP does not",
            },
        )
    model_trainable = sum(
        value.numel() for value in model.parameters() if value.requires_grad
    )
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "variant": variant,
        "seed": seed,
        "base_model": section["model_name"],
        "parent_adapter": str(parent) if parent else None,
        "examples": len(records),
        "optimizer_steps": step,
        "gradient_accumulation_steps": accumulation,
        "mean_training_loss": float(np.mean(losses)),
        "mean_lm_loss": float(np.mean(lm_losses)) if lm_losses else None,
        "mean_gate_loss": float(np.mean(gate_losses)) if gate_losses else None,
        "model_trainable_parameters": int(model_trainable),
        "gate_trainable_parameters": int(gate_trainable),
        "total_trainable_parameters": int(model_trainable + gate_trainable),
        "elapsed_seconds": time.time() - started,
        "finite": bool(np.isfinite(losses).all()),
        "gate_input_history_tokens": 0,
        "test_data_read": False,
    }
    _write_json(done, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, tokenizer, optimizer, scheduler, gate_head
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_all(config_path: Path, run_name: str, seeds: Sequence[int]) -> None:
    for seed in seeds:
        for variant in TRAIN_VARIANTS:
            train_variant(config_path, variant, int(seed), run_name)


@dataclass
class ForwardResult:
    scores: list[list[float]]
    gate_probabilities: list[list[float]] | None
    token_counts: list[int]
    elapsed_seconds: float


def _emotion_verbalizer_token_ids(
    tokenizer: Any,
    labels: Sequence[str],
    model_name: str,
) -> tuple[list[int], list[list[int]]]:
    """Return unique first-token verbalizers for one-forward classification.

    Qwen tokenizes several MELD label words into multiple tokens. The SFT loss
    nevertheless supervises the first completion token for every example. If
    those first tokens are unique, their seven logits form a one-forward class
    verbalizer without generating or scoring the remaining tokens.
    """
    encoded = [
        list(map(int, tokenizer.encode(label, add_special_tokens=False)))
        for label in labels
    ]
    empty = [label for label, token_ids in zip(labels, encoded) if not token_ids]
    if empty:
        raise ValueError(
            f"Emotion labels produced empty tokenizations for {model_name}: {empty}"
        )
    first_tokens = [token_ids[0] for token_ids in encoded]
    collisions = {
        token_id: [
            label
            for label, candidate in zip(labels, first_tokens)
            if candidate == token_id
        ]
        for token_id in set(first_tokens)
        if first_tokens.count(token_id) > 1
    }
    if collisions:
        raise ValueError(
            "Stage 7.1's one-forward production path requires unique first-token "
            f"emotion verbalizers for {model_name}; collisions={collisions}, "
            f"tokenizations={encoded}"
        )
    return first_tokens, encoded


class AdaptiveLLMEngine:
    def __init__(self, section: dict[str, Any], adapter: Path, with_gate: bool):
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and peft for Stage 7.1 inference"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 7.1 validation requires CUDA")
        self.section = section
        self.labels = list(section.get("labels", LABELS))
        self.tokenizer = _load_tokenizer(section)
        dtype = (
            torch.bfloat16
            if str(section.get("dtype", "bfloat16")) == "bfloat16"
            else torch.float16
        )
        base = AutoModelForCausalLM.from_pretrained(
            section["model_name"],
            dtype=dtype,
            trust_remote_code=bool(section.get("trust_remote_code", False)),
            local_files_only=bool(section.get("local_files_only", False)),
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.to("cuda")
        self.model.eval()
        self.model.config.use_cache = False
        self.gate_head: CurrentOnlyGateHead | None = None
        if with_gate:
            config_path = adapter / "gate_head_config.json"
            state_path = adapter / "gate_head.safetensors"
            if not config_path.exists() or not state_path.exists():
                raise FileNotFoundError(f"Missing gate-head files in {adapter}")
            gate_config = json.loads(config_path.read_text(encoding="utf-8"))
            if gate_config.get("classes") != list(GATE_LABELS):
                raise ValueError(f"Gate class mismatch in {config_path}")
            self.gate_head = CurrentOnlyGateHead(
                int(gate_config["hidden_size"]), float(gate_config.get("dropout", 0.10))
            ).to("cuda")
            self.gate_head.load_state_dict(load_file(str(state_path), device="cuda"))
            self.gate_head.eval()
        self.batch_size = int(section.get("inference_batch_size", 2))
        self.max_input_tokens = int(section.get("max_sequence_tokens", 768))
        first_tokens, encoded_labels = _emotion_verbalizer_token_ids(
            self.tokenizer,
            self.labels,
            str(section["model_name"]),
        )
        self.label_token_sequences = {
            label: token_ids for label, token_ids in zip(self.labels, encoded_labels)
        }
        self.label_token_ids = torch.tensor(
            first_tokens, dtype=torch.long, device="cuda"
        )

    def _render(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return str(
                _apply_chat_template(
                    self.tokenizer,
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(self.section.get("enable_thinking", False)),
                )
            )
        return (
            "\n".join(f"{item['role']}: {item['content']}" for item in messages)
            + "\nassistant:"
        )

    def prompts(self, targets: Sequence[Target], mode: str) -> list[str]:
        if mode == "current":
            return [
                self._render(CURRENT_SYSTEM, build_current_prompt(target, self.labels))
                for target in targets
            ]
        if mode == "recent3":
            return [
                self._render(RECENT3_SYSTEM, build_recent3_prompt(target, self.labels))
                for target in targets
            ]
        raise ValueError(f"Unknown prompt mode: {mode}")

    def forward_prompts(
        self,
        prompts: Sequence[str],
        *,
        need_gate: bool,
        batch_size: int | None = None,
        synchronize: bool = True,
    ) -> ForwardResult:
        if need_gate and self.gate_head is None:
            raise RuntimeError(
                "Gate probabilities requested from an adapter without a gate head"
            )
        scores: list[list[float]] = []
        gates: list[list[float]] = []
        token_counts: list[int] = []
        effective_batch = int(batch_size or self.batch_size)
        if synchronize:
            torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(prompts), effective_batch):
            encoded = self.tokenizer(
                list(prompts[start : start + effective_batch]),
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            token_counts.extend(
                int(value) for value in encoded["attention_mask"].sum(dim=1)
            )
            encoded = {
                key: value.to("cuda", non_blocking=True)
                for key, value in encoded.items()
            }
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(last_positions), device="cuda")
            with torch.inference_mode():
                output = self.model(
                    **encoded,
                    output_hidden_states=need_gate,
                    use_cache=False,
                )
                selected = output.logits[rows, last_positions].index_select(
                    -1, self.label_token_ids
                )
                scores.extend(selected.float().cpu().tolist())
                if need_gate:
                    hidden = output.hidden_states[-1][rows, last_positions]
                    probability = torch.softmax(self.gate_head(hidden.float()), dim=-1)
                    gates.extend(probability.cpu().tolist())
            del encoded, output, selected
        if synchronize:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        return ForwardResult(
            scores=scores,
            gate_probabilities=gates if need_gate else None,
            token_counts=token_counts,
            elapsed_seconds=elapsed,
        )

    def forward_targets(
        self,
        targets: Sequence[Target],
        mode: str,
        *,
        need_gate: bool,
        batch_size: int | None = None,
    ) -> ForwardResult:
        return self.forward_prompts(
            self.prompts(targets, mode),
            need_gate=need_gate,
            batch_size=batch_size,
        )


def _load_inference_targets(
    root: Path, section: dict[str, Any], paths: dict[str, Path]
) -> tuple[list[Target], list[Target]]:
    calibration = [
        _target_from_serialized(item)
        for item in _read_jsonl(paths["data"] / "calibration_targets.jsonl")
    ]
    validation_dialogues = read_jsonl(
        _resolve(root, section["processed_dir"]) / "validation.jsonl"
    )
    validation = _targets(validation_dialogues, max_history=3)
    return calibration, validation


def _execute_conditional_recent3(
    engine: AdaptiveLLMEngine,
    targets: Sequence[Target],
    current_scores: Sequence[Sequence[float]],
    read_mask: Sequence[bool],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Run recent-3 only for rows routed after the current-only forward pass."""
    if not (len(targets) == len(current_scores) == len(read_mask)):
        raise ValueError("Conditional recent-3 inputs are not aligned")
    selected_indices = [index for index, read in enumerate(read_mask) if read]
    selected_targets = [targets[index] for index in selected_indices]
    merged = [list(map(float, row)) for row in current_scores]
    selected_tokens = 0
    selected_seconds = 0.0
    if selected_targets:
        recent = engine.forward_targets(
            selected_targets,
            "recent3",
            need_gate=False,
        )
        if len(recent.scores) != len(selected_indices):
            raise ValueError("Conditional recent-3 output count is misaligned")
        for index, score in zip(selected_indices, recent.scores):
            merged[index] = list(map(float, score))
        selected_tokens = int(sum(recent.token_counts))
        selected_seconds = float(recent.elapsed_seconds)
    return merged, {
        "selected_targets": len(selected_indices),
        "selected_input_tokens": selected_tokens,
        "selected_forward_seconds": selected_seconds,
        "recent3_called_for_unselected_targets": False,
    }


def _mix_scores(
    current_scores: Sequence[Sequence[float]],
    recent_scores: Sequence[Sequence[float]],
    read_mask: Sequence[bool],
) -> list[list[float]]:
    if not (len(current_scores) == len(recent_scores) == len(read_mask)):
        raise ValueError("Adaptive score inputs are not aligned")
    return [
        list(map(float, recent if use_recent else current))
        for current, recent, use_recent in zip(current_scores, recent_scores, read_mask)
    ]


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _select_gate_threshold(
    p_skip: Sequence[float],
    context_need: Sequence[str],
    section: dict[str, Any],
) -> dict[str, Any]:
    probabilities = np.asarray(p_skip, dtype=np.float64)
    labels = list(context_need)
    required = np.asarray([value == "history_required" for value in labels])
    useful = np.asarray(
        [value in {"history_helpful", "history_required"} for value in labels]
    )
    current = np.asarray([value == "current_sufficient" for value in labels])
    minimum_required = float(section.get("threshold_minimum_required_recall", 0.90))
    minimum_useful = float(section.get("threshold_minimum_useful_recall", 0.80))
    minimum_skip = float(
        section.get("threshold_minimum_current_sufficient_skip_rate", 0.20)
    )
    candidates = sorted(set([0.0, 1.000001] + probabilities.tolist()))
    eligible: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for threshold in candidates:
        skip = probabilities >= threshold
        read = ~skip
        row = {
            "threshold": float(threshold),
            "required_recall": float(read[required].mean()) if required.any() else 1.0,
            "useful_recall": float(read[useful].mean()) if useful.any() else 1.0,
            "current_skip_rate": float(skip[current].mean()) if current.any() else 0.0,
            "context_use_rate": float(read.mean()),
        }
        all_rows.append(row)
        if (
            row["required_recall"] >= minimum_required
            and row["useful_recall"] >= minimum_useful
            and row["current_skip_rate"] >= minimum_skip
        ):
            eligible.append(row)
    if eligible:
        chosen = max(
            eligible,
            key=lambda item: (
                item["current_skip_rate"],
                item["required_recall"],
                item["useful_recall"],
                -item["context_use_rate"],
            ),
        )
        constraint_met = True
    else:
        # Fail closed: an unqualified gate may not skip context merely to look fast.
        chosen = next(item for item in all_rows if item["threshold"] > 1.0)
        constraint_met = False
    return {
        **chosen,
        "constraint_met": constraint_met,
        "selection_rule": (
            "maximize current_sufficient skip rate subject to preregistered "
            "required/useful recall constraints"
        ),
    }


def _gate_diagnostics(
    targets: Sequence[Target],
    gate_probabilities: Sequence[Sequence[float]],
    teacher_annotations: Sequence[dict[str, Any]],
    section: dict[str, Any],
) -> dict[str, Any]:
    lookup = {target.key: index for index, target in enumerate(targets)}
    rows: list[tuple[Target, list[float], str]] = []
    for annotation in teacher_annotations:
        key = f"{annotation['dialogue_id']}::{int(annotation['target_turn_id'])}"
        index = lookup.get(key)
        if index is None:
            continue
        target = targets[index]
        if not target.history:
            continue
        rows.append(
            (
                target,
                list(map(float, gate_probabilities[index])),
                str(annotation["context_need"]),
            )
        )
    if not rows:
        raise ValueError(
            "No history-bearing teacher calibration records for gate diagnostics"
        )
    probabilities = np.asarray([item[1] for item in rows], dtype=np.float64)
    context_need = [item[2] for item in rows]
    gold_gate = np.asarray(
        [GATE_TO_ID[context_need_to_gate(value)] for value in context_need]
    )
    predicted_gate = probabilities.argmax(axis=1)
    p_skip = probabilities[:, GATE_TO_ID["SKIP"]]
    gold_useful = np.asarray(
        [value in {"history_helpful", "history_required"} for value in context_need],
        dtype=np.int64,
    )
    context_ap = (
        float(average_precision_score(gold_useful, 1.0 - p_skip))
        if len(set(gold_useful.tolist())) > 1
        else None
    )
    threshold = _select_gate_threshold(p_skip, context_need, section)
    skip = p_skip >= float(threshold["threshold"])
    read = ~skip
    required = np.asarray([value == "history_required" for value in context_need])
    useful = gold_useful.astype(bool)
    current = np.asarray([value == "current_sufficient" for value in context_need])
    ambiguous = np.asarray([value == "ambiguous" for value in context_need])
    return {
        "records": len(rows),
        "context_positive_prevalence": float(gold_useful.mean()),
        "context_average_precision": context_ap,
        "context_ap_lift_over_prevalence": (
            float(context_ap - gold_useful.mean()) if context_ap is not None else None
        ),
        "gate_three_way_macro_f1": float(
            f1_score(gold_gate, predicted_gate, average="macro", zero_division=0)
        ),
        "threshold": float(threshold["threshold"]),
        "threshold_constraints_met": bool(threshold["constraint_met"]),
        "required_recall": float(read[required].mean()) if required.any() else 1.0,
        "helpful_required_recall": float(read[useful].mean()) if useful.any() else 1.0,
        "current_sufficient_skip_rate": float(skip[current].mean())
        if current.any()
        else 0.0,
        "ambiguous_read_rate": float(read[ambiguous].mean())
        if ambiguous.any()
        else 0.0,
        "calibration_context_use_rate": float(read.mean()),
        "context_need_distribution": dict(Counter(context_need)),
        "predicted_gate_distribution": dict(
            Counter(GATE_LABELS[index] for index in predicted_gate.tolist())
        ),
        "gate_input_history_tokens": 0,
    }


def _select_entropy_threshold(
    current_scores: Sequence[Sequence[float]],
    recent_scores: Sequence[Sequence[float]],
    gold_labels: Sequence[str],
    labels: Sequence[str],
    current_temperature: float,
) -> dict[str, Any]:
    entropies = np.asarray(
        [
            -float(np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))))
            for probability in (
                _softmax(row, current_temperature) for row in current_scores
            )
        ],
        dtype=np.float64,
    )
    thresholds = np.unique(np.quantile(entropies, np.linspace(0.0, 1.0, 101)))
    best: dict[str, Any] | None = None
    for threshold in thresholds:
        mask = entropies >= threshold
        scores = _mix_scores(current_scores, recent_scores, mask.tolist())
        metrics = _metrics(scores, gold_labels, labels, 1.0)
        objective = 0.5 * metrics["weighted_f1"] + 0.5 * metrics["macro_f1"]
        candidate = {
            "threshold": float(threshold),
            "objective": float(objective),
            "context_use_rate": float(mask.mean()),
        }
        if best is None or (candidate["objective"], -candidate["context_use_rate"]) > (
            best["objective"],
            -best["context_use_rate"],
        ):
            best = candidate
    if best is None:
        raise RuntimeError("Entropy threshold search produced no candidate")
    return best


def _benchmark_engine(
    engine: AdaptiveLLMEngine,
    targets: Sequence[Target],
    gate_threshold: float,
    section: dict[str, Any],
) -> dict[str, Any]:
    max_targets = int(section.get("benchmark_max_targets", 0))
    if 0 < max_targets < len(targets):
        indexes = np.linspace(0, len(targets) - 1, max_targets).round().astype(int)
        benchmark_targets = [targets[int(index)] for index in indexes]
        sampling = "fixed evenly-spaced validation targets"
    else:
        benchmark_targets = list(targets)
        sampling = "all validation targets"
    current_prompts = engine.prompts(benchmark_targets, "current")
    recent_prompts = engine.prompts(benchmark_targets, "recent3")
    history_available = np.asarray(
        [bool(target.history) for target in benchmark_targets]
    )
    warmup_examples = min(
        len(benchmark_targets), int(section.get("benchmark_warmup_examples", 8))
    )
    if warmup_examples:
        engine.forward_prompts(
            current_prompts[:warmup_examples], need_gate=True, batch_size=1
        )
        engine.forward_prompts(
            recent_prompts[:warmup_examples], need_gate=False, batch_size=1
        )
    repeats = int(section.get("benchmark_repeats", 3))
    output: dict[str, Any] = {}
    for batch_size in [
        int(value) for value in section.get("benchmark_batch_sizes", [1, 4])
    ]:
        timings: dict[str, list[float]] = {
            "current_only": [],
            "always_recent3": [],
            "adaptive": [],
        }
        adaptive_token_totals: list[int] = []
        route_rates: list[float] = []
        for _ in range(repeats):
            current_plain = engine.forward_prompts(
                current_prompts, need_gate=False, batch_size=batch_size
            )
            recent_all = engine.forward_prompts(
                recent_prompts, need_gate=False, batch_size=batch_size
            )
            current_gate = engine.forward_prompts(
                current_prompts, need_gate=True, batch_size=batch_size
            )
            gate_array = np.asarray(current_gate.gate_probabilities, dtype=np.float64)
            p_skip = gate_array[:, GATE_TO_ID["SKIP"]]
            read_mask = (p_skip < float(gate_threshold)) & history_available
            selected_prompts = [
                prompt
                for prompt, use_context in zip(recent_prompts, read_mask)
                if use_context
            ]
            recent_selected = engine.forward_prompts(
                selected_prompts, need_gate=False, batch_size=batch_size
            )
            timings["current_only"].append(current_plain.elapsed_seconds)
            timings["always_recent3"].append(recent_all.elapsed_seconds)
            timings["adaptive"].append(
                current_gate.elapsed_seconds + recent_selected.elapsed_seconds
            )
            adaptive_token_totals.append(
                int(sum(current_gate.token_counts) + sum(recent_selected.token_counts))
            )
            route_rates.append(float(read_mask.mean()))
        medians = {name: float(np.median(values)) for name, values in timings.items()}
        total = len(benchmark_targets)
        always_tokens = int(
            sum(
                len(
                    engine.tokenizer(
                        prompt,
                        truncation=True,
                        max_length=engine.max_input_tokens,
                    )["input_ids"]
                )
                for prompt in recent_prompts
            )
        )
        adaptive_tokens = int(round(float(np.median(adaptive_token_totals))))
        output[str(batch_size)] = {
            "targets": total,
            "repeats": repeats,
            "median_seconds": medians,
            "utterances_per_second": {
                name: _safe_divide(total, seconds) for name, seconds in medians.items()
            },
            "adaptive_speedup_vs_always_recent3": _safe_divide(
                medians["always_recent3"], medians["adaptive"]
            ),
            "adaptive_speedup_vs_current_only": _safe_divide(
                medians["current_only"], medians["adaptive"]
            ),
            "always_recent3_input_tokens": always_tokens,
            "adaptive_input_tokens": adaptive_tokens,
            "adaptive_token_reduction_vs_always_recent3": (
                1.0 - _safe_divide(adaptive_tokens, always_tokens)
            ),
            "adaptive_context_use_rate": float(np.mean(route_rates)),
            "measurement_scope": (
                "tokenization plus CUDA forward passes; excludes model loading and prompt-string construction"
            ),
            "sampling": sampling,
        }
    return output


def _metrics_from_scores(
    calibration_scores: Sequence[Sequence[float]],
    validation_scores: Sequence[Sequence[float]],
    calibration_gold: Sequence[str],
    validation_gold: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    temperature = _fit_temperature(
        calibration_scores, [labels.index(label) for label in calibration_gold]
    )
    return _metrics(validation_scores, validation_gold, labels, temperature)


def _compact_variant(metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {**metrics, **extra}


def evaluate_seed(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    max_targets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if not paths["data_audit"].exists():
        prepare_training_data(config_path, run_name)
    output_path = paths["scores"] / f"seed{seed}_validation.json"
    if output_path.exists() and not overwrite:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    for variant in TRAIN_VARIANTS:
        if not (_adapter_dir(paths, seed, variant) / "training_complete.json").exists():
            raise FileNotFoundError(f"Missing adapter seed={seed} variant={variant}")
    calibration, validation = _load_inference_targets(root, section, paths)
    teacher_calibration = _read_jsonl(
        paths["data"] / "teacher_calibration_annotations.jsonl"
    )
    if max_targets is not None:
        teacher_keys = {
            f"{item['dialogue_id']}::{int(item['target_turn_id'])}"
            for item in teacher_calibration
        }
        diagnostic_targets = [
            target
            for target in calibration
            if target.key in teacher_keys and bool(target.history)
        ][:max_targets]
        if not diagnostic_targets:
            raise ValueError("Smoke subset contains no teacher-labelled gate target")
        calibration = diagnostic_targets
        validation = validation[:max_targets]
    labels = list(section.get("labels", LABELS))
    gold_calibration = [target.gold for target in calibration]
    gold_validation = [target.gold for target in validation]
    history_calibration = np.asarray([bool(target.history) for target in calibration])
    history_validation = np.asarray([bool(target.history) for target in validation])
    variants: dict[str, dict[str, Any]] = {}

    # Shared emotion base: no context-label supervision.
    base_engine = AdaptiveLLMEngine(
        section, _adapter_dir(paths, seed, "emotion_base"), with_gate=False
    )
    base_cal_current = base_engine.forward_targets(
        calibration, "current", need_gate=False
    )
    base_cal_recent = base_engine.forward_targets(
        calibration, "recent3", need_gate=False
    )
    base_val_current = base_engine.forward_targets(
        validation, "current", need_gate=False
    )
    base_val_recent = base_engine.forward_targets(
        validation, "recent3", need_gate=False
    )
    variants["base_current"] = _metrics_from_scores(
        base_cal_current.scores,
        base_val_current.scores,
        gold_calibration,
        gold_validation,
        labels,
    )
    variants["base_recent3"] = _metrics_from_scores(
        base_cal_recent.scores,
        base_val_recent.scores,
        gold_calibration,
        gold_validation,
        labels,
    )
    entropy_temperature = variants["base_current"]["temperature"]
    entropy_choice = _select_entropy_threshold(
        base_cal_current.scores,
        base_cal_recent.scores,
        gold_calibration,
        labels,
        entropy_temperature,
    )
    calibration_entropy = np.asarray(
        [
            -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
            for p in (
                _softmax(row, entropy_temperature) for row in base_cal_current.scores
            )
        ]
    )
    validation_entropy = np.asarray(
        [
            -float(np.sum(p * np.log(np.clip(p, 1e-12, 1.0))))
            for p in (
                _softmax(row, entropy_temperature) for row in base_val_current.scores
            )
        ]
    )
    entropy_cal_read = (
        calibration_entropy >= float(entropy_choice["threshold"])
    ) & history_calibration
    entropy_val_read = (
        validation_entropy >= float(entropy_choice["threshold"])
    ) & history_validation
    entropy_cal_scores = _mix_scores(
        base_cal_current.scores, base_cal_recent.scores, entropy_cal_read.tolist()
    )
    entropy_val_scores = _mix_scores(
        base_val_current.scores, base_val_recent.scores, entropy_val_read.tolist()
    )
    variants["entropy_adaptive"] = _compact_variant(
        _metrics_from_scores(
            entropy_cal_scores,
            entropy_val_scores,
            gold_calibration,
            gold_validation,
            labels,
        ),
        threshold=float(entropy_choice["threshold"]),
        context_use_rate=float(entropy_val_read.mean()),
    )
    del base_engine
    gc.collect()
    torch.cuda.empty_cache()

    # Correct current-only context supervision.
    label_engine = AdaptiveLLMEngine(
        section, _adapter_dir(paths, seed, "label_gate"), with_gate=True
    )
    label_cal_current = label_engine.forward_targets(
        calibration, "current", need_gate=True
    )
    label_val_current = label_engine.forward_targets(
        validation, "current", need_gate=True
    )
    label_diagnostics = _gate_diagnostics(
        calibration,
        label_cal_current.gate_probabilities or [],
        teacher_calibration,
        section,
    )
    label_threshold = float(label_diagnostics["threshold"])
    label_cal_skip = (
        np.asarray(label_cal_current.gate_probabilities)[:, GATE_TO_ID["SKIP"]]
        >= label_threshold
    )
    label_val_skip = (
        np.asarray(label_val_current.gate_probabilities)[:, GATE_TO_ID["SKIP"]]
        >= label_threshold
    )
    label_cal_read = (~label_cal_skip) & history_calibration
    label_val_read = (~label_val_skip) & history_validation
    label_cal_adaptive, label_cal_execution = _execute_conditional_recent3(
        label_engine,
        calibration,
        label_cal_current.scores,
        label_cal_read.tolist(),
    )
    label_val_adaptive, label_val_execution = _execute_conditional_recent3(
        label_engine,
        validation,
        label_val_current.scores,
        label_val_read.tolist(),
    )
    # These full recent-3 passes are independent accuracy controls. They occur
    # after the adaptive decisions and are never used by label_adaptive.
    label_cal_recent = label_engine.forward_targets(
        calibration, "recent3", need_gate=False
    )
    label_val_recent = label_engine.forward_targets(
        validation, "recent3", need_gate=False
    )
    variants["label_current"] = _metrics_from_scores(
        label_cal_current.scores,
        label_val_current.scores,
        gold_calibration,
        gold_validation,
        labels,
    )
    variants["label_recent3"] = _metrics_from_scores(
        label_cal_recent.scores,
        label_val_recent.scores,
        gold_calibration,
        gold_validation,
        labels,
    )
    variants["label_adaptive"] = _compact_variant(
        _metrics_from_scores(
            label_cal_adaptive,
            label_val_adaptive,
            gold_calibration,
            gold_validation,
            labels,
        ),
        gate_threshold=label_threshold,
        context_use_rate=float(label_val_read.mean()),
        input_tokens_current=int(sum(label_val_current.token_counts)),
        input_tokens_recent_selected=int(label_val_execution["selected_input_tokens"]),
        input_tokens_adaptive=int(
            sum(label_val_current.token_counts)
            + int(label_val_execution["selected_input_tokens"])
        ),
        input_tokens_always_recent3=int(sum(label_val_recent.token_counts)),
        conditional_execution={
            "calibration": label_cal_execution,
            "validation": label_val_execution,
        },
    )
    benchmark = _benchmark_engine(label_engine, validation, label_threshold, section)
    emotion_label_tokenizations = dict(label_engine.label_token_sequences)
    del label_engine
    gc.collect()
    torch.cuda.empty_cache()

    # Capacity-matched shuffled context labels.
    shuffled_engine = AdaptiveLLMEngine(
        section, _adapter_dir(paths, seed, "shuffled_gate"), with_gate=True
    )
    shuffled_cal_current = shuffled_engine.forward_targets(
        calibration, "current", need_gate=True
    )
    shuffled_val_current = shuffled_engine.forward_targets(
        validation, "current", need_gate=True
    )
    shuffled_diagnostics = _gate_diagnostics(
        calibration,
        shuffled_cal_current.gate_probabilities or [],
        teacher_calibration,
        section,
    )
    shuffled_threshold = float(shuffled_diagnostics["threshold"])
    shuffled_cal_read = (
        np.asarray(shuffled_cal_current.gate_probabilities)[:, GATE_TO_ID["SKIP"]]
        < shuffled_threshold
    ) & history_calibration
    shuffled_val_read = (
        np.asarray(shuffled_val_current.gate_probabilities)[:, GATE_TO_ID["SKIP"]]
        < shuffled_threshold
    ) & history_validation
    shuffled_cal_adaptive, shuffled_cal_execution = _execute_conditional_recent3(
        shuffled_engine,
        calibration,
        shuffled_cal_current.scores,
        shuffled_cal_read.tolist(),
    )
    shuffled_val_adaptive, shuffled_val_execution = _execute_conditional_recent3(
        shuffled_engine,
        validation,
        shuffled_val_current.scores,
        shuffled_val_read.tolist(),
    )
    variants["shuffled_adaptive"] = _compact_variant(
        _metrics_from_scores(
            shuffled_cal_adaptive,
            shuffled_val_adaptive,
            gold_calibration,
            gold_validation,
            labels,
        ),
        gate_threshold=shuffled_threshold,
        context_use_rate=float(shuffled_val_read.mean()),
        conditional_execution={
            "calibration": shuffled_cal_execution,
            "validation": shuffled_val_execution,
        },
    )
    del shuffled_engine
    gc.collect()
    torch.cuda.empty_cache()

    result = {
        "status": "evaluated",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "split": "validation",
        "targets": len(validation),
        "dialogues": len({target.dialogue_id for target in validation}),
        "labels": labels,
        "variants": variants,
        "gate_diagnostics_on_heldout_train_dialogues": {
            "label_gate": label_diagnostics,
            "shuffled_gate": shuffled_diagnostics,
        },
        "runtime_benchmark": benchmark,
        "runtime_environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(section.get("dtype", "bfloat16")),
            "model_resident_before_timing": True,
            "emotion_scoring": "unique_first_token_verbalizer",
            "emotion_label_tokenizations": emotion_label_tokenizations,
        },
        "test_locked": True,
    }
    _write_json(output_path, result)
    print(f"Stage 7.1 seed {seed} validation written to {output_path}")
    return result


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": array.tolist(),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "count": len(array),
    }


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    runs = []
    for seed in seeds:
        path = paths["scores"] / f"seed{seed}_validation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 7.1 seed report: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    labels = list(section.get("labels", LABELS))
    aggregate: dict[str, Any] = {}
    for variant in REPORT_VARIANTS:
        aggregate[variant] = {}
        for metric in (
            "weighted_f1",
            "macro_f1",
            "accuracy",
            "nll",
            "brier",
            "ece",
            "temperature",
        ):
            aggregate[variant][metric] = _aggregate(
                [float(run["variants"][variant][metric]) for run in runs]
            )
        aggregate[variant]["per_emotion_f1"] = {
            label: _aggregate(
                [
                    float(run["variants"][variant]["per_emotion_f1"][label])
                    for run in runs
                ]
            )
            for label in labels
        }
        if all("context_use_rate" in run["variants"][variant] for run in runs):
            aggregate[variant]["context_use_rate"] = _aggregate(
                [float(run["variants"][variant]["context_use_rate"]) for run in runs]
            )

    gate_metrics = (
        "context_positive_prevalence",
        "context_average_precision",
        "context_ap_lift_over_prevalence",
        "gate_three_way_macro_f1",
        "required_recall",
        "helpful_required_recall",
        "current_sufficient_skip_rate",
        "ambiguous_read_rate",
        "calibration_context_use_rate",
    )
    gate_diagnostics: dict[str, Any] = {}
    for variant in ("label_gate", "shuffled_gate"):
        gate_diagnostics[variant] = {
            metric: _aggregate(
                [
                    float(
                        run["gate_diagnostics_on_heldout_train_dialogues"][variant][
                            metric
                        ]
                    )
                    for run in runs
                ]
            )
            for metric in gate_metrics
            if all(
                run["gate_diagnostics_on_heldout_train_dialogues"][variant].get(metric)
                is not None
                for run in runs
            )
        }
        gate_diagnostics[variant]["threshold_constraints_met_all_seeds"] = all(
            bool(
                run["gate_diagnostics_on_heldout_train_dialogues"][variant][
                    "threshold_constraints_met"
                ]
            )
            for run in runs
        )
        gate_diagnostics[variant]["records_per_seed"] = [
            int(run["gate_diagnostics_on_heldout_train_dialogues"][variant]["records"])
            for run in runs
        ]

    benchmark: dict[str, Any] = {}
    batch_sizes = sorted(
        {batch for run in runs for batch in run["runtime_benchmark"].keys()},
        key=int,
    )
    for batch in batch_sizes:
        benchmark[batch] = {
            "adaptive_speedup_vs_always_recent3": _aggregate(
                [
                    float(
                        run["runtime_benchmark"][batch][
                            "adaptive_speedup_vs_always_recent3"
                        ]
                    )
                    for run in runs
                ]
            ),
            "adaptive_token_reduction_vs_always_recent3": _aggregate(
                [
                    float(
                        run["runtime_benchmark"][batch][
                            "adaptive_token_reduction_vs_always_recent3"
                        ]
                    )
                    for run in runs
                ]
            ),
            "adaptive_context_use_rate": _aggregate(
                [
                    float(run["runtime_benchmark"][batch]["adaptive_context_use_rate"])
                    for run in runs
                ]
            ),
            "median_seconds_by_seed": {
                name: [
                    float(run["runtime_benchmark"][batch]["median_seconds"][name])
                    for run in runs
                ]
                for name in ("current_only", "always_recent3", "adaptive")
            },
        }

    validation_dialogues = read_jsonl(
        _resolve(root, section["processed_dir"]) / "validation.jsonl"
    )
    targets = _targets(validation_dialogues, max_history=3)
    gold = [target.gold for target in targets]
    dialogue_ids = [target.dialogue_id for target in targets]
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in (
        "label_adaptive",
        "label_current",
        "label_recent3",
        "entropy_adaptive",
        "shuffled_adaptive",
    ):
        probabilities = np.mean(
            np.asarray([run["variants"][variant]["probabilities"] for run in runs]),
            axis=0,
        )
        rows_by_variant[variant] = [
            {"probabilities": row.tolist()} for row in probabilities
        ]
    controls = {
        "current_only": "label_current",
        "always_recent3": "label_recent3",
        "entropy_router": "entropy_adaptive",
        "shuffled_gate": "shuffled_adaptive",
    }
    bootstrap = {
        name: _bootstrap_advantage(
            rows_by_variant["label_adaptive"],
            rows_by_variant[control],
            gold,
            dialogue_ids,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 2388)) + offset,
        )
        for offset, (name, control) in enumerate(controls.items())
    }
    _write_json(paths["bootstrap"], bootstrap)

    capacity_records = []
    for seed in seeds:
        for variant in TRAIN_VARIANTS:
            state = json.loads(
                (
                    _adapter_dir(paths, seed, variant) / "training_complete.json"
                ).read_text(encoding="utf-8")
            )
            capacity_records.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "optimizer_steps": int(state["optimizer_steps"]),
                    "total_trainable_parameters": int(
                        state["total_trainable_parameters"]
                    ),
                    "finite": bool(state["finite"]),
                }
            )
    branch_records = [
        item
        for item in capacity_records
        if item["variant"] in {"label_gate", "shuffled_gate"}
    ]
    capacity_audit = {
        "correct_vs_shuffled_exact_parameter_match": len(
            {item["total_trainable_parameters"] for item in branch_records}
        )
        == 1,
        "correct_vs_shuffled_exact_step_match": len(
            {item["optimizer_steps"] for item in branch_records}
        )
        == 1,
        "all_training_finite": all(item["finite"] for item in capacity_records),
        "records": capacity_records,
    }

    label_gate = gate_diagnostics["label_gate"]
    shuffled_gate = gate_diagnostics["shuffled_gate"]
    adaptive = aggregate["label_adaptive"]
    best_weighted_control = max(
        aggregate[name]["weighted_f1"]["mean"]
        for name in ("label_current", "label_recent3", "entropy_adaptive")
    )
    best_macro_control = max(
        aggregate[name]["macro_f1"]["mean"]
        for name in ("label_current", "label_recent3", "entropy_adaptive")
    )
    tolerance = float(section.get("classification_noninferiority_tolerance", 0.003))
    primary_batch = str(section.get("primary_latency_batch_size", 1))
    shuffled_bootstrap = bootstrap["shuffled_gate"]
    checks = {
        "gate_threshold_constraints_all_seeds": bool(
            label_gate["threshold_constraints_met_all_seeds"]
        ),
        "gate_context_ap_absolute": label_gate["context_average_precision"]["mean"]
        >= float(section.get("minimum_context_ap", 0.55)),
        "gate_context_ap_lift_over_prevalence": label_gate[
            "context_ap_lift_over_prevalence"
        ]["mean"]
        >= float(section.get("minimum_context_ap_lift", 0.05)),
        "correct_gate_ap_beats_shuffled": (
            label_gate["context_average_precision"]["mean"]
            - shuffled_gate["context_average_precision"]["mean"]
        )
        >= float(section.get("minimum_ap_advantage_over_shuffled", 0.02)),
        "gate_required_recall": label_gate["required_recall"]["mean"]
        >= float(section.get("minimum_required_recall", 0.90)),
        "gate_useful_recall": label_gate["helpful_required_recall"]["mean"]
        >= float(section.get("minimum_useful_recall", 0.80)),
        "gate_skips_sufficient_current": label_gate["current_sufficient_skip_rate"][
            "mean"
        ]
        >= float(section.get("minimum_current_skip_rate", 0.20)),
        "adaptive_weighted_f1_noninferior": adaptive["weighted_f1"]["mean"]
        >= best_weighted_control - tolerance,
        "adaptive_macro_f1_noninferior": adaptive["macro_f1"]["mean"]
        >= best_macro_control - tolerance,
        "correct_gate_beats_shuffled_weighted_ci": shuffled_bootstrap["weighted_f1"][
            "ci95"
        ][0]
        > 0,
        "correct_gate_beats_shuffled_macro_ci": shuffled_bootstrap["macro_f1"]["ci95"][
            0
        ]
        > 0,
        "adaptive_latency_speedup": benchmark[primary_batch][
            "adaptive_speedup_vs_always_recent3"
        ]["mean"]
        >= float(section.get("minimum_latency_speedup", 1.10)),
        "adaptive_input_token_reduction": benchmark[primary_batch][
            "adaptive_token_reduction_vs_always_recent3"
        ]["mean"]
        >= float(section.get("minimum_input_token_reduction", 0.10)),
        "capacity_and_training_audit": all(
            (
                capacity_audit["correct_vs_shuffled_exact_parameter_match"],
                capacity_audit["correct_vs_shuffled_exact_step_match"],
                capacity_audit["all_training_finite"],
            )
        ),
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "eligible_to_freeze_stage7_1_before_separate_test_package"
            if eligible
            else "stop_and_diagnose_current_only_gate"
        ),
        "eligible": eligible,
        "checks": checks,
        "preregistered_before_stage7_1_validation": True,
        "selection_data": "dialogue-disjoint held-out MELD-train calibration only",
        "next_action": (
            "Freeze the selected adaptive configuration; create a separate one-shot test package."
            if eligible
            else "Keep MELD test locked; diagnose only failed gate, accuracy, or latency checks."
        ),
        "test_locked": True,
    }
    result = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "dataset": "MELD",
        "model": section["model_name"],
        "seeds": seeds,
        "split": "validation",
        "variants": aggregate,
        "gate_diagnostics_on_heldout_train_dialogues": gate_diagnostics,
        "runtime_benchmark": benchmark,
        "paired_dialogue_bootstrap": bootstrap,
        "capacity_audit": capacity_audit,
        "decision": decision,
        "teacher_status": "GPT-5.6 Sol generated and self-audited; not human gold",
        "test_locked": True,
    }
    _write_json(paths["summary_json"], result)
    _write_json(paths["decision"], decision)

    lines = [
        "# Stage 7.1 — Current-Only Learned Context Gate",
        "",
        "> The gate sees only the current utterance. recent-3 is loaded only after a READ/UNCERTAIN route. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE | context use |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in REPORT_VARIANTS:
        row = aggregate[variant]
        context = row.get("context_use_rate", {}).get("mean")
        context_text = f"{context:.4f}" if context is not None else "n/a"
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {row['weighted_f1']['mean']:.4f} ± {row['weighted_f1']['std']:.4f} "
            f"| {row['macro_f1']['mean']:.4f} ± {row['macro_f1']['std']:.4f} "
            f"| {row['nll']['mean']:.4f} ± {row['nll']['std']:.4f} "
            f"| {row['brier']['mean']:.4f} ± {row['brier']['std']:.4f} "
            f"| {row['ece']['mean']:.4f} ± {row['ece']['std']:.4f} | {context_text} |"
        )
    lines.extend(["", "## Current-only gate diagnostics", ""])
    for variant in ("label_gate", "shuffled_gate"):
        values = gate_diagnostics[variant]
        lines.append(
            f"- {variant}: AP={values['context_average_precision']['mean']:.4f} ± "
            f"{values['context_average_precision']['std']:.4f}, required recall="
            f"{values['required_recall']['mean']:.4f}, useful recall="
            f"{values['helpful_required_recall']['mean']:.4f}, current skip rate="
            f"{values['current_sufficient_skip_rate']['mean']:.4f}"
        )
    lines.extend(["", "## Runtime", ""])
    for batch, values in benchmark.items():
        lines.append(
            f"- batch={batch}: speedup vs always-recent3="
            f"{values['adaptive_speedup_vs_always_recent3']['mean']:.3f}x, input-token reduction="
            f"{values['adaptive_token_reduction_vs_always_recent3']['mean']:.2%}"
        )
    lines.extend(["", "## Frozen decision", "", f"Status: **{decision['status']}**"])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "Teacher annotations are GPT-5.6 Sol generated and self-audited operational labels, not human gold labels.",
        ]
    )
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result


def status(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    result = {
        "prepared": paths["data_audit"].exists(),
        "adapters": {
            str(seed): {
                variant: (
                    _adapter_dir(paths, seed, variant) / "training_complete.json"
                ).exists()
                for variant in TRAIN_VARIANTS
            }
            for seed in seeds
        },
        "evaluations": {
            str(seed): (paths["scores"] / f"seed{seed}_validation.json").exists()
            for seed in seeds
        },
        "report": paths["summary_json"].exists(),
        "decision": paths["decision"].exists(),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def final_test_locked() -> None:
    raise RuntimeError(
        "MELD test is hard-locked in Stage 7.1. Do not edit this command. "
        "A separate frozen one-shot test package may be created only if every "
        "preregistered validation check passes."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--overwrite", action="store_true")
    train_all_parser = subparsers.add_parser("train-all")
    train_all_parser.add_argument("--seeds", type=int, nargs="+", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("report")
    subparsers.add_parser("status")
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--seed", type=int, default=42)
    subparsers.add_parser("final-test")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "preflight":
        preflight(args.config)
    elif args.command == "prepare":
        prepare_training_data(args.config)
    elif args.command == "train":
        train_variant(
            args.config, args.variant, args.seed, overwrite=bool(args.overwrite)
        )
    elif args.command == "train-all":
        train_all(args.config, "validation", args.seeds)
    elif args.command == "evaluate":
        evaluate_seed(args.config, args.seed, overwrite=bool(args.overwrite))
    elif args.command == "report":
        report(args.config)
    elif args.command == "status":
        status(args.config)
    elif args.command == "smoke":
        smoke_name = f"smoke_seed{args.seed}"
        prepare_training_data(args.config, smoke_name)
        for variant in TRAIN_VARIANTS:
            train_variant(
                args.config,
                variant,
                args.seed,
                smoke_name,
                max_steps_override=2,
                max_examples=32,
                overwrite=True,
            )
        evaluate_seed(
            args.config,
            args.seed,
            smoke_name,
            max_targets=8,
            overwrite=True,
        )
    elif args.command == "final-test":
        final_test_locked()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
