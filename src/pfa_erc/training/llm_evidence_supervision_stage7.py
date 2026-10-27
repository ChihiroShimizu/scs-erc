"""Stage 7: LLM-native, label-supervised local evidence reasoning for ERC.

The additional S6C-V2.3 annotations supervise *how an LLM retrieves context*.
They are never exposed at validation inference time and their self-audit scores
are never model inputs.  The model receives only the current utterance and the
last three causal turns, first emits a compact, machine-checkable retrieval
plan, then predicts the target emotion from the selected evidence.

This module deliberately does not import or train a BERT/RoBERTa model.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
)

from pfa_erc.data import Dialogue, Utterance, read_jsonl


VERSION = "2.0.0"
EXPERIMENT = "llm_evidence_supervision_stage7"

LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
CONTEXT_NEEDS = (
    "current_sufficient",
    "history_helpful",
    "history_required",
    "ambiguous",
)
ROLE_RELATIONS = (
    "direct_to_target",
    "about_target",
    "reply_dependency",
    "shared_event",
    "self_continuation",
    "unrelated",
    "uncertain",
)
UTILITIES = ("decisive", "helpful", "weak", "irrelevant", "misleading", "uncertain")
IMPACT_DIRECTIONS = ("positive", "negative", "mixed", "none", "uncertain")
TRANSITION_TYPES = (
    "valence_flip",
    "emotion_shift_same_valence",
    "intensity_change",
    "reinforcement",
    "none",
    "uncertain",
)
EVIDENCE_BASES = (
    "explicit_reply",
    "direct_address",
    "discourse_dependency",
    "semantic_context",
    "none",
    "uncertain",
)

TRAIN_VARIANTS = (
    "emotion_base",
    "generic_reasoning",
    "evidence_supervised",
    "full_supervised",
    "shuffled_supervised",
)

REPORT_VARIANTS = (
    "current_only",
    "always_recent3",
    "entropy_router",
    "generic_reasoning",
    "evidence_supervised",
    "full_no_impact",
    "full_supervised",
    "shuffled_supervised",
)

VARIANT_LABELS = {
    "current_only": "7A current-only LLM SFT",
    "always_recent3": "7B always recent-3",
    "entropy_router": "7C train-fitted entropy router",
    "generic_reasoning": "7D generic-reasoning control",
    "evidence_supervised": "7E evidence-label SFT",
    "full_no_impact": "7F full SFT, evidence only at read",
    "full_supervised": "7G full label-supervised reasoning",
    "shuffled_supervised": "7H shuffled-label control",
}

ANNOTATION_FIELDS = {
    "annotation_version",
    "source_annotation_id",
    "dialogue_id",
    "target_turn_id",
    "target_role",
    "context_need",
    "candidate_assessments",
    "selected_evidence_turn_id",
    "selection_reason",
    "overall_confidence",
}

CANDIDATE_FIELDS = {
    "candidate_turn_id",
    "distance",
    "candidate_speaker",
    "role_relation",
    "prediction_utility",
    "impact_direction",
    "transition_type",
    "impact_strength",
    "evidence_basis",
    "confidence",
    "short_reason",
}

PLANNER_SYSTEM = (
    "You are a causal local-evidence controller for emotion recognition. "
    "Use only the supplied current utterance and earlier turns. Do not use future "
    "turns, outside story knowledge, or a gold emotion. Return one compact JSON object."
)

EMOTION_SYSTEM = (
    "You are a dialogue emotion classifier. Use only the supplied causal text and "
    "controller output. Return exactly one emotion label and no explanation."
)


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[dict[str, str]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Any:
    """Use Qwen thinking control when supported, with a version-safe fallback."""
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


@dataclass(frozen=True)
class Target:
    dialogue_id: str
    turn_index: int
    utterance_id: str
    speaker: str
    text: str
    gold: str
    history: tuple[Utterance, ...]

    @property
    def key(self) -> str:
        return f"{self.dialogue_id}::{self.turn_index}"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _project_root(config_path: Path) -> Path:
    path = config_path.expanduser().resolve()
    return path.parent.parent if path.parent.name == "configs" else Path.cwd().resolve()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("llm_evidence_supervision_stage7")
    if not isinstance(value, dict):
        raise KeyError("Missing llm_evidence_supervision_stage7 configuration section")
    return value


def _paths(root: Path, section: dict[str, Any], run_name: str = "validation") -> dict[str, Path]:
    base = _resolve(root, section.get("artifact_dir", "artifacts/llm_evidence_stage7/meld"))
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "data": base / "prepared",
        "adapters": base / "adapters",
        "scores": base / "scores",
        "plans": base / "plans",
        "reports": base / "reports",
        "manifest": base / "run_manifest.json",
        "preflight": base / "reports" / "preflight.json",
        "data_audit": base / "reports" / "training_data_audit.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_validation_decision.json",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
    }


def _annotation_object(record: dict[str, Any]) -> dict[str, Any]:
    """Drop audit/provenance fields and retain only the final teacher annotation."""
    if "annotation_version" in record:
        value = record
    elif isinstance(record.get("final_annotation"), dict):
        value = record["final_annotation"]
    elif isinstance(record.get("annotation"), dict):
        value = record["annotation"]
    else:
        raise ValueError("Record has no complete S6C-V2.3 annotation payload")
    return dict(value)


def _targets(dialogues: Sequence[Dialogue], max_history: int = 3) -> list[Target]:
    result: list[Target] = []
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue.utterances):
            result.append(
                Target(
                    dialogue_id=dialogue.dialogue_id,
                    turn_index=index,
                    utterance_id=turn.utterance_id,
                    speaker=turn.speaker,
                    text=turn.text,
                    gold=turn.emotion,
                    history=tuple(dialogue.utterances[max(0, index - max_history) : index]),
                )
            )
    return result


def _target_map(dialogues: Sequence[Dialogue], max_history: int = 3) -> dict[str, Target]:
    values = _targets(dialogues, max_history=max_history)
    mapping = {target.key: target for target in values}
    if len(mapping) != len(values):
        raise ValueError("Duplicate dialogue/turn key in processed data")
    return mapping


def _stable_hash(text: str, seed: int) -> int:
    raw = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "big")


def _split_dialogues(
    dialogue_ids: Sequence[str], fraction: float, seed: int
) -> tuple[set[str], set[str]]:
    unique = sorted(set(dialogue_ids), key=lambda value: (_stable_hash(value, seed), value))
    count = max(1, min(len(unique) - 1, int(round(len(unique) * fraction))))
    calibration = set(unique[:count])
    return set(unique[count:]), calibration


def _selected_assessment(annotation: dict[str, Any]) -> dict[str, Any] | None:
    selected = annotation.get("selected_evidence_turn_id")
    if selected is None:
        return None
    for item in annotation.get("candidate_assessments", []):
        if int(item["candidate_turn_id"]) == int(selected):
            return item
    return None


def validate_teacher_annotation(annotation: dict[str, Any], target: Target) -> list[str]:
    issues: list[str] = []
    if set(annotation) != ANNOTATION_FIELDS:
        return [f"{target.key}: top-level schema mismatch"]
    if annotation.get("annotation_version") != "S6C-V2.3-local-impact":
        issues.append(f"{target.key}: wrong annotation version")
    if str(annotation.get("dialogue_id")) != target.dialogue_id:
        issues.append(f"{target.key}: dialogue mismatch")
    if int(annotation.get("target_turn_id", -1)) != target.turn_index:
        issues.append(f"{target.key}: target turn mismatch")
    if str(annotation.get("target_role")) != target.speaker:
        issues.append(f"{target.key}: target role mismatch")
    if annotation.get("context_need") not in CONTEXT_NEEDS:
        issues.append(f"{target.key}: invalid context_need")
    expected_ids = list(range(max(0, target.turn_index - 3), target.turn_index))[::-1]
    candidates = annotation.get("candidate_assessments")
    if not isinstance(candidates, list):
        return issues + [f"{target.key}: candidates are not a list"]
    actual_ids: list[int] = []
    history_map = {
        target.turn_index - len(target.history) + offset: turn
        for offset, turn in enumerate(target.history)
    }
    for item in candidates:
        if not isinstance(item, dict) or set(item) != CANDIDATE_FIELDS:
            issues.append(f"{target.key}: candidate schema mismatch")
            continue
        candidate_id = int(item["candidate_turn_id"])
        actual_ids.append(candidate_id)
        if candidate_id not in history_map:
            issues.append(f"{target.key}: noncausal candidate {candidate_id}")
            continue
        if str(item["candidate_speaker"]) != history_map[candidate_id].speaker:
            issues.append(f"{target.key}: candidate speaker mismatch")
        if int(item["distance"]) != target.turn_index - candidate_id:
            issues.append(f"{target.key}: candidate distance mismatch")
        if item["role_relation"] not in ROLE_RELATIONS:
            issues.append(f"{target.key}: invalid role_relation")
        if item["prediction_utility"] not in UTILITIES:
            issues.append(f"{target.key}: invalid prediction_utility")
        if item["impact_direction"] not in IMPACT_DIRECTIONS:
            issues.append(f"{target.key}: invalid impact_direction")
        if item["transition_type"] not in TRANSITION_TYPES:
            issues.append(f"{target.key}: invalid transition_type")
        if item["impact_strength"] not in {0, 1, 2, 3, None}:
            issues.append(f"{target.key}: invalid impact_strength")
        if item["evidence_basis"] not in EVIDENCE_BASES:
            issues.append(f"{target.key}: invalid evidence_basis")
    if actual_ids != expected_ids:
        issues.append(f"{target.key}: candidate coverage/order {actual_ids} != {expected_ids}")
    selected = annotation.get("selected_evidence_turn_id")
    if selected is not None and int(selected) not in actual_ids:
        issues.append(f"{target.key}: selected evidence outside recent window")
    if annotation["context_need"] == "current_sufficient" and selected is not None:
        issues.append(f"{target.key}: current_sufficient must not select evidence")
    if annotation["context_need"] in {"history_helpful", "history_required"}:
        chosen = _selected_assessment(annotation)
        if chosen is None or chosen.get("prediction_utility") not in {"helpful", "decisive"}:
            issues.append(f"{target.key}: required/helpful context lacks valid evidence")
    if target.turn_index == 0 and (candidates or selected is not None):
        issues.append(f"{target.key}: first turn contains history evidence")
    return issues


def _render_turns(target: Target, history: Sequence[tuple[int, Utterance]]) -> str:
    if not history:
        return "[no earlier turns]"
    return "\n".join(f"[{index}] {turn.speaker}: {turn.text}" for index, turn in history)


def _indexed_history(target: Target) -> list[tuple[int, Utterance]]:
    start = target.turn_index - len(target.history)
    return [(start + offset, turn) for offset, turn in enumerate(target.history)]


def build_retrieval_prompt(target: Target, full_fields: bool) -> str:
    fields = (
        "context_need, selected_evidence_turn_id, role_relation, prediction_utility, "
        "impact_direction, transition_type, impact_strength, evidence_basis"
        if full_fields
        else "context_need, selected_evidence_turn_id, prediction_utility"
    )
    return (
        "Task: learnable local evidence retrieval for ERC.\n"
        f"Recent causal window:\n{_render_turns(target, _indexed_history(target))}\n\n"
        f"Target [{target.turn_index}] {target.speaker}: {target.text}\n\n"
        "Decide whether the target emotion can be judged from the current utterance. "
        "If context is useful, select zero or one turn from the supplied recent window. "
        "Do not predict the emotion in this task. "
        f"Return JSON containing exactly: {fields}."
    )


def _plan_from_annotation(annotation: dict[str, Any], full_fields: bool) -> dict[str, Any]:
    chosen = _selected_assessment(annotation)
    payload: dict[str, Any] = {
        "context_need": annotation["context_need"],
        "selected_evidence_turn_id": annotation["selected_evidence_turn_id"],
        "prediction_utility": chosen["prediction_utility"] if chosen else "irrelevant",
    }
    if full_fields:
        payload.update(
            role_relation=chosen["role_relation"] if chosen else "unrelated",
            impact_direction=chosen["impact_direction"] if chosen else "none",
            transition_type=chosen["transition_type"] if chosen else "none",
            impact_strength=chosen["impact_strength"] if chosen else 0,
            evidence_basis=chosen["evidence_basis"] if chosen else "none",
        )
    return payload


def _selected_history(
    target: Target, selected_turn_id: int | None
) -> list[tuple[int, Utterance]]:
    if selected_turn_id is None:
        return []
    return [item for item in _indexed_history(target) if item[0] == int(selected_turn_id)]


def build_emotion_prompt(
    target: Target,
    labels: Sequence[str],
    *,
    mode: str,
    selected_turn_id: int | None = None,
    plan: dict[str, Any] | None = None,
) -> str:
    if mode == "current":
        history: list[tuple[int, Utterance]] = []
        guidance = "No context was supplied."
    elif mode in {"recent3", "generic"}:
        history = _indexed_history(target)
        guidance = (
            "Reason internally about whether the recent turns disambiguate the target."
            if mode == "generic"
            else "Use all recent turns only when they are relevant."
        )
    elif mode in {"selected", "structured"}:
        history = _selected_history(target, selected_turn_id)
        guidance = "The controller selected at most one local evidence turn."
    else:
        raise ValueError(f"Unknown emotion prompt mode: {mode}")

    controller = ""
    if mode == "structured" and plan:
        safe_fields = {
            key: plan.get(key)
            for key in (
                "context_need",
                "role_relation",
                "prediction_utility",
                "impact_direction",
                "transition_type",
                "impact_strength",
                "evidence_basis",
            )
        }
        controller = "\nController decision: " + json.dumps(
            safe_fields, ensure_ascii=False, separators=(",", ":")
        )
    return (
        f"Task: classify the target speaker's expressed emotion. {guidance}\n"
        f"Earlier evidence:\n{_render_turns(target, history)}{controller}\n\n"
        f"Target [{target.turn_index}] {target.speaker}: {target.text}\n\n"
        f"Return exactly one label from: {', '.join(labels)}.\nEmotion:"
    )


def _sft_record(system: str, prompt: str, completion: str, **meta: Any) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "completion": completion,
        "meta": meta,
    }


def _semantic_shuffle(
    annotations: Sequence[dict[str, Any]], seed: int
) -> dict[str, dict[str, Any]]:
    """Create structurally valid but semantically mismatched teacher targets.

    Candidate IDs/speakers remain tied to the receiving example; only semantic
    labels are copied by recency rank. This prevents an invalid future-ID
    control from making the shuffled baseline artificially easy to detect.
    """
    donors = list(annotations)
    random.Random(seed).shuffle(donors)
    if len(donors) > 1:
        receiver_keys = [
            f"{item['dialogue_id']}::{int(item['target_turn_id'])}" for item in annotations
        ]
        for shift in range(len(donors)):
            candidate_order = donors[shift:] + donors[:shift]
            donor_keys = [
                f"{item['dialogue_id']}::{int(item['target_turn_id'])}"
                for item in candidate_order
            ]
            if all(left != right for left, right in zip(receiver_keys, donor_keys)):
                donors = candidate_order
                break
    output: dict[str, dict[str, Any]] = {}
    semantic_fields = (
        "role_relation",
        "prediction_utility",
        "impact_direction",
        "transition_type",
        "impact_strength",
        "evidence_basis",
    )
    for receiver, donor in zip(annotations, donors):
        item = json.loads(json.dumps(receiver))
        receiver_candidates = item["candidate_assessments"]
        donor_candidates = donor["candidate_assessments"]
        for index, candidate in enumerate(receiver_candidates):
            if index >= len(donor_candidates):
                break
            for field in semantic_fields:
                candidate[field] = donor_candidates[index][field]
        donor_selected = donor.get("selected_evidence_turn_id")
        donor_distance = None
        if donor_selected is not None:
            donor_chosen = _selected_assessment(donor)
            donor_distance = int(donor_chosen["distance"]) if donor_chosen else None
        mapped = next(
            (
                candidate["candidate_turn_id"]
                for candidate in receiver_candidates
                if donor_distance is not None and int(candidate["distance"]) == donor_distance
            ),
            None,
        )
        donor_need = donor["context_need"]
        if donor_need in {"history_helpful", "history_required"} and mapped is None:
            donor_need = "ambiguous"
        item["context_need"] = donor_need
        item["selected_evidence_turn_id"] = (
            mapped if donor_need in {"history_helpful", "history_required"} else None
        )
        item["selection_reason"] = "Semantically shuffled training control."
        key = f"{receiver['dialogue_id']}::{int(receiver['target_turn_id'])}"
        output[key] = item
    return output


def preflight(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    config = _read_yaml(config_path)
    section = _section(config)
    paths = _paths(root, section, run_name)
    if str(section.get("dataset", "meld")).lower() != "meld":
        raise ValueError("Stage 7 currently accepts MELD only")
    processed = _resolve(root, section["processed_dir"])
    train_path = processed / "train.jsonl"
    validation_path = processed / "validation.jsonl"
    teacher_path = _resolve(root, section["teacher_annotation_file"])
    missing = [str(path) for path in (train_path, validation_path, teacher_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 7 input(s): " + ", ".join(missing))
    if "test" in str(teacher_path).lower():
        raise ValueError("Teacher path must not reference MELD test")
    records = _read_jsonl(teacher_path)
    expected_count = int(section.get("expected_teacher_records", 500))
    if len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} teacher records, found {len(records)}")
    expected_hash = str(section.get("expected_teacher_sha256", "")).strip().lower()
    actual_hash = _sha256(teacher_path)
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Teacher SHA256 mismatch: expected {expected_hash}, found {actual_hash}"
        )
    train_dialogues = read_jsonl(train_path)
    validation_dialogues = read_jsonl(validation_path)
    mapping = _target_map(train_dialogues, max_history=3)
    annotations: dict[str, dict[str, Any]] = {}
    issues: list[str] = []
    for record in records:
        annotation = _annotation_object(record)
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
        preview = "\n - ".join(issues[:20])
        raise ValueError(f"Teacher annotation audit failed ({len(issues)} issues):\n - {preview}")
    forbidden = [name for name in ("final_total_score", "original_total_score") if any(name in x for x in annotations.values())]
    result = {
        "status": "ready",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "model_family": "causal_llm_only",
        "model_name": section["model_name"],
        "roberta_or_bert_used": False,
        "teacher_source": "GPT-5.6 Sol generated and self-audited",
        "independent_verification": False,
        "teacher_records": len(annotations),
        "teacher_dialogues": len({item["dialogue_id"] for item in annotations.values()}),
        "teacher_sha256": actual_hash,
        "train_dialogues": len(train_dialogues),
        "train_targets": sum(len(item.utterances) for item in train_dialogues),
        "validation_dialogues": len(validation_dialogues),
        "validation_targets": sum(len(item.utterances) for item in validation_dialogues),
        "audit_scores_used_as_features": False,
        "forbidden_audit_fields_seen": forbidden,
        "causal_window": 3,
        "test_locked": True,
        "teacher_file": str(teacher_path),
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
            "model_name": section["model_name"],
            "seeds": list(section.get("seeds", [42, 43, 44])),
            "test_locked": True,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def prepare_training_data(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    preflight(config_path, run_name)
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    processed = _resolve(root, section["processed_dir"])
    train_dialogues = read_jsonl(processed / "train.jsonl")
    target_map = _target_map(train_dialogues, max_history=3)
    raw_teacher = _read_jsonl(_resolve(root, section["teacher_annotation_file"]))
    annotations = [_annotation_object(record) for record in raw_teacher]
    teacher_by_key = {
        f"{item['dialogue_id']}::{int(item['target_turn_id'])}": item for item in annotations
    }
    train_ids, calibration_ids = _split_dialogues(
        [dialogue.dialogue_id for dialogue in train_dialogues],
        float(section.get("calibration_dialogue_fraction", 0.10)),
        int(section.get("split_seed", 2026)),
    )
    labels = list(section.get("labels", LABELS))
    base_examples: list[dict[str, Any]] = []
    calibration_targets: list[dict[str, Any]] = []
    for target in target_map.values():
        if target.dialogue_id in train_ids:
            base_examples.append(
                _sft_record(
                    EMOTION_SYSTEM,
                    build_emotion_prompt(target, labels, mode="current"),
                    target.gold,
                    task="emotion_current",
                    key=target.key,
                    dialogue_id=target.dialogue_id,
                )
            )
        else:
            calibration_targets.append(
                {
                    "key": target.key,
                    "dialogue_id": target.dialogue_id,
                    "turn_index": target.turn_index,
                    "utterance_id": target.utterance_id,
                    "speaker": target.speaker,
                    "text": target.text,
                    "gold": target.gold,
                    "history": [
                        {
                            "turn_index": index,
                            "utterance_id": turn.utterance_id,
                            "speaker": turn.speaker,
                            "text": turn.text,
                        }
                        for index, turn in _indexed_history(target)
                    ],
                }
            )

    train_annotations = [
        value for key, value in teacher_by_key.items() if target_map[key].dialogue_id in train_ids
    ]
    calibration_annotations = [
        value
        for key, value in teacher_by_key.items()
        if target_map[key].dialogue_id in calibration_ids
    ]
    shuffled = _semantic_shuffle(
        train_annotations, int(section.get("shuffle_control_seed", 2077))
    )

    generic: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    shuffled_records: list[dict[str, Any]] = []
    replay_per_teacher = int(section.get("emotion_replay_per_teacher", 1))
    replay_pool = [example for example in base_examples]
    replay_rng = random.Random(int(section.get("split_seed", 2026)) + 71)
    replay_rng.shuffle(replay_pool)

    for index, annotation in enumerate(train_annotations):
        key = f"{annotation['dialogue_id']}::{int(annotation['target_turn_id'])}"
        target = target_map[key]
        lite_plan = _plan_from_annotation(annotation, full_fields=False)
        full_plan = _plan_from_annotation(annotation, full_fields=True)
        shuffled_plan = _plan_from_annotation(shuffled[key], full_fields=True)
        selected = annotation.get("selected_evidence_turn_id")
        shuffled_selected = shuffled[key].get("selected_evidence_turn_id")

        generic.append(
            _sft_record(
                EMOTION_SYSTEM,
                build_emotion_prompt(target, labels, mode="generic"),
                target.gold,
                task="generic_emotion",
                key=key,
            )
        )
        evidence.extend(
            [
                _sft_record(
                    PLANNER_SYSTEM,
                    build_retrieval_prompt(target, full_fields=False),
                    json.dumps(lite_plan, ensure_ascii=False, separators=(",", ":")),
                    task="retrieve_evidence",
                    key=key,
                ),
                _sft_record(
                    EMOTION_SYSTEM,
                    build_emotion_prompt(
                        target,
                        labels,
                        mode="selected",
                        selected_turn_id=selected,
                        plan=lite_plan,
                    ),
                    target.gold,
                    task="emotion_from_gold_evidence",
                    key=key,
                ),
            ]
        )
        full.extend(
            [
                _sft_record(
                    PLANNER_SYSTEM,
                    build_retrieval_prompt(target, full_fields=True),
                    json.dumps(full_plan, ensure_ascii=False, separators=(",", ":")),
                    task="retrieve_full_structure",
                    key=key,
                ),
                _sft_record(
                    EMOTION_SYSTEM,
                    build_emotion_prompt(
                        target,
                        labels,
                        mode="structured",
                        selected_turn_id=selected,
                        plan=full_plan,
                    ),
                    target.gold,
                    task="emotion_from_gold_structure",
                    key=key,
                ),
            ]
        )
        shuffled_records.extend(
            [
                _sft_record(
                    PLANNER_SYSTEM,
                    build_retrieval_prompt(target, full_fields=True),
                    json.dumps(shuffled_plan, ensure_ascii=False, separators=(",", ":")),
                    task="retrieve_shuffled_structure",
                    key=key,
                ),
                _sft_record(
                    EMOTION_SYSTEM,
                    build_emotion_prompt(
                        target,
                        labels,
                        mode="structured",
                        selected_turn_id=shuffled_selected,
                        plan=shuffled_plan,
                    ),
                    target.gold,
                    task="emotion_from_shuffled_structure",
                    key=key,
                ),
            ]
        )
        for replay_offset in range(replay_per_teacher):
            replay = replay_pool[(index * replay_per_teacher + replay_offset) % len(replay_pool)]
            evidence.append(dict(replay))
            full.append(dict(replay))
            shuffled_records.append(dict(replay))
            generic.append(dict(replay))

    outputs = {
        "emotion_base": base_examples,
        "generic_reasoning": generic,
        "evidence_supervised": evidence,
        "full_supervised": full,
        "shuffled_supervised": shuffled_records,
    }
    for name, values in outputs.items():
        _write_jsonl(paths["data"] / f"{name}_train.jsonl", values)
    _write_jsonl(paths["data"] / "router_calibration_targets.jsonl", calibration_targets)
    _write_jsonl(paths["data"] / "teacher_train_annotations.jsonl", train_annotations)
    _write_jsonl(paths["data"] / "teacher_calibration_annotations.jsonl", calibration_annotations)
    _write_jsonl(paths["data"] / "teacher_train_shuffled_annotations.jsonl", shuffled.values())

    audit = {
        "status": "passed",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "split_policy": "fixed dialogue-disjoint train/calibration split",
        "split_seed": int(section.get("split_seed", 2026)),
        "calibration_dialogue_fraction": float(
            section.get("calibration_dialogue_fraction", 0.10)
        ),
        "model_train_dialogues": len(train_ids),
        "router_calibration_dialogues": len(calibration_ids),
        "base_emotion_examples": len(base_examples),
        "router_calibration_targets": len(calibration_targets),
        "teacher_train_records": len(train_annotations),
        "teacher_calibration_records": len(calibration_annotations),
        "branch_examples": {name: len(values) for name, values in outputs.items()},
        "teacher_labels_hidden_at_validation_inference": True,
        "gold_emotion_hidden_from_retrieval_prompts": True,
        "audit_scores_used_as_features": False,
        "test_locked": True,
    }
    _write_json(paths["data_audit"], audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


class SFTDataset(torch.utils.data.Dataset):
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

    def _render_prompt(self, messages: Sequence[dict[str, str]]) -> list[int]:
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
        return list(self.tokenizer.encode(text + "\nassistant:", add_special_tokens=True))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        prompt_ids = self._render_prompt(record["messages"])
        completion = str(record["completion"])
        completion_ids = list(self.tokenizer.encode(completion, add_special_tokens=False))
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            completion_ids.append(int(eos_id))
        if len(completion_ids) >= self.max_length:
            raise ValueError("Completion exceeds max sequence length")
        prompt_ids = prompt_ids[-(self.max_length - len(completion_ids)) :]
        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch: Sequence[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    length = max(int(item["input_ids"].numel()) for item in batch)
    input_ids = torch.full((len(batch), length), pad_id, dtype=torch.long)
    attention = torch.zeros((len(batch), length), dtype=torch.long)
    labels = torch.full((len(batch), length), -100, dtype=torch.long)
    for row, item in enumerate(batch):
        size = int(item["input_ids"].numel())
        input_ids[row, :size] = item["input_ids"]
        attention[row, :size] = item["attention_mask"]
        labels[row, :size] = item["labels"]
    return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


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
            "Stage 7 requires transformers, accelerate and peft; run "
            "bash scripts/setup_llm_evidence_stage7.sh"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Full Stage 7 SFT requires a CUDA GPU")
    tokenizer = _load_tokenizer(section)
    dtype = torch.bfloat16 if str(section.get("dtype", "bfloat16")) == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        section["model_name"],
        torch_dtype=dtype,
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
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                )
            ),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    else:
        if not (parent_adapter / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing parent LoRA adapter: {parent_adapter}")
        model = PeftModel.from_pretrained(model, str(parent_adapter), is_trainable=True)
    model.to(torch.device("cuda"))
    return model, tokenizer


def _adapter_dir(paths: dict[str, Path], seed: int, variant: str) -> Path:
    return paths["adapters"] / f"seed{seed}" / variant


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
        raise ValueError(f"Unknown training variant {variant}; choose from {TRAIN_VARIANTS}")
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if not paths["data_audit"].exists():
        prepare_training_data(config_path, run_name)
    output_dir = _adapter_dir(paths, seed, variant)
    done = output_dir / "training_complete.json"
    if done.exists() and not overwrite:
        result = json.loads(done.read_text(encoding="utf-8"))
        print(json.dumps(result, indent=2))
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
    parent = None if variant == "emotion_base" else _adapter_dir(paths, seed, "emotion_base")
    if parent is not None and not (parent / "training_complete.json").exists():
        raise FileNotFoundError(
            f"Train emotion_base seed={seed} before branch {variant}: {parent}"
        )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, tokenizer = _load_trainable_model(section, parent)
    dataset = SFTDataset(
        records,
        tokenizer,
        int(section.get("max_sequence_tokens", 768)),
        enable_thinking=bool(section.get("enable_thinking", False)),
    )
    micro_batch = int(section.get("micro_batch_size", 1))
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: _collate(batch, int(tokenizer.pad_token_id)),
    )
    if variant == "emotion_base":
        default_steps = int(section.get("emotion_base_max_steps", 600))
        learning_rate = float(section.get("emotion_base_learning_rate", 2.0e-5))
    else:
        default_steps = int(section.get("branch_max_steps", 160))
        learning_rate = float(section.get("branch_learning_rate", 1.0e-5))
    max_steps = int(max_steps_override or default_steps)
    accumulation = int(section.get("gradient_accumulation_steps", 16))
    warmup = min(int(section.get("warmup_steps", 20)), max(0, max_steps - 1))
    from transformers import get_linear_schedule_with_warmup

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=float(section.get("weight_decay", 0.01)),
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, max_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro_step = 0
    losses: list[float] = []
    started = time.time()
    while step < max_steps:
        for batch in loader:
            batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
            output = model(**batch)
            loss = output.loss / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite Stage 7 loss at variant={variant} seed={seed} step={step}"
                )
            loss.backward()
            losses.append(float(loss.detach().cpu()) * accumulation)
            micro_step += 1
            if micro_step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(parameters, float(section.get("max_grad_norm", 1.0)))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 1 or step % int(section.get("log_every_steps", 10)) == 0:
                    recent = float(np.mean(losses[-min(len(losses), accumulation * 10) :]))
                    print(
                        f"[Stage7] {variant} seed={seed} step={step}/{max_steps} "
                        f"loss={recent:.5f} lr={scheduler.get_last_lr()[0]:.3e}",
                        flush=True,
                    )
                if step >= max_steps:
                    break
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
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
        "trainable_parameters": int(sum(value.numel() for value in parameters)),
        "elapsed_seconds": time.time() - started,
        "finite": bool(np.isfinite(losses).all()),
        "test_data_read": False,
    }
    _write_json(done, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del model, tokenizer, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _extract_json(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def normalize_plan(raw: dict[str, Any] | None, target: Target, full_fields: bool) -> dict[str, Any]:
    raw = raw or {}
    history_ids = {index for index, _ in _indexed_history(target)}
    context_need = str(raw.get("context_need", "ambiguous"))
    if context_need not in CONTEXT_NEEDS:
        context_need = "ambiguous"
    selected: int | None
    try:
        selected = int(raw.get("selected_evidence_turn_id"))
    except (TypeError, ValueError):
        selected = None
    if selected not in history_ids or context_need not in {"history_helpful", "history_required"}:
        selected = None
    plan: dict[str, Any] = {
        "parse_success": bool(raw),
        "context_need": context_need,
        "selected_evidence_turn_id": selected,
        "prediction_utility": str(raw.get("prediction_utility", "uncertain")),
    }
    if plan["prediction_utility"] not in UTILITIES:
        plan["prediction_utility"] = "uncertain"
    if full_fields:
        for name, allowed, default in (
            ("role_relation", ROLE_RELATIONS, "uncertain"),
            ("impact_direction", IMPACT_DIRECTIONS, "uncertain"),
            ("transition_type", TRANSITION_TYPES, "uncertain"),
            ("evidence_basis", EVIDENCE_BASES, "uncertain"),
        ):
            value = str(raw.get(name, default))
            plan[name] = value if value in allowed else default
        strength = raw.get("impact_strength")
        plan["impact_strength"] = strength if strength in {0, 1, 2, 3, None} else None
    return plan


class LLMEngine:
    def __init__(self, section: dict[str, Any], adapter: Path):
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("Install peft and transformers for Stage 7 inference") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 7 validation requires CUDA")
        self.section = section
        self.labels = list(section.get("labels", LABELS))
        self.tokenizer = _load_tokenizer(section)
        dtype = torch.bfloat16 if str(section.get("dtype", "bfloat16")) == "bfloat16" else torch.float16
        base = AutoModelForCausalLM.from_pretrained(
            section["model_name"],
            torch_dtype=dtype,
            trust_remote_code=bool(section.get("trust_remote_code", False)),
            local_files_only=bool(section.get("local_files_only", False)),
            low_cpu_mem_usage=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
        self.model.to("cuda")
        self.model.eval()
        self.batch_size = int(section.get("inference_batch_size", 2))
        self.max_input_tokens = int(section.get("max_sequence_tokens", 768))
        self.label_ids = [
            self.tokenizer.encode(label, add_special_tokens=False) for label in self.labels
        ]
        self.fast_labels = all(len(value) == 1 for value in self.label_ids)

    def _render(self, system: str, prompt: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return _apply_chat_template(
                self.tokenizer,
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(self.section.get("enable_thinking", False)),
            )
        return f"system: {system}\nuser: {prompt}\nassistant:"

    def generate_plans(
        self, targets: Sequence[Target], full_fields: bool
    ) -> list[dict[str, Any]]:
        prompts = [self._render(PLANNER_SYSTEM, build_retrieval_prompt(target, full_fields)) for target in targets]
        plans: list[dict[str, Any]] = []
        for start in range(0, len(prompts), self.batch_size):
            batch_targets = targets[start : start + self.batch_size]
            batch_prompts = prompts[start : start + self.batch_size]
            previous_padding_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = "left"
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            self.tokenizer.padding_side = previous_padding_side
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=int(self.section.get("planner_max_new_tokens", 120)),
                    do_sample=False,
                    pad_token_id=int(self.tokenizer.pad_token_id),
                    eos_token_id=int(self.tokenizer.eos_token_id),
                )
            width = encoded["input_ids"].shape[1]
            for target, row in zip(batch_targets, generated[:, width:]):
                text = self.tokenizer.decode(row, skip_special_tokens=True).strip()
                plan = normalize_plan(_extract_json(text), target, full_fields)
                plan["raw_response"] = text
                plans.append(plan)
            del encoded, generated
        return plans

    def score_emotions(self, prompts: Sequence[str]) -> list[list[float]]:
        rendered = [self._render(EMOTION_SYSTEM, prompt) for prompt in prompts]
        return self._score_single(rendered) if self.fast_labels else self._score_multi(rendered)

    def _score_single(self, prompts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        ids = torch.tensor([value[0] for value in self.label_ids], device="cuda")
        for start in range(0, len(prompts), self.batch_size):
            encoded = self.tokenizer(
                prompts[start : start + self.batch_size],
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            last = encoded["attention_mask"].sum(dim=1) - 1
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            rows = torch.arange(logits.shape[0], device="cuda")
            selected = logits[rows, last].index_select(-1, ids).float().cpu().numpy()
            results.extend(selected.tolist())
            del encoded, logits, selected
        return results

    def _score_multi(self, prompts: Sequence[str]) -> list[list[float]]:
        output = np.zeros((len(prompts), len(self.labels)), dtype=np.float64)
        pairs: list[tuple[int, int, list[int], int]] = []
        reserve = max(len(value) for value in self.label_ids)
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max(2, self.max_input_tokens - reserve),
            )
            for label_index, label_ids in enumerate(self.label_ids):
                pairs.append((prompt_index, label_index, prompt_ids + label_ids, len(prompt_ids)))
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            length = max(len(item[2]) for item in batch)
            input_ids = torch.full(
                (len(batch), length), int(self.tokenizer.pad_token_id), dtype=torch.long
            )
            attention = torch.zeros((len(batch), length), dtype=torch.long)
            for row, (_, _, ids, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1
            input_ids, attention = input_ids.to("cuda"), attention.to("cuda")
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, (prompt_index, label_index, ids, prompt_length) in enumerate(batch):
                values = [
                    log_probs[row, position - 1, ids[position]]
                    for position in range(prompt_length, len(ids))
                ]
                output[prompt_index, label_index] = float(torch.stack(values).mean().cpu())
            del input_ids, attention, logits, log_probs
        return output.tolist()


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64) / float(temperature)
    values -= values.max()
    exp = np.exp(values)
    return exp / exp.sum()


def _fit_temperature(scores: Sequence[Sequence[float]], gold_ids: Sequence[int]) -> float:
    score_array = np.asarray(scores, dtype=np.float64)
    gold = np.asarray(gold_ids, dtype=np.int64)
    coarse = np.geomspace(0.25, 6.0, 160)
    losses: list[float] = []
    for temperature in coarse:
        logits = score_array / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        losses.append(float(-log_probs[np.arange(len(gold)), gold].mean()))
    best = float(coarse[int(np.argmin(losses))])
    fine = np.linspace(max(0.05, best * 0.8), best * 1.2, 120)
    fine_losses: list[float] = []
    for temperature in fine:
        logits = score_array / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        fine_losses.append(float(-log_probs[np.arange(len(gold)), gold].mean()))
    return float(fine[int(np.argmin(fine_losses))])


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(-(values * np.log(np.clip(values, 1e-12, 1.0))).sum())


def _ece(probabilities: np.ndarray, gold: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    total = len(gold)
    result = 0.0
    for lower, upper in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float((predictions[mask] == gold[mask]).mean()) - float(confidence[mask].mean())
            )
    return result


def _metrics(
    score_rows: Sequence[Sequence[float]], gold_labels: Sequence[str], labels: Sequence[str], temperature: float
) -> dict[str, Any]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    gold = np.asarray([label_to_id[label] for label in gold_labels], dtype=np.int64)
    probabilities = np.stack([_softmax(row, temperature) for row in score_rows])
    predictions = probabilities.argmax(axis=1)
    one_hot = np.eye(len(labels))[gold]
    per_emotion = {
        label: float(f1_score(gold, predictions, labels=[index], average="macro", zero_division=0))
        for index, label in enumerate(labels)
    }
    return {
        "weighted_f1": float(f1_score(gold, predictions, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, predictions, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predictions)),
        "nll": float(-np.log(np.clip(probabilities[np.arange(len(gold)), gold], 1e-12, 1)).mean()),
        "brier": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
        "ece": _ece(probabilities, gold),
        "temperature": float(temperature),
        "per_emotion_f1": per_emotion,
        "predictions": [labels[index] for index in predictions],
        "probabilities": probabilities.tolist(),
    }


def _target_from_serialized(record: dict[str, Any]) -> Target:
    history = tuple(
        Utterance(str(item["utterance_id"]), str(item["speaker"]), str(item["text"]), "")
        for item in record.get("history", [])
    )
    return Target(
        str(record["dialogue_id"]),
        int(record["turn_index"]),
        str(record["utterance_id"]),
        str(record["speaker"]),
        str(record["text"]),
        str(record["gold"]),
        history,
    )


def _save_score_records(
    path: Path,
    targets: Sequence[Target],
    scores: Sequence[Sequence[float]],
    plans: Sequence[dict[str, Any]] | None = None,
) -> None:
    records = []
    for index, (target, score) in enumerate(zip(targets, scores)):
        record: dict[str, Any] = {
            "key": target.key,
            "dialogue_id": target.dialogue_id,
            "turn_index": target.turn_index,
            "gold": target.gold,
            "scores": list(map(float, score)),
        }
        if plans is not None:
            record["plan"] = plans[index]
        records.append(record)
    _write_jsonl(path, records)


def _load_inference_targets(root: Path, section: dict[str, Any], paths: dict[str, Path]) -> tuple[list[Target], list[Target]]:
    calibration = [
        _target_from_serialized(record)
        for record in _read_jsonl(paths["data"] / "router_calibration_targets.jsonl")
    ]
    validation_dialogues = read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl")
    validation = _targets(validation_dialogues, max_history=3)
    return calibration, validation


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
    output = paths["scores"] / f"seed{seed}_validation.json"
    if output.exists() and not overwrite:
        result = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    for variant in TRAIN_VARIANTS:
        if not (_adapter_dir(paths, seed, variant) / "training_complete.json").exists():
            raise FileNotFoundError(f"Missing trained adapter seed={seed} variant={variant}")
    calibration, validation = _load_inference_targets(root, section, paths)
    if max_targets is not None:
        calibration = calibration[:max_targets]
        validation = validation[:max_targets]
    labels = list(section.get("labels", LABELS))
    variant_scores: dict[str, dict[str, Any]] = {}
    plan_cache: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def run_adapter(train_variant: str, jobs: Sequence[tuple[str, str, bool]]) -> None:
        engine = LLMEngine(section, _adapter_dir(paths, seed, train_variant))
        generated_cache: dict[tuple[str, bool], list[dict[str, Any]]] = {}
        for report_name, mode, full_fields in jobs:
            plans_by_split: dict[str, list[dict[str, Any]]] = {}
            scores_by_split: dict[str, list[list[float]]] = {}
            for split_name, targets in (("calibration", calibration), ("validation", validation)):
                plans: list[dict[str, Any]] | None = None
                if mode in {"selected", "structured"}:
                    cache_key = (split_name, full_fields)
                    if cache_key not in generated_cache:
                        generated_cache[cache_key] = engine.generate_plans(
                            targets, full_fields=full_fields
                        )
                    plans = generated_cache[cache_key]
                    prompts = [
                        build_emotion_prompt(
                            target,
                            labels,
                            mode=mode,
                            selected_turn_id=plan["selected_evidence_turn_id"],
                            plan=plan,
                        )
                        for target, plan in zip(targets, plans)
                    ]
                    plans_by_split[split_name] = plans
                else:
                    prompts = [build_emotion_prompt(target, labels, mode=mode) for target in targets]
                scores = engine.score_emotions(prompts)
                scores_by_split[split_name] = scores
                _save_score_records(
                    paths["scores"] / f"seed{seed}_{report_name}_{split_name}.jsonl",
                    targets,
                    scores,
                    plans,
                )
            gold_cal = [labels.index(target.gold) for target in calibration]
            temperature = _fit_temperature(scores_by_split["calibration"], gold_cal)
            metrics = _metrics(
                scores_by_split["validation"],
                [target.gold for target in validation],
                labels,
                temperature,
            )
            metrics["raw_scores_validation"] = scores_by_split["validation"]
            metrics["raw_scores_calibration"] = scores_by_split["calibration"]
            variant_scores[report_name] = metrics
            if plans_by_split:
                plan_cache[report_name] = plans_by_split
        del engine
        gc.collect()
        torch.cuda.empty_cache()

    run_adapter("emotion_base", (("current_only", "current", False), ("always_recent3", "recent3", False)))
    run_adapter("generic_reasoning", (("generic_reasoning", "generic", False),))
    run_adapter("evidence_supervised", (("evidence_supervised", "selected", False),))
    run_adapter(
        "full_supervised",
        (("full_no_impact", "selected", True), ("full_supervised", "structured", True)),
    )
    run_adapter("shuffled_supervised", (("shuffled_supervised", "structured", True),))

    # Entropy routing threshold is selected only on the held-out MELD-train calibration dialogues.
    current_cal = variant_scores["current_only"]["raw_scores_calibration"]
    recent_cal = variant_scores["always_recent3"]["raw_scores_calibration"]
    current_val = variant_scores["current_only"]["raw_scores_validation"]
    recent_val = variant_scores["always_recent3"]["raw_scores_validation"]
    current_temp = variant_scores["current_only"]["temperature"]
    entropies = np.asarray([_entropy(_softmax(row, current_temp)) for row in current_cal])
    thresholds = np.unique(np.quantile(entropies, np.linspace(0.0, 1.0, 101)))
    best_threshold = float(thresholds[0])
    best_score = -math.inf
    for threshold in thresholds:
        routed = [
            recent if entropy >= threshold else current
            for current, recent, entropy in zip(current_cal, recent_cal, entropies)
        ]
        candidate = _metrics(routed, [target.gold for target in calibration], labels, 1.0)
        selection = 0.5 * candidate["macro_f1"] + 0.5 * candidate["weighted_f1"]
        if selection > best_score:
            best_score = selection
            best_threshold = float(threshold)
    val_entropy = [_entropy(_softmax(row, current_temp)) for row in current_val]
    routed_val = [
        recent if entropy >= best_threshold else current
        for current, recent, entropy in zip(current_val, recent_val, val_entropy)
    ]
    routed_cal = [
        recent if entropy >= best_threshold else current
        for current, recent, entropy in zip(current_cal, recent_cal, entropies)
    ]
    routed_temperature = _fit_temperature(
        routed_cal, [labels.index(target.gold) for target in calibration]
    )
    variant_scores["entropy_router"] = _metrics(
        routed_val, [target.gold for target in validation], labels, routed_temperature
    )
    variant_scores["entropy_router"]["threshold"] = best_threshold
    variant_scores["entropy_router"]["context_use_rate"] = float(
        np.mean(np.asarray(val_entropy) >= best_threshold)
    )

    teacher_calibration = {
        f"{item['dialogue_id']}::{int(item['target_turn_id'])}": item
        for item in _read_jsonl(paths["data"] / "teacher_calibration_annotations.jsonl")
    }
    plan_diagnostics: dict[str, Any] = {}
    target_lookup = {target.key: index for index, target in enumerate(calibration)}
    for report_name in ("evidence_supervised", "full_supervised", "shuffled_supervised"):
        plans = plan_cache.get(report_name, {}).get("calibration", [])
        gold_context: list[int] = []
        context_score: list[float] = []
        selected_correct: list[int] = []
        selected_correct_when_needed: list[int] = []
        parse_success: list[int] = []
        relation_gold: list[str] = []
        relation_pred: list[str] = []
        direction_gold: list[str] = []
        direction_pred: list[str] = []
        strength_gold: list[float] = []
        strength_pred: list[float] = []
        for key, annotation in teacher_calibration.items():
            if key not in target_lookup or target_lookup[key] >= len(plans):
                continue
            plan = plans[target_lookup[key]]
            helpful = annotation["context_need"] in {"history_helpful", "history_required"}
            gold_context.append(int(helpful))
            context_score.append(
                1.0 if plan["context_need"] == "history_required" else 0.75 if plan["context_need"] == "history_helpful" else 0.25 if plan["context_need"] == "ambiguous" else 0.0
            )
            selected_correct.append(
                int(plan.get("selected_evidence_turn_id") == annotation.get("selected_evidence_turn_id"))
            )
            if helpful:
                selected_correct_when_needed.append(
                    int(
                        plan.get("selected_evidence_turn_id")
                        == annotation.get("selected_evidence_turn_id")
                    )
                )
            parse_success.append(int(plan.get("parse_success", False)))
            if report_name != "evidence_supervised":
                chosen = _selected_assessment(annotation)
                relation_gold.append(chosen["role_relation"] if chosen else "unrelated")
                relation_pred.append(plan.get("role_relation", "uncertain"))
                direction_gold.append(chosen["impact_direction"] if chosen else "none")
                direction_pred.append(plan.get("impact_direction", "uncertain"))
                gold_strength = chosen["impact_strength"] if chosen and chosen["impact_strength"] is not None else 0
                pred_strength = plan.get("impact_strength")
                strength_gold.append(float(gold_strength))
                strength_pred.append(float(pred_strength if pred_strength is not None else 0))
        ap = (
            float(average_precision_score(gold_context, context_score))
            if len(set(gold_context)) > 1
            else None
        )
        diagnostics: dict[str, Any] = {
            "records": len(gold_context),
            "parse_success_rate": float(np.mean(parse_success)) if parse_success else None,
            "context_average_precision": ap,
            "selection_exact_accuracy_all": float(np.mean(selected_correct)) if selected_correct else None,
            "selected_evidence_accuracy": (
                float(np.mean(selected_correct_when_needed))
                if selected_correct_when_needed
                else None
            ),
        }
        if relation_gold:
            diagnostics.update(
                role_relation_macro_f1=float(
                    f1_score(relation_gold, relation_pred, average="macro", zero_division=0)
                ),
                impact_direction_macro_f1=float(
                    f1_score(direction_gold, direction_pred, average="macro", zero_division=0)
                ),
                impact_strength_mae=float(mean_absolute_error(strength_gold, strength_pred)),
            )
        plan_diagnostics[report_name] = diagnostics

    compact: dict[str, Any] = {}
    for name, metrics in variant_scores.items():
        compact[name] = {
            key: value
            for key, value in metrics.items()
            if key not in {"raw_scores_validation", "raw_scores_calibration", "probabilities", "predictions"}
        }
        compact[name]["predictions"] = metrics["predictions"]
        compact[name]["probabilities"] = metrics["probabilities"]
    result = {
        "status": "evaluated",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "split": "validation",
        "targets": len(validation),
        "dialogues": len({target.dialogue_id for target in validation}),
        "labels": labels,
        "variants": compact,
        "controller_diagnostics_on_heldout_train_dialogues": plan_diagnostics,
        "test_locked": True,
    }
    _write_json(output, result)
    print(f"Stage 7 seed {seed} validation written to {output}")
    return result


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    return {
        "values": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": len(values),
    }


def _bootstrap_advantage(
    proposed: Sequence[dict[str, Any]],
    control: Sequence[dict[str, Any]],
    gold: Sequence[str],
    dialogues: Sequence[str],
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, dialogue in enumerate(dialogues):
        groups[dialogue].append(index)
    ids = sorted(groups)
    rng = np.random.default_rng(seed)

    def metric(rows: Sequence[dict[str, Any]], indexes: Sequence[int], name: str) -> float:
        probabilities = np.asarray([rows[index]["probabilities"] for index in indexes])
        y_true = np.asarray([labels.index(gold[index]) for index in indexes])
        if name == "macro_f1":
            return float(f1_score(y_true, probabilities.argmax(1), average="macro", zero_division=0))
        if name == "weighted_f1":
            return float(f1_score(y_true, probabilities.argmax(1), average="weighted", zero_division=0))
        if name == "nll":
            return float(-np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-12, 1)).mean())
        raise ValueError(name)

    result: dict[str, Any] = {}
    all_indexes = list(range(len(gold)))
    for name in ("weighted_f1", "macro_f1", "nll"):
        sign = -1.0 if name == "nll" else 1.0
        point = sign * (metric(proposed, all_indexes, name) - metric(control, all_indexes, name))
        values: list[float] = []
        for _ in range(samples):
            chosen = rng.choice(ids, size=len(ids), replace=True)
            indexes = [index for dialogue in chosen for index in groups[str(dialogue)]]
            values.append(sign * (metric(proposed, indexes, name) - metric(control, indexes, name)))
        result[name] = {
            "point_advantage": point,
            "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "probability_positive": float(np.mean(np.asarray(values) > 0)),
        }
    return result


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    runs = []
    for seed in seeds:
        path = paths["scores"] / f"seed{seed}_validation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 7 validation seed report: {path}")
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    labels = list(section.get("labels", LABELS))
    validation_dialogues = read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl")
    targets = _targets(validation_dialogues, max_history=3)
    gold = [target.gold for target in targets]
    dialogue_ids = [target.dialogue_id for target in targets]
    aggregate: dict[str, Any] = {}
    for variant in REPORT_VARIANTS:
        aggregate[variant] = {}
        for metric in ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece", "temperature"):
            aggregate[variant][metric] = _aggregate(
                [float(run["variants"][variant][metric]) for run in runs]
            )
        aggregate[variant]["per_emotion_f1"] = {
            label: _aggregate(
                [float(run["variants"][variant]["per_emotion_f1"][label]) for run in runs]
            )
            for label in labels
        }
    diagnostics: dict[str, Any] = {}
    for variant in ("evidence_supervised", "full_supervised", "shuffled_supervised"):
        metric_names = sorted(
            {
                key
                for run in runs
                for key, value in run["controller_diagnostics_on_heldout_train_dialogues"][variant].items()
                if key != "records" and value is not None
            }
        )
        diagnostics[variant] = {
            name: _aggregate(
                [
                    float(run["controller_diagnostics_on_heldout_train_dialogues"][variant][name])
                    for run in runs
                    if run["controller_diagnostics_on_heldout_train_dialogues"][variant].get(name) is not None
                ]
            )
            for name in metric_names
        }

    # Average probabilities over fixed seeds before resampling dialogues.
    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in ("generic_reasoning", "evidence_supervised", "full_no_impact", "full_supervised", "shuffled_supervised"):
        mean_probabilities = np.mean(
            np.asarray([run["variants"][variant]["probabilities"] for run in runs]), axis=0
        )
        rows_by_variant[variant] = [
            {"probabilities": row.tolist()} for row in mean_probabilities
        ]
    controls = {
        "generic_reasoning": "generic_reasoning",
        "evidence_supervision": "evidence_supervised",
        "evidence_only_read": "full_no_impact",
        "shuffled_teacher": "shuffled_supervised",
    }
    bootstrap = {
        control: _bootstrap_advantage(
            rows_by_variant["full_supervised"],
            rows_by_variant[variant],
            gold,
            dialogue_ids,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 2088)) + offset,
        )
        for offset, (control, variant) in enumerate(controls.items())
    }
    _write_json(paths["bootstrap"], bootstrap)

    capacity_records: list[dict[str, Any]] = []
    for seed in seeds:
        for variant in TRAIN_VARIANTS:
            state_path = _adapter_dir(paths, seed, variant) / "training_complete.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            capacity_records.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "trainable_parameters": int(state["trainable_parameters"]),
                    "optimizer_steps": int(state["optimizer_steps"]),
                    "finite": bool(state["finite"]),
                }
            )
    branch_records = [
        item for item in capacity_records if item["variant"] != "emotion_base"
    ]
    capacity_audit = {
        "exact_trainable_parameter_match": len(
            {item["trainable_parameters"] for item in branch_records}
        )
        == 1,
        "exact_branch_optimizer_step_match": len(
            {item["optimizer_steps"] for item in branch_records}
        )
        == 1,
        "all_training_finite": all(item["finite"] for item in capacity_records),
        "records": capacity_records,
    }

    full = aggregate["full_supervised"]
    generic = aggregate["generic_reasoning"]
    evidence_sft = aggregate["evidence_supervised"]
    evidence_read = aggregate["full_no_impact"]
    shuffled = aggregate["shuffled_supervised"]
    full_diag = diagnostics.get("full_supervised", {})
    checks = {
        "full_preserves_current_weighted_f1": full["weighted_f1"]["mean"]
        >= aggregate["current_only"]["weighted_f1"]["mean"]
        - float(section.get("classification_tolerance", 0.003)),
        "full_beats_generic_macro": full["macro_f1"]["mean"]
        > generic["macro_f1"]["mean"],
        "full_training_adds_over_evidence_sft": full["macro_f1"]["mean"]
        > evidence_sft["macro_f1"]["mean"],
        "structured_fields_add_over_evidence_read": full["macro_f1"]["mean"]
        > evidence_read["macro_f1"]["mean"],
        "correct_teacher_beats_shuffled": full["macro_f1"]["mean"]
        > shuffled["macro_f1"]["mean"],
        "controller_parse_success": full_diag.get("parse_success_rate", {}).get("mean", 0.0)
        >= float(section.get("minimum_parse_success", 0.98)),
        "controller_context_ap": full_diag.get("context_average_precision", {}).get("mean", 0.0)
        >= float(section.get("minimum_context_ap", 0.55)),
        "controller_evidence_accuracy": full_diag.get("selected_evidence_accuracy", {}).get("mean", 0.0)
        >= float(section.get("minimum_evidence_accuracy", 0.45)),
        "capacity_and_training_audit": all(
            (
                capacity_audit["exact_trainable_parameter_match"],
                capacity_audit["exact_branch_optimizer_step_match"],
                capacity_audit["all_training_finite"],
            )
        ),
    }
    eligible = all(checks.values())
    decision = {
        "status": "eligible_for_frozen_test" if eligible else "stop_and_diagnose_llm_evidence_supervision",
        "eligible": eligible,
        "checks": checks,
        "selection_basis": "fixed dialogue-disjoint MELD-train calibration and untouched MELD validation",
        "next_action": (
            "Freeze full_supervised and prepare a separate confirmed test command."
            if eligible
            else "Keep MELD test locked; inspect failed retrieval/supervision contrast only."
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
        "controller_diagnostics_on_heldout_train_dialogues": diagnostics,
        "paired_dialogue_bootstrap": bootstrap,
        "capacity_audit": capacity_audit,
        "decision": decision,
        "teacher_status": "GPT-5.6 Sol generated and self-audited; not human gold",
        "test_locked": True,
    }
    _write_json(paths["summary_json"], result)
    _write_json(paths["decision"], decision)

    lines = [
        "# Stage 7 — LLM-Native Label-Supervised Evidence Reasoning",
        "",
        "> MELD validation-only. Extra annotations supervise retrieval/reasoning; they are absent at inference. Test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in REPORT_VARIANTS:
        row = aggregate[variant]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {row['weighted_f1']['mean']:.4f} ± {row['weighted_f1']['std']:.4f} "
            f"| {row['macro_f1']['mean']:.4f} ± {row['macro_f1']['std']:.4f} "
            f"| {row['nll']['mean']:.4f} ± {row['nll']['std']:.4f} "
            f"| {row['brier']['mean']:.4f} ± {row['brier']['std']:.4f} "
            f"| {row['ece']['mean']:.4f} ± {row['ece']['std']:.4f} |"
        )
    lines.extend(["", "## Validation per-emotion F1", ""])
    lines.append("| Emotion | " + " | ".join(VARIANT_LABELS[value] for value in REPORT_VARIANTS) + " |")
    lines.append("|---|" + "---:|" * len(REPORT_VARIANTS))
    for label in labels:
        cells = [
            f"{aggregate[variant]['per_emotion_f1'][label]['mean']:.4f} ± "
            f"{aggregate[variant]['per_emotion_f1'][label]['std']:.4f}"
            for variant in REPORT_VARIANTS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Controller diagnostics (held-out MELD-train dialogues)", ""])
    for variant, values in diagnostics.items():
        rendered = ", ".join(
            f"{name}={metric['mean']:.4f} ± {metric['std']:.4f}"
            for name, metric in values.items()
        )
        lines.append(f"- {VARIANT_LABELS[variant]}: {rendered}")
    lines.extend(["", "## Frozen decision", "", f"Status: **{decision['status']}**"])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "The teacher labels are GPT-5.6 Sol generated and self-audited operational labels, not human gold labels. Self-audit scores and provenance are excluded from training features.",
        ]
    )
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result


def train_all(config_path: Path, run_name: str, seeds: Sequence[int]) -> None:
    for seed in seeds:
        train_variant(config_path, "emotion_base", seed, run_name)
        for variant in TRAIN_VARIANTS[1:]:
            train_variant(config_path, variant, seed, run_name)


def status(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    state = {
        "prepared": paths["data_audit"].exists(),
        "adapters": {
            str(seed): {
                variant: (_adapter_dir(paths, seed, variant) / "training_complete.json").exists()
                for variant in TRAIN_VARIANTS
            }
            for seed in seeds
        },
        "evaluations": {
            str(seed): (paths["scores"] / f"seed{seed}_validation.json").exists()
            for seed in seeds
        },
        "report": paths["summary_json"].exists(),
        "test_locked": True,
    }
    print(json.dumps(state, indent=2))
    return state


def final_test_locked() -> None:
    raise RuntimeError(
        "Stage 7 v2.0.0 intentionally cannot read MELD test. First finish validation, "
        "satisfy the pre-registered gate, freeze one method, and create a separate hash-locked test patch."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("prepare")
    subparsers.add_parser("status")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--variant", choices=TRAIN_VARIANTS, required=True)
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--max-examples", type=int)
    train_parser.add_argument("--overwrite", action="store_true")
    all_parser = subparsers.add_parser("train-all")
    all_parser.add_argument("--seeds", type=int, nargs="*")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--max-targets", type=int)
    evaluate_parser.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("report")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--seed", type=int, default=42)
    subparsers.add_parser("final-test")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.config, args.run_name)
    elif args.command == "prepare":
        prepare_training_data(args.config, args.run_name)
    elif args.command == "status":
        status(args.config, args.run_name)
    elif args.command == "train":
        train_variant(
            args.config,
            args.variant,
            args.seed,
            args.run_name,
            max_steps_override=args.max_steps,
            max_examples=args.max_examples,
            overwrite=args.overwrite,
        )
    elif args.command == "train-all":
        section = _section(_read_yaml(args.config))
        seeds = args.seeds or [int(value) for value in section.get("seeds", [42, 43, 44])]
        train_all(args.config, args.run_name, seeds)
    elif args.command == "evaluate":
        evaluate_seed(
            args.config,
            args.seed,
            args.run_name,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
        )
    elif args.command == "report":
        report(args.config, args.run_name)
    elif args.command == "smoke":
        smoke_name = f"smoke_seed{args.seed}"
        prepare_training_data(args.config, smoke_name)
        train_variant(args.config, "emotion_base", args.seed, smoke_name, max_steps_override=2, max_examples=16, overwrite=True)
        for variant in TRAIN_VARIANTS[1:]:
            train_variant(args.config, variant, args.seed, smoke_name, max_steps_override=2, max_examples=16, overwrite=True)
        evaluate_seed(args.config, args.seed, smoke_name, max_targets=8, overwrite=True)
        print(json.dumps({"status": "smoke_passed", "run_name": smoke_name, "test_locked": True}, indent=2))
    elif args.command == "final-test":
        final_test_locked()


if __name__ == "__main__":
    main()
