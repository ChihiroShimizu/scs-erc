"""Stage 7.2: early current-only, label-and-utility supervised context gate.

The router sees only the current utterance and exits after the first few Qwen
decoder layers.  It then chooses exactly one complete emotion path: current
only (SKIP) or causal recent-three (READ).  The emotion LoRA inherited from
Stage 7.1 is frozen.  Teacher context labels and MELD-train counterfactual
utility outcomes supervise only a small gate head and are absent at inference.

This module has no MELD-test implementation.  Validation thresholds are fitted
on dialogue-disjoint MELD-train calibration data and test remains hard-locked.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
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
from sklearn.metrics import average_precision_score

from pfa_erc.data import read_jsonl
from pfa_erc.training.llm_evidence_supervision_stage7 import (
    LABELS,
    Target,
    _annotation_object,
    _bootstrap_advantage,
    _fit_temperature,
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
from pfa_erc.training.llm_current_only_gate_stage7_1 import (
    AdaptiveLLMEngine,
    VERSION as STAGE7_1_VERSION,
    _select_entropy_threshold,
    _serialize_target,
)


VERSION = "1.0.1"
EXPERIMENT = "llm_early_utility_gate_stage7_2"

HEAD_VARIANTS = (
    "semantic_gate",
    "utility_gate",
    "joint_gate",
    "shuffled_joint_gate",
)
REPORT_VARIANTS = (
    "base_current",
    "base_recent3",
    "entropy_adaptive",
    "semantic_adaptive",
    "utility_adaptive",
    "joint_adaptive",
    "matched_shuffled",
    "matched_random",
)
VARIANT_LABELS = {
    "base_current": "7.2A frozen-base current-only",
    "base_recent3": "7.2B frozen-base always recent-3",
    "entropy_adaptive": "7.2C full-pass entropy router",
    "semantic_adaptive": "7.2D early semantic-label gate",
    "utility_adaptive": "7.2E early counterfactual-utility gate",
    "joint_adaptive": "7.2F early joint label+utility gate",
    "matched_shuffled": "7.2G matched-rate shuffled-label gate",
    "matched_random": "7.2H matched-rate random gate",
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
    value = config.get("llm_early_utility_gate_stage7_2")
    if not isinstance(value, dict):
        raise KeyError("Missing llm_early_utility_gate_stage7_2 configuration section")
    return value


def _paths(
    root: Path, section: dict[str, Any], run_name: str = "validation"
) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get(
            "artifact_dir", "artifacts/llm_early_utility_gate_stage7_2/meld"
        ),
    )
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "data": base / "prepared",
        "cache": base / "cache",
        "heads": base / "heads",
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


def _stage7_1_adapter(
    root: Path, section: dict[str, Any], seed: int
) -> Path:
    base = _resolve(
        root,
        section.get(
            "stage7_1_artifact_dir",
            "artifacts/llm_current_only_gate_stage7_1/meld",
        ),
    )
    return base / "adapters" / f"seed{seed}" / "emotion_base"


def _cache_dir(paths: dict[str, Path], seed: int) -> Path:
    return paths["cache"] / f"seed{seed}"


def _head_dir(paths: dict[str, Path], seed: int, variant: str) -> Path:
    return paths["heads"] / f"seed{seed}" / variant


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def teacher_read_label(context_need: str) -> int:
    if context_need == "current_sufficient":
        return 0
    if context_need in {"history_helpful", "history_required", "ambiguous"}:
        return 1
    raise ValueError(f"Unknown context_need value: {context_need}")


def teacher_sample_weight(context_need: str) -> float:
    return {
        "current_sufficient": 1.0,
        "history_helpful": 1.5,
        "history_required": 2.0,
        "ambiguous": 1.0,
    }[context_need]


def _log_probability(scores: Sequence[float], gold_index: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values -= values.max()
    return float(values[gold_index] - np.log(np.exp(values).sum()))


def counterfactual_utility_target(
    current_scores: Sequence[float],
    recent_scores: Sequence[float],
    gold_index: int,
    current_tokens: int,
    recent_tokens: int,
    history_available: bool,
    *,
    cost_lambda: float,
    read_margin: float,
    weight_scale: float,
    maximum_weight: float,
) -> dict[str, Any]:
    """Create an out-of-sample, cost-aware READ target from both actions.

    Gold is used only to construct the training reward.  The inference gate
    receives neither gold nor either complete-path score.
    """
    current_nll = -_log_probability(current_scores, gold_index)
    recent_nll = -_log_probability(recent_scores, gold_index)
    nll_gain = current_nll - recent_nll
    current_prediction = int(np.argmax(current_scores))
    recent_prediction = int(np.argmax(recent_scores))
    current_correct = current_prediction == int(gold_index)
    recent_correct = recent_prediction == int(gold_index)
    extra_ratio = max(0.0, (float(recent_tokens) - float(current_tokens))) / max(
        float(recent_tokens), 1.0
    )
    net_gain = nll_gain - float(cost_lambda) * extra_ratio
    if not history_available:
        read = False
        rule = "no_history_forces_skip"
    elif recent_correct and not current_correct:
        read = True
        rule = "recent_corrects_current_error"
    elif current_correct and not recent_correct:
        read = False
        rule = "recent_harms_correct_current"
    else:
        read = net_gain > float(read_margin)
        rule = "cost_adjusted_nll_gain"
    scale = max(float(weight_scale), 1e-8)
    weight = 1.0 + min(abs(net_gain) / scale, max(0.0, maximum_weight - 1.0))
    return {
        "label": int(read),
        "sample_weight": float(weight),
        "current_nll": float(current_nll),
        "recent_nll": float(recent_nll),
        "nll_gain": float(nll_gain),
        "extra_token_ratio": float(extra_ratio),
        "cost_adjusted_gain": float(net_gain),
        "current_correct": bool(current_correct),
        "recent_correct": bool(recent_correct),
        "rule": rule,
    }


def layer_equivalent_cost_ratio(
    current_tokens: Sequence[int],
    recent_tokens: Sequence[int],
    read_mask: Sequence[bool],
    early_layers: int,
    total_layers: int,
) -> float:
    if not (
        len(current_tokens) == len(recent_tokens) == len(read_mask)
    ):
        raise ValueError("Cost inputs are not aligned")
    if total_layers <= 0 or not 0 < early_layers < total_layers:
        raise ValueError("early_layers must be between zero and total_layers")
    gate_fraction = float(early_layers) / float(total_layers)
    adaptive = 0.0
    baseline = float(sum(recent_tokens))
    for current, recent, read in zip(current_tokens, recent_tokens, read_mask):
        adaptive += float(current) * gate_fraction
        adaptive += float(recent if read else current)
    return adaptive / baseline if baseline else 0.0


def _normalise_source_weights(values: Sequence[float], target_mass: float) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or float(array.sum()) <= 0:
        raise ValueError("Cannot normalise empty or non-positive sample weights")
    return (array * (float(target_mass) / float(array.sum()))).tolist()


def _derangement_permutation(labels: Sequence[int], seed: int) -> tuple[list[int], float]:
    original = [int(value) for value in labels]
    if len(original) < 2 or len(set(original)) < 2:
        return list(range(len(original))), 0.0
    rng = random.Random(int(seed))
    best = list(range(len(original)))
    best_changed = -1
    for _ in range(2000):
        candidate = list(range(len(original)))
        rng.shuffle(candidate)
        changed = sum(
            original[index] != original[source]
            for index, source in enumerate(candidate)
        )
        if changed > best_changed:
            best = candidate
            best_changed = changed
        if changed == len(original):
            break
    return best, best_changed / len(original)


def _derange_binary(labels: Sequence[int], seed: int) -> tuple[list[int], float]:
    permutation, changed = _derangement_permutation(labels, seed)
    return [int(labels[index]) for index in permutation], changed


def _exact_top_k_mask(
    probabilities: Sequence[float], history_available: Sequence[bool], count: int
) -> np.ndarray:
    probability = np.asarray(probabilities, dtype=np.float64)
    history = np.asarray(history_available, dtype=bool)
    candidates = np.flatnonzero(history)
    requested = max(0, min(int(count), len(candidates)))
    order = sorted(candidates.tolist(), key=lambda index: (-probability[index], index))
    mask = np.zeros(len(probability), dtype=bool)
    if requested:
        mask[np.asarray(order[:requested], dtype=np.int64)] = True
    return mask


def _exact_random_mask(
    history_available: Sequence[bool], count: int, seed: int
) -> np.ndarray:
    history = np.asarray(history_available, dtype=bool)
    candidates = np.flatnonzero(history).tolist()
    rng = random.Random(int(seed))
    rng.shuffle(candidates)
    mask = np.zeros(len(history), dtype=bool)
    for index in candidates[: max(0, min(int(count), len(candidates)))]:
        mask[index] = True
    return mask


def preflight(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if str(section.get("dataset", "meld")).lower() != "meld":
        raise ValueError("Stage 7.2 accepts MELD only")
    if STAGE7_1_VERSION != "2.1.1":
        raise ValueError(
            f"Stage 7.2 requires Stage 7.1 v2.1.1, found {STAGE7_1_VERSION}"
        )
    if section.get("test_locked") is not True:
        raise ValueError("Stage 7.2 requires test_locked: true")
    early_layers = int(section.get("early_exit_layers", 2))
    if not 1 <= early_layers <= int(section.get("maximum_allowed_early_layers", 2)):
        raise ValueError("early_exit_layers exceeds the preregistered lightweight limit")
    processed = _resolve(root, section["processed_dir"])
    teacher_path = _resolve(root, section["teacher_annotation_file"])
    required = [processed / "train.jsonl", processed / "validation.jsonl", teacher_path]
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    adapters = [_stage7_1_adapter(root, section, seed) for seed in seeds]
    required.extend(adapter / "training_complete.json" for adapter in adapters)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 7.2 dependency: " + ", ".join(missing))
    if "test" in str(teacher_path).lower():
        raise ValueError("Teacher annotation path may not reference MELD test")
    raw_teacher = _read_jsonl(teacher_path)
    expected = int(section.get("expected_teacher_records", 500))
    if len(raw_teacher) != expected:
        raise ValueError(f"Expected {expected} teacher records, found {len(raw_teacher)}")
    actual_hash = _sha256(teacher_path)
    expected_hash = str(section.get("expected_teacher_sha256", "")).lower().strip()
    if expected_hash and actual_hash != expected_hash:
        raise ValueError(
            f"Teacher SHA256 mismatch: expected {expected_hash}, found {actual_hash}"
        )
    train_dialogues = read_jsonl(processed / "train.jsonl")
    validation_dialogues = read_jsonl(processed / "validation.jsonl")
    mapping = _target_map(train_dialogues, max_history=3)
    seen: set[str] = set()
    issues: list[str] = []
    for raw in raw_teacher:
        annotation = _annotation_object(raw)
        key = f"{annotation.get('dialogue_id')}::{int(annotation.get('target_turn_id', -1))}"
        if key in seen:
            issues.append(f"duplicate teacher key {key}")
            continue
        seen.add(key)
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
    base_states = []
    for seed, adapter in zip(seeds, adapters):
        state = json.loads(
            (adapter / "training_complete.json").read_text(encoding="utf-8")
        )
        if (
            state.get("variant") != "emotion_base"
            or not state.get("finite", False)
            or state.get("base_model") != section["model_name"]
        ):
            raise ValueError(f"Invalid Stage 7.1 emotion base: {adapter}")
        base_states.append(
            {
                "seed": seed,
                "adapter": str(adapter),
                "tree_sha256": _tree_sha256(adapter),
                "optimizer_steps": state.get("optimizer_steps"),
            }
        )
    result = {
        "status": "ready",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "model_name": section["model_name"],
        "model_family": "Qwen causal LLM with early hidden-state gate",
        "roberta_or_bert_used": False,
        "gate_input": "current utterance only",
        "history_visible_before_route": False,
        "complete_emotion_forwards_per_target": 1,
        "early_exit_layers": early_layers,
        "emotion_base_frozen": True,
        "teacher_source": "GPT-5.6 Sol generated and self-audited",
        "teacher_records": len(raw_teacher),
        "teacher_sha256": actual_hash,
        "stage7_1_emotion_bases": base_states,
        "train_dialogues": len(train_dialogues),
        "validation_dialogues": len(validation_dialogues),
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
            "seeds": seeds,
            "base_states": base_states,
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
    train_ids, stage7_1_holdout_ids = _split_dialogues(
        [dialogue.dialogue_id for dialogue in train_dialogues],
        float(section.get("stage7_1_calibration_dialogue_fraction", 0.20)),
        int(section.get("stage7_1_split_seed", 2171)),
    )
    utility_ids, threshold_ids = _split_dialogues(
        sorted(stage7_1_holdout_ids),
        float(section.get("threshold_fraction_of_stage7_1_holdout", 0.40)),
        int(section.get("utility_split_seed", 7212)),
    )
    if train_ids & utility_ids or train_ids & threshold_ids or utility_ids & threshold_ids:
        raise AssertionError("Stage 7.2 dialogue partitions overlap")
    teacher_train = [
        item for item in annotations if str(item["dialogue_id"]) in train_ids
    ]
    teacher_threshold = [
        item for item in annotations if str(item["dialogue_id"]) in threshold_ids
    ]
    semantic_targets = [
        target_map[f"{item['dialogue_id']}::{int(item['target_turn_id'])}"]
        for item in teacher_train
    ]
    all_targets = list(target_map.values())
    utility_targets = [
        target for target in all_targets if target.dialogue_id in utility_ids
    ]
    threshold_targets = [
        target for target in all_targets if target.dialogue_id in threshold_ids
    ]
    _write_jsonl(
        paths["data"] / "semantic_train_targets.jsonl",
        (_serialize_target(target) for target in semantic_targets),
    )
    _write_jsonl(
        paths["data"] / "utility_train_targets.jsonl",
        (_serialize_target(target) for target in utility_targets),
    )
    _write_jsonl(
        paths["data"] / "threshold_calibration_targets.jsonl",
        (_serialize_target(target) for target in threshold_targets),
    )
    _write_jsonl(paths["data"] / "teacher_train_annotations.jsonl", teacher_train)
    _write_jsonl(
        paths["data"] / "teacher_threshold_annotations.jsonl", teacher_threshold
    )
    audit = {
        "status": "passed",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "split_policy": (
            "reuse Stage 7.1 dialogue-disjoint 80/20 split; divide the untouched "
            "20% into utility-training and threshold-calibration dialogues"
        ),
        "stage7_1_split_seed": int(section.get("stage7_1_split_seed", 2171)),
        "utility_split_seed": int(section.get("utility_split_seed", 7212)),
        "model_train_dialogues": len(train_ids),
        "utility_train_dialogues": len(utility_ids),
        "threshold_calibration_dialogues": len(threshold_ids),
        "semantic_teacher_examples": len(semantic_targets),
        "counterfactual_utility_examples": len(utility_targets),
        "threshold_calibration_examples": len(threshold_targets),
        "teacher_threshold_examples": len(teacher_threshold),
        "partitions_pairwise_disjoint": True,
        "utility_predictions_out_of_sample_for_stage7_1_emotion_base": True,
        "gate_prompt_contains_history": False,
        "gate_prompt_contains_gold_emotion": False,
        "gold_used_only_to_construct_train_utility_reward": True,
        "teacher_labels_hidden_at_validation_inference": True,
        "validation_used_for_threshold_selection": False,
        "test_locked": True,
    }
    _write_json(paths["data_audit"], audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


class EarlyBinaryGateHead(nn.Module):
    """Small binary router operating on an early Qwen hidden state."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck: int = 256,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(int(hidden_size))
        self.projection = nn.Sequential(
            nn.Linear(int(hidden_size), int(bottleneck)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(bottleneck), 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden.float())).squeeze(-1)


class _EarlyExitSignal(RuntimeError):
    pass


def _get_nested_attribute(value: Any, path: str) -> Any:
    current = value
    for component in path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def locate_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Locate Qwen decoder blocks across PEFT/Transformers wrapper versions."""
    candidates = (
        "base_model.model.model.layers",
        "base_model.model.model.model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        layers = _get_nested_attribute(model, path)
        if isinstance(layers, (nn.ModuleList, list, tuple)) and len(layers) > 1:
            return layers
    for name, module in model.named_modules():
        if name.endswith("model.layers") and isinstance(module, nn.ModuleList):
            if len(module) > 1:
                return module
    raise RuntimeError(
        "Could not locate Qwen decoder layers. Stage 7.2 supports the PEFT "
        "Qwen2/Qwen2.5 wrapper layouts tested with transformers>=4.51,<5."
    )


def _last_non_padding_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the final attended token for both left- and right-padded batches.

    Stage 7.2 v1.0.0 used ``attention_mask.sum(1) - 1``, which is correct only
    for right padding.  Qwen deployments may use left padding, so the old
    expression could select an interior or padding position for shorter rows.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    if bool((attention_mask.sum(dim=1) <= 0).any()):
        raise ValueError("Every encoded prompt must contain an attended token")
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    masked = positions.masked_fill(
        attention_mask.to(dtype=torch.bool).logical_not(), -1
    )
    return masked.max(dim=1).values


@dataclass
class EarlyFeatureResult:
    features: torch.Tensor
    token_counts: list[int]
    elapsed_seconds: float


class EarlyFeatureExtractor:
    """Abort Qwen immediately after a configured decoder layer."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        early_layers: int,
        max_input_tokens: int,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = locate_decoder_layers(model)
        self.total_layers = len(self.layers)
        self.early_layers = int(early_layers)
        self.max_input_tokens = int(max_input_tokens)
        if not 0 < self.early_layers < self.total_layers:
            raise ValueError(
                f"early_layers={self.early_layers} is invalid for "
                f"total_layers={self.total_layers}"
            )

    def extract(
        self,
        prompts: Sequence[str],
        *,
        batch_size: int,
        keep_on_device: bool = False,
    ) -> EarlyFeatureResult:
        collected: list[torch.Tensor] = []
        token_counts: list[int] = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(prompts), int(batch_size)):
            encoded = self.tokenizer(
                list(prompts[start : start + int(batch_size)]),
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
            last_positions = _last_non_padding_positions(encoded["attention_mask"])
            rows = torch.arange(len(last_positions), device="cuda")
            captured: dict[str, torch.Tensor] = {}

            def stop_after_layer(
                _module: nn.Module, _inputs: tuple[Any, ...], output: Any
            ) -> None:
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not isinstance(hidden, torch.Tensor):
                    raise TypeError("Qwen decoder layer did not return a tensor")
                captured["hidden"] = hidden
                raise _EarlyExitSignal()

            handle = self.layers[self.early_layers - 1].register_forward_hook(
                stop_after_layer
            )
            try:
                with torch.inference_mode():
                    self.model(**encoded, use_cache=False, output_hidden_states=False)
            except _EarlyExitSignal:
                pass
            finally:
                handle.remove()
            if "hidden" not in captured:
                raise RuntimeError("Early-exit hook did not execute")
            selected = captured["hidden"][rows, last_positions].detach()
            collected.append(selected if keep_on_device else selected.cpu())
            del encoded, captured, selected
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if not collected:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            device = "cuda" if keep_on_device else "cpu"
            features = torch.empty((0, hidden_size), device=device)
        else:
            features = torch.cat(collected, dim=0)
        return EarlyFeatureResult(features, token_counts, elapsed)


def _load_targets(path: Path) -> list[Target]:
    return [_target_from_serialized(item) for item in _read_jsonl(path)]


def _cache_group(
    engine: AdaptiveLLMEngine,
    extractor: EarlyFeatureExtractor,
    targets: Sequence[Target],
    *,
    include_complete_scores: bool,
    batch_size: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    prompts = engine.prompts(targets, "current")
    feature_result = extractor.extract(
        prompts, batch_size=batch_size, keep_on_device=False
    )
    record: dict[str, Any] = {
        "keys": [target.key for target in targets],
        "gold": [target.gold for target in targets],
        "history_available": [bool(target.history) for target in targets],
        "gate_current_tokens": feature_result.token_counts,
        "early_feature_seconds": feature_result.elapsed_seconds,
    }
    if include_complete_scores:
        current = engine.forward_targets(
            targets, "current", need_gate=False, batch_size=batch_size
        )
        recent = engine.forward_targets(
            targets, "recent3", need_gate=False, batch_size=batch_size
        )
        record.update(
            {
                "current_scores": current.scores,
                "recent_scores": recent.scores,
                "current_tokens": current.token_counts,
                "recent_tokens": recent.token_counts,
                "current_forward_seconds": current.elapsed_seconds,
                "recent_forward_seconds": recent.elapsed_seconds,
            }
        )
    return feature_result.features.to(dtype=torch.float16).contiguous(), record


def cache_seed(
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
    output_dir = _cache_dir(paths, seed)
    done = output_dir / "cache_complete.json"
    if done.exists() and not overwrite:
        result = json.loads(done.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if output_dir.exists() and overwrite:
        backup = output_dir.with_name(
            output_dir.name + ".backup_" + time.strftime("%Y%m%dT%H%M%S")
        )
        shutil.move(str(output_dir), str(backup))
    groups = {
        "semantic_train": _load_targets(
            paths["data"] / "semantic_train_targets.jsonl"
        ),
        "utility_train": _load_targets(
            paths["data"] / "utility_train_targets.jsonl"
        ),
        "threshold_calibration": _load_targets(
            paths["data"] / "threshold_calibration_targets.jsonl"
        ),
        "validation": _targets(
            read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl"),
            max_history=3,
        ),
    }
    if max_targets is not None:
        limit = int(max_targets)
        teacher_keys = {
            f"{item['dialogue_id']}::{int(item['target_turn_id'])}"
            for item in _read_jsonl(
                paths["data"] / "teacher_threshold_annotations.jsonl"
            )
        }
        diagnostic_first = sorted(
            groups["threshold_calibration"],
            key=lambda target: (target.key not in teacher_keys, target.key),
        )
        groups["threshold_calibration"] = diagnostic_first[:limit]
        for name in ("semantic_train", "utility_train", "validation"):
            groups[name] = groups[name][:limit]
    adapter = _stage7_1_adapter(root, section, seed)
    engine = AdaptiveLLMEngine(section, adapter, with_gate=False)
    extractor = EarlyFeatureExtractor(
        engine.model,
        engine.tokenizer,
        int(section.get("early_exit_layers", 2)),
        int(section.get("max_sequence_tokens", 768)),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    group_files: dict[str, Any] = {}
    hidden_size: int | None = None
    batch_size = int(section.get("feature_batch_size", 8))
    for name, targets in groups.items():
        print(
            f"[Stage7.2] seed={seed} caching {name}: {len(targets)} targets",
            flush=True,
        )
        tensor, record = _cache_group(
            engine,
            extractor,
            targets,
            include_complete_scores=name != "semantic_train",
            batch_size=batch_size,
        )
        if hidden_size is None:
            hidden_size = int(tensor.shape[1])
        elif hidden_size != int(tensor.shape[1]):
            raise ValueError("Early feature hidden sizes differ across partitions")
        feature_path = output_dir / f"{name}_features.safetensors"
        metadata_path = output_dir / f"{name}_metadata.json"
        save_file({"features": tensor}, str(feature_path))
        _write_json(metadata_path, record)
        group_files[name] = {
            "features": str(feature_path),
            "features_sha256": _sha256(feature_path),
            "metadata": str(metadata_path),
            "metadata_sha256": _sha256(metadata_path),
        }
        del tensor, record
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "adapter": str(adapter),
        "adapter_tree_sha256": _tree_sha256(adapter),
        "early_exit_layers": extractor.early_layers,
        "total_decoder_layers": extractor.total_layers,
        "hidden_size": int(hidden_size or 0),
        "groups": {name: int(len(values)) for name, values in groups.items()},
        "complete_emotion_scores_cached_for": [
            name for name in groups if name != "semantic_train"
        ],
        "semantic_train_gold_scored": False,
        "group_files": group_files,
        "test_data_read": False,
    }
    _write_json(done, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del engine, extractor
    gc.collect()
    torch.cuda.empty_cache()
    return result


@dataclass
class HeadTrainingRows:
    features: torch.Tensor
    labels: torch.Tensor
    weights: torch.Tensor
    sources: list[str]


def _utility_rows_from_cache(
    section: dict[str, Any], metadata: dict[str, Any], labels: Sequence[str]
) -> tuple[list[int], list[float], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    targets: list[int] = []
    weights: list[float] = []
    for index, gold in enumerate(metadata["gold"]):
        row = counterfactual_utility_target(
            metadata["current_scores"][index],
            metadata["recent_scores"][index],
            labels.index(gold),
            int(metadata["current_tokens"][index]),
            int(metadata["recent_tokens"][index]),
            bool(metadata["history_available"][index]),
            cost_lambda=float(section.get("utility_cost_lambda", 0.05)),
            read_margin=float(section.get("utility_read_margin", 0.0)),
            weight_scale=float(section.get("utility_weight_scale", 0.20)),
            maximum_weight=float(section.get("utility_maximum_weight", 3.0)),
        )
        row["key"] = metadata["keys"][index]
        rows.append(row)
        targets.append(int(row["label"]))
        weights.append(float(row["sample_weight"]))
    return targets, weights, rows


def _assemble_head_rows(
    variant: str,
    semantic_features: torch.Tensor,
    semantic_labels: Sequence[int],
    semantic_weights: Sequence[float],
    utility_features: torch.Tensor,
    utility_labels: Sequence[int],
    utility_weights: Sequence[float],
    shuffle_seed: int,
) -> tuple[HeadTrainingRows, dict[str, Any]]:
    if variant not in HEAD_VARIANTS:
        raise ValueError(f"Unknown Stage 7.2 head variant: {variant}")
    semantic_y = list(map(int, semantic_labels))
    utility_y = list(map(int, utility_labels))
    semantic_w = list(map(float, semantic_weights))
    utility_w = list(map(float, utility_weights))
    changed = {"semantic": 0.0, "utility": 0.0}
    if variant == "shuffled_joint_gate":
        semantic_permutation, changed["semantic"] = _derangement_permutation(
            semantic_y, int(shuffle_seed)
        )
        utility_permutation, changed["utility"] = _derangement_permutation(
            utility_y, int(shuffle_seed) + 1
        )
        semantic_y = [semantic_y[index] for index in semantic_permutation]
        semantic_w = [semantic_w[index] for index in semantic_permutation]
        utility_y = [utility_y[index] for index in utility_permutation]
        utility_w = [utility_w[index] for index in utility_permutation]
    if variant == "semantic_gate":
        x = semantic_features
        y = semantic_y
        w = _normalise_source_weights(semantic_w, 1.0)
        sources = ["teacher_semantic"] * len(y)
    elif variant == "utility_gate":
        x = utility_features
        y = utility_y
        w = _normalise_source_weights(utility_w, 1.0)
        sources = ["counterfactual_utility"] * len(y)
    else:
        x = torch.cat([semantic_features, utility_features], dim=0)
        y = semantic_y + utility_y
        w = _normalise_source_weights(semantic_w, 0.5) + _normalise_source_weights(
            utility_w, 0.5
        )
        sources = ["teacher_semantic"] * len(semantic_y) + [
            "counterfactual_utility"
        ] * len(utility_y)
    result = HeadTrainingRows(
        features=x.float(),
        labels=torch.tensor(y, dtype=torch.float32),
        weights=torch.tensor(w, dtype=torch.float32),
        sources=sources,
    )
    audit = {
        "examples": len(y),
        "read_prevalence": float(np.mean(y)) if y else None,
        "source_distribution": dict(Counter(sources)),
        "shuffled_changed_rate": changed,
        "shuffle_preserves_semantic_marginals": Counter(semantic_y)
        == Counter(semantic_labels),
        "shuffle_preserves_utility_marginals": Counter(utility_y)
        == Counter(utility_labels),
        "shuffle_moves_label_weight_pairs_together": variant
        == "shuffled_joint_gate",
    }
    return result, audit


def _train_one_head(
    section: dict[str, Any],
    rows: HeadTrainingRows,
    seed: int,
    output_dir: Path,
    variant: str,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = EarlyBinaryGateHead(
        int(rows.features.shape[1]),
        int(section.get("gate_bottleneck", 256)),
        float(section.get("gate_dropout", 0.10)),
    ).to(device)
    features = rows.features
    targets = rows.labels
    weights = rows.weights
    source_indexes = {
        source: torch.tensor(
            [index for index, value in enumerate(rows.sources) if value == source],
            dtype=torch.long,
        )
        for source in sorted(set(rows.sources))
    }
    batch_size = int(section.get("gate_batch_size", 64))
    steps = int(max_steps_override or section.get("gate_max_steps", 500))
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(section.get("gate_learning_rate", 3.0e-4)),
        weight_decay=float(section.get("gate_weight_decay", 0.01)),
    )
    from transformers import get_cosine_schedule_with_warmup

    warmup = min(int(section.get("gate_warmup_steps", 25)), max(0, steps - 1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)
    losses: list[float] = []
    head.train()
    started = time.time()
    for step in range(steps):
        effective_batch = min(batch_size, len(features))
        if len(source_indexes) == 1:
            pool = next(iter(source_indexes.values()))
            positions = torch.randint(
                0, len(pool), (effective_batch,), generator=generator
            )
            indexes = pool.index_select(0, positions)
        else:
            names = sorted(source_indexes)
            first_count = effective_batch // 2
            counts = [first_count, effective_batch - first_count]
            sampled = []
            for name, count in zip(names, counts):
                pool = source_indexes[name]
                positions = torch.randint(
                    0, len(pool), (count,), generator=generator
                )
                sampled.append(pool.index_select(0, positions))
            indexes = torch.cat(sampled)
            indexes = indexes.index_select(
                0, torch.randperm(len(indexes), generator=generator)
            )
        x = features.index_select(0, indexes).to(device, non_blocking=True)
        y = targets.index_select(0, indexes).to(device, non_blocking=True)
        w = weights.index_select(0, indexes).to(device, non_blocking=True)
        logits = head(x)
        raw = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        loss = (raw * w).sum() / w.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite Stage 7.2 loss variant={variant} step={step}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head.parameters(), float(section.get("gate_max_grad_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        if step == 0 or (step + 1) % int(section.get("log_every_steps", 50)) == 0:
            print(
                f"[Stage7.2] {variant} step={step + 1}/{steps} "
                f"loss={np.mean(losses[-50:]):.6f}",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in head.state_dict().items()},
        str(output_dir / "gate_head.safetensors"),
    )
    _write_json(
        output_dir / "gate_head_config.json",
        {
            "hidden_size": int(rows.features.shape[1]),
            "bottleneck": int(section.get("gate_bottleneck", 256)),
            "dropout": float(section.get("gate_dropout", 0.10)),
            "classes": ["SKIP", "READ"],
            "input": f"current-only Qwen hidden state after layer {int(section.get('early_exit_layers', 2))}",
        },
    )
    parameters = sum(value.numel() for value in head.parameters())
    return {
        "status": "complete",
        "variant": variant,
        "optimizer_steps": steps,
        "examples": len(features),
        "trainable_parameters": int(parameters),
        "mean_loss": float(np.mean(losses)),
        "finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": time.time() - started,
        "emotion_base_trainable_parameters": 0,
        "emotion_base_frozen": True,
        "source_balanced_batches": len(source_indexes) > 1,
    }


def train_heads(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    overwrite: bool = False,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    cache = _cache_dir(paths, seed)
    if not (cache / "cache_complete.json").exists():
        raise FileNotFoundError(f"Run cache-seed before training heads: {cache}")
    semantic_features = load_file(
        str(cache / "semantic_train_features.safetensors"), device="cpu"
    )["features"]
    utility_features = load_file(
        str(cache / "utility_train_features.safetensors"), device="cpu"
    )["features"]
    semantic_meta = json.loads(
        (cache / "semantic_train_metadata.json").read_text(encoding="utf-8")
    )
    utility_meta = json.loads(
        (cache / "utility_train_metadata.json").read_text(encoding="utf-8")
    )
    teacher = _read_jsonl(paths["data"] / "teacher_train_annotations.jsonl")
    teacher_by_key = {
        f"{item['dialogue_id']}::{int(item['target_turn_id'])}": item
        for item in teacher
    }
    semantic_labels = []
    semantic_weights = []
    for key in semantic_meta["keys"]:
        annotation = teacher_by_key[key]
        context_need = str(annotation["context_need"])
        semantic_labels.append(teacher_read_label(context_need))
        semantic_weights.append(teacher_sample_weight(context_need))
    labels = list(section.get("labels", LABELS))
    utility_labels, utility_weights, utility_rows = _utility_rows_from_cache(
        section, utility_meta, labels
    )
    _write_json(cache / "utility_training_targets.json", utility_rows)
    output: dict[str, Any] = {}
    audits: dict[str, Any] = {}
    for variant in HEAD_VARIANTS:
        output_dir = _head_dir(paths, seed, variant)
        done = output_dir / "training_complete.json"
        if done.exists() and not overwrite:
            output[variant] = json.loads(done.read_text(encoding="utf-8"))
            continue
        if output_dir.exists() and overwrite:
            backup = output_dir.with_name(
                output_dir.name + ".backup_" + time.strftime("%Y%m%dT%H%M%S")
            )
            shutil.move(str(output_dir), str(backup))
        rows, audit = _assemble_head_rows(
            variant,
            semantic_features,
            semantic_labels,
            semantic_weights,
            utility_features,
            utility_labels,
            utility_weights,
            int(section.get("shuffle_control_seed", 7272)) + seed,
        )
        training_seed = int(seed) + {
            "semantic_gate": 0,
            "utility_gate": 1009,
            "joint_gate": 2018,
            "shuffled_joint_gate": 2018,
        }[variant]
        state = _train_one_head(
            section,
            rows,
            training_seed,
            output_dir,
            variant,
            max_steps_override=max_steps_override,
        )
        state.update(audit)
        state["seed"] = int(seed)
        state["initialization_seed"] = training_seed
        state["version"] = VERSION
        state["test_data_read"] = False
        _write_json(done, state)
        output[variant] = state
        audits[variant] = audit
        del rows
    summary = {
        "status": "complete",
        "seed": int(seed),
        "variants": output,
        "semantic_label_distribution": dict(Counter(semantic_labels)),
        "utility_label_distribution": dict(Counter(utility_labels)),
        "utility_rule_distribution": dict(
            Counter(str(item["rule"]) for item in utility_rows)
        ),
        "emotion_base_frozen": True,
        "validation_cache_loaded": False,
        "test_data_read": False,
    }
    _write_json(paths["reports"] / f"seed{seed}_head_training_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_head(path: Path, device: torch.device) -> EarlyBinaryGateHead:
    config = json.loads((path / "gate_head_config.json").read_text(encoding="utf-8"))
    if config.get("classes") != ["SKIP", "READ"]:
        raise ValueError(f"Unexpected Stage 7.2 classes in {path}")
    head = EarlyBinaryGateHead(
        int(config["hidden_size"]),
        int(config["bottleneck"]),
        float(config.get("dropout", 0.10)),
    ).to(device)
    head.load_state_dict(load_file(str(path / "gate_head.safetensors"), device=str(device)))
    head.eval()
    return head


def _head_probabilities(
    head: EarlyBinaryGateHead,
    features: torch.Tensor,
    batch_size: int = 512,
) -> np.ndarray:
    device = next(head.parameters()).device
    result: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            values = features[start : start + int(batch_size)].to(
                device, dtype=torch.float32, non_blocking=True
            )
            probability = torch.sigmoid(head(values))
            result.append(probability.float().cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float64)


def _mix_scores(
    current_scores: Sequence[Sequence[float]],
    recent_scores: Sequence[Sequence[float]],
    read_mask: Sequence[bool],
) -> list[list[float]]:
    if not (
        len(current_scores) == len(recent_scores) == len(read_mask)
    ):
        raise ValueError("Adaptive score inputs are not aligned")
    return [
        list(map(float, recent if read else current))
        for current, recent, read in zip(current_scores, recent_scores, read_mask)
    ]


def _teacher_diagnostic_arrays(
    metadata: dict[str, Any], teacher_annotations: Sequence[dict[str, Any]]
) -> dict[str, np.ndarray]:
    teacher = {
        f"{item['dialogue_id']}::{int(item['target_turn_id'])}": str(
            item["context_need"]
        )
        for item in teacher_annotations
    }
    indexes = [
        index
        for index, key in enumerate(metadata["keys"])
        if key in teacher and bool(metadata["history_available"][index])
    ]
    context_need = [teacher[metadata["keys"][index]] for index in indexes]
    return {
        "indexes": np.asarray(indexes, dtype=np.int64),
        "required": np.asarray(
            [value == "history_required" for value in context_need], dtype=bool
        ),
        "useful": np.asarray(
            [value in {"history_helpful", "history_required"} for value in context_need],
            dtype=bool,
        ),
        "current": np.asarray(
            [value == "current_sufficient" for value in context_need], dtype=bool
        ),
        "positive": np.asarray(
            [teacher_read_label(value) for value in context_need], dtype=np.int64
        ),
    }


def _threshold_diagnostics(
    probabilities: Sequence[float],
    read_mask: Sequence[bool],
    metadata: dict[str, Any],
    teacher_annotations: Sequence[dict[str, Any]],
    utility_labels: Sequence[int],
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    read = np.asarray(read_mask, dtype=bool)
    teacher = _teacher_diagnostic_arrays(metadata, teacher_annotations)
    indexes = teacher["indexes"]
    teacher_probability = probability[indexes]
    teacher_read = read[indexes]
    positive = teacher["positive"]
    context_ap = (
        float(average_precision_score(positive, teacher_probability))
        if len(set(positive.tolist())) > 1
        else None
    )
    utility = np.asarray(utility_labels, dtype=np.int64)
    utility_ap = (
        float(average_precision_score(utility, probability))
        if len(set(utility.tolist())) > 1
        else None
    )
    return {
        "teacher_records": int(len(indexes)),
        "context_positive_prevalence": float(positive.mean()) if len(positive) else None,
        "context_average_precision": context_ap,
        "context_ap_lift_over_prevalence": (
            float(context_ap - positive.mean()) if context_ap is not None else None
        ),
        "required_recall": (
            float(teacher_read[teacher["required"]].mean())
            if teacher["required"].any()
            else 1.0
        ),
        "useful_recall": (
            float(teacher_read[teacher["useful"]].mean())
            if teacher["useful"].any()
            else 1.0
        ),
        "current_sufficient_skip_rate": (
            float((~teacher_read[teacher["current"]]).mean())
            if teacher["current"].any()
            else 0.0
        ),
        "utility_positive_prevalence": float(utility.mean()),
        "utility_average_precision": utility_ap,
        "utility_ap_lift_over_prevalence": (
            float(utility_ap - utility.mean()) if utility_ap is not None else None
        ),
        "context_use_rate": float(read.mean()),
    }


def select_cost_aware_threshold(
    probabilities: Sequence[float],
    metadata: dict[str, Any],
    teacher_annotations: Sequence[dict[str, Any]],
    utility_labels: Sequence[int],
    labels: Sequence[str],
    section: dict[str, Any],
    early_layers: int,
    total_layers: int,
) -> dict[str, Any]:
    probability = np.asarray(probabilities, dtype=np.float64)
    history = np.asarray(metadata["history_available"], dtype=bool)
    gold = list(metadata["gold"])
    always = _metrics(metadata["recent_scores"], gold, labels, 1.0)
    candidates = sorted(set([0.0, 1.000001] + probability.tolist()))
    teacher = _teacher_diagnostic_arrays(metadata, teacher_annotations)
    rows: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for threshold in candidates:
        read = (probability >= float(threshold)) & history
        mixed = _mix_scores(
            metadata["current_scores"], metadata["recent_scores"], read.tolist()
        )
        metric = _metrics(mixed, gold, labels, 1.0)
        teacher_read = read[teacher["indexes"]]
        required_recall = (
            float(teacher_read[teacher["required"]].mean())
            if teacher["required"].any()
            else 1.0
        )
        useful_recall = (
            float(teacher_read[teacher["useful"]].mean())
            if teacher["useful"].any()
            else 1.0
        )
        current_skip = (
            float((~teacher_read[teacher["current"]]).mean())
            if teacher["current"].any()
            else 0.0
        )
        cost_ratio = layer_equivalent_cost_ratio(
            metadata["current_tokens"],
            metadata["recent_tokens"],
            read.tolist(),
            early_layers,
            total_layers,
        )
        row = {
            "threshold": float(threshold),
            "weighted_f1": float(metric["weighted_f1"]),
            "macro_f1": float(metric["macro_f1"]),
            "context_use_rate": float(read.mean()),
            "history_eligible_read_rate": (
                float(read[history].mean()) if history.any() else 0.0
            ),
            "required_recall": required_recall,
            "useful_recall": useful_recall,
            "current_sufficient_skip_rate": current_skip,
            "layer_equivalent_cost_ratio": float(cost_ratio),
        }
        tolerance = float(
            section.get("threshold_classification_noninferiority_tolerance", 0.01)
        )
        row["classification_noninferior"] = bool(
            row["weighted_f1"] >= always["weighted_f1"] - tolerance
            and row["macro_f1"] >= always["macro_f1"] - tolerance
        )
        row["semantic_constraints_met"] = bool(
            row["required_recall"]
            >= float(section.get("threshold_minimum_required_recall", 0.90))
            and row["useful_recall"]
            >= float(section.get("threshold_minimum_useful_recall", 0.80))
            and row["current_sufficient_skip_rate"]
            >= float(section.get("threshold_minimum_current_skip_rate", 0.20))
        )
        row["cost_constraint_met"] = bool(
            row["layer_equivalent_cost_ratio"]
            <= float(section.get("threshold_maximum_cost_ratio", 0.90))
        )
        row["constraint_met"] = bool(
            row["classification_noninferior"]
            and row["semantic_constraints_met"]
            and row["cost_constraint_met"]
        )
        rows.append(row)
        if row["constraint_met"]:
            eligible.append(row)
    if eligible:
        chosen = min(
            eligible,
            key=lambda item: (
                item["layer_equivalent_cost_ratio"],
                -(item["weighted_f1"] + item["macro_f1"]),
                item["context_use_rate"],
            ),
        )
        selection_status = "all_preregistered_constraints_met"
    else:
        cost_weight = float(section.get("threshold_fallback_cost_weight", 0.10))
        semantic_candidates = [item for item in rows if item["semantic_constraints_met"]]
        pool = semantic_candidates or rows
        chosen = max(
            pool,
            key=lambda item: (
                0.5 * (item["weighted_f1"] + item["macro_f1"])
                - cost_weight * item["layer_equivalent_cost_ratio"],
                -item["layer_equivalent_cost_ratio"],
            ),
        )
        selection_status = "diagnostic_fallback_constraints_failed"
    frontier = []
    for candidate in rows:
        dominated = any(
            other["layer_equivalent_cost_ratio"]
            <= candidate["layer_equivalent_cost_ratio"]
            and other["weighted_f1"] >= candidate["weighted_f1"]
            and other["macro_f1"] >= candidate["macro_f1"]
            and (
                other["layer_equivalent_cost_ratio"]
                < candidate["layer_equivalent_cost_ratio"]
                or other["weighted_f1"] > candidate["weighted_f1"]
                or other["macro_f1"] > candidate["macro_f1"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(candidate)
    frontier.sort(
        key=lambda item: (
            item["layer_equivalent_cost_ratio"],
            -item["weighted_f1"],
            -item["macro_f1"],
        )
    )
    return {
        **chosen,
        "selection_status": selection_status,
        "candidate_count": len(rows),
        "eligible_candidate_count": len(eligible),
        "always_recent3_weighted_f1": float(always["weighted_f1"]),
        "always_recent3_macro_f1": float(always["macro_f1"]),
        "selection_data": "dialogue-disjoint MELD-train threshold calibration",
        "validation_outcomes_used": False,
        "pareto_frontier": frontier,
    }


def _metrics_from_scores(
    calibration_scores: Sequence[Sequence[float]],
    validation_scores: Sequence[Sequence[float]],
    calibration_gold: Sequence[str],
    validation_gold: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    temperature = _fit_temperature(
        calibration_scores, [labels.index(value) for value in calibration_gold]
    )
    return _metrics(validation_scores, validation_gold, labels, temperature)


@dataclass
class ProductionResult:
    scores: list[list[float]]
    read_mask: list[bool]
    read_probabilities: list[float]
    elapsed_seconds: float
    gate_seconds: float
    full_current_seconds: float
    full_recent_seconds: float
    raw_input_tokens: int
    layer_equivalent_tokens: float


class EarlyGateProductionEngine:
    """Real gate-first route with one and only one complete emotion pass."""

    def __init__(
        self,
        section: dict[str, Any],
        adapter: Path,
        head_path: Path,
    ):
        self.engine = AdaptiveLLMEngine(section, adapter, with_gate=False)
        self.extractor = EarlyFeatureExtractor(
            self.engine.model,
            self.engine.tokenizer,
            int(section.get("early_exit_layers", 2)),
            int(section.get("max_sequence_tokens", 768)),
        )
        self.head = _load_head(head_path, torch.device("cuda"))
        self.section = section

    def route_and_score(
        self,
        targets: Sequence[Target],
        threshold: float,
        batch_size: int,
    ) -> ProductionResult:
        current_prompts = self.engine.prompts(targets, "current")
        feature = self.extractor.extract(
            current_prompts, batch_size=batch_size, keep_on_device=True
        )
        torch.cuda.synchronize()
        gate_started = time.perf_counter()
        with torch.inference_mode():
            read_probability = torch.sigmoid(self.head(feature.features.float()))
        torch.cuda.synchronize()
        probability = read_probability.float().cpu().numpy()
        history = np.asarray([bool(target.history) for target in targets], dtype=bool)
        read = (probability >= float(threshold)) & history
        head_seconds = time.perf_counter() - gate_started
        current_indexes = np.flatnonzero(~read).tolist()
        recent_indexes = np.flatnonzero(read).tolist()
        scores: list[list[float] | None] = [None] * len(targets)
        current_seconds = 0.0
        recent_seconds = 0.0
        full_tokens = 0
        if current_indexes:
            current_result = self.engine.forward_targets(
                [targets[index] for index in current_indexes],
                "current",
                need_gate=False,
                batch_size=batch_size,
            )
            current_seconds = current_result.elapsed_seconds
            full_tokens += int(sum(current_result.token_counts))
            for index, row in zip(current_indexes, current_result.scores):
                scores[index] = list(map(float, row))
        if recent_indexes:
            recent_result = self.engine.forward_targets(
                [targets[index] for index in recent_indexes],
                "recent3",
                need_gate=False,
                batch_size=batch_size,
            )
            recent_seconds = recent_result.elapsed_seconds
            full_tokens += int(sum(recent_result.token_counts))
            for index, row in zip(recent_indexes, recent_result.scores):
                scores[index] = list(map(float, row))
        if any(row is None for row in scores):
            raise RuntimeError("Production route failed to score every target")
        gate_tokens = int(sum(feature.token_counts))
        gate_fraction = self.extractor.early_layers / self.extractor.total_layers
        return ProductionResult(
            scores=[row for row in scores if row is not None],
            read_mask=read.tolist(),
            read_probabilities=probability.tolist(),
            elapsed_seconds=(
                feature.elapsed_seconds + head_seconds + current_seconds + recent_seconds
            ),
            gate_seconds=feature.elapsed_seconds + head_seconds,
            full_current_seconds=current_seconds,
            full_recent_seconds=recent_seconds,
            raw_input_tokens=gate_tokens + full_tokens,
            layer_equivalent_tokens=gate_tokens * gate_fraction + full_tokens,
        )


def _benchmark_primary(
    production: EarlyGateProductionEngine,
    targets: Sequence[Target],
    threshold: float,
    section: dict[str, Any],
) -> dict[str, Any]:
    maximum = int(section.get("benchmark_max_targets", 256))
    if 0 < maximum < len(targets):
        indexes = np.linspace(0, len(targets) - 1, maximum).round().astype(int)
        selected = [targets[int(index)] for index in indexes]
        sampling = "fixed evenly-spaced validation targets"
    else:
        selected = list(targets)
        sampling = "all validation targets"
    warmup = min(len(selected), int(section.get("benchmark_warmup_examples", 8)))
    if warmup:
        production.route_and_score(selected[:warmup], threshold, 1)
        production.engine.forward_targets(
            selected[:warmup], "recent3", need_gate=False, batch_size=1
        )
    output: dict[str, Any] = {}
    for batch_size in [int(value) for value in section.get("benchmark_batch_sizes", [1, 4])]:
        adaptive_seconds = []
        recent_seconds = []
        current_seconds = []
        adaptive_raw_tokens = []
        adaptive_equivalent_tokens = []
        read_rates = []
        repeats = int(section.get("benchmark_repeats", 3))
        recent_tokens = 0
        for _ in range(repeats):
            current = production.engine.forward_targets(
                selected, "current", need_gate=False, batch_size=batch_size
            )
            recent = production.engine.forward_targets(
                selected, "recent3", need_gate=False, batch_size=batch_size
            )
            adaptive = production.route_and_score(selected, threshold, batch_size)
            current_seconds.append(current.elapsed_seconds)
            recent_seconds.append(recent.elapsed_seconds)
            adaptive_seconds.append(adaptive.elapsed_seconds)
            adaptive_raw_tokens.append(adaptive.raw_input_tokens)
            adaptive_equivalent_tokens.append(adaptive.layer_equivalent_tokens)
            read_rates.append(float(np.mean(adaptive.read_mask)))
            recent_tokens = int(sum(recent.token_counts))
        current_median = float(np.median(current_seconds))
        recent_median = float(np.median(recent_seconds))
        adaptive_median = float(np.median(adaptive_seconds))
        equivalent = float(np.median(adaptive_equivalent_tokens))
        output[str(batch_size)] = {
            "targets": len(selected),
            "repeats": repeats,
            "median_seconds": {
                "current_only": current_median,
                "always_recent3": recent_median,
                "early_joint_adaptive": adaptive_median,
            },
            "adaptive_speedup_vs_always_recent3": (
                recent_median / adaptive_median if adaptive_median else 0.0
            ),
            "adaptive_context_use_rate": float(np.mean(read_rates)),
            "always_recent3_input_tokens": recent_tokens,
            "adaptive_raw_input_tokens": int(round(float(np.median(adaptive_raw_tokens)))),
            "adaptive_layer_equivalent_tokens": equivalent,
            "layer_equivalent_token_reduction_vs_always_recent3": (
                1.0 - equivalent / recent_tokens if recent_tokens else 0.0
            ),
            "complete_emotion_forwards_per_target": 1,
            "partial_gate_layers": production.extractor.early_layers,
            "total_decoder_layers": production.extractor.total_layers,
            "sampling": sampling,
            "measurement_scope": (
                "tokenization, early Qwen layers, gate head, branch partitioning, "
                "and synchronized CUDA complete-path forward; model loading excluded"
            ),
        }
    return output


def evaluate_seed(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    overwrite: bool = False,
    max_targets: int | None = None,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    output_path = paths["scores"] / f"seed{seed}_validation.json"
    if output_path.exists() and not overwrite:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    cache = _cache_dir(paths, seed)
    if not (cache / "cache_complete.json").exists():
        raise FileNotFoundError(f"Missing Stage 7.2 cache for seed={seed}")
    for variant in HEAD_VARIANTS:
        if not (_head_dir(paths, seed, variant) / "training_complete.json").exists():
            raise FileNotFoundError(
                f"Missing Stage 7.2 head seed={seed} variant={variant}"
            )
    cache_state = json.loads((cache / "cache_complete.json").read_text(encoding="utf-8"))
    calibration_features = load_file(
        str(cache / "threshold_calibration_features.safetensors"), device="cpu"
    )["features"]
    validation_features = load_file(
        str(cache / "validation_features.safetensors"), device="cpu"
    )["features"]
    calibration = json.loads(
        (cache / "threshold_calibration_metadata.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (cache / "validation_metadata.json").read_text(encoding="utf-8")
    )
    labels = list(section.get("labels", LABELS))
    teacher_calibration = _read_jsonl(
        paths["data"] / "teacher_threshold_annotations.jsonl"
    )
    utility_calibration, _, utility_rows = _utility_rows_from_cache(
        section, calibration, labels
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage 7.2 validation/runtime benchmark requires CUDA")
    calibration_probabilities: dict[str, np.ndarray] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    heads: dict[str, EarlyBinaryGateHead] = {}
    for variant in HEAD_VARIANTS:
        head = _load_head(_head_dir(paths, seed, variant), device)
        heads[variant] = head
        calibration_probabilities[variant] = _head_probabilities(
            head, calibration_features
        )
        validation_probabilities[variant] = _head_probabilities(
            head, validation_features
        )
    thresholds: dict[str, dict[str, Any]] = {}
    total_layers = int(cache_state["total_decoder_layers"])
    early_layers = int(cache_state["early_exit_layers"])
    for variant in ("semantic_gate", "utility_gate", "joint_gate"):
        thresholds[variant] = select_cost_aware_threshold(
            calibration_probabilities[variant],
            calibration,
            teacher_calibration,
            utility_calibration,
            labels,
            section,
            early_layers,
            total_layers,
        )
    history_calibration = np.asarray(calibration["history_available"], dtype=bool)
    history_validation = np.asarray(validation["history_available"], dtype=bool)
    masks_calibration: dict[str, np.ndarray] = {}
    masks_validation: dict[str, np.ndarray] = {}
    for variant in ("semantic_gate", "utility_gate", "joint_gate"):
        threshold = float(thresholds[variant]["threshold"])
        masks_calibration[variant] = (
            calibration_probabilities[variant] >= threshold
        ) & history_calibration
        masks_validation[variant] = (
            validation_probabilities[variant] >= threshold
        ) & history_validation
    joint_cal_count = int(masks_calibration["joint_gate"].sum())
    joint_val_count = int(masks_validation["joint_gate"].sum())
    masks_calibration["shuffled_joint_gate"] = _exact_top_k_mask(
        calibration_probabilities["shuffled_joint_gate"],
        history_calibration,
        joint_cal_count,
    )
    masks_validation["shuffled_joint_gate"] = _exact_top_k_mask(
        validation_probabilities["shuffled_joint_gate"],
        history_validation,
        joint_val_count,
    )
    random_seed = int(section.get("matched_random_seed", 7281)) + int(seed)
    masks_calibration["matched_random"] = _exact_random_mask(
        history_calibration, joint_cal_count, random_seed
    )
    masks_validation["matched_random"] = _exact_random_mask(
        history_validation, joint_val_count, random_seed + 1
    )

    current_temperature = _fit_temperature(
        calibration["current_scores"],
        [labels.index(value) for value in calibration["gold"]],
    )
    entropy_choice = _select_entropy_threshold(
        calibration["current_scores"],
        calibration["recent_scores"],
        calibration["gold"],
        labels,
        current_temperature,
    )
    calibration_entropy = np.asarray(
        [
            -float(np.sum(value * np.log(np.clip(value, 1e-12, 1.0))))
            for value in (
                _softmax(row, current_temperature)
                for row in calibration["current_scores"]
            )
        ]
    )
    validation_entropy = np.asarray(
        [
            -float(np.sum(value * np.log(np.clip(value, 1e-12, 1.0))))
            for value in (
                _softmax(row, current_temperature)
                for row in validation["current_scores"]
            )
        ]
    )
    entropy_calibration = (
        calibration_entropy >= float(entropy_choice["threshold"])
    ) & history_calibration
    entropy_validation = (
        validation_entropy >= float(entropy_choice["threshold"])
    ) & history_validation

    def adaptive_metrics(cal_mask: np.ndarray, val_mask: np.ndarray) -> dict[str, Any]:
        cal_scores = _mix_scores(
            calibration["current_scores"],
            calibration["recent_scores"],
            cal_mask.tolist(),
        )
        val_scores = _mix_scores(
            validation["current_scores"],
            validation["recent_scores"],
            val_mask.tolist(),
        )
        result = _metrics_from_scores(
            cal_scores,
            val_scores,
            calibration["gold"],
            validation["gold"],
            labels,
        )
        result["context_use_rate"] = float(val_mask.mean())
        result["history_eligible_read_rate"] = (
            float(val_mask[history_validation].mean())
            if history_validation.any()
            else 0.0
        )
        result["complete_emotion_forwards_per_target"] = 1
        return result

    variants: dict[str, dict[str, Any]] = {}
    variants["base_current"] = _metrics_from_scores(
        calibration["current_scores"],
        validation["current_scores"],
        calibration["gold"],
        validation["gold"],
        labels,
    )
    variants["base_recent3"] = _metrics_from_scores(
        calibration["recent_scores"],
        validation["recent_scores"],
        calibration["gold"],
        validation["gold"],
        labels,
    )
    variants["entropy_adaptive"] = adaptive_metrics(
        entropy_calibration, entropy_validation
    )
    variants["entropy_adaptive"].update(
        {
            "threshold": float(entropy_choice["threshold"]),
            "execution": "complete current pass plus selected complete recent3 pass",
            "complete_emotion_forwards_per_target": "1 or 2",
        }
    )
    variants["semantic_adaptive"] = adaptive_metrics(
        masks_calibration["semantic_gate"], masks_validation["semantic_gate"]
    )
    variants["utility_adaptive"] = adaptive_metrics(
        masks_calibration["utility_gate"], masks_validation["utility_gate"]
    )
    variants["joint_adaptive"] = adaptive_metrics(
        masks_calibration["joint_gate"], masks_validation["joint_gate"]
    )
    variants["matched_shuffled"] = adaptive_metrics(
        masks_calibration["shuffled_joint_gate"],
        masks_validation["shuffled_joint_gate"],
    )
    variants["matched_random"] = adaptive_metrics(
        masks_calibration["matched_random"], masks_validation["matched_random"]
    )
    for report_name, head_name in (
        ("semantic_adaptive", "semantic_gate"),
        ("utility_adaptive", "utility_gate"),
        ("joint_adaptive", "joint_gate"),
    ):
        variants[report_name]["gate_threshold"] = float(
            thresholds[head_name]["threshold"]
        )
        variants[report_name]["threshold_constraints_met"] = bool(
            thresholds[head_name]["constraint_met"]
        )
        variants[report_name]["layer_equivalent_cost_ratio_calibration"] = float(
            thresholds[head_name]["layer_equivalent_cost_ratio"]
        )
    variants["matched_shuffled"]["matched_joint_validation_read_count"] = joint_val_count
    variants["matched_random"]["matched_joint_validation_read_count"] = joint_val_count

    diagnostics: dict[str, Any] = {}
    for variant in HEAD_VARIANTS:
        mask = (
            masks_calibration[variant]
            if variant in masks_calibration
            else masks_calibration["joint_gate"]
        )
        diagnostics[variant] = _threshold_diagnostics(
            calibration_probabilities[variant],
            mask,
            calibration,
            teacher_calibration,
            utility_calibration,
        )
    diagnostics["joint_gate"]["threshold_selection"] = thresholds["joint_gate"]
    diagnostics["semantic_gate"]["threshold_selection"] = thresholds["semantic_gate"]
    diagnostics["utility_gate"]["threshold_selection"] = thresholds["utility_gate"]

    production = EarlyGateProductionEngine(
        section,
        _stage7_1_adapter(root, section, seed),
        _head_dir(paths, seed, "joint_gate"),
    )
    benchmark_targets = _targets(
        read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl"),
        max_history=3,
    )
    if max_targets is not None:
        benchmark_targets = benchmark_targets[: int(max_targets)]
    if len(benchmark_targets) != len(validation["keys"]):
        raise ValueError("Validation cache and processed validation are misaligned")
    benchmark = _benchmark_primary(
        production,
        benchmark_targets,
        float(thresholds["joint_gate"]["threshold"]),
        section,
    )
    head_training = {
        variant: json.loads(
            (_head_dir(paths, seed, variant) / "training_complete.json").read_text(
                encoding="utf-8"
            )
        )
        for variant in HEAD_VARIANTS
    }
    _write_json(
        paths["reports"] / f"seed{seed}_threshold_pareto_frontiers.json",
        {
            variant: {
                "selected": {
                    key: value
                    for key, value in thresholds[variant].items()
                    if key != "pareto_frontier"
                },
                "pareto_frontier": thresholds[variant]["pareto_frontier"],
            }
            for variant in ("semantic_gate", "utility_gate", "joint_gate")
        },
    )
    result = {
        "status": "evaluated",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "split": "validation",
        "targets": len(validation["keys"]),
        "dialogues": len({key.rsplit("::", 1)[0] for key in validation["keys"]}),
        "labels": labels,
        "variants": variants,
        "thresholds_frozen_on_meld_train": thresholds,
        "gate_diagnostics_on_meld_train_threshold_split": diagnostics,
        "counterfactual_utility_calibration": {
            "records": len(utility_rows),
            "read_prevalence": float(np.mean(utility_calibration)),
            "rule_distribution": dict(
                Counter(str(item["rule"]) for item in utility_rows)
            ),
        },
        "runtime_benchmark": benchmark,
        "architecture_audit": {
            "gate_reads_current_only": True,
            "history_loaded_after_route_only": True,
            "early_exit_layers": early_layers,
            "total_decoder_layers": total_layers,
            "emotion_base_frozen": True,
            "complete_emotion_forwards_per_primary_target": 1,
            "teacher_or_gold_available_at_validation_route": False,
        },
        "head_training_audit": head_training,
        "runtime_environment": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": str(section.get("dtype", "bfloat16")),
            "emotion_scoring": "unique first-token verbalizer",
        },
        "test_locked": True,
    }
    _write_json(output_path, result)
    print(f"Stage 7.2 seed {seed} validation written to {output_path}")
    del production, heads, calibration_features, validation_features
    gc.collect()
    torch.cuda.empty_cache()
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
            raise FileNotFoundError(f"Missing Stage 7.2 seed report: {path}")
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

    diagnostic_metrics = (
        "context_positive_prevalence",
        "context_average_precision",
        "context_ap_lift_over_prevalence",
        "required_recall",
        "useful_recall",
        "current_sufficient_skip_rate",
        "utility_positive_prevalence",
        "utility_average_precision",
        "utility_ap_lift_over_prevalence",
        "context_use_rate",
    )
    diagnostics: dict[str, Any] = {}
    for variant in HEAD_VARIANTS:
        diagnostics[variant] = {}
        for metric in diagnostic_metrics:
            values = [
                run["gate_diagnostics_on_meld_train_threshold_split"][variant].get(
                    metric
                )
                for run in runs
            ]
            if all(value is not None for value in values):
                diagnostics[variant][metric] = _aggregate(
                    [float(value) for value in values]
                )

    runtime: dict[str, Any] = {}
    batch_sizes = sorted(
        {batch for run in runs for batch in run["runtime_benchmark"]}, key=int
    )
    for batch in batch_sizes:
        runtime[batch] = {
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
            "layer_equivalent_token_reduction_vs_always_recent3": _aggregate(
                [
                    float(
                        run["runtime_benchmark"][batch][
                            "layer_equivalent_token_reduction_vs_always_recent3"
                        ]
                    )
                    for run in runs
                ]
            ),
            "adaptive_context_use_rate": _aggregate(
                [
                    float(
                        run["runtime_benchmark"][batch][
                            "adaptive_context_use_rate"
                        ]
                    )
                    for run in runs
                ]
            ),
            "median_seconds_by_seed": {
                name: [
                    float(run["runtime_benchmark"][batch]["median_seconds"][name])
                    for run in runs
                ]
                for name in (
                    "current_only",
                    "always_recent3",
                    "early_joint_adaptive",
                )
            },
        }

    validation_dialogues = read_jsonl(
        _resolve(root, section["processed_dir"]) / "validation.jsonl"
    )
    targets = _targets(validation_dialogues, max_history=3)
    gold = [target.gold for target in targets]
    dialogue_ids = [target.dialogue_id for target in targets]
    averaged_rows: dict[str, list[dict[str, Any]]] = {}
    for variant in REPORT_VARIANTS:
        probabilities = np.mean(
            np.asarray(
                [run["variants"][variant]["probabilities"] for run in runs],
                dtype=np.float64,
            ),
            axis=0,
        )
        averaged_rows[variant] = [
            {"probabilities": row.tolist()} for row in probabilities
        ]
    controls = {
        "current_only": "base_current",
        "always_recent3": "base_recent3",
        "entropy_router": "entropy_adaptive",
        "semantic_gate": "semantic_adaptive",
        "utility_gate": "utility_adaptive",
        "matched_shuffled": "matched_shuffled",
        "matched_random": "matched_random",
    }
    bootstrap = {
        name: _bootstrap_advantage(
            averaged_rows["joint_adaptive"],
            averaged_rows[control],
            gold,
            dialogue_ids,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 7290)) + offset,
        )
        for offset, (name, control) in enumerate(controls.items())
    }
    _write_json(paths["bootstrap"], bootstrap)

    training_records = []
    for seed in seeds:
        for variant in HEAD_VARIANTS:
            state = json.loads(
                (
                    _head_dir(paths, seed, variant) / "training_complete.json"
                ).read_text(encoding="utf-8")
            )
            training_records.append(
                {
                    "seed": seed,
                    "variant": variant,
                    "optimizer_steps": int(state["optimizer_steps"]),
                    "trainable_parameters": int(state["trainable_parameters"]),
                    "finite": bool(state["finite"]),
                    "emotion_base_frozen": bool(state["emotion_base_frozen"]),
                    "initialization_seed": int(state["initialization_seed"]),
                }
            )
    matched = [
        item
        for item in training_records
        if item["variant"] in {"joint_gate", "shuffled_joint_gate"}
    ]
    capacity_audit = {
        "joint_vs_shuffled_exact_parameter_match": len(
            {item["trainable_parameters"] for item in matched}
        )
        == 1,
        "joint_vs_shuffled_exact_step_match": len(
            {item["optimizer_steps"] for item in matched}
        )
        == 1,
        "joint_vs_shuffled_exact_initialization_match": all(
            next(
                item["initialization_seed"]
                for item in matched
                if item["seed"] == seed and item["variant"] == "joint_gate"
            )
            == next(
                item["initialization_seed"]
                for item in matched
                if item["seed"] == seed
                and item["variant"] == "shuffled_joint_gate"
            )
            for seed in seeds
        ),
        "all_training_finite": all(item["finite"] for item in training_records),
        "all_emotion_bases_frozen": all(
            item["emotion_base_frozen"] for item in training_records
        ),
        "records": training_records,
    }
    architecture_audit = {
        "all_seeds_current_only_before_route": all(
            run["architecture_audit"]["gate_reads_current_only"]
            and run["architecture_audit"]["history_loaded_after_route_only"]
            for run in runs
        ),
        "one_complete_emotion_forward_all_seeds": all(
            int(run["architecture_audit"]["complete_emotion_forwards_per_primary_target"])
            == 1
            for run in runs
        ),
        "early_layer_limit_all_seeds": all(
            int(run["architecture_audit"]["early_exit_layers"])
            <= int(section.get("maximum_allowed_early_layers", 2))
            for run in runs
        ),
        "emotion_base_frozen_all_seeds": all(
            run["architecture_audit"]["emotion_base_frozen"] for run in runs
        ),
    }

    primary = aggregate["joint_adaptive"]
    strongest_weighted = max(
        aggregate[name]["weighted_f1"]["mean"]
        for name in ("base_recent3", "entropy_adaptive")
    )
    strongest_macro = max(
        aggregate[name]["macro_f1"]["mean"]
        for name in ("base_recent3", "entropy_adaptive")
    )
    tolerance = float(section.get("validation_noninferiority_tolerance", 0.01))
    joint_diag = diagnostics["joint_gate"]
    shuffled_diag = diagnostics["shuffled_joint_gate"]
    primary_batch = str(section.get("primary_latency_batch_size", 1))
    matched_shuffled = bootstrap["matched_shuffled"]
    matched_random = bootstrap["matched_random"]
    checks = {
        "joint_threshold_constraints_all_seeds": all(
            bool(
                run["thresholds_frozen_on_meld_train"]["joint_gate"][
                    "constraint_met"
                ]
            )
            for run in runs
        ),
        "joint_context_ap_absolute": joint_diag["context_average_precision"]["mean"]
        >= float(section.get("minimum_context_ap", 0.60)),
        "joint_context_ap_lift": joint_diag["context_ap_lift_over_prevalence"]["mean"]
        >= float(section.get("minimum_context_ap_lift", 0.05)),
        "joint_context_ap_beats_shuffled": (
            joint_diag["context_average_precision"]["mean"]
            - shuffled_diag["context_average_precision"]["mean"]
        )
        >= float(section.get("minimum_context_ap_advantage_over_shuffled", 0.02)),
        "joint_utility_ap_lift": joint_diag["utility_ap_lift_over_prevalence"]["mean"]
        >= float(section.get("minimum_utility_ap_lift", 0.05)),
        "joint_required_recall": joint_diag["required_recall"]["mean"]
        >= float(section.get("minimum_required_recall", 0.90)),
        "joint_useful_recall": joint_diag["useful_recall"]["mean"]
        >= float(section.get("minimum_useful_recall", 0.80)),
        "joint_skips_sufficient_current": joint_diag[
            "current_sufficient_skip_rate"
        ]["mean"]
        >= float(section.get("minimum_current_skip_rate", 0.20)),
        "joint_weighted_f1_noninferior": primary["weighted_f1"]["mean"]
        >= strongest_weighted - tolerance,
        "joint_macro_f1_noninferior": primary["macro_f1"]["mean"]
        >= strongest_macro - tolerance,
        "joint_beats_matched_shuffled_weighted": matched_shuffled["weighted_f1"][
            "probability_positive"
        ]
        >= float(section.get("minimum_bootstrap_probability_positive", 0.95)),
        "joint_beats_matched_shuffled_macro": matched_shuffled["macro_f1"][
            "probability_positive"
        ]
        >= float(section.get("minimum_bootstrap_probability_positive", 0.95)),
        "joint_beats_matched_random_weighted": matched_random["weighted_f1"][
            "probability_positive"
        ]
        >= float(section.get("minimum_bootstrap_probability_positive", 0.95)),
        "joint_beats_matched_random_macro": matched_random["macro_f1"][
            "probability_positive"
        ]
        >= float(section.get("minimum_bootstrap_probability_positive", 0.95)),
        "adaptive_latency_speedup": runtime[primary_batch][
            "adaptive_speedup_vs_always_recent3"
        ]["mean"]
        >= float(section.get("minimum_latency_speedup", 1.10)),
        "layer_equivalent_token_reduction": runtime[primary_batch][
            "layer_equivalent_token_reduction_vs_always_recent3"
        ]["mean"]
        >= float(section.get("minimum_layer_equivalent_token_reduction", 0.10)),
        "architecture_audit": all(architecture_audit.values()),
        "capacity_and_training_audit": all(
            value for key, value in capacity_audit.items() if key != "records"
        ),
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "eligible_to_freeze_stage7_2_and_start_cost_aware_stage7_3"
            if eligible
            else "stop_and_diagnose_early_utility_gate"
        ),
        "eligible": eligible,
        "checks": checks,
        "preregistered_before_stage7_2_validation": True,
        "selection_data": "dialogue-disjoint MELD-train only",
        "next_action": (
            "Freeze this gate as the SFT warm start for Stage 7.3 contextual-bandit training."
            if eligible
            else "Keep MELD test locked; diagnose failed semantic, utility, accuracy, or runtime checks."
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
        "gate_diagnostics_on_meld_train_threshold_split": diagnostics,
        "runtime_benchmark": runtime,
        "paired_dialogue_bootstrap": bootstrap,
        "architecture_audit": architecture_audit,
        "capacity_audit": capacity_audit,
        "decision": decision,
        "teacher_status": (
            "GPT-5.6 Sol generated and self-audited operational labels; not human gold"
        ),
        "test_locked": True,
    }
    _write_json(paths["summary_json"], result)
    _write_json(paths["decision"], decision)

    lines = [
        "# Stage 7.2 — Early Label + Utility Context Gate",
        "",
        "> The first two Qwen layers see only the current utterance. The gate then selects exactly one complete emotion path. MELD test remains locked.",
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
    lines.extend(["", "## Gate diagnostics (MELD-train threshold split)", ""])
    for variant in HEAD_VARIANTS:
        value = diagnostics[variant]
        lines.append(
            f"- {variant}: context AP={value['context_average_precision']['mean']:.4f}, "
            f"utility AP={value['utility_average_precision']['mean']:.4f}, "
            f"required recall={value['required_recall']['mean']:.4f}, "
            f"useful recall={value['useful_recall']['mean']:.4f}, "
            f"current skip={value['current_sufficient_skip_rate']['mean']:.4f}"
        )
    lines.extend(["", "## Runtime", ""])
    for batch, value in runtime.items():
        lines.append(
            f"- batch={batch}: speedup={value['adaptive_speedup_vs_always_recent3']['mean']:.3f}x, "
            f"layer-equivalent token reduction={value['layer_equivalent_token_reduction_vs_always_recent3']['mean']:.2%}, "
            f"context use={value['adaptive_context_use_rate']['mean']:.2%}"
        )
    lines.extend(["", "## Frozen decision", "", f"Status: **{decision['status']}**"])
    for name, passed in checks.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "Teacher labels are GPT-5.6 Sol generated and self-audited operational labels, not human gold labels.",
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
        "seeds": {
            str(seed): {
                "cache": (_cache_dir(paths, seed) / "cache_complete.json").exists(),
                "heads": {
                    variant: (
                        _head_dir(paths, seed, variant) / "training_complete.json"
                    ).exists()
                    for variant in HEAD_VARIANTS
                },
                "evaluation": (
                    paths["scores"] / f"seed{seed}_validation.json"
                ).exists(),
            }
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
        "MELD test is hard-locked in Stage 7.2. Stage 7.2 success only permits "
        "a separately preregistered Stage 7.3 contextual-bandit experiment; it "
        "does not authorize test evaluation."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("prepare")
    cache = subparsers.add_parser("cache-seed")
    cache.add_argument("--seed", type=int, required=True)
    cache.add_argument("--overwrite", action="store_true")
    train = subparsers.add_parser("train-heads")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--overwrite", action="store_true")
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
    elif args.command == "cache-seed":
        cache_seed(
            args.config,
            args.seed,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "train-heads":
        train_heads(
            args.config,
            args.seed,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "evaluate":
        evaluate_seed(
            args.config,
            args.seed,
            overwrite=bool(args.overwrite),
        )
    elif args.command == "report":
        report(args.config)
    elif args.command == "status":
        status(args.config)
    elif args.command == "smoke":
        smoke_name = f"smoke_seed{args.seed}"
        prepare_training_data(args.config, smoke_name)
        cache_seed(
            args.config,
            args.seed,
            smoke_name,
            max_targets=16,
            overwrite=True,
        )
        train_heads(
            args.config,
            args.seed,
            smoke_name,
            overwrite=True,
            max_steps_override=3,
        )
        evaluate_seed(
            args.config,
            args.seed,
            smoke_name,
            overwrite=True,
            max_targets=16,
        )
    elif args.command == "final-test":
        final_test_locked()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
