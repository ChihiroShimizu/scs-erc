from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from pfa_erc.data import Dialogue, read_jsonl
from pfa_erc.training.llm_context_utility_stage1 import (
    TargetSpec,
    _metric_bundle,
    _read_jsonl_dicts,
    _write_json,
    make_target_specs,
)
from pfa_erc.training.llm_selective_context_stage2 import (
    TARGET_FEATURE_NAMES,
    _calibrated_payload,
    _choose_uncertainty_threshold,
    _fit_shared_temperature,
    _gold_nll,
    _variant_record,
    target_features,
)


STAGE2R_VERSION = "1.2.0"

DISCOURSE_MARKERS = {
    "actually",
    "although",
    "anyway",
    "because",
    "but",
    "however",
    "instead",
    "so",
    "then",
    "therefore",
    "though",
    "well",
    "yet",
}
COREFERENCE_WORDS = {
    "he",
    "her",
    "hers",
    "him",
    "his",
    "it",
    "its",
    "she",
    "that",
    "them",
    "their",
    "theirs",
    "these",
    "they",
    "this",
    "those",
}
NEGATION_WORDS = {
    "ain't",
    "cannot",
    "can't",
    "didn't",
    "doesn't",
    "don't",
    "never",
    "no",
    "not",
    "won't",
}
EMOTION_WORDS = {
    "afraid",
    "angry",
    "annoyed",
    "disgusted",
    "excited",
    "glad",
    "gross",
    "happy",
    "hate",
    "hurt",
    "mad",
    "sad",
    "scared",
    "sorry",
    "surprised",
    "upset",
    "worried",
    "wow",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _root_and_config(config_path: Path) -> tuple[Path, dict[str, Any]]:
    path = config_path.expanduser().resolve()
    return path.parent.parent, _read_yaml(path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("llm_context_router_stage2r")
    if not isinstance(section, dict):
        raise KeyError("Missing llm_context_router_stage2r configuration section")
    return section


def _labels(config: dict[str, Any]) -> list[str]:
    labels = list(_section(config).get("labels", []))
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("llm_context_router_stage2r.labels must contain unique labels")
    return labels


def _paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    section = _section(config)
    base = _resolve(root, section.get("artifact_dir", "artifacts/llm_context_router_stage2r/meld"))
    return {
        "base": base,
        "train_records": _resolve(root, section["stage2_train_records"]),
        "validation_records": _resolve(root, section["stage1_validation_records"]),
        "models": base / "models" / "stage2r_models.json",
        "report_json": base / "reports" / "validation_summary.json",
        "report_md": base / "reports" / "validation_summary.md",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
        "decision": base / "reports" / "frozen_validation_decision.json",
    }


def _load_split(root: Path, config: dict[str, Any], split: str) -> list[Dialogue]:
    if split == "test":
        raise ValueError("Stage 2R is train/validation-only; MELD test remains locked")
    path = _resolve(root, config["data"]["processed_dir"]) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    return read_jsonl(path)


def _target_maps(
    dialogues: Sequence[Dialogue], max_history_turns: int
) -> tuple[dict[str, TargetSpec], dict[tuple[str, int], TargetSpec]]:
    by_key: dict[str, TargetSpec] = {}
    by_turn: dict[tuple[str, int], TargetSpec] = {}
    for dialogue in dialogues:
        for target in make_target_specs(dialogue, max_history_turns=max_history_turns):
            by_key[target.key] = target
            by_turn[(target.dialogue_id, target.turn_index)] = target
    return by_key, by_turn


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def _jaccard(left: str, right: str) -> float:
    a, b = set(_tokens(left)), set(_tokens(right))
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std(ddof=0)) if len(array) else None,
        "count": int(len(array)),
    }


def _current_probabilities(
    record_by_turn: dict[tuple[str, int], dict[str, Any]],
    dialogue_id: str,
    turn_index: int,
    labels: Sequence[str],
    temperature: float,
) -> tuple[list[float], float]:
    record = record_by_turn.get((dialogue_id, turn_index))
    if record is None:
        return [1.0 / len(labels)] * len(labels), 0.0
    payload = _calibrated_payload(record["current"]["scores"], labels, temperature)
    return list(payload["probabilities"]), 1.0


def _previous_same_speaker_index(target: TargetSpec) -> int | None:
    for history_index in range(len(target.history) - 1, -1, -1):
        if target.history[history_index].speaker == target.speaker:
            return target.turn_index - (len(target.history) - history_index)
    return None


def _rate(tokens: Sequence[str], vocabulary: set[str]) -> float:
    return sum(token in vocabulary for token in tokens) / max(len(tokens), 1)


CUE_EXTRA_FEATURE_NAMES = [
    *[
        f"last_turn_p_{label}"
        for label in ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
    ],
    *[
        f"same_speaker_p_{label}"
        for label in ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
    ],
    "last_turn_available",
    "same_speaker_state_available",
    "current_last_probability_l1",
    "current_last_probability_dot",
    "current_last_same_argmax",
    "current_same_probability_l1",
    "current_same_probability_dot",
    "current_same_same_argmax",
    "current_last_lexical_jaccard",
    "current_history_mean_jaccard",
    "current_history_max_jaccard",
    "current_same_speaker_max_jaccard",
    "history_unique_speaker_fraction",
    "immediate_speaker_change",
    "speaker_reentry",
    "discourse_marker_density",
    "coreference_density",
    "negation_density",
    "explicit_emotion_word_density",
    "previous_turn_is_question",
    "topic_discontinuity_proxy",
]


def cue_aware_features(
    base_features: Sequence[float],
    target: TargetSpec,
    record_by_turn: dict[tuple[str, int], dict[str, Any]],
    labels: Sequence[str],
    temperature: float,
) -> list[float]:
    """Return causal input-only cue proxies; no gold emotion is read here."""
    current = np.asarray(base_features[: len(labels)], dtype=np.float64)
    last_index = target.turn_index - 1
    last, last_available = _current_probabilities(
        record_by_turn, target.dialogue_id, last_index, labels, temperature
    )
    same_index = _previous_same_speaker_index(target)
    same, same_available = _current_probabilities(
        record_by_turn,
        target.dialogue_id,
        same_index if same_index is not None else -1,
        labels,
        temperature,
    )
    last_array = np.asarray(last, dtype=np.float64)
    same_array = np.asarray(same, dtype=np.float64)
    history_jaccards = [_jaccard(target.text, turn.text) for turn in target.history]
    same_jaccards = [
        _jaccard(target.text, turn.text)
        for turn in target.history
        if turn.speaker == target.speaker
    ]
    speakers = {turn.speaker for turn in target.history}
    last_turn = target.history[-1] if target.history else None
    tokens = _tokens(target.text)
    history_max = max(history_jaccards, default=0.0)
    return [
        *map(float, last_array),
        *map(float, same_array),
        float(last_available),
        float(same_available),
        float(np.abs(current - last_array).sum()),
        float(np.dot(current, last_array)),
        float(np.argmax(current) == np.argmax(last_array)),
        float(np.abs(current - same_array).sum()),
        float(np.dot(current, same_array)),
        float(np.argmax(current) == np.argmax(same_array)),
        _jaccard(target.text, last_turn.text) if last_turn else 0.0,
        float(np.mean(history_jaccards)) if history_jaccards else 0.0,
        history_max,
        max(same_jaccards, default=0.0),
        len(speakers) / max(len(target.history), 1),
        float(last_turn is not None and last_turn.speaker != target.speaker),
        float(any(turn.speaker == target.speaker for turn in target.history) and last_turn is not None and last_turn.speaker != target.speaker),
        _rate(tokens, DISCOURSE_MARKERS),
        _rate(tokens, COREFERENCE_WORDS),
        _rate(tokens, NEGATION_WORDS),
        _rate(tokens, EMOTION_WORDS),
        float(last_turn is not None and "?" in last_turn.text),
        1.0 - history_max if target.history else 0.0,
    ]


def _build_examples(
    records: Sequence[dict[str, Any]],
    dialogues: Sequence[Dialogue],
    labels: Sequence[str],
    temperature: float,
    max_history_turns: int,
) -> list[dict[str, Any]]:
    target_by_key, _ = _target_maps(dialogues, max_history_turns)
    record_by_turn = {
        (record["dialogue_id"], int(record["turn_index"])): record for record in records
    }
    output: list[dict[str, Any]] = []
    for record in records:
        target = target_by_key.get(record["key"])
        if target is None:
            continue
        current = _calibrated_payload(record["current"]["scores"], labels, temperature)
        full = _calibrated_payload(record["full"]["scores"], labels, temperature)
        base = target_features(record, target, labels, temperature, max_history_turns)
        cue = cue_aware_features(base, target, record_by_turn, labels, temperature)
        current_nll = _gold_nll(current, record["gold"], labels)
        full_nll = _gold_nll(full, record["gold"], labels)
        output.append(
            {
                "key": record["key"],
                "dialogue_id": record["dialogue_id"],
                "turn_index": int(record["turn_index"]),
                "gold": record["gold"],
                "history_count": int(record.get("history_count", len(target.history))),
                "current": current,
                "full": full,
                "utility": float(current_nll - full_nll),
                "base_features": [float(value) for value in base],
                "cue_features": [float(value) for value in (*base, *cue)],
            }
        )
    if not output:
        raise ValueError("No scored targets could be aligned to the processed dialogue split")
    return output


def _serialize_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    model = Ridge(alpha=alpha).fit(scaler.transform(x), y)
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": np.asarray(model.coef_, dtype=np.float64).tolist(),
        "intercept": float(model.intercept_),
        "alpha": float(alpha),
    }


def _predict_linear(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    normalized = (x - mean) / np.where(scale == 0.0, 1.0, scale)
    return normalized @ np.asarray(model["coef"], dtype=np.float64) + float(model["intercept"])


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_direct_gate(
    x: np.ndarray,
    current_gold: np.ndarray,
    full_gold: np.ndarray,
    context_cost: float,
    seed: int,
    epochs: int,
    learning_rate: float,
    l2: float,
    gradient_clip: float,
) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 0.01, size=normalized.shape[1])
    bias = -0.5
    first_w = np.zeros_like(weights)
    second_w = np.zeros_like(weights)
    first_b = 0.0
    second_b = 0.0
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    for step in range(1, epochs + 1):
        gate = _sigmoid(normalized @ weights + bias)
        mixture_gold = np.clip(
            (1.0 - gate) * current_gold + gate * full_gold, 1e-12, 1.0
        )
        dloss_dgate = -(full_gold - current_gold) / mixture_gold + context_cost
        dloss_dlogit = dloss_dgate * gate * (1.0 - gate)
        grad_w = normalized.T @ dloss_dlogit / len(x) + l2 * weights
        grad_b = float(dloss_dlogit.mean())
        norm = float(np.sqrt(np.dot(grad_w, grad_w) + grad_b * grad_b))
        if norm > gradient_clip:
            scale = gradient_clip / max(norm, 1e-12)
            grad_w *= scale
            grad_b *= scale
        first_w = beta1 * first_w + (1.0 - beta1) * grad_w
        second_w = beta2 * second_w + (1.0 - beta2) * (grad_w**2)
        first_b = beta1 * first_b + (1.0 - beta1) * grad_b
        second_b = beta2 * second_b + (1.0 - beta2) * (grad_b**2)
        correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
        weights -= learning_rate * (first_w / correction1) / (
            np.sqrt(second_w / correction2) + epsilon
        )
        bias -= learning_rate * (first_b / correction1) / (
            math.sqrt(second_b / correction2) + epsilon
        )
    gate = _sigmoid(normalized @ weights + bias)
    mixture_gold = np.clip((1.0 - gate) * current_gold + gate * full_gold, 1e-12, 1.0)
    return {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "context_cost": float(context_cost),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "l2": float(l2),
        "train_objective": float((-np.log(mixture_gold) + context_cost * gate).mean()),
        "train_gate_mean": float(gate.mean()),
    }


def _predict_gate(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    normalized = (x - mean) / np.where(scale == 0.0, 1.0, scale)
    return _sigmoid(
        normalized @ np.asarray(model["weights"], dtype=np.float64) + float(model["bias"])
    )


def _fold_ids(dialogues: Sequence[str], folds: int, seed: int) -> np.ndarray:
    unique = sorted(set(dialogues))
    random.Random(seed).shuffle(unique)
    assignment = {dialogue: index % folds for index, dialogue in enumerate(unique)}
    return np.asarray([assignment[dialogue] for dialogue in dialogues], dtype=np.int64)


def _records_for_branch(
    examples: Sequence[dict[str, Any]],
    labels: Sequence[str],
    scores: Sequence[float],
    threshold: float,
    name: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for example, score in zip(examples, scores):
        use_context = example["history_count"] > 0 and float(score) >= threshold
        branch = example["full"] if use_context else example["current"]
        output.append(
            _variant_record(example, {name: branch}, labels)
            | {
                "history_count": example["history_count"],
                "router_score": float(score),
                "uses_context": bool(use_context),
                "utility": float(example["utility"]),
            }
        )
    return output


def _threshold_candidates(scores: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    values = scores[eligible]
    if not len(values):
        raise ValueError("No history-eligible targets available for threshold selection")
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 101))
    return np.unique(np.concatenate(([values.min() - 1e-9], quantiles, [values.max() + 1e-9])))


def _select_threshold(
    examples: Sequence[dict[str, Any]],
    labels: Sequence[str],
    scores: np.ndarray,
    context_penalty: float,
) -> dict[str, Any]:
    eligible = np.asarray([example["history_count"] > 0 for example in examples], dtype=bool)
    best: tuple[float, float, float, dict[str, Any]] | None = None
    for threshold in _threshold_candidates(scores, eligible):
        records = _records_for_branch(examples, labels, scores, float(threshold), "routed")
        metrics = _metric_bundle(records, "routed", labels)
        context_rate = float(np.mean([record["uses_context"] for record in records]))
        objective = 0.5 * (metrics["weighted_f1"] + metrics["macro_f1"]) - context_penalty * context_rate
        candidate = (objective, -context_rate, -float(threshold), metrics)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    assert best is not None
    return {
        "threshold": -best[2],
        "objective": best[0],
        "context_use_rate": -best[1],
        "weighted_f1": best[3]["weighted_f1"],
        "macro_f1": best[3]["macro_f1"],
        "fit_split": "train_oof",
    }


def _eligible_arrays(
    examples: Sequence[dict[str, Any]], feature_key: str, labels: Sequence[str]
) -> dict[str, Any]:
    indices = np.asarray(
        [index for index, example in enumerate(examples) if example["history_count"] > 0],
        dtype=np.int64,
    )
    x = np.asarray([examples[index][feature_key] for index in indices], dtype=np.float64)
    current_gold = np.asarray(
        [examples[index]["current"]["probabilities"][labels.index(examples[index]["gold"])] for index in indices],
        dtype=np.float64,
    )
    full_gold = np.asarray(
        [examples[index]["full"]["probabilities"][labels.index(examples[index]["gold"])] for index in indices],
        dtype=np.float64,
    )
    return {
        "indices": indices,
        "x": x,
        "current_gold": current_gold,
        "full_gold": full_gold,
        "utility": np.asarray([examples[index]["utility"] for index in indices]),
        "dialogues": [examples[index]["dialogue_id"] for index in indices],
    }


def _oof_ridge(
    arrays: dict[str, Any], folds: int, seed: int, alpha: float
) -> np.ndarray:
    fold = _fold_ids(arrays["dialogues"], folds, seed)
    predictions = np.zeros(len(arrays["indices"]), dtype=np.float64)
    for fold_id in range(folds):
        train = fold != fold_id
        held_out = fold == fold_id
        model = _serialize_ridge(arrays["x"][train], arrays["utility"][train], alpha)
        predictions[held_out] = _predict_linear(model, arrays["x"][held_out])
    return predictions


def _oof_gate(
    arrays: dict[str, Any],
    folds: int,
    seed: int,
    context_cost: float,
    trainer: dict[str, Any],
) -> np.ndarray:
    fold = _fold_ids(arrays["dialogues"], folds, seed)
    predictions = np.zeros(len(arrays["indices"]), dtype=np.float64)
    for fold_id in range(folds):
        train = fold != fold_id
        held_out = fold == fold_id
        model = _fit_direct_gate(
            arrays["x"][train],
            arrays["current_gold"][train],
            arrays["full_gold"][train],
            context_cost=context_cost,
            seed=seed * 101 + fold_id,
            **trainer,
        )
        predictions[held_out] = _predict_gate(model, arrays["x"][held_out])
    return predictions


def _expand_scores(count: int, indices: np.ndarray, eligible_scores: np.ndarray) -> np.ndarray:
    # A finite sentinel keeps JSON prediction artifacts standards-compliant.
    # Non-history turns are still forced to current-only by _records_for_branch.
    scores = np.full(count, -1e9, dtype=np.float64)
    scores[indices] = eligible_scores
    return scores


def _fit_seed(
    examples: Sequence[dict[str, Any]], labels: Sequence[str], section: dict[str, Any], seed: int
) -> dict[str, Any]:
    folds = int(section.get("oof_dialogue_folds", 5))
    context_penalty = float(section.get("selection_context_penalty", 0.005))
    alpha = float(section.get("ridge_alpha", 1.0))
    trainer = {
        "epochs": int(section.get("gate_epochs", 500)),
        "learning_rate": float(section.get("gate_learning_rate", 0.03)),
        "l2": float(section.get("gate_l2", 1e-3)),
        "gradient_clip": float(section.get("gate_gradient_clip", 5.0)),
    }
    base = _eligible_arrays(examples, "base_features", labels)
    cue = _eligible_arrays(examples, "cue_features", labels)

    ridge_oof = _oof_ridge(cue, folds, seed, alpha)
    ridge_selection = _select_threshold(
        examples,
        labels,
        _expand_scores(len(examples), cue["indices"], ridge_oof),
        context_penalty,
    )

    gate_results: dict[str, Any] = {}
    cost_grid = [float(value) for value in section.get("context_cost_grid", [0.0, 0.005, 0.01])]
    for name, arrays in (("base_gate", base), ("cue_gate", cue)):
        candidates: list[dict[str, Any]] = []
        for cost in cost_grid:
            oof = _oof_gate(arrays, folds, seed, cost, trainer)
            selection = _select_threshold(
                examples,
                labels,
                _expand_scores(len(examples), arrays["indices"], oof),
                context_penalty,
            )
            candidates.append({"cost": cost, "selection": selection, "oof_scores": oof})
        selected = max(
            candidates,
            key=lambda item: (
                item["selection"]["objective"],
                -item["selection"]["context_use_rate"],
                -item["cost"],
            ),
        )
        final_model = _fit_direct_gate(
            arrays["x"],
            arrays["current_gold"],
            arrays["full_gold"],
            context_cost=float(selected["cost"]),
            seed=seed,
            **trainer,
        )
        gate_results[name] = {
            "model": final_model,
            "selection": selected["selection"],
            "selected_context_cost": float(selected["cost"]),
            "candidate_summary": [
                {"context_cost": item["cost"], **item["selection"]} for item in candidates
            ],
        }

    ridge_model = _serialize_ridge(cue["x"], cue["utility"], alpha)
    return {
        "seed": seed,
        "utility_regression": {"model": ridge_model, "selection": ridge_selection},
        **gate_results,
    }


def check(config_path: Path) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    paths = _paths(root, config)
    missing = [str(paths[name]) for name in ("train_records", "validation_records") if not paths[name].exists()]
    train = _read_jsonl_dicts(paths["train_records"])
    validation = _read_jsonl_dicts(paths["validation_records"])
    result = {
        "status": "ready" if not missing and len(train) >= int(section.get("minimum_train_targets", 1000)) else "blocked",
        "experiment": "LLM_context_router_stage2R",
        "version": STAGE2R_VERSION,
        "train_records": len(train),
        "validation_records": len(validation),
        "required_train_records": int(section.get("minimum_train_targets", 1000)),
        "missing": missing,
        "requires_new_llm_scoring": False,
        "requires_gpu": False,
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ready":
        raise RuntimeError("Stage 2R preflight failed; complete Stage 1/2 cached scoring first")
    return result


def fit(config_path: Path) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    labels = _labels(config)
    paths = _paths(root, config)
    records = _read_jsonl_dicts(paths["train_records"])
    minimum = int(section.get("minimum_train_targets", 1000))
    if len(records) < minimum:
        raise ValueError(f"Need at least {minimum} Stage 2 train records, found {len(records)}")
    train_dialogues = _load_split(root, config, "train")
    temperature = _fit_shared_temperature(records, labels)
    examples = _build_examples(
        records,
        train_dialogues,
        labels,
        float(temperature["temperature"]),
        int(section.get("max_history_turns", 12)),
    )
    uncertainty = _choose_uncertainty_threshold(records, labels, float(temperature["temperature"]))
    seeds = [int(seed) for seed in section.get("seeds", [42, 43, 44])]
    seed_models = [_fit_seed(examples, labels, section, seed) for seed in seeds]
    payload = {
        "experiment": "LLM_context_router_stage2R",
        "version": STAGE2R_VERSION,
        "fit_split": "train",
        "validation_used_for_fitting": False,
        "test_locked": True,
        "labels": labels,
        "temperature": temperature,
        "uncertainty_router": uncertainty,
        "base_feature_names": TARGET_FEATURE_NAMES,
        "cue_feature_names": TARGET_FEATURE_NAMES + CUE_EXTRA_FEATURE_NAMES,
        "cue_semantics": "causal input-only proxy features; not scene labels and not EmotionCueCoT annotations",
        "train_targets": len(examples),
        "history_eligible_train_targets": sum(example["history_count"] > 0 for example in examples),
        "seeds": seeds,
        "seed_models": seed_models,
    }
    _write_json(paths["models"], payload)
    print(f"Stage 2R fitted models: {paths['models']}")
    return payload


VARIANT_LABELS = {
    "current_only": "2R-0 current-only",
    "always_full": "2R-1 always-full",
    "entropy_router": "2R-A frozen entropy router",
    "utility_regression": "2R-B utility regression",
    "base_gate": "2R-C direct gate",
    "cue_gate": "2R-D cue-aware direct gate",
    "oracle": "Oracle current/full",
}


def _fixed_records(
    examples: Sequence[dict[str, Any]], labels: Sequence[str], branch: str
) -> list[dict[str, Any]]:
    output = []
    for example in examples:
        if branch == "oracle":
            payload = example["full"] if example["utility"] > 0.0 else example["current"]
        else:
            payload = example[branch]
        output.append(
            _variant_record(example, {branch: payload}, labels)
            | {
                "history_count": example["history_count"],
                "uses_context": branch == "always_full" and example["history_count"] > 0,
                "router_score": None,
                "utility": example["utility"],
            }
        )
    return output


def _router_diagnostic(
    records: Sequence[dict[str, Any]], utility_epsilon: float
) -> dict[str, Any]:
    eligible = [record for record in records if record["history_count"] > 0]
    truth = np.asarray([record["utility"] > utility_epsilon for record in eligible], dtype=np.int64)
    scores = np.asarray([record["router_score"] for record in eligible], dtype=np.float64)
    routed = np.asarray([record["uses_context"] for record in eligible], dtype=np.int64)
    result = {
        "count": len(eligible),
        "positive_rate": float(truth.mean()) if len(truth) else None,
        "context_use_rate": float(routed.mean()) if len(routed) else None,
        "score_mean": float(scores.mean()) if len(scores) else None,
        "score_std": float(scores.std(ddof=0)) if len(scores) else None,
    }
    if len(set(truth.tolist())) == 2:
        result.update(
            {
                "average_precision": float(average_precision_score(truth, scores)),
                "roc_auc": float(roc_auc_score(truth, scores)),
                "f1": float(f1_score(truth, routed, zero_division=0)),
            }
        )
    else:
        result.update({"average_precision": None, "roc_auc": None, "f1": None})
    return result


def _aggregate_seed_reports(seed_reports: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_names = ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece")
    for variant in VARIANT_LABELS:
        metrics = [report["metrics"][variant] for report in seed_reports]
        diagnostics = [report["diagnostics"].get(variant) for report in seed_reports]
        payload = {
            "label": VARIANT_LABELS[variant],
            **{metric: _mean_std([item[metric] for item in metrics]) for metric in metric_names},
            "per_emotion_f1": {
                label: _mean_std([item["per_emotion_f1"][label] for item in metrics])
                for label in labels
            },
        }
        if all(item is not None for item in diagnostics):
            payload["router"] = {
                key: _mean_std([item[key] for item in diagnostics if item[key] is not None])
                for key in ("average_precision", "roc_auc", "f1", "context_use_rate", "score_mean")
            }
        output[variant] = payload
    return output


def _bootstrap(
    seed_records: Sequence[dict[str, list[dict[str, Any]]]],
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    dialogue_ids = sorted({record["dialogue_id"] for record in seed_records[0]["cue_gate"]})
    indexed: list[dict[str, dict[str, list[dict[str, Any]]]]] = []
    for report in seed_records:
        variants: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for variant in ("entropy_router", "cue_gate"):
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for record in report[variant]:
                grouped[record["dialogue_id"]].append(record)
            variants[variant] = grouped
        indexed.append(variants)
    values = {metric: [] for metric in ("weighted_f1", "macro_f1", "nll", "brier")}
    rng = random.Random(seed)
    for _ in range(samples):
        sampled_ids = [rng.choice(dialogue_ids) for _ in dialogue_ids]
        seed_deltas = {metric: [] for metric in values}
        for variants in indexed:
            baseline = [record for dialogue in sampled_ids for record in variants["entropy_router"][dialogue]]
            proposed = [record for dialogue in sampled_ids for record in variants["cue_gate"][dialogue]]
            base_metrics = _metric_bundle(baseline, "entropy_router", labels)
            proposed_metrics = _metric_bundle(proposed, "cue_gate", labels)
            for metric in values:
                delta = (
                    base_metrics[metric] - proposed_metrics[metric]
                    if metric in {"nll", "brier"}
                    else proposed_metrics[metric] - base_metrics[metric]
                )
                seed_deltas[metric].append(float(delta))
        for metric in values:
            values[metric].append(float(np.mean(seed_deltas[metric])))
    return {
        "method": "paired dialogue-level bootstrap averaged over fixed model seeds",
        "proposed": "2R-D cue-aware direct gate",
        "control": "2R-A frozen entropy router",
        "dialogues": len(dialogue_ids),
        "samples": samples,
        "seed": seed,
        "advantage": {
            metric: {
                "mean": float(np.mean(samples_values)),
                "ci95": [float(np.quantile(samples_values, 0.025)), float(np.quantile(samples_values, 0.975))],
                "probability_positive": float(np.mean(np.asarray(samples_values) > 0.0)),
            }
            for metric, samples_values in values.items()
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 2R — Continuous Context-Utility Routing",
        "",
        "> Train-fitted, validation-only. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE | router AP | context use |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"].values():
        router = item.get("router", {})
        ap = router.get("average_precision", {}).get("mean")
        use = router.get("context_use_rate", {}).get("mean")
        prefix = (
            f"| {item['label']} "
            f"| {item['weighted_f1']['mean']:.4f} ± {item['weighted_f1']['std']:.4f} "
            f"| {item['macro_f1']['mean']:.4f} ± {item['macro_f1']['std']:.4f} "
            f"| {item['nll']['mean']:.4f} ± {item['nll']['std']:.4f} "
            f"| {item['brier']['mean']:.4f} ± {item['brier']['std']:.4f} "
            f"| {item['ece']['mean']:.4f} ± {item['ece']['std']:.4f} "
        )
        suffix = f"| {ap:.4f} | {use:.1%} |" if ap is not None and use is not None else "| n/a | n/a |"
        lines.append(prefix + suffix)
    decision = report["decision"]
    lines.extend(
        [
            "",
            "## Frozen validation gate",
            "",
            f"- Status: **{decision['status']}**",
            f"- Eligible for the next cross-dataset stage: **{decision['eligible']}**",
            f"- MELD test locked: **{decision['test_locked']}**",
            "",
            "The cue-aware features are causal, input-only proxies. They are not manually annotated scene labels and are not EmotionCueCoT chains of thought.",
        ]
    )
    return "\n".join(lines) + "\n"


def evaluate(config_path: Path) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    labels = _labels(config)
    paths = _paths(root, config)
    if not paths["models"].exists():
        raise FileNotFoundError(f"Run fit first: {paths['models']}")
    models = json.loads(paths["models"].read_text(encoding="utf-8"))
    if models.get("test_locked") is not True or models.get("fit_split") != "train":
        raise ValueError("Refusing non-frozen or non-train-fitted Stage 2R model artifacts")
    records = _read_jsonl_dicts(paths["validation_records"])
    dialogues = _load_split(root, config, "validation")
    examples = _build_examples(
        records,
        dialogues,
        labels,
        float(models["temperature"]["temperature"]),
        int(section.get("max_history_turns", 12)),
    )
    utility_epsilon = float(section.get("utility_epsilon", 0.02))
    entropy_scores = np.asarray(
        [
            _entropy(example["current"]["probabilities"]) / math.log(len(labels))
            if example["history_count"] > 0
            else -1.0
            for example in examples
        ],
        dtype=np.float64,
    )
    entropy_threshold = float(models["uncertainty_router"]["threshold"])
    fixed_current = _fixed_records(examples, labels, "current")
    for record in fixed_current:
        record["current_only"] = record.pop("current")
    fixed_full = _fixed_records(examples, labels, "full")
    for record in fixed_full:
        record["always_full"] = record.pop("full")
    fixed_oracle = _fixed_records(examples, labels, "oracle")

    seed_reports: list[dict[str, Any]] = []
    seed_record_sets: list[dict[str, list[dict[str, Any]]]] = []
    for seed_payload in models["seed_models"]:
        seed = int(seed_payload["seed"])
        cue_x = np.asarray([example["cue_features"] for example in examples], dtype=np.float64)
        base_x = np.asarray([example["base_features"] for example in examples], dtype=np.float64)
        scores = {
            "entropy_router": entropy_scores,
            "utility_regression": _predict_linear(seed_payload["utility_regression"]["model"], cue_x),
            "base_gate": _predict_gate(seed_payload["base_gate"]["model"], base_x),
            "cue_gate": _predict_gate(seed_payload["cue_gate"]["model"], cue_x),
        }
        thresholds = {
            "entropy_router": entropy_threshold,
            "utility_regression": float(seed_payload["utility_regression"]["selection"]["threshold"]),
            "base_gate": float(seed_payload["base_gate"]["selection"]["threshold"]),
            "cue_gate": float(seed_payload["cue_gate"]["selection"]["threshold"]),
        }
        records_by_variant: dict[str, list[dict[str, Any]]] = {
            "current_only": fixed_current,
            "always_full": fixed_full,
            "oracle": fixed_oracle,
        }
        for variant in scores:
            records_by_variant[variant] = _records_for_branch(
                examples, labels, scores[variant], thresholds[variant], variant
            )
        metrics = {
            variant: _metric_bundle(records_by_variant[variant], variant, labels)
            for variant in VARIANT_LABELS
        }
        diagnostics = {
            variant: _router_diagnostic(records_by_variant[variant], utility_epsilon)
            for variant in ("entropy_router", "utility_regression", "base_gate", "cue_gate")
        }
        seed_reports.append({"seed": seed, "metrics": metrics, "diagnostics": diagnostics})
        seed_record_sets.append(records_by_variant)
        prediction_file = paths["base"] / "validation" / f"predictions_seed{seed}.jsonl"
        prediction_file.parent.mkdir(parents=True, exist_ok=True)
        with prediction_file.open("w", encoding="utf-8") as handle:
            for index, example in enumerate(examples):
                payload: dict[str, Any] = {
                    "key": example["key"],
                    "dialogue_id": example["dialogue_id"],
                    "gold": example["gold"],
                    "history_count": example["history_count"],
                    "utility": example["utility"],
                }
                for variant in VARIANT_LABELS:
                    record = records_by_variant[variant][index]
                    payload[variant] = {
                        "prediction": record[variant]["prediction"],
                        "probabilities": record[variant]["probabilities"],
                        "uses_context": record.get("uses_context"),
                        "router_score": record.get("router_score"),
                    }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    variants = _aggregate_seed_reports(seed_reports, labels)
    bootstrap = _bootstrap(
        seed_record_sets,
        labels,
        int(section.get("bootstrap_samples", 2000)),
        int(section.get("bootstrap_seed", 2028)),
    )
    _write_json(paths["bootstrap"], bootstrap)

    baseline = variants["entropy_router"]
    proposed = variants["cue_gate"]
    macro_deltas = [
        report["metrics"]["cue_gate"]["macro_f1"] - report["metrics"]["entropy_router"]["macro_f1"]
        for report in seed_reports
    ]
    checks = {
        "macro_not_lower_than_entropy_router": proposed["macro_f1"]["mean"] >= baseline["macro_f1"]["mean"],
        "weighted_f1_drop_within_tolerance": proposed["weighted_f1"]["mean"] >= baseline["weighted_f1"]["mean"] - float(section.get("max_weighted_f1_drop", 0.003)),
        "nll_within_tolerance": proposed["nll"]["mean"] <= baseline["nll"]["mean"] + float(section.get("max_nll_increase", 0.01)),
        "router_average_precision": proposed["router"]["average_precision"]["mean"] >= float(section.get("minimum_router_ap", 0.55)),
        "context_use_rate": proposed["router"]["context_use_rate"]["mean"] <= float(section.get("maximum_context_use_rate", 0.45)),
        "positive_macro_seeds": sum(delta > 0.0 for delta in macro_deltas) >= int(section.get("minimum_positive_macro_seeds", 2)),
    }
    eligible = all(checks.values())
    decision = {
        "status": "eligible_for_cross_dataset_replication" if eligible else "stop_and_diagnose_router",
        "eligible": eligible,
        "checks": checks,
        "macro_advantage_by_seed": macro_deltas,
        "selection_basis": "train-only OOF thresholds; one-time MELD validation evaluation",
        "test_locked": True,
        "next_action": "Replicate the frozen router on EmoryNLP before considering any final test." if eligible else "Do not inspect MELD test; analyze router labels/features and revise only on train/validation.",
    }
    _write_json(paths["decision"], decision)
    report = {
        "experiment": "LLM_context_router_stage2R",
        "version": STAGE2R_VERSION,
        "split": "validation",
        "seeds": models["seeds"],
        "test_locked": True,
        "temperature": models["temperature"],
        "targets": len(examples),
        "history_eligible_targets": sum(example["history_count"] > 0 for example in examples),
        "variants": variants,
        "seed_reports": seed_reports,
        "paired_bootstrap": bootstrap,
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["report_json"], report)
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_md"].write_text(_markdown(report), encoding="utf-8")
    print(_markdown(report))
    print(f"JSON report: {paths['report_json']}")
    return report


def verify(config_path: Path) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    paths = _paths(root, config)
    required = ("models", "report_json", "report_md", "bootstrap", "decision")
    missing = [str(paths[name]) for name in required if not paths[name].exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2R artifacts: {missing}")
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    models = json.loads(paths["models"].read_text(encoding="utf-8"))
    checks = {
        "fit_split_is_train": models.get("fit_split") == "train",
        "validation_not_used_for_fitting": models.get("validation_used_for_fitting") is False,
        "test_locked_in_models": models.get("test_locked") is True,
        "test_locked_in_report": report.get("test_locked") is True,
        "has_three_or_more_seeds": len(models.get("seeds", [])) >= 3,
        "has_cue_gate": "cue_gate" in report.get("variants", {}),
        "has_bootstrap": int(report.get("paired_bootstrap", {}).get("samples", 0)) > 0,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 2R verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2R continuous context-utility routing")
    parser.add_argument("stage", choices=("check", "fit", "evaluate", "verify", "all-validation"))
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "check":
        check(arguments.config)
    elif arguments.stage == "fit":
        fit(arguments.config)
    elif arguments.stage == "evaluate":
        evaluate(arguments.config)
    elif arguments.stage == "verify":
        verify(arguments.config)
    else:
        check(arguments.config)
        fit(arguments.config)
        evaluate(arguments.config)
        verify(arguments.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
