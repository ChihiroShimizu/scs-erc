from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pfa_erc.data import Dialogue, Utterance, read_jsonl
from pfa_erc.training.llm_context_utility_stage1 import (
    CausalLabelScorer,
    TargetSpec,
    _append_jsonl,
    _metric_bundle,
    _prediction_payload,
    _read_jsonl_dicts,
    _record_for_target,
    _softmax,
    _write_json,
    build_prompt,
    make_target_specs,
)


STAGE2_VERSION = "1.1.0"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _root_and_config(config_path: Path) -> tuple[Path, dict[str, Any]]:
    path = config_path.expanduser().resolve()
    return path.parent.parent, _read_yaml(path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _paths(root: Path, config: dict[str, Any], run_name: str = "validation") -> dict[str, Path]:
    section = config["llm_selective_context"]
    base = _resolve(root, section.get("artifact_dir", "artifacts/llm_selective_context_stage2/meld"))
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "train_records": base / "train" / "utility_records.jsonl",
        "train_manifest": base / "train" / "manifest.json",
        "models": base / "models" / "selective_context_models.json",
        "validation_records": base / "validation" / "selected_context_records.jsonl",
        "report_json": base / "reports" / "validation_summary.json",
        "report_md": base / "reports" / "validation_summary.md",
    }


def _load_split(root: Path, config: dict[str, Any], split: str) -> list[Dialogue]:
    if split == "test":
        raise ValueError("Stage 2 is train/validation-only; MELD test remains locked")
    path = _resolve(root, config["data"]["processed_dir"]) / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")
    return read_jsonl(path)


def _labels(config: dict[str, Any]) -> list[str]:
    labels = list(config["llm_selective_context"].get("labels", []))
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("llm_selective_context.labels must contain unique labels")
    return labels


def _word_tokens(text: str) -> set[str]:
    return {token.lower().strip(".,!?;:'\"()[]{}") for token in text.split() if token.strip()}


def _jaccard(left: str, right: str) -> float:
    a, b = _word_tokens(left), _word_tokens(right)
    if not a and not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))


def _margin(probabilities: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(probabilities, dtype=np.float64))
    return float(ordered[-1] - ordered[-2])


def _calibrated_payload(scores: Sequence[float], labels: Sequence[str], temperature: float) -> dict[str, Any]:
    return _prediction_payload([float(value) / temperature for value in scores], labels)


def _gold_nll(payload: dict[str, Any], gold: str, labels: Sequence[str]) -> float:
    return float(-math.log(max(payload["probabilities"][labels.index(gold)], 1e-12)))


def _fit_shared_temperature(records: Sequence[dict[str, Any]], labels: Sequence[str]) -> dict[str, Any]:
    score_rows: list[list[float]] = []
    gold_ids: list[int] = []
    for record in records:
        for branch in ("current", "full"):
            score_rows.append(record[branch]["scores"])
            gold_ids.append(labels.index(record["gold"]))
    scores = np.asarray(score_rows, dtype=np.float64)
    gold = np.asarray(gold_ids, dtype=np.int64)

    def nll(temperature: float) -> float:
        logits = scores / temperature
        logits -= logits.max(axis=1, keepdims=True)
        log_denom = np.log(np.exp(logits).sum(axis=1))
        return float(np.mean(log_denom - logits[np.arange(len(gold)), gold]))

    coarse = np.geomspace(0.20, 10.0, 240)
    coarse_losses = np.asarray([nll(float(value)) for value in coarse])
    best = float(coarse[int(np.argmin(coarse_losses))])
    lower, upper = max(0.10, best / 1.15), min(20.0, best * 1.15)
    fine = np.linspace(lower, upper, 200)
    fine_losses = np.asarray([nll(float(value)) for value in fine])
    temperature = float(fine[int(np.argmin(fine_losses))])
    return {
        "temperature": temperature,
        "uncalibrated_nll": nll(1.0),
        "calibrated_nll": nll(temperature),
        "improvement": nll(1.0) - nll(temperature),
        "fit_examples": len(gold),
        "fit_split": "train",
        "shared_across_branches": True,
    }


TARGET_FEATURE_NAMES = [
    "current_p_anger",
    "current_p_disgust",
    "current_p_fear",
    "current_p_joy",
    "current_p_neutral",
    "current_p_sadness",
    "current_p_surprise",
    "current_entropy",
    "current_margin",
    "current_confidence",
    "history_fraction",
    "same_speaker_history_fraction",
    "speaker_gap_fraction",
    "target_log_word_length",
    "target_has_question",
    "target_has_exclamation",
]

SELECTOR_EXTRA_FEATURE_NAMES = [
    "source_same_speaker",
    "source_gap_fraction",
    "source_log_word_length",
    "source_target_jaccard",
    "source_entropy",
    "source_target_probability_l1",
    "source_target_probability_dot",
    "source_target_same_argmax",
]


def target_features(
    record: dict[str, Any],
    target: TargetSpec,
    labels: Sequence[str],
    temperature: float,
    max_history_turns: int,
) -> list[float]:
    payload = _calibrated_payload(record["current"]["scores"], labels, temperature)
    probabilities = list(payload["probabilities"])
    same_count = sum(turn.speaker == target.speaker for turn in target.history)
    previous_same = [i for i, turn in enumerate(target.history) if turn.speaker == target.speaker]
    speaker_gap = len(target.history) - previous_same[-1] if previous_same else max_history_turns
    return [
        *probabilities,
        _entropy(probabilities) / math.log(len(labels)),
        _margin(probabilities),
        max(probabilities),
        len(target.history) / max(max_history_turns, 1),
        same_count / max(len(target.history), 1),
        min(speaker_gap, max_history_turns) / max(max_history_turns, 1),
        math.log1p(len(target.text.split())) / math.log(65.0),
        float("?" in target.text),
        float("!" in target.text),
    ]


def selector_features(
    target_vector: Sequence[float],
    target: TargetSpec,
    source: Utterance,
    turn_gap: int,
    source_probabilities: Sequence[float],
    labels: Sequence[str],
    max_history_turns: int,
) -> list[float]:
    target_probabilities = np.asarray(target_vector[: len(labels)], dtype=np.float64)
    source_probabilities_array = np.asarray(source_probabilities, dtype=np.float64)
    return [
        *target_vector,
        float(source.speaker == target.speaker),
        min(turn_gap, max_history_turns) / max(max_history_turns, 1),
        math.log1p(len(source.text.split())) / math.log(65.0),
        _jaccard(source.text, target.text),
        _entropy(source_probabilities_array) / math.log(len(labels)),
        float(np.abs(source_probabilities_array - target_probabilities).sum()),
        float(np.dot(source_probabilities_array, target_probabilities)),
        float(np.argmax(source_probabilities_array) == np.argmax(target_probabilities)),
    ]


def _dialogue_target_maps(
    dialogues: Sequence[Dialogue], max_history_turns: int
) -> tuple[dict[str, TargetSpec], dict[tuple[str, int], TargetSpec]]:
    by_key: dict[str, TargetSpec] = {}
    by_turn: dict[tuple[str, int], TargetSpec] = {}
    for dialogue in dialogues:
        for target in make_target_specs(dialogue, max_history_turns=max_history_turns):
            by_key[target.key] = target
            by_turn[(target.dialogue_id, target.turn_index)] = target
    return by_key, by_turn


def _serialize_binary_model(
    features: Sequence[Sequence[float]], labels: Sequence[int], seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError("Binary supervision must contain both classes")
    scaler = StandardScaler().fit(x)
    normalized = scaler.transform(x)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        solver="lbfgs",
    ).fit(normalized, y)
    probabilities = model.predict_proba(normalized)[:, 1]
    return (
        {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "positive_rate": float(y.mean()),
            "train_examples": len(y),
            "train_average_precision": float(average_precision_score(y, probabilities)),
            "train_roc_auc": float(roc_auc_score(y, probabilities)),
            "train_f1_at_050": float(f1_score(y, probabilities >= 0.5, zero_division=0)),
        },
        probabilities,
    )


def _predict_binary(model: dict[str, Any], features: Sequence[Sequence[float]]) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    mean = np.asarray(model["mean"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    coef = np.asarray(model["coef"], dtype=np.float64)
    logits = ((x - mean) / np.where(scale == 0, 1.0, scale)) @ coef + float(model["intercept"])
    logits = np.clip(logits, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-logits))


def _source_probabilities(
    record_by_turn: dict[tuple[str, int], dict[str, Any]],
    dialogue_id: str,
    source_turn_index: int,
    labels: Sequence[str],
    temperature: float,
) -> list[float]:
    source_record = record_by_turn.get((dialogue_id, source_turn_index))
    if source_record is None:
        return [1.0 / len(labels)] * len(labels)
    return list(
        _calibrated_payload(source_record["current"]["scores"], labels, temperature)[
            "probabilities"
        ]
    )


def generate_train(
    config_path: Path,
    max_targets: int | None,
    overwrite: bool,
    model_override: str | None,
    run_name: str,
) -> None:
    root, config = _root_and_config(config_path)
    section = config["llm_selective_context"]
    labels = _labels(config)
    dialogues = _load_split(root, config, "train")
    random.Random(int(section.get("dialogue_shuffle_seed", 2026))).shuffle(dialogues)
    paths = _paths(root, config, run_name)
    if overwrite and paths["train_records"].exists():
        paths["train_records"].unlink()
    existing = _read_jsonl_dicts(paths["train_records"])
    completed = {record["key"] for record in existing}
    model_config = dict(section)
    if model_override:
        model_config["model_name"] = model_override
    scorer = CausalLabelScorer(model_config, labels)
    max_history = int(section.get("max_history_turns", 12))
    preamble = str(section["prompt_preamble"])
    utility_epsilon = float(section.get("utility_epsilon", 0.02))
    deletion_epsilon = float(section.get("deletion_epsilon", 0.01))
    _write_json(
        paths["train_manifest"],
        {
            "experiment": "LLM_selective_context_stage2",
            "version": STAGE2_VERSION,
            "split": "train",
            "test_locked": True,
            "scorer": scorer.audit,
            "max_history_turns": max_history,
            "max_targets": max_targets,
        },
    )
    processed = len(existing)
    stop = False
    for dialogue in tqdm(dialogues, desc="Stage 2 train utility", unit="dialogue"):
        pending: list[TargetSpec] = []
        for target in make_target_specs(dialogue, max_history_turns=max_history):
            if target.key in completed:
                continue
            if max_targets is not None and processed + len(pending) >= max_targets:
                stop = True
                break
            pending.append(target)
        prompts: list[str] = []
        spans: list[tuple[int, int]] = []
        for target in pending:
            start = len(prompts)
            prompts.extend(
                [
                    build_prompt(target, labels, [], preamble),
                    build_prompt(target, labels, target.history, preamble),
                ]
            )
            for deletion_index in range(len(target.history)):
                history = target.history[:deletion_index] + target.history[deletion_index + 1 :]
                prompts.append(build_prompt(target, labels, history, preamble))
            spans.append((start, len(prompts)))
        if pending:
            scores = scorer.score_prompts(prompts)
            records: list[dict[str, Any]] = []
            for target, (start, end) in zip(pending, spans):
                target_scores = scores[start:end]
                record = _record_for_target(
                    target,
                    labels,
                    target_scores,
                    utility_epsilon,
                    deletion_epsilon,
                )
                for deletion, raw_scores in zip(record["deletions"], target_scores[2:]):
                    deletion["deleted_scores"] = [float(value) for value in raw_scores]
                records.append(record)
            _append_jsonl(paths["train_records"], records)
            processed += len(records)
        if stop:
            break
    print(f"Train utility records: {paths['train_records']} ({processed} targets)")


def _variant_record(
    base: dict[str, Any], payloads: dict[str, dict[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    result = {
        "key": base["key"],
        "dialogue_id": base["dialogue_id"],
        "gold": base["gold"],
    }
    for name, payload in payloads.items():
        copied = dict(payload)
        copied["nll"] = _gold_nll(copied, base["gold"], labels)
        result[name] = copied
    return result


def _choose_uncertainty_threshold(
    records: Sequence[dict[str, Any]], labels: Sequence[str], temperature: float
) -> dict[str, Any]:
    adjusted: list[dict[str, Any]] = []
    entropies: list[float] = []
    for record in records:
        current = _calibrated_payload(record["current"]["scores"], labels, temperature)
        full = _calibrated_payload(record["full"]["scores"], labels, temperature)
        adjusted.append(_variant_record(record, {"current": current, "full": full}, labels))
        entropies.append(_entropy(current["probabilities"]) / math.log(len(labels)))
    candidates = np.linspace(0.10, 0.95, 86)
    best: tuple[float, float, dict[str, float]] | None = None
    for threshold in candidates:
        routed = []
        for record, entropy in zip(adjusted, entropies):
            branch = record["full"] if entropy >= threshold else record["current"]
            routed.append({**record, "routed": branch})
        metrics = _metric_bundle(routed, "routed", labels)
        objective = 0.5 * (metrics["weighted_f1"] + metrics["macro_f1"])
        candidate = (objective, -float(threshold), metrics)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    assert best is not None
    return {
        "threshold": -best[1],
        "train_selection": best[0],
        "train_weighted_f1": best[2]["weighted_f1"],
        "train_macro_f1": best[2]["macro_f1"],
        "fit_split": "train",
    }


def fit_models(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = config["llm_selective_context"]
    labels = _labels(config)
    paths = _paths(root, config, run_name)
    records = _read_jsonl_dicts(paths["train_records"])
    minimum = int(section.get("minimum_train_targets", 1000))
    if len(records) < minimum:
        raise ValueError(f"Need at least {minimum} train records, found {len(records)}")
    dialogues = _load_split(root, config, "train")
    max_history = int(section.get("max_history_turns", 12))
    target_by_key, target_by_turn = _dialogue_target_maps(dialogues, max_history)
    record_by_turn = {
        (record["dialogue_id"], int(record["turn_index"])): record for record in records
    }
    calibration = _fit_shared_temperature(records, labels)
    temperature = float(calibration["temperature"])
    utility_epsilon = float(section.get("utility_epsilon", 0.02))
    deletion_epsilon = float(section.get("deletion_epsilon", 0.01))

    router_x: list[list[float]] = []
    router_y: list[int] = []
    selector_x: list[list[float]] = []
    selector_y: list[int] = []
    for record in records:
        target = target_by_key.get(record["key"])
        if target is None or not target.history:
            continue
        vector = target_features(record, target, labels, temperature, max_history)
        current = _calibrated_payload(record["current"]["scores"], labels, temperature)
        full = _calibrated_payload(record["full"]["scores"], labels, temperature)
        utility = _gold_nll(current, record["gold"], labels) - _gold_nll(
            full, record["gold"], labels
        )
        router_x.append(vector)
        router_y.append(int(utility > utility_epsilon))
        for deletion in record.get("deletions", []):
            history_index = int(deletion["history_index"])
            if history_index >= len(target.history):
                continue
            source = target.history[history_index]
            source_turn_index = target.turn_index - int(deletion["turn_gap"])
            source_probs = _source_probabilities(
                record_by_turn,
                target.dialogue_id,
                source_turn_index,
                labels,
                temperature,
            )
            selector_x.append(
                selector_features(
                    vector,
                    target,
                    source,
                    int(deletion["turn_gap"]),
                    source_probs,
                    labels,
                    max_history,
                )
            )
            # Stage 1 validation records predate storage of deleted raw scores.
            # Keep the train and validation selector target definition identical.
            delta = float(deletion["delta_nll"])
            selector_y.append(int(delta > deletion_epsilon))

    seed = int(section.get("model_seed", 42))
    router, _ = _serialize_binary_model(router_x, router_y, seed)
    selector, _ = _serialize_binary_model(selector_x, selector_y, seed)
    train_manifest = json.loads(paths["train_manifest"].read_text(encoding="utf-8"))
    payload = {
        "experiment": "LLM_selective_context_stage2",
        "version": STAGE2_VERSION,
        "fit_split": "train",
        "test_locked": True,
        "labels": labels,
        "model_name": train_manifest["scorer"]["model_name"],
        "temperature": calibration,
        "target_feature_names": TARGET_FEATURE_NAMES,
        "selector_feature_names": TARGET_FEATURE_NAMES + SELECTOR_EXTRA_FEATURE_NAMES,
        "router": router,
        "selector": selector,
        "router_threshold": float(section.get("router_threshold", 0.50)),
        "selector_threshold": float(section.get("selector_threshold", 0.50)),
        "max_selected_turns": int(section.get("max_selected_turns", 3)),
        "uncertainty_router": _choose_uncertainty_threshold(records, labels, temperature),
        "train_targets": len(records),
    }
    _write_json(paths["models"], payload)
    print(json.dumps(payload, indent=2))
    print(f"Model file: {paths['models']}")
    return payload


def _validation_stage1_paths(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    section = config["llm_selective_context"]
    records = _resolve(root, section["stage1_validation_records"])
    manifest = _resolve(root, section["stage1_validation_manifest"])
    return records, manifest


def _selection_plan(
    record: dict[str, Any],
    target: TargetSpec,
    record_by_turn: dict[tuple[str, int], dict[str, Any]],
    model_payload: dict[str, Any],
    labels: Sequence[str],
    max_history: int,
) -> dict[str, Any]:
    temperature = float(model_payload["temperature"]["temperature"])
    target_vector = target_features(record, target, labels, temperature, max_history)
    router_probability = float(_predict_binary(model_payload["router"], [target_vector])[0])
    selector_vectors: list[list[float]] = []
    for history_index, source in enumerate(target.history):
        turn_gap = len(target.history) - history_index
        source_turn_index = target.turn_index - turn_gap
        source_probs = _source_probabilities(
            record_by_turn,
            target.dialogue_id,
            source_turn_index,
            labels,
            temperature,
        )
        selector_vectors.append(
            selector_features(
                target_vector,
                target,
                source,
                turn_gap,
                source_probs,
                labels,
                max_history,
            )
        )
    selector_probabilities = (
        _predict_binary(model_payload["selector"], selector_vectors).tolist()
        if selector_vectors
        else []
    )
    threshold = float(model_payload["selector_threshold"])
    candidates = [
        (index, probability)
        for index, probability in enumerate(selector_probabilities)
        if probability >= threshold
    ]
    candidates.sort(key=lambda item: (-item[1], -item[0]))
    selected_indices = sorted(
        index for index, _ in candidates[: int(model_payload["max_selected_turns"])]
    )
    return {
        "key": record["key"],
        "router_probability": router_probability,
        "selector_probabilities": selector_probabilities,
        "selected_history_indices": selected_indices,
    }


def evaluate_validation(
    config_path: Path,
    run_name: str,
    max_targets: int | None,
    overwrite: bool,
    model_override: str | None,
) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = config["llm_selective_context"]
    labels = _labels(config)
    paths = _paths(root, config, run_name)
    if not paths["models"].exists():
        raise FileNotFoundError(f"Run fit first: {paths['models']}")
    model_payload = json.loads(paths["models"].read_text(encoding="utf-8"))
    requested_model = model_override or str(section["model_name"])
    if requested_model != model_payload["model_name"]:
        raise ValueError(
            f"Model mismatch: fitted artifacts use {model_payload['model_name']!r}, "
            f"but evaluation requested {requested_model!r}"
        )
    stage1_records_path, _ = _validation_stage1_paths(root, config)
    stage1_records = _read_jsonl_dicts(stage1_records_path)
    if max_targets is not None:
        stage1_records = stage1_records[:max_targets]
    dialogues = _load_split(root, config, "validation")
    max_history = int(section.get("max_history_turns", 12))
    target_by_key, _ = _dialogue_target_maps(dialogues, max_history)
    record_by_turn = {
        (record["dialogue_id"], int(record["turn_index"])): record
        for record in stage1_records
    }
    if overwrite and paths["validation_records"].exists():
        paths["validation_records"].unlink()
    existing = _read_jsonl_dicts(paths["validation_records"])
    existing_by_key = {record["key"]: record for record in existing}
    plans: list[dict[str, Any]] = []
    prompt_plan: list[tuple[dict[str, Any], TargetSpec, list[Utterance]]] = []
    for record in stage1_records:
        if record["key"] in existing_by_key:
            continue
        target = target_by_key.get(record["key"])
        if target is None:
            raise KeyError(f"Could not reconstruct target {record['key']}")
        plan = _selection_plan(
            record, target, record_by_turn, model_payload, labels, max_history
        )
        selected = [target.history[index] for index in plan["selected_history_indices"]]
        plans.append(plan)
        prompt_plan.append((plan, target, selected))

    if prompt_plan:
        model_config = dict(section)
        model_config["model_name"] = requested_model
        scorer = CausalLabelScorer(model_config, labels)
        preamble = str(section["prompt_preamble"])
        prompts = [
            build_prompt(target, labels, selected, preamble)
            for _, target, selected in prompt_plan
        ]
        selected_scores = scorer.score_prompts(prompts)
        output = []
        for (plan, _, _), scores in zip(prompt_plan, selected_scores):
            output.append({**plan, "selected_scores": [float(value) for value in scores]})
        _append_jsonl(paths["validation_records"], output)

    return summarize(config_path, run_name, max_targets=max_targets)


def _branch_records(
    stage1_records: Sequence[dict[str, Any]],
    selected_by_key: dict[str, dict[str, Any]],
    model_payload: dict[str, Any],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    temperature = float(model_payload["temperature"]["temperature"])
    router_threshold = float(model_payload["router_threshold"])
    uncertainty_threshold = float(model_payload["uncertainty_router"]["threshold"])
    output: list[dict[str, Any]] = []
    for record in stage1_records:
        selected_record = selected_by_key[record["key"]]
        current = _calibrated_payload(record["current"]["scores"], labels, temperature)
        full = _calibrated_payload(record["full"]["scores"], labels, temperature)
        selected = _calibrated_payload(selected_record["selected_scores"], labels, temperature)
        entropy = _entropy(current["probabilities"]) / math.log(len(labels))
        uncertainty = full if entropy >= uncertainty_threshold else current
        router_full = (
            full
            if float(selected_record["router_probability"]) >= router_threshold
            else current
        )
        use_selected = (
            float(selected_record["router_probability"]) >= router_threshold
            and bool(selected_record["selected_history_indices"])
        )
        router_selector = selected if use_selected else current
        selector_only = (
            selected if bool(selected_record["selected_history_indices"]) else current
        )
        current_nll = _gold_nll(current, record["gold"], labels)
        full_nll = _gold_nll(full, record["gold"], labels)
        oracle = full if full_nll < current_nll else current
        output.append(
            _variant_record(
                record,
                {
                    "current_only": current,
                    "always_full": full,
                    "uncertainty_router": uncertainty,
                    "learned_router_full": router_full,
                    "selector_only": selector_only,
                    "router_selector": router_selector,
                    "oracle_current_full": oracle,
                },
                labels,
            )
            | {
                "router_probability": float(selected_record["router_probability"]),
                "selector_probabilities": selected_record["selector_probabilities"],
                "selected_history_indices": selected_record["selected_history_indices"],
                "history_count": int(record["history_count"]),
                "raw_deletions": record.get("deletions", []),
            }
        )
    return output


def _flip_stats(records: Sequence[dict[str, Any]], branch: str) -> dict[str, int]:
    good = sum(
        record["current_only"]["prediction"] != record["gold"]
        and record[branch]["prediction"] == record["gold"]
        for record in records
    )
    bad = sum(
        record["current_only"]["prediction"] == record["gold"]
        and record[branch]["prediction"] != record["gold"]
        for record in records
    )
    return {"good": good, "bad": bad, "net": good - bad}


def _diagnostics(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    model_payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    section = config["llm_selective_context"]
    temperature = float(model_payload["temperature"]["temperature"])
    utility_epsilon = float(section.get("utility_epsilon", 0.02))
    deletion_epsilon = float(section.get("deletion_epsilon", 0.01))
    router_y: list[int] = []
    router_p: list[float] = []
    selector_y: list[int] = []
    selector_p: list[float] = []
    for record in records:
        current_nll = record["current_only"]["nll"]
        full_nll = record["always_full"]["nll"]
        if record["history_count"]:
            router_y.append(int(current_nll - full_nll > utility_epsilon))
            router_p.append(float(record["router_probability"]))
        for deletion, probability in zip(
            record["raw_deletions"], record["selector_probabilities"]
        ):
            selector_y.append(int(float(deletion["delta_nll"]) > deletion_epsilon))
            selector_p.append(float(probability))

    def binary_report(y: Sequence[int], p: Sequence[float]) -> dict[str, Any]:
        if not y or len(set(y)) < 2:
            return {"count": len(y), "average_precision": None, "roc_auc": None, "f1": None}
        return {
            "count": len(y),
            "positive_rate": float(np.mean(y)),
            "average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)),
            "f1": float(f1_score(y, np.asarray(p) >= 0.5, zero_division=0)),
        }

    selected_counts = [len(record["selected_history_indices"]) for record in records]
    history_counts = [record["history_count"] for record in records]
    return {
        "shared_temperature": temperature,
        "router": binary_report(router_y, router_p),
        "selector": binary_report(selector_y, selector_p),
        "mean_selected_turns": float(np.mean(selected_counts)),
        "context_use_rate": float(
            np.mean(
                [
                    record["router_probability"] >= model_payload["router_threshold"]
                    and bool(record["selected_history_indices"])
                    for record in records
                ]
            )
        ),
        "turn_retention_rate": float(sum(selected_counts) / max(sum(history_counts), 1)),
    }


def _bootstrap(
    records: Sequence[dict[str, Any]], labels: Sequence[str], samples: int, seed: int
) -> dict[str, Any]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dialogue[record["dialogue_id"]].append(record)
    dialogue_ids = sorted(by_dialogue)
    rng = random.Random(seed)
    controls = ("current_only", "always_full")
    values = {
        control: {metric: [] for metric in ("weighted_f1", "macro_f1", "nll", "brier")}
        for control in controls
    }
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for dialogue_id in (rng.choice(dialogue_ids) for _ in dialogue_ids):
            sampled.extend(by_dialogue[dialogue_id])
        method = _metric_bundle(sampled, "router_selector", labels)
        for control in controls:
            baseline = _metric_bundle(sampled, control, labels)
            for metric in values[control]:
                delta = (
                    baseline[metric] - method[metric]
                    if metric in {"nll", "brier"}
                    else method[metric] - baseline[metric]
                )
                values[control][metric].append(float(delta))
    return {
        "samples": samples,
        "seed": seed,
        "dialogues": len(dialogue_ids),
        "router_selector_advantage": {
            control: {
                metric: {
                    "mean": float(np.mean(metric_values)),
                    "ci95": [
                        float(np.percentile(metric_values, 2.5)),
                        float(np.percentile(metric_values, 97.5)),
                    ],
                    "probability_positive": float(np.mean(np.asarray(metric_values) > 0)),
                }
                for metric, metric_values in control_values.items()
            }
            for control, control_values in values.items()
        },
    }


def summarize(config_path: Path, run_name: str, max_targets: int | None = None) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = config["llm_selective_context"]
    labels = _labels(config)
    paths = _paths(root, config, run_name)
    stage1_path, _ = _validation_stage1_paths(root, config)
    stage1_records = _read_jsonl_dicts(stage1_path)
    if max_targets is not None:
        stage1_records = stage1_records[:max_targets]
    selected = _read_jsonl_dicts(paths["validation_records"])
    selected_by_key = {record["key"]: record for record in selected}
    missing = [record["key"] for record in stage1_records if record["key"] not in selected_by_key]
    if missing:
        raise ValueError(f"Selected-context scores missing for {len(missing)} targets")
    model_payload = json.loads(paths["models"].read_text(encoding="utf-8"))
    records = _branch_records(stage1_records, selected_by_key, model_payload, labels)
    branches = (
        "current_only",
        "always_full",
        "uncertainty_router",
        "learned_router_full",
        "selector_only",
        "router_selector",
        "oracle_current_full",
    )
    metrics = {branch: _metric_bundle(records, branch, labels) for branch in branches}
    flips = {branch: _flip_stats(records, branch) for branch in branches if branch != "current_only"}
    diagnostics = _diagnostics(records, labels, model_payload, config)
    bootstrap = _bootstrap(
        records,
        labels,
        int(section.get("bootstrap_samples", 2000)),
        int(section.get("bootstrap_seed", 2027)),
    )
    method = metrics["router_selector"]
    current = metrics["current_only"]
    full = metrics["always_full"]
    tolerance = float(section.get("classification_tolerance", 0.005))
    eligible = (
        method["macro_f1"] >= max(current["macro_f1"], full["macro_f1"]) - tolerance
        and method["weighted_f1"] >= current["weighted_f1"] - tolerance
        and method["nll"] <= min(current["nll"], full["nll"]) + 0.02
        and diagnostics["router"].get("average_precision", 0.0) >= 0.55
        and diagnostics["turn_retention_rate"] <= 0.50
    )
    summary = {
        "experiment": "LLM_selective_context_stage2",
        "version": STAGE2_VERSION,
        "split": "validation",
        "test_locked": True,
        "targets": len(records),
        "dialogues": len({record["dialogue_id"] for record in records}),
        "metrics": metrics,
        "flips_vs_current": flips,
        "diagnostics": diagnostics,
        "paired_dialogue_bootstrap": bootstrap,
        "selection_gate": {
            "eligible": eligible,
            "classification_tolerance": tolerance,
            "rule": (
                "Preserve best macro within tolerance, preserve current weighted F1, "
                "match best NLL within 0.02, router AP >= 0.55, retention <= 0.50."
            ),
        },
        "temperature_fit": model_payload["temperature"],
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["report_json"], summary)
    markdown = _markdown(summary)
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_md"].write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON report: {paths['report_json']}")
    return summary


def _markdown(summary: dict[str, Any]) -> str:
    labels = {
        "current_only": "2A current-only",
        "always_full": "2B always-full",
        "uncertainty_router": "2C uncertainty router",
        "learned_router_full": "2D learned router + full",
        "selector_only": "2E selector-only",
        "router_selector": "2F router + selector",
        "oracle_current_full": "Oracle current/full",
    }
    lines = [
        "# Stage 2 — Selective LLM Context Routing",
        "",
        "> Train-supervised, validation-only evaluation. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels.items():
        metric = summary["metrics"][key]
        lines.append(
            f"| {label} | {metric['weighted_f1']:.4f} | {metric['macro_f1']:.4f} | "
            f"{metric['accuracy']:.4f} | {metric['nll']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |"
        )
    diagnostic = summary["diagnostics"]
    lines.extend(
        [
            "",
            "## Mechanism diagnostics",
            "",
            f"- Shared train-fitted temperature: **{diagnostic['shared_temperature']:.4f}**",
            f"- Router AP / F1: **{diagnostic['router']['average_precision']:.4f} / {diagnostic['router']['f1']:.4f}**",
            f"- Selector AP / F1: **{diagnostic['selector']['average_precision']:.4f} / {diagnostic['selector']['f1']:.4f}**",
            f"- Context use rate: **{diagnostic['context_use_rate']:.1%}**",
            f"- Turn retention rate: **{diagnostic['turn_retention_rate']:.1%}**",
            f"- Mean selected turns: **{diagnostic['mean_selected_turns']:.3f}**",
            "",
            "## Frozen validation decision",
            "",
            f"Eligible for the next stage: **{summary['selection_gate']['eligible']}**",
            "",
            summary["selection_gate"]["rule"],
            "",
        ]
    )
    return "\n".join(lines)


def check(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    labels = _labels(config)
    train = _load_split(root, config, "train")
    validation = _load_split(root, config, "validation")
    paths = _paths(root, config, run_name)
    stage1_records, stage1_manifest = _validation_stage1_paths(root, config)
    stage1_model = None
    if stage1_manifest.exists():
        stage1_payload = json.loads(stage1_manifest.read_text(encoding="utf-8"))
        stage1_model = stage1_payload.get("scorer", {}).get("model_name")
    configured_model = config["llm_selective_context"]["model_name"]
    model_matches_stage1 = stage1_model in {None, configured_model}
    result = {
        "status": (
            "ready"
            if stage1_records.exists() and stage1_manifest.exists() and model_matches_stage1
            else "blocked"
        ),
        "labels": labels,
        "train_dialogues": len(train),
        "validation_dialogues": len(validation),
        "stage1_validation_records": str(stage1_records),
        "stage1_records_exist": stage1_records.exists(),
        "stage1_manifest_exists": stage1_manifest.exists(),
        "stage1_model_name": stage1_model,
        "configured_model_name": configured_model,
        "model_matches_stage1": model_matches_stage1,
        "stage2_train_records_exist": paths["train_records"].exists(),
        "stage2_models_exist": paths["models"].exists(),
        "test_locked": True,
        "cuda_available": torch.cuda.is_available(),
    }
    print(json.dumps(result, indent=2))
    if result["status"] != "ready":
        raise FileNotFoundError("Stage 1 validation records/manifest are required")
    return result


def verify(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    paths = _paths(root, config, run_name)
    required = [paths["train_records"], paths["models"], paths["validation_records"], paths["report_json"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 2 artifacts: {missing}")
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    checks = {
        "validation_only": report.get("split") == "validation",
        "test_locked": report.get("test_locked") is True,
        "has_all_variants": len(report.get("metrics", {})) == 7,
        "has_bootstrap": report.get("paired_dialogue_bootstrap", {}).get("samples", 0) > 0,
        "shared_temperature": report.get("temperature_fit", {}).get("shared_across_branches") is True,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 2 verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 2 selective LLM context routing")
    parser.add_argument(
        "stage",
        choices=("check", "generate-train", "fit", "evaluate", "summarize", "verify"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "check":
        check(args.config, args.run_name)
    elif args.stage == "generate-train":
        generate_train(
            args.config,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
            model_override=args.model_name,
            run_name=args.run_name,
        )
    elif args.stage == "fit":
        fit_models(args.config, args.run_name)
    elif args.stage == "evaluate":
        evaluate_validation(
            args.config,
            run_name=args.run_name,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
            model_override=args.model_name,
        )
    elif args.stage == "summarize":
        summarize(args.config, args.run_name, max_targets=args.max_targets)
    else:
        verify(args.config, args.run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
