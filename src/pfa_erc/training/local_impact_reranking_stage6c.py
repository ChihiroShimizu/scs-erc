"""Stage 6C: self-audited local-impact evidence reranking.

This validation-only mechanism experiment consumes the final S6C V2.3
annotations, but never consumes their self-audit scores or provenance as model
features.  It reuses Stage 6B's cached current-only and single-evidence Qwen
predictions.  No language model is trained or queried in this stage.

Any mapping learned from annotation fields to observed evidence usefulness is
fit leave-one-complete-dialogue-out.  MELD test is inaccessible by design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pfa_erc.training import hard_role_oracle_stage4 as stage4

VERSION = "1.9.0"
EXPERIMENT = "local_impact_reranking_stage6c"

UTILITY_ORDER = {
    "misleading": -2.0,
    "irrelevant": -1.0,
    "uncertain": 0.0,
    "weak": 1.0,
    "helpful": 2.0,
    "decisive": 3.0,
}
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

FINAL_BRANCHES = (
    "current_only",
    "full_history",
    "stage6b_local_general",
    "stage6b_local_predicted_role",
    "annotation_selected",
    "annotation_random_matched",
    "utility_ranked",
    "strength_blend",
    "learned_utility",
    "learned_full_impact",
    "learned_shuffled_impact",
    "learned_nearest_matched",
    "oracle_current_or_candidate",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key(dialogue_id: str, turn_id: int) -> str:
    return f"{dialogue_id}::{int(turn_id)}"


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "configs":
        return resolved.parent.parent
    return Path.cwd().resolve()


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get(
            "artifact_dir",
            "artifacts/local_impact_reranking_stage6c/meld_validation",
        ),
    )
    return {
        "base": base,
        "audit": base / "reports" / "annotation_audit.json",
        "oof": base / "scores" / "crossfit_predictions_validation.jsonl",
        "fold_models": base / "reports" / "crossfit_fold_models.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
        "verification": base / "reports" / "artifact_verification.json",
        "manifest": base / "run_manifest.json",
    }


def _annotation_object(record: dict[str, Any]) -> dict[str, Any]:
    """Accept only the annotation payload; audit scores are intentionally dropped."""
    if "annotation_version" in record:
        value = record
    elif isinstance(record.get("final_annotation"), dict):
        value = record["final_annotation"]
    elif isinstance(record.get("annotation"), dict):
        value = record["annotation"]
    else:
        raise ValueError("Record does not contain a complete S6C V2.3 annotation")
    return dict(value)


def _load_context(
    config_path: Path,
) -> tuple[
    Path,
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Path],
]:
    root = _project_root(config_path)
    config = _read_yaml(config_path)
    section = config.get("local_impact_reranking_stage6c", {})
    if section.get("dataset", "meld") != "meld":
        raise ValueError("Stage 6C accepts only MELD")
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 6C is validation-only; refusing non-validation data")

    labels = list(
        section.get(
            "labels",
            ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        )
    )
    source_paths = {
        "annotation": _resolve(root, section["annotation_file"]),
        "provenance": _resolve(root, section["provenance_file"]),
        "review_queue": _resolve(root, section["review_queue_file"]),
        "stage6b_raw": _resolve(root, section["stage6b_raw_score_file"]),
        "stage6b_oof": _resolve(root, section["stage6b_oof_score_file"]),
        "stage6b_summary": _resolve(root, section["stage6b_summary_file"]),
    }
    missing = [str(path) for path in source_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6C prerequisite(s): {missing}")
    forbidden = [str(path) for path in source_paths.values() if "test" in path.name.lower()]
    if forbidden:
        raise ValueError(f"Stage 6C refuses test-named inputs: {forbidden}")

    raw_annotations = _read_jsonl(source_paths["annotation"])
    annotations: dict[str, dict[str, Any]] = {}
    for record in raw_annotations:
        annotation = _annotation_object(record)
        key = _key(annotation["dialogue_id"], annotation["target_turn_id"])
        if key in annotations:
            raise ValueError(f"Duplicate annotation key: {key}")
        annotations[key] = annotation

    provenance = _read_jsonl(source_paths["provenance"])
    review_queue = _read_jsonl(source_paths["review_queue"])
    raw_scores = _read_jsonl(source_paths["stage6b_raw"])
    stage6b_oof = _read_jsonl(source_paths["stage6b_oof"])
    raw_by_key = {str(item["key"]): item for item in raw_scores}
    oof_by_key = {str(item["key"]): item for item in stage6b_oof}
    if len(raw_by_key) != len(raw_scores):
        raise ValueError("Duplicate key in Stage 6B raw scores")
    if len(oof_by_key) != len(stage6b_oof):
        raise ValueError("Duplicate key in Stage 6B OOF scores")
    if set(annotations) != set(raw_by_key) or set(annotations) != set(oof_by_key):
        raise ValueError(
            "V2.3 annotation keys must exactly match Stage 6B raw and OOF score keys"
        )
    merged_raw: list[dict[str, Any]] = []
    for key in sorted(raw_by_key, key=lambda value: (value.rsplit("::", 1)[0], int(value.rsplit("::", 1)[1]))):
        item = dict(raw_by_key[key])
        oof = oof_by_key[key]
        item["stage6b_local_general"] = oof["local_general"]
        item["stage6b_local_predicted_role"] = oof["local_predicted_role"]
        merged_raw.append(item)

    return (
        root,
        section,
        labels,
        merged_raw,
        stage6b_oof,
        annotations,
        provenance,
        review_queue,
        source_paths,
    )


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _words(value: Any) -> int:
    return len(str(value).strip().split())


def _candidate_pair_map(raw: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["source_turn_id"]): item for item in raw.get("pair_scores", [])}


def validate_annotation(
    annotation: dict[str, Any], raw: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    key = str(raw["key"])
    if set(annotation) != ANNOTATION_FIELDS:
        issues.append(
            f"{key}: annotation fields mismatch missing={sorted(ANNOTATION_FIELDS-set(annotation))} "
            f"extra={sorted(set(annotation)-ANNOTATION_FIELDS)}"
        )
        return issues
    target = int(raw["turn_id"])
    if annotation["annotation_version"] != "S6C-V2.3-local-impact":
        issues.append(f"{key}: wrong annotation_version")
    if _key(annotation["dialogue_id"], annotation["target_turn_id"]) != key:
        issues.append(f"{key}: annotation key mismatch")
    if int(annotation["target_turn_id"]) != target:
        issues.append(f"{key}: target_turn_id mismatch")
    if str(annotation["target_role"]) != str(raw["speaker"]):
        issues.append(f"{key}: target_role mismatch")
    expected_source_id = f"meld-hard-role:{annotation['dialogue_id']}:{target}"
    if str(annotation["source_annotation_id"]) != expected_source_id:
        issues.append(f"{key}: source_annotation_id mismatch")
    if annotation["context_need"] not in CONTEXT_NEEDS:
        issues.append(f"{key}: invalid context_need")
    if not _is_number(annotation["overall_confidence"]) or not 0 <= float(
        annotation["overall_confidence"]
    ) <= 1:
        issues.append(f"{key}: invalid overall_confidence")
    if _words(annotation["selection_reason"]) > 30:
        issues.append(f"{key}: selection_reason exceeds 30 words")

    candidates = annotation["candidate_assessments"]
    if not isinstance(candidates, list):
        issues.append(f"{key}: candidate_assessments is not a list")
        return issues
    expected_ids = list(range(max(0, target - 3), target))[::-1]
    actual_ids: list[int] = []
    pair_map = _candidate_pair_map(raw)
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            issues.append(f"{key}: candidate {index} is not an object")
            continue
        if set(candidate) != CANDIDATE_FIELDS:
            issues.append(f"{key}: candidate {index} fields mismatch")
            continue
        source = int(candidate["candidate_turn_id"])
        actual_ids.append(source)
        if source not in pair_map:
            issues.append(f"{key}: source {source} absent from Stage 6B causal pairs")
        else:
            expected_speaker = pair_map[source].get("candidate", {}).get("speaker")
            if expected_speaker is not None and str(candidate["candidate_speaker"]) != str(
                expected_speaker
            ):
                issues.append(f"{key}: source {source} speaker mismatch")
        if int(candidate["distance"]) != target - source:
            issues.append(f"{key}: source {source} distance mismatch")
        if candidate["role_relation"] not in ROLE_RELATIONS:
            issues.append(f"{key}: source {source} invalid role_relation")
        if candidate["prediction_utility"] not in UTILITY_ORDER:
            issues.append(f"{key}: source {source} invalid prediction_utility")
        if candidate["impact_direction"] not in IMPACT_DIRECTIONS:
            issues.append(f"{key}: source {source} invalid impact_direction")
        if candidate["transition_type"] not in TRANSITION_TYPES:
            issues.append(f"{key}: source {source} invalid transition_type")
        strength = candidate["impact_strength"]
        if strength not in {0, 1, 2, 3, None}:
            issues.append(f"{key}: source {source} invalid impact_strength")
        if candidate["evidence_basis"] not in EVIDENCE_BASES:
            issues.append(f"{key}: source {source} invalid evidence_basis")
        if not _is_number(candidate["confidence"]) or not 0 <= float(
            candidate["confidence"]
        ) <= 1:
            issues.append(f"{key}: source {source} invalid confidence")
        if _words(candidate["short_reason"]) > 30:
            issues.append(f"{key}: source {source} reason exceeds 30 words")
        if candidate["impact_direction"] == "none" and not (
            candidate["transition_type"] == "none" and strength == 0
        ):
            issues.append(f"{key}: source {source} none-impact dependency violation")
        if candidate["role_relation"] == "unrelated" and candidate[
            "prediction_utility"
        ] not in {"irrelevant", "misleading", "uncertain"}:
            issues.append(f"{key}: source {source} unrelated utility violation")
        if candidate["prediction_utility"] == "decisive" and float(
            candidate["confidence"]
        ) < 0.70:
            issues.append(f"{key}: source {source} decisive confidence below 0.70")

    if actual_ids != expected_ids:
        issues.append(f"{key}: candidate order/coverage {actual_ids} != {expected_ids}")
    selected = annotation["selected_evidence_turn_id"]
    if selected is not None:
        selected = int(selected)
        if selected not in actual_ids:
            issues.append(f"{key}: selected evidence is outside candidate set")
        else:
            chosen = next(
                item for item in candidates if int(item["candidate_turn_id"]) == selected
            )
            if chosen["role_relation"] == "unrelated" or chosen[
                "prediction_utility"
            ] in {"irrelevant", "misleading"}:
                issues.append(f"{key}: selected evidence is invalid")
    if annotation["context_need"] == "current_sufficient" and selected is not None:
        issues.append(f"{key}: current_sufficient selected evidence")
    if annotation["context_need"] in {"history_helpful", "history_required"}:
        if selected is None:
            issues.append(f"{key}: context-needed record lacks selected evidence")
        elif selected not in actual_ids:
            issues.append(f"{key}: context-needed selection is outside candidate set")
        elif next(
            item for item in candidates if int(item["candidate_turn_id"]) == selected
        )["prediction_utility"] not in {"helpful", "decisive"}:
            issues.append(f"{key}: context-needed selection is not helpful/decisive")
    if target == 0 and (candidates or selected is not None):
        issues.append(f"{key}: first turn contains history evidence")
    return issues


def check(config_path: Path) -> dict[str, Any]:
    (
        root,
        section,
        labels,
        raw,
        _,
        annotations,
        provenance,
        review_queue,
        source_paths,
    ) = _load_context(config_path)
    result = {
        "status": "ready",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "targets": len(annotations),
        "dialogues": len({item["dialogue_id"] for item in raw}),
        "provenance_records": len(provenance),
        "review_queue_records": len(review_queue),
        "labels": labels,
        "reuses_stage6b_cached_predictions": True,
        "requires_gpu": False,
        "annotation_file": str(source_paths["annotation"]),
        "project_root": str(root),
        "ridge_lambda": float(section.get("ridge_lambda", 1.0)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def audit(config_path: Path) -> dict[str, Any]:
    (
        root,
        section,
        _,
        raw,
        _,
        annotations,
        provenance,
        review_queue,
        source_paths,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    raw_by_key = {item["key"]: item for item in raw}
    issues = [
        issue
        for key, annotation in annotations.items()
        for issue in validate_annotation(annotation, raw_by_key[key])
    ]
    selected = [
        annotation["selected_evidence_turn_id"]
        for annotation in annotations.values()
        if annotation["selected_evidence_turn_id"] is not None
    ]
    candidates = [
        candidate
        for annotation in annotations.values()
        for candidate in annotation["candidate_assessments"]
    ]
    checks = {
        "exactly_100_targets": len(annotations) == 100,
        "exact_key_alignment": set(annotations) == set(raw_by_key),
        "provenance_coverage": len(provenance) == len(annotations),
        "review_queue_empty": len(review_queue) == 0,
        "schema_and_causal_window_valid": not issues,
        "validation_only": section.get("split") == "validation",
        "test_locked": True,
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "checks": checks,
        "targets": len(annotations),
        "dialogues": len({item["dialogue_id"] for item in raw}),
        "candidate_assessments": len(candidates),
        "selected_evidence_targets": len(selected),
        "context_need_distribution": dict(
            Counter(item["context_need"] for item in annotations.values())
        ),
        "prediction_utility_distribution": dict(
            Counter(item["prediction_utility"] for item in candidates)
        ),
        "impact_strength_distribution": {
            str(key): value
            for key, value in Counter(item["impact_strength"] for item in candidates).items()
        },
        "impact_direction_distribution": dict(
            Counter(item["impact_direction"] for item in candidates)
        ),
        "selected_distance_distribution": dict(
            Counter(
                int(annotation["target_turn_id"])
                - int(annotation["selected_evidence_turn_id"])
                for annotation in annotations.values()
                if annotation["selected_evidence_turn_id"] is not None
            )
        ),
        "structural_errors": issues,
        "source_hashes": {name: _sha256(path) for name, path in source_paths.items()},
        "model_input_invariant": (
            "Only final_annotation payloads are consumed; self-audit scores and "
            "provenance never enter any feature vector."
        ),
    }
    _write_json(paths["audit"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Stage 6C annotation audit failed")
    return report


def _one_hot(value: str, values: Sequence[str]) -> list[float]:
    return [1.0 if value == item else 0.0 for item in values]


def feature_names(feature_set: str) -> list[str]:
    names = [
        "utility_ordinal",
        "candidate_confidence",
        "inverse_distance",
        "annotator_selected",
    ]
    names.extend(f"context_need={item}" for item in CONTEXT_NEEDS)
    names.extend(f"role_relation={item}" for item in ROLE_RELATIONS)
    names.extend(f"evidence_basis={item}" for item in EVIDENCE_BASES)
    if feature_set == "full":
        names.append("impact_strength")
        names.extend(f"impact_direction={item}" for item in IMPACT_DIRECTIONS)
        names.extend(f"transition_type={item}" for item in TRANSITION_TYPES)
    elif feature_set != "utility":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return names


def candidate_features(
    annotation: dict[str, Any],
    candidate: dict[str, Any],
    feature_set: str,
    impact_override: dict[str, Any] | None = None,
) -> np.ndarray:
    selected = annotation["selected_evidence_turn_id"]
    values: list[float] = [
        UTILITY_ORDER[candidate["prediction_utility"]] / 3.0,
        float(candidate["confidence"]),
        1.0 / max(int(candidate["distance"]), 1),
        float(selected is not None and int(selected) == int(candidate["candidate_turn_id"])),
    ]
    values.extend(_one_hot(annotation["context_need"], CONTEXT_NEEDS))
    values.extend(_one_hot(candidate["role_relation"], ROLE_RELATIONS))
    values.extend(_one_hot(candidate["evidence_basis"], EVIDENCE_BASES))
    if feature_set == "full":
        impact = impact_override or candidate
        strength = impact.get("impact_strength")
        values.append(0.5 if strength is None else float(strength) / 3.0)
        values.extend(_one_hot(str(impact["impact_direction"]), IMPACT_DIRECTIONS))
        values.extend(_one_hot(str(impact["transition_type"]), TRANSITION_TYPES))
    elif feature_set != "utility":
        raise ValueError(f"Unknown feature set: {feature_set}")
    return np.asarray(values, dtype=np.float64)


def _gold_nll(payload: dict[str, Any], gold: str, labels: Sequence[str]) -> float:
    index = labels.index(gold)
    return -math.log(max(float(payload["probabilities"][index]), 1e-12))


def _candidate_payload(raw: dict[str, Any], source: int | None) -> dict[str, Any]:
    if source is None:
        return raw["current_only"]
    return raw["single_predictions"].get(str(source), raw["current_only"])


def _candidate_rows(
    records: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    labels: Sequence[str],
    feature_set: str,
    impact_map: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x_values: list[np.ndarray] = []
    y_values: list[float] = []
    for raw in records:
        annotation = annotations[raw["key"]]
        current_nll = _gold_nll(raw["current_only"], raw["gold"], labels)
        for candidate in annotation["candidate_assessments"]:
            source = int(candidate["candidate_turn_id"])
            payload = _candidate_payload(raw, source)
            if payload is raw["current_only"] and str(source) not in raw["single_predictions"]:
                continue
            override = impact_map.get((raw["key"], source)) if impact_map else None
            x_values.append(
                candidate_features(annotation, candidate, feature_set, override)
            )
            y_values.append(
                current_nll - _gold_nll(payload, raw["gold"], labels)
            )
    if not x_values:
        raise ValueError("No candidate rows available for cross-fitted regression")
    return np.stack(x_values), np.asarray(y_values, dtype=np.float64)


def fit_ridge(
    records: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    labels: Sequence[str],
    feature_set: str,
    ridge_lambda: float,
    impact_map: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    x_values, y_values = _candidate_rows(
        records, annotations, labels, feature_set, impact_map
    )
    mean = x_values.mean(axis=0)
    scale = x_values.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (x_values - mean) / scale
    design = np.column_stack([np.ones(len(normalized)), normalized])
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge_lambda)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(design.T @ design + penalty) @ design.T @ y_values
    predictions = design @ coefficients
    return {
        "feature_set": feature_set,
        "feature_names": feature_names(feature_set),
        "mean": mean,
        "scale": scale,
        "intercept": float(coefficients[0]),
        "coefficients": coefficients[1:],
        "training_pairs": len(y_values),
        "training_delta_nll_mean": float(y_values.mean()),
        "training_rmse": float(np.sqrt(np.mean((predictions - y_values) ** 2))),
    }


def predict_ridge(model: dict[str, Any], values: np.ndarray) -> float:
    normalized = (values - model["mean"]) / model["scale"]
    return float(model["intercept"] + normalized @ model["coefficients"])


def _impact_shuffle_map(
    annotations: dict[str, dict[str, Any]], seed: int
) -> dict[tuple[str, int], dict[str, Any]]:
    by_distance: dict[int, list[tuple[tuple[str, int], dict[str, Any]]]] = defaultdict(list)
    for key, annotation in annotations.items():
        for candidate in annotation["candidate_assessments"]:
            pair_key = (key, int(candidate["candidate_turn_id"]))
            by_distance[int(candidate["distance"])].append(
                (
                    pair_key,
                    {
                        "impact_strength": candidate["impact_strength"],
                        "impact_direction": candidate["impact_direction"],
                        "transition_type": candidate["transition_type"],
                    },
                )
            )
    output: dict[tuple[str, int], dict[str, Any]] = {}
    for distance, rows in sorted(by_distance.items()):
        ordered = sorted(rows, key=lambda item: item[0])
        if len(ordered) == 1:
            output[ordered[0][0]] = ordered[0][1]
            continue
        rng = random.Random(seed + distance)
        shift = 1 + rng.randrange(len(ordered) - 1)
        values = [item[1] for item in ordered]
        for index, (pair_key, _) in enumerate(ordered):
            output[pair_key] = values[(index + shift) % len(values)]
    return output


def _context_requested(annotation: dict[str, Any]) -> bool:
    return annotation["context_need"] in {"history_helpful", "history_required"}


def annotation_selected_source(annotation: dict[str, Any]) -> int | None:
    selected = annotation["selected_evidence_turn_id"]
    return int(selected) if _context_requested(annotation) and selected is not None else None


def utility_ranked_source(annotation: dict[str, Any]) -> int | None:
    if not _context_requested(annotation):
        return None
    eligible = [
        candidate
        for candidate in annotation["candidate_assessments"]
        if candidate["prediction_utility"] in {"helpful", "decisive"}
        and candidate["role_relation"] != "unrelated"
    ]
    if not eligible:
        return None
    chosen = max(
        eligible,
        key=lambda item: (
            UTILITY_ORDER[item["prediction_utility"]],
            float(item["confidence"]),
            -int(item["distance"]),
        ),
    )
    return int(chosen["candidate_turn_id"])


def _learned_source(
    raw: dict[str, Any],
    annotation: dict[str, Any],
    model: dict[str, Any],
    impact_map: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> tuple[int | None, float | None]:
    if not _context_requested(annotation):
        return None, None
    scored: list[tuple[float, int, int]] = []
    for candidate in annotation["candidate_assessments"]:
        source = int(candidate["candidate_turn_id"])
        override = impact_map.get((raw["key"], source)) if impact_map else None
        values = candidate_features(
            annotation, candidate, model["feature_set"], override
        )
        score = predict_ridge(model, values)
        scored.append((score, -int(candidate["distance"]), source))
    if not scored:
        return None, None
    score, _, source = max(scored)
    return (source, score) if score > 0.0 else (None, score)


def _deterministic_source(
    key: str, candidates: Sequence[dict[str, Any]], seed: int
) -> int | None:
    if not candidates:
        return None
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    index = int(digest[:16], 16) % len(candidates)
    return int(candidates[index]["candidate_turn_id"])


def _probability_payload(probabilities: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    values = np.maximum(values, 1e-12)
    values /= values.sum()
    index = int(np.argmax(values))
    return {
        "scores": [float(math.log(value)) for value in values],
        "probabilities": [float(value) for value in values],
        "prediction": labels[index],
        "confidence": float(values[index]),
    }


def geometric_blend(
    current: dict[str, Any],
    context: dict[str, Any],
    alpha: float,
    labels: Sequence[str],
) -> dict[str, Any]:
    alpha = min(max(float(alpha), 0.0), 1.0)
    current_p = np.maximum(np.asarray(current["probabilities"], dtype=np.float64), 1e-12)
    context_p = np.maximum(np.asarray(context["probabilities"], dtype=np.float64), 1e-12)
    logits = (1.0 - alpha) * np.log(current_p) + alpha * np.log(context_p)
    logits -= logits.max()
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum()
    return _probability_payload(probabilities, labels)


def _selected_strength_fraction(
    annotation: dict[str, Any], null_fraction: float
) -> float:
    source = annotation_selected_source(annotation)
    if source is None:
        return 0.0
    candidate = next(
        item
        for item in annotation["candidate_assessments"]
        if int(item["candidate_turn_id"]) == source
    )
    strength = candidate["impact_strength"]
    return float(null_fraction) if strength is None else float(strength) / 3.0


def choose_alpha_scale(
    training: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    labels: Sequence[str],
    grid: Sequence[float],
    null_fraction: float,
) -> float:
    candidates: list[tuple[float, float]] = []
    for scale in sorted({float(value) for value in grid}):
        losses: list[float] = []
        for raw in training:
            annotation = annotations[raw["key"]]
            source = annotation_selected_source(annotation)
            if source is None:
                payload = raw["current_only"]
            else:
                alpha = min(1.0, scale * _selected_strength_fraction(annotation, null_fraction))
                payload = geometric_blend(
                    raw["current_only"], _candidate_payload(raw, source), alpha, labels
                )
            losses.append(_gold_nll(payload, raw["gold"], labels))
        candidates.append((float(np.mean(losses)), scale))
    return min(candidates)[1]


def _oracle_payload(raw: dict[str, Any], annotation: dict[str, Any], labels: Sequence[str]) -> tuple[dict[str, Any], int | None]:
    choices: list[tuple[float, int | None, dict[str, Any]]] = [
        (_gold_nll(raw["current_only"], raw["gold"], labels), None, raw["current_only"])
    ]
    for candidate in annotation["candidate_assessments"]:
        source = int(candidate["candidate_turn_id"])
        payload = _candidate_payload(raw, source)
        choices.append((_gold_nll(payload, raw["gold"], labels), source, payload))
    _, source, payload = min(choices, key=lambda item: item[0])
    return payload, source


def crossfit_predictions(
    raw_records: Sequence[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dialogue_ids = sorted({str(item["dialogue_id"]) for item in raw_records})
    if len(dialogue_ids) < 2:
        raise ValueError("At least two complete dialogues are required for cross-fitting")
    ridge_lambda = float(section.get("ridge_lambda", 1.0))
    alpha_grid = section.get("strength_alpha_scale_grid", [0.25, 0.5, 0.75, 1.0])
    null_fraction = float(section.get("null_strength_fraction", 0.5))
    random_seed = int(section.get("matched_random_seed", 2053))
    shuffle_map = _impact_shuffle_map(
        annotations, int(section.get("impact_shuffle_seed", 2051))
    )
    output: list[dict[str, Any]] = []
    fold_models: list[dict[str, Any]] = []
    for heldout in dialogue_ids:
        training = [item for item in raw_records if item["dialogue_id"] != heldout]
        testing = [item for item in raw_records if item["dialogue_id"] == heldout]
        utility_model = fit_ridge(
            training, annotations, labels, "utility", ridge_lambda
        )
        full_model = fit_ridge(
            training, annotations, labels, "full", ridge_lambda
        )
        shuffled_model = fit_ridge(
            training,
            annotations,
            labels,
            "full",
            ridge_lambda,
            impact_map=shuffle_map,
        )
        alpha_scale = choose_alpha_scale(
            training, annotations, labels, alpha_grid, null_fraction
        )
        fold_models.append(
            {
                "heldout_dialogue": heldout,
                "training_dialogues": len(dialogue_ids) - 1,
                "alpha_scale": alpha_scale,
                "utility_model": _serializable_model(utility_model),
                "full_model": _serializable_model(full_model),
                "shuffled_model": _serializable_model(shuffled_model),
            }
        )

        for raw in testing:
            annotation = annotations[raw["key"]]
            annotation_source = annotation_selected_source(annotation)
            utility_source = utility_ranked_source(annotation)
            learned_utility_source, utility_delta = _learned_source(
                raw, annotation, utility_model
            )
            learned_full_source, full_delta = _learned_source(
                raw, annotation, full_model
            )
            shuffled_source, shuffled_delta = _learned_source(
                raw, annotation, shuffled_model, shuffle_map
            )
            candidates = annotation["candidate_assessments"]
            annotation_random = (
                _deterministic_source(raw["key"] + ":annotation", candidates, random_seed)
                if annotation_source is not None
                else None
            )
            nearest_source = (
                max(int(item["candidate_turn_id"]) for item in candidates)
                if learned_full_source is not None and candidates
                else None
            )
            strength_fraction = _selected_strength_fraction(annotation, null_fraction)
            strength_payload = (
                geometric_blend(
                    raw["current_only"],
                    _candidate_payload(raw, annotation_source),
                    min(1.0, alpha_scale * strength_fraction),
                    labels,
                )
                if annotation_source is not None
                else raw["current_only"]
            )
            oracle_payload, oracle_source = _oracle_payload(raw, annotation, labels)

            row = {
                "key": raw["key"],
                "dialogue_id": raw["dialogue_id"],
                "turn_id": int(raw["turn_id"]),
                "speaker": raw["speaker"],
                "gold": raw["gold"],
                "context_need": annotation["context_need"],
                "current_only": raw["current_only"],
                "full_history": raw["full_history"],
                "stage6b_local_general": raw["stage6b_local_general"],
                "stage6b_local_predicted_role": raw[
                    "stage6b_local_predicted_role"
                ],
                "annotation_selected": _candidate_payload(raw, annotation_source),
                "annotation_random_matched": _candidate_payload(raw, annotation_random),
                "utility_ranked": _candidate_payload(raw, utility_source),
                "strength_blend": strength_payload,
                "learned_utility": _candidate_payload(raw, learned_utility_source),
                "learned_full_impact": _candidate_payload(raw, learned_full_source),
                "learned_shuffled_impact": _candidate_payload(raw, shuffled_source),
                "learned_nearest_matched": _candidate_payload(raw, nearest_source),
                "oracle_current_or_candidate": oracle_payload,
                "crossfit_heldout_dialogue": heldout,
                "strength_alpha_scale": alpha_scale,
                "selected_source": {
                    "annotation_selected": annotation_source,
                    "annotation_random_matched": annotation_random,
                    "utility_ranked": utility_source,
                    "learned_utility": learned_utility_source,
                    "learned_full_impact": learned_full_source,
                    "learned_shuffled_impact": shuffled_source,
                    "learned_nearest_matched": nearest_source,
                    "oracle_current_or_candidate": oracle_source,
                },
                "predicted_delta_nll": {
                    "learned_utility": utility_delta,
                    "learned_full_impact": full_delta,
                    "learned_shuffled_impact": shuffled_delta,
                },
                "annotation_selected_source": annotation_source,
                "candidate_count": len(candidates),
            }
            output.append(row)
    output.sort(key=lambda item: (item["dialogue_id"], int(item["turn_id"])))
    return output, fold_models


def _serializable_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_set": model["feature_set"],
        "feature_names": model["feature_names"],
        "intercept": model["intercept"],
        "coefficients": [float(value) for value in model["coefficients"]],
        "training_pairs": model["training_pairs"],
        "training_delta_nll_mean": model["training_delta_nll_mean"],
        "training_rmse": model["training_rmse"],
    }


def _bootstrap(
    records: Sequence[dict[str, Any]],
    proposed: str,
    control: str,
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    return stage4.paired_dialogue_bootstrap(
        records, proposed, control, labels, samples, seed
    )


def _branch_diagnostics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    branches = (
        "annotation_selected",
        "annotation_random_matched",
        "utility_ranked",
        "learned_utility",
        "learned_full_impact",
        "learned_shuffled_impact",
        "learned_nearest_matched",
    )
    output: dict[str, Any] = {}
    for branch in branches:
        sources = [item["selected_source"][branch] for item in records]
        selected = [source for source in sources if source is not None]
        gaps = [
            int(item["turn_id"]) - int(source)
            for item, source in zip(records, sources)
            if source is not None
        ]
        agreement = [
            source == item["annotation_selected_source"]
            for item, source in zip(records, sources)
            if source is not None or item["annotation_selected_source"] is not None
        ]
        output[branch] = {
            "context_use_rate": len(selected) / max(len(records), 1),
            "selected_targets": len(selected),
            "mean_selected_gap": float(np.mean(gaps)) if gaps else None,
            "agreement_with_annotation_selection": (
                float(np.mean(agreement)) if agreement else None
            ),
        }
    return output


def _mean_fold_coefficients(
    fold_models: Sequence[dict[str, Any]], model_name: str
) -> dict[str, float]:
    if not fold_models:
        return {}
    names = fold_models[0][model_name]["feature_names"]
    matrix = np.asarray(
        [item[model_name]["coefficients"] for item in fold_models],
        dtype=np.float64,
    )
    return {
        name: float(value) for name, value in zip(names, matrix.mean(axis=0))
    }


def _flip_counts(
    records: Sequence[dict[str, Any]], proposed: str, control: str
) -> dict[str, int]:
    good = sum(
        item[proposed]["prediction"] == item["gold"]
        and item[control]["prediction"] != item["gold"]
        for item in records
    )
    bad = sum(
        item[proposed]["prediction"] != item["gold"]
        and item[control]["prediction"] == item["gold"]
        for item in records
    )
    return {"good": good, "bad": bad, "net": good - bad}


def _supported(comparison: dict[str, Any], probability: float) -> bool:
    metrics = comparison["bootstrap"]["metrics"]
    return (
        metrics["macro_f1"]["probability_positive"] >= probability
        or metrics["nll"]["probability_positive"] >= probability
    )


def _markdown(summary: dict[str, Any]) -> str:
    names = {
        "current_only": "6C-0 current only",
        "full_history": "6C-1 full history",
        "stage6b_local_general": "6B local general control",
        "stage6b_local_predicted_role": "6B predicted-role control",
        "annotation_selected": "6C-2 V2.3 selected evidence",
        "annotation_random_matched": "6C-3 matched random evidence",
        "utility_ranked": "6C-4 utility-ranked evidence",
        "strength_blend": "6C-5 strength-weighted blend",
        "learned_utility": "6C-6 cross-fitted utility model",
        "learned_full_impact": "6C-7 full local-impact model",
        "learned_shuffled_impact": "6C-8 shuffled-impact control",
        "learned_nearest_matched": "6C-9 nearest-turn matched control",
        "oracle_current_or_candidate": "current/candidate outcome Oracle",
    }
    metrics = summary["metrics"]
    lines = [
        "# Stage 6C — Self-Audited Local-Impact Evidence Reranking",
        "",
        "> Complete-dialogue MELD validation-only mechanism test. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in FINAL_BRANCHES:
        item = metrics[branch]
        lines.append(
            f"| {names[branch]} | {item['weighted_f1']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['nll']:.4f} | "
            f"{item['brier']:.4f} | {item['ece']:.4f} |"
        )
    lines.extend(["", "## Pre-registered paired mechanism contrasts", ""])
    for name, comparison in summary["comparisons"].items():
        macro = comparison["bootstrap"]["metrics"]["macro_f1"]
        nll = comparison["bootstrap"]["metrics"]["nll"]
        flips = comparison["flips"]
        lines.append(
            f"- **{name}**: macro {macro['point_advantage']:+.4f} "
            f"(95% CI {macro['ci95'][0]:+.4f} to {macro['ci95'][1]:+.4f}, "
            f"P>0={macro['probability_positive']:.3f}); NLL "
            f"{nll['point_advantage']:+.4f} (P>0={nll['probability_positive']:.3f}); "
            f"good/bad/net={flips['good']}/{flips['bad']}/{flips['net']}."
        )
    lines.extend(["", "## Routing diagnostics", ""])
    for branch, values in summary["routing_diagnostics"].items():
        lines.append(
            f"- {branch}: use={values['context_use_rate']:.1%}, "
            f"mean gap={values['mean_selected_gap']}, "
            f"agreement with V2.3={values['agreement_with_annotation_selection']}"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"Status: **{summary['decision']['status']}**",
        ]
    )
    for name, passed in summary["decision"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "The V2.3 annotations are self-audited LLM operational labels, not human gold emotion-state changes. Audit scores and provenance are excluded from all model features. All learned mappings are leave-one-complete-dialogue-out.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path) -> dict[str, Any]:
    (
        root,
        section,
        labels,
        raw,
        _,
        annotations,
        _,
        review_queue,
        source_paths,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["audit"].exists():
        raise FileNotFoundError("Run Stage 6C audit before analyze")
    audit_report = _read_json(paths["audit"])
    if audit_report.get("status") != "passed":
        raise RuntimeError("Stage 6C annotation audit did not pass")

    oof, fold_models = crossfit_predictions(raw, annotations, labels, section)
    _write_jsonl(paths["oof"], oof)
    _write_json(paths["fold_models"], {"folds": fold_models})
    metrics = {
        branch: stage4.metric_bundle(oof, branch, labels) for branch in FINAL_BRANCHES
    }
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2055))
    specs = {
        "annotated_evidence_vs_random_matched": (
            "annotation_selected",
            "annotation_random_matched",
        ),
        "annotated_evidence_vs_stage6b_general": (
            "annotation_selected",
            "stage6b_local_general",
        ),
        "strength_blend_vs_annotated_evidence": (
            "strength_blend",
            "annotation_selected",
        ),
        "learned_utility_vs_stage6b_general": (
            "learned_utility",
            "stage6b_local_general",
        ),
        "full_impact_vs_utility_model": (
            "learned_full_impact",
            "learned_utility",
        ),
        "correct_impact_vs_shuffled_impact": (
            "learned_full_impact",
            "learned_shuffled_impact",
        ),
        "full_impact_vs_nearest_matched": (
            "learned_full_impact",
            "learned_nearest_matched",
        ),
        "full_impact_vs_current": ("learned_full_impact", "current_only"),
    }
    comparisons: dict[str, Any] = {}
    for offset, (name, (proposed, control)) in enumerate(specs.items()):
        comparisons[name] = {
            "proposed": proposed,
            "control": control,
            "flips": _flip_counts(oof, proposed, control),
            "bootstrap": _bootstrap(
                oof, proposed, control, labels, samples, seed + offset
            ),
        }

    probability = float(section.get("minimum_bootstrap_probability", 0.90))
    tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    checks = {
        "self_audited_annotation_gate": audit_report.get("status") == "passed"
        and not review_queue,
        "complete_dialogue_crossfit": len(oof) == len(annotations)
        and all(item["crossfit_heldout_dialogue"] == item["dialogue_id"] for item in oof),
        "full_impact_preserves_current_weighted_f1": metrics[
            "learned_full_impact"
        ]["weighted_f1"]
        >= metrics["current_only"]["weighted_f1"] - tolerance,
        "annotated_evidence_beats_random_matched": _supported(
            comparisons["annotated_evidence_vs_random_matched"], probability
        ),
        "impact_fields_add_over_utility": _supported(
            comparisons["full_impact_vs_utility_model"], probability
        ),
        "correct_impact_beats_shuffled_impact": _supported(
            comparisons["correct_impact_vs_shuffled_impact"], probability
        ),
        "full_impact_beats_nearest_matched": _supported(
            comparisons["full_impact_vs_nearest_matched"], probability
        ),
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "advance_to_train_scale_local_impact_reasoner"
            if eligible
            else "stop_and_diagnose_local_impact_labels"
        ),
        "eligible": eligible,
        "checks": checks,
        "selection_basis": (
            "fixed S6C V2.3 self-audited annotations; leave-one-complete-dialogue-out "
            "ridge mapping and strength interpolation; validation only"
        ),
        "test_locked": True,
        "next_action": (
            "Expand the same causal annotation protocol on MELD train and train a deployable local-impact reasoner."
            if eligible
            else "Keep MELD test locked; inspect whether selection, strength, or direction fields failed their matched controls."
        ),
    }
    routing = _branch_diagnostics(oof)
    summary = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "oracle_diagnostic": True,
        "annotation_status": "self_audited_llm_operational_labels",
        "targets": len(oof),
        "dialogues": len({item["dialogue_id"] for item in oof}),
        "labels": labels,
        "metrics": metrics,
        "comparisons": comparisons,
        "routing_diagnostics": routing,
        "fold_alpha_scales": dict(
            Counter(str(item["alpha_scale"]) for item in fold_models)
        ),
        "mean_utility_coefficients": _mean_fold_coefficients(
            fold_models, "utility_model"
        ),
        "mean_full_impact_coefficients": _mean_fold_coefficients(
            fold_models, "full_model"
        ),
        "decision": decision,
    }
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["decision"], decision)
    manifest = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "source_hashes": {name: _sha256(path) for name, path in source_paths.items()},
        "oof_sha256": _sha256(paths["oof"]),
        "annotation_fields_used": [
            "context_need",
            "selected_evidence_turn_id",
            "role_relation",
            "prediction_utility",
            "impact_direction",
            "transition_type",
            "impact_strength",
            "evidence_basis",
            "confidence",
        ],
        "annotation_fields_explicitly_excluded": [
            "self_audit_score",
            "self_audit_action",
            "provenance",
            "gold_emotion",
            "future_turns",
        ],
        "causal_invariants": [
            "only the latest three turns strictly before the target are candidates",
            "at most one evidence turn is read",
            "all learned mappings exclude the held-out complete dialogue",
            "self-audit scores and provenance never enter model features",
            "MELD test is never opened",
        ],
    }
    _write_json(paths["manifest"], manifest)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, section, _, raw, _, annotations, _, review_queue, _ = _load_context(
        config_path
    )
    paths = _paths(root, section)
    required = [
        paths["audit"],
        paths["oof"],
        paths["fold_models"],
        paths["summary_json"],
        paths["summary_md"],
        paths["decision"],
        paths["manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6C artifact(s): {missing}")
    oof = _read_jsonl(paths["oof"])
    summary = _read_json(paths["summary_json"])
    finite = all(
        math.isfinite(float(summary["metrics"][branch][metric]))
        for branch in FINAL_BRANCHES
        for metric in ("weighted_f1", "macro_f1", "nll", "brier", "ece")
    )
    no_future = all(
        source is None or int(source) < int(item["turn_id"])
        for item in oof
        for source in item["selected_source"].values()
    )
    max_distance = all(
        source is None or 0 < int(item["turn_id"]) - int(source) <= 3
        for item in oof
        for source in item["selected_source"].values()
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "review_queue_empty": len(review_queue) == 0,
        "complete_oof_coverage": len(oof) == len(annotations) == len(raw),
        "all_variants_present": all(
            branch in summary.get("metrics", {}) for branch in FINAL_BRANCHES
        ),
        "finite_metrics": finite,
        "no_current_or_future_evidence": no_future,
        "latest_three_window_respected": max_distance,
        "complete_dialogue_crossfit": all(
            item["crossfit_heldout_dialogue"] == item["dialogue_id"] for item in oof
        ),
        "paired_bootstrap_present": all(
            item.get("bootstrap") is not None
            for item in summary.get("comparisons", {}).values()
        ),
        "audit_scores_absent_from_predictions": all(
            "audit" not in key.lower() and "provenance" not in key.lower()
            for item in oof
            for key in item
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scientific_decision": summary.get("decision", {}).get("status"),
        "test_locked": True,
    }
    _write_json(paths["verification"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 6C artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only self-audited local-impact reranking"
    )
    parser.add_argument(
        "stage", choices=("check", "audit", "analyze", "verify", "all-validation")
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "check":
        check(arguments.config)
    elif arguments.stage == "audit":
        audit(arguments.config)
    elif arguments.stage == "analyze":
        analyze(arguments.config)
    elif arguments.stage == "verify":
        verify(arguments.config)
    else:
        check(arguments.config)
        audit(arguments.config)
        analyze(arguments.config)
        verify(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
