"""Stage 6B: recency-bounded role-conditioned evidence search.

This validation-only mechanism test implements a local causal search rather
than a role-memory lookup.  When the current-only prediction is ambiguous, the
model inspects only the last K turns and scans them from nearest to farthest.
It stops at the first turn that passes a target-relevance threshold and the
requested role condition.  One evidence turn is used at most.

Every learned branch shares the same cross-fitted K, entropy gate, and
relevance threshold.  Therefore comparisons change the candidate mask rather
than silently changing how often context is used.  MELD test is inaccessible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pfa_erc.training import hard_role_oracle_stage4 as stage4
from pfa_erc.training import role_conditioned_retrieval_stage6a as stage6a

VERSION = "1.8.0"
LOCAL_METHODS = ("general", "gold_role", "predicted_role", "wrong_role")
FINAL_BRANCHES = (
    "current_only",
    "full_history",
    "last_turn_routed",
    "recent_window_routed",
    "local_general",
    "local_gold_role",
    "local_predicted_role",
    "local_wrong_role",
    "global_gold_role_6a",
    "global_predicted_role_6a",
    "oracle_evidence",
    "verified_local_evidence",
    "gold_context_local_role",
    "oracle_current_or_local",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
                raise TypeError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get(
            "artifact_dir",
            "artifacts/recency_bounded_role_search_stage6b/meld_validation",
        ),
    )
    return {
        "base": base,
        "audit": base / "reports" / "local_distance_audit.json",
        "raw_scores": base / "scores" / "local_predictions_validation.jsonl",
        "oof_scores": base / "scores" / "crossfit_predictions_validation.jsonl",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
        "verification": base / "reports" / "artifact_verification.json",
        "manifest": base / "run_manifest.json",
    }


def _key(dialogue_id: str, turn_id: int) -> str:
    return f"{dialogue_id}::{turn_id}"


def _load_context(
    config_path: Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    root, config, labels, dialogues, verified, quality = stage4._load_context(config_path)
    section = config.get("recency_bounded_role_search_stage6b", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 6B is validation-only; refusing a non-validation split")
    if section.get("dataset", "meld") != "meld":
        raise ValueError("Stage 6B accepts only the annotated MELD validation pilot")

    stage4_scores_path = _resolve(root, section["stage4_score_file"])
    selector_path = _resolve(root, section["stage6a_selector_score_file"])
    global_oof_path = _resolve(root, section["stage6a_oof_score_file"])
    missing = [
        str(path)
        for path in (stage4_scores_path, selector_path, global_oof_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Complete Stage 4 and Stage 6A before Stage 6B; missing: " + str(missing)
        )
    stage4_scores = _read_jsonl(stage4_scores_path)
    selectors = _read_jsonl(selector_path)
    global_oof = _read_jsonl(global_oof_path)
    expected = set(verified)
    if {item["key"] for item in stage4_scores} != expected:
        raise ValueError("Stage 4 score keys do not match verified targets")
    if {item["key"] for item in global_oof} != expected:
        raise ValueError("Stage 6A OOF score keys do not match verified targets")
    if {item["target_key"] for item in selectors} - expected:
        raise ValueError("Stage 6A selector scores contain unknown target keys")
    return (
        root,
        config,
        section,
        labels,
        dialogues,
        verified,
        quality,
        stage4_scores,
        selectors,
        global_oof,
    )


def _specs(
    dialogues: Sequence[dict[str, Any]],
    verified: dict[str, dict[str, Any]],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    return stage6a.build_target_specs(
        dialogues,
        verified,
        int(section.get("selector_history_limit", 12)),
        int(section.get("role_prefix_turns", 3)),
    )


def check(config_path: Path) -> dict[str, Any]:
    (
        root,
        _,
        section,
        labels,
        dialogues,
        verified,
        quality,
        _,
        selectors,
        _,
    ) = _load_context(config_path)
    result = {
        "status": "ready",
        "experiment": "recency_bounded_role_search_stage6b",
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "complete_dialogues": len(dialogues),
        "targets": len(verified),
        "selector_pairs_reused": len(selectors),
        "window_grid": section.get("window_grid"),
        "labels": labels,
        "annotation_quality_status": quality.get("status"),
        "model_name": section.get("model_name"),
        "project_root": str(root),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def evidence_distance_audit(
    specs: Sequence[dict[str, Any]], window_grid: Sequence[int]
) -> dict[str, Any]:
    gaps: list[int] = []
    target_gaps: dict[str, list[int]] = {}
    role_gaps: list[int] = []
    for spec in specs:
        values = [
            int(spec["turn_id"]) - int(source)
            for source in spec["oracle_evidence_ids"]
            if int(source) < int(spec["turn_id"])
        ]
        if values:
            target_gaps[spec["key"]] = values
            gaps.extend(values)
        role_gaps.extend(
            int(spec["turn_id"]) - int(pair["source_turn_id"])
            for pair in spec["pairs"]
            if pair["gold_role_match"]
        )

    def bucket(gap: int) -> str:
        if gap == 1:
            return "1"
        if gap == 2:
            return "2"
        if gap <= 3:
            return "3"
        if gap <= 5:
            return "4-5"
        return "6+"

    evidence_buckets = Counter(bucket(value) for value in gaps)
    role_buckets = Counter(bucket(value) for value in role_gaps)
    coverage: dict[str, Any] = {}
    for window in sorted({int(value) for value in window_grid}):
        covered_turns = sum(value <= window for value in gaps)
        covered_targets = sum(any(value <= window for value in values) for values in target_gaps.values())
        coverage[str(window)] = {
            "evidence_turn_coverage": covered_turns / max(len(gaps), 1),
            "evidence_target_coverage": covered_targets / max(len(target_gaps), 1),
            "covered_evidence_turns": covered_turns,
            "covered_targets": covered_targets,
        }
    return {
        "verified_evidence_turns": len(gaps),
        "targets_with_verified_evidence": len(target_gaps),
        "gold_role_event_pairs": len(role_gaps),
        "evidence_gap_distribution": dict(evidence_buckets),
        "role_event_gap_distribution": dict(role_buckets),
        "coverage_by_window": coverage,
    }


def audit(config_path: Path) -> dict[str, Any]:
    (
        root,
        _,
        section,
        _,
        dialogues,
        verified,
        quality,
        _,
        selectors,
        _,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    window_grid = sorted({int(value) for value in section.get("window_grid", [1, 2, 3, 5])})
    specs = _specs(dialogues, verified, section)
    distance = evidence_distance_audit(specs, window_grid)
    max_window = max(window_grid)
    local_pairs = [
        item
        for item in selectors
        if int(item["target_turn_id"]) - int(item["source_turn_id"]) <= max_window
    ]
    report = {
        "status": "passed",
        "experiment": "recency_bounded_role_search_stage6b",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "complete_dialogues": len(dialogues),
        "targets": len(specs),
        "window_grid": window_grid,
        "maximum_local_window": max_window,
        "reused_selector_pairs": len(selectors),
        "local_selector_pairs": len(local_pairs),
        "annotation_quality_gate": quality.get("eligible_for_oracle") is True,
        "distance_audit": distance,
        "interpretation": (
            "Role labels constrain candidates only inside the recent local window; "
            "they are not accumulated into a long-term emotion state."
        ),
    }
    _write_json(paths["audit"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def select_backward_candidate(
    pair_scores: Sequence[dict[str, Any]],
    method: str,
    window: int,
    relevance_threshold: float,
    predicted_role_threshold: float = 0.5,
) -> int | None:
    """Return the nearest qualifying source; a farther high score cannot win."""
    if window <= 0:
        raise ValueError("window must be positive")
    ordered = sorted(
        pair_scores, key=lambda item: int(item["source_turn_id"]), reverse=True
    )
    for item in ordered:
        gap = int(item["target_turn_id"]) - int(item["source_turn_id"])
        if gap <= 0 or gap > window:
            continue
        if float(item["relevance_probability"]) < relevance_threshold:
            continue
        if method == "general":
            role_ok = True
        elif method == "gold_role":
            role_ok = bool(item["gold_role_match"])
        elif method == "predicted_role":
            role_ok = bool(item["role_predictable"]) and float(
                item["role_probability"]
            ) >= predicted_role_threshold
        elif method == "wrong_role":
            role_ok = bool(item["wrong_role_match"])
        else:
            raise ValueError(f"Unknown local method: {method}")
        if role_ok:
            return int(item["source_turn_id"])
    return None


def _make_scorer(section: dict[str, Any], labels: Sequence[str]) -> Any:
    from pfa_erc.training.llm_context_utility_stage1 import CausalLabelScorer

    return CausalLabelScorer(section, labels)


def _by_target(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[str(item["target_key"])].append(item)
    return grouped


def _candidate(turn_id: int, utterance: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": int(turn_id),
        "speaker": str(utterance["speaker"]),
        "text": str(utterance["text"]),
    }


def score_predictions(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    (
        root,
        _,
        section,
        labels,
        dialogues,
        verified,
        _,
        stage4_scores,
        selectors,
        global_oof,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    if overwrite and paths["raw_scores"].exists():
        paths["raw_scores"].unlink()
    stage4_by_key = {item["key"]: item for item in stage4_scores}
    selector_by_target = _by_target(selectors)
    global_by_key = {item["key"]: item for item in global_oof}
    existing = _read_jsonl(paths["raw_scores"])
    completed = {item["key"] for item in existing}
    scorer = _make_scorer(section, labels)
    temperature = float(section.get("shared_temperature", 1.0))
    window_grid = sorted({int(value) for value in section.get("window_grid", [1, 2, 3, 5])})
    max_window = max(window_grid)
    preamble = str(section.get("prompt_preamble", "You are a careful emotion classifier."))

    for dialogue_number, dialogue in enumerate(dialogues, 1):
        dialogue_id = dialogue["dialogue_id"]
        utterances = dialogue["utterances"]
        output: list[dict[str, Any]] = []
        for target_turn, utterance in enumerate(utterances):
            target_key = _key(dialogue_id, target_turn)
            if target_key in completed:
                continue
            stage4_record = stage4_by_key[target_key]
            global_record = global_by_key[target_key]
            target = _candidate(target_turn, utterance)
            pair_rows = [
                item
                for item in selector_by_target.get(target_key, [])
                if 0 < target_turn - int(item["source_turn_id"]) <= max_window
            ]
            pair_rows.sort(key=lambda item: int(item["source_turn_id"]))
            prompts: list[str] = []
            layout: list[tuple[str, str]] = []
            single_predictions: dict[str, dict[str, Any]] = {}
            window_predictions: dict[str, dict[str, Any]] = {}

            for pair in pair_rows:
                source = int(pair["source_turn_id"])
                evidence = [_candidate(source, utterances[source])]
                prompts.append(stage4.build_prompt(target, evidence, [], labels, preamble))
                layout.append(("single", str(source)))

            for window in window_grid:
                start = max(0, target_turn - window)
                evidence = [
                    _candidate(index, utterances[index])
                    for index in range(start, target_turn)
                ]
                if not evidence:
                    window_predictions[str(window)] = stage4_record["current_only"]
                    continue
                prompts.append(stage4.build_prompt(target, evidence, [], labels, preamble))
                layout.append(("window", str(window)))

            if prompts:
                scores = scorer.score_prompts(prompts)
                for (kind, identifier), values in zip(layout, scores):
                    payload = stage4._probability_payload(values, labels, temperature)
                    if kind == "single":
                        single_predictions[identifier] = payload
                    else:
                        window_predictions[identifier] = payload

            output.append(
                {
                    "key": target_key,
                    "dialogue_id": dialogue_id,
                    "turn_id": target_turn,
                    "speaker": str(utterance["speaker"]),
                    "gold": str(utterance["emotion"]),
                    "context_level": stage6a.context_level(
                        verified[target_key]["annotation"]
                    ),
                    "oracle_evidence_ids": stage4.oracle_evidence_ids(
                        verified[target_key]["annotation"]
                    ),
                    "history_count": target_turn,
                    "pair_scores": pair_rows,
                    "single_predictions": single_predictions,
                    "window_predictions": window_predictions,
                    "current_only": stage4_record["current_only"],
                    "full_history": stage4_record["full_history"],
                    "oracle_evidence": stage4_record["oracle_evidence"],
                    "global_gold_role_6a": global_record["gold_role_routed"],
                    "global_predicted_role_6a": global_record[
                        "predicted_role_routed"
                    ],
                }
            )
        if output:
            _append_jsonl(paths["raw_scores"], output)
            completed.update(item["key"] for item in output)
        print(
            f"[Stage 6B prediction] dialogue {dialogue_number}/{len(dialogues)} "
            f"{dialogue_id}: {len(output)} target(s)",
            flush=True,
        )

    final = _read_jsonl(paths["raw_scores"])
    if {item["key"] for item in final} != set(verified):
        raise ValueError("Stage 6B prediction coverage does not match verified targets")
    result = {
        "status": "complete",
        "targets": len(final),
        "window_grid": window_grid,
        "score_file": str(paths["raw_scores"]),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _entropy_use(record: dict[str, Any], threshold: float) -> bool:
    return stage6a.normalized_entropy(record["current_only"]["probabilities"]) >= threshold


def _single_payload(record: dict[str, Any], source: int | None) -> dict[str, Any]:
    if source is None:
        return record["current_only"]
    return record["single_predictions"].get(str(source), record["current_only"])


def materialize_local(
    record: dict[str, Any],
    method: str,
    parameters: dict[str, Any],
    predicted_role_threshold: float,
) -> tuple[dict[str, Any], bool, int | None]:
    source = select_backward_candidate(
        record["pair_scores"],
        method,
        int(parameters["window"]),
        float(parameters["relevance_threshold"]),
        predicted_role_threshold,
    )
    used = _entropy_use(record, float(parameters["entropy_threshold"])) and source is not None
    return (
        _single_payload(record, source) if used else record["current_only"],
        used,
        source if used else None,
    )


def _mean_nll(
    records: Sequence[dict[str, Any]], branch: str, labels: Sequence[str]
) -> float:
    label_to_id = {label: index for index, label in enumerate(labels)}
    return float(
        np.mean(
            [
                -math.log(
                    max(
                        float(item[branch]["probabilities"][label_to_id[item["gold"]]]),
                        1e-12,
                    )
                )
                for item in records
            ]
        )
    )


def _materialize_general_for_selection(
    records: Sequence[dict[str, Any]],
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in records:
        payload, used, source = materialize_local(item, "general", parameters, 0.5)
        row = dict(item)
        row["_candidate"] = payload
        row["_used"] = used
        row["_gap"] = int(item["turn_id"]) - source if source is not None else 0
        output.append(row)
    return output


def choose_shared_parameters(
    training_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> dict[str, Any]:
    if not training_records:
        raise ValueError("Cross-fitting requires non-empty training dialogues")
    windows = sorted({int(value) for value in section.get("window_grid", [1, 2, 3, 5])})
    entropy_grid = sorted(
        {float(value) for value in section.get("entropy_threshold_grid", [0.0, 0.55, 0.7, 0.82, 0.92])}
    )
    relevance_grid = sorted(
        {float(value) for value in section.get("relevance_threshold_grid", [0.0, 0.25, 0.5, 0.75])}
    )
    context_cost = float(section.get("context_use_penalty", 0.002))
    distance_cost = float(section.get("distance_penalty", 0.001))
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for window in windows:
        for entropy_threshold in entropy_grid:
            for relevance_threshold in relevance_grid:
                parameters = {
                    "window": window,
                    "entropy_threshold": entropy_threshold,
                    "relevance_threshold": relevance_threshold,
                }
                rows = _materialize_general_for_selection(training_records, parameters)
                nll = _mean_nll(rows, "_candidate", labels)
                use_rate = float(np.mean([item["_used"] for item in rows]))
                mean_gap = float(np.mean([item["_gap"] for item in rows]))
                macro = stage4.metric_bundle(rows, "_candidate", labels)["macro_f1"]
                objective = (
                    nll
                    + context_cost * use_rate
                    + distance_cost * mean_gap / max(window, 1)
                )
                result = parameters | {
                    "objective": objective,
                    "nll": nll,
                    "macro_f1": macro,
                    "context_use_rate": use_rate,
                    "mean_gap": mean_gap,
                }
                tie_break = (
                    objective,
                    nll,
                    -macro,
                    float(window),
                    -relevance_threshold,
                    -entropy_threshold,
                )
                candidates.append((tie_break, result))
    return min(candidates, key=lambda item: item[0])[1]


def _nearest_verified_source(record: dict[str, Any], window: int) -> int | None:
    eligible = [
        int(source)
        for source in record["oracle_evidence_ids"]
        if 0 < int(record["turn_id"]) - int(source) <= window
        and str(source) in record["single_predictions"]
    ]
    return max(eligible) if eligible else None


def crossfit_predictions(
    raw_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    dialogue_ids = sorted({item["dialogue_id"] for item in raw_records})
    if len(dialogue_ids) < 2:
        raise ValueError("At least two complete dialogues are required")
    predicted_role_threshold = float(section.get("predicted_role_threshold", 0.5))
    output: list[dict[str, Any]] = []
    for heldout in dialogue_ids:
        training = [item for item in raw_records if item["dialogue_id"] != heldout]
        testing = [item for item in raw_records if item["dialogue_id"] == heldout]
        parameters = choose_shared_parameters(training, labels, section)
        for item in testing:
            row = {
                key: value
                for key, value in item.items()
                if key not in {"single_predictions", "window_predictions", "pair_scores"}
            }
            row["crossfit_heldout_dialogue"] = heldout
            row["shared_parameters"] = parameters
            row["context_used"] = {}
            row["selected_source"] = {}
            row["selected_gap"] = {}
            candidate_sources: dict[str, int | None] = {}
            for method in LOCAL_METHODS:
                payload, used, source = materialize_local(
                    item, method, parameters, predicted_role_threshold
                )
                branch = f"local_{method}"
                row[branch] = payload
                row["context_used"][branch] = used
                row["selected_source"][branch] = source
                row["selected_gap"][branch] = (
                    int(item["turn_id"]) - source if source is not None else None
                )
                candidate_sources[method] = select_backward_candidate(
                    item["pair_scores"],
                    method,
                    int(parameters["window"]),
                    float(parameters["relevance_threshold"]),
                    predicted_role_threshold,
                )

            use_context = _entropy_use(item, float(parameters["entropy_threshold"]))
            last_payload = item["window_predictions"].get("1", item["current_only"])
            recent_payload = item["window_predictions"].get(
                str(parameters["window"]), item["current_only"]
            )
            has_history = int(item["history_count"]) > 0
            row["last_turn_routed"] = (
                last_payload if use_context and has_history else item["current_only"]
            )
            row["recent_window_routed"] = (
                recent_payload if use_context and has_history else item["current_only"]
            )

            verified_source = _nearest_verified_source(item, int(parameters["window"]))
            verified_used = use_context and verified_source is not None
            row["verified_local_evidence"] = (
                _single_payload(item, verified_source)
                if verified_used
                else item["current_only"]
            )
            row["context_used"]["verified_local_evidence"] = verified_used
            row["selected_source"]["verified_local_evidence"] = (
                verified_source if verified_used else None
            )

            gold_needed = item["context_level"] in {
                "history_helpful",
                "history_required",
            }
            gold_source = candidate_sources["gold_role"]
            row["gold_context_local_role"] = (
                _single_payload(item, gold_source)
                if gold_needed and gold_source is not None
                else item["current_only"]
            )

            label_index = labels.index(item["gold"])
            current_nll = -math.log(
                max(float(item["current_only"]["probabilities"][label_index]), 1e-12)
            )
            local_oracle_payload = _single_payload(item, verified_source)
            local_nll = -math.log(
                max(float(local_oracle_payload["probabilities"][label_index]), 1e-12)
            )
            row["oracle_current_or_local"] = (
                local_oracle_payload if verified_source is not None and local_nll < current_nll else item["current_only"]
            )

            row["gold_general_common_candidate"] = (
                candidate_sources["gold_role"] is not None
                and candidate_sources["general"] is not None
            )
            row["predicted_general_common_candidate"] = (
                candidate_sources["predicted_role"] is not None
                and candidate_sources["general"] is not None
            )
            row["gold_wrong_common_candidate"] = (
                candidate_sources["gold_role"] is not None
                and candidate_sources["wrong_role"] is not None
            )
            row["verified_local_available"] = verified_source is not None
            output.append(row)
    return sorted(output, key=lambda item: (item["dialogue_id"], item["turn_id"]))


def _safe_ap(gold: Sequence[bool], scores: Sequence[float]) -> float | None:
    return stage6a._safe_ap(gold, scores)


def _safe_f1(gold: Sequence[bool], scores: Sequence[float]) -> float | None:
    return stage6a._safe_f1(gold, scores)


def _retrieval_diagnostics(
    records: Sequence[dict[str, Any]], selectors: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    role_rows = [item for item in selectors if item["role_predictable"]]
    diagnostics: dict[str, Any] = {
        "predicted_role_average_precision": _safe_ap(
            [item["gold_role_match"] for item in role_rows],
            [item["role_probability"] for item in role_rows],
        ),
        "predicted_role_f1_at_050": _safe_f1(
            [item["gold_role_match"] for item in role_rows],
            [item["role_probability"] for item in role_rows],
        ),
        "relevance_average_precision": _safe_ap(
            [item["gold_evidence_match"] for item in selectors],
            [item["relevance_probability"] for item in selectors],
        ),
        "relevance_f1_at_050": _safe_f1(
            [item["gold_evidence_match"] for item in selectors],
            [item["relevance_probability"] for item in selectors],
        ),
    }
    for branch in (
        "local_general",
        "local_gold_role",
        "local_predicted_role",
        "local_wrong_role",
    ):
        selected = [item["selected_source"][branch] for item in records]
        used = [item["context_used"][branch] for item in records]
        valid_sources = [source for source in selected if source is not None]
        hits = sum(
            source is not None and int(source) in set(item["oracle_evidence_ids"])
            for item, source in zip(records, selected)
        )
        total_gold = sum(len(set(item["oracle_evidence_ids"])) for item in records)
        gaps = [
            item["selected_gap"][branch]
            for item in records
            if item["selected_gap"][branch] is not None
        ]
        diagnostics[branch] = {
            "context_use_rate": float(np.mean(used)),
            "selected_targets": len(valid_sources),
            "mean_selected_gap": float(np.mean(gaps)) if gaps else None,
            "evidence_precision": hits / max(len(valid_sources), 1),
            "evidence_recall_all_verified": hits / max(total_gold, 1),
        }
    return diagnostics


def _subset_metrics(
    records: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, dict[str, Any]]:
    subsets = {
        "all": list(records),
        "context_needed": [
            item
            for item in records
            if item["context_level"] in {"history_helpful", "history_required"}
        ],
        "current_sufficient": [
            item for item in records if item["context_level"] == "current_sufficient"
        ],
        "gold_general_common": [
            item for item in records if item["gold_general_common_candidate"]
        ],
        "predicted_general_common": [
            item for item in records if item["predicted_general_common_candidate"]
        ],
        "gold_wrong_common": [
            item for item in records if item["gold_wrong_common_candidate"]
        ],
        "verified_local_available": [
            item for item in records if item["verified_local_available"]
        ],
    }
    return {
        name: {
            "count": len(rows),
            "dialogues": len({item["dialogue_id"] for item in rows}),
            "metrics": (
                {
                    branch: stage4.metric_bundle(rows, branch, labels)
                    for branch in FINAL_BRANCHES
                }
                if rows
                else {}
            ),
        }
        for name, rows in subsets.items()
    }


def _bootstrap(
    records: Sequence[dict[str, Any]],
    proposed: str,
    control: str,
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not records:
        return None
    return stage4.paired_dialogue_bootstrap(
        records, proposed, control, labels, samples, seed
    )


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["subsets"]["all"]["metrics"]
    names = {
        "current_only": "6B-0 current only",
        "full_history": "6B-1 full history",
        "last_turn_routed": "6B-2 routed last turn",
        "recent_window_routed": "6B-3 routed recent window",
        "local_general": "6B-4 local backward search",
        "local_gold_role": "6B-5 local gold-role search",
        "local_predicted_role": "6B-6 local predicted-role search",
        "local_wrong_role": "6B-7 local wrong-role search",
        "global_gold_role_6a": "6A global gold-role control",
        "oracle_evidence": "verified evidence Oracle",
        "verified_local_evidence": "verified local evidence Oracle",
        "gold_context_local_role": "gold context + local role Oracle",
        "oracle_current_or_local": "current/local outcome Oracle",
    }
    lines = [
        "# Stage 6B — Recency-Bounded Role-Conditioned Evidence Search",
        "",
        "> Complete-dialogue MELD validation-only mechanism test. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch, label in names.items():
        item = metrics[branch]
        lines.append(
            f"| {label} | {item['weighted_f1']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['nll']:.4f} | {item['brier']:.4f} | {item['ece']:.4f} |"
        )
    lines.extend(["", "## Evidence-distance audit", ""])
    distance = summary["distance_audit"]
    lines.append(
        f"Verified evidence turns/targets: **{distance['verified_evidence_turns']} / "
        f"{distance['targets_with_verified_evidence']}**."
    )
    for window, values in distance["coverage_by_window"].items():
        lines.append(
            f"- last {window}: turn coverage={values['evidence_turn_coverage']:.1%}, "
            f"target coverage={values['evidence_target_coverage']:.1%}"
        )
    lines.extend(["", "## Paired mechanism contrasts", ""])
    for name, item in summary["comparisons"].items():
        bootstrap = item["bootstrap"]
        if bootstrap is None:
            lines.append(f"- **{name}**: not estimable.")
            continue
        macro = bootstrap["metrics"]["macro_f1"]
        nll = bootstrap["metrics"]["nll"]
        lines.append(
            f"- **{name}** ({item['targets']} targets): macro {macro['point_advantage']:+.4f} "
            f"(95% CI {macro['ci95'][0]:+.4f} to {macro['ci95'][1]:+.4f}, "
            f"P>0={macro['probability_positive']:.3f}); NLL {nll['point_advantage']:+.4f} "
            f"(P>0={nll['probability_positive']:.3f})."
        )
    lines.extend(["", "## Retrieval diagnostics", ""])
    diagnostics = summary["diagnostics"]
    lines.append(
        f"- Predicted-role AP/F1@0.5: **{diagnostics['predicted_role_average_precision']} / "
        f"{diagnostics['predicted_role_f1_at_050']}**"
    )
    lines.append(
        f"- Relevance AP/F1@0.5: **{diagnostics['relevance_average_precision']} / "
        f"{diagnostics['relevance_f1_at_050']}**"
    )
    for branch in ("local_general", "local_gold_role", "local_predicted_role", "local_wrong_role"):
        item = diagnostics[branch]
        lines.append(
            f"- {branch}: use={item['context_use_rate']:.1%}, gap={item['mean_selected_gap']}, "
            f"evidence P/R={item['evidence_precision']:.3f}/"
            f"{item['evidence_recall_all_verified']:.3f}"
        )
    lines.extend(["", "## Frozen decision", "", f"Status: **{summary['decision']['status']}**"])
    for name, passed in summary["decision"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "All learned local branches share one dialogue-cross-fitted window, entropy gate, and relevance threshold. Gold-role and gold-context branches are Oracles, not deployable claims.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path) -> dict[str, Any]:
    (
        root,
        _,
        section,
        labels,
        dialogues,
        verified,
        quality,
        _,
        selectors,
        _,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["raw_scores"].exists() or not paths["audit"].exists():
        raise FileNotFoundError("Run audit and score-predictions before analyze")
    raw = _read_jsonl(paths["raw_scores"])
    if {item["key"] for item in raw} != set(verified):
        raise ValueError("Stage 6B raw coverage mismatch")
    oof = crossfit_predictions(raw, labels, section)
    _write_jsonl(paths["oof_scores"], oof)
    audit_report = _read_json(paths["audit"])
    subsets = _subset_metrics(oof, labels)
    max_local_window = max(
        int(value) for value in section.get("window_grid", [1, 2, 3, 5])
    )
    local_selectors = [
        item
        for item in selectors
        if 0
        < int(item["target_turn_id"]) - int(item["source_turn_id"])
        <= max_local_window
    ]
    diagnostics = _retrieval_diagnostics(oof, local_selectors)
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2047))
    comparison_specs = {
        "local_gold_vs_global_gold": (
            list(oof),
            "local_gold_role",
            "global_gold_role_6a",
        ),
        "local_general_vs_recent_window": (
            list(oof),
            "local_general",
            "recent_window_routed",
        ),
        "local_gold_vs_general_matched": (
            [item for item in oof if item["gold_general_common_candidate"]],
            "local_gold_role",
            "local_general",
        ),
        "local_predicted_vs_general_matched": (
            [item for item in oof if item["predicted_general_common_candidate"]],
            "local_predicted_role",
            "local_general",
        ),
        "local_gold_vs_wrong_matched": (
            [item for item in oof if item["gold_wrong_common_candidate"]],
            "local_gold_role",
            "local_wrong_role",
        ),
        "verified_local_vs_local_gold": (
            [item for item in oof if item["verified_local_available"]],
            "verified_local_evidence",
            "local_gold_role",
        ),
        "gold_context_local_vs_current": (
            list(oof),
            "gold_context_local_role",
            "current_only",
        ),
    }
    comparisons: dict[str, Any] = {}
    for offset, (name, (rows, proposed, control)) in enumerate(comparison_specs.items()):
        comparisons[name] = {
            "proposed": proposed,
            "control": control,
            "targets": len(rows),
            "bootstrap": _bootstrap(
                rows, proposed, control, labels, samples, seed + offset
            ),
        }

    probability = float(section.get("minimum_bootstrap_probability", 0.90))
    w_tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    nll_tolerance = float(section.get("predicted_gold_nll_tolerance", 0.05))
    local_coverage_threshold = float(
        section.get("minimum_evidence_target_coverage_k5", 0.70)
    )
    all_metrics = subsets["all"]["metrics"]

    def supported(name: str) -> bool:
        bootstrap = comparisons[name]["bootstrap"]
        if bootstrap is None:
            return False
        return (
            bootstrap["metrics"]["macro_f1"]["probability_positive"] >= probability
            or bootstrap["metrics"]["nll"]["probability_positive"] >= probability
        )

    distance = audit_report["distance_audit"]
    k5 = distance["coverage_by_window"].get("5", {})
    role_ap = diagnostics["predicted_role_average_precision"]
    checks = {
        "annotation_quality_gate": quality.get("eligible_for_oracle") is True,
        "complete_crossfit_coverage": len(oof) == len(verified),
        "recent_five_cover_verified_targets": float(
            k5.get("evidence_target_coverage", 0.0)
        )
        >= local_coverage_threshold,
        "local_gold_preserves_current_weighted_f1": all_metrics["local_gold_role"][
            "weighted_f1"
        ]
        >= all_metrics["current_only"]["weighted_f1"] - w_tolerance,
        "local_search_adds_over_global_role_search": supported(
            "local_gold_vs_global_gold"
        ),
        "gold_role_adds_over_general_local_search": supported(
            "local_gold_vs_general_matched"
        ),
        "predicted_role_average_precision": role_ap is not None
        and role_ap >= float(section.get("minimum_predicted_role_ap", 0.55)),
        "predicted_role_adds_over_general_local_search": supported(
            "local_predicted_vs_general_matched"
        ),
        "predicted_local_within_gold_nll_tolerance": all_metrics[
            "local_predicted_role"
        ]["nll"]
        <= all_metrics["local_gold_role"]["nll"] + nll_tolerance,
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "advance_to_train_scale_local_evidence_reasoner"
            if eligible
            else "stop_and_diagnose_local_evidence_search"
        ),
        "eligible": eligible,
        "checks": checks,
        "selection_basis": (
            "one shared leave-one-complete-dialogue-out parameter set learned "
            "from general local search; every role branch reuses it"
        ),
        "next_action": (
            "Expand train-only local evidence labels and train the deployable reasoner."
            if eligible
            else "Keep MELD test locked; inspect evidence-distance coverage and matched local contrasts."
        ),
        "test_locked": True,
    }
    summary = {
        "experiment": "recency_bounded_role_search_stage6b",
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "oracle_diagnostic": True,
        "targets": len(oof),
        "dialogues": len(dialogues),
        "labels": labels,
        "distance_audit": distance,
        "subsets": subsets,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
        "shared_parameter_counts": {
            f"K={key[0]},entropy={key[1]:.2f},relevance={key[2]:.2f}": count
            for key, count in Counter(
                (
                    item["shared_parameters"]["window"],
                    item["shared_parameters"]["entropy_threshold"],
                    item["shared_parameters"]["relevance_threshold"],
                )
                for item in oof
            ).items()
        },
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["decision"], decision)
    manifest = {
        "experiment": summary["experiment"],
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "annotation_sha256": _sha256(_resolve(root, section["annotation_file"])),
        "stage6a_selector_sha256": _sha256(
            _resolve(root, section["stage6a_selector_score_file"])
        ),
        "stage6a_oof_sha256": _sha256(
            _resolve(root, section["stage6a_oof_score_file"])
        ),
        "raw_predictions_sha256": _sha256(paths["raw_scores"]),
        "oof_predictions_sha256": _sha256(paths["oof_scores"]),
        "causal_invariants": [
            "only turns strictly earlier than the target are selectable",
            "only the last cross-fitted K turns form the local candidate window",
            "nearer qualifying evidence always beats farther qualifying evidence",
            "all learned role branches share K, entropy gate, and relevance threshold",
            "held-out dialogues are excluded from parameter selection",
            "MELD test is never opened",
        ],
    }
    _write_json(paths["manifest"], manifest)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, _, section, _, _, verified, _, _, _, _ = _load_context(config_path)
    paths = _paths(root, section)
    required = [
        paths["audit"],
        paths["raw_scores"],
        paths["oof_scores"],
        paths["summary_json"],
        paths["summary_md"],
        paths["decision"],
        paths["manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6B artifact(s): {missing}")
    summary = _read_json(paths["summary_json"])
    raw = _read_jsonl(paths["raw_scores"])
    oof = _read_jsonl(paths["oof_scores"])
    no_future = all(
        int(pair["source_turn_id"]) < int(item["turn_id"])
        for item in raw
        for pair in item["pair_scores"]
    )
    local_window_respected = all(
        gap is None or int(gap) <= int(item["shared_parameters"]["window"])
        for item in oof
        for gap in item["selected_gap"].values()
    )
    finite = all(
        math.isfinite(float(summary["subsets"]["all"]["metrics"][branch][metric]))
        for branch in FINAL_BRANCHES
        for metric in ("weighted_f1", "macro_f1", "nll", "brier", "ece")
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "complete_raw_coverage": len(raw) == len(verified),
        "complete_oof_coverage": len(oof) == len(verified),
        "all_final_variants": all(
            branch in summary.get("subsets", {}).get("all", {}).get("metrics", {})
            for branch in FINAL_BRANCHES
        ),
        "no_future_candidates": no_future,
        "local_window_respected": local_window_respected,
        "crossfit_heldout_dialogue_marked": all(
            item["crossfit_heldout_dialogue"] == item["dialogue_id"] for item in oof
        ),
        "finite_metrics": finite,
        "paired_bootstrap_present_or_not_estimable": all(
            item.get("bootstrap") is not None or int(item.get("targets", 0)) == 0
            for item in summary.get("comparisons", {}).values()
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
        raise RuntimeError("Stage 6B artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only recency-bounded role evidence search"
    )
    parser.add_argument(
        "stage",
        choices=(
            "check",
            "audit",
            "score-predictions",
            "analyze",
            "verify",
            "all-validation",
        ),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "check":
        check(arguments.config)
    elif arguments.stage == "audit":
        audit(arguments.config)
    elif arguments.stage == "score-predictions":
        score_predictions(arguments.config, overwrite=arguments.overwrite)
    elif arguments.stage == "analyze":
        analyze(arguments.config)
    elif arguments.stage == "verify":
        verify(arguments.config)
    else:
        check(arguments.config)
        audit(arguments.config)
        score_predictions(arguments.config, overwrite=arguments.overwrite)
        analyze(arguments.config)
        verify(arguments.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
