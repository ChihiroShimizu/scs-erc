"""Stage 6A: validation-only role-conditioned minimal evidence retrieval.

The experiment tests a narrower hypothesis than long-term role-state tracking:
when a target utterance is ambiguous, historical events explicitly routed to
the current speaker should form a better evidence pool than indiscriminate
history.  V2.2 annotations provide only hard role relevance and causal evidence;
they never provide an emotion direction or target label.

MELD test data are inaccessible by construction.  All parameter choices are
leave-one-complete-dialogue-out and the gold-dependent Oracles are excluded from
selection.
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
from sklearn.metrics import average_precision_score, f1_score

VERSION = "1.7.0"
METHODS = ("general", "gold_role", "predicted_role", "wrong_role")
BASE_BRANCHES = ("current_only", "full_history", "oracle_evidence")
FINAL_BRANCHES = (
    "current_only",
    "full_history",
    "general_always",
    "general_routed",
    "gold_role_always",
    "gold_role_routed",
    "predicted_role_always",
    "predicted_role_routed",
    "wrong_role_always",
    "wrong_role_routed",
    "oracle_evidence",
    "oracle_context_role",
    "oracle_current_or_evidence",
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


def _key(dialogue_id: str, turn_id: int) -> str:
    return f"{dialogue_id}::{turn_id}"


def hard_role_targets(annotation: dict[str, Any]) -> set[str]:
    """Return roles with both a direct interaction and a grounded event write.

    This is an operational role-relevance label, not a claim that the role's
    internal emotion actually changed.
    """
    return {
        str(item["role"])
        for item in annotation.get("role_assessments", [])
        if item.get("interaction_target") == "yes"
        and item.get("event_memory_write") == "yes"
    }


def context_level(annotation: dict[str, Any]) -> str:
    return str(
        annotation.get("context_utility", {})
        .get("speaker_emotion", {})
        .get("level", "ambiguous")
    )


def build_relevance_prompt(candidate: dict[str, Any], target: dict[str, Any]) -> str:
    return (
        "You are selecting minimal causal evidence for emotion recognition.\n"
        "Judge whether the earlier turn is necessary or materially useful for "
        "interpreting the target speaker's expressed emotion. Mere topic or name "
        "overlap is not sufficient.\n\n"
        f"Earlier turn [{candidate['turn_id']}]: {candidate['speaker']}: "
        f"{candidate['text']}\n"
        f"Target turn [{target['turn_id']}]: {target['speaker']}: {target['text']}\n\n"
        "Is the earlier turn useful evidence? Answer exactly yes or no.\nAnswer:"
    )


def build_role_prompt(
    causal_source_prefix: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    role: str,
) -> str:
    """Build a source-time-only role prompt; the future target text is absent."""
    prefix = "\n".join(
        f"[{item['turn_id']}] {item['speaker']}: {item['text']}"
        for item in causal_source_prefix
    )
    return (
        "You are detecting a grounded role-relevant dialogue event. Use only the "
        "conversation prefix below. Do not guess an emotion.\n\n"
        f"Conversation through event [{candidate['turn_id']}]:\n{prefix}\n\n"
        f"Candidate role: {role}\n"
        "Does the final event directly address, reply to, act upon, or deliver "
        "material information to this role that could be useful when the role "
        "speaks later? Answer exactly yes or no.\nAnswer:"
    )


def _yes_probability(scores: Sequence[float]) -> float:
    values = np.asarray(scores, dtype=np.float64)
    values -= values.max()
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    return float(probabilities[0])


def normalized_entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    values = values / values.sum()
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-12))))
    return entropy / math.log(len(values))


def route_payload(
    record: dict[str, Any], branch: str, threshold: float
) -> tuple[dict[str, Any], bool]:
    use_context = (
        normalized_entropy(record["current_only"]["probabilities"]) >= threshold
        and len(record["selected_ids"].get(branch, [])) > 0
    )
    return (record[branch] if use_context else record["current_only"], use_context)


def _candidate_dict(turn_id: int, utterance: dict[str, Any]) -> dict[str, Any]:
    return {
        "turn_id": int(turn_id),
        "speaker": str(utterance["speaker"]),
        "text": str(utterance["text"]),
    }


def build_target_specs(
    dialogues: Sequence[dict[str, Any]],
    verified: dict[str, dict[str, Any]],
    max_history_turns: int,
    role_prefix_turns: int,
) -> list[dict[str, Any]]:
    """Construct causal retrieval pairs from complete annotated dialogues."""
    specs: list[dict[str, Any]] = []
    for dialogue in dialogues:
        dialogue_id = dialogue["dialogue_id"]
        utterances = dialogue["utterances"]
        for target_turn, utterance in enumerate(utterances):
            target_key = _key(dialogue_id, target_turn)
            target_record = verified[target_key]
            target_annotation = target_record["annotation"]
            target = {
                "turn_id": target_turn,
                "speaker": str(utterance["speaker"]),
                "text": str(utterance["text"]),
            }
            first = max(0, target_turn - max_history_turns) if max_history_turns > 0 else 0
            oracle_ids = stage4.oracle_evidence_ids(target_annotation)
            pairs: list[dict[str, Any]] = []
            for source_turn in range(first, target_turn):
                candidate = _candidate_dict(source_turn, utterances[source_turn])
                source_annotation = verified[_key(dialogue_id, source_turn)]["annotation"]
                intended_roles = hard_role_targets(source_annotation)
                roles_seen_at_source = {
                    str(item["speaker"]) for item in utterances[: source_turn + 1]
                }
                role_predictable = (
                    target["speaker"] in roles_seen_at_source
                    and candidate["speaker"] != target["speaker"]
                )
                prefix_start = max(0, source_turn - max(role_prefix_turns - 1, 0))
                causal_prefix = [
                    _candidate_dict(index, utterances[index])
                    for index in range(prefix_start, source_turn + 1)
                ]
                pairs.append(
                    {
                        "pair_id": f"{target_key}::source={source_turn}",
                        "target_key": target_key,
                        "dialogue_id": dialogue_id,
                        "target_turn_id": target_turn,
                        "source_turn_id": source_turn,
                        "target_speaker": target["speaker"],
                        "candidate": candidate,
                        "gold_role_match": target["speaker"] in intended_roles,
                        "wrong_role_match": (
                            target["speaker"] not in intended_roles
                            and bool(intended_roles)
                        ),
                        "gold_evidence_match": source_turn in oracle_ids,
                        "role_predictable": role_predictable,
                        "relevance_prompt": build_relevance_prompt(candidate, target),
                        "role_prompt": (
                            build_role_prompt(causal_prefix, candidate, target["speaker"])
                            if role_predictable
                            else None
                        ),
                    }
                )
            specs.append(
                {
                    "key": target_key,
                    "dialogue_id": dialogue_id,
                    "turn_id": target_turn,
                    "speaker": target["speaker"],
                    "text": target["text"],
                    "gold": str(utterance["emotion"]),
                    "context_level": context_level(target_annotation),
                    "oracle_evidence_ids": oracle_ids,
                    "pairs": pairs,
                }
            )
    return specs


def rank_candidate_ids(
    pair_scores: Sequence[dict[str, Any]], method: str, k: int
) -> list[int]:
    if k <= 0:
        return []
    if method == "general":
        candidates = list(pair_scores)
        score = lambda item: float(item["relevance_probability"])
    elif method == "gold_role":
        candidates = [item for item in pair_scores if item["gold_role_match"]]
        score = lambda item: float(item["relevance_probability"])
    elif method == "predicted_role":
        candidates = [item for item in pair_scores if item["role_predictable"]]
        score = lambda item: float(item["relevance_probability"]) * float(
            item["role_probability"]
        )
    elif method == "wrong_role":
        candidates = [item for item in pair_scores if item["wrong_role_match"]]
        score = lambda item: float(item["relevance_probability"])
    else:
        raise ValueError(f"Unknown retrieval method: {method}")
    ranked = sorted(
        candidates,
        key=lambda item: (score(item), int(item["source_turn_id"])),
        reverse=True,
    )
    return sorted(int(item["source_turn_id"]) for item in ranked[:k])


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get(
            "artifact_dir",
            "artifacts/role_conditioned_retrieval_stage6a/meld_validation",
        ),
    )
    return {
        "base": base,
        "audit": base / "reports" / "candidate_topology_audit.json",
        "selector_scores": base / "scores" / "selector_pair_scores_validation.jsonl",
        "raw_scores": base / "scores" / "retrieval_predictions_validation.jsonl",
        "oof_scores": base / "scores" / "crossfit_predictions_validation.jsonl",
        "manifest": base / "run_manifest.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
        "verification": base / "reports" / "artifact_verification.json",
    }


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
]:
    root, config, labels, dialogues, verified, quality = stage4._load_context(config_path)
    section = config.get("role_conditioned_retrieval_stage6a", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 6A is validation-only; refusing a non-validation split")
    if section.get("dataset", "meld") != "meld":
        raise ValueError("Stage 6A currently accepts only the annotated MELD validation pilot")
    stage4_score_file = _resolve(root, section["stage4_score_file"])
    if not stage4_score_file.exists():
        raise FileNotFoundError(
            f"Missing Stage 4 scores: {stage4_score_file}. Complete Stage 4 first."
        )
    records = _read_jsonl(stage4_score_file)
    if {record["key"] for record in records} != set(verified):
        raise ValueError("Stage 4 score keys do not match the verified annotation trajectory")
    return root, config, section, labels, dialogues, verified, quality


def _specs(
    dialogues: Sequence[dict[str, Any]],
    verified: dict[str, dict[str, Any]],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    return build_target_specs(
        dialogues,
        verified,
        int(section.get("max_history_turns", 12)),
        int(section.get("role_prefix_turns", 3)),
    )


def check(config_path: Path) -> dict[str, Any]:
    root, _, section, labels, dialogues, verified, quality = _load_context(config_path)
    result = {
        "status": "ready",
        "experiment": "role_conditioned_retrieval_stage6a",
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "complete_dialogues": len(dialogues),
        "targets": len(verified),
        "labels": labels,
        "annotation_quality_status": quality.get("status"),
        "model_name": section.get("model_name"),
        "project_root": str(root),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def audit(config_path: Path) -> dict[str, Any]:
    root, _, section, _, dialogues, verified, quality = _load_context(config_path)
    paths = _paths(root, section)
    specs = _specs(dialogues, verified, section)
    context_needed = [
        item
        for item in specs
        if item["context_level"] in {"history_helpful", "history_required"}
    ]
    all_pairs = [pair for item in specs for pair in item["pairs"]]
    role_targets = [
        item for item in specs if any(pair["gold_role_match"] for pair in item["pairs"])
    ]
    oracle_evidence = sum(len(item["oracle_evidence_ids"]) for item in specs)
    role_evidence_hits = sum(
        pair["gold_role_match"] and pair["gold_evidence_match"] for pair in all_pairs
    )
    report = {
        "status": "passed",
        "experiment": "role_conditioned_retrieval_stage6a",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "complete_dialogues": len(dialogues),
        "targets": len(specs),
        "candidate_pairs": len(all_pairs),
        "context_needed_targets": len(context_needed),
        "current_sufficient_targets": sum(
            item["context_level"] == "current_sufficient" for item in specs
        ),
        "targets_with_gold_role_candidates": len(role_targets),
        "role_candidate_coverage": len(role_targets) / max(len(specs), 1),
        "context_needed_role_candidate_coverage": sum(
            any(pair["gold_role_match"] for pair in item["pairs"])
            for item in context_needed
        )
        / max(len(context_needed), 1),
        "wrong_role_eligible_targets": sum(
            any(pair["wrong_role_match"] for pair in item["pairs"]) for item in specs
        ),
        "oracle_evidence_turns": oracle_evidence,
        "oracle_evidence_covered_by_role_events": role_evidence_hits,
        "role_event_oracle_evidence_recall": role_evidence_hits
        / max(oracle_evidence, 1),
        "annotation_quality_gate": quality.get("eligible_for_oracle") is True,
        "interpretation": (
            "interaction_target=yes plus event_memory_write=yes is treated as a "
            "role-relevant event, not a true emotion-change label"
        ),
    }
    _write_json(paths["audit"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _make_scorer(section: dict[str, Any], labels: Sequence[str]) -> Any:
    from pfa_erc.training.llm_context_utility_stage1 import CausalLabelScorer

    return CausalLabelScorer(section, labels)


def score_selectors(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    root, _, section, _, dialogues, verified, _ = _load_context(config_path)
    paths = _paths(root, section)
    if overwrite and paths["selector_scores"].exists():
        paths["selector_scores"].unlink()
    specs = _specs(dialogues, verified, section)
    expected = {pair["pair_id"] for item in specs for pair in item["pairs"]}
    existing = _read_jsonl(paths["selector_scores"])
    completed = {item["pair_id"] for item in existing}
    scorer = _make_scorer(section, ("yes", "no"))

    for dialogue_number, dialogue in enumerate(dialogues, 1):
        dialogue_specs = [
            item for item in specs if item["dialogue_id"] == dialogue["dialogue_id"]
        ]
        pending = [
            pair
            for item in dialogue_specs
            for pair in item["pairs"]
            if pair["pair_id"] not in completed
        ]
        if not pending:
            continue
        prompts: list[str] = []
        layout: list[tuple[dict[str, Any], bool]] = []
        for pair in pending:
            prompts.append(pair["relevance_prompt"])
            has_role_prompt = pair["role_prompt"] is not None
            if has_role_prompt:
                prompts.append(pair["role_prompt"])
            layout.append((pair, has_role_prompt))
        scores = scorer.score_prompts(prompts)
        cursor = 0
        rows: list[dict[str, Any]] = []
        for pair, has_role_prompt in layout:
            relevance_scores = scores[cursor]
            cursor += 1
            role_scores: Sequence[float] | None = None
            if has_role_prompt:
                role_scores = scores[cursor]
                cursor += 1
            rows.append(
                {
                    key: value
                    for key, value in pair.items()
                    if key not in {"relevance_prompt", "role_prompt", "candidate"}
                }
                | {
                    "candidate_speaker": pair["candidate"]["speaker"],
                    "candidate_text": pair["candidate"]["text"],
                    "relevance_scores": [float(value) for value in relevance_scores],
                    "relevance_probability": _yes_probability(relevance_scores),
                    "role_scores": (
                        [float(value) for value in role_scores]
                        if role_scores is not None
                        else None
                    ),
                    "role_probability": (
                        _yes_probability(role_scores) if role_scores is not None else 0.0
                    ),
                }
            )
        _append_jsonl(paths["selector_scores"], rows)
        completed.update(item["pair_id"] for item in rows)
        print(
            f"[Stage 6A selector] dialogue {dialogue_number}/{len(dialogues)} "
            f"{dialogue['dialogue_id']}: {len(rows)} pair(s)",
            flush=True,
        )

    final = _read_jsonl(paths["selector_scores"])
    actual = {item["pair_id"] for item in final}
    if actual != expected or len(final) != len(expected):
        raise ValueError(
            f"Selector coverage mismatch: expected {len(expected)}, got "
            f"{len(actual)} unique/{len(final)} rows"
        )
    result = {
        "status": "complete",
        "pairs": len(final),
        "prompts_scored": sum(1 + int(item["role_predictable"]) for item in final),
        "selector_score_file": str(paths["selector_scores"]),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _by_target(records: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["target_key"]].append(item)
    return grouped


def score_predictions(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    root, _, section, labels, dialogues, verified, _ = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["selector_scores"].exists():
        raise FileNotFoundError("Run score-selectors before score-predictions")
    if overwrite and paths["raw_scores"].exists():
        paths["raw_scores"].unlink()
    selector_records = _read_jsonl(paths["selector_scores"])
    selector_by_target = _by_target(selector_records)
    specs = _specs(dialogues, verified, section)
    stage4_records = {
        item["key"]: item
        for item in _read_jsonl(_resolve(root, section["stage4_score_file"]))
    }
    existing = _read_jsonl(paths["raw_scores"])
    completed = {item["key"] for item in existing}
    scorer = _make_scorer(section, labels)
    temperature = float(section.get("shared_temperature", 1.0))
    top_k_grid = sorted({int(value) for value in section.get("top_k_grid", [1, 2, 3])})
    if not top_k_grid or min(top_k_grid) <= 0:
        raise ValueError("top_k_grid must contain positive integers")
    max_history = int(section.get("max_history_turns", 12))
    preamble = str(section.get("prompt_preamble", "You are a careful emotion classifier."))

    for dialogue_number, dialogue in enumerate(dialogues, 1):
        utterances = dialogue["utterances"]
        dialogue_specs = [
            item
            for item in specs
            if item["dialogue_id"] == dialogue["dialogue_id"]
            and item["key"] not in completed
        ]
        output: list[dict[str, Any]] = []
        for spec in dialogue_specs:
            stage4_record = stage4_records[spec["key"]]
            pair_rows = selector_by_target.get(spec["key"], [])
            by_id = {
                index: _candidate_dict(index, utterances[index])
                for index in range(max(0, spec["turn_id"] - max_history), spec["turn_id"])
            }
            selected_ids: dict[str, list[int]] = {}
            prompts: list[str] = []
            prompt_branches: list[str] = []
            payloads: dict[str, dict[str, Any]] = {
                branch: stage4_record[branch] for branch in BASE_BRANCHES
            }
            target = {
                "turn_id": spec["turn_id"],
                "speaker": spec["speaker"],
                "text": spec["text"],
            }
            for method in METHODS:
                for k in top_k_grid:
                    branch = f"{method}_k{k}"
                    ids = rank_candidate_ids(pair_rows, method, k)
                    selected_ids[branch] = ids
                    if not ids:
                        payloads[branch] = stage4_record["current_only"]
                        continue
                    evidence = [by_id[value] for value in ids]
                    prompts.append(
                        stage4.build_prompt(target, evidence, [], labels, preamble)
                    )
                    prompt_branches.append(branch)
            if prompts:
                scores = scorer.score_prompts(prompts)
                for branch, values in zip(prompt_branches, scores):
                    payloads[branch] = stage4._probability_payload(
                        values, labels, temperature
                    )
            output.append(
                {
                    "key": spec["key"],
                    "dialogue_id": spec["dialogue_id"],
                    "turn_id": spec["turn_id"],
                    "speaker": spec["speaker"],
                    "gold": spec["gold"],
                    "context_level": spec["context_level"],
                    "oracle_evidence_ids": spec["oracle_evidence_ids"],
                    "history_count": len(pair_rows),
                    "gold_role_candidate_count": sum(
                        item["gold_role_match"] for item in pair_rows
                    ),
                    "wrong_role_candidate_count": sum(
                        item["wrong_role_match"] for item in pair_rows
                    ),
                    "selected_ids": selected_ids,
                }
                | payloads
            )
        if output:
            _append_jsonl(paths["raw_scores"], output)
            completed.update(item["key"] for item in output)
        print(
            f"[Stage 6A prediction] dialogue {dialogue_number}/{len(dialogues)} "
            f"{dialogue['dialogue_id']}: {len(output)} target(s)",
            flush=True,
        )

    final = _read_jsonl(paths["raw_scores"])
    if {item["key"] for item in final} != {item["key"] for item in specs}:
        raise ValueError("Prediction score coverage does not match Stage 6A targets")
    result = {
        "status": "complete",
        "targets": len(final),
        "top_k_grid": top_k_grid,
        "prediction_score_file": str(paths["raw_scores"]),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _mean_nll(records: Sequence[dict[str, Any]], branch: str, labels: Sequence[str]) -> float:
    label_to_id = {label: index for index, label in enumerate(labels)}
    values = []
    for item in records:
        index = label_to_id[item["gold"]]
        values.append(-math.log(max(float(item[branch]["probabilities"][index]), 1e-12)))
    return float(np.mean(values))


def _materialize_routed(
    records: Sequence[dict[str, Any]],
    method: str,
    k: int,
    threshold: float,
) -> list[dict[str, Any]]:
    branch = f"{method}_k{k}"
    output = []
    for item in records:
        payload, used = route_payload(item, branch, threshold)
        row = dict(item)
        row["_candidate"] = payload
        row["_context_used"] = used
        row["_evidence_count"] = len(item["selected_ids"].get(branch, [])) if used else 0
        output.append(row)
    return output


def _selection_objective(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    context_cost: float,
    evidence_cost: float,
    max_k: int,
) -> tuple[float, float, float, float]:
    nll = _mean_nll(records, "_candidate", labels)
    use_rate = float(np.mean([item["_context_used"] for item in records]))
    mean_evidence = float(np.mean([item["_evidence_count"] for item in records]))
    objective = nll + context_cost * use_rate + evidence_cost * mean_evidence / max(max_k, 1)
    macro = stage4.metric_bundle(records, "_candidate", labels)["macro_f1"]
    return objective, nll, -macro, use_rate


def choose_parameters(
    training_records: Sequence[dict[str, Any]],
    method: str,
    labels: Sequence[str],
    k_grid: Sequence[int],
    threshold_grid: Sequence[float],
    context_cost: float,
    evidence_cost: float,
    routed: bool,
) -> dict[str, Any]:
    if not training_records:
        raise ValueError("Cross-fitting requires non-empty training dialogues")
    candidates: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    thresholds = threshold_grid if routed else [0.0]
    for k in k_grid:
        for threshold in thresholds:
            materialized = _materialize_routed(
                training_records, method, int(k), float(threshold)
            )
            objective = _selection_objective(
                materialized,
                labels,
                context_cost,
                evidence_cost,
                max(k_grid),
            )
            parameter = {
                "k": int(k),
                "entropy_threshold": float(threshold),
                "objective": float(objective[0]),
                "nll": float(objective[1]),
                "macro_f1": float(-objective[2]),
                "context_use_rate": float(objective[3]),
            }
            candidates.append((objective + (float(k), float(threshold)), parameter))
    return min(candidates, key=lambda item: item[0])[1]


def crossfit_predictions(
    raw_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    dialogue_ids = sorted({item["dialogue_id"] for item in raw_records})
    if len(dialogue_ids) < 2:
        raise ValueError("At least two complete dialogues are required for cross-fitting")
    k_grid = sorted({int(value) for value in section.get("top_k_grid", [1, 2, 3])})
    threshold_grid = sorted(
        {float(value) for value in section.get("entropy_threshold_grid", [0.0, 0.6, 0.75, 0.9])}
    )
    context_cost = float(section.get("context_use_penalty", 0.002))
    evidence_cost = float(section.get("evidence_count_penalty", 0.001))
    output: list[dict[str, Any]] = []
    for heldout in dialogue_ids:
        training = [item for item in raw_records if item["dialogue_id"] != heldout]
        testing = [item for item in raw_records if item["dialogue_id"] == heldout]
        choices: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            choices[f"{method}_always"] = choose_parameters(
                training,
                method,
                labels,
                k_grid,
                threshold_grid,
                context_cost,
                evidence_cost,
                routed=False,
            )
            choices[f"{method}_routed"] = choose_parameters(
                training,
                method,
                labels,
                k_grid,
                threshold_grid,
                context_cost,
                evidence_cost,
                routed=True,
            )
        for item in testing:
            row = {
                key: value
                for key, value in item.items()
                if not any(key.startswith(f"{method}_k") for method in METHODS)
            }
            row["crossfit_heldout_dialogue"] = heldout
            row["crossfit_parameters"] = choices
            row["context_used"] = {}
            row["retrieved_ids"] = {}
            for method in METHODS:
                for mode in ("always", "routed"):
                    final_branch = f"{method}_{mode}"
                    parameter = choices[final_branch]
                    raw_branch = f"{method}_k{parameter['k']}"
                    payload, used = route_payload(
                        item, raw_branch, parameter["entropy_threshold"]
                    )
                    row[final_branch] = payload
                    row["context_used"][final_branch] = used
                    row["retrieved_ids"][final_branch] = (
                        item["selected_ids"].get(raw_branch, []) if used else []
                    )
            gold_needed = item["context_level"] in {
                "history_helpful",
                "history_required",
            }
            row["oracle_context_role"] = (
                row["gold_role_always"]
                if gold_needed and row["retrieved_ids"]["gold_role_always"]
                else item["current_only"]
            )
            label_index = labels.index(item["gold"])
            current_nll = -math.log(
                max(float(item["current_only"]["probabilities"][label_index]), 1e-12)
            )
            oracle_nll = -math.log(
                max(float(item["oracle_evidence"]["probabilities"][label_index]), 1e-12)
            )
            row["oracle_current_or_evidence"] = (
                item["oracle_evidence"] if oracle_nll < current_nll else item["current_only"]
            )
            output.append(row)
    return sorted(output, key=lambda item: (item["dialogue_id"], item["turn_id"]))


def _safe_ap(gold: Sequence[bool], scores: Sequence[float]) -> float | None:
    values = np.asarray(gold, dtype=np.int64)
    if len(values) == 0 or len(set(values.tolist())) < 2:
        return None
    return float(average_precision_score(values, np.asarray(scores, dtype=np.float64)))


def _safe_f1(gold: Sequence[bool], scores: Sequence[float], threshold: float = 0.5) -> float | None:
    values = np.asarray(gold, dtype=np.int64)
    if len(values) == 0:
        return None
    predicted = np.asarray(scores, dtype=np.float64) >= threshold
    return float(f1_score(values, predicted, zero_division=0))


def _retrieval_diagnostics(
    oof: Sequence[dict[str, Any]], selector_scores: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    role_gold = [item["gold_role_match"] for item in selector_scores if item["role_predictable"]]
    role_scores = [
        item["role_probability"] for item in selector_scores if item["role_predictable"]
    ]
    evidence_gold = [item["gold_evidence_match"] for item in selector_scores]
    evidence_scores = [item["relevance_probability"] for item in selector_scores]
    diagnostics: dict[str, Any] = {
        "predicted_role_average_precision": _safe_ap(role_gold, role_scores),
        "predicted_role_f1_at_050": _safe_f1(role_gold, role_scores),
        "relevance_average_precision": _safe_ap(evidence_gold, evidence_scores),
        "relevance_f1_at_050": _safe_f1(evidence_gold, evidence_scores),
    }
    for branch in ("general_routed", "gold_role_routed", "predicted_role_routed"):
        selected = [item["retrieved_ids"][branch] for item in oof]
        gold_sets = [set(item["oracle_evidence_ids"]) for item in oof]
        hits = sum(len(set(ids) & gold) for ids, gold in zip(selected, gold_sets))
        total_selected = sum(len(ids) for ids in selected)
        total_gold = sum(len(gold) for gold in gold_sets)
        diagnostics[branch] = {
            "context_use_rate": float(
                np.mean([item["context_used"][branch] for item in oof])
            ),
            "mean_selected_turns": float(np.mean([len(ids) for ids in selected])),
            "evidence_precision": hits / max(total_selected, 1),
            "evidence_recall": hits / max(total_gold, 1),
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
        "role_candidate_available": [
            item for item in records if item["gold_role_candidate_count"] > 0
        ],
        "wrong_role_eligible": [
            item for item in records if item["wrong_role_candidate_count"] > 0
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


def _bootstrap_or_none(
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
        "current_only": "6A-0 current only",
        "full_history": "6A-1 full history",
        "general_routed": "6A-2 general minimal retrieval",
        "gold_role_routed": "6A-3 gold role-conditioned retrieval",
        "predicted_role_routed": "6A-4 predicted role-conditioned retrieval",
        "wrong_role_routed": "6A-5 wrong-role retrieval",
        "oracle_evidence": "6A-6 verified evidence Oracle",
        "oracle_context_role": "6A-7 gold context-need + role retrieval",
        "oracle_current_or_evidence": "6A-8 current/evidence outcome Oracle",
    }
    shown = tuple(names)
    lines = [
        "# Stage 6A — Role-Conditioned Minimal Evidence Retrieval",
        "",
        "> Complete-dialogue MELD validation-only mechanism test. MELD test remains locked.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for branch in shown:
        item = metrics[branch]
        lines.append(
            f"| {names[branch]} | {item['weighted_f1']:.4f} | "
            f"{item['macro_f1']:.4f} | {item['nll']:.4f} | "
            f"{item['brier']:.4f} | {item['ece']:.4f} |"
        )
    lines.extend(["", "## Paired mechanism contrasts", ""])
    for name, item in summary["comparisons"].items():
        if item["bootstrap"] is None:
            lines.append(f"- **{name}**: not estimable on this pilot subset.")
            continue
        macro = item["bootstrap"]["metrics"]["macro_f1"]
        nll = item["bootstrap"]["metrics"]["nll"]
        lines.append(
            f"- **{name}**: macro advantage {macro['point_advantage']:+.4f} "
            f"(95% CI {macro['ci95'][0]:+.4f} to {macro['ci95'][1]:+.4f}, "
            f"P>0={macro['probability_positive']:.3f}); NLL advantage "
            f"{nll['point_advantage']:+.4f} (P>0={nll['probability_positive']:.3f})."
        )
    diagnostics = summary["diagnostics"]
    lines.extend(
        [
            "",
            "## Retrieval diagnostics",
            "",
            f"- Predicted-role AP/F1@0.5: **{diagnostics['predicted_role_average_precision']} / {diagnostics['predicted_role_f1_at_050']}**",
            f"- Pairwise relevance AP/F1@0.5: **{diagnostics['relevance_average_precision']} / {diagnostics['relevance_f1_at_050']}**",
        ]
    )
    for branch in ("general_routed", "gold_role_routed", "predicted_role_routed"):
        item = diagnostics[branch]
        lines.append(
            f"- {branch}: use={item['context_use_rate']:.1%}, "
            f"mean turns={item['mean_selected_turns']:.3f}, "
            f"evidence P/R={item['evidence_precision']:.3f}/{item['evidence_recall']:.3f}"
        )
    lines.extend(["", "## Frozen decision", "", f"Status: **{summary['decision']['status']}**"])
    for name, passed in summary["decision"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "The V2.2 role label is an operational role-relevance label, not a verified change in a person's internal emotion. Gold-role and gold-context branches are Oracles and are not deployable results.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path) -> dict[str, Any]:
    root, _, section, labels, dialogues, verified, quality = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["selector_scores"].exists() or not paths["raw_scores"].exists():
        raise FileNotFoundError("Run score-selectors and score-predictions before analyze")
    selector_scores = _read_jsonl(paths["selector_scores"])
    raw_records = _read_jsonl(paths["raw_scores"])
    if {item["key"] for item in raw_records} != set(verified):
        raise ValueError("Raw prediction coverage mismatch")
    oof = crossfit_predictions(raw_records, labels, section)
    _write_jsonl(paths["oof_scores"], oof)
    subsets = _subset_metrics(oof, labels)
    diagnostics = _retrieval_diagnostics(oof, selector_scores)
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2041))
    all_records = list(oof)
    wrong_eligible = [item for item in oof if item["wrong_role_candidate_count"] > 0]
    comparison_specs = {
        "gold_role_vs_general": (all_records, "gold_role_routed", "general_routed"),
        "predicted_role_vs_general": (
            all_records,
            "predicted_role_routed",
            "general_routed",
        ),
        "gold_role_vs_full": (all_records, "gold_role_routed", "full_history"),
        "predicted_role_vs_current": (
            all_records,
            "predicted_role_routed",
            "current_only",
        ),
        "gold_role_vs_wrong_role_eligible": (
            wrong_eligible,
            "gold_role_routed",
            "wrong_role_routed",
        ),
        "oracle_evidence_vs_gold_role": (
            all_records,
            "oracle_evidence",
            "gold_role_routed",
        ),
    }
    comparisons: dict[str, Any] = {}
    for offset, (name, (rows, proposed, control)) in enumerate(comparison_specs.items()):
        comparisons[name] = {
            "proposed": proposed,
            "control": control,
            "targets": len(rows),
            "bootstrap": _bootstrap_or_none(
                rows, proposed, control, labels, samples, seed + offset
            ),
        }

    threshold = float(section.get("minimum_bootstrap_probability", 0.90))
    w_tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    predicted_gap = float(section.get("predicted_gold_nll_tolerance", 0.05))
    all_metrics = subsets["all"]["metrics"]

    def probability(name: str, metric: str) -> float:
        bootstrap = comparisons[name]["bootstrap"]
        if bootstrap is None:
            return 0.0
        return float(bootstrap["metrics"][metric]["probability_positive"])

    gold_supported = (
        probability("gold_role_vs_general", "nll") >= threshold
        or probability("gold_role_vs_general", "macro_f1") >= threshold
    )
    predicted_supported = (
        probability("predicted_role_vs_general", "nll") >= threshold
        or probability("predicted_role_vs_general", "macro_f1") >= threshold
    )
    role_ap = diagnostics["predicted_role_average_precision"]
    checks = {
        "annotation_quality_gate": quality.get("eligible_for_oracle") is True,
        "complete_crossfit_coverage": len(oof) == len(verified),
        "gold_role_preserves_current_weighted_f1": all_metrics["gold_role_routed"][
            "weighted_f1"
        ]
        >= all_metrics["current_only"]["weighted_f1"] - w_tolerance,
        "gold_role_filter_adds_over_general_retrieval": gold_supported,
        "gold_role_adds_over_full_history": (
            probability("gold_role_vs_full", "nll") >= threshold
            or probability("gold_role_vs_full", "macro_f1") >= threshold
        ),
        "predicted_role_average_precision": role_ap is not None
        and role_ap >= float(section.get("minimum_predicted_role_ap", 0.55)),
        "predicted_role_adds_over_general_retrieval": predicted_supported,
        "predicted_role_within_gold_nll_tolerance": all_metrics[
            "predicted_role_routed"
        ]["nll"]
        <= all_metrics["gold_role_routed"]["nll"] + predicted_gap,
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "advance_to_train_scale_role_conditioned_retrieval"
            if eligible
            else "stop_and_diagnose_role_conditioned_retrieval"
        ),
        "eligible": eligible,
        "checks": checks,
        "selection_basis": (
            "leave-one-complete-dialogue-out NLL plus frozen context/evidence cost; "
            "gold-dependent Oracles excluded"
        ),
        "next_action": (
            "Expand train-only role-event annotations and train the deployable role router."
            if eligible
            else "Keep MELD test locked; inspect role-candidate coverage, relevance AP, and predicted-role AP before scaling."
        ),
        "test_locked": True,
    }
    summary = {
        "experiment": "role_conditioned_retrieval_stage6a",
        "version": VERSION,
        "dataset": "meld",
        "split": "validation",
        "test_locked": True,
        "oracle_diagnostic": True,
        "labels": labels,
        "targets": len(oof),
        "dialogues": len(dialogues),
        "subsets": subsets,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
        "parameter_counts": {
            branch: {
                f"k={key[0]},threshold={key[1]:.2f}": count
                for key, count in Counter(
                    (
                        item["crossfit_parameters"][branch]["k"],
                        item["crossfit_parameters"][branch]["entropy_threshold"],
                    )
                    for item in oof
                ).items()
            }
            for branch in (
                "general_routed",
                "gold_role_routed",
                "predicted_role_routed",
                "wrong_role_routed",
            )
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
        "annotation_sha256": _sha256(
            _resolve(root, section["annotation_file"])
        ),
        "stage4_scores_sha256": _sha256(
            _resolve(root, section["stage4_score_file"])
        ),
        "selector_scores_sha256": _sha256(paths["selector_scores"]),
        "raw_predictions_sha256": _sha256(paths["raw_scores"]),
        "oof_predictions_sha256": _sha256(paths["oof_scores"]),
        "causal_invariants": [
            "only source turns strictly earlier than the target are retrievable",
            "predicted-role prompts contain no future target utterance",
            "each held-out dialogue is excluded from parameter selection",
            "gold outcome Oracle is excluded from model selection",
        ],
    }
    _write_json(paths["manifest"], manifest)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, _, section, _, _, verified, _ = _load_context(config_path)
    paths = _paths(root, section)
    required = [
        paths["audit"],
        paths["selector_scores"],
        paths["raw_scores"],
        paths["oof_scores"],
        paths["manifest"],
        paths["summary_json"],
        paths["summary_md"],
        paths["decision"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 6A artifact(s): {missing}")
    summary = _read_json(paths["summary_json"])
    raw = _read_jsonl(paths["raw_scores"])
    oof = _read_jsonl(paths["oof_scores"])
    selector = _read_jsonl(paths["selector_scores"])
    no_future_candidates = all(
        int(item["source_turn_id"]) < int(item["target_turn_id"]) for item in selector
    )
    crossfit_exclusion = all(
        item["crossfit_heldout_dialogue"] == item["dialogue_id"] for item in oof
    )
    finite = all(
        math.isfinite(float(summary["subsets"]["all"]["metrics"][branch][metric]))
        for branch in FINAL_BRANCHES
        for metric in ("weighted_f1", "macro_f1", "nll", "brier", "ece")
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "complete_raw_score_coverage": len(raw) == len(verified),
        "complete_oof_score_coverage": len(oof) == len(verified),
        "all_final_variants": all(
            branch in summary.get("subsets", {}).get("all", {}).get("metrics", {})
            for branch in FINAL_BRANCHES
        ),
        "no_future_candidates": no_future_candidates,
        "crossfit_heldout_dialogue_marked": crossfit_exclusion,
        "finite_metrics": finite,
        "paired_bootstrap_present": all(
            item.get("bootstrap") is not None
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
        raise RuntimeError("Stage 6A artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only role-conditioned minimal evidence retrieval"
    )
    parser.add_argument(
        "stage",
        choices=(
            "check",
            "audit",
            "score-selectors",
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
    elif arguments.stage == "score-selectors":
        score_selectors(arguments.config, overwrite=arguments.overwrite)
    elif arguments.stage == "score-predictions":
        score_predictions(arguments.config, overwrite=arguments.overwrite)
    elif arguments.stage == "analyze":
        analyze(arguments.config)
    elif arguments.stage == "verify":
        verify(arguments.config)
    else:
        check(arguments.config)
        audit(arguments.config)
        score_selectors(arguments.config, overwrite=arguments.overwrite)
        score_predictions(arguments.config, overwrite=arguments.overwrite)
        analyze(arguments.config)
        verify(arguments.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
