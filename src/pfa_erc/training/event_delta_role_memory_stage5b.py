"""Stage 5B: cross-fitted event-delta role-memory diagnostic.

Stage 5A estimated an absolute emotion distribution after each routed event.
Its selected update weight collapsed to one, so accumulated memory became the
last event.  Stage 5B instead scores the same recipient immediately before and
after an event and stores only the centered log-probability difference.

Hard V2.2 annotations route evidence and recipients only.  They never provide
emotion directions or magnitudes.  MELD test is inaccessible by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pfa_erc.training import hard_role_oracle_stage4 as stage4
from pfa_erc.training import latent_role_belief_stage5a as stage5a

VERSION = "1.6.0"
BRANCHES = (
    "current_only",
    "oracle_evidence_direct",
    "stage5a_absolute_state",
    "last_event_delta",
    "accumulated_event_delta",
    "random_evidence_delta",
    "wrong_recipient_delta",
    "oracle_current_or_delta",
)
EVALUATION_SUBSETS = (
    "all",
    "affected_next_turn",
    "single_event_affected",
    "multi_event_affected",
    "wrong_recipient_eligible",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get("artifact_dir", "artifacts/event_delta_role_memory_stage5b/meld_validation"),
    )
    return {
        "base": base,
        "audit": base / "reports" / "s5b_event_topology_audit.json",
        "pair_scores": base / "scores" / "s5b_paired_event_beliefs_validation.jsonl",
        "oof_scores": base / "scores" / "s5b_crossfit_predictions_validation.jsonl",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
        "manifest": base / "run_manifest.json",
    }


def _load_context(
    config_path: Path,
) -> tuple[
    Path,
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    config = _read_yaml(config_path)
    section = config.get("event_delta_role_memory_stage5b", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 5B is validation-only; refusing any non-validation split")
    root, _, labels, dialogues, verified, quality, stage5a_section = stage5a._load_context(
        config_path
    )
    stage5a_path = _resolve(root, section["stage5a_oof_score_file"])
    if not stage5a_path.exists():
        raise FileNotFoundError(
            f"Missing Stage 5A OOF scores: {stage5a_path}. Complete Stage 5A first."
        )
    stage5a_rows = _read_jsonl(stage5a_path)
    expected = set(verified)
    actual = {item.get("key") for item in stage5a_rows}
    if actual != expected or len(stage5a_rows) != len(expected):
        raise ValueError(
            f"Stage 5A OOF coverage mismatch: {len(actual)} unique/{len(stage5a_rows)} rows, "
            f"expected {len(expected)}"
        )
    return (
        root,
        config,
        labels,
        dialogues,
        verified,
        quality,
        stage5a_section,
        section,
    )


def _history_items(
    dialogue: dict[str, Any], evidence_ids: Sequence[int]
) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": int(turn_id),
            "speaker": dialogue["utterances"][int(turn_id)]["speaker"],
            "text": dialogue["utterances"][int(turn_id)]["text"],
        }
        for turn_id in evidence_ids
    ]


def _observed_lines(items: Sequence[dict[str, Any]]) -> str:
    if not items:
        return "[no preceding evidence selected]"
    return "\n".join(
        f"[{item['turn_id']}] {item['speaker']}: {item['text']}" for item in items
    )


def build_paired_event_prompts(
    *,
    evidence: Sequence[dict[str, Any]],
    event_turn_id: int,
    event_speaker: str,
    event_text: str,
    recipient: str,
    labels: Sequence[str],
    preamble: str,
) -> tuple[str, str]:
    """Build matched before/after prompts without future text or gold labels."""
    shared = (
        f"{preamble.strip()}\n\n"
        f"Role being tracked: {recipient}\n"
        "Observed causal dialogue prefix:\n"
    )
    suffix = (
        "\n\nEstimate the tracked role's predictive emotion immediately after the final "
        "observed line. If no line is available, use an uninformed pre-event belief "
        "rather than inventing context. "
        "This is a predictive distribution, not a claim about a hidden true mental state. "
        f"Return exactly one label from: {', '.join(labels)}.\n"
        "Emotion:"
    )
    evidence_text = _observed_lines(evidence)
    event_line = f"[{int(event_turn_id)}] {event_speaker}: {event_text}"
    before = shared + evidence_text + suffix
    # Retaining the identical evidence text (including the empty-evidence
    # marker) makes the current event the only dialogue-content difference.
    after = shared + evidence_text + "\n" + event_line + suffix
    return before, after


def build_event_pair_specs(
    dialogues: Sequence[dict[str, Any]],
    verified: dict[str, dict[str, Any]],
    labels: Sequence[str],
    stage5a_section: dict[str, Any],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    dialogue_map = {item["dialogue_id"]: item for item in dialogues}
    base_specs = stage5a.extract_event_specs(
        dialogues, verified, labels, stage5a_section
    )
    preamble = str(
        section.get(
            "paired_prompt_preamble",
            "You estimate causal predictive emotion beliefs for dialogue participants. "
            "Use only supplied text, do not invent missing context, and do not explain.",
        )
    )
    specs: list[dict[str, Any]] = []
    for base in base_specs:
        dialogue = dialogue_map[base["dialogue_id"]]
        oracle_evidence = _history_items(dialogue, base["oracle_evidence_ids"])
        random_evidence = _history_items(dialogue, base["random_evidence_ids"])
        oracle_before, oracle_after = build_paired_event_prompts(
            evidence=oracle_evidence,
            event_turn_id=int(base["event_turn_id"]),
            event_speaker=base["event_speaker"],
            event_text=base["event_text"],
            recipient=base["recipient"],
            labels=labels,
            preamble=preamble,
        )
        random_before, random_after = build_paired_event_prompts(
            evidence=random_evidence,
            event_turn_id=int(base["event_turn_id"]),
            event_speaker=base["event_speaker"],
            event_text=base["event_text"],
            recipient=base["recipient"],
            labels=labels,
            preamble=preamble,
        )
        spec = {key: value for key, value in base.items() if not key.endswith("_prompt")}
        spec.update(
            {
                "oracle_before_prompt": oracle_before,
                "oracle_after_prompt": oracle_after,
                "random_before_prompt": random_before,
                "random_after_prompt": random_after,
            }
        )
        specs.append(spec)
    return specs


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Probability vector must be finite, nonnegative, and one-dimensional")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Probability vector has zero mass")
    return array / total


def event_log_delta(before: Sequence[float], after: Sequence[float]) -> np.ndarray:
    before_array = _normalize(before)
    after_array = _normalize(after)
    if len(before_array) != len(after_array):
        raise ValueError("Before and after distributions have different dimensions")
    delta = np.log(np.clip(after_array, 1e-12, 1.0)) - np.log(
        np.clip(before_array, 1e-12, 1.0)
    )
    return delta - float(delta.mean())


def apply_event_deltas(
    anchor: Sequence[float],
    events: Sequence[dict[str, Any]],
    *,
    current_turn_id: int,
    read_weight: float,
    decay: float,
    delta_clip: float,
    last_only: bool,
) -> np.ndarray:
    if not 0.0 <= read_weight <= 1.0:
        raise ValueError("read_weight must lie in [0,1]")
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must lie in [0,1]")
    if delta_clip <= 0:
        raise ValueError("delta_clip must be positive")
    anchor_array = _normalize(anchor)
    if not events or read_weight == 0.0:
        return anchor_array
    selected = [max(events, key=lambda item: int(item["event_turn_id"]))] if last_only else list(events)
    total = np.zeros_like(anchor_array)
    for event in selected:
        delta = np.asarray(event["delta_logits"], dtype=np.float64)
        if delta.shape != anchor_array.shape or not np.all(np.isfinite(delta)):
            raise ValueError("Invalid event-delta vector")
        age = max(int(current_turn_id) - int(event["event_turn_id"]) - 1, 0)
        total += (decay**age) * np.clip(delta, -delta_clip, delta_clip)
    logits = np.log(np.clip(anchor_array, 1e-12, 1.0)) + read_weight * total
    logits -= float(np.max(logits))
    return _normalize(np.exp(logits))


def _payload(probabilities: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    values = _normalize(probabilities)
    index = int(np.argmax(values))
    return {
        "probabilities": values.tolist(),
        "prediction": labels[index],
        "confidence": float(values[index]),
    }


def _gold_nll(payload: dict[str, Any], gold: str, labels: Sequence[str]) -> float:
    index = labels.index(gold)
    return -math.log(max(float(payload["probabilities"][index]), 1e-12))


def check(config_path: Path) -> dict[str, Any]:
    root, _, labels, dialogues, verified, quality, _, section = _load_context(config_path)
    result = {
        "status": "ready",
        "experiment": "event_delta_role_memory_stage5b",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "emotion_delta_labels_used": False,
        "model_name": section["model_name"],
        "targets": len(verified),
        "complete_dialogues": len(dialogues),
        "labels": labels,
        "annotation_quality_status": quality.get("status"),
        "stage5a_oof_score_file": str(_resolve(root, section["stage5a_oof_score_file"])),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def audit(config_path: Path) -> dict[str, Any]:
    root, _, labels, dialogues, verified, _, stage5a_section, section = _load_context(
        config_path
    )
    paths = _paths(root, section)
    specs = build_event_pair_specs(
        dialogues, verified, labels, stage5a_section, section
    )
    observable = [item for item in specs if item["next_recipient_key"] is not None]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in observable:
        by_target[item["next_recipient_key"]].append(item)
    single_targets = [key for key, values in by_target.items() if len(values) == 1]
    multi_targets = [key for key, values in by_target.items() if len(values) >= 2]
    wrong_targets = {
        item["next_recipient_key"]
        for item in observable
        if item.get("wrong_recipient_available")
    }
    causal_order = all(
        all(int(evidence_id) < int(item["event_turn_id"]) for evidence_id in item["oracle_evidence_ids"])
        and all(int(evidence_id) < int(item["event_turn_id"]) for evidence_id in item["random_evidence_ids"])
        and (
            item["next_recipient_turn_id"] is None
            or int(item["event_turn_id"]) < int(item["next_recipient_turn_id"])
        )
        for item in specs
    )
    report = {
        "status": "passed" if causal_order else "failed",
        "experiment": "event_delta_role_memory_stage5b",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "event_writes": len(specs),
        "observable_event_recipient_pairs": len(observable),
        "unique_affected_next_turn_targets": len(by_target),
        "single_event_targets": len(single_targets),
        "multi_event_targets": len(multi_targets),
        "multi_event_counts": dict(
            sorted(Counter(len(values) for values in by_target.values() if len(values) >= 2).items())
        ),
        "wrong_recipient_eligible_targets": len(wrong_targets),
        "causal_order_verified": causal_order,
        "paired_prompts_per_event": 4,
        "total_prompts": 4 * len(specs),
    }
    _write_json(paths["audit"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("Stage 5B causal topology audit failed")
    return report


def _make_scorer(section: dict[str, Any], labels: Sequence[str]) -> Any:
    from pfa_erc.training.llm_context_utility_stage1 import CausalLabelScorer

    return CausalLabelScorer(section, labels)


def score_pairs(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    root, _, labels, dialogues, verified, _, stage5a_section, section = _load_context(
        config_path
    )
    paths = _paths(root, section)
    specs = build_event_pair_specs(
        dialogues, verified, labels, stage5a_section, section
    )
    if overwrite and paths["pair_scores"].exists():
        paths["pair_scores"].unlink()
    existing = _read_jsonl(paths["pair_scores"])
    completed = {item["event_id"] for item in existing}
    pending = [item for item in specs if item["event_id"] not in completed]
    if pending:
        scorer = _make_scorer(section, labels)
        temperature = float(section.get("shared_temperature", 1.0))
        batch_events = int(section.get("event_checkpoint_batch", 8))
        for start in range(0, len(pending), batch_events):
            chunk = pending[start : start + batch_events]
            prompts = [
                prompt
                for item in chunk
                for prompt in (
                    item["oracle_before_prompt"],
                    item["oracle_after_prompt"],
                    item["random_before_prompt"],
                    item["random_after_prompt"],
                )
            ]
            scores = scorer.score_prompts(prompts)
            cursor = 0
            rows: list[dict[str, Any]] = []
            for item in chunk:
                row = {key: value for key, value in item.items() if not key.endswith("_prompt")}
                oracle_before = stage4._probability_payload(scores[cursor], labels, temperature)
                oracle_after = stage4._probability_payload(scores[cursor + 1], labels, temperature)
                random_before = stage4._probability_payload(scores[cursor + 2], labels, temperature)
                random_after = stage4._probability_payload(scores[cursor + 3], labels, temperature)
                cursor += 4
                row.update(
                    {
                        "oracle_before_belief": oracle_before,
                        "oracle_after_belief": oracle_after,
                        "random_before_belief": random_before,
                        "random_after_belief": random_after,
                        "oracle_delta_logits": event_log_delta(
                            oracle_before["probabilities"], oracle_after["probabilities"]
                        ).tolist(),
                        "random_delta_logits": event_log_delta(
                            random_before["probabilities"], random_after["probabilities"]
                        ).tolist(),
                    }
                )
                rows.append(row)
            _append_jsonl(paths["pair_scores"], rows)
            print(
                f"[Stage 5B] paired beliefs {min(start + len(chunk), len(pending))}/"
                f"{len(pending)} pending ({len(completed) + start + len(chunk)}/{len(specs)} total)",
                flush=True,
            )
    records = _read_jsonl(paths["pair_scores"])
    expected = {item["event_id"] for item in specs}
    actual = {item.get("event_id") for item in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"Paired-score coverage mismatch: {len(actual)} unique/{len(records)} rows, "
            f"expected {len(expected)}"
        )
    result = {
        "status": "complete",
        "events": len(records),
        "prompts_scored": 4 * len(records),
        "pair_score_file": str(paths["pair_scores"]),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _event_entry(event: dict[str, Any], field: str) -> dict[str, Any]:
    return {
        "event_id": event["event_id"],
        "event_turn_id": int(event["event_turn_id"]),
        "delta_logits": list(event[field]),
    }


def simulate_dialogue(
    dialogue: dict[str, Any],
    stage4_by_key: dict[str, dict[str, Any]],
    stage5a_by_key: dict[str, dict[str, Any]],
    events_by_turn: dict[tuple[str, int], list[dict[str, Any]]],
    labels: Sequence[str],
    *,
    read_weight: float,
    decay: float,
    delta_clip: float,
) -> list[dict[str, Any]]:
    """Predict each turn before applying writes caused by that same turn."""
    states: dict[str, dict[str, list[dict[str, Any]]]] = {
        "correct": defaultdict(list),
        "random": defaultdict(list),
        "wrong": defaultdict(list),
    }
    wrong_eligible: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    dialogue_id = dialogue["dialogue_id"]
    for turn_id, utterance in enumerate(dialogue["utterances"]):
        key = stage4._key(dialogue_id, turn_id)
        stage4_row = stage4_by_key[key]
        stage5a_row = stage5a_by_key[key]
        speaker = utterance["speaker"]
        gold = utterance["emotion"]
        anchor = stage4_row["current_only"]["probabilities"]
        correct_events = list(states["correct"].get(speaker, []))
        random_events = list(states["random"].get(speaker, []))
        wrong_events = list(states["wrong"].get(speaker, []))
        row: dict[str, Any] = {
            "key": key,
            "dialogue_id": dialogue_id,
            "turn_id": turn_id,
            "speaker": speaker,
            "gold": gold,
            "event_count_read": len(correct_events),
            "affected_next_turn": bool(correct_events),
            "single_event_affected": len(correct_events) == 1,
            "multi_event_affected": len(correct_events) >= 2,
            "wrong_recipient_eligible": bool(wrong_eligible.get(speaker)),
            "causal_event_ids_read": [item["event_id"] for item in correct_events],
            "wrong_eligible_event_ids_read": list(wrong_eligible.get(speaker, [])),
            "current_only": stage4_row["current_only"],
            "oracle_evidence_direct": stage4_row["oracle_evidence"],
            "stage5a_absolute_state": stage5a_row["correct_role_state"],
        }
        row["last_event_delta"] = _payload(
            apply_event_deltas(
                anchor,
                correct_events,
                current_turn_id=turn_id,
                read_weight=read_weight,
                decay=decay,
                delta_clip=delta_clip,
                last_only=True,
            ),
            labels,
        )
        row["accumulated_event_delta"] = _payload(
            apply_event_deltas(
                anchor,
                correct_events,
                current_turn_id=turn_id,
                read_weight=read_weight,
                decay=decay,
                delta_clip=delta_clip,
                last_only=False,
            ),
            labels,
        )
        row["random_evidence_delta"] = _payload(
            apply_event_deltas(
                anchor,
                random_events,
                current_turn_id=turn_id,
                read_weight=read_weight,
                decay=decay,
                delta_clip=delta_clip,
                last_only=False,
            ),
            labels,
        )
        row["wrong_recipient_delta"] = _payload(
            apply_event_deltas(
                anchor,
                wrong_events,
                current_turn_id=turn_id,
                read_weight=read_weight,
                decay=decay,
                delta_clip=delta_clip,
                last_only=False,
            ),
            labels,
        )
        current_nll = _gold_nll(row["current_only"], gold, labels)
        delta_nll = _gold_nll(row["accumulated_event_delta"], gold, labels)
        chosen = (
            row["accumulated_event_delta"] if delta_nll < current_nll else row["current_only"]
        )
        row["oracle_current_or_delta"] = dict(chosen)
        row["oracle_used_delta"] = delta_nll < current_nll
        rows.append(row)

        # The observed utterance closes the current external-effect interval.
        # Any event caused by this same turn is written only after prediction.
        for branch_state in states.values():
            branch_state[speaker] = []
        wrong_eligible[speaker] = []
        for event in events_by_turn.get((dialogue_id, turn_id), []):
            recipient = event["recipient"]
            states["correct"][recipient].append(_event_entry(event, "oracle_delta_logits"))
            states["random"][recipient].append(_event_entry(event, "random_delta_logits"))
            wrong_recipient = event.get("wrong_recipient")
            if wrong_recipient:
                states["wrong"][wrong_recipient].append(
                    _event_entry(event, "oracle_delta_logits")
                )
                wrong_eligible[recipient].append(event["event_id"])
    return rows


def _subset(rows: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "all":
        return list(rows)
    if name in {
        "affected_next_turn",
        "single_event_affected",
        "multi_event_affected",
        "wrong_recipient_eligible",
    }:
        return [item for item in rows if item[name]]
    raise ValueError(f"Unknown Stage 5B subset: {name}")


def _parameter_grid(section: dict[str, Any]) -> list[tuple[float, float]]:
    weights = [float(value) for value in section.get("read_weight_grid", [0.05, 0.1, 0.2])]
    decays = [float(value) for value in section.get("decay_grid", [0.5, 0.75, 1.0])]
    grid = [(weight, decay) for weight in weights for decay in decays]
    if not grid or any(not 0 <= weight <= 1 or not 0 <= decay <= 1 for weight, decay in grid):
        raise ValueError("Stage 5B read-weight/decay grids must lie in [0,1]")
    return grid


def crossfit_predictions(
    dialogues: Sequence[dict[str, Any]],
    stage4_records: Sequence[dict[str, Any]],
    stage5a_records: Sequence[dict[str, Any]],
    event_scores: Sequence[dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stage4_by_key = {item["key"]: item for item in stage4_records}
    stage5a_by_key = {item["key"]: item for item in stage5a_records}
    events_by_turn: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in event_scores:
        events_by_turn[(event["dialogue_id"], int(event["event_turn_id"]))].append(event)
    grid = _parameter_grid(section)
    delta_clip = float(section.get("delta_clip", 1.0))
    selected: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    dialogue_ids = [item["dialogue_id"] for item in dialogues]
    for heldout in dialogue_ids:
        training_dialogues = [item for item in dialogues if item["dialogue_id"] != heldout]
        candidates: list[tuple[float, float, float, float, int]] = []
        for read_weight, decay in grid:
            train_rows = [
                row
                for dialogue in training_dialogues
                for row in simulate_dialogue(
                    dialogue,
                    stage4_by_key,
                    stage5a_by_key,
                    events_by_turn,
                    labels,
                    read_weight=read_weight,
                    decay=decay,
                    delta_clip=delta_clip,
                )
            ]
            affected = _subset(train_rows, "affected_next_turn")
            selection_rows = affected if affected else train_rows
            metrics = stage4.metric_bundle(
                selection_rows, "accumulated_event_delta", labels
            )
            # Primary NLL, then macro F1, then the smaller correction and
            # stronger decay as fixed simplicity tie-breakers.
            candidates.append(
                (
                    float(metrics["nll"]),
                    -float(metrics["macro_f1"]),
                    read_weight,
                    decay,
                    len(selection_rows),
                )
            )
        best = min(candidates)
        _, _, read_weight, decay, selection_targets = best
        heldout_dialogue = next(item for item in dialogues if item["dialogue_id"] == heldout)
        heldout_rows = simulate_dialogue(
            heldout_dialogue,
            stage4_by_key,
            stage5a_by_key,
            events_by_turn,
            labels,
            read_weight=read_weight,
            decay=decay,
            delta_clip=delta_clip,
        )
        for row in heldout_rows:
            row["crossfit_heldout_dialogue"] = heldout
            row["selected_read_weight"] = read_weight
            row["selected_decay"] = decay
            row["delta_clip"] = delta_clip
        oof.extend(heldout_rows)
        selected.append(
            {
                "heldout_dialogue": heldout,
                "training_dialogues": sorted(
                    item["dialogue_id"] for item in training_dialogues
                ),
                "selection_targets": selection_targets,
                "selection_branch": "accumulated_event_delta",
                "selected_read_weight": read_weight,
                "selected_decay": decay,
                "delta_clip": delta_clip,
                "selection_nll": best[0],
                "selection_macro_f1": -best[1],
            }
        )
    expected = set(stage4_by_key)
    actual = {item["key"] for item in oof}
    if actual != expected or len(oof) != len(expected):
        raise ValueError("Stage 5B cross-fitted prediction coverage is incomplete")
    return oof, selected


def _comparison(
    rows: Sequence[dict[str, Any]],
    proposed: str,
    control: str,
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        return {"proposed": proposed, "control": control, "targets": 0, "bootstrap": None}
    proposed_metrics = stage4.metric_bundle(rows, proposed, labels)
    control_metrics = stage4.metric_bundle(rows, control, labels)
    return {
        "proposed": proposed,
        "control": control,
        "targets": len(rows),
        "dialogues": len({item["dialogue_id"] for item in rows}),
        "point_advantage": stage4.paired_advantage(proposed_metrics, control_metrics),
        "bootstrap": stage4.paired_dialogue_bootstrap(
            rows, proposed, control, labels, samples, seed
        ),
    }


def _nll_supported(comparison: dict[str, Any], probability: float) -> bool:
    bootstrap = comparison.get("bootstrap")
    return bool(
        bootstrap
        and bootstrap["metrics"]["nll"]["probability_positive"] >= probability
    )


def _markdown(summary: dict[str, Any]) -> str:
    names = {
        "current_only": "5B-0 current only",
        "oracle_evidence_direct": "5B-1 evidence direct",
        "stage5a_absolute_state": "5B-2 Stage 5A absolute state",
        "last_event_delta": "5B-3 last event delta",
        "accumulated_event_delta": "5B-4 accumulated event deltas",
        "random_evidence_delta": "5B-5 random-evidence deltas",
        "wrong_recipient_delta": "5B-6 wrong-recipient deltas",
        "oracle_current_or_delta": "Oracle current/delta choice",
    }
    headings = {
        "all": "All annotated targets",
        "affected_next_turn": "Affected recipient's next utterance",
        "single_event_affected": "Exactly one pending event",
        "multi_event_affected": "Two or more pending events",
        "wrong_recipient_eligible": "Wrong-recipient eligible targets",
    }
    lines = [
        "# Stage 5B — Cross-Fitted Event-Delta Role Memory",
        "",
        "> MELD validation-only. Event effects are paired frozen-model differences, not emotion-delta annotations. Test remains locked.",
        "",
    ]
    for subset_name in EVALUATION_SUBSETS:
        block = summary["subsets"][subset_name]
        lines.extend(
            [
                f"## {headings[subset_name]}",
                "",
                f"Targets/dialogues: **{block['targets']} / {block['dialogues']}**",
                "",
                "| Variant | wF1 | macro F1 | accuracy | NLL | Brier | ECE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for branch in BRANCHES:
            if not block["targets"]:
                break
            metric = block["metrics"][branch]
            lines.append(
                f"| {names[branch]} | {metric['weighted_f1']:.4f} | "
                f"{metric['macro_f1']:.4f} | {metric['accuracy']:.4f} | "
                f"{metric['nll']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |"
            )
        lines.append("")
    lines.extend(["## Pre-registered paired-NLL contrasts", ""])
    for name, comparison in summary["comparisons"].items():
        bootstrap = comparison.get("bootstrap")
        if not bootstrap:
            lines.append(f"- **{name}**: unavailable (zero eligible targets).")
            continue
        nll = bootstrap["metrics"]["nll"]
        macro = bootstrap["metrics"]["macro_f1"]
        lines.append(
            f"- **{name}** ({comparison['targets']} targets): NLL advantage "
            f"{nll['point_advantage']:+.4f} (95% CI {nll['ci95'][0]:+.4f} to "
            f"{nll['ci95'][1]:+.4f}, P>0={nll['probability_positive']:.3f}); "
            f"macro-F1 advantage {macro['point_advantage']:+.4f}."
        )
    lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- Event writes / paired prompts: **{summary['audit']['event_writes']} / {summary['audit']['total_prompts']}**",
            f"- Affected targets: **{summary['audit']['unique_affected_next_turn_targets']}**",
            f"- Single-event / multi-event targets: **{summary['audit']['single_event_targets']} / {summary['audit']['multi_event_targets']}**",
            f"- Wrong-recipient eligible targets: **{summary['audit']['wrong_recipient_eligible_targets']}**",
            f"- Selected read weights: **{summary['selected_parameter_counts']['read_weight']}**",
            f"- Selected decays: **{summary['selected_parameter_counts']['decay']}**",
            f"- Oracle delta-use rate on affected targets: **{summary['oracle_delta_use_rate_affected']:.1%}**",
            "",
            "## Frozen decision",
            "",
            f"Status: **{summary['decision']['status']}**",
            "",
            "Primary advancement checks:",
        ]
    )
    for name, passed in summary["decision"]["primary_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(["", "Secondary mechanism checks (not used to unlock test):"])
    for name, passed in summary["decision"]["secondary_checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
    lines.extend(
        [
            "",
            "The Oracle row uses target gold only to measure headroom and is excluded from selection and claims.",
            "This diagnostic cannot access MELD test.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path) -> dict[str, Any]:
    (
        root,
        config,
        labels,
        dialogues,
        verified,
        _,
        stage5a_section,
        section,
    ) = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["audit"].exists():
        audit(config_path)
    audit_report = _read_json(paths["audit"])
    event_scores = _read_jsonl(paths["pair_scores"])
    specs = build_event_pair_specs(
        dialogues, verified, labels, stage5a_section, section
    )
    if {item.get("event_id") for item in event_scores} != {
        item["event_id"] for item in specs
    }:
        raise ValueError("Run Stage 5B score-pairs before analyze")
    stage4_records = _read_jsonl(_resolve(root, stage5a_section["stage4_score_file"]))
    stage5a_records = _read_jsonl(_resolve(root, section["stage5a_oof_score_file"]))
    oof, parameters = crossfit_predictions(
        dialogues,
        stage4_records,
        stage5a_records,
        event_scores,
        labels,
        section,
    )
    _write_jsonl(paths["oof_scores"], oof)
    subsets: dict[str, Any] = {}
    for subset_name in EVALUATION_SUBSETS:
        rows = _subset(oof, subset_name)
        subsets[subset_name] = {
            "targets": len(rows),
            "dialogues": len({item["dialogue_id"] for item in rows}),
            "metrics": (
                {branch: stage4.metric_bundle(rows, branch, labels) for branch in BRANCHES}
                if rows
                else {}
            ),
        }
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2052))
    affected = _subset(oof, "affected_next_turn")
    multi = _subset(oof, "multi_event_affected")
    wrong = _subset(oof, "wrong_recipient_eligible")
    comparisons = {
        "delta_vs_current_affected": _comparison(
            affected,
            "accumulated_event_delta",
            "current_only",
            labels,
            samples,
            seed,
        ),
        "delta_vs_stage5a_absolute_affected": _comparison(
            affected,
            "accumulated_event_delta",
            "stage5a_absolute_state",
            labels,
            samples,
            seed + 1,
        ),
        "delta_vs_direct_evidence_affected": _comparison(
            affected,
            "accumulated_event_delta",
            "oracle_evidence_direct",
            labels,
            samples,
            seed + 2,
        ),
        "correct_vs_random_delta_affected": _comparison(
            affected,
            "accumulated_event_delta",
            "random_evidence_delta",
            labels,
            samples,
            seed + 3,
        ),
        "accumulated_vs_last_multi_event": _comparison(
            multi,
            "accumulated_event_delta",
            "last_event_delta",
            labels,
            samples,
            seed + 4,
        ),
        "correct_vs_wrong_recipient_eligible": _comparison(
            wrong,
            "accumulated_event_delta",
            "wrong_recipient_delta",
            labels,
            samples,
            seed + 5,
        ),
    }
    probability = float(section.get("minimum_bootstrap_probability", 0.90))
    tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    all_metrics = subsets["all"]["metrics"]
    primary_checks = {
        "annotation_quality_gate": True,
        "crossfit_covers_every_dialogue": len(parameters) == len(dialogues),
        "minimum_affected_targets": len(affected)
        >= int(section.get("minimum_affected_targets", 30)),
        "weighted_f1_preserved_overall": all_metrics["accumulated_event_delta"][
            "weighted_f1"
        ]
        >= all_metrics["current_only"]["weighted_f1"] - tolerance,
        "event_delta_beats_current": _nll_supported(
            comparisons["delta_vs_current_affected"], probability
        ),
        "event_delta_beats_stage5a_absolute": _nll_supported(
            comparisons["delta_vs_stage5a_absolute_affected"], probability
        ),
        "event_delta_beats_direct_evidence": _nll_supported(
            comparisons["delta_vs_direct_evidence_affected"], probability
        ),
        "verified_evidence_beats_random_delta": _nll_supported(
            comparisons["correct_vs_random_delta_affected"], probability
        ),
    }
    secondary_checks = {
        "minimum_multi_event_targets": len(multi)
        >= int(section.get("minimum_multi_event_targets", 5)),
        "accumulation_beats_last_event": _nll_supported(
            comparisons["accumulated_vs_last_multi_event"], probability
        ),
        "minimum_wrong_recipient_targets": len(wrong)
        >= int(section.get("minimum_wrong_recipient_targets", 8)),
        "correct_recipient_beats_wrong_recipient": _nll_supported(
            comparisons["correct_vs_wrong_recipient_eligible"], probability
        ),
    }
    eligible = all(primary_checks.values())
    decision = {
        "status": (
            "advance_to_train_split_delta_router"
            if eligible
            else "stop_and_diagnose_event_delta_memory"
        ),
        "eligible": eligible,
        "primary_endpoint": "paired affected-next-turn NLL",
        "primary_checks": primary_checks,
        "secondary_checks": secondary_checks,
        "secondary_checks_unlock_test": False,
        "selection_basis": (
            "leave-one-complete-dialogue-out NLL selection; fixed paired before/after prompts"
        ),
        "test_locked": True,
        "next_action": (
            "Scale structural annotations on MELD train and learn routing without test access."
            if eligible
            else "Keep MELD test locked; inspect paired deltas and Oracle headroom before scaling."
        ),
    }
    read_counts = Counter(str(item["selected_read_weight"]) for item in parameters)
    decay_counts = Counter(str(item["selected_decay"]) for item in parameters)
    summary = {
        "experiment": "event_delta_role_memory_stage5b",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "emotion_delta_labels_used": False,
        "paired_frozen_model_differences": True,
        "labels": labels,
        "targets": len(oof),
        "dialogues": len(dialogues),
        "audit": audit_report,
        "subsets": subsets,
        "comparisons": comparisons,
        "crossfit_parameters": parameters,
        "selected_parameter_counts": {
            "read_weight": dict(read_counts),
            "decay": dict(decay_counts),
        },
        "oracle_delta_use_rate_affected": float(
            np.mean([bool(item["oracle_used_delta"]) for item in affected])
        ),
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["decision"], decision)
    annotation_path = _resolve(
        root, config["hard_role_oracle_stage4"]["annotation_file"]
    )
    manifest = {
        "experiment": "event_delta_role_memory_stage5b",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "annotation_file_sha256": _sha256(annotation_path),
        "stage4_scores_sha256": _sha256(
            _resolve(root, stage5a_section["stage4_score_file"])
        ),
        "stage5a_oof_sha256": _sha256(
            _resolve(root, section["stage5a_oof_score_file"])
        ),
        "paired_event_scores_sha256": _sha256(paths["pair_scores"]),
        "stage5b_oof_sha256": _sha256(paths["oof_scores"]),
        "causal_invariant": "predict turn t before writes caused by turn t",
        "crossfit_invariant": "held-out dialogue labels never select its parameters",
        "emotion_delta_labels_used": False,
        "oracle_excluded_from_selection": True,
    }
    _write_json(paths["manifest"], manifest)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, _, _, dialogues, verified, _, _, section = _load_context(config_path)
    paths = _paths(root, section)
    required = [
        paths["audit"],
        paths["pair_scores"],
        paths["oof_scores"],
        paths["summary_json"],
        paths["summary_md"],
        paths["decision"],
        paths["manifest"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5B artifact(s): {missing}")
    summary = _read_json(paths["summary_json"])
    pair_scores = _read_jsonl(paths["pair_scores"])
    oof = _read_jsonl(paths["oof_scores"])
    parameters = summary.get("crossfit_parameters", [])
    heldout_ok = all(
        item["heldout_dialogue"] not in item["training_dialogues"] for item in parameters
    )
    finite = all(
        math.isfinite(float(metric[name]))
        for block in summary.get("subsets", {}).values()
        for metric in block.get("metrics", {}).values()
        for name in ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece")
    )
    centered_deltas = all(
        abs(float(np.mean(item[field]))) < 1e-8
        for item in pair_scores
        for field in ("oracle_delta_logits", "random_delta_logits")
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "complete_oof_coverage": len(oof) == len(verified),
        "complete_pair_score_coverage": len(pair_scores)
        == int(summary["audit"]["event_writes"]),
        "all_eight_variants": all(
            branch in summary["subsets"]["all"].get("metrics", {})
            for branch in BRANCHES
        ),
        "finite_metrics": finite,
        "centered_finite_event_deltas": centered_deltas,
        "complete_dialogues": len(parameters) == len(dialogues),
        "heldout_dialogue_excluded_from_selection": heldout_ok,
        "no_emotion_delta_annotations": summary.get("emotion_delta_labels_used") is False,
        "oracle_excluded_from_selection": all(
            item.get("selection_branch") == "accumulated_event_delta"
            for item in parameters
        ),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scientific_decision": summary.get("decision", {}).get("status"),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 5B artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5B event-delta role memory")
    parser.add_argument(
        "stage",
        choices=("check", "audit", "score-pairs", "analyze", "verify", "all-validation"),
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
    elif arguments.stage == "score-pairs":
        score_pairs(arguments.config, overwrite=arguments.overwrite)
    elif arguments.stage == "analyze":
        analyze(arguments.config)
    elif arguments.stage == "verify":
        verify(arguments.config)
    else:
        check(arguments.config)
        audit(arguments.config)
        score_pairs(arguments.config, overwrite=arguments.overwrite)
        analyze(arguments.config)
        verify(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
