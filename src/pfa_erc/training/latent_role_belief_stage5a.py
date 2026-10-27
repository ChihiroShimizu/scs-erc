"""Stage 5A: cross-fitted Oracle-routed latent role-belief diagnostic.

This module deliberately treats per-role emotion as a *predictive latent state*,
not as an annotated or observable psychological quantity.  V2.2 hard labels
provide only causal evidence and recipient routing.  A frozen causal LM scores
an event-conditioned emotion distribution, and that distribution is mixed into
the recipient's previous predictive state.  The state is evaluated when the
recipient next speaks.

The experiment is MELD-validation-only.  This module never opens MELD test.
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

VERSION = "1.5.0"
STATE_BRANCHES = (
    "current_only",
    "oracle_evidence_direct",
    "speaker_state_only",
    "last_event_state",
    "correct_role_state",
    "random_evidence_state",
    "wrong_role_state",
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
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
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
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get("artifact_dir", "artifacts/latent_role_belief_stage5a/meld_validation"),
    )
    return {
        "base": base,
        "pair_records": base / "pairs" / "s5a_event_recipient_next_turn_pairs.jsonl",
        "pair_audit": base / "reports" / "s5a_pair_audit.json",
        "event_scores": base / "scores" / "s5a_event_beliefs_validation.jsonl",
        "oof_scores": base / "scores" / "s5a_crossfit_predictions_validation.jsonl",
        "manifest": base / "run_manifest.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
    }


def _event_id(dialogue_id: str, turn_id: int, recipient: str) -> str:
    material = json.dumps([dialogue_id, int(turn_id), recipient], ensure_ascii=False)
    suffix = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    return f"{dialogue_id}::{turn_id}::{suffix}"


def _deterministic_wrong_recipient(
    roles_seen_before: Sequence[str], source_speaker: str, intended: Sequence[str], key: str, seed: int
) -> str | None:
    candidates = sorted(set(roles_seen_before) - set(intended) - {source_speaker})
    if not candidates:
        return None
    material = f"{seed}|{key}".encode()
    choice = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return candidates[choice % len(candidates)]


def _history_lines(history: Sequence[dict[str, Any]]) -> str:
    if not history:
        return "[none]"
    return "\n".join(
        f"[{item['turn_id']}] {item['speaker']}: {item['text']}" for item in history
    )


def build_event_belief_prompt(
    *,
    evidence: Sequence[dict[str, Any]],
    event_speaker: str,
    event_text: str,
    recipient: str,
    labels: Sequence[str],
    preamble: str,
) -> str:
    """Create a causal event prompt without a future utterance or gold label."""
    return (
        f"{preamble.strip()}\n\n"
        "Causally relevant dialogue evidence before the event:\n"
        f"{_history_lines(evidence)}\n\n"
        "Current interaction event:\n"
        f"{event_speaker}: {event_text}\n\n"
        f"Role whose predictive belief may be affected: {recipient}\n\n"
        "Estimate which emotion would be most plausible for this role immediately after the "
        "event, using only the text above. This is a predictive belief, not a claim about an "
        "unobserved true mental state. "
        f"Return exactly one label from: {', '.join(labels)}.\n"
        "Emotion:"
    )


def extract_event_specs(
    dialogues: Sequence[dict[str, Any]],
    verified: dict[str, dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    random_seed = int(section.get("random_seed", 2041))
    preamble = str(
        section.get(
            "event_prompt_preamble",
            "You maintain a causal predictive belief for each participant in a dialogue. "
            "Do not invent missing context and do not explain.",
        )
    )
    specs: list[dict[str, Any]] = []
    for dialogue in dialogues:
        dialogue_id = dialogue["dialogue_id"]
        utterances = dialogue["utterances"]
        roles_seen: list[str] = []
        for turn_id, utterance in enumerate(utterances):
            key = stage4._key(dialogue_id, turn_id)
            annotation = verified[key]["annotation"]
            history = [
                {"turn_id": index, "speaker": item["speaker"], "text": item["text"]}
                for index, item in enumerate(utterances[:turn_id])
            ]
            by_id = {item["turn_id"]: item for item in history}
            oracle_ids = stage4.oracle_evidence_ids(annotation)
            if any(index not in by_id for index in oracle_ids):
                raise ValueError(f"Non-causal Oracle evidence at {key}: {oracle_ids}")
            random_ids = stage4.deterministic_random_evidence(
                list(by_id), len(oracle_ids), key, random_seed
            )
            intended = [
                item["role"]
                for item in annotation.get("role_assessments", [])
                if item.get("interaction_target") == "yes"
                and item.get("event_memory_write") == "yes"
            ]
            for recipient in intended:
                next_turn = next(
                    (
                        future_id
                        for future_id in range(turn_id + 1, len(utterances))
                        if utterances[future_id]["speaker"] == recipient
                    ),
                    None,
                )
                event_id = _event_id(dialogue_id, turn_id, recipient)
                wrong_recipient = _deterministic_wrong_recipient(
                    roles_seen,
                    utterance["speaker"],
                    intended,
                    event_id,
                    random_seed,
                )
                oracle_evidence = [by_id[index] for index in oracle_ids]
                random_evidence = [by_id[index] for index in random_ids]
                specs.append(
                    {
                        "event_id": event_id,
                        "dialogue_id": dialogue_id,
                        "event_turn_id": turn_id,
                        "event_key": key,
                        "event_speaker": utterance["speaker"],
                        "event_text": utterance["text"],
                        "recipient": recipient,
                        "next_recipient_turn_id": next_turn,
                        "next_recipient_key": (
                            stage4._key(dialogue_id, next_turn) if next_turn is not None else None
                        ),
                        "turn_gap": next_turn - turn_id if next_turn is not None else None,
                        "wrong_recipient": wrong_recipient,
                        "wrong_recipient_available": wrong_recipient is not None,
                        "oracle_evidence_ids": oracle_ids,
                        "random_evidence_ids": random_ids,
                        "oracle_prompt": build_event_belief_prompt(
                            evidence=oracle_evidence,
                            event_speaker=utterance["speaker"],
                            event_text=utterance["text"],
                            recipient=recipient,
                            labels=labels,
                            preamble=preamble,
                        ),
                        "random_prompt": build_event_belief_prompt(
                            evidence=random_evidence,
                            event_speaker=utterance["speaker"],
                            event_text=utterance["text"],
                            recipient=recipient,
                            labels=labels,
                            preamble=preamble,
                        ),
                    }
                )
            if utterance["speaker"] not in roles_seen:
                roles_seen.append(utterance["speaker"])
    if len({item["event_id"] for item in specs}) != len(specs):
        raise ValueError("Duplicate Stage 5A event IDs")
    return specs


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
]:
    config = _read_yaml(config_path)
    section = config.get("latent_role_belief_stage5a", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 5A is validation-only; refusing any non-validation split")
    root, _, labels, dialogues, verified, quality = stage4._load_context(config_path)
    stage4_score_path = _resolve(root, section["stage4_score_file"])
    if not stage4_score_path.exists():
        raise FileNotFoundError(
            f"Missing Stage 4 score file: {stage4_score_path}. Run Stage 4 first."
        )
    stage4_records = _read_jsonl(stage4_score_path)
    expected = set(verified)
    actual = {item.get("key") for item in stage4_records}
    if actual != expected or len(stage4_records) != len(expected):
        raise ValueError(
            f"Stage 4 score coverage mismatch: {len(actual)} unique/{len(stage4_records)} rows, "
            f"expected {len(expected)}"
        )
    return root, config, labels, dialogues, verified, quality, section


def check(config_path: Path) -> dict[str, Any]:
    root, _, labels, dialogues, verified, quality, section = _load_context(config_path)
    result = {
        "status": "ready",
        "experiment": "latent_role_belief_stage5a",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "model_name": section["model_name"],
        "targets": len(verified),
        "complete_dialogues": len(dialogues),
        "labels": labels,
        "annotation_quality_status": quality.get("status"),
        "stage4_score_file": str(_resolve(root, section["stage4_score_file"])),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def audit_pairs(config_path: Path) -> dict[str, Any]:
    root, _, labels, dialogues, verified, _, section = _load_context(config_path)
    paths = _paths(root, section)
    specs = extract_event_specs(dialogues, verified, labels, section)
    pair_rows = [
        {key: value for key, value in item.items() if not key.endswith("_prompt")}
        for item in specs
        if item["next_recipient_turn_id"] is not None
    ]
    _write_jsonl(paths["pair_records"], pair_rows)
    gaps = [int(item["turn_gap"]) for item in pair_rows]
    wrong_pairs = [item for item in pair_rows if item["wrong_recipient_available"]]
    audit = {
        "status": "passed",
        "experiment": "latent_role_belief_stage5a",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "complete_dialogues": len(dialogues),
        "targets": len(verified),
        "event_writes": len(specs),
        "observable_event_recipient_pairs": len(pair_rows),
        "unique_observable_targets": len(
            {item["next_recipient_key"] for item in pair_rows}
        ),
        "wrong_recipient_eligible_pairs": len(wrong_pairs),
        "wrong_recipient_coverage_among_observable": len(wrong_pairs) / max(len(pair_rows), 1),
        "turn_gap": {
            "mean": float(np.mean(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "max": max(gaps) if gaps else None,
        },
        "pair_file": str(paths["pair_records"]),
    }
    _write_json(paths["pair_audit"], audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return audit


def _make_scorer(section: dict[str, Any], labels: Sequence[str]) -> Any:
    from pfa_erc.training.llm_context_utility_stage1 import CausalLabelScorer

    return CausalLabelScorer(section, labels)


def score_events(config_path: Path, overwrite: bool = False) -> dict[str, Any]:
    root, _, labels, dialogues, verified, _, section = _load_context(config_path)
    paths = _paths(root, section)
    specs = extract_event_specs(dialogues, verified, labels, section)
    if overwrite and paths["event_scores"].exists():
        paths["event_scores"].unlink()
    existing = _read_jsonl(paths["event_scores"])
    completed = {item["event_id"] for item in existing}
    pending = [item for item in specs if item["event_id"] not in completed]
    if pending:
        scorer = _make_scorer(section, labels)
        temperature = float(section.get("shared_temperature", 1.0))
        batch_events = int(section.get("event_checkpoint_batch", 16))
        for start in range(0, len(pending), batch_events):
            chunk = pending[start : start + batch_events]
            prompts = [
                prompt
                for item in chunk
                for prompt in (item["oracle_prompt"], item["random_prompt"])
            ]
            scores = scorer.score_prompts(prompts)
            rows: list[dict[str, Any]] = []
            cursor = 0
            for item in chunk:
                row = {key: value for key, value in item.items() if not key.endswith("_prompt")}
                row["oracle_event_belief"] = stage4._probability_payload(
                    scores[cursor], labels, temperature
                )
                row["random_event_belief"] = stage4._probability_payload(
                    scores[cursor + 1], labels, temperature
                )
                cursor += 2
                rows.append(row)
            _append_jsonl(paths["event_scores"], rows)
            print(
                f"[Stage 5A] event beliefs {min(start + len(chunk), len(pending))}/"
                f"{len(pending)} pending ({len(completed) + start + len(chunk)}/{len(specs)} total)",
                flush=True,
            )
    records = _read_jsonl(paths["event_scores"])
    actual = {item["event_id"] for item in records}
    expected = {item["event_id"] for item in specs}
    if actual != expected or len(records) != len(expected):
        raise ValueError(
            f"Event-score coverage mismatch: {len(actual)} unique/{len(records)} rows, "
            f"expected {len(expected)}"
        )
    result = {
        "status": "complete",
        "events": len(records),
        "event_score_file": str(paths["event_scores"]),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _normalize(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("Probability vector must be finite, nonnegative, and one-dimensional")
    total = float(array.sum())
    if total <= 0:
        raise ValueError("Probability vector has zero mass")
    return array / total


def update_state(
    previous: Sequence[float] | None, event_belief: Sequence[float], weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("State update weight must be in [0, 1]")
    event = _normalize(event_belief)
    prior = np.full_like(event, 1.0 / len(event)) if previous is None else _normalize(previous)
    return _normalize((1.0 - weight) * prior + weight * event)


def fuse_anchor_state(
    anchor: Sequence[float], state: Sequence[float] | None, weight: float
) -> np.ndarray:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("State read weight must be in [0, 1]")
    anchor_array = _normalize(anchor)
    if state is None or weight == 0.0:
        return anchor_array
    state_array = _normalize(state)
    log_values = (1.0 - weight) * np.log(np.clip(anchor_array, 1e-12, 1.0))
    log_values += weight * np.log(np.clip(state_array, 1e-12, 1.0))
    log_values -= np.max(log_values)
    return _normalize(np.exp(log_values))


def _payload(probabilities: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    values = _normalize(probabilities)
    index = int(np.argmax(values))
    return {
        "probabilities": values.tolist(),
        "prediction": labels[index],
        "confidence": float(values[index]),
    }


def simulate_dialogue(
    dialogue: dict[str, Any],
    stage4_by_key: dict[str, dict[str, Any]],
    events_by_turn: dict[tuple[str, int], list[dict[str, Any]]],
    labels: Sequence[str],
    update_weight: float,
    read_weight: float,
) -> list[dict[str, Any]]:
    """Run one causal trajectory; the current event is applied after prediction."""
    states: dict[str, dict[str, np.ndarray]] = {
        name: {}
        for name in (
            "speaker_state_only",
            "last_event_state",
            "correct_role_state",
            "random_evidence_state",
            "wrong_role_state",
        )
    }
    pending_correct: dict[str, list[str]] = defaultdict(list)
    pending_wrong_eligible: dict[str, list[str]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    dialogue_id = dialogue["dialogue_id"]
    for turn_id, utterance in enumerate(dialogue["utterances"]):
        key = stage4._key(dialogue_id, turn_id)
        source = stage4_by_key[key]
        speaker = utterance["speaker"]
        anchor = source["current_only"]["probabilities"]
        row: dict[str, Any] = {
            "key": key,
            "dialogue_id": dialogue_id,
            "turn_id": turn_id,
            "speaker": speaker,
            "gold": utterance["emotion"],
            "affected_next_turn": bool(pending_correct.get(speaker)),
            "wrong_recipient_eligible": bool(pending_wrong_eligible.get(speaker)),
            "causal_event_ids_read": list(pending_correct.get(speaker, [])),
            "wrong_eligible_event_ids_read": list(pending_wrong_eligible.get(speaker, [])),
        }
        row["current_only"] = source["current_only"]
        row["oracle_evidence_direct"] = source["oracle_evidence"]
        for branch in (
            "speaker_state_only",
            "last_event_state",
            "correct_role_state",
            "random_evidence_state",
            "wrong_role_state",
        ):
            combined = fuse_anchor_state(anchor, states[branch].get(speaker), read_weight)
            row[branch] = _payload(combined, labels)
        rows.append(row)

        # The target speaker's own current utterance anchors their state only
        # after the current prediction has been created.
        anchor_array = _normalize(anchor)
        for branch_state in states.values():
            branch_state[speaker] = anchor_array.copy()
        pending_correct[speaker] = []
        pending_wrong_eligible[speaker] = []

        # External event updates are also applied only after predicting turn t.
        for event in events_by_turn.get((dialogue_id, turn_id), []):
            recipient = event["recipient"]
            oracle_belief = event["oracle_event_belief"]["probabilities"]
            random_belief = event["random_event_belief"]["probabilities"]
            states["correct_role_state"][recipient] = update_state(
                states["correct_role_state"].get(recipient), oracle_belief, update_weight
            )
            # This matched control retains the role's most recent self anchor but
            # discards any earlier external events.  It isolates accumulation
            # from merely seeing the last routed event.
            states["last_event_state"][recipient] = update_state(
                states["speaker_state_only"].get(recipient),
                oracle_belief,
                update_weight,
            )
            states["random_evidence_state"][recipient] = update_state(
                states["random_evidence_state"].get(recipient), random_belief, update_weight
            )
            pending_correct[recipient].append(event["event_id"])
            wrong_recipient = event.get("wrong_recipient")
            if wrong_recipient:
                states["wrong_role_state"][wrong_recipient] = update_state(
                    states["wrong_role_state"].get(wrong_recipient),
                    oracle_belief,
                    update_weight,
                )
                pending_wrong_eligible[recipient].append(event["event_id"])
    return rows


def _subset(rows: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    if name == "all":
        return list(rows)
    if name == "affected_next_turn":
        return [item for item in rows if item["affected_next_turn"]]
    if name == "wrong_recipient_eligible":
        return [item for item in rows if item["wrong_recipient_eligible"]]
    raise ValueError(f"Unknown Stage 5A subset: {name}")


def _parameter_grid(section: dict[str, Any]) -> list[tuple[float, float]]:
    updates = [float(value) for value in section.get("update_weight_grid", [0.25, 0.5, 0.75])]
    reads = [float(value) for value in section.get("read_weight_grid", [0.1, 0.25, 0.5])]
    grid = [(left, right) for left in updates for right in reads]
    if not grid or any(not 0 <= left <= 1 or not 0 <= right <= 1 for left, right in grid):
        raise ValueError("Stage 5A update/read grids must be nonempty and lie in [0,1]")
    return grid


def crossfit_predictions(
    dialogues: Sequence[dict[str, Any]],
    stage4_records: Sequence[dict[str, Any]],
    event_scores: Sequence[dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Leave one complete dialogue out; never select parameters on its labels."""
    stage4_by_key = {item["key"]: item for item in stage4_records}
    events_by_turn: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in event_scores:
        events_by_turn[(event["dialogue_id"], int(event["event_turn_id"]))].append(event)
    grid = _parameter_grid(section)
    dialogue_ids = [item["dialogue_id"] for item in dialogues]
    selected: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    for heldout in dialogue_ids:
        training_dialogues = [item for item in dialogues if item["dialogue_id"] != heldout]
        candidates: list[tuple[float, float, float, float, int]] = []
        for update_weight, read_weight in grid:
            train_rows = [
                row
                for dialogue in training_dialogues
                for row in simulate_dialogue(
                    dialogue,
                    stage4_by_key,
                    events_by_turn,
                    labels,
                    update_weight,
                    read_weight,
                )
            ]
            affected = _subset(train_rows, "affected_next_turn")
            selection_rows = affected if affected else train_rows
            metrics = stage4.metric_bundle(selection_rows, "correct_role_state", labels)
            candidates.append(
                (
                    float(metrics["nll"]),
                    -float(metrics["macro_f1"]),
                    update_weight,
                    read_weight,
                    len(selection_rows),
                )
            )
        best = min(candidates)
        _, _, update_weight, read_weight, selection_targets = best
        heldout_dialogue = next(item for item in dialogues if item["dialogue_id"] == heldout)
        heldout_rows = simulate_dialogue(
            heldout_dialogue,
            stage4_by_key,
            events_by_turn,
            labels,
            update_weight,
            read_weight,
        )
        for row in heldout_rows:
            row["crossfit_heldout_dialogue"] = heldout
            row["selected_update_weight"] = update_weight
            row["selected_read_weight"] = read_weight
        oof.extend(heldout_rows)
        selected.append(
            {
                "heldout_dialogue": heldout,
                "training_dialogues": sorted(
                    item["dialogue_id"] for item in training_dialogues
                ),
                "selection_targets": selection_targets,
                "selected_update_weight": update_weight,
                "selected_read_weight": read_weight,
                "selection_nll": best[0],
                "selection_macro_f1": -best[1],
            }
        )
    if {item["key"] for item in oof} != set(stage4_by_key) or len(oof) != len(stage4_by_key):
        raise ValueError("Cross-fitted prediction coverage is incomplete")
    return oof, selected


def _bootstrap_safe(
    rows: Sequence[dict[str, Any]],
    proposed: str,
    control: str,
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if not rows:
        return None
    return stage4.paired_dialogue_bootstrap(
        rows, proposed, control, labels, samples, seed
    )


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
        "bootstrap": _bootstrap_safe(rows, proposed, control, labels, samples, seed),
    }


def _supported(comparison: dict[str, Any], probability: float) -> bool:
    """Apply the frozen primary endpoint: paired NLL advantage.

    F1 remains a secondary descriptive outcome.  Using a single primary
    endpoint prevents choosing whichever metric happens to pass afterward.
    """
    bootstrap = comparison.get("bootstrap")
    if not bootstrap:
        return False
    return bootstrap["metrics"]["nll"]["probability_positive"] >= probability


def _markdown(summary: dict[str, Any]) -> str:
    names = {
        "current_only": "5A-0 current only",
        "oracle_evidence_direct": "5A-1 evidence direct",
        "speaker_state_only": "5A-2 speaker persistence",
        "last_event_state": "5A-3 last routed event only",
        "correct_role_state": "5A-4 accumulated correct-role state",
        "random_evidence_state": "5A-5 random-evidence state",
        "wrong_role_state": "5A-6 wrong-recipient state",
    }
    lines = [
        "# Stage 5A — Cross-Fitted Latent Role-Belief Oracle",
        "",
        "> MELD validation-only diagnostic. Emotion-state deltas are latent and are not human/LLM labels. MELD test remains locked.",
        "",
    ]
    for subset_name, heading in (
        ("all", "All annotated targets"),
        ("affected_next_turn", "Affected recipient's next utterance"),
        ("wrong_recipient_eligible", "Wrong-recipient eligible next utterance"),
    ):
        block = summary["subsets"][subset_name]
        lines.extend(
            [
                f"## {heading}",
                "",
                f"Targets/dialogues: **{block['targets']} / {block['dialogues']}**",
                "",
                "| Variant | wF1 | macro F1 | accuracy | NLL | Brier | ECE |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if block["targets"]:
            for branch in STATE_BRANCHES:
                metric = block["metrics"][branch]
                lines.append(
                    f"| {names[branch]} | {metric['weighted_f1']:.4f} | "
                    f"{metric['macro_f1']:.4f} | {metric['accuracy']:.4f} | "
                    f"{metric['nll']:.4f} | {metric['brier']:.4f} | {metric['ece']:.4f} |"
                )
        lines.append("")
    lines.extend(["## Pre-registered mechanism contrasts", ""])
    for name, item in summary["comparisons"].items():
        if not item.get("bootstrap"):
            lines.append(f"- **{name}**: unavailable (no eligible targets).")
            continue
        macro = item["bootstrap"]["metrics"]["macro_f1"]
        nll = item["bootstrap"]["metrics"]["nll"]
        lines.append(
            f"- **{name}** ({item['targets']} targets): macro advantage "
            f"{macro['point_advantage']:+.4f} (95% CI {macro['ci95'][0]:+.4f} to "
            f"{macro['ci95'][1]:+.4f}, P>0={macro['probability_positive']:.3f}); "
            f"NLL advantage {nll['point_advantage']:+.4f} "
            f"(P>0={nll['probability_positive']:.3f})."
        )
    audit = summary["pair_audit"]
    lines.extend(
        [
            "",
            "## Causal-pair and cross-fitting diagnostics",
            "",
            f"- Event writes: **{audit['event_writes']}**",
            f"- Observable event→recipient-next-turn pairs: **{audit['observable_event_recipient_pairs']}**",
            f"- Unique affected next-turn targets: **{audit['unique_observable_targets']}**",
            f"- Wrong-recipient eligible pairs: **{audit['wrong_recipient_eligible_pairs']}**",
            f"- Cross-fitted held-out dialogues: **{len(summary['crossfit_parameters'])}**",
            f"- Selected update weights: **{summary['selected_parameter_counts']['update_weight']}**",
            f"- Selected read weights: **{summary['selected_parameter_counts']['read_weight']}**",
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
            "This is a cross-fitted Oracle diagnostic, not a deployable model and not a test-set claim.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(config_path: Path) -> dict[str, Any]:
    root, _, labels, dialogues, verified, _, section = _load_context(config_path)
    paths = _paths(root, section)
    if not paths["pair_audit"].exists():
        audit_pairs(config_path)
    pair_audit = _read_json(paths["pair_audit"])
    event_scores = _read_jsonl(paths["event_scores"])
    specs = extract_event_specs(dialogues, verified, labels, section)
    if {item["event_id"] for item in event_scores} != {item["event_id"] for item in specs}:
        raise ValueError("Run score-events before Stage 5A analysis")
    stage4_records = _read_jsonl(_resolve(root, section["stage4_score_file"]))
    oof, crossfit_parameters = crossfit_predictions(
        dialogues, stage4_records, event_scores, labels, section
    )
    _write_jsonl(paths["oof_scores"], oof)
    subsets: dict[str, Any] = {}
    for subset_name in ("all", "affected_next_turn", "wrong_recipient_eligible"):
        rows = _subset(oof, subset_name)
        subsets[subset_name] = {
            "targets": len(rows),
            "dialogues": len({item["dialogue_id"] for item in rows}),
            "metrics": (
                {branch: stage4.metric_bundle(rows, branch, labels) for branch in STATE_BRANCHES}
                if rows
                else {}
            ),
        }
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2042))
    comparisons = {
        "correct_state_vs_current_affected": _comparison(
            _subset(oof, "affected_next_turn"),
            "correct_role_state",
            "current_only",
            labels,
            samples,
            seed,
        ),
        "correct_state_vs_evidence_direct_affected": _comparison(
            _subset(oof, "affected_next_turn"),
            "correct_role_state",
            "oracle_evidence_direct",
            labels,
            samples,
            seed + 1,
        ),
        "correct_state_vs_speaker_state_affected": _comparison(
            _subset(oof, "affected_next_turn"),
            "correct_role_state",
            "speaker_state_only",
            labels,
            samples,
            seed + 2,
        ),
        "accumulated_state_vs_last_event_affected": _comparison(
            _subset(oof, "affected_next_turn"),
            "correct_role_state",
            "last_event_state",
            labels,
            samples,
            seed + 3,
        ),
        "correct_state_vs_random_state_affected": _comparison(
            _subset(oof, "affected_next_turn"),
            "correct_role_state",
            "random_evidence_state",
            labels,
            samples,
            seed + 4,
        ),
        "correct_state_vs_wrong_state_eligible": _comparison(
            _subset(oof, "wrong_recipient_eligible"),
            "correct_role_state",
            "wrong_role_state",
            labels,
            samples,
            seed + 5,
        ),
    }
    update_counts = Counter(str(item["selected_update_weight"]) for item in crossfit_parameters)
    read_counts = Counter(str(item["selected_read_weight"]) for item in crossfit_parameters)
    probability = float(section.get("minimum_bootstrap_probability", 0.90))
    tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    all_metrics = subsets["all"]["metrics"]
    checks = {
        "annotation_quality_gate": True,
        "crossfit_covers_every_dialogue": len(crossfit_parameters) == len(dialogues),
        "observable_pair_count": pair_audit["observable_event_recipient_pairs"]
        >= int(section.get("minimum_observable_pairs", 15)),
        "wrong_recipient_pair_count": pair_audit["wrong_recipient_eligible_pairs"]
        >= int(section.get("minimum_wrong_recipient_pairs", 8)),
        "weighted_f1_preserved_overall": all_metrics["correct_role_state"]["weighted_f1"]
        >= all_metrics["current_only"]["weighted_f1"] - tolerance,
        "correct_state_beats_current_on_affected": _supported(
            comparisons["correct_state_vs_current_affected"], probability
        ),
        "correct_state_adds_over_direct_evidence": _supported(
            comparisons["correct_state_vs_evidence_direct_affected"], probability
        ),
        "correct_state_adds_over_speaker_persistence": _supported(
            comparisons["correct_state_vs_speaker_state_affected"], probability
        ),
        "accumulation_adds_over_last_event": _supported(
            comparisons["accumulated_state_vs_last_event_affected"], probability
        ),
        "correct_evidence_beats_random_state": _supported(
            comparisons["correct_state_vs_random_state_affected"], probability
        ),
        "correct_recipient_beats_wrong_recipient": _supported(
            comparisons["correct_state_vs_wrong_state_eligible"], probability
        ),
    }
    eligible = all(checks.values())
    decision = {
        "status": (
            "advance_to_train_split_role_router"
            if eligible
            else "stop_and_diagnose_latent_role_belief"
        ),
        "eligible": eligible,
        "checks": checks,
        "selection_basis": (
            "leave-one-complete-dialogue-out parameter selection; fixed V2.2 Oracle routing"
        ),
        "test_locked": True,
        "next_action": (
            "Annotate/train a structural recipient router on MELD train only."
            if eligible
            else "Keep MELD test locked; diagnose state update and eligible causal pairs before scaling."
        ),
    }
    summary = {
        "experiment": "latent_role_belief_stage5a",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "latent_state_not_annotation": True,
        "labels": labels,
        "targets": len(oof),
        "dialogues": len(dialogues),
        "pair_audit": pair_audit,
        "subsets": subsets,
        "comparisons": comparisons,
        "crossfit_parameters": crossfit_parameters,
        "selected_parameter_counts": {
            "update_weight": dict(update_counts),
            "read_weight": dict(read_counts),
        },
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["decision"], decision)
    manifest = {
        "experiment": "latent_role_belief_stage5a",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "annotation_file_sha256": _sha256(
            _resolve(
                root,
                _read_yaml(config_path)["hard_role_oracle_stage4"]["annotation_file"],
            )
        ),
        "stage4_scores_sha256": _sha256(_resolve(root, section["stage4_score_file"])),
        "event_scores_sha256": _sha256(paths["event_scores"]),
        "oof_scores_sha256": _sha256(paths["oof_scores"]),
        "causal_invariant": "predict turn t before any write caused by turn t",
        "crossfit_invariant": "held-out dialogue labels never select its update/read weights",
        "emotion_delta_labels_used": False,
    }
    _write_json(paths["manifest"], manifest)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, _, _, dialogues, verified, _, section = _load_context(config_path)
    paths = _paths(root, section)
    required = [
        paths["pair_records"],
        paths["pair_audit"],
        paths["event_scores"],
        paths["oof_scores"],
        paths["manifest"],
        paths["summary_json"],
        paths["summary_md"],
        paths["decision"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 5A artifact(s): {missing}")
    summary = _read_json(paths["summary_json"])
    oof = _read_jsonl(paths["oof_scores"])
    event_scores = _read_jsonl(paths["event_scores"])
    heldout_ok = all(
        item["heldout_dialogue"] not in item["training_dialogues"]
        for item in summary.get("crossfit_parameters", [])
    )
    finite = all(
        math.isfinite(float(metric[name]))
        for block in summary.get("subsets", {}).values()
        for metric in block.get("metrics", {}).values()
        for name in ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece")
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "complete_oof_coverage": len(oof) == len(verified),
        "complete_event_score_coverage": len(event_scores)
        == int(summary["pair_audit"]["event_writes"]),
        "all_seven_variants": all(
            branch in summary["subsets"]["all"].get("metrics", {})
            for branch in STATE_BRANCHES
        ),
        "finite_metrics": finite,
        "complete_dialogues": len(summary.get("crossfit_parameters", [])) == len(dialogues),
        "heldout_dialogue_excluded_from_selection": heldout_ok,
        "no_emotion_delta_annotations": summary.get("latent_state_not_annotation") is True,
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scientific_decision": summary.get("decision", {}).get("status"),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 5A artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 5A latent role-belief Oracle")
    parser.add_argument(
        "stage",
        choices=("check", "audit-pairs", "score-events", "analyze", "verify", "all-validation"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "check":
        check(args.config)
    elif args.stage == "audit-pairs":
        audit_pairs(args.config)
    elif args.stage == "score-events":
        score_events(args.config, overwrite=args.overwrite)
    elif args.stage == "analyze":
        analyze(args.config)
    elif args.stage == "verify":
        verify(args.config)
    else:
        check(args.config)
        audit_pairs(args.config)
        score_events(args.config, overwrite=args.overwrite)
        analyze(args.config)
        verify(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
