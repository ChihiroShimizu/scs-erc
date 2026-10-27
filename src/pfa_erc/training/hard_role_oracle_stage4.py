"""Stage 4: validation-only hard-label Oracle for selective role memory.

This diagnostic answers three deliberately narrow questions before another
learned controller is built:

1. Do verified causal evidence turns outperform count-matched random turns?
2. Does writing concrete events to the correct recipient outperform writing the
   same events to an incorrect recipient?
3. Does causal role memory add value beyond selected evidence alone?

The MELD test split is inaccessible by construction.  Predictions for turn t
are built before applying any memory write caused by turn t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score


VERSION = "1.4.1"
BRANCHES = (
    "current_only",
    "full_history",
    "oracle_evidence",
    "oracle_role_memory",
    "random_evidence",
    "wrong_recipient",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(record)
    return records


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _key(dialogue_id: str, turn_id: int) -> str:
    return f"{dialogue_id}::{turn_id}"


def deterministic_random_evidence(
    history_ids: Sequence[int],
    count: int,
    target_key: str,
    seed: int,
) -> list[int]:
    """Draw a reproducible count-matched evidence set without Python hash()."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if count > len(history_ids):
        raise ValueError("count cannot exceed available history")
    material = f"{seed}|{target_key}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    rng = random.Random(local_seed)
    return sorted(rng.sample(list(history_ids), count))


def oracle_evidence_ids(annotation: dict[str, Any]) -> list[int]:
    """Union all verified hard causal evidence; no future/gold information."""
    values: list[int] = []
    context = annotation.get("context_utility", {})
    for field in ("speaker_emotion", "recipient_resolution"):
        values.extend(context.get(field, {}).get("evidence_turn_ids", []))
    values.extend(annotation.get("discourse_parent_ids", []))
    for role in annotation.get("role_assessments", []):
        values.extend(role.get("history_evidence_turn_ids", []))
    return sorted(set(int(value) for value in values))


def apply_causal_writes(
    memory: dict[str, list[dict[str, Any]]],
    annotation: dict[str, Any],
    source_speaker: str,
    source_text: str,
    turn_id: int,
    max_entries: int,
    *,
    wrong_recipient: bool,
    roles_seen_before: Sequence[str],
    target_key: str,
    random_seed: int,
) -> dict[str, int]:
    """Apply writes after prediction and return attempted/applied diagnostics.

    In the wrong-recipient control the same grounded event is written to a
    deterministic, already-seen role that is neither the speaker nor any
    intended recipient.  If no such role exists the write is skipped.
    """
    intended = [
        role["role"]
        for role in annotation.get("role_assessments", [])
        if role.get("interaction_target") == "yes"
        and role.get("event_memory_write") == "yes"
    ]
    excluded = set(intended) | {source_speaker}
    candidates = sorted(set(roles_seen_before) - excluded)
    attempted = len(intended)
    applied = 0
    for index, recipient in enumerate(intended):
        actual_recipient = recipient
        if wrong_recipient:
            if not candidates:
                continue
            material = f"{random_seed}|{target_key}|{index}".encode("utf-8")
            choice = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
            actual_recipient = candidates[choice % len(candidates)]
        entry = {
            "source_turn_id": int(turn_id),
            "source_speaker": source_speaker,
            "text": source_text,
            "intended_recipient": recipient,
            "stored_for_role": actual_recipient,
        }
        memory.setdefault(actual_recipient, []).append(entry)
        if max_entries > 0:
            memory[actual_recipient] = memory[actual_recipient][-max_entries:]
        applied += 1
    return {"attempted": attempted, "applied": applied}


def _history_lines(history: Sequence[dict[str, Any]]) -> str:
    if not history:
        return "[none]"
    return "\n".join(
        f"[{turn['turn_id']}] {turn['speaker']}: {turn['text']}" for turn in history
    )


def build_prompt(
    target: dict[str, Any],
    evidence: Sequence[dict[str, Any]],
    memory: Sequence[dict[str, Any]],
    labels: Sequence[str],
    preamble: str,
) -> str:
    memory_text = "[none]"
    if memory:
        memory_text = "\n".join(
            (
                f"[grounded memory from turn {item['source_turn_id']}] "
                f"{item['source_speaker']}: {item['text']}"
            )
            for item in memory
        )
    return (
        f"{preamble.strip()}\n\n"
        "Selected earlier evidence (causal turns only):\n"
        f"{_history_lines(evidence)}\n\n"
        f"Grounded event memory for the target speaker {target['speaker']}:\n"
        f"{memory_text}\n\n"
        "Target utterance:\n"
        f"{target['speaker']}: {target['text']}\n\n"
        "Classify the target speaker's emotion using only the text above. "
        f"Return exactly one label from: {', '.join(labels)}.\n"
        "Emotion:"
    )


def build_dialogue_plans(
    dialogue: dict[str, Any],
    verified: dict[str, dict[str, Any]],
    labels: Sequence[str],
    section: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build all causal prompts for one complete annotated dialogue."""
    utterances = dialogue["utterances"]
    max_history = int(section.get("max_history_turns", 12))
    max_memory = int(section.get("max_memory_entries_per_role", 3))
    random_seed = int(section.get("random_seed", 2033))
    preamble = str(
        section.get(
            "prompt_preamble",
            "You are a careful dialogue emotion classifier. Do not invent missing context and do not explain.",
        )
    )
    memory: dict[str, list[dict[str, Any]]] = {}
    wrong_memory: dict[str, list[dict[str, Any]]] = {}
    roles_seen: list[str] = []
    plans: list[dict[str, Any]] = []

    for turn_id, utterance in enumerate(utterances):
        key = _key(dialogue["dialogue_id"], turn_id)
        if key not in verified:
            raise ValueError(f"Incomplete annotation trajectory: missing {key}")
        verified_record = verified[key]
        source = verified_record["input"]
        annotation = verified_record["annotation"]
        target = source["target"]
        if target.get("speaker") != utterance.get("speaker") or target.get("text") != utterance.get("text"):
            raise ValueError(f"Annotation/source mismatch at {key}")
        history = [
            {
                "turn_id": index,
                "speaker": prior["speaker"],
                "text": prior["text"],
            }
            for index, prior in enumerate(utterances[:turn_id])
        ]
        by_id = {item["turn_id"]: item for item in history}
        oracle_ids = oracle_evidence_ids(annotation)
        if any(value not in by_id for value in oracle_ids):
            raise ValueError(f"Annotation cites non-causal evidence at {key}: {oracle_ids}")
        random_ids = deterministic_random_evidence(
            list(by_id), len(oracle_ids), key, random_seed
        )
        full_ids = list(by_id)[-max_history:] if max_history > 0 else list(by_id)
        target_speaker = target["speaker"]

        evidence_sets = {
            "current_only": [],
            "full_history": [by_id[value] for value in full_ids],
            "oracle_evidence": [by_id[value] for value in oracle_ids],
            "oracle_role_memory": [by_id[value] for value in oracle_ids],
            "random_evidence": [by_id[value] for value in random_ids],
            "wrong_recipient": [by_id[value] for value in oracle_ids],
        }
        memory_sets = {
            "current_only": [],
            "full_history": [],
            "oracle_evidence": [],
            "oracle_role_memory": list(memory.get(target_speaker, [])),
            "random_evidence": [],
            "wrong_recipient": list(wrong_memory.get(target_speaker, [])),
        }
        prompts = {
            branch: build_prompt(target, evidence_sets[branch], memory_sets[branch], labels, preamble)
            for branch in BRANCHES
        }

        # Causality invariant: the current turn is written only after every
        # prediction prompt for this turn has been constructed.
        correct_write = apply_causal_writes(
            memory,
            annotation,
            target_speaker,
            target["text"],
            turn_id,
            max_memory,
            wrong_recipient=False,
            roles_seen_before=roles_seen,
            target_key=key,
            random_seed=random_seed,
        )
        wrong_write = apply_causal_writes(
            wrong_memory,
            annotation,
            target_speaker,
            target["text"],
            turn_id,
            max_memory,
            wrong_recipient=True,
            roles_seen_before=roles_seen,
            target_key=key,
            random_seed=random_seed,
        )
        if target_speaker not in roles_seen:
            roles_seen.append(target_speaker)
        plans.append(
            {
                "key": key,
                "dialogue_id": dialogue["dialogue_id"],
                "turn_id": turn_id,
                "utterance_id": str(utterance.get("utterance_id", turn_id)),
                "speaker": target_speaker,
                "gold": utterance["emotion"],
                "history_count": len(history),
                "oracle_evidence_ids": oracle_ids,
                "random_evidence_ids": random_ids,
                "role_memory_read_count": len(memory_sets["oracle_role_memory"]),
                "wrong_memory_read_count": len(memory_sets["wrong_recipient"]),
                "correct_write_attempts": correct_write["attempted"],
                "correct_writes": correct_write["applied"],
                "wrong_write_attempts": wrong_write["attempted"],
                "wrong_writes": wrong_write["applied"],
                "prompts": prompts,
            }
        )
    return plans


def _probability_payload(
    scores: Sequence[float], labels: Sequence[str], temperature: float
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64) / temperature
    values -= np.max(values)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum()
    index = int(np.argmax(probabilities))
    return {
        "scores": [float(value) for value in scores],
        "probabilities": [float(value) for value in probabilities],
        "prediction": labels[index],
        "confidence": float(probabilities[index]),
    }


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if np.any(selected):
            result += float(selected.mean()) * abs(
                float(correct[selected].mean()) - float(confidence[selected].mean())
            )
    return float(result)


def metric_bundle(
    records: Sequence[dict[str, Any]], branch: str, labels: Sequence[str]
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot calculate metrics for zero records")
    label_to_id = {label: index for index, label in enumerate(labels)}
    gold = np.asarray([label_to_id[record["gold"]] for record in records])
    predicted = np.asarray(
        [label_to_id[record[branch]["prediction"]] for record in records]
    )
    probabilities = np.asarray(
        [record[branch]["probabilities"] for record in records], dtype=np.float64
    )
    ids = list(range(len(labels)))
    per_emotion = f1_score(gold, predicted, labels=ids, average=None, zero_division=0)
    one_hot = np.eye(len(labels), dtype=np.float64)[gold]
    nll = -np.log(np.maximum(probabilities[np.arange(len(gold)), gold], 1e-12))
    confidence = probabilities.max(axis=1)
    correct = predicted == gold
    return {
        "weighted_f1": float(f1_score(gold, predicted, labels=ids, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(gold, predicted, labels=ids, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(gold, predicted)),
        "nll": float(nll.mean()),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _ece(confidence, correct),
        "per_emotion_f1": {
            label: float(value) for label, value in zip(labels, per_emotion)
        },
    }


def paired_advantage(
    proposed: dict[str, Any], control: dict[str, Any]
) -> dict[str, float]:
    return {
        "weighted_f1": proposed["weighted_f1"] - control["weighted_f1"],
        "macro_f1": proposed["macro_f1"] - control["macro_f1"],
        "accuracy": proposed["accuracy"] - control["accuracy"],
        "nll": control["nll"] - proposed["nll"],
        "brier": control["brier"] - proposed["brier"],
        "ece": control["ece"] - proposed["ece"],
    }


def _ci(values: Sequence[float]) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired_dialogue_bootstrap(
    records: Sequence[dict[str, Any]],
    proposed_branch: str,
    control_branch: str,
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dialogue[record["dialogue_id"]].append(record)
    dialogue_ids = sorted(by_dialogue)
    rng = random.Random(seed)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for _ in dialogue_ids:
            sampled.extend(by_dialogue[rng.choice(dialogue_ids)])
        proposed = metric_bundle(sampled, proposed_branch, labels)
        control = metric_bundle(sampled, control_branch, labels)
        for metric, value in paired_advantage(proposed, control).items():
            values[metric].append(value)
    point = paired_advantage(
        metric_bundle(records, proposed_branch, labels),
        metric_bundle(records, control_branch, labels),
    )
    return {
        "proposed": proposed_branch,
        "control": control_branch,
        "dialogues": len(dialogue_ids),
        "samples": samples,
        "seed": seed,
        "metrics": {
            metric: {
                "point_advantage": point[metric],
                "ci95": _ci(series),
                "probability_positive": float(np.mean(np.asarray(series) > 0)),
            }
            for metric, series in values.items()
        },
    }


def _paths(root: Path, section: dict[str, Any]) -> dict[str, Path]:
    base = _resolve(root, section.get("artifact_dir", "artifacts/hard_role_oracle_stage4/meld_validation"))
    return {
        "base": base,
        "scores": base / "scores" / "validation.jsonl",
        "manifest": base / "run_manifest.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "decision": base / "reports" / "frozen_decision.json",
    }


def _load_context(config_path: Path) -> tuple[Path, dict[str, Any], list[str], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    root = config_path.resolve().parent.parent
    config = _read_yaml(config_path)
    section = config.get("hard_role_oracle_stage4", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 4 is validation-only; refusing any non-validation split")
    labels = list(section.get("labels", []))
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("hard_role_oracle_stage4.labels must be unique")
    processed_dir = _resolve(root, config["data"]["processed_dir"])
    validation_path = processed_dir / "validation.jsonl"
    if not validation_path.exists():
        raise FileNotFoundError(f"Missing validation data: {validation_path}")
    # The literal test path is intentionally never opened by this module.
    dialogues = _read_jsonl(validation_path)
    dialogue_map = {item["dialogue_id"]: item for item in dialogues}

    annotation_path = _resolve(root, section["annotation_file"])
    quality_path = _resolve(root, section["quality_report"])
    if not annotation_path.exists() or not quality_path.exists():
        raise FileNotFoundError(
            "Run V2.2 sampling, annotation, verification, and finalization before Stage 4 scoring"
        )
    quality = _read_json(quality_path)
    if quality.get("split") != "validation":
        raise ValueError("Annotation quality report is not validation-only")
    if quality.get("eligible_for_oracle") is not True:
        raise RuntimeError("V2.2 hard annotations did not pass the frozen quality gate")
    verified_records = _read_jsonl(annotation_path)
    verified: dict[str, dict[str, Any]] = {}
    selected_ids: list[str] = []
    for record in verified_records:
        dialogue_id = record.get("dialogue_id")
        turn_id = record.get("turn_id")
        if not isinstance(dialogue_id, str) or not dialogue_id.startswith("validation:"):
            raise ValueError(f"Invalid annotated validation dialogue {dialogue_id!r}")
        if not isinstance(turn_id, int):
            raise ValueError(f"Invalid annotated turn ID {turn_id!r}")
        key = _key(dialogue_id, turn_id)
        if key in verified:
            raise ValueError(f"Duplicate verified annotation {key}")
        verified[key] = record
        if dialogue_id not in selected_ids:
            selected_ids.append(dialogue_id)
    selected: list[dict[str, Any]] = []
    for dialogue_id in selected_ids:
        if dialogue_id not in dialogue_map:
            raise ValueError(f"Annotated dialogue absent from processed validation data: {dialogue_id}")
        dialogue = dialogue_map[dialogue_id]
        expected = {_key(dialogue_id, index) for index in range(len(dialogue["utterances"]))}
        actual = {key for key in verified if key.startswith(f"{dialogue_id}::")}
        if actual != expected:
            raise ValueError(f"Incomplete annotated trajectory for {dialogue_id}")
        selected.append(dialogue)
    return root, config, labels, selected, verified, quality


def check(config_path: Path) -> dict[str, Any]:
    root, config, labels, dialogues, verified, quality = _load_context(config_path)
    section = config["hard_role_oracle_stage4"]
    result = {
        "status": "ready",
        "experiment": "hard_role_oracle_stage4",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "model_name": section["model_name"],
        "labels": labels,
        "complete_dialogues": len(dialogues),
        "targets": len(verified),
        "annotation_quality_status": quality["status"],
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "project_root": str(root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _make_scorer(section: dict[str, Any], labels: Sequence[str]) -> Any:
    try:
        from pfa_erc.training.llm_context_utility_stage1 import CausalLabelScorer
    except ImportError as exc:
        raise RuntimeError(
            "Stage 4 requires the installed Stage 1 CausalLabelScorer. "
            "Install this patch over the existing pfa-erc v0.10.0 project with python3 -m pip install -e ."
        ) from exc
    return CausalLabelScorer(section, labels)


def score(config_path: Path, overwrite: bool) -> dict[str, Any]:
    root, config, labels, dialogues, verified, quality = _load_context(config_path)
    section = config["hard_role_oracle_stage4"]
    paths = _paths(root, section)
    if overwrite and paths["scores"].exists():
        paths["scores"].unlink()
    existing = _read_jsonl(paths["scores"]) if paths["scores"].exists() else []
    completed = {record["key"] for record in existing}
    scorer = _make_scorer(section, labels)
    temperature = float(section.get("shared_temperature", 1.0))
    if temperature <= 0:
        raise ValueError("shared_temperature must be positive")
    manifest = {
        "experiment": "hard_role_oracle_stage4",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "annotation_file": str(_resolve(root, section["annotation_file"])),
        "annotation_sha256": _sha256(_resolve(root, section["annotation_file"])),
        "quality_report_sha256": _sha256(_resolve(root, section["quality_report"])),
        "branches": list(BRANCHES),
        "temperature": temperature,
        "scorer": scorer.audit,
        "causal_invariant": "predict turn t before applying writes from turn t",
    }
    _write_json(paths["manifest"], manifest)

    for dialogue_number, dialogue in enumerate(dialogues, 1):
        plans = build_dialogue_plans(dialogue, verified, labels, section)
        pending = [plan for plan in plans if plan["key"] not in completed]
        if not pending:
            continue
        flat_prompts = [plan["prompts"][branch] for plan in pending for branch in BRANCHES]
        flat_scores = scorer.score_prompts(flat_prompts)
        output: list[dict[str, Any]] = []
        cursor = 0
        for plan in pending:
            record = {key: value for key, value in plan.items() if key != "prompts"}
            for branch in BRANCHES:
                record[branch] = _probability_payload(flat_scores[cursor], labels, temperature)
                cursor += 1
            output.append(record)
        _append_jsonl(paths["scores"], output)
        print(
            f"[Stage 4] dialogue {dialogue_number}/{len(dialogues)} "
            f"{dialogue['dialogue_id']}: {len(output)} target(s)",
            flush=True,
        )
    return summarize(config_path)


def _flip_counts(
    records: Sequence[dict[str, Any]], proposed: str, control: str
) -> dict[str, int]:
    good = sum(
        record[proposed]["prediction"] == record["gold"]
        and record[control]["prediction"] != record["gold"]
        for record in records
    )
    bad = sum(
        record[proposed]["prediction"] != record["gold"]
        and record[control]["prediction"] == record["gold"]
        for record in records
    )
    return {"good": good, "bad": bad, "net": good - bad}


def _markdown(summary: dict[str, Any]) -> str:
    labels = summary["labels"]
    metrics = summary["metrics"]
    names = {
        "current_only": "4A current only",
        "full_history": "4B full causal history",
        "oracle_evidence": "4C verified evidence Oracle",
        "oracle_role_memory": "4D verified role-memory Oracle",
        "random_evidence": "4E count-matched random evidence",
        "wrong_recipient": "4F wrong-recipient memory",
    }
    lines = [
        "# Stage 4 — Hard-Label Role-Memory Oracle",
        "",
        "> Complete-dialogue MELD validation pilot. The MELD test split remains locked.",
        "",
        "| Variant | wF1 | macro F1 | accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        item = metrics[branch]
        lines.append(
            f"| {names[branch]} | {item['weighted_f1']:.4f} | {item['macro_f1']:.4f} | "
            f"{item['accuracy']:.4f} | {item['nll']:.4f} | {item['brier']:.4f} | {item['ece']:.4f} |"
        )
    lines.extend(["", "## Pre-registered causal contrasts", ""])
    for name, comparison in summary["comparisons"].items():
        macro = comparison["bootstrap"]["metrics"]["macro_f1"]
        nll = comparison["bootstrap"]["metrics"]["nll"]
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
            "## Diagnostics",
            "",
            f"- Targets / complete dialogues: **{summary['targets']} / {summary['dialogues']}**",
            f"- Mean Oracle evidence retention: **{diagnostics['evidence_retention_rate']:.1%}**",
            f"- Role-memory read rate: **{diagnostics['role_memory_read_rate']:.1%}**",
            f"- Correct event writes: **{diagnostics['correct_writes']}**",
            f"- Wrong-recipient control coverage: **{diagnostics['wrong_recipient_coverage']:.1%}**",
            "",
            "## Frozen pilot decision",
            "",
            f"Status: **{summary['decision']['status']}**",
        ]
    )
    for check_name, passed in summary["decision"]["checks"].items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {check_name}")
    lines.extend(["", "## Per-emotion F1", ""])
    lines.append("| Emotion | Current | Full | Evidence Oracle | Role-memory Oracle | Random | Wrong recipient |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        lines.append(
            f"| {label} | "
            + " | ".join(f"{metrics[branch]['per_emotion_f1'][label]:.4f}" for branch in BRANCHES)
            + " |"
        )
    lines.extend(
        [
            "",
            "This is an Oracle/pilot mechanism test, not a deployable prediction result. No MELD test result may be selected from it.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(config_path: Path) -> dict[str, Any]:
    root, config, labels, dialogues, verified, quality = _load_context(config_path)
    section = config["hard_role_oracle_stage4"]
    paths = _paths(root, section)
    if not paths["scores"].exists():
        raise FileNotFoundError(f"Missing score file: {paths['scores']}")
    records = _read_jsonl(paths["scores"])
    expected_keys = set(verified)
    actual_keys = {record["key"] for record in records}
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise ValueError(
            f"Score coverage mismatch: expected {len(expected_keys)}, got {len(actual_keys)} unique/{len(records)} rows"
        )
    metrics = {branch: metric_bundle(records, branch, labels) for branch in BRANCHES}
    samples = int(section.get("bootstrap_samples", 2000))
    seed = int(section.get("bootstrap_seed", 2034))
    pairs = {
        "evidence_vs_random": ("oracle_evidence", "random_evidence"),
        "role_memory_vs_evidence": ("oracle_role_memory", "oracle_evidence"),
        "correct_vs_wrong_recipient": ("oracle_role_memory", "wrong_recipient"),
        "role_memory_vs_current": ("oracle_role_memory", "current_only"),
        "role_memory_vs_full": ("oracle_role_memory", "full_history"),
    }
    comparisons: dict[str, Any] = {}
    for offset, (name, (proposed, control)) in enumerate(pairs.items()):
        comparisons[name] = {
            "proposed": proposed,
            "control": control,
            "point_advantage": paired_advantage(metrics[proposed], metrics[control]),
            "flips": _flip_counts(records, proposed, control),
            "bootstrap": paired_dialogue_bootstrap(
                records, proposed, control, labels, samples, seed + offset
            ),
        }

    total_history = sum(record["history_count"] for record in records)
    total_evidence = sum(len(record["oracle_evidence_ids"]) for record in records)
    write_attempts = sum(record["correct_write_attempts"] for record in records)
    wrong_attempts = sum(record["wrong_write_attempts"] for record in records)
    wrong_writes = sum(record["wrong_writes"] for record in records)
    diagnostics = {
        "evidence_turns": total_evidence,
        "available_history_turns": total_history,
        "evidence_retention_rate": total_evidence / max(total_history, 1),
        "targets_with_role_memory": sum(record["role_memory_read_count"] > 0 for record in records),
        "role_memory_read_rate": sum(record["role_memory_read_count"] > 0 for record in records) / len(records),
        "mean_role_memory_entries": float(np.mean([record["role_memory_read_count"] for record in records])),
        "correct_write_attempts": write_attempts,
        "correct_writes": sum(record["correct_writes"] for record in records),
        "wrong_write_attempts": wrong_attempts,
        "wrong_writes": wrong_writes,
        "wrong_recipient_coverage": wrong_writes / max(wrong_attempts, 1),
    }

    probability = float(section.get("minimum_bootstrap_probability", 0.90))
    tolerance = float(section.get("weighted_f1_tolerance", 0.01))
    nll_tolerance = float(section.get("nll_tolerance", 0.01))

    def supported(name: str) -> bool:
        values = comparisons[name]["bootstrap"]["metrics"]
        return (
            values["macro_f1"]["probability_positive"] >= probability
            or values["nll"]["probability_positive"] >= probability
        )

    checks = {
        "annotation_quality_gate": quality.get("eligible_for_oracle") is True,
        "weighted_f1_preserved_vs_current": metrics["oracle_role_memory"]["weighted_f1"] >= metrics["current_only"]["weighted_f1"] - tolerance,
        "nll_within_best_current_full_tolerance": metrics["oracle_role_memory"]["nll"] <= min(metrics["current_only"]["nll"], metrics["full_history"]["nll"]) + nll_tolerance,
        "verified_evidence_beats_random": supported("evidence_vs_random"),
        "role_memory_adds_over_evidence": supported("role_memory_vs_evidence"),
        "correct_recipient_beats_wrong_recipient": supported("correct_vs_wrong_recipient"),
        "wrong_recipient_control_coverage": diagnostics["wrong_recipient_coverage"] >= float(section.get("minimum_wrong_recipient_coverage", 0.30)),
    }
    eligible = all(checks.values())
    decision = {
        "status": "advance_to_learned_hard_role_controller" if eligible else "stop_and_diagnose_hard_role_oracle",
        "eligible": eligible,
        "checks": checks,
        "selection_basis": "pre-registered complete-dialogue MELD validation pilot",
        "next_action": (
            "Train a learned hard-role controller on train-only annotations and replicate cross-dataset."
            if eligible
            else "Keep MELD test locked; inspect which hard mechanism contrast failed before expanding annotation."
        ),
        "test_locked": True,
    }
    summary = {
        "experiment": "hard_role_oracle_stage4",
        "version": VERSION,
        "split": "validation",
        "test_locked": True,
        "oracle_diagnostic": True,
        "labels": labels,
        "targets": len(records),
        "dialogues": len(dialogues),
        "metrics": metrics,
        "comparisons": comparisons,
        "diagnostics": diagnostics,
        "decision": decision,
        "artifacts": {key: str(value) for key, value in paths.items()},
    }
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown(summary), encoding="utf-8")
    _write_json(paths["decision"], decision)
    print(_markdown(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Decision: {paths['decision']}")
    return summary


def verify(config_path: Path) -> dict[str, Any]:
    root, config, labels, dialogues, verified, _ = _load_context(config_path)
    section = config["hard_role_oracle_stage4"]
    paths = _paths(root, section)
    required = [paths["scores"], paths["manifest"], paths["summary_json"], paths["summary_md"], paths["decision"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 4 artifact(s): {missing}")
    records = _read_jsonl(paths["scores"])
    summary = _read_json(paths["summary_json"])
    finite_metrics = all(
        math.isfinite(float(summary["metrics"][branch][metric]))
        for branch in BRANCHES
        for metric in ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece")
    )
    checks = {
        "validation_only": summary.get("split") == "validation",
        "test_locked": summary.get("test_locked") is True,
        "complete_score_coverage": len(records) == len(verified),
        "all_six_variants": all(branch in summary.get("metrics", {}) for branch in BRANCHES),
        "finite_metrics": finite_metrics,
        "paired_bootstrap_present": all(
            int(item["bootstrap"].get("samples", 0)) > 0
            for item in summary.get("comparisons", {}).values()
        ),
        "complete_dialogues": int(summary.get("dialogues", 0)) == len(dialogues),
    }
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "scientific_decision": summary.get("decision", {}).get("status"),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 4 artifact verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only hard role-memory Oracle")
    parser.add_argument(
        "stage", choices=("check", "score", "summarize", "verify", "all-validation")
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "check":
        check(args.config)
    elif args.stage == "score":
        score(args.config, overwrite=args.overwrite)
    elif args.stage == "summarize":
        summarize(args.config)
    elif args.stage == "verify":
        verify(args.config)
    else:
        check(args.config)
        score(args.config, overwrite=args.overwrite)
        verify(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
