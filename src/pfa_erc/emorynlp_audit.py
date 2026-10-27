from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .data import Dialogue, EMORYNLP_LABELS, prepare_emorynlp


OFFICIAL_COUNTS = {
    "train": {"dialogues": 713, "utterances": 9934},
    "validation": {"dialogues": 99, "utterances": 1344},
    "test": {"dialogues": 85, "utterances": 1328},
}


def _scene_hash(dialogue: Dialogue) -> str:
    normalized = "\n".join(
        f"{utterance.speaker.strip().lower()}\t"
        f"{' '.join(utterance.text.lower().split())}"
        for utterance in dialogue.utterances
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _gap_bucket(gap: int) -> str:
    if gap == 1:
        return "1"
    if gap <= 3:
        return "2-3"
    if gap <= 7:
        return "4-7"
    return "8+"


def _split_profile(dialogues: list[Dialogue]) -> dict[str, Any]:
    labels: Counter[str] = Counter()
    gap_counts: Counter[str] = Counter()
    global_speakers: set[str] = set()
    scene_speaker_counts: list[int] = []
    eligible = 0
    first_appearance = 0
    multi_speaker = 0
    unknown_speaker = 0
    consecutive_same_speaker = 0
    utterance_count = 0

    for dialogue in dialogues:
        last_seen: dict[str, int] = {}
        scene_speakers: set[str] = set()
        previous_speaker: str | None = None
        for turn_index, utterance in enumerate(dialogue.utterances):
            speaker = utterance.speaker
            utterance_count += 1
            labels[utterance.emotion] += 1
            global_speakers.add(speaker)
            scene_speakers.add(speaker)
            multi_speaker += int("+" in speaker)
            unknown_speaker += int(not speaker or speaker.lower() == "unknown")
            consecutive_same_speaker += int(
                previous_speaker is not None and previous_speaker == speaker
            )
            previous_speaker = speaker
            if speaker in last_seen:
                eligible += 1
                gap_counts[_gap_bucket(turn_index - last_seen[speaker])] += 1
            else:
                first_appearance += 1
            last_seen[speaker] = turn_index
        scene_speaker_counts.append(len(scene_speakers))

    return {
        "dialogues": len(dialogues),
        "utterances": utterance_count,
        "average_dialogue_length": (
            utterance_count / len(dialogues) if dialogues else 0.0
        ),
        "unique_speakers": len(global_speakers),
        "average_speakers_per_dialogue": (
            sum(scene_speaker_counts) / len(scene_speaker_counts)
            if scene_speaker_counts
            else 0.0
        ),
        "maximum_speakers_in_dialogue": (
            max(scene_speaker_counts) if scene_speaker_counts else 0
        ),
        "speaker_history_eligible_targets": eligible,
        "speaker_history_eligible_rate": (
            eligible / utterance_count if utterance_count else 0.0
        ),
        "speaker_first_appearance_targets": first_appearance,
        "speaker_recurrence_gap_counts": {
            bucket: gap_counts[bucket]
            for bucket in ("1", "2-3", "4-7", "8+")
        },
        "consecutive_same_speaker_turns": consecutive_same_speaker,
        "multi_speaker_utterances": multi_speaker,
        "unknown_speaker_utterances": unknown_speaker,
        "label_counts": {
            label: labels[label]
            for label in EMORYNLP_LABELS
        },
        "label_rates": {
            label: labels[label] / utterance_count if utterance_count else 0.0
            for label in EMORYNLP_LABELS
        },
    }


def build_emorynlp_audit(
    raw_dir: str | Path,
    *,
    strict_official: bool = False,
) -> dict[str, Any]:
    splits, labels = prepare_emorynlp(raw_dir)
    profiles = {
        split: _split_profile(dialogues)
        for split, dialogues in splits.items()
    }
    count_mismatches: list[dict[str, Any]] = []
    for split, expected in OFFICIAL_COUNTS.items():
        observed = profiles[split]
        for field, expected_value in expected.items():
            if observed[field] != expected_value:
                count_mismatches.append(
                    {
                        "split": split,
                        "field": field,
                        "expected": expected_value,
                        "observed": observed[field],
                    }
                )

    scene_ids = {
        split: {
            dialogue.dialogue_id.split(":", 1)[-1]
            for dialogue in dialogues
        }
        for split, dialogues in splits.items()
    }
    scene_hashes = {
        split: {_scene_hash(dialogue) for dialogue in dialogues}
        for split, dialogues in splits.items()
    }
    overlap: dict[str, Any] = {}
    split_names = ("train", "validation", "test")
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}_vs_{right}"
            overlap[key] = {
                "scene_id_overlap": len(scene_ids[left] & scene_ids[right]),
                "exact_scene_content_overlap": len(
                    scene_hashes[left] & scene_hashes[right]
                ),
            }

    payload = {
        "dataset": "emorynlp",
        "labels": labels,
        "scene_boundary_policy": (
            "Each official scene is one independent dialogue. "
            "All speaker-state and transition memory resets at every scene."
        ),
        "splits": profiles,
        "official_release_counts": OFFICIAL_COUNTS,
        "official_count_mismatches": count_mismatches,
        "cross_split_overlap": overlap,
        "checks": {
            "official_counts_match": not count_mismatches,
            "no_scene_id_overlap": all(
                row["scene_id_overlap"] == 0 for row in overlap.values()
            ),
            "no_exact_scene_content_overlap": all(
                row["exact_scene_content_overlap"] == 0
                for row in overlap.values()
            ),
            "no_unknown_speaker": all(
                row["unknown_speaker_utterances"] == 0
                for row in profiles.values()
            ),
            "labels_match_protocol": labels == EMORYNLP_LABELS,
        },
    }
    failures = [
        name
        for name, passed in payload["checks"].items()
        if not passed
    ]
    payload["status"] = "passed" if not failures else "failed"
    payload["failed_checks"] = failures
    if strict_official and failures:
        raise ValueError(
            "EmoryNLP audit failed: "
            + ", ".join(failures)
            + ". Inspect the generated audit before training."
        )
    return payload


def save_emorynlp_audit(
    raw_dir: str | Path,
    output_file: str | Path,
    *,
    strict_official: bool = False,
) -> dict[str, Any]:
    payload = build_emorynlp_audit(
        raw_dir,
        strict_official=strict_official,
    )
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload
