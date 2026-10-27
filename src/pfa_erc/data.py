from __future__ import annotations

import csv
import hashlib
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class Utterance:
    utterance_id: str
    speaker: str
    text: str
    emotion: str


@dataclass(frozen=True)
class Dialogue:
    dialogue_id: str
    utterances: list[Utterance]


MELD_LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
EMORYNLP_LABELS = ["sad", "mad", "scared", "powerful", "peaceful", "joyful", "neutral"]
IEMOCAP_4_LABELS = ["anger", "happiness", "sadness", "neutral"]
IEMOCAP_6_LABELS = ["anger", "happiness", "sadness", "neutral", "excited", "frustrated"]
EMOTIONLINES_LABELS = [
    "anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"
]
DAILYDIALOG_LABELS = [
    "neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"
]


def _normal(value: Any) -> str:
    return str(value).strip()


def write_jsonl(dialogues: Iterable[Dialogue], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for dialogue in dialogues:
            handle.write(json.dumps(asdict(dialogue), ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[Dialogue]:
    dialogues: list[Dialogue] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            utterances = [Utterance(**item) for item in raw["utterances"]]
            if not utterances:
                raise ValueError(f"Empty dialogue at {path}:{line_number}")
            dialogues.append(Dialogue(str(raw["dialogue_id"]), utterances))
    return dialogues


def load_processed_splits(processed_dir: str | Path) -> dict[str, list[Dialogue]]:
    processed_dir = Path(processed_dir)
    result: dict[str, list[Dialogue]] = {}
    for split in ("train", "validation", "test"):
        path = processed_dir / f"{split}.jsonl"
        if path.exists():
            result[split] = read_jsonl(path)
    if "train" not in result or "test" not in result:
        raise FileNotFoundError(
            f"Expected train.jsonl and test.jsonl under {processed_dir}. Run prepare first."
        )
    return result


def _find_one(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        direct = root / candidate
        if direct.exists():
            return direct
        matches = list(root.rglob(candidate))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"None of {candidates} found under {root}")


def _meld_from_csv(path: Path, split: str) -> list[Dialogue]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        dialogue_id = _normal(row.get("Dialogue_ID", row.get("dialogue_id", "")))
        groups.setdefault(dialogue_id, []).append(row)

    dialogues: list[Dialogue] = []
    for dialogue_id, items in groups.items():
        items.sort(key=lambda x: int(x.get("Utterance_ID", x.get("utterance_id", 0))))
        utterances = [
            Utterance(
                utterance_id=_normal(row.get("Utterance_ID", row.get("utterance_id", i))),
                speaker=_normal(row.get("Speaker", row.get("speaker", "unknown"))),
                text=_normal(row.get("Utterance", row.get("text", ""))),
                emotion=_normal(row.get("Emotion", row.get("emotion", "neutral"))).lower(),
            )
            for i, row in enumerate(items)
        ]
        dialogues.append(Dialogue(f"{split}:{dialogue_id}", utterances))
    return dialogues


def prepare_meld(raw_dir: str | Path) -> tuple[dict[str, list[Dialogue]], list[str]]:
    root = Path(raw_dir)
    try:
        files = {
            "train": _find_one(root, ["train_sent_emo.csv"]),
            "validation": _find_one(root, ["dev_sent_emo.csv", "valid_sent_emo.csv"]),
            "test": _find_one(root, ["test_sent_emo.csv"]),
        }
    except FileNotFoundError as error:
        expected = "train_sent_emo.csv, dev_sent_emo.csv, test_sent_emo.csv"
        raise FileNotFoundError(
            f"MELD annotation CSV files are missing under {root.resolve()}. "
            f"Expected: {expected}. Download the annotations from "
            "https://github.com/declare-lab/MELD/tree/master/data/MELD and place "
            "them in that directory (subdirectories are also accepted). Only the "
            "CSV annotations are required for this text-only reproduction."
        ) from error
    return {split: _meld_from_csv(path, split) for split, path in files.items()}, MELD_LABELS


def _walk_emory_scenes(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        utterances = node.get("utterances")
        if isinstance(utterances, list) and utterances:
            first = utterances[0]
            if isinstance(first, dict) and any(k in first for k in ("transcript", "utterance", "text")):
                yield node
                return
        for value in node.values():
            yield from _walk_emory_scenes(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk_emory_scenes(value)


def _emory_from_json(path: Path, split: str) -> list[Dialogue]:
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    dialogues: list[Dialogue] = []
    for scene_index, scene in enumerate(_walk_emory_scenes(raw)):
        raw_utterances = scene["utterances"]
        utterances: list[Utterance] = []
        for i, item in enumerate(raw_utterances):
            speakers = item.get("speakers", item.get("speaker", "unknown"))
            if isinstance(speakers, list):
                speaker = "+".join(map(str, speakers))
            else:
                speaker = str(speakers)
            label = item.get("emotion", item.get("label", "neutral"))
            text = item.get("transcript", item.get("utterance", item.get("text", "")))
            utterance_id = item.get("utterance_id", item.get("id", i))
            utterances.append(
                Utterance(_normal(utterance_id), _normal(speaker), _normal(text), _normal(label).lower())
            )
        scene_id = scene.get("scene_id", scene.get("id", scene_index))
        dialogues.append(Dialogue(f"{split}:{scene_id}", utterances))
    if not dialogues:
        raise ValueError(f"Could not find EmoryNLP scenes with utterances in {path}")
    return dialogues


def prepare_emorynlp(raw_dir: str | Path) -> tuple[dict[str, list[Dialogue]], list[str]]:
    root = Path(raw_dir)
    files = {
        "train": _find_one(root, ["emorynlp_train_final.json", "train.json"]),
        "validation": _find_one(root, ["emorynlp_dev_final.json", "dev.json", "validation.json"]),
        "test": _find_one(root, ["emorynlp_test_final.json", "test.json"]),
    }
    return {split: _emory_from_json(path, split) for split, path in files.items()}, EMORYNLP_LABELS


def _emotionlines_from_json(
    path: Path,
    split: str,
    include_non_neutral: bool,
) -> list[Dialogue]:
    """Read the official EmotionLines Friends/EmotionPush JSON format."""
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(raw, dict):
        for key in ("dialogues", "conversations", "data"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected a list of dialogues in {path}")

    dialogues: list[Dialogue] = []
    for dialogue_index, item in enumerate(raw):
        turns = item.get("utterances", item.get("messages", [])) if isinstance(item, dict) else item
        if not isinstance(turns, list):
            continue
        utterances: list[Utterance] = []
        for turn_index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            emotion = _normal(turn.get("emotion", turn.get("label", "neutral"))).lower()
            emotion = {
                "angry": "anger",
                "happy": "joy",
                "happiness": "joy",
                "sad": "sadness",
                "surprised": "surprise",
                "fearful": "fear",
                "disgusted": "disgust",
                "non_neutral": "non-neutral",
            }.get(emotion, emotion)
            if emotion in {"non-neutral", "non_neutral"} and not include_non_neutral:
                continue
            text = _normal(
                turn.get("utterance", turn.get("text", turn.get("message", "")))
            )
            if not text:
                continue
            utterances.append(
                Utterance(
                    utterance_id=_normal(turn.get("utterance_id", turn.get("id", turn_index))),
                    speaker=_normal(turn.get("speaker", turn.get("user", "unknown"))),
                    text=text,
                    emotion=emotion,
                )
            )
        if utterances:
            raw_id = item.get("dialogue_id", item.get("id", dialogue_index)) if isinstance(item, dict) else dialogue_index
            dialogues.append(Dialogue(f"{split}:{raw_id}", utterances))
    if not dialogues:
        raise ValueError(f"No EmotionLines dialogues were found in {path}")
    return dialogues


def prepare_emotionlines(
    raw_dir: str | Path,
    subset: str = "emotionpush",
    include_non_neutral: bool = False,
) -> tuple[dict[str, list[Dialogue]], list[str]]:
    root = Path(raw_dir)
    subset = subset.lower()
    if subset not in {"emotionpush", "friends"}:
        raise ValueError("data.emotionlines_subset must be 'emotionpush' or 'friends'")
    prefixes = [subset, subset.capitalize(), subset.upper()]

    def candidates(split_names: list[str]) -> list[str]:
        names: list[str] = []
        for prefix in prefixes:
            for split_name in split_names:
                names.extend(
                    [
                        f"{prefix}_{split_name}.json",
                        f"{prefix}-{split_name}.json",
                        f"{split_name}_{prefix}.json",
                    ]
                )
        names.extend(f"{split_name}.json" for split_name in split_names)
        return names

    files = {
        "train": _find_one(root, candidates(["train"])),
        "validation": _find_one(root, candidates(["dev", "valid", "validation"])),
        "test": _find_one(root, candidates(["test"])),
    }
    labels = list(EMOTIONLINES_LABELS)
    if include_non_neutral:
        labels.append("non-neutral")
    splits = {
        split: _emotionlines_from_json(path, split, include_non_neutral)
        for split, path in files.items()
    }
    return splits, labels


def _dailydialog_split(
    root: Path,
    split: str,
    *,
    mismatch_policy: str = "error",
    alignment_audit: list[dict[str, Any]] | None = None,
) -> list[Dialogue]:
    if mismatch_policy not in {"error", "skip_dialogue"}:
        raise ValueError(
            "DailyDialog mismatch_policy must be 'error' or 'skip_dialogue', "
            f"got {mismatch_policy!r}"
        )

    aliases = {
        "train": ["train"],
        "validation": ["validation", "valid", "dev"],
        "test": ["test"],
    }[split]

    # Prefer the official split-specific filenames. The ACL attachment also
    # contains unsplit auxiliary corpora such as EMNLP_dataset/dialogues_text.txt;
    # silently using those for every split would cause both parsing errors and
    # train/validation/test leakage.
    pair: tuple[Path, Path] | None = None
    for alias in aliases:
        text_name = f"dialogues_{alias}.txt"
        emotion_name = f"dialogues_emotion_{alias}.txt"
        for text_candidate in sorted(root.rglob(text_name)):
            emotion_candidate = text_candidate.with_name(emotion_name)
            if emotion_candidate.exists():
                pair = (text_candidate, emotion_candidate)
                break
        if pair is not None:
            break

    # Also support already-normalised layouts such as
    # train/dialogues_text.txt + train/dialogues_emotion.txt.
    if pair is None:
        for directory in sorted(path for path in root.rglob("*") if path.is_dir()):
            if directory.name.lower() not in aliases:
                continue
            text_candidate = directory / "dialogues_text.txt"
            emotion_candidate = directory / "dialogues_emotion.txt"
            if text_candidate.exists() and emotion_candidate.exists():
                pair = (text_candidate, emotion_candidate)
                break

    if pair is None:
        nested_archives = [
            str(path)
            for alias in aliases
            for path in root.rglob(f"{alias}.zip")
        ]
        archive_hint = (
            " Nested split archives still need extraction: "
            + ", ".join(nested_archives)
            if nested_archives
            else ""
        )
        raise FileNotFoundError(
            f"Could not find split-specific DailyDialog files for {split!r} "
            f"under {root}.{archive_hint}"
        )

    text_path, emotion_path = pair
    text_lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    emotion_lines = emotion_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(text_lines) != len(emotion_lines):
        raise ValueError(
            f"DailyDialog text/emotion line counts differ: {text_path} vs {emotion_path}"
        )
    id_to_emotion = dict(enumerate(DAILYDIALOG_LABELS))
    dialogues: list[Dialogue] = []
    skipped_records: list[dict[str, Any]] = []
    for dialogue_index, (text_line, emotion_line) in enumerate(zip(text_lines, emotion_lines)):
        # DailyDialog uses a trailing __eou__ marker. Remove only the final
        # terminator field; filtering every blank field changes turn indices
        # on the few records that contain an empty utterance while their
        # emotion label is still present.
        text_fields = text_line.split("__eou__")
        while text_fields and not text_fields[-1].strip():
            text_fields.pop()
        texts = [piece.strip() for piece in text_fields]
        emotion_ids = [int(piece) for piece in emotion_line.split()]
        if len(texts) != len(emotion_ids):
            message = (
                "DailyDialog turn/label mismatch at "
                f"{text_path}:{dialogue_index + 1}: "
                f"{len(texts)} turns vs {len(emotion_ids)} labels"
            )
            record = {
                "split": split,
                "source_file": str(text_path),
                "line_number": dialogue_index + 1,
                "text_turns": len(texts),
                "emotion_labels": len(emotion_ids),
                "action": mismatch_policy,
            }
            if alignment_audit is not None:
                alignment_audit.append(record)
            if mismatch_policy == "error":
                raise ValueError(message)
            skipped_records.append(record)
            continue
        utterances = [
            Utterance(str(index), f"speaker_{index % 2}", text, id_to_emotion[label])
            for index, (text, label) in enumerate(zip(texts, emotion_ids))
        ]
        dialogues.append(Dialogue(f"{split}:{dialogue_index}", utterances))
    if skipped_records:
        line_numbers = ", ".join(str(item["line_number"]) for item in skipped_records)
        warnings.warn(
            f"Skipped {len(skipped_records)} malformed DailyDialog {split} dialogue(s) "
            f"from {text_path}; source line(s): {line_numbers}. "
            "See dailydialog_alignment_audit.json in the processed directory.",
            RuntimeWarning,
            stacklevel=2,
        )
    return dialogues


def prepare_dailydialog(
    raw_dir: str | Path,
    *,
    mismatch_policy: str = "error",
    alignment_audit: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, list[Dialogue]], list[str]]:
    root = Path(raw_dir)
    splits = {
        split: _dailydialog_split(
            root,
            split,
            mismatch_policy=mismatch_policy,
            alignment_audit=alignment_audit,
        )
        for split in ("train", "validation", "test")
    }
    return splits, DAILYDIALOG_LABELS


_IEMOCAP_EVAL_RE = re.compile(
    r"^\[[^\]]+\]\s+(?P<utterance>\S+)\s+(?P<label>[a-z]{3})\s+\["
)
_IEMOCAP_TEXT_RE = re.compile(r"^(?P<utterance>\S+)\s+\[[^\]]+\]:\s*(?P<text>.*)$")


def _iemocap_label(raw: str, protocol: str) -> str | None:
    if protocol == "paper_4class":
        return {
            "ang": "anger",
            "hap": "happiness",
            "exc": "happiness",
            "sad": "sadness",
            "neu": "neutral",
        }.get(raw)
    if protocol == "standard_6class":
        return {
            "ang": "anger",
            "hap": "happiness",
            "sad": "sadness",
            "neu": "neutral",
            "exc": "excited",
            "fru": "frustrated",
        }.get(raw)
    raise ValueError(f"Unknown IEMOCAP protocol: {protocol}")


def _stable_validation(dialogue_id: str) -> bool:
    digest = hashlib.sha1(dialogue_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:2], "big") % 10 == 0


def prepare_iemocap(
    raw_dir: str | Path,
    protocol: str = "paper_4class",
    test_session: str = "Session5",
    defer_test: bool = False,
) -> tuple[dict[str, list[Dialogue]], list[str]]:
    root = Path(raw_dir)
    all_dialogues: list[tuple[str, Dialogue]] = []
    for session_dir in sorted(root.glob("Session[1-5]")):
        # A locked confirmatory protocol must not even parse held-out
        # transcripts or labels before model/configuration freezing.
        if defer_test and session_dir.name == test_session:
            continue
        emotion_dir = session_dir / "dialog" / "EmoEvaluation"
        transcript_dir = session_dir / "dialog" / "transcriptions"
        if not emotion_dir.exists() or not transcript_dir.exists():
            continue
        labels: dict[str, str] = {}
        for evaluation_file in emotion_dir.glob("*.txt"):
            for line in evaluation_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = _IEMOCAP_EVAL_RE.match(line)
                if match:
                    mapped = _iemocap_label(match.group("label"), protocol)
                    if mapped is not None:
                        labels[match.group("utterance")] = mapped
        for transcript_file in sorted(transcript_dir.glob("*.txt")):
            utterances: list[Utterance] = []
            for line in transcript_file.read_text(encoding="utf-8", errors="ignore").splitlines():
                match = _IEMOCAP_TEXT_RE.match(line)
                if not match or match.group("utterance") not in labels:
                    continue
                utterance_id = match.group("utterance")
                speaker_match = re.search(r"_([FM])\d+$", utterance_id)
                speaker = (
                    f"{session_dir.name}_{speaker_match.group(1)}"
                    if speaker_match
                    else "unknown"
                )
                utterances.append(
                    Utterance(utterance_id, speaker, match.group("text").strip(), labels[utterance_id])
                )
            if utterances:
                dialogue_id = f"{session_dir.name}:{transcript_file.stem}"
                all_dialogues.append((session_dir.name, Dialogue(dialogue_id, utterances)))
    if not all_dialogues:
        raise FileNotFoundError(f"No IEMOCAP transcripts found under {root}/Session[1-5]")

    splits: dict[str, list[Dialogue]] = {"train": [], "validation": [], "test": []}
    for session, dialogue in all_dialogues:
        if session == test_session:
            splits["test"].append(dialogue)
        elif _stable_validation(dialogue.dialogue_id):
            splits["validation"].append(dialogue)
        else:
            splits["train"].append(dialogue)
    labels = IEMOCAP_4_LABELS if protocol == "paper_4class" else IEMOCAP_6_LABELS
    return splits, labels


def make_synthetic_splits() -> tuple[dict[str, list[Dialogue]], list[str]]:
    """Create a tiny license-free dataset for pipeline smoke tests."""
    labels = ["neutral", "joy", "sadness", "anger"]
    templates = [
        ("A", "The meeting starts at nine.", "neutral"),
        ("B", "Great, I am glad it worked.", "joy"),
        ("A", "The result was disappointing.", "sadness"),
        ("B", "This delay is frustrating.", "anger"),
        ("A", "Let us try one more time.", "neutral"),
        ("B", "Good news, the test passed.", "joy"),
    ]

    def create(split: str, count: int) -> list[Dialogue]:
        output = []
        for d in range(count):
            turns = [
                Utterance(str(i), speaker, f"{text} Example {d}.", label)
                for i, (speaker, text, label) in enumerate(templates)
            ]
            output.append(Dialogue(f"{split}:{d}", turns))
        return output

    return {"train": create("train", 8), "validation": create("validation", 2), "test": create("test", 2)}, labels


def validate_splits(splits: dict[str, list[Dialogue]], labels: list[str]) -> dict[str, Any]:
    label_set = set(labels)
    stats: dict[str, Any] = {"labels": labels, "splits": {}}
    for split, dialogues in splits.items():
        ids: set[str] = set()
        utterance_count = 0
        unknown_labels: set[str] = set()
        for dialogue in dialogues:
            if dialogue.dialogue_id in ids:
                raise ValueError(f"Duplicate dialogue id in {split}: {dialogue.dialogue_id}")
            ids.add(dialogue.dialogue_id)
            utterance_count += len(dialogue.utterances)
            unknown_labels.update(u.emotion for u in dialogue.utterances if u.emotion not in label_set)
        if unknown_labels:
            raise ValueError(f"Unknown labels in {split}: {sorted(unknown_labels)}")
        stats["splits"][split] = {
            "dialogues": len(dialogues),
            "utterances": utterance_count,
            "average_length": utterance_count / max(len(dialogues), 1),
        }
    return stats


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    dataset = data_config["dataset"].lower()
    alignment_audit: list[dict[str, Any]] = []
    dailydialog_mismatch_policy = str(
        data_config.get("dailydialog_mismatch_policy", "error")
    )
    if dataset == "meld":
        splits, labels = prepare_meld(data_config["raw_dir"])
    elif dataset == "emorynlp":
        splits, labels = prepare_emorynlp(data_config["raw_dir"])
    elif dataset == "iemocap":
        splits, labels = prepare_iemocap(
            data_config["raw_dir"],
            protocol=data_config.get("iemocap_protocol", "paper_4class"),
            test_session=data_config.get("iemocap_test_session", "Session5"),
            defer_test=bool(data_config.get("iemocap_defer_test", False)),
        )
    elif dataset == "emotionlines":
        splits, labels = prepare_emotionlines(
            data_config["raw_dir"],
            subset=data_config.get("emotionlines_subset", "emotionpush"),
            include_non_neutral=bool(data_config.get("include_non_neutral", False)),
        )
    elif dataset == "dailydialog":
        splits, labels = prepare_dailydialog(
            data_config["raw_dir"],
            mismatch_policy=dailydialog_mismatch_policy,
            alignment_audit=alignment_audit,
        )
    elif dataset == "synthetic":
        splits, labels = make_synthetic_splits()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    stats = validate_splits(splits, labels)
    output_dir = Path(data_config["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "dailydialog":
        audit_payload = {
            "policy": dailydialog_mismatch_policy,
            "skipped_dialogues": sum(
                item["action"] == "skip_dialogue" for item in alignment_audit
            ),
            "records": alignment_audit,
        }
        audit_name = "dailydialog_alignment_audit.json"
        (output_dir / audit_name).write_text(
            json.dumps(audit_payload, indent=2),
            encoding="utf-8",
        )
        stats["alignment_audit"] = {
            "policy": dailydialog_mismatch_policy,
            "skipped_dialogues": audit_payload["skipped_dialogues"],
            "file": audit_name,
        }
    for split, dialogues in splits.items():
        write_jsonl(dialogues, output_dir / f"{split}.jsonl")
    (output_dir / "label_map.json").write_text(
        json.dumps({label: index for index, label in enumerate(labels)}, indent=2), encoding="utf-8"
    )
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
