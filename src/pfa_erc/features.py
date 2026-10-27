from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from .data import Dialogue, load_processed_splits


CONTEXT_LAYOUTS = {"chronological", "current_first_recent_first"}
TRUNCATION_SIDES = {"left", "right"}
ENCODER_SOURCES = {"pretrained_base", "sequence_classifier_backbone"}


@dataclass(frozen=True)
class FormattedUtterance:
    text: str
    current_span: tuple[int, int]


def format_utterance_with_span(
    dialogue: Dialogue,
    index: int,
    context_turns: int = 8,
    layout: str = "chronological",
) -> FormattedUtterance:
    """Format one example and retain the character span of the current turn."""
    if not 0 <= index < len(dialogue.utterances):
        raise IndexError(f"Utterance index {index} is outside dialogue {dialogue.dialogue_id}")
    if context_turns < 0:
        raise ValueError("context_turns must be non-negative")
    if layout not in CONTEXT_LAYOUTS:
        raise ValueError(
            f"Unsupported context layout {layout!r}; expected one of {sorted(CONTEXT_LAYOUTS)}"
        )

    start = max(0, index - context_turns)
    current = dialogue.utterances[index]
    history = dialogue.utterances[start:index]
    if layout == "chronological":
        turns = [*history, current]
    else:
        turns = [current, *reversed(history)]

    separator = " </s> "
    pieces: list[str] = []
    current_span: tuple[int, int] | None = None
    cursor = 0
    for turn in turns:
        piece = f"{turn.speaker}: {turn.text.strip()}"
        if turn is current:
            current_span = (cursor, cursor + len(piece))
        pieces.append(piece)
        cursor += len(piece) + len(separator)
    if current_span is None:
        raise AssertionError("Current utterance was not included in its formatted context")
    return FormattedUtterance(separator.join(pieces), current_span)


def format_utterance(
    dialogue: Dialogue,
    index: int,
    context_turns: int = 8,
    layout: str = "chronological",
) -> str:
    """Format one turn with speaker names and up to eight prior turns.

    RoBERTa itself adds the outer ``<s>`` and ``</s>`` tokens. Internal ``</s>``
    tokens separate turns. The default keeps the legacy chronological layout;
    ``current_first_recent_first`` protects the current turn during right-side
    truncation and orders history from most recent to oldest.
    """
    return format_utterance_with_span(dialogue, index, context_turns, layout).text


def _validate_feature_layout(config: dict[str, Any]) -> tuple[str, str]:
    features = config["features"]
    layout = str(features.get("context_layout", "chronological"))
    truncation_side = str(features.get("truncation_side", "right"))
    if layout not in CONTEXT_LAYOUTS:
        raise ValueError(
            f"features.context_layout must be one of {sorted(CONTEXT_LAYOUTS)}, got {layout!r}"
        )
    if truncation_side not in TRUNCATION_SIDES:
        raise ValueError(
            "features.truncation_side must be 'left' or 'right', "
            f"got {truncation_side!r}"
        )
    return layout, truncation_side


def _feature_encoder_spec(
    config: dict[str, Any], label_to_id: dict[str, int]
) -> tuple[str, str, dict[str, Any] | None]:
    """Resolve an encoder and validate 3A lineage before 3B extraction."""
    features = config["features"]
    source = str(features.get("encoder_source", "pretrained_base"))
    if source not in ENCODER_SOURCES:
        raise ValueError(
            f"features.encoder_source must be one of {sorted(ENCODER_SOURCES)}, "
            f"got {source!r}"
        )
    if source == "pretrained_base":
        return str(features.get("model_name", "roberta-large")), source, None

    raw_checkpoint = features.get("encoder_checkpoint")
    if not raw_checkpoint:
        raise KeyError(
            "features.encoder_checkpoint is required when encoder_source is "
            "sequence_classifier_backbone"
        )
    checkpoint = Path(raw_checkpoint)
    state_path = checkpoint / "pfa_erc_training.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} is missing. Complete Experiment 3A before extracting 3B features."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("experiment") != "roberta_current_utterance_3a":
        raise ValueError(
            f"{checkpoint} is not an Experiment 3A current-utterance checkpoint"
        )
    if state.get("input_format") != "speaker_and_text":
        raise ValueError(
            "3B feature formatting is 'Speaker: text', but the 3A checkpoint used "
            f"input_format={state.get('input_format')!r}"
        )
    context_turns = int(config["data"].get("context_turns", 8))
    if context_turns != 0:
        raise ValueError(
            "Experiment 3B requires data.context_turns=0 so each embedding uses only "
            "the current utterance"
        )
    max_length = int(config["data"].get("max_length", 512))
    if max_length != int(state["max_length"]):
        raise ValueError(
            "3B max_length must match its 3A checkpoint: "
            f"config={max_length}, checkpoint={state['max_length']}"
        )
    layout, truncation_side = _validate_feature_layout(config)
    if layout != "chronological" or truncation_side != "right":
        raise ValueError(
            "Experiment 3B must use context_layout=chronological and "
            "truncation_side=right to match 3A"
        )
    checkpoint_labels = {
        str(label): int(index) for label, index in state["label_to_id"].items()
    }
    normalized_labels = {str(label): int(index) for label, index in label_to_id.items()}
    if checkpoint_labels != normalized_labels:
        raise ValueError("The 3A checkpoint label map differs from processed label_map.json")

    best_metrics = state.get("best_validation_metrics", {})
    lineage = {
        "experiment": state["experiment"],
        "checkpoint": str(checkpoint),
        "training_state": str(state_path),
        "seed": int(state.get("seed", 42)),
        "best_epoch": state.get("best_epoch"),
        "best_validation_weighted_f1": best_metrics.get("weighted_f1"),
        "input_format": state["input_format"],
        "max_length": int(state["max_length"]),
    }
    return str(checkpoint), source, lineage


def _load_feature_encoder(
    model_reference: str,
    encoder_source: str,
    trust_remote_code: bool,
) -> torch.nn.Module:
    if encoder_source == "pretrained_base":
        return AutoModel.from_pretrained(
            model_reference, trust_remote_code=trust_remote_code
        )
    if encoder_source == "sequence_classifier_backbone":
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_reference, trust_remote_code=trust_remote_code
        )
        encoder = classifier.base_model
        del classifier
        return encoder
    raise ValueError(f"Unsupported encoder source: {encoder_source!r}")


class _TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> str:
        return self.texts[index]


def _encode_dialogues(
    dialogues: list[Dialogue],
    model: torch.nn.Module,
    tokenizer: Any,
    label_to_id: dict[str, int],
    config: dict[str, Any],
    device: torch.device,
    classifier_model: torch.nn.Module | None = None,
) -> list[dict[str, Any]]:
    context_turns = int(config["data"].get("context_turns", 8))
    layout, truncation_side = _validate_feature_layout(config)
    tokenizer.truncation_side = truncation_side
    texts: list[str] = []
    for dialogue in dialogues:
        for utterance_index in range(len(dialogue.utterances)):
            texts.append(
                format_utterance(
                    dialogue,
                    utterance_index,
                    context_turns,
                    layout=layout,
                )
            )

    def collate(batch: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=int(config["data"].get("max_length", 512)),
            return_tensors="pt",
        )

    loader = DataLoader(
        _TextDataset(texts),
        batch_size=int(config["features"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(config["experiment"].get("num_workers", 0)),
        collate_fn=collate,
    )
    embeddings: list[torch.Tensor] = []
    classifier_logits: list[torch.Tensor] = []
    use_amp = bool(config["features"].get("use_amp", True)) and device.type == "cuda"
    model.eval()
    if classifier_model is not None:
        classifier_model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="RoBERTa embeddings", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if classifier_model is None:
                    cls_embedding = model(**batch).last_hidden_state[:, 0]
                    logits = None
                else:
                    output = classifier_model(
                        **batch,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                    cls_embedding = output.hidden_states[-1][:, 0]
                    logits = output.logits
            embeddings.extend(cls_embedding.detach().float().cpu().unbind(0))
            if logits is not None:
                classifier_logits.extend(logits.detach().float().cpu().unbind(0))

    result: list[dict[str, Any]] = []
    cursor = 0
    for dialogue in dialogues:
        length = len(dialogue.utterances)
        dialogue_embeddings = torch.stack(embeddings[cursor : cursor + length])
        dialogue_logits = (
            torch.stack(classifier_logits[cursor : cursor + length])
            if classifier_model is not None
            else None
        )
        cursor += length
        speaker_to_id: dict[str, int] = {}
        speaker_ids: list[int] = []
        for turn in dialogue.utterances:
            if turn.speaker not in speaker_to_id:
                speaker_to_id[turn.speaker] = len(speaker_to_id)
            speaker_ids.append(speaker_to_id[turn.speaker])
        max_speakers = int(config["data"].get("max_speakers", 16))
        if len(speaker_to_id) > max_speakers:
            raise ValueError(
                f"Dialogue {dialogue.dialogue_id} has {len(speaker_to_id)} speakers; "
                f"increase data.max_speakers (currently {max_speakers})."
            )
        item = {
            "dialogue_id": dialogue.dialogue_id,
            "embeddings": dialogue_embeddings,
            "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
            "speaker_names": [u.speaker for u in dialogue.utterances],
            "labels": torch.tensor([label_to_id[u.emotion] for u in dialogue.utterances]),
            "label_names": [u.emotion for u in dialogue.utterances],
            "texts": [u.text for u in dialogue.utterances],
            "utterance_ids": [u.utterance_id for u in dialogue.utterances],
        }
        if dialogue_logits is not None:
            item["current_logits"] = dialogue_logits
        result.append(item)
    return result


def extract_features(config: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    processed_dir = Path(config["data"]["processed_dir"])
    label_to_id = json.loads((processed_dir / "label_map.json").read_text(encoding="utf-8"))
    available_splits = load_processed_splits(processed_dir)
    requested_splits = config["features"].get("splits")
    if requested_splits is None:
        splits = available_splits
    else:
        requested = [str(split) for split in requested_splits]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError(
                "features.splits must contain distinct split names"
            )
        unknown = sorted(set(requested) - set(available_splits))
        if unknown:
            raise KeyError(
                f"Unavailable feature splits {unknown}; found "
                f"{sorted(available_splits)}"
            )
        splits = {split: available_splits[split] for split in requested}
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model_name, encoder_source, source_lineage = _feature_encoder_spec(
        config, label_to_id
    )
    trust_remote_code = bool(config["features"].get("trust_remote_code", False))
    cache_classifier_logits = bool(
        config["features"].get("cache_classifier_logits", False)
    )
    if cache_classifier_logits and encoder_source != "sequence_classifier_backbone":
        raise ValueError(
            "features.cache_classifier_logits requires "
            "encoder_source=sequence_classifier_backbone"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    classifier_model: torch.nn.Module | None = None
    if cache_classifier_logits:
        classifier_model = AutoModelForSequenceClassification.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        ).to(device_obj)
        model = classifier_model.base_model
    else:
        model = _load_feature_encoder(
            model_name, encoder_source, trust_remote_code
        ).to(device_obj)
    parameters_owner = classifier_model if classifier_model is not None else model
    for parameter in parameters_owner.parameters():
        parameter.requires_grad_(False)

    encoded = {
        split: _encode_dialogues(
            dialogues,
            model,
            tokenizer,
            label_to_id,
            config,
            device_obj,
            classifier_model=classifier_model,
        )
        for split, dialogues in splits.items()
    }
    first_dialogue = next(
        (
            dialogue
            for split_dialogues in encoded.values()
            for dialogue in split_dialogues
        ),
        None,
    )
    if first_dialogue is None:
        raise ValueError("Feature extraction selected no dialogues")
    embedding_dim = int(first_dialogue["embeddings"].shape[-1])
    if embedding_dim != int(config["model"]["embedding_dim"]):
        raise ValueError(
            f"Config model.embedding_dim={config['model']['embedding_dim']}, "
            f"but {model_name} produced {embedding_dim}."
        )
    payload = {
        "format_version": 1,
        "model_name": model_name,
        "encoder_source": encoder_source,
        "source_lineage": source_lineage,
        "embedding_dim": embedding_dim,
        "label_to_id": label_to_id,
        "splits": encoded,
        "feature_config": {
            "context_turns": config["data"].get("context_turns", 8),
            "max_length": config["data"].get("max_length", 512),
            "context_layout": config["features"].get(
                "context_layout", "chronological"
            ),
            "truncation_side": config["features"].get("truncation_side", "right"),
            "encoder_source": encoder_source,
            "encoder_checkpoint": config["features"].get("encoder_checkpoint"),
            "cache_classifier_logits": cache_classifier_logits,
            "splits": list(encoded),
        },
    }
    output_path = Path(config["data"]["feature_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return {
        "feature_file": str(output_path),
        "embedding_dim": embedding_dim,
        "encoder_source": encoder_source,
        "source_lineage": source_lineage,
        "cache_classifier_logits": cache_classifier_logits,
        "dialogues": {split: len(items) for split, items in encoded.items()},
    }


def _offset_overlap_count(
    offsets: list[list[int]] | list[tuple[int, int]],
    span: tuple[int, int],
) -> int:
    span_start, span_end = span
    return sum(
        int(int(end) > span_start and int(start) < span_end)
        for start, end in offsets
        if int(end) > int(start)
    )


def _empty_audit_bucket() -> dict[str, int]:
    return {
        "utterances": 0,
        "truncated": 0,
        "current_fully_retained": 0,
        "current_partially_retained": 0,
        "current_missing": 0,
        "full_tokens_sum": 0,
        "max_full_tokens": 0,
        "current_tokens": 0,
        "retained_current_tokens": 0,
    }


def _update_audit_bucket(
    bucket: dict[str, int],
    full_tokens: int,
    current_tokens: int,
    retained_current_tokens: int,
    max_length: int,
) -> None:
    bucket["utterances"] += 1
    bucket["full_tokens_sum"] += full_tokens
    bucket["max_full_tokens"] = max(bucket["max_full_tokens"], full_tokens)
    bucket["current_tokens"] += current_tokens
    bucket["retained_current_tokens"] += retained_current_tokens
    if full_tokens > max_length:
        bucket["truncated"] += 1
    if retained_current_tokens >= current_tokens:
        bucket["current_fully_retained"] += 1
    elif retained_current_tokens > 0:
        bucket["current_partially_retained"] += 1
    else:
        bucket["current_missing"] += 1


def _finalize_audit_bucket(bucket: dict[str, int]) -> dict[str, int | float]:
    utterances = max(bucket["utterances"], 1)
    current_tokens = max(bucket["current_tokens"], 1)
    return {
        "utterances": bucket["utterances"],
        "truncated": bucket["truncated"],
        "truncation_rate": bucket["truncated"] / utterances,
        "current_fully_retained": bucket["current_fully_retained"],
        "current_partially_retained": bucket["current_partially_retained"],
        "current_missing": bucket["current_missing"],
        "current_full_retention_rate": bucket["current_fully_retained"] / utterances,
        "current_token_retention_rate": (
            bucket["retained_current_tokens"] / current_tokens
        ),
        "mean_full_tokens": bucket["full_tokens_sum"] / utterances,
        "max_full_tokens": bucket["max_full_tokens"],
    }


def _audit_strategy(
    splits: dict[str, list[Dialogue]],
    tokenizer: Any,
    context_turns: int,
    max_length: int,
    strategy: dict[str, str],
    batch_size: int,
    top_k: int,
) -> dict[str, Any]:
    layout = strategy["context_layout"]
    truncation_side = strategy["truncation_side"]
    if layout not in CONTEXT_LAYOUTS or truncation_side not in TRUNCATION_SIDES:
        raise ValueError(f"Invalid audit strategy: {strategy}")

    previous_side = tokenizer.truncation_side
    tokenizer.truncation_side = truncation_side
    split_results: dict[str, Any] = {}
    all_splits = _empty_audit_bucket()
    all_splits_by_emotion: dict[str, dict[str, int]] = {}
    all_problems: list[dict[str, Any]] = []
    try:
        for split, dialogues in splits.items():
            records: list[dict[str, Any]] = []
            for dialogue in dialogues:
                for index, utterance in enumerate(dialogue.utterances):
                    formatted = format_utterance_with_span(
                        dialogue,
                        index,
                        context_turns=context_turns,
                        layout=layout,
                    )
                    records.append(
                        {
                            "text": formatted.text,
                            "current_span": formatted.current_span,
                            "dialogue_id": dialogue.dialogue_id,
                            "utterance_id": utterance.utterance_id,
                            "emotion": utterance.emotion,
                            "split": split,
                        }
                    )

            overall = _empty_audit_bucket()
            by_emotion: dict[str, dict[str, int]] = {}
            problems: list[dict[str, Any]] = []
            for start in tqdm(
                range(0, len(records), batch_size),
                desc=f"Context audit {strategy['name']} {split}",
                leave=False,
            ):
                chunk = records[start : start + batch_size]
                texts = [str(item["text"]) for item in chunk]
                full = tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=False,
                    return_offsets_mapping=True,
                )
                truncated = tokenizer(
                    texts,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    return_offsets_mapping=True,
                )
                for item, full_ids, full_offsets, kept_offsets in zip(
                    chunk,
                    full["input_ids"],
                    full["offset_mapping"],
                    truncated["offset_mapping"],
                ):
                    current_tokens = _offset_overlap_count(
                        full_offsets, item["current_span"]
                    )
                    retained_current_tokens = _offset_overlap_count(
                        kept_offsets, item["current_span"]
                    )
                    if current_tokens == 0:
                        raise RuntimeError(
                            "Tokenizer produced no offset-mapped token for current utterance "
                            f"{item['dialogue_id']}:{item['utterance_id']}"
                        )
                    full_tokens = len(full_ids)
                    emotion_bucket = by_emotion.setdefault(
                        str(item["emotion"]), _empty_audit_bucket()
                    )
                    aggregate_emotion_bucket = all_splits_by_emotion.setdefault(
                        str(item["emotion"]), _empty_audit_bucket()
                    )
                    _update_audit_bucket(
                        overall,
                        full_tokens,
                        current_tokens,
                        retained_current_tokens,
                        max_length,
                    )
                    _update_audit_bucket(
                        emotion_bucket,
                        full_tokens,
                        current_tokens,
                        retained_current_tokens,
                        max_length,
                    )
                    _update_audit_bucket(
                        all_splits,
                        full_tokens,
                        current_tokens,
                        retained_current_tokens,
                        max_length,
                    )
                    _update_audit_bucket(
                        aggregate_emotion_bucket,
                        full_tokens,
                        current_tokens,
                        retained_current_tokens,
                        max_length,
                    )
                    if retained_current_tokens < current_tokens:
                        problem = {
                            "split": item["split"],
                            "dialogue_id": item["dialogue_id"],
                            "utterance_id": item["utterance_id"],
                            "emotion": item["emotion"],
                            "full_tokens": full_tokens,
                            "current_tokens": current_tokens,
                            "retained_current_tokens": retained_current_tokens,
                            "retention_rate": retained_current_tokens / current_tokens,
                        }
                        problems.append(problem)
                        all_problems.append(problem)

            problems.sort(key=lambda item: (item["retention_rate"], -item["full_tokens"]))
            split_results[split] = {
                "overall": _finalize_audit_bucket(overall),
                "by_emotion": {
                    emotion: _finalize_audit_bucket(bucket)
                    for emotion, bucket in sorted(by_emotion.items())
                },
                "worst_current_truncations": problems[:top_k],
            }
    finally:
        tokenizer.truncation_side = previous_side

    all_problems.sort(key=lambda item: (item["retention_rate"], -item["full_tokens"]))
    return {
        "context_layout": layout,
        "truncation_side": truncation_side,
        "overall": _finalize_audit_bucket(all_splits),
        "by_emotion": {
            emotion: _finalize_audit_bucket(bucket)
            for emotion, bucket in sorted(all_splits_by_emotion.items())
        },
        "worst_current_truncations": all_problems[:top_k],
        "splits": split_results,
    }


def audit_context_truncation(config: dict[str, Any]) -> dict[str, Any]:
    """Measure whether RoBERTa truncation removes tokens from the current turn."""
    processed_dir = Path(config["data"]["processed_dir"])
    splits = load_processed_splits(processed_dir)
    features = config["features"]
    model_name = str(features.get("model_name", "roberta-large"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=bool(features.get("trust_remote_code", False)),
    )
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("Context audit requires a fast tokenizer with offset mappings")

    default_strategies = [
        {
            "name": "legacy_current_last_right",
            "context_layout": "chronological",
            "truncation_side": "right",
        },
        {
            "name": "current_last_left",
            "context_layout": "chronological",
            "truncation_side": "left",
        },
        {
            "name": "current_first_recent_first_right",
            "context_layout": "current_first_recent_first",
            "truncation_side": "right",
        },
    ]
    strategies = features.get("audit_strategies", default_strategies)
    names: set[str] = set()
    normalized: list[dict[str, str]] = []
    for raw in strategies:
        if not isinstance(raw, dict):
            raise TypeError("Each features.audit_strategies entry must be a mapping")
        strategy = {
            "name": str(raw["name"]),
            "context_layout": str(raw["context_layout"]),
            "truncation_side": str(raw["truncation_side"]),
        }
        if strategy["name"] in names:
            raise ValueError(f"Duplicate context-audit strategy name: {strategy['name']}")
        names.add(strategy["name"])
        normalized.append(strategy)

    context_turns = int(config["data"].get("context_turns", 8))
    max_length = int(config["data"].get("max_length", 512))
    batch_size = int(features.get("audit_batch_size", 128))
    top_k = int(features.get("audit_top_k", 20))
    if context_turns < 0:
        raise ValueError("data.context_turns must be non-negative")
    if max_length < 3:
        raise ValueError("data.max_length must be at least 3")
    if batch_size < 1:
        raise ValueError("features.audit_batch_size must be at least 1")
    if top_k < 0:
        raise ValueError("features.audit_top_k must be non-negative")

    strategy_results = {
        strategy["name"]: _audit_strategy(
            splits,
            tokenizer,
            context_turns,
            max_length,
            strategy,
            batch_size,
            top_k,
        )
        for strategy in normalized
    }
    result = {
        "dataset": config["data"]["dataset"],
        "model_name": model_name,
        "context_turns": context_turns,
        "max_length": max_length,
        "comparison": {
            name: {
                "context_layout": strategy_result["context_layout"],
                "truncation_side": strategy_result["truncation_side"],
                **strategy_result["overall"],
            }
            for name, strategy_result in strategy_results.items()
        },
        "strategies": strategy_results,
    }
    report_file = Path(
        features.get(
            "audit_report_file",
            Path(config["experiment"].get("output_dir", "artifacts"))
            / "audits"
            / f"{config['data']['dataset']}_context_truncation.json",
        )
    )
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result["report_file"] = str(report_file)
    return result


def make_toy_feature_store(config: dict[str, Any]) -> dict[str, Any]:
    """Write deterministic random features for a fast, offline model smoke test."""
    from .data import make_synthetic_splits

    splits, labels = make_synthetic_splits()
    label_to_id = {label: index for index, label in enumerate(labels)}
    embedding_dim = int(config["model"]["embedding_dim"])
    generator = torch.Generator().manual_seed(int(config["experiment"].get("seed", 42)))
    encoded: dict[str, list[dict[str, Any]]] = {}
    for split, dialogues in splits.items():
        encoded[split] = []
        for dialogue in dialogues:
            speaker_to_id: dict[str, int] = {}
            speaker_ids: list[int] = []
            for utterance in dialogue.utterances:
                speaker_to_id.setdefault(utterance.speaker, len(speaker_to_id))
                speaker_ids.append(speaker_to_id[utterance.speaker])
            encoded[split].append(
                {
                    "dialogue_id": dialogue.dialogue_id,
                    "embeddings": torch.randn(
                        len(dialogue.utterances), embedding_dim, generator=generator
                    ),
                    "speaker_ids": torch.tensor(speaker_ids, dtype=torch.long),
                    "speaker_names": [u.speaker for u in dialogue.utterances],
                    "labels": torch.tensor([label_to_id[u.emotion] for u in dialogue.utterances]),
                    "label_names": [u.emotion for u in dialogue.utterances],
                    "texts": [u.text for u in dialogue.utterances],
                    "utterance_ids": [u.utterance_id for u in dialogue.utterances],
                }
            )
    output_path = Path(config["data"]["feature_file"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_name": "toy-random-features",
            "embedding_dim": embedding_dim,
            "label_to_id": label_to_id,
            "splits": encoded,
            "feature_config": {"synthetic": True},
        },
        output_path,
    )
    return {"feature_file": str(output_path), "embedding_dim": embedding_dim}
