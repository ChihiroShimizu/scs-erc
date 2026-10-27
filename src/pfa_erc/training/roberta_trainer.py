from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..data import Dialogue, load_processed_splits
from .common import (
    EarlyStopping,
    classification_metrics,
    get_device,
    linear_warmup_decay_lambda,
    make_grad_scaler,
    save_json,
    seed_everything,
)


INPUT_FORMATS = {"speaker_and_text", "text_only"}


def _format_current_utterance(speaker: str, text: str, input_format: str) -> str:
    text = text.strip()
    if input_format == "speaker_and_text":
        return f"{speaker.strip()}: {text}"
    if input_format == "text_only":
        return text
    raise ValueError(
        f"Unsupported train_roberta.input_format {input_format!r}; "
        f"expected one of {sorted(INPUT_FORMATS)}"
    )


def _flatten_current_utterances(
    dialogues: list[Dialogue],
    label_to_id: dict[str, int],
    input_format: str,
) -> list[dict[str, Any]]:
    """Flatten dialogues while retaining gold transition groups for evaluation only."""
    records: list[dict[str, Any]] = []
    for dialogue in dialogues:
        previous_emotion_by_speaker: dict[str, str] = {}
        for utterance in dialogue.utterances:
            if utterance.emotion not in label_to_id:
                raise ValueError(
                    f"Unknown label {utterance.emotion!r} in dialogue {dialogue.dialogue_id}"
                )
            previous_emotion = previous_emotion_by_speaker.get(utterance.speaker)
            if previous_emotion is None:
                transition_group = "unknown"
            elif previous_emotion == utterance.emotion:
                transition_group = "stable"
            else:
                transition_group = "transition"
            records.append(
                {
                    "text": _format_current_utterance(
                        utterance.speaker,
                        utterance.text,
                        input_format,
                    ),
                    "label": label_to_id[utterance.emotion],
                    "label_name": utterance.emotion,
                    "dialogue_id": dialogue.dialogue_id,
                    "utterance_id": utterance.utterance_id,
                    "speaker": utterance.speaker,
                    "transition_group": transition_group,
                }
            )
            previous_emotion_by_speaker[utterance.speaker] = utterance.emotion
    return records


class CurrentUtteranceDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class CurrentUtteranceCollator:
    def __init__(self, tokenizer: Any, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = self.tokenizer(
            [str(item["text"]) for item in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [int(item["label"]) for item in batch], dtype=torch.long
        )
        encoded["transition_groups"] = [
            str(item["transition_group"]) for item in batch
        ]
        encoded["dialogue_ids"] = [str(item["dialogue_id"]) for item in batch]
        encoded["utterance_ids"] = [str(item["utterance_id"]) for item in batch]
        return encoded


def _label_maps(processed_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    path = processed_dir / "label_map.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run pfa-erc prepare first.")
    label_to_id = {
        str(label): int(index)
        for label, index in json.loads(path.read_text(encoding="utf-8")).items()
    }
    expected = list(range(len(label_to_id)))
    if sorted(label_to_id.values()) != expected:
        raise ValueError("label_map.json ids must be contiguous and start at zero")
    return label_to_id, {index: label for label, index in label_to_id.items()}


def _training_config(config: dict[str, Any]) -> dict[str, Any]:
    if "train_roberta" not in config:
        raise KeyError("Config is missing the train_roberta section")
    train_config = config["train_roberta"]
    input_format = str(train_config.get("input_format", "speaker_and_text"))
    if input_format not in INPUT_FORMATS:
        raise ValueError(
            f"train_roberta.input_format must be one of {sorted(INPUT_FORMATS)}"
        )
    numeric_minimums = {
        "max_length": (int(train_config.get("max_length", 128)), 3),
        "batch_size": (int(train_config.get("batch_size", 8)), 1),
        "gradient_accumulation_steps": (
            int(train_config.get("gradient_accumulation_steps", 1)),
            1,
        ),
        "epochs": (int(train_config.get("epochs", 5)), 1),
        "patience": (int(train_config.get("patience", 2)), 1),
    }
    for key, (value, minimum) in numeric_minimums.items():
        if value < minimum:
            raise ValueError(f"train_roberta.{key} must be at least {minimum}")
    return train_config


def _loader(
    records: list[dict[str, Any]],
    tokenizer: Any,
    config: dict[str, Any],
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    train_config = _training_config(config)
    generator = torch.Generator().manual_seed(seed)
    workers = int(config["experiment"].get("num_workers", 0))
    return DataLoader(
        CurrentUtteranceDataset(records),
        batch_size=int(train_config.get("batch_size", 8)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=workers,
        collate_fn=CurrentUtteranceCollator(
            tokenizer, int(train_config.get("max_length", 128))
        ),
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    ignored = {"transition_groups", "dialogue_ids", "utterance_ids"}
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key not in ignored and torch.is_tensor(value)
    }


def _subset_metrics(
    labels: list[int], predictions: list[int], id_to_label: dict[int, str]
) -> dict[str, Any]:
    if not labels:
        return {
            "weighted_f1": None,
            "macro_f1": None,
            "accuracy": None,
            "classification_report": {},
            "utterances": 0,
        }
    metrics = classification_metrics(labels, predictions, id_to_label)
    metrics["utterances"] = len(labels)
    return metrics


def _amp_dtype(train_config: dict[str, Any]) -> torch.dtype:
    name = str(train_config.get("amp_dtype", "float16"))
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError("train_roberta.amp_dtype must be 'float16' or 'bfloat16'")


def _evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    stable_labels: list[int] = []
    stable_predictions: list[int] = []
    transition_labels: list[int] = []
    transition_predictions: list[int] = []
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="RoBERTa evaluate", leave=False):
            model_inputs = _model_inputs(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(**model_inputs)
            logits = output.logits
            batch_labels = model_inputs["labels"]
            batch_predictions = logits.argmax(dim=-1)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(batch_predictions.cpu().tolist())
            for index, group in enumerate(batch["transition_groups"]):
                label = int(batch_labels[index])
                prediction = int(batch_predictions[index])
                if group == "stable":
                    stable_labels.append(label)
                    stable_predictions.append(prediction)
                elif group == "transition":
                    transition_labels.append(label)
                    transition_predictions.append(prediction)
            examples = int(batch_labels.shape[0])
            total_loss += float(output.loss) * examples
            total_examples += examples

    metrics = classification_metrics(labels, predictions, id_to_label)
    metrics["loss"] = total_loss / max(total_examples, 1)
    metrics["utterances"] = total_examples
    metrics["same_speaker_stable"] = _subset_metrics(
        stable_labels, stable_predictions, id_to_label
    )
    metrics["same_speaker_transition"] = _subset_metrics(
        transition_labels, transition_predictions, id_to_label
    )
    return metrics


def _save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
    training_state: dict[str, Any],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    save_json(training_state, checkpoint_dir / "pfa_erc_training.json")


def train_roberta(
    config: dict[str, Any],
    device: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fine-tune RoBERTa on isolated current utterances for Experiment 3A."""
    train_config = _training_config(config)
    seed = int(seed if seed is not None else config["experiment"].get("seed", 42))
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device_obj = get_device(device)
    processed_dir = Path(config["data"]["processed_dir"])
    label_to_id, id_to_label = _label_maps(processed_dir)
    splits = load_processed_splits(processed_dir)
    input_format = str(train_config.get("input_format", "speaker_and_text"))
    records = {
        split: _flatten_current_utterances(dialogues, label_to_id, input_format)
        for split, dialogues in splits.items()
    }

    model_name = str(
        train_config.get(
            "model_name", config.get("features", {}).get("model_name", "roberta-large")
        )
    )
    trust_remote_code = bool(train_config.get("trust_remote_code", False))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label=id_to_label,
        trust_remote_code=trust_remote_code,
    )
    if bool(train_config.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
    model.to(device_obj)

    train_loader = _loader(records["train"], tokenizer, config, True, seed, device_obj)
    validation_split = "validation" if "validation" in records else "train"
    validation_loader = _loader(
        records[validation_split], tokenizer, config, False, seed, device_obj
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-5)),
        weight_decay=float(train_config.get("weight_decay", 0.01)),
    )
    accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_steps = max(optimizer_steps_per_epoch * int(train_config.get("epochs", 5)), 1)
    warmup_steps = int(total_steps * float(train_config.get("warmup_ratio", 0.1)))
    scheduler = LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay_lambda(step, warmup_steps, total_steps),
    )
    use_amp = bool(train_config.get("use_amp", True)) and device_obj.type == "cuda"
    amp_dtype = _amp_dtype(train_config)
    scaler = make_grad_scaler(
        device_obj, use_amp and amp_dtype == torch.float16
    )
    stopper = EarlyStopping(int(train_config.get("patience", 2)), mode="max")
    checkpoint_dir = Path(train_config["checkpoint"])
    history: list[dict[str, float | int]] = []
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, int(train_config.get("epochs", 5)) + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"RoBERTa train {epoch}", leave=False)
        ):
            model_inputs = _model_inputs(batch, device_obj)
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            with torch.autocast(
                device_type=device_obj.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                output = model(**model_inputs)
                loss = output.loss
                backward_loss = loss / group_size
            scaler.scale(backward_loss).backward()
            examples = int(model_inputs["labels"].shape[0])
            running_loss += float(loss.detach()) * examples
            running_examples += examples

            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config.get("grad_clip", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        validation = _evaluate_loader(
            model,
            validation_loader,
            device_obj,
            id_to_label,
            use_amp,
            amp_dtype,
        )
        train_loss = running_loss / max(running_examples, 1)
        weighted_f1 = float(validation["weighted_f1"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": float(validation["loss"]),
                "validation_weighted_f1": weighted_f1,
                "validation_macro_f1": float(validation["macro_f1"]),
            }
        )
        print(
            f"[RoBERTa-3A] seed={seed} epoch={epoch:03d} "
            f"train_loss={train_loss:.5f} val_wF1={weighted_f1:.4f} "
            f"val_macroF1={float(validation['macro_f1']):.4f}"
        )
        improved, should_stop = stopper.update(weighted_f1)
        if improved:
            _save_checkpoint(
                model,
                tokenizer,
                checkpoint_dir,
                {
                    "format_version": 1,
                    "experiment": "roberta_current_utterance_3a",
                    "model_name": model_name,
                    "seed": seed,
                    "best_epoch": epoch,
                    "best_validation_metrics": validation,
                    "label_to_id": label_to_id,
                    "input_format": input_format,
                    "max_length": int(train_config.get("max_length", 128)),
                    "config": config,
                },
            )
        if should_stop:
            break

    return {
        "checkpoint": str(checkpoint_dir),
        "seed": seed,
        "model_name": model_name,
        "best_validation_weighted_f1": stopper.best,
        "epochs_run": len(history),
        "training_utterances": len(records["train"]),
        "validation_utterances": len(records[validation_split]),
        "history": history,
    }


def evaluate_roberta(
    config: dict[str, Any],
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
    save_report: bool = True,
) -> dict[str, Any]:
    """Evaluate the saved Experiment 3A checkpoint on a processed split."""
    train_config = _training_config(config)
    checkpoint_dir = Path(checkpoint_path or train_config["checkpoint"])
    state_path = checkpoint_dir / "pfa_erc_training.json"
    if not state_path.exists():
        raise FileNotFoundError(
            f"{state_path} is missing. Run pfa-erc train-roberta first."
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    input_format = str(train_config.get("input_format", "speaker_and_text"))
    max_length = int(train_config.get("max_length", 128))
    if input_format != state["input_format"] or max_length != int(state["max_length"]):
        raise ValueError(
            "Evaluation input settings differ from the saved checkpoint: "
            f"expected input_format={state['input_format']!r}, "
            f"max_length={state['max_length']}"
        )

    seed = int(state.get("seed", config["experiment"].get("seed", 42)))
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device_obj = get_device(device)
    processed_dir = Path(config["data"]["processed_dir"])
    processed_label_to_id, _ = _label_maps(processed_dir)
    label_to_id = {str(key): int(value) for key, value in state["label_to_id"].items()}
    if processed_label_to_id != label_to_id:
        raise ValueError("Processed label_map.json differs from the saved checkpoint")
    id_to_label = {index: label for label, index in label_to_id.items()}
    splits = load_processed_splits(processed_dir)
    split = split or config.get("evaluation_roberta", {}).get("split", "test")
    if split not in splits:
        raise KeyError(f"Split {split!r} is unavailable; found {sorted(splits)}")
    records = _flatten_current_utterances(splits[split], label_to_id, input_format)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(
        device_obj
    )
    loader = _loader(records, tokenizer, config, False, seed, device_obj)
    use_amp = bool(train_config.get("use_amp", True)) and device_obj.type == "cuda"
    metrics = _evaluate_loader(
        model,
        loader,
        device_obj,
        id_to_label,
        use_amp,
        _amp_dtype(train_config),
    )
    result = {
        "checkpoint": str(checkpoint_dir),
        "split": split,
        "seed": seed,
        "input_format": input_format,
        "max_length": max_length,
        "metrics": metrics,
    }
    if save_report:
        report_file = config.get("evaluation_roberta", {}).get("report_file")
        if not report_file:
            raise KeyError("Config is missing evaluation_roberta.report_file")
        save_json(result, report_file)
    return result

