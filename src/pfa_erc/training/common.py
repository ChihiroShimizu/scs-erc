from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def get_device(requested: str | None = None) -> torch.device:
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def make_grad_scaler(device: torch.device, enabled: bool) -> Any:
    """Create an AMP scaler with the current API and retain older torch support."""
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_feature_store(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {"format_version", "embedding_dim", "label_to_id", "splits"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Invalid feature store; missing {sorted(missing)}")
    return payload


class ForecastWindowDataset(Dataset):
    def __init__(
        self,
        dialogues: list[dict[str, Any]],
        history_steps: int,
        future_steps: int,
    ) -> None:
        self.dialogues = dialogues
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.index: list[tuple[int, int]] = []
        for dialogue_index, dialogue in enumerate(dialogues):
            length = int(dialogue["embeddings"].shape[0])
            self.index.extend((dialogue_index, anchor) for anchor in range(max(length - 1, 0)))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        dialogue_index, anchor = self.index[item]
        dialogue = self.dialogues[dialogue_index]
        start = max(0, anchor - self.history_steps + 1)
        target_end = min(anchor + 1 + self.future_steps, len(dialogue["embeddings"]))
        transition_targets: list[float] = []
        transition_known: list[bool] = []
        for target_index in range(anchor + 1, target_end):
            target_speaker = dialogue["speaker_ids"][target_index]
            previous_same_speaker = torch.nonzero(
                dialogue["speaker_ids"][:target_index].eq(target_speaker), as_tuple=False
            ).flatten()
            if previous_same_speaker.numel() == 0:
                transition_targets.append(0.0)
                transition_known.append(False)
                continue
            previous_index = int(previous_same_speaker[-1])
            changed = bool(dialogue["labels"][target_index] != dialogue["labels"][previous_index])
            transition_targets.append(float(changed))
            transition_known.append(True)
        return {
            "history": dialogue["embeddings"][start : anchor + 1],
            "speaker_ids": dialogue["speaker_ids"][start : anchor + 1],
            "target": dialogue["embeddings"][anchor + 1 : target_end],
            "transition_targets": torch.tensor(transition_targets, dtype=torch.float32),
            "transition_known": torch.tensor(transition_known, dtype=torch.bool),
            "future_steps": self.future_steps,
        }


def collate_forecast(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_history = max(item["history"].shape[0] for item in batch)
    future_steps = int(batch[0]["future_steps"])
    embedding_dim = batch[0]["history"].shape[-1]
    history = torch.zeros(batch_size, max_history, embedding_dim)
    speakers = torch.zeros(batch_size, max_history, dtype=torch.long)
    history_mask = torch.zeros(batch_size, max_history, dtype=torch.bool)
    target = torch.zeros(batch_size, future_steps, embedding_dim)
    target_mask = torch.zeros(batch_size, future_steps, dtype=torch.bool)
    transition_targets = torch.zeros(batch_size, future_steps)
    transition_mask = torch.zeros(batch_size, future_steps, dtype=torch.bool)
    for i, item in enumerate(batch):
        history_length = item["history"].shape[0]
        target_length = item["target"].shape[0]
        history[i, -history_length:] = item["history"]
        speakers[i, -history_length:] = item["speaker_ids"]
        history_mask[i, -history_length:] = True
        target[i, :target_length] = item["target"]
        target_mask[i, :target_length] = True
        if "transition_targets" in item:
            transition_targets[i, :target_length] = item["transition_targets"]
        if "transition_known" in item:
            transition_mask[i, :target_length] = item["transition_known"]
    return {
        "history": history,
        "speaker_ids": speakers,
        "history_mask": history_mask,
        "target": target,
        "target_mask": target_mask,
        "transition_targets": transition_targets,
        "transition_mask": transition_mask & target_mask,
    }


class DialogueFeatureDataset(Dataset):
    def __init__(self, dialogues: list[dict[str, Any]]) -> None:
        self.dialogues = dialogues

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.dialogues[index]


def collate_dialogues(batch: list[dict[str, Any]]) -> dict[str, Any]:
    batch_size = len(batch)
    max_length = max(item["embeddings"].shape[0] for item in batch)
    embedding_dim = batch[0]["embeddings"].shape[-1]
    embeddings = torch.zeros(batch_size, max_length, embedding_dim)
    speaker_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    has_current_logits = ["current_logits" in item for item in batch]
    if any(has_current_logits) and not all(has_current_logits):
        raise ValueError("A dialogue batch cannot mix items with and without current_logits")
    current_logits = None
    if all(has_current_logits):
        num_classes = int(batch[0]["current_logits"].shape[-1])
        current_logits = torch.zeros(batch_size, max_length, num_classes)
    for i, item in enumerate(batch):
        length = item["embeddings"].shape[0]
        embeddings[i, :length] = item["embeddings"]
        speaker_ids[i, :length] = item["speaker_ids"]
        labels[i, :length] = item["labels"]
        mask[i, :length] = True
        if current_logits is not None:
            if item["current_logits"].shape != (length, num_classes):
                raise ValueError(
                    f"Invalid current_logits shape for {item['dialogue_id']}: "
                    f"{tuple(item['current_logits'].shape)}"
                )
            current_logits[i, :length] = item["current_logits"]
    result = {
        "embeddings": embeddings,
        "speaker_ids": speaker_ids,
        "labels": labels,
        "mask": mask,
        "dialogue_ids": [item["dialogue_id"] for item in batch],
    }
    if current_logits is not None:
        result["current_logits"] = current_logits
    return result


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def classification_metrics(
    labels: list[int], predictions: list[int], id_to_label: dict[int, str]
) -> dict[str, Any]:
    ordered_ids = sorted(id_to_label)
    target_names = [id_to_label[index] for index in ordered_ids]
    return {
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=ordered_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=ordered_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=ordered_ids,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    }


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def linear_warmup_decay_lambda(current_step: int, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return current_step / max(warmup_steps, 1)
    progress = (current_step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.0, 1.0 - progress)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0

    def update(self, value: float) -> tuple[bool, bool]:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience
