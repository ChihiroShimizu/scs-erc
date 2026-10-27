from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models import ContextUtilityRouter
from .common import (
    classification_metrics,
    get_device,
    load_feature_store,
    save_json,
    seed_everything,
)
from .erc_trainer import (
    _build_model,
    _dialogue_fold,
    _probability_quality,
    _validate_current_logit_features,
    train_erc,
)
from .influence_audit import (
    _baseline_logits,
    _dialogue_prefix,
    _forward_target_logits,
)


VARIANT_LABELS = {
    "anchor": "13A current-only Anchor",
    "full_history": "13B always full history",
    "entropy_router": "13C uncertainty router",
    "utility_router": "13D counterfactual utility router",
    "utility_sparse": "13E utility router + sparse selector",
    "oracle": "Diagnostic context oracle",
}
CANDIDATE_VARIANTS = ("utility_router", "utility_sparse")


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    if "experiment13" not in config:
        raise KeyError("Experiment 13 requires an experiment13 config section")
    return config["experiment13"]


def _artifact_dir(config: dict[str, Any]) -> Path:
    path = Path(_settings(config)["artifact_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path


def _feature_path(
    config: dict[str, Any],
    seed: int,
    split: str = "development",
) -> Path:
    settings = _settings(config)
    if split == "test" and settings.get("test_feature_file_pattern"):
        value = str(settings["test_feature_file_pattern"])
    elif settings.get("feature_file_pattern"):
        value = str(settings["feature_file_pattern"])
    else:
        value = str(config["data"]["feature_file"])
    return Path(value.format(seed=int(seed)))


def _seed_list(config: dict[str, Any]) -> list[int]:
    seeds = [int(value) for value in _settings(config).get("seeds", [42, 43, 44])]
    if len(seeds) < 3:
        raise ValueError("Experiment 13 requires at least three independent seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Experiment 13 seeds must be unique")
    return seeds


def _teacher_checkpoint(
    root: Path,
    seed: int,
    fold: int | None,
) -> Path:
    suffix = "full" if fold is None else f"fold{fold}"
    return root / "teachers" / f"context_seed{seed}_{suffix}.pt"


def _label_paths(root: Path, seed: int, split: str) -> tuple[Path, Path]:
    directory = root / "labels"
    return (
        directory / f"{split}_seed{seed}_targets.jsonl",
        directory / f"{split}_seed{seed}_pairs.jsonl",
    )


def _router_checkpoint(root: Path, seed: int) -> Path:
    return root / "routers" / f"router_seed{seed}.pt"


def _seed_report(root: Path, seed: int, split: str) -> Path:
    return root / "reports" / f"{split}_seed{seed}.json"


def _seed_records(root: Path, seed: int, split: str) -> Path:
    return root / "records" / f"{split}_seed{seed}.jsonl"


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(value)
    return records


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utility_class(value: float, threshold: float) -> int:
    if value > threshold:
        return ContextUtilityRouter.HELPFUL
    if value < -threshold:
        return ContextUtilityRouter.HARMFUL
    return ContextUtilityRouter.UNCERTAIN


def _dialogue_lookup(
    dialogues: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dialogue in dialogues:
        key = str(dialogue.get("dialogue_id"))
        if key in result:
            raise ValueError(f"Duplicate dialogue_id in feature store: {key}")
        result[key] = dialogue
    return result


def _turn_index_by_utterance(
    dialogue: dict[str, Any],
    utterance_id: str,
) -> int:
    utterance_ids = dialogue.get("utterance_ids")
    if utterance_ids is None:
        return int(utterance_id)
    matches = [
        index
        for index, value in enumerate(utterance_ids)
        if str(value) == str(utterance_id)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve utterance {utterance_id!r} in "
            f"dialogue {dialogue.get('dialogue_id')!r}"
        )
    return matches[0]


def _history_summary(
    dialogue: dict[str, Any],
    target_turn: int,
    max_history_turns: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    embeddings = dialogue["embeddings"].float()
    speakers = dialogue["speaker_ids"].long()
    current_speaker = int(speakers[target_turn])
    start = max(0, target_turn - max_history_turns)
    history_indices = list(range(start, target_turn))
    same_indices = [
        index
        for index in history_indices
        if int(speakers[index]) == current_speaker
    ]
    other_indices = [
        index
        for index in history_indices
        if int(speakers[index]) != current_speaker
    ]
    zero = torch.zeros_like(embeddings[target_turn])
    same = (
        embeddings[same_indices].mean(dim=0)
        if same_indices
        else zero.clone()
    )
    other = (
        embeddings[other_indices].mean(dim=0)
        if other_indices
        else zero.clone()
    )
    same_gap = target_turn - same_indices[-1] if same_indices else 0
    denominator = math.log1p(max(max_history_turns, 1))
    scalar = torch.tensor(
        [
            float(bool(same_indices)),
            float(bool(other_indices)),
            min(target_turn, max_history_turns) / max(max_history_turns, 1),
            math.log1p(same_gap) / denominator if same_gap else 0.0,
        ],
        dtype=torch.float32,
    )
    return same, other, scalar


def _target_tensors(
    dialogue: dict[str, Any],
    target_turn: int,
    max_history_turns: int,
) -> dict[str, torch.Tensor]:
    same, other, scalar = _history_summary(
        dialogue, target_turn, max_history_turns
    )
    return {
        "current": dialogue["embeddings"][target_turn].float(),
        "same_history": same,
        "other_history": other,
        "anchor_logits": dialogue["current_logits"][target_turn].float(),
        "scalar_features": scalar,
    }


def _pair_tensors(
    dialogue: dict[str, Any],
    target_turn: int,
    history_turn: int,
    max_history_turns: int,
) -> dict[str, torch.Tensor]:
    gap = target_turn - history_turn
    if gap <= 0:
        raise ValueError("History must precede its target")
    denominator = math.log1p(max(max_history_turns, 1))
    return {
        "current": dialogue["embeddings"][target_turn].float(),
        "history": dialogue["embeddings"][history_turn].float(),
        "current_anchor_logits": dialogue["current_logits"][target_turn].float(),
        "history_anchor_logits": dialogue["current_logits"][history_turn].float(),
        "scalar_features": torch.tensor(
            [
                float(
                    dialogue["speaker_ids"][target_turn]
                    == dialogue["speaker_ids"][history_turn]
                ),
                min(math.log1p(gap) / denominator, 1.0),
            ],
            dtype=torch.float32,
        ),
    }


def _selected_prefix(
    dialogue: dict[str, Any],
    target_turn: int,
    selected_history: list[int],
) -> dict[str, Any]:
    indices = sorted(
        {
            int(index)
            for index in selected_history
            if 0 <= int(index) < target_turn
        }
    )
    indices.append(target_turn)
    turn_fields = {
        "embeddings",
        "speaker_ids",
        "speaker_names",
        "labels",
        "label_names",
        "texts",
        "utterance_ids",
        "current_logits",
    }
    output: dict[str, Any] = {}
    index_tensor = torch.tensor(indices, dtype=torch.long)
    for key, value in dialogue.items():
        if key not in turn_fields:
            output[key] = value
        elif torch.is_tensor(value):
            output[key] = value.index_select(0, index_tensor)
        elif isinstance(value, list):
            output[key] = [value[index] for index in indices]
        else:
            output[key] = value
    return output


def _load_context_model(
    config: dict[str, Any],
    store: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    label_to_id = {
        str(key): int(value)
        for key, value in checkpoint.get(
            "label_to_id", store["label_to_id"]
        ).items()
    }
    model_config = checkpoint.get("config", config)
    if bool(model_config["model"].get("use_hitft", False)):
        raise ValueError("Experiment 13 context teachers must not use HiTFT")
    model = _build_model(model_config, len(label_to_id), device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _train_teacher_models(
    config: dict[str, Any],
    device: str | None,
) -> dict[str, Any]:
    settings = _settings(config)
    root = _artifact_dir(config)
    folds = int(settings.get("oof_folds", 3))
    fold_seed = int(settings.get("oof_fold_seed", 13013))
    reuse = bool(settings.get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seed_list(config):
        for fold in range(folds):
            checkpoint = _teacher_checkpoint(root, seed, fold)
            if reuse and checkpoint.exists():
                runs.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "checkpoint": str(checkpoint),
                        "reused": True,
                    }
                )
                continue
            local = copy.deepcopy(config)
            local["experiment"]["name"] = (
                f"{config['experiment']['name']}_teacher_seed{seed}_fold{fold}"
            )
            local["data"]["feature_file"] = str(
                _feature_path(config, seed)
            )
            local["train_erc"]["checkpoint"] = str(checkpoint)
            local["train_erc"]["oof_folds"] = folds
            local["train_erc"]["oof_fold_seed"] = fold_seed
            local["train_erc"]["oof_holdout_fold"] = fold
            result = train_erc(local, device=device, seed=seed)
            runs.append({"seed": seed, "fold": fold, **result, "reused": False})

        full_checkpoint = _teacher_checkpoint(root, seed, None)
        if reuse and full_checkpoint.exists():
            runs.append(
                {
                    "seed": seed,
                    "fold": None,
                    "checkpoint": str(full_checkpoint),
                    "reused": True,
                }
            )
        else:
            local = copy.deepcopy(config)
            local["experiment"]["name"] = (
                f"{config['experiment']['name']}_teacher_seed{seed}_full"
            )
            local["data"]["feature_file"] = str(
                _feature_path(config, seed)
            )
            local["train_erc"]["checkpoint"] = str(full_checkpoint)
            local["train_erc"].pop("oof_holdout_fold", None)
            result = train_erc(local, device=device, seed=seed)
            runs.append(
                {"seed": seed, "fold": None, **result, "reused": False}
            )
    report = {
        "experiment": "13 Context Utility Routing",
        "stage": "train-teachers",
        "folds": folds,
        "fold_seed": fold_seed,
        "runs": runs,
    }
    save_json(report, root / "reports" / "teacher_training.json")
    return report


def _generate_labels_for_seed(
    config: dict[str, Any],
    seed: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    settings = _settings(config)
    root = _artifact_dir(config)
    store = load_feature_store(_feature_path(config, seed))
    _validate_current_logit_features(store, len(store["label_to_id"]))
    if split not in {"train", "validation"}:
        raise ValueError(
            "Development labels may be generated only for train/validation"
        )
    dialogues = list(store["splits"][split])
    max_history = int(settings.get("max_history_turns", 12))
    threshold = float(settings.get("utility_threshold", 0.01))
    batch_size = int(settings.get("counterfactual_batch_size", 128))
    folds = int(settings.get("oof_folds", 3))
    fold_seed = int(settings.get("oof_fold_seed", 13013))
    use_amp = bool(settings.get("use_amp", True)) and device.type == "cuda"

    if split == "train":
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for dialogue in dialogues:
            grouped[_dialogue_fold(dialogue, folds, fold_seed)].append(dialogue)
        checkpoint_groups = [
            (fold, grouped.get(fold, []), _teacher_checkpoint(root, seed, fold))
            for fold in range(folds)
        ]
    else:
        checkpoint_groups = [
            (None, dialogues, _teacher_checkpoint(root, seed, None))
        ]

    target_records: list[dict[str, Any]] = []
    pair_records: list[dict[str, Any]] = []
    for fold, fold_dialogues, checkpoint_path in checkpoint_groups:
        if not fold_dialogues:
            continue
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing teacher checkpoint: {checkpoint_path}. "
                "Run Experiment 13 train-teachers first."
            )
        model, checkpoint = _load_context_model(
            config, store, checkpoint_path, device
        )
        for dialogue in tqdm(
            fold_dialogues,
            desc=f"13 labels {split} seed={seed} fold={fold}",
            leave=False,
        ):
            length = int(dialogue["embeddings"].shape[0])
            if length < 2:
                continue
            full_logits = _baseline_logits(model, dialogue, device, use_amp)
            full_log_probability = torch.log_softmax(full_logits, dim=-1)
            no_context_samples = [
                _selected_prefix(dialogue, target_turn, [])
                for target_turn in range(1, length)
            ]
            no_context_logits = _forward_target_logits(
                model,
                no_context_samples,
                batch_size,
                device,
                use_amp,
            )
            no_context_log_probability = torch.log_softmax(
                no_context_logits, dim=-1
            )
            dialogue_id = str(dialogue.get("dialogue_id"))
            utterance_ids = dialogue.get(
                "utterance_ids", list(range(length))
            )
            deletion_samples: list[dict[str, Any]] = []
            deletion_metadata: list[tuple[int, int, float]] = []
            for target_turn in range(1, length):
                gold = int(dialogue["labels"][target_turn])
                no_context_nll = -float(
                    no_context_log_probability[target_turn - 1, gold]
                )
                full_nll = -float(full_log_probability[target_turn, gold])
                target_utility = no_context_nll - full_nll
                target_records.append(
                    {
                        "seed": seed,
                        "split": split,
                        "teacher_fold": fold,
                        "teacher_checkpoint": str(checkpoint_path),
                        "teacher_epoch": int(checkpoint.get("epoch", -1)),
                        "dialogue_id": dialogue_id,
                        "target_utterance_id": str(utterance_ids[target_turn]),
                        "target_turn": target_turn,
                        "gold_label": gold,
                        "no_context_nll": no_context_nll,
                        "full_context_nll": full_nll,
                        "target_utility": target_utility,
                        "utility_class": _utility_class(
                            target_utility, threshold
                        ),
                    }
                )
                start = max(0, target_turn - max_history)
                for history_turn in range(start, target_turn):
                    deletion_samples.append(
                        _dialogue_prefix(
                            dialogue,
                            target_turn,
                            removed_turn=history_turn,
                        )
                    )
                    deletion_metadata.append(
                        (target_turn, history_turn, full_nll)
                    )
            if deletion_samples:
                deletion_logits = _forward_target_logits(
                    model,
                    deletion_samples,
                    batch_size,
                    device,
                    use_amp,
                )
                deletion_log_probability = torch.log_softmax(
                    deletion_logits, dim=-1
                )
                for row, (target_turn, history_turn, full_nll) in enumerate(
                    deletion_metadata
                ):
                    gold = int(dialogue["labels"][target_turn])
                    deletion_nll = -float(
                        deletion_log_probability[row, gold]
                    )
                    pair_utility = deletion_nll - full_nll
                    pair_records.append(
                        {
                            "seed": seed,
                            "split": split,
                            "teacher_fold": fold,
                            "dialogue_id": dialogue_id,
                            "target_utterance_id": str(
                                utterance_ids[target_turn]
                            ),
                            "history_utterance_id": str(
                                utterance_ids[history_turn]
                            ),
                            "target_turn": target_turn,
                            "history_turn": history_turn,
                            "gap": target_turn - history_turn,
                            "same_speaker": bool(
                                dialogue["speaker_ids"][target_turn]
                                == dialogue["speaker_ids"][history_turn]
                            ),
                            "full_context_nll": full_nll,
                            "deletion_nll": deletion_nll,
                            "pair_utility": pair_utility,
                            "utility_class": _utility_class(
                                pair_utility, threshold
                            ),
                        }
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    target_path, pair_path = _label_paths(root, seed, split)
    _write_jsonl(target_records, target_path)
    _write_jsonl(pair_records, pair_path)
    class_counts = {
        str(label): sum(
            int(record["utility_class"]) == label
            for record in target_records
        )
        for label in range(ContextUtilityRouter.NUM_UTILITY_CLASSES)
    }
    report = {
        "experiment": "13 Context Utility Routing",
        "stage": "build-labels",
        "seed": seed,
        "split": split,
        "targets": len(target_records),
        "pairs": len(pair_records),
        "target_class_counts": class_counts,
        "utility_threshold": threshold,
        "target_file": str(target_path),
        "pair_file": str(pair_path),
        "out_of_fold": split == "train",
        "gold_labels_are_targets_only": True,
        "future_turns_used": False,
    }
    save_json(
        report,
        root / "reports" / f"labels_{split}_seed{seed}.json",
    )
    return report


def _generate_labels(
    config: dict[str, Any],
    split: str,
    device: str | None,
) -> dict[str, Any]:
    device_obj = get_device(device)
    return {
        "experiment": "13 Context Utility Routing",
        "stage": "build-labels",
        "split": split,
        "runs": [
            _generate_labels_for_seed(
                config, seed, split, device_obj
            )
            for seed in _seed_list(config)
        ],
    }


class _TargetUtilityDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        dialogues: dict[str, dict[str, Any]],
        max_history_turns: int,
        cue_labels: dict[tuple[str, str], float] | None = None,
    ) -> None:
        self.records = records
        self.dialogues = dialogues
        self.max_history_turns = max_history_turns
        self.cue_labels = cue_labels or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        dialogue_id = str(record["dialogue_id"])
        dialogue = self.dialogues[dialogue_id]
        target_turn = int(record["target_turn"])
        item = _target_tensors(
            dialogue, target_turn, self.max_history_turns
        )
        item.update(
            {
                "target_utility": torch.tensor(
                    float(record["target_utility"]), dtype=torch.float32
                ),
                "utility_class": torch.tensor(
                    int(record["utility_class"]), dtype=torch.long
                ),
                "cue_slow": torch.tensor(
                    self.cue_labels.get(
                        (
                            dialogue_id,
                            str(record["target_utterance_id"]),
                        ),
                        -1.0,
                    ),
                    dtype=torch.float32,
                ),
            }
        )
        return item


class _PairUtilityDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        dialogues: dict[str, dict[str, Any]],
        max_history_turns: int,
    ) -> None:
        self.records = records
        self.dialogues = dialogues
        self.max_history_turns = max_history_turns

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        dialogue = self.dialogues[str(record["dialogue_id"])]
        item = _pair_tensors(
            dialogue,
            int(record["target_turn"]),
            int(record["history_turn"]),
            self.max_history_turns,
        )
        item["pair_utility"] = torch.tensor(
            float(record["pair_utility"]), dtype=torch.float32
        )
        item["utility_class"] = torch.tensor(
            int(record["utility_class"]), dtype=torch.long
        )
        return item


def _internal_dev(dialogue_id: str, seed: int, percent: int) -> bool:
    digest = hashlib.sha256(
        f"router-dev:{seed}:{dialogue_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % 100 < percent


def _split_records(
    records: list[dict[str, Any]],
    seed: int,
    dev_percent: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [
        record
        for record in records
        if not _internal_dev(str(record["dialogue_id"]), seed, dev_percent)
    ]
    dev = [
        record
        for record in records
        if _internal_dev(str(record["dialogue_id"]), seed, dev_percent)
    ]
    if not train or not dev:
        raise ValueError("Router internal split produced an empty partition")
    return train, dev


def _load_cue_labels(
    config: dict[str, Any],
) -> dict[tuple[str, str], float]:
    settings = _settings(config)
    value = settings.get("emotioncue_train_file")
    if not value:
        return {}
    path = Path(value)
    records = _read_jsonl(path)
    result: dict[tuple[str, str], float] = {}
    rejected = 0
    for record in records:
        split = str(record.get("split", "")).lower()
        if split != "train":
            rejected += 1
            continue
        mode = str(
            record.get("thinking_mode", record.get("mode", ""))
        ).lower()
        if mode not in {"fast", "slow"}:
            continue
        dialogue_id = str(record.get("dialogue_id", ""))
        utterance_id = str(record.get("utterance_id", ""))
        if not dialogue_id or not utterance_id:
            continue
        result[(dialogue_id, utterance_id)] = float(mode == "slow")
    if rejected and not bool(
        settings.get("allow_ignored_nontrain_cues", False)
    ):
        raise ValueError(
            "EmotionCue supervision file contains validation/test records. "
            "Provide a train-only export or explicitly allow them to be "
            "ignored; non-train rationales must never enter training."
        )
    return result


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _target_loss(
    model: ContextUtilityRouter,
    batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    settings: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model.forward_target(
        batch["current"],
        batch["same_history"],
        batch["other_history"],
        batch["anchor_logits"],
        batch["scalar_features"],
    )
    regression = nn.functional.smooth_l1_loss(
        output["utility"], batch["target_utility"]
    )
    classification = nn.functional.cross_entropy(
        output["class_logits"],
        batch["utility_class"],
        weight=class_weights,
    )
    cue_mask = batch["cue_slow"].ge(0)
    cue_loss = regression.new_zeros(())
    if cue_mask.any():
        cue_loss = nn.functional.binary_cross_entropy_with_logits(
            output["cue_slow_logit"][cue_mask],
            batch["cue_slow"][cue_mask],
        )
    loss = (
        float(settings.get("target_regression_weight", 1.0)) * regression
        + float(settings.get("target_class_weight", 1.0)) * classification
        + float(settings.get("emotioncue_aux_weight", 0.0)) * cue_loss
    )
    return loss, {
        "target_regression": float(regression.detach()),
        "target_classification": float(classification.detach()),
        "cue_loss": float(cue_loss.detach()),
    }


def _pair_loss(
    model: ContextUtilityRouter,
    batch: dict[str, torch.Tensor],
    settings: dict[str, Any],
) -> tuple[torch.Tensor, float]:
    prediction = model.forward_pair(
        batch["current"],
        batch["history"],
        batch["current_anchor_logits"],
        batch["history_anchor_logits"],
        batch["scalar_features"],
    )
    raw = nn.functional.smooth_l1_loss(
        prediction, batch["pair_utility"]
    )
    return float(settings.get("pair_regression_weight", 1.0)) * raw, float(
        raw.detach()
    )


def _router_dev_metrics(
    model: ContextUtilityRouter,
    target_loader: DataLoader,
    pair_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    target_gold: list[int] = []
    target_prediction: list[int] = []
    target_utility_gold: list[float] = []
    target_utility_prediction: list[float] = []
    pair_gold: list[float] = []
    pair_prediction: list[float] = []
    with torch.no_grad():
        for batch in target_loader:
            batch = _to_device(batch, device)
            output = model.forward_target(
                batch["current"],
                batch["same_history"],
                batch["other_history"],
                batch["anchor_logits"],
                batch["scalar_features"],
            )
            target_gold.extend(batch["utility_class"].cpu().tolist())
            target_prediction.extend(
                output["class_logits"].argmax(dim=-1).cpu().tolist()
            )
            target_utility_gold.extend(
                batch["target_utility"].cpu().tolist()
            )
            target_utility_prediction.extend(output["utility"].cpu().tolist())
        for batch in pair_loader:
            batch = _to_device(batch, device)
            prediction = model.forward_pair(
                batch["current"],
                batch["history"],
                batch["current_anchor_logits"],
                batch["history_anchor_logits"],
                batch["scalar_features"],
            )
            pair_gold.extend(batch["pair_utility"].cpu().tolist())
            pair_prediction.extend(prediction.cpu().tolist())
    target_mae = float(
        np.mean(
            np.abs(
                np.asarray(target_utility_gold, dtype=float)
                - np.asarray(target_utility_prediction, dtype=float)
            )
        )
    )
    pair_mae = float(
        np.mean(
            np.abs(
                np.asarray(pair_gold, dtype=float)
                - np.asarray(pair_prediction, dtype=float)
            )
        )
    )
    macro = float(
        f1_score(
            target_gold,
            target_prediction,
            labels=list(range(ContextUtilityRouter.NUM_UTILITY_CLASSES)),
            average="macro",
            zero_division=0,
        )
    )
    return {
        "target_mae": target_mae,
        "pair_mae": pair_mae,
        "target_macro_f1": macro,
        "selection_loss": target_mae + pair_mae + 0.25 * (1.0 - macro),
    }


def _train_router_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings = _settings(config)
    root = _artifact_dir(config)
    store = load_feature_store(_feature_path(config, seed))
    dialogues = _dialogue_lookup(list(store["splits"]["train"]))
    target_path, pair_path = _label_paths(root, seed, "train")
    target_records = _read_jsonl(target_path)
    pair_records = _read_jsonl(pair_path)
    dev_percent = int(settings.get("router_dev_percent", 15))
    target_train, target_dev = _split_records(
        target_records, seed, dev_percent
    )
    pair_train, pair_dev = _split_records(pair_records, seed, dev_percent)
    max_history = int(settings.get("max_history_turns", 12))
    cue_labels = _load_cue_labels(config)

    target_train_dataset = _TargetUtilityDataset(
        target_train, dialogues, max_history, cue_labels
    )
    target_dev_dataset = _TargetUtilityDataset(
        target_dev, dialogues, max_history, cue_labels
    )
    pair_train_dataset = _PairUtilityDataset(
        pair_train, dialogues, max_history
    )
    pair_dev_dataset = _PairUtilityDataset(pair_dev, dialogues, max_history)
    batch_size = int(settings.get("router_batch_size", 256))
    generator = torch.Generator().manual_seed(seed)
    loader_args = {
        "batch_size": batch_size,
        "num_workers": int(config["experiment"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    target_train_loader = DataLoader(
        target_train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    target_dev_loader = DataLoader(
        target_dev_dataset, shuffle=False, **loader_args
    )
    pair_train_loader = DataLoader(
        pair_train_dataset,
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    pair_dev_loader = DataLoader(
        pair_dev_dataset, shuffle=False, **loader_args
    )

    model = ContextUtilityRouter.from_config(
        config,
        embedding_dim=int(store["embedding_dim"]),
        num_classes=len(store["label_to_id"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(settings.get("router_learning_rate", 3e-4)),
        weight_decay=float(settings.get("router_weight_decay", 1e-3)),
    )
    counts = torch.bincount(
        torch.tensor(
            [int(record["utility_class"]) for record in target_train],
            dtype=torch.long,
        ),
        minlength=ContextUtilityRouter.NUM_UTILITY_CLASSES,
    ).float()
    class_weights = counts.clamp_min(1.0).pow(-0.5)
    class_weights = (class_weights / class_weights.mean()).to(device)
    epochs = int(settings.get("router_epochs", 30))
    patience = int(settings.get("router_patience", 6))
    grad_clip = float(settings.get("router_grad_clip", 1.0))
    checkpoint_path = _router_checkpoint(root, seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        target_running = 0.0
        target_batches = 0
        for batch in target_train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _target_loss(
                model, batch, class_weights, settings
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            target_running += float(loss.detach())
            target_batches += 1
        pair_running = 0.0
        pair_batches = 0
        for batch in pair_train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = _pair_loss(model, batch, settings)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            pair_running += float(loss.detach())
            pair_batches += 1
        dev = _router_dev_metrics(
            model, target_dev_loader, pair_dev_loader, device
        )
        epoch_record = {
            "epoch": epoch,
            "target_train_loss": target_running / max(target_batches, 1),
            "pair_train_loss": pair_running / max(pair_batches, 1),
            **dev,
        }
        history.append(epoch_record)
        print(
            f"[Experiment13 router] seed={seed} epoch={epoch:03d} "
            f"targetMAE={dev['target_mae']:.4f} "
            f"pairMAE={dev['pair_mae']:.4f} "
            f"macroF1={dev['target_macro_f1']:.4f} "
            f"selection={dev['selection_loss']:.4f}"
        )
        if dev["selection_loss"] < best:
            best = dev["selection_loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "dev_metrics": dev,
                    "embedding_dim": int(store["embedding_dim"]),
                    "num_classes": len(store["label_to_id"]),
                    "label_to_id": store["label_to_id"],
                    "settings": copy.deepcopy(settings),
                    "cue_examples": len(cue_labels),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return {
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "best_selection_loss": best,
        "epochs_run": len(history),
        "target_train_records": len(target_train),
        "target_dev_records": len(target_dev),
        "pair_train_records": len(pair_train),
        "pair_dev_records": len(pair_dev),
        "cue_examples": len(cue_labels),
        "history": history,
    }


def _train_routers(
    config: dict[str, Any],
    device: str | None,
) -> dict[str, Any]:
    settings = _settings(config)
    root = _artifact_dir(config)
    device_obj = get_device(device)
    reuse = bool(settings.get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seed_list(config):
        checkpoint = _router_checkpoint(root, seed)
        if reuse and checkpoint.exists():
            stored = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            runs.append(
                {
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "reused": True,
                    "best_selection_loss": stored.get(
                        "dev_metrics", {}
                    ).get("selection_loss"),
                }
            )
        else:
            seed_everything(
                seed,
                bool(config["experiment"].get("deterministic", True)),
            )
            runs.append(
                {
                    **_train_router_seed(
                        config, seed, device_obj
                    ),
                    "reused": False,
                }
            )
    report = {
        "experiment": "13 Context Utility Routing",
        "stage": "train-routers",
        "runs": runs,
    }
    save_json(report, root / "reports" / "router_training.json")
    return report


def _load_router(
    config: dict[str, Any],
    seed: int,
    store: dict[str, Any],
    device: torch.device,
) -> tuple[ContextUtilityRouter, dict[str, Any]]:
    checkpoint_path = _router_checkpoint(_artifact_dir(config), seed)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing router checkpoint: {checkpoint_path}. "
            "Run Experiment 13 train-routers first."
        )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model = ContextUtilityRouter.from_config(
        config,
        embedding_dim=int(checkpoint.get("embedding_dim", store["embedding_dim"])),
        num_classes=int(
            checkpoint.get("num_classes", len(store["label_to_id"]))
        ),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def _anchor_entropy(logits: torch.Tensor) -> float:
    probability = torch.softmax(logits.float(), dim=-1).clamp_min(1e-8)
    entropy = -(probability * probability.log()).sum()
    return float(entropy / math.log(probability.numel()))


def _router_target_prediction(
    router: ContextUtilityRouter,
    dialogue: dict[str, Any],
    target_turn: int,
    max_history: int,
    device: torch.device,
) -> dict[str, float | int]:
    item = {
        key: value.unsqueeze(0).to(device)
        for key, value in _target_tensors(
            dialogue, target_turn, max_history
        ).items()
    }
    with torch.no_grad():
        output = router.forward_target(
            item["current"],
            item["same_history"],
            item["other_history"],
            item["anchor_logits"],
            item["scalar_features"],
        )
        probability = torch.softmax(
            output["class_logits"].float(), dim=-1
        )[0]
    return {
        "predicted_utility": float(output["utility"][0]),
        "predicted_class": int(probability.argmax()),
        "helpful_probability": float(
            probability[ContextUtilityRouter.HELPFUL]
        ),
        "harmful_probability": float(
            probability[ContextUtilityRouter.HARMFUL]
        ),
        "cue_slow_probability": float(
            torch.sigmoid(output["cue_slow_logit"][0])
        ),
    }


def _select_history(
    router: ContextUtilityRouter,
    dialogue: dict[str, Any],
    target_turn: int,
    max_history: int,
    top_k: int,
    threshold: float,
    min_keep: int,
    device: torch.device,
) -> tuple[list[int], list[dict[str, float | int]]]:
    start = max(0, target_turn - max_history)
    candidates = list(range(start, target_turn))
    if not candidates:
        return [], []
    tensors = [
        _pair_tensors(
            dialogue, target_turn, history_turn, max_history
        )
        for history_turn in candidates
    ]
    batch = {
        key: torch.stack([item[key] for item in tensors]).to(device)
        for key in tensors[0]
    }
    with torch.no_grad():
        scores = router.forward_pair(
            batch["current"],
            batch["history"],
            batch["current_anchor_logits"],
            batch["history_anchor_logits"],
            batch["scalar_features"],
        ).float().cpu()
    ranked = sorted(
        zip(candidates, scores.tolist()),
        key=lambda item: (float(item[1]), -int(item[0])),
        reverse=True,
    )
    selected = [
        int(history_turn)
        for history_turn, score in ranked
        if float(score) > threshold
    ][:top_k]
    if len(selected) < min_keep:
        selected = [
            int(history_turn)
            for history_turn, _ in ranked[: min(min_keep, top_k)]
        ]
    diagnostics = [
        {"history_turn": int(history_turn), "predicted_utility": float(score)}
        for history_turn, score in ranked
    ]
    return sorted(selected), diagnostics


def _probability_from_logits(logits: list[float], temperature: float = 1.0) -> list[float]:
    tensor = torch.tensor(logits, dtype=torch.float32) / max(
        float(temperature), 1e-6
    )
    return torch.softmax(tensor, dim=-1).tolist()


def _metrics_for_records(
    records: list[dict[str, Any]],
    variant: str,
    id_to_label: dict[int, str],
    temperature: float = 1.0,
) -> dict[str, Any]:
    labels = [int(record["gold_label"]) for record in records]
    logits = [record["variant_logits"][variant] for record in records]
    probabilities = [
        _probability_from_logits(value, temperature) for value in logits
    ]
    predictions = [
        int(np.argmax(probability)) for probability in probabilities
    ]
    result = classification_metrics(labels, predictions, id_to_label)
    result["probability_quality"] = _probability_quality(
        labels, probabilities, len(id_to_label)
    )
    result["utterances"] = len(labels)
    return result


def _subset_metrics(
    records: list[dict[str, Any]],
    variant: str,
    id_to_label: dict[int, str],
    temperature: float = 1.0,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ("helpful", "harmful", "uncertain"):
        subset = [
            record
            for record in records
            if record["true_utility_group"] == group
        ]
        output[group] = (
            _metrics_for_records(
                subset, variant, id_to_label, temperature
            )
            if subset
            else {"utterances": 0}
        )
    return output


def _pair_selector_metrics(
    config: dict[str, Any],
    router: ContextUtilityRouter,
    store: dict[str, Any],
    seed: int,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    settings = _settings(config)
    _, pair_path = _label_paths(_artifact_dir(config), seed, split)
    if not pair_path.exists():
        return {
            "available": False,
            "reason": "Run build-validation-labels for selector diagnostics.",
        }
    records = _read_jsonl(pair_path)
    dialogues = _dialogue_lookup(list(store["splits"][split]))
    dataset = _PairUtilityDataset(
        records,
        dialogues,
        int(settings.get("max_history_turns", 12)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(settings.get("router_batch_size", 256)),
        shuffle=False,
        num_workers=int(config["experiment"].get("num_workers", 0)),
    )
    gold: list[float] = []
    prediction: list[float] = []
    classes: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            value = router.forward_pair(
                batch["current"],
                batch["history"],
                batch["current_anchor_logits"],
                batch["history_anchor_logits"],
                batch["scalar_features"],
            )
            gold.extend(batch["pair_utility"].cpu().tolist())
            prediction.extend(value.float().cpu().tolist())
            classes.extend(batch["utility_class"].cpu().tolist())
    gold_array = np.asarray(gold)
    prediction_array = np.asarray(prediction)
    helpful = np.asarray(
        [value == ContextUtilityRouter.HELPFUL for value in classes],
        dtype=int,
    )
    harmful = np.asarray(
        [value == ContextUtilityRouter.HARMFUL for value in classes],
        dtype=int,
    )
    return {
        "available": True,
        "pairs": len(records),
        "mae": float(np.mean(np.abs(gold_array - prediction_array))),
        "sign_accuracy": float(
            np.mean(np.sign(gold_array) == np.sign(prediction_array))
        ),
        "helpful_average_precision": (
            float(average_precision_score(helpful, prediction_array))
            if len(set(helpful.tolist())) > 1
            else None
        ),
        "harmful_average_precision": (
            float(average_precision_score(harmful, -prediction_array))
            if len(set(harmful.tolist())) > 1
            else None
        ),
    }


def _evaluate_seed(
    config: dict[str, Any],
    seed: int,
    split: str,
    device: torch.device,
    save: bool = True,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Experiment 13 evaluation supports validation/test")
    settings = _settings(config)
    root = _artifact_dir(config)
    started_at = time.perf_counter()
    store = load_feature_store(_feature_path(config, seed, split))
    _validate_current_logit_features(store, len(store["label_to_id"]))
    dialogues = list(store["splits"][split])
    context_checkpoint = _teacher_checkpoint(root, seed, None)
    context_model, context_payload = _load_context_model(
        config, store, context_checkpoint, device
    )
    router, router_payload = _load_router(
        config, seed, store, device
    )
    max_history = int(settings.get("max_history_turns", 12))
    top_k = int(settings.get("selector_top_k", 4))
    pair_threshold = float(settings.get("selector_threshold", 0.0))
    min_keep = int(settings.get("selector_min_keep", 1))
    helpful_threshold = float(
        settings.get("router_helpful_probability_threshold", 0.5)
    )
    entropy_threshold = float(
        settings.get("entropy_router_threshold", 0.6)
    )
    utility_threshold = float(settings.get("utility_threshold", 0.01))
    use_amp = bool(settings.get("use_amp", True)) and device.type == "cuda"

    pending_sparse: list[dict[str, Any]] = []
    pending_indices: list[int] = []
    records: list[dict[str, Any]] = []
    for dialogue in tqdm(
        dialogues,
        desc=f"13 evaluate {split} seed={seed}",
        leave=False,
    ):
        full_logits = _baseline_logits(
            context_model, dialogue, device, use_amp
        )
        length = int(dialogue["embeddings"].shape[0])
        no_context_by_turn: dict[int, torch.Tensor] = {
            0: full_logits[0].float()
        }
        if length > 1:
            no_context_samples = [
                _selected_prefix(dialogue, target_turn, [])
                for target_turn in range(1, length)
            ]
            no_context_logits = _forward_target_logits(
                context_model,
                no_context_samples,
                int(settings.get("counterfactual_batch_size", 128)),
                device,
                use_amp,
            )
            no_context_by_turn.update(
                {
                    target_turn: no_context_logits[target_turn - 1].float()
                    for target_turn in range(1, length)
                }
            )
        utterance_ids = dialogue.get(
            "utterance_ids", list(range(length))
        )
        for target_turn in range(length):
            gold = int(dialogue["labels"][target_turn])
            anchor_logits = dialogue["current_logits"][target_turn].float()
            full_target_logits = full_logits[target_turn].float()
            anchor_nll = -float(
                torch.log_softmax(anchor_logits, dim=-1)[gold]
            )
            full_nll = -float(
                torch.log_softmax(full_target_logits, dim=-1)[gold]
            )
            no_context_logits = no_context_by_turn[target_turn]
            no_context_nll = -float(
                torch.log_softmax(no_context_logits, dim=-1)[gold]
            )
            true_utility = no_context_nll - full_nll
            branch_utility = anchor_nll - full_nll
            true_class = _utility_class(true_utility, utility_threshold)
            if true_class == ContextUtilityRouter.HELPFUL:
                group = "helpful"
            elif true_class == ContextUtilityRouter.HARMFUL:
                group = "harmful"
            else:
                group = "uncertain"

            if target_turn == 0:
                router_output: dict[str, float | int] = {
                    "predicted_utility": 0.0,
                    "predicted_class": ContextUtilityRouter.UNCERTAIN,
                    "helpful_probability": 0.0,
                    "harmful_probability": 0.0,
                    "cue_slow_probability": 0.0,
                }
                selected: list[int] = []
                pair_diagnostics: list[dict[str, float | int]] = []
            else:
                router_output = _router_target_prediction(
                    router,
                    dialogue,
                    target_turn,
                    max_history,
                    device,
                )
                selected, pair_diagnostics = _select_history(
                    router,
                    dialogue,
                    target_turn,
                    max_history,
                    top_k,
                    pair_threshold,
                    min_keep,
                    device,
                )
            use_utility = (
                float(router_output["helpful_probability"])
                >= helpful_threshold
            )
            use_entropy = _anchor_entropy(anchor_logits) >= entropy_threshold
            record = {
                "seed": seed,
                "split": split,
                "dialogue_id": str(dialogue.get("dialogue_id")),
                "target_utterance_id": str(utterance_ids[target_turn]),
                "target_turn": target_turn,
                "gold_label": gold,
                "anchor_nll": anchor_nll,
                "no_context_nll": no_context_nll,
                "full_context_nll": full_nll,
                "true_context_utility": true_utility,
                "true_branch_utility": branch_utility,
                "true_utility_class": true_class,
                "true_utility_group": group,
                "anchor_entropy": _anchor_entropy(anchor_logits),
                **router_output,
                "entropy_used_context": use_entropy,
                "utility_used_context": use_utility,
                "selected_history_turns": selected,
                "selected_history_count": len(selected),
                "available_history_count": min(target_turn, max_history),
                "pair_diagnostics": pair_diagnostics,
                "variant_logits": {
                    "anchor": anchor_logits.tolist(),
                    "full_history": full_target_logits.tolist(),
                    "entropy_router": (
                        full_target_logits if use_entropy else anchor_logits
                    ).tolist(),
                    "utility_router": (
                        full_target_logits if use_utility else anchor_logits
                    ).tolist(),
                    "utility_sparse": anchor_logits.tolist(),
                    "oracle": (
                        full_target_logits
                        if full_nll < anchor_nll
                        else anchor_logits
                    ).tolist(),
                },
            }
            records.append(record)
            if target_turn > 0:
                pending_indices.append(len(records) - 1)
                pending_sparse.append(
                    _selected_prefix(
                        dialogue, target_turn, selected
                    )
                )

    if pending_sparse:
        sparse_logits = _forward_target_logits(
            context_model,
            pending_sparse,
            int(settings.get("counterfactual_batch_size", 128)),
            device,
            use_amp,
        )
        for index, value in zip(pending_indices, sparse_logits):
            if bool(records[index]["utility_used_context"]):
                records[index]["variant_logits"]["utility_sparse"] = (
                    value.float().tolist()
                )

    id_to_label = {
        int(value): str(key)
        for key, value in store["label_to_id"].items()
    }
    variants: dict[str, Any] = {}
    for variant in VARIANT_LABELS:
        metrics = _metrics_for_records(records, variant, id_to_label)
        metrics["utility_subsets"] = _subset_metrics(
            records, variant, id_to_label
        )
        variants[variant] = metrics

    eligible_records = [
        record for record in records if int(record["target_turn"]) > 0
    ]
    router_gold = [
        int(record["true_utility_class"]) for record in eligible_records
    ]
    router_prediction = [
        int(record["predicted_class"]) for record in eligible_records
    ]
    helpful_gold = [
        int(value == ContextUtilityRouter.HELPFUL)
        for value in router_gold
    ]
    harmful_gold = [
        int(value == ContextUtilityRouter.HARMFUL)
        for value in router_gold
    ]
    helpful_score = [
        float(record["helpful_probability"])
        for record in eligible_records
    ]
    harmful_score = [
        float(record["harmful_probability"])
        for record in eligible_records
    ]
    router_metrics = {
        "targets": len(eligible_records),
        "three_class_macro_f1": float(
            f1_score(
                router_gold,
                router_prediction,
                labels=list(
                    range(ContextUtilityRouter.NUM_UTILITY_CLASSES)
                ),
                average="macro",
                zero_division=0,
            )
        ),
        "helpful_average_precision": (
            float(average_precision_score(helpful_gold, helpful_score))
            if len(set(helpful_gold)) > 1
            else None
        ),
        "harmful_average_precision": (
            float(average_precision_score(harmful_gold, harmful_score))
            if len(set(harmful_gold)) > 1
            else None
        ),
        "context_selection_rate": float(
            np.mean(
                [
                    bool(record["utility_used_context"])
                    for record in eligible_records
                ]
            )
        ),
        "mean_selected_history": float(
            np.mean(
                [
                    int(record["selected_history_count"])
                    for record in eligible_records
                ]
            )
        ),
        "mean_available_history": float(
            np.mean(
                [
                    int(record["available_history_count"])
                    for record in eligible_records
                ]
            )
        ),
        "selector_retention_rate": float(
            sum(
                int(record["selected_history_count"])
                for record in eligible_records
            )
            / max(
                sum(
                    int(record["available_history_count"])
                    for record in eligible_records
                ),
                1,
            )
        ),
        "effective_retention_rate": float(
            sum(
                int(record["selected_history_count"])
                if bool(record["utility_used_context"])
                else 0
                for record in eligible_records
            )
            / max(
                sum(
                    int(record["available_history_count"])
                    for record in eligible_records
                ),
                1,
            )
        ),
    }
    selector_metrics = _pair_selector_metrics(
        config, router, store, seed, split, device
    )
    elapsed_seconds = time.perf_counter() - started_at
    report = {
        "experiment": "13 Context Utility Routing",
        "stage": f"evaluate-{split}",
        "seed": seed,
        "split": split,
        "context_checkpoint": str(context_checkpoint),
        "context_epoch": int(context_payload.get("epoch", -1)),
        "router_checkpoint": str(_router_checkpoint(root, seed)),
        "router_epoch": int(router_payload.get("epoch", -1)),
        "settings": {
            "max_history_turns": max_history,
            "selector_top_k": top_k,
            "selector_threshold": pair_threshold,
            "selector_min_keep": min_keep,
            "router_helpful_probability_threshold": helpful_threshold,
            "entropy_router_threshold": entropy_threshold,
            "utility_threshold": utility_threshold,
        },
        "variants": variants,
        "router": router_metrics,
        "selector": selector_metrics,
        "efficiency": {
            "elapsed_seconds": elapsed_seconds,
            "utterances_per_second": len(records)
            / max(elapsed_seconds, 1e-8),
            "context_trainable_parameters": sum(
                parameter.numel()
                for parameter in context_model.parameters()
                if parameter.requires_grad
            ),
            "router_trainable_parameters": sum(
                parameter.numel()
                for parameter in router.parameters()
                if parameter.requires_grad
            ),
            "effective_history_retention": router_metrics[
                "effective_retention_rate"
            ],
        },
        "test_used_for_selection": False,
    }
    if save:
        save_json(report, _seed_report(root, seed, split))
        _write_jsonl(records, _seed_records(root, seed, split))
    return report


def _evaluate_all(
    config: dict[str, Any],
    split: str,
    device: str | None,
) -> dict[str, Any]:
    device_obj = get_device(device)
    reports = [
        _evaluate_seed(config, seed, split, device_obj)
        for seed in _seed_list(config)
    ]
    return {
        "experiment": "13 Context Utility Routing",
        "stage": f"evaluate-{split}",
        "split": split,
        "reports": reports,
    }


def _scalar_summary(values: Iterable[float | int | None]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return {
        "values": finite,
        "mean": mean(finite) if finite else None,
        "std": pstdev(finite) if finite else None,
        "count": len(finite),
    }


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _fit_scalar_temperature(
    records: list[dict[str, Any]],
    variant: str,
) -> float:
    if not records:
        return 1.0
    logits = torch.tensor(
        [record["variant_logits"][variant] for record in records],
        dtype=torch.float64,
    )
    labels = torch.tensor(
        [int(record["gold_label"]) for record in records],
        dtype=torch.long,
    )
    temperatures = torch.logspace(
        math.log10(0.25),
        math.log10(5.0),
        steps=601,
        dtype=torch.float64,
    )
    best_temperature = 1.0
    best_nll = float("inf")
    for chunk in temperatures.split(75):
        scaled = logits.unsqueeze(0) / chunk[:, None, None]
        log_probability = torch.log_softmax(scaled, dim=-1)
        row = torch.arange(labels.numel())
        nll = -log_probability[:, row, labels].mean(dim=-1)
        index = int(nll.argmin())
        value = float(nll[index])
        if value < best_nll:
            best_nll = value
            best_temperature = float(chunk[index])
    return best_temperature


def _cross_fitted_temperature(
    records: list[dict[str, Any]],
    variant: str,
    id_to_label: dict[int, str],
    seed: int,
    folds: int,
) -> dict[str, Any]:
    if folds < 2:
        raise ValueError("calibration_folds must be at least two")
    probability_by_index: list[list[float] | None] = [None] * len(records)
    fold_temperatures: list[float] = []
    for fold in range(folds):
        fit_records = [
            record
            for record in records
            if _dialogue_fold(
                {"dialogue_id": record["dialogue_id"]},
                folds,
                seed,
            )
            != fold
        ]
        held_indices = [
            index
            for index, record in enumerate(records)
            if _dialogue_fold(
                {"dialogue_id": record["dialogue_id"]},
                folds,
                seed,
            )
            == fold
        ]
        if not fit_records or not held_indices:
            continue
        temperature = _fit_scalar_temperature(fit_records, variant)
        fold_temperatures.append(temperature)
        for index in held_indices:
            probability_by_index[index] = _probability_from_logits(
                records[index]["variant_logits"][variant],
                temperature,
            )
    if any(value is None for value in probability_by_index):
        raise ValueError(
            "Cross-fitted calibration left validation examples uncovered"
        )
    probabilities = [
        value for value in probability_by_index if value is not None
    ]
    labels = [int(record["gold_label"]) for record in records]
    predictions = [int(np.argmax(value)) for value in probabilities]
    metrics = classification_metrics(labels, predictions, id_to_label)
    metrics["probability_quality"] = _probability_quality(
        labels, probabilities, len(id_to_label)
    )
    metrics["utterances"] = len(labels)
    return {
        "folds": folds,
        "fold_temperatures": fold_temperatures,
        "deployment_temperature": _fit_scalar_temperature(records, variant),
        "cross_fitted_metrics": metrics,
    }


def _variant_summary(
    reports: list[dict[str, Any]],
    variant: str,
    calibrations: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    def values(path: tuple[str, ...]) -> list[Any]:
        return [
            _nested_value(report["variants"][variant], path)
            for report in reports
        ]

    calibrated = [
        calibrations[int(report["seed"])]["cross_fitted_metrics"]
        for report in reports
    ]
    return {
        "label": VARIANT_LABELS[variant],
        "weighted_f1": _scalar_summary(values(("weighted_f1",))),
        "macro_f1": _scalar_summary(values(("macro_f1",))),
        "accuracy": _scalar_summary(values(("accuracy",))),
        "nll": _scalar_summary(values(("probability_quality", "nll"))),
        "brier": _scalar_summary(values(("probability_quality", "brier"))),
        "ece": _scalar_summary(values(("probability_quality", "ece"))),
        "cross_fitted_calibrated_nll": _scalar_summary(
            [
                item["probability_quality"]["nll"]
                for item in calibrated
            ]
        ),
        "cross_fitted_calibrated_brier": _scalar_summary(
            [
                item["probability_quality"]["brier"]
                for item in calibrated
            ]
        ),
        "cross_fitted_calibrated_ece": _scalar_summary(
            [
                item["probability_quality"]["ece"]
                for item in calibrated
            ]
        ),
        "deployment_temperatures": _scalar_summary(
            [
                calibrations[int(report["seed"])]
                ["deployment_temperature"]
                for report in reports
            ]
        ),
    }


def _subset_nll(
    report: dict[str, Any],
    variant: str,
    group: str,
) -> float | None:
    value = _nested_value(
        report,
        (
            "variants",
            variant,
            "utility_subsets",
            group,
            "probability_quality",
            "nll",
        ),
    )
    return None if value is None else float(value)


def _candidate_eligibility(
    reports: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    variant: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    max_drop = float(settings.get("max_classification_drop", 0.003))
    min_router_ap = float(settings.get("min_router_helpful_ap", 0.55))
    min_selector_ap = float(settings.get("min_selector_helpful_ap", 0.50))
    max_retention = float(settings.get("max_sparse_retention", 0.50))
    anchor = summaries["anchor"]
    candidate = summaries[variant]
    helpful_advantages: list[float] = []
    harmful_advantages: list[float] = []
    macro_advantages: list[float] = []
    router_ap: list[float] = []
    selector_ap: list[float] = []
    retention: list[float] = []
    for report in reports:
        anchor_helpful = _subset_nll(report, "anchor", "helpful")
        candidate_helpful = _subset_nll(report, variant, "helpful")
        full_harmful = _subset_nll(report, "full_history", "harmful")
        candidate_harmful = _subset_nll(report, variant, "harmful")
        if anchor_helpful is not None and candidate_helpful is not None:
            helpful_advantages.append(anchor_helpful - candidate_helpful)
        if full_harmful is not None and candidate_harmful is not None:
            harmful_advantages.append(full_harmful - candidate_harmful)
        macro_advantages.append(
            float(report["variants"][variant]["macro_f1"])
            - float(report["variants"]["anchor"]["macro_f1"])
        )
        helpful_ap = report["router"].get("helpful_average_precision")
        if helpful_ap is not None:
            router_ap.append(float(helpful_ap))
        selector_value = report["selector"].get(
            "helpful_average_precision"
        )
        if selector_value is not None:
            selector_ap.append(float(selector_value))
        retention.append(
            float(report["router"].get("effective_retention_rate", 1.0))
        )

    weighted_drop = float(anchor["weighted_f1"]["mean"]) - float(
        candidate["weighted_f1"]["mean"]
    )
    macro_drop = float(anchor["macro_f1"]["mean"]) - float(
        candidate["macro_f1"]["mean"]
    )
    helpful_summary = _scalar_summary(helpful_advantages)
    harmful_summary = _scalar_summary(harmful_advantages)
    router_summary = _scalar_summary(router_ap)
    selector_summary = _scalar_summary(selector_ap)
    retention_summary = _scalar_summary(retention)
    checks = {
        "classification_preserved": (
            weighted_drop <= max_drop and macro_drop <= max_drop
        ),
        "macro_nonnegative_in_two_seeds": sum(
            value >= 0.0 for value in macro_advantages
        )
        >= 2,
        "helpful_context_used": (
            helpful_summary["mean"] is not None
            and helpful_summary["mean"] > 0.0
            and sum(value > 0.0 for value in helpful_advantages) >= 2
        ),
        "harmful_context_avoided": (
            harmful_summary["mean"] is not None
            and harmful_summary["mean"] > 0.0
            and sum(value > 0.0 for value in harmful_advantages) >= 2
        ),
        "router_predictive": (
            router_summary["mean"] is not None
            and router_summary["mean"] >= min_router_ap
        ),
        "selector_predictive": (
            variant != "utility_sparse"
            or (
                selector_summary["mean"] is not None
                and selector_summary["mean"] >= min_selector_ap
            )
        ),
        "sparse_enough": (
            variant != "utility_sparse"
            or (
                retention_summary["mean"] is not None
                and retention_summary["mean"] <= max_retention
            )
        ),
    }
    return {
        "weighted_f1_drop_vs_anchor": weighted_drop,
        "macro_f1_drop_vs_anchor": macro_drop,
        "macro_f1_advantage_by_seed": macro_advantages,
        "helpful_subset_nll_advantage_vs_anchor": helpful_summary,
        "harmful_subset_nll_advantage_vs_full_history": harmful_summary,
        "router_helpful_average_precision": router_summary,
        "selector_helpful_average_precision": selector_summary,
        "effective_history_retention": retention_summary,
        "checks": checks,
        "eligible": all(checks.values()),
    }


def _summarize_validation(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    root = _artifact_dir(config)
    seeds = _seed_list(config)
    report_paths = [
        _seed_report(root, seed, "validation") for seed in seeds
    ]
    record_paths = [
        _seed_records(root, seed, "validation") for seed in seeds
    ]
    missing = [
        str(path)
        for path in [*report_paths, *record_paths]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Run Experiment 13 evaluate-validation first; missing: "
            + ", ".join(missing)
        )
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in report_paths
    ]
    records_by_seed = {
        seed: _read_jsonl(path)
        for seed, path in zip(seeds, record_paths)
    }
    store = load_feature_store(_feature_path(config, seeds[0]))
    id_to_label = {
        int(value): str(key)
        for key, value in store["label_to_id"].items()
    }
    calibration_folds = int(settings.get("calibration_folds", 5))
    calibration: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_LABELS:
        per_seed = {
            seed: _cross_fitted_temperature(
                records_by_seed[seed],
                variant,
                id_to_label,
                seed=int(settings.get("calibration_fold_seed", 13015))
                + seed,
                folds=calibration_folds,
            )
            for seed in seeds
        }
        calibration[variant] = {
            str(seed): value for seed, value in per_seed.items()
        }
        summaries[variant] = _variant_summary(
            reports, variant, per_seed
        )

    eligibility = {
        variant: _candidate_eligibility(
            reports, summaries, variant, settings
        )
        for variant in CANDIDATE_VARIANTS
    }
    eligible = [
        variant
        for variant in CANDIDATE_VARIANTS
        if eligibility[variant]["eligible"]
    ]
    selection_scores = {
        variant: (
            0.4 * float(summaries[variant]["weighted_f1"]["mean"])
            + 0.6 * float(summaries[variant]["macro_f1"]["mean"])
        )
        for variant in CANDIDATE_VARIANTS
    }
    fixed_candidate = settings.get("fixed_candidate")
    if fixed_candidate is not None:
        fixed_candidate = str(fixed_candidate)
        if fixed_candidate not in CANDIDATE_VARIANTS:
            raise ValueError(
                "experiment13.fixed_candidate must be utility_router or "
                "utility_sparse"
            )
        selected = (
            fixed_candidate
            if fixed_candidate in eligible
            else None
        )
        selection_mode = "externally_frozen_candidate"
    else:
        selected = (
            max(eligible, key=lambda value: selection_scores[value])
            if eligible
            else None
        )
        selection_mode = "meld_development_selection"
    status = "ready" if selected is not None else "no_eligible_candidate"
    output = {
        "status": status,
        "experiment": "13 Context Utility Routing",
        "split": "validation",
        "seeds": seeds,
        "utility_definition": (
            "NLL(no-history OOF teacher) - NLL(full-history OOF teacher)"
        ),
        "variant_summaries": summaries,
        "calibration": calibration,
        "eligibility": eligibility,
        "selection_scores": selection_scores,
        "selected_variant": selected,
        "selected_label": VARIANT_LABELS.get(selected),
        "selection_mode": selection_mode,
        "fixed_candidate": fixed_candidate,
        "selection_rule": {
            "score": "0.4 * weighted_F1 + 0.6 * macro_F1",
            "max_classification_drop": float(
                settings.get("max_classification_drop", 0.003)
            ),
            "min_router_helpful_ap": float(
                settings.get("min_router_helpful_ap", 0.55)
            ),
            "min_selector_helpful_ap": float(
                settings.get("min_selector_helpful_ap", 0.50)
            ),
            "max_sparse_retention": float(
                settings.get("max_sparse_retention", 0.50)
            ),
            "tie_break": "higher pre-registered validation score",
        },
        "efficiency": {
            "utterances_per_second": _scalar_summary(
                [
                    report["efficiency"]["utterances_per_second"]
                    for report in reports
                ]
            ),
            "context_trainable_parameters": _scalar_summary(
                [
                    report["efficiency"][
                        "context_trainable_parameters"
                    ]
                    for report in reports
                ]
            ),
            "router_trainable_parameters": _scalar_summary(
                [
                    report["efficiency"][
                        "router_trainable_parameters"
                    ]
                    for report in reports
                ]
            ),
        },
        "emotioncue_auxiliary": {
            "enabled": bool(settings.get("emotioncue_train_file")),
            "loss_weight": float(
                settings.get("emotioncue_aux_weight", 0.0)
            ),
            "interpretation": (
                "train-only weak supervision; never a gold relevance label"
            ),
        },
        "test_locked": True,
        "test_read": False,
    }
    path = root / "reports" / "validation_summary.json"
    save_json(output, path)
    output["report_file"] = str(path)
    return output


def _configuration_check(config: dict[str, Any]) -> dict[str, Any]:
    settings = _settings(config)
    seeds = _seed_list(config)
    development_files = {
        seed: _feature_path(config, seed) for seed in seeds
    }
    test_files = {
        seed: _feature_path(config, seed, "test") for seed in seeds
    }
    missing_files = [
        str(path)
        for path in {*development_files.values(), *test_files.values()}
        if not path.exists()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing feature/logit cache(s): " + ", ".join(missing_files)
        )
    stores = {
        seed: load_feature_store(path)
        for seed, path in development_files.items()
    }
    for seed, store in stores.items():
        _validate_current_logit_features(store, len(store["label_to_id"]))
        missing_splits = {
            split for split in ("train", "validation")
            if split not in store["splits"]
        }
        if not settings.get("test_feature_file_pattern"):
            missing_splits.update(
                split for split in ("test",)
                if split not in store["splits"]
            )
        if missing_splits:
            raise ValueError(
                f"Seed {seed} feature store is missing splits: "
                f"{sorted(missing_splits)}"
            )
    store = stores[seeds[0]]
    reference_labels = store["label_to_id"]
    if any(
        item["label_to_id"] != reference_labels
        for item in stores.values()
    ):
        raise ValueError("Per-seed feature stores disagree on label IDs")
    folds = int(settings.get("oof_folds", 3))
    if folds < 2:
        raise ValueError("experiment13.oof_folds must be at least two")
    fold_seed = int(settings.get("oof_fold_seed", 13013))
    fold_counts = {
        str(fold): sum(
            _dialogue_fold(dialogue, folds, fold_seed) == fold
            for dialogue in store["splits"]["train"]
        )
        for fold in range(folds)
    }
    if any(value == 0 for value in fold_counts.values()):
        raise ValueError(f"An OOF fold is empty: {fold_counts}")
    if bool(config["model"].get("use_hitft", False)):
        raise ValueError("Experiment 13 context teachers must disable HiTFT")
    return {
        "status": "passed",
        "experiment": "13 Context Utility Routing",
        "development_feature_files": {
            str(seed): {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for seed, path in development_files.items()
        },
        "test_feature_files": {
            str(seed): str(path)
            for seed, path in test_files.items()
        },
        "embedding_dim": int(store["embedding_dim"]),
        "labels": store["label_to_id"],
        "split_dialogues": {
            split: len(dialogues)
            for split, dialogues in store["splits"].items()
        },
        "seeds": seeds,
        "oof_folds": folds,
        "oof_fold_counts": fold_counts,
        "test_metrics_read": False,
    }


def _check_final_inputs(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    summary_path = root / "reports" / "validation_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            "Run Experiment 13 all-validation before check-final"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ready" or not summary.get(
        "selected_variant"
    ):
        raise RuntimeError(
            "No pre-registered Experiment 13 candidate passed validation; "
            "the test split remains locked."
        )
    inputs: list[dict[str, Any]] = []
    for seed in _seed_list(config):
        teacher = _teacher_checkpoint(root, seed, None)
        router = _router_checkpoint(root, seed)
        for path in (teacher, router):
            if not path.exists():
                raise FileNotFoundError(path)
        deployment_temperature = _nested_value(
            summary,
            (
                "calibration",
                str(summary["selected_variant"]),
                str(seed),
                "deployment_temperature",
            ),
        )
        anchor_temperature = _nested_value(
            summary,
            (
                "calibration",
                "anchor",
                str(seed),
                "deployment_temperature",
            ),
        )
        if deployment_temperature is None or anchor_temperature is None:
            raise ValueError("Validation temperatures are incomplete")
        inputs.append(
            {
                "seed": seed,
                "development_feature": str(
                    _feature_path(config, seed)
                ),
                "development_feature_sha256": _sha256(
                    _feature_path(config, seed)
                ),
                "test_feature": str(
                    _feature_path(config, seed, "test")
                ),
                "test_feature_sha256": _sha256(
                    _feature_path(config, seed, "test")
                ),
                "teacher": str(teacher),
                "teacher_sha256": _sha256(teacher),
                "router": str(router),
                "router_sha256": _sha256(router),
                "selected_temperature": float(deployment_temperature),
                "anchor_temperature": float(anchor_temperature),
            }
        )
    manifest = {
        "status": "ready",
        "experiment": "13 Context Utility Routing",
        "selected_variant": summary["selected_variant"],
        "selected_label": summary["selected_label"],
        "validation_summary": str(summary_path),
        "validation_summary_sha256": _sha256(summary_path),
        "config_file": config.get("_config_path"),
        "config_sha256": (
            _sha256(Path(config["_config_path"]))
            if config.get("_config_path")
            else None
        ),
        "inputs": inputs,
        "test_read": False,
    }
    save_json(manifest, root / "reports" / "final_manifest.json")
    return manifest


def _metric_summary(
    per_seed: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "weighted_f1": _scalar_summary(
            [value["weighted_f1"] for value in per_seed]
        ),
        "macro_f1": _scalar_summary(
            [value["macro_f1"] for value in per_seed]
        ),
        "accuracy": _scalar_summary(
            [value["accuracy"] for value in per_seed]
        ),
        "nll": _scalar_summary(
            [
                value["probability_quality"]["nll"]
                for value in per_seed
            ]
        ),
        "brier": _scalar_summary(
            [
                value["probability_quality"]["brier"]
                for value in per_seed
            ]
        ),
        "ece": _scalar_summary(
            [
                value["probability_quality"]["ece"]
                for value in per_seed
            ]
        ),
    }


def _metric_from_sample(
    sample: list[tuple[int, dict[str, Any]]],
    variant: str,
    metric: str,
    temperatures: dict[int, float],
) -> float:
    labels = [int(record["gold_label"]) for _, record in sample]
    probabilities = [
        _probability_from_logits(
            record["variant_logits"][variant],
            temperatures[int(seed)],
        )
        for seed, record in sample
    ]
    predictions = [int(np.argmax(value)) for value in probabilities]
    if metric == "weighted_f1":
        return float(
            f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            )
        )
    if metric == "macro_f1":
        return float(
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
    if metric == "accuracy":
        return float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    if metric == "nll":
        return float(
            np.mean(
                [
                    -math.log(max(probability[label], 1e-12))
                    for probability, label in zip(probabilities, labels)
                ]
            )
        )
    if metric == "brier":
        return float(
            np.mean(
                [
                    sum(
                        (
                            probability[index] - float(index == label)
                        )
                        ** 2
                        for index in range(len(probability))
                    )
                    for probability, label in zip(probabilities, labels)
                ]
            )
        )
    raise ValueError(f"Unsupported bootstrap metric: {metric}")


def _hierarchical_bootstrap(
    records_by_seed: dict[int, list[dict[str, Any]]],
    candidate: str,
    baseline: str,
    candidate_temperatures: dict[int, float],
    baseline_temperatures: dict[int, float],
    samples: int,
    random_seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(random_seed)
    seeds = sorted(records_by_seed)
    clusters: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in seeds:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records_by_seed[seed]:
            grouped[str(record["dialogue_id"])].append(record)
        clusters[seed] = dict(grouped)
    draws: dict[str, list[float]] = {
        metric: []
        for metric in (
            "weighted_f1",
            "macro_f1",
            "accuracy",
            "nll",
            "brier",
        )
    }
    for _ in range(samples):
        sample: list[tuple[int, dict[str, Any]]] = []
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        for raw_seed in sampled_seeds:
            seed = int(raw_seed)
            dialogue_ids = list(clusters[seed])
            sampled_dialogues = rng.choice(
                dialogue_ids,
                size=len(dialogue_ids),
                replace=True,
            )
            for dialogue_id in sampled_dialogues:
                sample.extend(
                    (seed, record)
                    for record in clusters[seed][str(dialogue_id)]
                )
        for metric in draws:
            candidate_value = _metric_from_sample(
                sample,
                candidate,
                metric,
                candidate_temperatures,
            )
            baseline_value = _metric_from_sample(
                sample,
                baseline,
                metric,
                baseline_temperatures,
            )
            advantage = (
                baseline_value - candidate_value
                if metric in {"nll", "brier"}
                else candidate_value - baseline_value
            )
            draws[metric].append(advantage)
    output: dict[str, Any] = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=float)
        output[metric] = {
            "mean_advantage": float(array.mean()),
            "ci95": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "probability_advantage_gt_zero": float(np.mean(array > 0.0)),
            "bootstrap_samples": samples,
        }
    return output


def _final_test(
    config: dict[str, Any],
    device: str | None,
    confirm: bool,
    resume: bool,
) -> dict[str, Any]:
    if not confirm:
        raise PermissionError(
            "Final test is one-time and locked. Re-run with "
            "--confirm-final-test after validation selection is frozen."
        )
    settings = _settings(config)
    root = _artifact_dir(config)
    manifest = _check_final_inputs(config)
    lock_path = root / "reports" / "final_test.lock.json"
    final_path = root / "reports" / "final_test.json"
    if final_path.exists():
        return json.loads(final_path.read_text(encoding="utf-8"))
    if lock_path.exists():
        existing_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if not resume:
            raise RuntimeError(
                "A final-test run already opened the test split. Use "
                "--resume only to finish that identical frozen run."
            )
        immutable_keys = (
            "selected_variant",
            "validation_summary_sha256",
            "config_sha256",
            "inputs",
        )
        changed = [
            key
            for key in immutable_keys
            if existing_lock.get(key) != manifest.get(key)
        ]
        if changed:
            raise RuntimeError(
                "Refusing to resume with changed frozen inputs: "
                + ", ".join(changed)
            )
    lock = {
        **manifest,
        "status": "running",
        "test_read": True,
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(lock, lock_path)

    summary = json.loads(
        Path(manifest["validation_summary"]).read_text(encoding="utf-8")
    )
    selected = str(manifest["selected_variant"])
    seeds = _seed_list(config)
    device_obj = get_device(device)
    for seed in seeds:
        report_path = _seed_report(root, seed, "test")
        records_path = _seed_records(root, seed, "test")
        if resume and report_path.exists() and records_path.exists():
            continue
        _evaluate_seed(config, seed, "test", device_obj, save=True)

    store = load_feature_store(
        _feature_path(config, seeds[0], "test")
    )
    id_to_label = {
        int(value): str(key)
        for key, value in store["label_to_id"].items()
    }
    records_by_seed = {
        seed: _read_jsonl(_seed_records(root, seed, "test"))
        for seed in seeds
    }
    test_reports = {
        seed: json.loads(
            _seed_report(root, seed, "test").read_text(encoding="utf-8")
        )
        for seed in seeds
    }
    variants_to_report = ("anchor", "full_history", selected)
    raw_per_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variants_to_report
    }
    calibrated_per_variant: dict[str, list[dict[str, Any]]] = {
        variant: [] for variant in variants_to_report
    }
    temperatures: dict[str, dict[int, float]] = {}
    for variant in variants_to_report:
        temperatures[variant] = {
            seed: float(
                summary["calibration"][variant][str(seed)]
                ["deployment_temperature"]
            )
            for seed in seeds
        }
        for seed in seeds:
            raw_per_variant[variant].append(
                _metrics_for_records(
                    records_by_seed[seed], variant, id_to_label
                )
            )
            calibrated_per_variant[variant].append(
                _metrics_for_records(
                    records_by_seed[seed],
                    variant,
                    id_to_label,
                    temperatures[variant][seed],
                )
            )
    variant_report = {
        variant: {
            "label": VARIANT_LABELS[variant],
            "raw": _metric_summary(raw_per_variant[variant]),
            "calibrated": _metric_summary(
                calibrated_per_variant[variant]
            ),
            "temperatures": _scalar_summary(
                temperatures[variant].values()
            ),
        }
        for variant in variants_to_report
    }
    bootstrap_samples = int(settings.get("bootstrap_samples", 2000))
    bootstrap_seed = int(settings.get("bootstrap_seed", 13016))
    bootstrap = {
        "selected_vs_anchor": _hierarchical_bootstrap(
            records_by_seed,
            selected,
            "anchor",
            temperatures[selected],
            temperatures["anchor"],
            bootstrap_samples,
            bootstrap_seed,
        ),
        "selected_vs_full_history": _hierarchical_bootstrap(
            records_by_seed,
            selected,
            "full_history",
            temperatures[selected],
            temperatures["full_history"],
            bootstrap_samples,
            bootstrap_seed + 1,
        ),
    }
    output = {
        "status": "complete",
        "experiment": "13 Context Utility Routing",
        "split": "test",
        "selected_variant": selected,
        "selected_label": VARIANT_LABELS[selected],
        "seeds": seeds,
        "variants": variant_report,
        "paired_hierarchical_bootstrap": bootstrap,
        "efficiency": {
            "utterances_per_second": _scalar_summary(
                [
                    test_reports[seed]["efficiency"][
                        "utterances_per_second"
                    ]
                    for seed in seeds
                ]
            ),
            "context_trainable_parameters": _scalar_summary(
                [
                    test_reports[seed]["efficiency"][
                        "context_trainable_parameters"
                    ]
                    for seed in seeds
                ]
            ),
            "router_trainable_parameters": _scalar_summary(
                [
                    test_reports[seed]["efficiency"][
                        "router_trainable_parameters"
                    ]
                    for seed in seeds
                ]
            ),
            "effective_history_retention": _scalar_summary(
                [
                    test_reports[seed]["efficiency"][
                        "effective_history_retention"
                    ]
                    for seed in seeds
                ]
            ),
        },
        "manifest": manifest,
        "test_used_for_selection": False,
        "test_read": True,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(output, final_path)
    save_json(
        {
            **lock,
            "status": "complete",
            "completed_at_utc": output["completed_at_utc"],
            "final_report": str(final_path),
            "final_report_sha256": _sha256(final_path),
        },
        lock_path,
    )
    return output


def run_experiment13(
    config: dict[str, Any],
    stage: str,
    device: str | None = None,
    confirm_final_test: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the pre-registered context-utility routing protocol."""
    if stage == "check":
        report = _configuration_check(config)
        save_json(
            report,
            _artifact_dir(config) / "reports" / "input_audit.json",
        )
        return report
    if stage == "train-teachers":
        return _train_teacher_models(config, device)
    if stage == "build-train-labels":
        return _generate_labels(config, "train", device)
    if stage == "build-validation-labels":
        return _generate_labels(config, "validation", device)
    if stage == "train-routers":
        return _train_routers(config, device)
    if stage == "evaluate-validation":
        return _evaluate_all(config, "validation", device)
    if stage == "summarize-validation":
        return _summarize_validation(config)
    if stage == "all-validation":
        _configuration_check(config)
        _train_teacher_models(config, device)
        _generate_labels(config, "train", device)
        _generate_labels(config, "validation", device)
        _train_routers(config, device)
        _evaluate_all(config, "validation", device)
        return _summarize_validation(config)
    if stage == "check-final":
        return _check_final_inputs(config)
    if stage == "final-test":
        return _final_test(
            config,
            device,
            confirm=confirm_final_test,
            resume=resume,
        )
    if stage == "summarize-final":
        path = _artifact_dir(config) / "reports" / "final_test.json"
        if not path.exists():
            raise FileNotFoundError(
                "Frozen Experiment 13 final test has not been run"
            )
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown Experiment 13 stage: {stage}")


__all__ = [
    "run_experiment13",
    "_dialogue_fold",
    "_selected_prefix",
    "_utility_class",
]
