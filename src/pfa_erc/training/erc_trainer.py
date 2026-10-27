from __future__ import annotations

import hashlib
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import HiTFTForecaster, PFAERC
from .common import (
    DialogueFeatureDataset,
    EarlyStopping,
    classification_metrics,
    collate_dialogues,
    get_device,
    linear_warmup_decay_lambda,
    load_feature_store,
    make_grad_scaler,
    move_to_device,
    save_json,
    seed_everything,
)


def _build_model(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> PFAERC:
    hitft = None
    if bool(config["model"].get("use_hitft", True)):
        hitft = HiTFTForecaster(config)
        checkpoint = torch.load(
            config["train_hitft"]["checkpoint"], map_location="cpu", weights_only=False
        )
        hitft.load_state_dict(checkpoint["model_state"])
    return PFAERC(config, num_classes=num_classes, hitft=hitft).to(device)


def _loader(
    dialogues: list[dict[str, Any]],
    config: dict[str, Any],
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DialogueFeatureDataset(dialogues),
        batch_size=int(config["train_erc"].get("batch_size", 8)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=int(config["experiment"].get("num_workers", 0)),
        collate_fn=collate_dialogues,
        pin_memory=device.type == "cuda",
    )


def _dialogue_fold(dialogue: dict[str, Any], folds: int, seed: int) -> int:
    """Assign a dialogue to a deterministic fold without using Python's hash.

    Python randomizes ``hash(str)`` between processes.  A SHA-256 assignment
    makes Experiment 13 out-of-fold teachers reproducible across machines.
    """
    if folds < 2:
        raise ValueError("folds must be at least 2")
    dialogue_id = str(dialogue.get("dialogue_id", ""))
    digest = hashlib.sha256(f"{seed}:{dialogue_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _oof_training_split(
    dialogues: list[dict[str, Any]],
    train_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Optionally remove one deterministic dialogue fold from ERC training."""
    holdout = train_config.get("oof_holdout_fold")
    if holdout is None:
        return dialogues, None
    folds = int(train_config.get("oof_folds", 3))
    fold_seed = int(train_config.get("oof_fold_seed", 13013))
    holdout = int(holdout)
    if not 0 <= holdout < folds:
        raise ValueError(
            f"train_erc.oof_holdout_fold must be in [0, {folds}), got {holdout}"
        )
    selected = [
        dialogue
        for dialogue in dialogues
        if _dialogue_fold(dialogue, folds, fold_seed) != holdout
    ]
    held_out = len(dialogues) - len(selected)
    if not selected or held_out == 0:
        raise ValueError(
            "OOF split produced an empty train or holdout partition; "
            "check dialogue IDs and fold settings"
        )
    return selected, {
        "folds": folds,
        "holdout_fold": holdout,
        "fold_seed": fold_seed,
        "train_dialogues": len(selected),
        "held_out_dialogues": held_out,
    }


def _validate_current_logit_features(
    store: dict[str, Any], num_classes: int
) -> None:
    feature_config = store.get("feature_config", {})
    if not bool(feature_config.get("cache_classifier_logits", False)):
        raise ValueError(
            "This logit-anchored model requires a feature cache with 3A "
            "current_logits. Re-run the extract stage with the 3C config."
        )
    for split, dialogues in store["splits"].items():
        for dialogue in dialogues:
            logits = dialogue.get("current_logits")
            length = int(dialogue["embeddings"].shape[0])
            if logits is None or tuple(logits.shape) != (length, num_classes):
                raise ValueError(
                    f"Invalid or missing current_logits in {split}:"
                    f"{dialogue.get('dialogue_id', 'unknown')}"
                )


def _same_speaker_transition_masks(
    speaker_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mark turns whose gold emotion is stable/changed for the same speaker.

    The first observed utterance from each speaker has no reference emotion and
    is excluded from both subsets. This is evaluation-only and is never passed
    into the model.
    """
    known = torch.zeros_like(valid_mask)
    transition = torch.zeros_like(valid_mask)
    for batch_index in range(speaker_ids.shape[0]):
        previous_label_by_speaker: dict[int, int] = {}
        for turn_index in torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten():
            index = int(turn_index)
            speaker = int(speaker_ids[batch_index, index])
            label = int(labels[batch_index, index])
            if speaker in previous_label_by_speaker:
                known[batch_index, index] = True
                transition[batch_index, index] = (
                    label != previous_label_by_speaker[speaker]
                )
            previous_label_by_speaker[speaker] = label
    return known, transition


def _context_helpfulness_targets(
    anchor_logits: torch.Tensor,
    context_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    """Return where the context branch beats the fixed 3A anchor.

    The target is computed only from detached training/evaluation logits. A
    context prediction is considered helpful when its gold-label negative log
    likelihood is smaller than the anchor NLL by at least ``margin``.
    """
    with torch.no_grad():
        safe_labels = labels.clamp_min(0).unsqueeze(-1)
        anchor_nll = -torch.log_softmax(anchor_logits.detach().float(), dim=-1).gather(
            -1, safe_labels
        ).squeeze(-1)
        context_nll = -torch.log_softmax(context_logits.detach().float(), dim=-1).gather(
            -1, safe_labels
        ).squeeze(-1)
        return valid_mask & (context_nll + margin < anchor_nll)


def _build_class_weights(
    dialogues: list[dict[str, Any]],
    num_classes: int,
    train_config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Build bounded inverse-frequency weights from the training split only."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for dialogue in dialogues:
        labels = torch.as_tensor(dialogue["labels"], dtype=torch.long).flatten()
        valid = labels[(labels >= 0) & (labels < num_classes)]
        counts += torch.bincount(valid, minlength=num_classes)
    strategy = str(train_config.get("class_weight_strategy", "none"))
    if strategy == "none":
        return None, counts
    if strategy != "inverse_frequency":
        raise ValueError(
            "train_erc.class_weight_strategy must be 'none' or "
            f"'inverse_frequency', got {strategy!r}"
        )
    if torch.any(counts == 0):
        missing = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
        raise ValueError(
            f"Cannot class-balance ERC: training classes {missing} have no samples"
        )

    power = float(train_config.get("class_weight_power", 0.5))
    minimum = float(train_config.get("class_weight_min", 0.25))
    maximum = float(train_config.get("class_weight_max", 3.0))
    if not 0.0 < power <= 1.0:
        raise ValueError("train_erc.class_weight_power must be in (0, 1]")
    if not 0.0 < minimum <= maximum:
        raise ValueError(
            "train_erc class-weight bounds must satisfy 0 < min <= max"
        )

    weights = counts.float().pow(-power)
    weights = weights / weights.mean()
    weights = weights.clamp(min=minimum, max=maximum)
    return weights.to(device), counts


def _selection_metric_weights(train_config: dict[str, Any]) -> dict[str, float]:
    allowed = (
        "overall_weighted_f1",
        "overall_macro_f1",
        "transition_weighted_f1",
        "transition_macro_f1",
    )
    configured = train_config.get("selection_metric_weights")
    if configured is None:
        transition_weight = float(
            train_config.get("transition_selection_weight", 0.0)
        )
        if not 0.0 <= transition_weight <= 1.0:
            raise ValueError(
                "train_erc.transition_selection_weight must be between 0 and 1"
            )
        return {
            "overall_weighted_f1": 1.0 - transition_weight,
            "overall_macro_f1": 0.0,
            "transition_weighted_f1": transition_weight,
            "transition_macro_f1": 0.0,
        }
    if not isinstance(configured, dict):
        raise ValueError("train_erc.selection_metric_weights must be a mapping")
    unknown = sorted(set(configured) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown ERC selection metrics: {unknown}")
    weights = {name: float(configured.get(name, 0.0)) for name in allowed}
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("ERC selection metric weights cannot be negative")
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("ERC selection metric weights must sum to 1.0")
    return weights


def _validation_selection_score(
    validation: dict[str, Any], weights: dict[str, float]
) -> tuple[float, dict[str, float]]:
    overall_weighted = float(validation["weighted_f1"])
    overall_macro = float(validation["macro_f1"])
    transition = validation["same_speaker_transition"]
    transition_weighted = transition["weighted_f1"]
    transition_macro = transition["macro_f1"]
    values = {
        "overall_weighted_f1": overall_weighted,
        "overall_macro_f1": overall_macro,
        "transition_weighted_f1": (
            overall_weighted
            if transition_weighted is None
            else float(transition_weighted)
        ),
        "transition_macro_f1": (
            overall_macro if transition_macro is None else float(transition_macro)
        ),
    }
    score = sum(weights[name] * values[name] for name in weights)
    return score, values


def _subset_classification_metrics(
    labels: list[int],
    predictions: list[int],
    id_to_label: dict[int, str],
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


def _gate_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "quantiles": []}
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": len(values),
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "quantiles": [
            float(value)
            for value in torch.quantile(
                tensor, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
            )
        ],
    }


def _probability_quality(
    labels: list[int],
    probabilities: list[list[float]],
    num_classes: int,
    bins: int = 10,
) -> dict[str, Any]:
    """Return proper scoring rules without using them as model inputs."""
    if not labels:
        return {
            "nll": None,
            "brier": None,
            "ece": None,
            "top1_accuracy": None,
            "macro_f1": None,
            "utterances": 0,
        }
    label_tensor = torch.tensor(labels, dtype=torch.long)
    probability = torch.tensor(probabilities, dtype=torch.float32).clamp_min(1e-8)
    probability = probability / probability.sum(dim=-1, keepdim=True)
    nll = -probability[
        torch.arange(label_tensor.numel()), label_tensor
    ].log().mean()
    one_hot = nn.functional.one_hot(
        label_tensor, num_classes=num_classes
    ).to(probability.dtype)
    brier = (probability - one_hot).square().sum(dim=-1).mean()
    confidence, prediction = probability.max(dim=-1)
    correctness = prediction.eq(label_tensor).to(probability.dtype)
    _, _, macro_f1, _ = precision_recall_fscore_support(
        label_tensor.cpu().tolist(),
        prediction.cpu().tolist(),
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )
    ece = probability.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        in_bin = confidence.ge(lower) & (
            confidence.le(upper) if index == bins - 1 else confidence.lt(upper)
        )
        if in_bin.any():
            ece = ece + in_bin.float().mean() * (
                correctness[in_bin].mean() - confidence[in_bin].mean()
            ).abs()
    return {
        "nll": float(nll),
        "brier": float(brier),
        "ece": float(ece),
        "top1_accuracy": float(correctness.mean()),
        "macro_f1": float(macro_f1),
        "utterances": len(labels),
    }


def _history_prior_baselines(
    labels: list[int],
    num_classes: int,
    class_prior: torch.Tensor,
) -> dict[str, Any]:
    """Evaluate fixed priors on the same history-eligible turns as q_t."""
    if class_prior.ndim != 1 or class_prior.numel() != num_classes:
        raise ValueError("class_prior must contain one value per class")
    normalized = class_prior.float().clamp_min(1e-8)
    normalized = normalized / normalized.sum()
    uniform = [1.0 / num_classes] * num_classes
    marginal = normalized.cpu().tolist()
    uniform_quality = _probability_quality(
        labels,
        [uniform] * len(labels),
        num_classes,
    )
    marginal_quality = _probability_quality(
        labels,
        [marginal] * len(labels),
        num_classes,
    )
    return {
        "uniform": uniform_quality,
        "class_marginal": marginal_quality,
    }


def _transition_detector_metrics(
    targets: list[int], probabilities: list[float]
) -> dict[str, Any]:
    if not targets:
        return {
            "known_targets": 0,
            "average_precision": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "target_rate": None,
            "predicted_rate": None,
        }
    predictions = [int(probability >= 0.5) for probability in probabilities]
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )
    average_precision = (
        float(average_precision_score(targets, probabilities))
        if len(set(targets)) > 1
        else None
    )
    return {
        "known_targets": len(targets),
        "average_precision": average_precision,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "target_rate": sum(targets) / len(targets),
        "predicted_rate": sum(predictions) / len(predictions),
    }


def _binary_metrics_by_emotion(
    emotion_labels: list[int],
    targets: list[int],
    probabilities: list[float],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for label_id, label_name in sorted(id_to_label.items()):
        indices = [
            index
            for index, observed_label in enumerate(emotion_labels)
            if observed_label == label_id
        ]
        output[label_name] = _transition_detector_metrics(
            [targets[index] for index in indices],
            [probabilities[index] for index in indices],
        )
    return output


def _fusion_flip_metrics(
    labels: list[int],
    anchor_predictions: list[int],
    fusion_predictions: list[int],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    if not (
        len(labels) == len(anchor_predictions) == len(fusion_predictions)
    ):
        raise ValueError("Fusion flip inputs must have matching lengths")

    def summarize(indices: list[int]) -> dict[str, Any]:
        beneficial = sum(
            anchor_predictions[index] != labels[index]
            and fusion_predictions[index] == labels[index]
            for index in indices
        )
        harmful = sum(
            anchor_predictions[index] == labels[index]
            and fusion_predictions[index] != labels[index]
            for index in indices
        )
        return {
            "support": len(indices),
            "beneficial_flips": beneficial,
            "harmful_flips": harmful,
            "net_flips": beneficial - harmful,
            "beneficial_rate": beneficial / len(indices) if indices else None,
            "harmful_rate": harmful / len(indices) if indices else None,
        }

    output = summarize(list(range(len(labels))))
    output["per_emotion"] = {
        label_name: summarize(
            [index for index, label in enumerate(labels) if label == label_id]
        )
        for label_id, label_name in sorted(id_to_label.items())
    }
    return output


def _evaluate_loader(
    model: PFAERC,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
    use_amp: bool,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []
    stable_labels: list[int] = []
    stable_predictions: list[int] = []
    transition_labels: list[int] = []
    transition_predictions: list[int] = []
    anchor_predictions: list[int] = []
    anchor_probabilities: list[list[float]] = []
    anchor_stable_predictions: list[int] = []
    anchor_transition_predictions: list[int] = []
    context_predictions: list[int] = []
    context_probabilities: list[list[float]] = []
    context_stable_predictions: list[int] = []
    context_transition_predictions: list[int] = []
    overall_gates: list[float] = []
    stable_gates: list[float] = []
    transition_gates: list[float] = []
    overall_base_gates: list[float] = []
    stable_base_gates: list[float] = []
    transition_base_gates: list[float] = []
    overall_capped_base_gates: list[float] = []
    stable_capped_base_gates: list[float] = []
    transition_capped_base_gates: list[float] = []
    transition_targets: list[int] = []
    transition_probabilities: list[float] = []
    helpfulness_targets: list[int] = []
    helpfulness_probabilities: list[float] = []
    helpfulness_emotion_labels: list[int] = []
    stable_helpfulness_targets: list[int] = []
    stable_helpfulness_probabilities: list[float] = []
    transition_helpfulness_targets: list[int] = []
    transition_helpfulness_probabilities: list[float] = []
    overall_transition_suppression: list[float] = []
    stable_transition_suppression: list[float] = []
    transition_transition_suppression: list[float] = []
    overall_helpfulness_modulation: list[float] = []
    stable_helpfulness_modulation: list[float] = []
    transition_helpfulness_modulation: list[float] = []
    raw_residual_rms: list[float] = []
    stable_raw_residual_rms: list[float] = []
    transition_raw_residual_rms: list[float] = []
    applied_residual_rms: list[float] = []
    stable_applied_residual_rms: list[float] = []
    transition_applied_residual_rms: list[float] = []
    class_gate_by_output: list[list[float]] = [
        [] for _ in range(len(id_to_label))
    ]
    directed_self_gates: list[float] = []
    directed_self_available_gates: list[float] = []
    directed_other_gates: list[float] = []
    directed_other_available_gates: list[float] = []
    directed_transition_gates: list[float] = []
    directed_transition_known_gates: list[float] = []
    directed_js_divergence: list[float] = []
    directed_matrix_diagonal: list[float] = []
    directed_matrix_delta_rms: list[float] = []
    directed_prior_labels: list[int] = []
    directed_prior_probabilities: list[list[float]] = []
    directed_anchor_entropy: list[float] = []
    directed_prior_entropy: list[float] = []
    directed_speaker_gap: list[float] = []
    directed_self_delta_rms: list[float] = []
    directed_other_delta_rms: list[float] = []
    directed_transition_delta_rms: list[float] = []
    directed_self_available = 0
    directed_other_available = 0
    directed_speaker_known = 0
    total_loss = 0.0
    total_turns = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="ERC evaluate", leave=False):
            batch = move_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if model.use_current_logit_anchor:
                    logits, auxiliary = model(
                        batch["embeddings"],
                        batch["speaker_ids"],
                        batch["mask"],
                        current_logits=batch.get("current_logits"),
                        return_aux=True,
                    )
                else:
                    logits = model(
                        batch["embeddings"],
                        batch["speaker_ids"],
                        batch["mask"],
                    )
                    auxiliary = None
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["labels"].reshape(-1),
                    ignore_index=-100,
                )
            valid = batch["mask"]
            predicted_labels = logits.argmax(dim=-1)
            labels.extend(batch["labels"][valid].cpu().tolist())
            predictions.extend(predicted_labels[valid].cpu().tolist())
            probabilities.extend(
                torch.softmax(logits.float(), dim=-1)[valid].cpu().tolist()
            )
            known, transition = _same_speaker_transition_masks(
                batch["speaker_ids"], batch["labels"], valid
            )
            stable = known & ~transition
            stable_labels.extend(batch["labels"][stable].cpu().tolist())
            stable_predictions.extend(predicted_labels[stable].cpu().tolist())
            transition_labels.extend(batch["labels"][transition].cpu().tolist())
            transition_predictions.extend(predicted_labels[transition].cpu().tolist())
            if auxiliary is not None:
                anchor_predicted = batch["current_logits"].argmax(dim=-1)
                context_predicted = auxiliary["context_logits"].argmax(dim=-1)
                anchor_probability = torch.softmax(
                    batch["current_logits"].float(), dim=-1
                )
                context_probability = torch.softmax(
                    auxiliary["context_logits"].float(), dim=-1
                )
                anchor_predictions.extend(anchor_predicted[valid].cpu().tolist())
                anchor_probabilities.extend(
                    anchor_probability[valid].cpu().tolist()
                )
                anchor_stable_predictions.extend(
                    anchor_predicted[stable].cpu().tolist()
                )
                anchor_transition_predictions.extend(
                    anchor_predicted[transition].cpu().tolist()
                )
                context_predictions.extend(context_predicted[valid].cpu().tolist())
                context_probabilities.extend(
                    context_probability[valid].cpu().tolist()
                )
                context_stable_predictions.extend(
                    context_predicted[stable].cpu().tolist()
                )
                context_transition_predictions.extend(
                    context_predicted[transition].cpu().tolist()
                )
                gate = auxiliary["context_gate"].squeeze(-1)
                base_gate = auxiliary["base_context_gate"].squeeze(-1)
                capped_base_gate = auxiliary["capped_base_context_gate"].squeeze(-1)
                probability = auxiliary["predicted_transition_probability"]
                transition_suppression = auxiliary[
                    "transition_suppression"
                ].squeeze(-1)
                helpfulness_modulation = auxiliary[
                    "helpfulness_modulation"
                ].squeeze(-1)
                overall_gates.extend(gate[valid].float().cpu().tolist())
                stable_gates.extend(gate[stable].float().cpu().tolist())
                transition_gates.extend(gate[transition].float().cpu().tolist())
                overall_base_gates.extend(base_gate[valid].float().cpu().tolist())
                stable_base_gates.extend(base_gate[stable].float().cpu().tolist())
                transition_base_gates.extend(
                    base_gate[transition].float().cpu().tolist()
                )
                overall_capped_base_gates.extend(
                    capped_base_gate[valid].float().cpu().tolist()
                )
                stable_capped_base_gates.extend(
                    capped_base_gate[stable].float().cpu().tolist()
                )
                transition_capped_base_gates.extend(
                    capped_base_gate[transition].float().cpu().tolist()
                )
                transition_targets.extend(transition[known].long().cpu().tolist())
                transition_probabilities.extend(
                    probability[known].float().cpu().tolist()
                )
                overall_transition_suppression.extend(
                    transition_suppression[valid].float().cpu().tolist()
                )
                stable_transition_suppression.extend(
                    transition_suppression[stable].float().cpu().tolist()
                )
                transition_transition_suppression.extend(
                    transition_suppression[transition].float().cpu().tolist()
                )
                overall_helpfulness_modulation.extend(
                    helpfulness_modulation[valid].float().cpu().tolist()
                )
                stable_helpfulness_modulation.extend(
                    helpfulness_modulation[stable].float().cpu().tolist()
                )
                transition_helpfulness_modulation.extend(
                    helpfulness_modulation[transition].float().cpu().tolist()
                )
                if model.use_class_residual:
                    raw = auxiliary["raw_class_residual"].float()
                    applied = auxiliary["class_residual"].float()
                    class_gate = auxiliary["class_gate"].float()
                    raw_turn_rms = raw.square().mean(dim=-1).sqrt()
                    applied_turn_rms = applied.square().mean(dim=-1).sqrt()
                    raw_residual_rms.extend(raw_turn_rms[valid].cpu().tolist())
                    stable_raw_residual_rms.extend(
                        raw_turn_rms[stable].cpu().tolist()
                    )
                    transition_raw_residual_rms.extend(
                        raw_turn_rms[transition].cpu().tolist()
                    )
                    applied_residual_rms.extend(
                        applied_turn_rms[valid].cpu().tolist()
                    )
                    stable_applied_residual_rms.extend(
                        applied_turn_rms[stable].cpu().tolist()
                    )
                    transition_applied_residual_rms.extend(
                        applied_turn_rms[transition].cpu().tolist()
                    )
                    for class_index in range(len(id_to_label)):
                        class_gate_by_output[class_index].extend(
                            class_gate[..., class_index][valid].cpu().tolist()
                        )
                if model.use_directed_update:
                    self_available = auxiliary["self_available"] & valid
                    other_available = auxiliary["other_available"] & valid
                    speaker_known = auxiliary["speaker_known"] & valid
                    self_gate = auxiliary["self_gate"].squeeze(-1).float()
                    other_gate = auxiliary["other_gate"].squeeze(-1).float()
                    directed_gate = auxiliary["transition_gate"].squeeze(-1).float()
                    directed_self_gates.extend(self_gate[valid].cpu().tolist())
                    directed_self_available_gates.extend(
                        self_gate[self_available].cpu().tolist()
                    )
                    directed_other_gates.extend(other_gate[valid].cpu().tolist())
                    directed_other_available_gates.extend(
                        other_gate[other_available].cpu().tolist()
                    )
                    directed_transition_gates.extend(
                        directed_gate[valid].cpu().tolist()
                    )
                    directed_transition_known_gates.extend(
                        directed_gate[speaker_known].cpu().tolist()
                    )
                    directed_js_divergence.extend(
                        auxiliary["js_divergence"].squeeze(-1)[speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    matrix_diagonal = (
                        auxiliary["transition_matrix"]
                        .diagonal(dim1=-2, dim2=-1)
                        .mean(dim=-1)
                    )
                    directed_matrix_diagonal.extend(
                        matrix_diagonal[speaker_known].float().cpu().tolist()
                    )
                    directed_matrix_delta_rms.extend(
                        auxiliary["matrix_delta_rms"][speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    directed_prior_labels.extend(
                        batch["labels"][speaker_known].long().cpu().tolist()
                    )
                    directed_prior_probabilities.extend(
                        auxiliary["dynamic_prior"][speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    directed_anchor_entropy.extend(
                        auxiliary["anchor_entropy"].squeeze(-1)[speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    directed_prior_entropy.extend(
                        auxiliary["history_prior_entropy"]
                        .squeeze(-1)[speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    directed_speaker_gap.extend(
                        auxiliary["speaker_gap"][speaker_known]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    for key, target in (
                        ("self_delta", directed_self_delta_rms),
                        ("other_delta", directed_other_delta_rms),
                        ("transition_delta", directed_transition_delta_rms),
                    ):
                        rms = auxiliary[key].float().square().mean(dim=-1).sqrt()
                        target.extend(rms[valid].cpu().tolist())
                    directed_self_available += int(self_available.sum())
                    directed_other_available += int(other_available.sum())
                    directed_speaker_known += int(speaker_known.sum())
                if model.uses_helpfulness_gate:
                    helpful = _context_helpfulness_targets(
                        batch["current_logits"],
                        auxiliary["context_logits"],
                        batch["labels"],
                        valid,
                        model.context_helpfulness_margin,
                    )
                    helpfulness_targets.extend(helpful[valid].long().cpu().tolist())
                    helpfulness_probabilities.extend(
                        auxiliary["predicted_helpfulness_probability"][valid]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    helpfulness_emotion_labels.extend(
                        batch["labels"][valid].long().cpu().tolist()
                    )
                    stable_helpfulness_targets.extend(
                        helpful[stable].long().cpu().tolist()
                    )
                    stable_helpfulness_probabilities.extend(
                        auxiliary["predicted_helpfulness_probability"][stable]
                        .float()
                        .cpu()
                        .tolist()
                    )
                    transition_helpfulness_targets.extend(
                        helpful[transition].long().cpu().tolist()
                    )
                    transition_helpfulness_probabilities.extend(
                        auxiliary["predicted_helpfulness_probability"][transition]
                        .float()
                        .cpu()
                        .tolist()
                    )
            turns = int(valid.sum())
            total_loss += float(loss) * turns
            total_turns += turns
    metrics = classification_metrics(labels, predictions, id_to_label)
    metrics["loss"] = total_loss / max(total_turns, 1)
    metrics["utterances"] = total_turns
    metrics["probability_quality"] = _probability_quality(
        labels, probabilities, len(id_to_label)
    )
    metrics["same_speaker_stable"] = _subset_classification_metrics(
        stable_labels, stable_predictions, id_to_label
    )
    metrics["same_speaker_transition"] = _subset_classification_metrics(
        transition_labels, transition_predictions, id_to_label
    )
    if model.use_current_logit_anchor:
        metrics["anchor_only"] = {
            **_subset_classification_metrics(
                labels, anchor_predictions, id_to_label
            ),
            "probability_quality": _probability_quality(
                labels, anchor_probabilities, len(id_to_label)
            ),
            "same_speaker_stable": _subset_classification_metrics(
                stable_labels, anchor_stable_predictions, id_to_label
            ),
            "same_speaker_transition": _subset_classification_metrics(
                transition_labels, anchor_transition_predictions, id_to_label
            ),
        }
        metrics["context_only"] = {
            **_subset_classification_metrics(
                labels, context_predictions, id_to_label
            ),
            "probability_quality": _probability_quality(
                labels, context_probabilities, len(id_to_label)
            ),
            "same_speaker_stable": _subset_classification_metrics(
                stable_labels, context_stable_predictions, id_to_label
            ),
            "same_speaker_transition": _subset_classification_metrics(
                transition_labels, context_transition_predictions, id_to_label
            ),
        }
        metrics["logit_fusion"] = {
            "gate_mode": model.context_gate_mode,
            "transition_gate_power": model.transition_gate_power,
            "context_gate_max": model.context_gate_max,
            "context_gate": {
                "overall": _gate_summary(overall_gates),
                "same_speaker_stable": _gate_summary(stable_gates),
                "same_speaker_transition": _gate_summary(transition_gates),
            },
            "base_context_gate": {
                "overall": _gate_summary(overall_base_gates),
                "same_speaker_stable": _gate_summary(stable_base_gates),
                "same_speaker_transition": _gate_summary(transition_base_gates),
            },
            "capped_base_context_gate": {
                "overall": _gate_summary(overall_capped_base_gates),
                "same_speaker_stable": _gate_summary(stable_capped_base_gates),
                "same_speaker_transition": _gate_summary(
                    transition_capped_base_gates
                ),
            },
            "transition_detector": _transition_detector_metrics(
                transition_targets, transition_probabilities
            ),
            "transition_suppression": {
                "overall": _gate_summary(overall_transition_suppression),
                "same_speaker_stable": _gate_summary(
                    stable_transition_suppression
                ),
                "same_speaker_transition": _gate_summary(
                    transition_transition_suppression
                ),
            },
            "helpfulness_modulation": {
                "overall": _gate_summary(overall_helpfulness_modulation),
                "same_speaker_stable": _gate_summary(
                    stable_helpfulness_modulation
                ),
                "same_speaker_transition": _gate_summary(
                    transition_helpfulness_modulation
                ),
            },
            "fusion_flips": {
                "overall": _fusion_flip_metrics(
                    labels, anchor_predictions, predictions, id_to_label
                ),
                "same_speaker_stable": _fusion_flip_metrics(
                    stable_labels,
                    anchor_stable_predictions,
                    stable_predictions,
                    id_to_label,
                ),
                "same_speaker_transition": _fusion_flip_metrics(
                    transition_labels,
                    anchor_transition_predictions,
                    transition_predictions,
                    id_to_label,
                ),
            },
        }
        if model.uses_helpfulness_gate:
            helpfulness_metrics = _transition_detector_metrics(
                helpfulness_targets, helpfulness_probabilities
            )
            helpfulness_metrics["same_speaker_stable"] = (
                _transition_detector_metrics(
                    stable_helpfulness_targets,
                    stable_helpfulness_probabilities,
                )
            )
            helpfulness_metrics["same_speaker_transition"] = (
                _transition_detector_metrics(
                    transition_helpfulness_targets,
                    transition_helpfulness_probabilities,
                )
            )
            helpfulness_metrics["per_emotion"] = _binary_metrics_by_emotion(
                helpfulness_emotion_labels,
                helpfulness_targets,
                helpfulness_probabilities,
                id_to_label,
            )
            metrics["logit_fusion"]["helpfulness_detector"] = helpfulness_metrics
        if model.use_class_residual:
            metrics["logit_fusion"]["class_residual"] = {
                "raw_rms": {
                    "overall": _gate_summary(raw_residual_rms),
                    "same_speaker_stable": _gate_summary(
                        stable_raw_residual_rms
                    ),
                    "same_speaker_transition": _gate_summary(
                        transition_raw_residual_rms
                    ),
                },
                "applied_rms": {
                    "overall": _gate_summary(applied_residual_rms),
                    "same_speaker_stable": _gate_summary(
                        stable_applied_residual_rms
                    ),
                    "same_speaker_transition": _gate_summary(
                        transition_applied_residual_rms
                    ),
                },
                "class_gate_by_output": {
                    id_to_label[class_index]: _gate_summary(values)
                    for class_index, values in enumerate(class_gate_by_output)
                },
            }
        if model.use_directed_update:
            assert model.directed_affect_updater is not None
            history_prior_quality = _probability_quality(
                directed_prior_labels,
                directed_prior_probabilities,
                len(id_to_label),
            )
            history_prior_baselines = _history_prior_baselines(
                directed_prior_labels,
                len(id_to_label),
                model.directed_affect_updater.class_prior,
            )
            prior_nll = history_prior_quality["nll"]
            uniform_nll = history_prior_baselines["uniform"]["nll"]
            marginal_nll = history_prior_baselines["class_marginal"]["nll"]
            metrics["logit_fusion"]["directed_update"] = {
                "variant": model.directed_affect_updater.variant,
                "matrix_mode": model.directed_affect_updater.matrix_mode,
                "prior_mode": model.directed_affect_updater.prior_mode,
                "matrix_feature_mode": (
                    model.directed_affect_updater.matrix_feature_mode
                ),
                "fusion_mode": model.directed_affect_updater.fusion_mode,
                "self_gate": {
                    "all_valid": _gate_summary(directed_self_gates),
                    "available_only": _gate_summary(
                        directed_self_available_gates
                    ),
                },
                "other_gate": {
                    "all_valid": _gate_summary(directed_other_gates),
                    "available_only": _gate_summary(
                        directed_other_available_gates
                    ),
                },
                "transition_gate": {
                    "all_valid": _gate_summary(directed_transition_gates),
                    "known_speaker_only": _gate_summary(
                        directed_transition_known_gates
                    ),
                },
                "availability_rate": {
                    "self": directed_self_available / max(total_turns, 1),
                    "other": directed_other_available / max(total_turns, 1),
                    "known_speaker_prior": directed_speaker_known
                    / max(total_turns, 1),
                },
                "anchor_prior_js_divergence": _gate_summary(
                    directed_js_divergence
                ),
                "transition_matrix_diagonal": _gate_summary(
                    directed_matrix_diagonal
                ),
                "matrix_delta_rms": _gate_summary(
                    directed_matrix_delta_rms
                ),
                "history_prior_quality": history_prior_quality,
                "history_prior_baselines": history_prior_baselines,
                "history_prior_nll_improvement": {
                    "over_uniform": (
                        None
                        if prior_nll is None or uniform_nll is None
                        else float(uniform_nll - prior_nll)
                    ),
                    "over_class_marginal": (
                        None
                        if prior_nll is None or marginal_nll is None
                        else float(marginal_nll - prior_nll)
                    ),
                },
                "anchor_entropy": _gate_summary(
                    directed_anchor_entropy
                ),
                "history_prior_entropy": _gate_summary(
                    directed_prior_entropy
                ),
                "normalized_speaker_gap": _gate_summary(
                    directed_speaker_gap
                ),
                "applied_delta_rms": {
                    "self": _gate_summary(directed_self_delta_rms),
                    "other": _gate_summary(directed_other_delta_rms),
                    "transition": _gate_summary(
                        directed_transition_delta_rms
                    ),
                },
            }
    return metrics


def train_erc(
    config: dict[str, Any],
    device: str | None = None,
    seed: int | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    seed = int(seed if seed is not None else config["experiment"].get("seed", 42))
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device_obj = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    label_to_id = {str(key): int(value) for key, value in store["label_to_id"].items()}
    id_to_label = {value: key for key, value in label_to_id.items()}
    model = _build_model(config, len(label_to_id), device_obj)
    if model.use_current_logit_anchor:
        _validate_current_logit_features(store, len(label_to_id))
    train_items, oof_partition = _oof_training_split(
        list(store["splits"]["train"]),
        config["train_erc"],
    )
    train_loader = _loader(train_items, config, True, seed, device_obj)
    validation_items = store["splits"].get("validation", store["splits"]["train"])
    validation_loader = _loader(validation_items, config, False, seed, device_obj)

    train_config = config["train_erc"]
    class_weights, class_counts = _build_class_weights(
        train_items,
        len(label_to_id),
        train_config,
        device_obj,
    )
    balance_main_loss = bool(train_config.get("class_balance_main_loss", False))
    balance_context_loss = bool(
        train_config.get("class_balance_context_loss", False)
    )
    main_class_weights = class_weights if balance_main_loss else None
    context_class_weights = class_weights if balance_context_loss else None
    selection_weights = _selection_metric_weights(train_config)
    class_balance = {
        "strategy": str(train_config.get("class_weight_strategy", "none")),
        "power": float(train_config.get("class_weight_power", 0.5)),
        "minimum": float(train_config.get("class_weight_min", 0.25)),
        "maximum": float(train_config.get("class_weight_max", 3.0)),
        "main_loss": balance_main_loss,
        "context_loss": balance_context_loss,
        "counts": {
            id_to_label[index]: int(class_counts[index])
            for index in sorted(id_to_label)
        },
        "weights": (
            {
                id_to_label[index]: float(class_weights[index].detach().cpu())
                for index in sorted(id_to_label)
            }
            if class_weights is not None
            else None
        ),
    }
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(train_config.get("learning_rate", 5e-5)),
        weight_decay=float(train_config.get("weight_decay", 1e-3)),
    )
    total_steps = max(len(train_loader) * int(train_config.get("epochs", 20)), 1)
    warmup_steps = int(total_steps * float(train_config.get("warmup_ratio", 0.2)))
    scheduler = LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay_lambda(step, warmup_steps, total_steps),
    )
    use_amp = bool(train_config.get("use_amp", True)) and device_obj.type == "cuda"
    scaler = make_grad_scaler(device_obj, use_amp)
    stopper = EarlyStopping(int(train_config.get("patience", 8)), mode="max")
    checkpoint = Path(checkpoint_path or train_config["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    helpfulness_loss_weight = float(train_config.get("helpfulness_loss_weight", 0.0))
    residual_l1_weight = float(train_config.get("residual_l1_weight", 0.0))
    directed_prior_loss_weight = float(
        train_config.get("directed_prior_loss_weight", 0.0)
    )
    if helpfulness_loss_weight < 0.0:
        raise ValueError("train_erc.helpfulness_loss_weight cannot be negative")
    if residual_l1_weight < 0.0:
        raise ValueError("train_erc.residual_l1_weight cannot be negative")
    if directed_prior_loss_weight < 0.0:
        raise ValueError(
            "train_erc.directed_prior_loss_weight cannot be negative"
        )
    if directed_prior_loss_weight > 0.0:
        if not model.use_directed_update or model.directed_affect_updater is None:
            raise ValueError(
                "directed_prior_loss_weight requires the directed probabilistic updater"
            )
        if model.directed_affect_updater.matrix_feature_mode != "history_only":
            raise ValueError(
                "Supervised prior loss is restricted to history_only matrix features"
            )
    best_validation_weighted_f1 = float("-inf")
    best_validation_macro_f1 = float("-inf")

    for epoch in range(1, int(train_config.get("epochs", 20)) + 1):
        model.train()
        running_loss = 0.0
        running_main_loss = 0.0
        running_context_loss = 0.0
        running_transition_loss = 0.0
        running_helpfulness_loss = 0.0
        running_residual_l1 = 0.0
        running_directed_prior_loss = 0.0
        running_directed_prior_turns = 0
        running_turns = 0
        for batch in tqdm(train_loader, desc=f"ERC train {epoch}", leave=False):
            batch = move_to_device(batch, device_obj)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_obj.type, dtype=torch.float16, enabled=use_amp):
                if model.use_current_logit_anchor:
                    logits, auxiliary = model(
                        batch["embeddings"],
                        batch["speaker_ids"],
                        batch["mask"],
                        labels=batch["labels"],
                        teacher_forcing=float(train_config.get("teacher_forcing", 0.5)),
                        current_logits=batch.get("current_logits"),
                        return_aux=True,
                    )
                else:
                    logits = model(
                        batch["embeddings"],
                        batch["speaker_ids"],
                        batch["mask"],
                        labels=batch["labels"],
                        teacher_forcing=float(train_config.get("teacher_forcing", 0.5)),
                    )
                    auxiliary = None
                main_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["labels"].reshape(-1),
                    weight=main_class_weights,
                    ignore_index=-100,
                )
                context_loss = main_loss.new_zeros(())
                transition_loss = main_loss.new_zeros(())
                helpfulness_loss = main_loss.new_zeros(())
                residual_l1 = main_loss.new_zeros(())
                directed_prior_loss = main_loss.new_zeros(())
                directed_prior_turns = 0
                if auxiliary is not None:
                    context_logits = auxiliary["context_logits"]
                    context_loss = nn.functional.cross_entropy(
                        context_logits.reshape(-1, context_logits.shape[-1]),
                        batch["labels"].reshape(-1),
                        weight=context_class_weights,
                        ignore_index=-100,
                    )
                    known, transition = _same_speaker_transition_masks(
                        batch["speaker_ids"], batch["labels"], batch["mask"]
                    )
                    if known.any():
                        transition_loss = nn.functional.binary_cross_entropy_with_logits(
                            auxiliary["predicted_transition_logits"][known],
                            transition[known].to(logits.dtype),
                        )
                    if model.uses_helpfulness_gate:
                        helpful = _context_helpfulness_targets(
                            batch["current_logits"],
                            context_logits,
                            batch["labels"],
                            batch["mask"],
                            model.context_helpfulness_margin,
                        )
                        helpfulness_loss = (
                            nn.functional.binary_cross_entropy_with_logits(
                                auxiliary["predicted_helpfulness_logits"][batch["mask"]],
                                helpful[batch["mask"]].to(logits.dtype),
                            )
                        )
                    if model.use_class_residual:
                        residual_l1 = auxiliary["class_residual"][
                            batch["mask"]
                        ].abs().mean()
                    if model.use_directed_update:
                        speaker_known = (
                            auxiliary["speaker_known"] & batch["mask"]
                        )
                        if speaker_known.any():
                            directed_prior_turns = int(speaker_known.sum())
                            directed_prior_loss = nn.functional.nll_loss(
                                auxiliary["dynamic_prior"][speaker_known]
                                .float()
                                .clamp_min(1e-8)
                                .log(),
                                batch["labels"][speaker_known],
                            )
                loss = (
                    main_loss
                    + float(train_config.get("context_loss_weight", 0.0)) * context_loss
                    + float(train_config.get("transition_loss_weight", 0.0))
                    * transition_loss
                    + helpfulness_loss_weight * helpfulness_loss
                    + residual_l1_weight * residual_l1
                    + directed_prior_loss_weight * directed_prior_loss
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, float(train_config.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            turns = int(batch["mask"].sum())
            running_loss += float(loss.detach()) * turns
            running_main_loss += float(main_loss.detach()) * turns
            running_context_loss += float(context_loss.detach()) * turns
            running_transition_loss += float(transition_loss.detach()) * turns
            running_helpfulness_loss += float(helpfulness_loss.detach()) * turns
            running_residual_l1 += float(residual_l1.detach()) * turns
            running_directed_prior_loss += float(
                directed_prior_loss.detach()
            ) * directed_prior_turns
            running_directed_prior_turns += directed_prior_turns
            running_turns += turns

        validation = _evaluate_loader(model, validation_loader, device_obj, id_to_label, use_amp)
        train_loss = running_loss / max(running_turns, 1)
        train_main_loss = running_main_loss / max(running_turns, 1)
        train_context_loss = running_context_loss / max(running_turns, 1)
        train_transition_loss = running_transition_loss / max(running_turns, 1)
        train_helpfulness_loss = running_helpfulness_loss / max(running_turns, 1)
        train_residual_l1 = running_residual_l1 / max(running_turns, 1)
        train_directed_prior_loss = (
            running_directed_prior_loss
            / max(running_directed_prior_turns, 1)
        )
        selection_score, selection_values = _validation_selection_score(
            validation, selection_weights
        )
        weighted_f1 = selection_values["overall_weighted_f1"]
        macro_f1 = selection_values["overall_macro_f1"]
        transition_weighted_f1 = selection_values["transition_weighted_f1"]
        transition_macro_f1 = selection_values["transition_macro_f1"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_main_loss": train_main_loss,
                "train_context_loss": train_context_loss,
                "train_transition_loss": train_transition_loss,
                "train_helpfulness_loss": train_helpfulness_loss,
                "train_residual_l1": train_residual_l1,
                "train_directed_prior_loss": train_directed_prior_loss,
                "train_directed_prior_turns": running_directed_prior_turns,
                "validation_loss": float(validation["loss"]),
                "validation_weighted_f1": weighted_f1,
                "validation_macro_f1": macro_f1,
                "validation_transition_weighted_f1": transition_weighted_f1,
                "validation_transition_macro_f1": transition_macro_f1,
                "validation_selection_score": selection_score,
                "validation_helpfulness_average_precision": _nested_value(
                    validation,
                    (
                        "logit_fusion",
                        "helpfulness_detector",
                        "average_precision",
                    ),
                ),
                "validation_helpfulness_f1": _nested_value(
                    validation,
                    ("logit_fusion", "helpfulness_detector", "f1"),
                ),
            }
        )
        gate_text = ""
        if "logit_fusion" in validation:
            gate_mean = validation["logit_fusion"]["context_gate"]["overall"]["mean"]
            gate_text = f" gate={float(gate_mean):.4f}"
            helpfulness = validation["logit_fusion"].get(
                "helpfulness_detector"
            )
            if helpfulness is not None:
                help_ap = helpfulness.get("average_precision")
                help_f1 = helpfulness.get("f1")
                gate_text += (
                    f" helpAP={float(help_ap):.4f}"
                    if help_ap is not None
                    else " helpAP=n/a"
                )
                gate_text += (
                    f" helpF1={float(help_f1):.4f}"
                    if help_f1 is not None
                    else " helpF1=n/a"
                )
            residual = validation["logit_fusion"].get("class_residual")
            if residual is not None:
                residual_rms = residual["applied_rms"]["overall"]["mean"]
                gate_text += f" residualRMS={float(residual_rms):.4f}"
            directed_update = validation["logit_fusion"].get(
                "directed_update"
            )
            if directed_update is not None:
                prior_nll = directed_update.get(
                    "history_prior_quality", {}
                ).get("nll")
                gate_text += (
                    f" priorNLL={float(prior_nll):.4f}"
                    if prior_nll is not None
                    else " priorNLL=n/a"
                )
        print(
            f"[PFA-ERC] seed={seed} epoch={epoch:03d} train_loss={train_loss:.5f} "
            f"val_wF1={weighted_f1:.4f} val_macroF1={macro_f1:.4f} "
            f"val_transition_wF1={transition_weighted_f1:.4f} "
            f"val_transition_macroF1={transition_macro_f1:.4f} "
            f"selection={selection_score:.4f}{gate_text}"
        )
        improved, should_stop = stopper.update(selection_score)
        if improved:
            best_validation_weighted_f1 = weighted_f1
            best_validation_macro_f1 = macro_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation,
                    "validation_selection_score": selection_score,
                    "selection_metric_weights": selection_weights,
                    "class_balance": class_balance,
                    "oof_partition": oof_partition,
                    "label_to_id": label_to_id,
                    "config": config,
                    "seed": seed,
                },
                checkpoint,
            )
        if should_stop:
            break
    return {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "best_validation_weighted_f1": best_validation_weighted_f1,
        "best_validation_macro_f1": best_validation_macro_f1,
        "best_validation_selection_score": stopper.best,
        "selection_metric_weights": selection_weights,
        "class_balance": class_balance,
        "oof_partition": oof_partition,
        "epochs_run": len(history),
        "history": history,
    }


def evaluate_erc(
    config: dict[str, Any],
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
    save_report: bool = True,
) -> dict[str, Any]:
    device_obj = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    checkpoint_file = Path(checkpoint_path or config["train_erc"]["checkpoint"])
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    label_to_id = {str(key): int(value) for key, value in checkpoint["label_to_id"].items()}
    id_to_label = {value: key for key, value in label_to_id.items()}
    model = _build_model(config, len(label_to_id), device_obj)
    if model.use_current_logit_anchor:
        _validate_current_logit_features(store, len(label_to_id))
    model.load_state_dict(checkpoint["model_state"])
    split = split or config["evaluation"].get("split", "test")
    loader = _loader(
        store["splits"][split], config, False, int(checkpoint.get("seed", 42)), device_obj
    )
    use_amp = bool(config["train_erc"].get("use_amp", True)) and device_obj.type == "cuda"
    metrics = _evaluate_loader(model, loader, device_obj, id_to_label, use_amp)
    result = {
        "checkpoint": str(checkpoint_file),
        "split": split,
        "seed": int(checkpoint.get("seed", 42)),
        "metrics": metrics,
    }
    if save_report:
        save_json(result, config["evaluation"]["report_file"])
    return result


def _nested_value(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _scalar_summary(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    return {
        "values": numeric,
        "mean": mean(numeric) if numeric else None,
        "std": pstdev(numeric) if numeric else None,
        "count": len(numeric),
    }


def _summarize_emotion_reports(
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    output_prefixes = {
        "fusion": (),
        "anchor": ("anchor_only",),
        "context": ("context_only",),
    }
    subset_suffixes = {
        "overall": ("classification_report",),
        "stable": ("same_speaker_stable", "classification_report"),
        "transition": (
            "same_speaker_transition",
            "classification_report",
        ),
    }
    excluded = {"accuracy", "macro avg", "weighted avg"}
    summary: dict[str, Any] = {}
    for subset, suffix in subset_suffixes.items():
        summary[subset] = {}
        for output, prefix in output_prefixes.items():
            reports = []
            for item in metrics:
                report = _nested_value(item, (*prefix, *suffix))
                if isinstance(report, dict) and report:
                    reports.append(report)
            if not reports:
                continue
            emotions = [key for key in reports[0] if key not in excluded]
            summary[subset][output] = {
                emotion: {
                    metric: _scalar_summary(
                        [
                            report.get(emotion, {}).get(metric)
                            for report in reports
                        ]
                    )
                    for metric in ("precision", "recall", "f1-score", "support")
                }
                for emotion in emotions
            }
    return summary


def _summarize_seed_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric_paths = {
        "overall_weighted_f1": ("weighted_f1",),
        "overall_macro_f1": ("macro_f1",),
        "overall_accuracy": ("accuracy",),
        "stable_weighted_f1": ("same_speaker_stable", "weighted_f1"),
        "stable_macro_f1": ("same_speaker_stable", "macro_f1"),
        "transition_weighted_f1": (
            "same_speaker_transition",
            "weighted_f1",
        ),
        "transition_macro_f1": (
            "same_speaker_transition",
            "macro_f1",
        ),
        "overall_nll": ("probability_quality", "nll"),
        "overall_brier": ("probability_quality", "brier"),
        "overall_ece": ("probability_quality", "ece"),
        "anchor_nll": ("anchor_only", "probability_quality", "nll"),
        "anchor_brier": ("anchor_only", "probability_quality", "brier"),
        "anchor_ece": ("anchor_only", "probability_quality", "ece"),
        "context_nll": ("context_only", "probability_quality", "nll"),
        "context_brier": ("context_only", "probability_quality", "brier"),
        "context_ece": ("context_only", "probability_quality", "ece"),
        "anchor_overall_weighted_f1": ("anchor_only", "weighted_f1"),
        "anchor_overall_macro_f1": ("anchor_only", "macro_f1"),
        "anchor_stable_weighted_f1": (
            "anchor_only",
            "same_speaker_stable",
            "weighted_f1",
        ),
        "anchor_transition_weighted_f1": (
            "anchor_only",
            "same_speaker_transition",
            "weighted_f1",
        ),
        "anchor_stable_macro_f1": (
            "anchor_only",
            "same_speaker_stable",
            "macro_f1",
        ),
        "anchor_transition_macro_f1": (
            "anchor_only",
            "same_speaker_transition",
            "macro_f1",
        ),
        "context_overall_weighted_f1": ("context_only", "weighted_f1"),
        "context_overall_macro_f1": ("context_only", "macro_f1"),
        "context_stable_weighted_f1": (
            "context_only",
            "same_speaker_stable",
            "weighted_f1",
        ),
        "context_transition_weighted_f1": (
            "context_only",
            "same_speaker_transition",
            "weighted_f1",
        ),
        "context_stable_macro_f1": (
            "context_only",
            "same_speaker_stable",
            "macro_f1",
        ),
        "context_transition_macro_f1": (
            "context_only",
            "same_speaker_transition",
            "macro_f1",
        ),
        "gate_overall_mean": (
            "logit_fusion",
            "context_gate",
            "overall",
            "mean",
        ),
        "gate_stable_mean": (
            "logit_fusion",
            "context_gate",
            "same_speaker_stable",
            "mean",
        ),
        "gate_transition_mean": (
            "logit_fusion",
            "context_gate",
            "same_speaker_transition",
            "mean",
        ),
        "base_gate_overall_mean": (
            "logit_fusion",
            "base_context_gate",
            "overall",
            "mean",
        ),
        "base_gate_stable_mean": (
            "logit_fusion",
            "base_context_gate",
            "same_speaker_stable",
            "mean",
        ),
        "base_gate_transition_mean": (
            "logit_fusion",
            "base_context_gate",
            "same_speaker_transition",
            "mean",
        ),
        "capped_base_gate_overall_mean": (
            "logit_fusion",
            "capped_base_context_gate",
            "overall",
            "mean",
        ),
        "capped_base_gate_stable_mean": (
            "logit_fusion",
            "capped_base_context_gate",
            "same_speaker_stable",
            "mean",
        ),
        "capped_base_gate_transition_mean": (
            "logit_fusion",
            "capped_base_context_gate",
            "same_speaker_transition",
            "mean",
        ),
        "transition_detector_average_precision": (
            "logit_fusion",
            "transition_detector",
            "average_precision",
        ),
        "transition_detector_f1": (
            "logit_fusion",
            "transition_detector",
            "f1",
        ),
        "transition_detector_precision": (
            "logit_fusion",
            "transition_detector",
            "precision",
        ),
        "transition_detector_recall": (
            "logit_fusion",
            "transition_detector",
            "recall",
        ),
        "helpfulness_detector_average_precision": (
            "logit_fusion",
            "helpfulness_detector",
            "average_precision",
        ),
        "helpfulness_detector_f1": (
            "logit_fusion",
            "helpfulness_detector",
            "f1",
        ),
        "helpfulness_detector_stable_average_precision": (
            "logit_fusion",
            "helpfulness_detector",
            "same_speaker_stable",
            "average_precision",
        ),
        "helpfulness_detector_stable_f1": (
            "logit_fusion",
            "helpfulness_detector",
            "same_speaker_stable",
            "f1",
        ),
        "helpfulness_detector_transition_average_precision": (
            "logit_fusion",
            "helpfulness_detector",
            "same_speaker_transition",
            "average_precision",
        ),
        "helpfulness_detector_transition_f1": (
            "logit_fusion",
            "helpfulness_detector",
            "same_speaker_transition",
            "f1",
        ),
        "helpfulness_modulation_overall_mean": (
            "logit_fusion",
            "helpfulness_modulation",
            "overall",
            "mean",
        ),
        "helpfulness_modulation_stable_mean": (
            "logit_fusion",
            "helpfulness_modulation",
            "same_speaker_stable",
            "mean",
        ),
        "helpfulness_modulation_transition_mean": (
            "logit_fusion",
            "helpfulness_modulation",
            "same_speaker_transition",
            "mean",
        ),
        "fusion_beneficial_flips": (
            "logit_fusion",
            "fusion_flips",
            "overall",
            "beneficial_flips",
        ),
        "fusion_harmful_flips": (
            "logit_fusion",
            "fusion_flips",
            "overall",
            "harmful_flips",
        ),
        "fusion_net_flips": (
            "logit_fusion",
            "fusion_flips",
            "overall",
            "net_flips",
        ),
        "stable_fusion_net_flips": (
            "logit_fusion",
            "fusion_flips",
            "same_speaker_stable",
            "net_flips",
        ),
        "transition_fusion_net_flips": (
            "logit_fusion",
            "fusion_flips",
            "same_speaker_transition",
            "net_flips",
        ),
        "raw_residual_rms": (
            "logit_fusion",
            "class_residual",
            "raw_rms",
            "overall",
            "mean",
        ),
        "applied_residual_rms": (
            "logit_fusion",
            "class_residual",
            "applied_rms",
            "overall",
            "mean",
        ),
        "stable_applied_residual_rms": (
            "logit_fusion",
            "class_residual",
            "applied_rms",
            "same_speaker_stable",
            "mean",
        ),
        "transition_applied_residual_rms": (
            "logit_fusion",
            "class_residual",
            "applied_rms",
            "same_speaker_transition",
            "mean",
        ),
        "directed_self_gate": (
            "logit_fusion",
            "directed_update",
            "self_gate",
            "available_only",
            "mean",
        ),
        "directed_other_gate": (
            "logit_fusion",
            "directed_update",
            "other_gate",
            "available_only",
            "mean",
        ),
        "directed_transition_gate": (
            "logit_fusion",
            "directed_update",
            "transition_gate",
            "known_speaker_only",
            "mean",
        ),
        "directed_self_available_rate": (
            "logit_fusion",
            "directed_update",
            "availability_rate",
            "self",
        ),
        "directed_other_available_rate": (
            "logit_fusion",
            "directed_update",
            "availability_rate",
            "other",
        ),
        "directed_anchor_prior_js": (
            "logit_fusion",
            "directed_update",
            "anchor_prior_js_divergence",
            "mean",
        ),
        "directed_matrix_diagonal": (
            "logit_fusion",
            "directed_update",
            "transition_matrix_diagonal",
            "mean",
        ),
        "directed_matrix_delta_rms": (
            "logit_fusion",
            "directed_update",
            "matrix_delta_rms",
            "mean",
        ),
        "directed_history_prior_nll": (
            "logit_fusion",
            "directed_update",
            "history_prior_quality",
            "nll",
        ),
        "directed_history_prior_brier": (
            "logit_fusion",
            "directed_update",
            "history_prior_quality",
            "brier",
        ),
        "directed_history_prior_ece": (
            "logit_fusion",
            "directed_update",
            "history_prior_quality",
            "ece",
        ),
        "directed_history_prior_top1_accuracy": (
            "logit_fusion",
            "directed_update",
            "history_prior_quality",
            "top1_accuracy",
        ),
        "directed_history_prior_macro_f1": (
            "logit_fusion",
            "directed_update",
            "history_prior_quality",
            "macro_f1",
        ),
        "directed_uniform_prior_nll": (
            "logit_fusion",
            "directed_update",
            "history_prior_baselines",
            "uniform",
            "nll",
        ),
        "directed_class_marginal_prior_nll": (
            "logit_fusion",
            "directed_update",
            "history_prior_baselines",
            "class_marginal",
            "nll",
        ),
        "directed_prior_nll_improvement_over_uniform": (
            "logit_fusion",
            "directed_update",
            "history_prior_nll_improvement",
            "over_uniform",
        ),
        "directed_prior_nll_improvement_over_class_marginal": (
            "logit_fusion",
            "directed_update",
            "history_prior_nll_improvement",
            "over_class_marginal",
        ),
        "directed_anchor_entropy": (
            "logit_fusion",
            "directed_update",
            "anchor_entropy",
            "mean",
        ),
        "directed_history_prior_entropy": (
            "logit_fusion",
            "directed_update",
            "history_prior_entropy",
            "mean",
        ),
    }
    metrics = [run["evaluation"]["metrics"] for run in runs]
    summary = {
        name: _scalar_summary([_nested_value(item, path) for item in metrics])
        for name, path in metric_paths.items()
    }
    summary["best_validation_selection_score"] = _scalar_summary(
        [run["training"].get("best_validation_selection_score") for run in runs]
    )
    summary["best_validation_weighted_f1"] = _scalar_summary(
        [run["training"].get("best_validation_weighted_f1") for run in runs]
    )
    summary["best_validation_macro_f1"] = _scalar_summary(
        [run["training"].get("best_validation_macro_f1") for run in runs]
    )
    summary["epochs_run"] = _scalar_summary(
        [run["training"].get("epochs_run") for run in runs]
    )
    summary["classification_by_emotion"] = _summarize_emotion_reports(metrics)
    return summary


def _validate_reusable_seed_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    seed: int,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(checkpoint.get("seed", -1)) != seed:
        raise ValueError(
            f"Existing checkpoint {checkpoint_path} has seed="
            f"{checkpoint.get('seed')}, expected {seed}"
        )
    expected_experiment = str(config["experiment"]["name"])
    compatible_experiments = {
        str(name)
        for name in config.get("evaluation", {}).get(
            "compatible_checkpoint_experiments", [expected_experiment]
        )
    }
    compatible_experiments.add(expected_experiment)
    saved_experiment = str(
        checkpoint.get("config", {}).get("experiment", {}).get("name", "")
    )
    if saved_experiment not in compatible_experiments:
        raise ValueError(
            f"Existing checkpoint {checkpoint_path} belongs to experiment "
            f"{saved_experiment!r}, expected one of "
            f"{sorted(compatible_experiments)!r}. Move it aside before restarting "
            "the seed sweep."
        )
    saved_config = checkpoint.get("config", {})
    expected_model = config.get("model", {})
    saved_model = saved_config.get("model", {})
    expected_train = config.get("train_erc", {})
    saved_train = saved_config.get("train_erc", {})
    lineage_values = {
        "logit_fusion": (
            str(expected_model.get("logit_fusion", "none")),
            str(saved_model.get("logit_fusion", "none")),
        ),
        "context_gate_mode": (
            str(expected_model.get("context_gate_mode", "transition")),
            str(saved_model.get("context_gate_mode", "transition")),
        ),
        "transition_gate_power": (
            float(expected_model.get("transition_gate_power", 1.0)),
            float(saved_model.get("transition_gate_power", 1.0)),
        ),
        "context_gate_init": (
            float(expected_model.get("context_gate_init", 0.1)),
            float(saved_model.get("context_gate_init", 0.1)),
        ),
        "context_gate_max": (
            float(expected_model.get("context_gate_max", 1.0)),
            float(saved_model.get("context_gate_max", 1.0)),
        ),
        "context_helpfulness_margin": (
            float(expected_model.get("context_helpfulness_margin", 0.0)),
            float(saved_model.get("context_helpfulness_margin", 0.0)),
        ),
        "helpfulness_prior": (
            float(expected_model.get("helpfulness_prior", 0.55)),
            float(saved_model.get("helpfulness_prior", 0.55)),
        ),
        "helpfulness_gate_power": (
            float(expected_model.get("helpfulness_gate_power", 1.0)),
            float(saved_model.get("helpfulness_gate_power", 1.0)),
        ),
        "helpfulness_loss_weight": (
            float(expected_train.get("helpfulness_loss_weight", 0.0)),
            float(
                saved_train.get("helpfulness_loss_weight", 0.0)
            ),
        ),
        "residual_use_change": (
            bool(expected_model.get("residual_use_change", False)),
            bool(saved_model.get("residual_use_change", False)),
        ),
        "residual_use_trigger": (
            bool(expected_model.get("residual_use_trigger", False)),
            bool(saved_model.get("residual_use_trigger", False)),
        ),
        "residual_use_vector_gate": (
            bool(expected_model.get("residual_use_vector_gate", False)),
            bool(saved_model.get("residual_use_vector_gate", False)),
        ),
        "residual_use_transition_feature": (
            bool(expected_model.get("residual_use_transition_feature", False)),
            bool(saved_model.get("residual_use_transition_feature", False)),
        ),
        "residual_gate_init": (
            float(expected_model.get("residual_gate_init", 0.5)),
            float(saved_model.get("residual_gate_init", 0.5)),
        ),
        "residual_scale": (
            float(expected_model.get("residual_scale", 1.0)),
            float(saved_model.get("residual_scale", 1.0)),
        ),
        "residual_l1_weight": (
            float(expected_train.get("residual_l1_weight", 0.0)),
            float(saved_train.get("residual_l1_weight", 0.0)),
        ),
        "directed_variant": (
            str(expected_model.get("directed_variant", "full")),
            str(saved_model.get("directed_variant", "full")),
        ),
        "directed_matrix_mode": (
            str(expected_model.get("directed_matrix_mode", "dynamic")),
            str(saved_model.get("directed_matrix_mode", "dynamic")),
        ),
        "directed_matrix_feature_mode": (
            str(
                expected_model.get(
                    "directed_matrix_feature_mode",
                    "current_conditioned",
                )
            ),
            str(
                saved_model.get(
                    "directed_matrix_feature_mode",
                    "current_conditioned",
                )
            ),
        ),
        "directed_fusion_mode": (
            str(expected_model.get("directed_fusion_mode", "")),
            str(saved_model.get("directed_fusion_mode", "")),
        ),
        "directed_class_prior": (
            tuple(
                float(value)
                for value in expected_model.get(
                    "directed_class_prior", []
                )
            ),
            tuple(
                float(value)
                for value in saved_model.get(
                    "directed_class_prior", []
                )
            ),
        ),
        "directed_prior_mode": (
            str(expected_model.get("directed_prior_mode", "speaker")),
            str(saved_model.get("directed_prior_mode", "speaker")),
        ),
        "directed_gate_init": (
            float(expected_model.get("directed_gate_init", 0.05)),
            float(saved_model.get("directed_gate_init", 0.05)),
        ),
        "directed_transition_gate_init": (
            float(
                expected_model.get(
                    "directed_transition_gate_init",
                    expected_model.get("directed_gate_init", 0.05),
                )
            ),
            float(
                saved_model.get(
                    "directed_transition_gate_init",
                    saved_model.get("directed_gate_init", 0.05),
                )
            ),
        ),
        "directed_correction_scale": (
            float(expected_model.get("directed_correction_scale", 1.0)),
            float(saved_model.get("directed_correction_scale", 1.0)),
        ),
        "directed_transition_scale": (
            float(expected_model.get("directed_transition_scale", 1.0)),
            float(saved_model.get("directed_transition_scale", 1.0)),
        ),
        "directed_transition_diagonal_bias": (
            float(
                expected_model.get("directed_transition_diagonal_bias", 2.0)
            ),
            float(saved_model.get("directed_transition_diagonal_bias", 2.0)),
        ),
        "directed_detach_belief_history": (
            bool(expected_model.get("directed_detach_belief_history", True)),
            bool(saved_model.get("directed_detach_belief_history", True)),
        ),
        "directed_prior_loss_weight": (
            float(expected_train.get("directed_prior_loss_weight", 0.0)),
            float(saved_train.get("directed_prior_loss_weight", 0.0)),
        ),
        "class_weight_strategy": (
            str(expected_train.get("class_weight_strategy", "none")),
            str(saved_train.get("class_weight_strategy", "none")),
        ),
        "class_weight_power": (
            float(expected_train.get("class_weight_power", 0.5)),
            float(saved_train.get("class_weight_power", 0.5)),
        ),
        "class_weight_min": (
            float(expected_train.get("class_weight_min", 0.25)),
            float(saved_train.get("class_weight_min", 0.25)),
        ),
        "class_weight_max": (
            float(expected_train.get("class_weight_max", 3.0)),
            float(saved_train.get("class_weight_max", 3.0)),
        ),
        "class_balance_main_loss": (
            bool(expected_train.get("class_balance_main_loss", False)),
            bool(saved_train.get("class_balance_main_loss", False)),
        ),
        "class_balance_context_loss": (
            bool(expected_train.get("class_balance_context_loss", False)),
            bool(saved_train.get("class_balance_context_loss", False)),
        ),
        "selection_metric_weights": (
            tuple(_selection_metric_weights(expected_train).items()),
            tuple(_selection_metric_weights(saved_train).items()),
        ),
    }
    mismatches = [
        f"{key}: saved={saved!r}, expected={expected!r}"
        for key, (expected, saved) in lineage_values.items()
        if saved != expected
    ]
    if mismatches:
        raise ValueError(
            f"Existing checkpoint {checkpoint_path} has incompatible settings: "
            + "; ".join(mismatches)
        )


def train_seed_sweep(config: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    base_checkpoint = Path(config["train_erc"]["checkpoint"])
    report_file = Path(config["evaluation"]["report_file"])
    seeds = [
        int(seed)
        for seed in config["evaluation"].get("seeds", [13, 21, 42, 87, 100])
    ]
    reuse_existing = bool(
        config["evaluation"].get("reuse_existing_checkpoints", False)
    )
    save_individual = bool(
        config["evaluation"].get("save_individual_seed_reports", True)
    )
    protocol = {
        "seeds": seeds,
        "evaluation_split": str(config["evaluation"].get("split", "test")),
        "fixed_feature_file": str(config["data"]["feature_file"]),
        "fixed_anchor_checkpoint": str(
            config.get("features", {}).get("encoder_checkpoint", "")
        ),
        "reuse_existing_checkpoints": reuse_existing,
        "compatible_checkpoint_experiments": config["evaluation"].get(
            "compatible_checkpoint_experiments", []
        ),
        "class_balance": {
            "strategy": str(
                config["train_erc"].get("class_weight_strategy", "none")
            ),
            "power": float(config["train_erc"].get("class_weight_power", 0.5)),
            "minimum": float(config["train_erc"].get("class_weight_min", 0.25)),
            "maximum": float(config["train_erc"].get("class_weight_max", 3.0)),
            "main_loss": bool(
                config["train_erc"].get("class_balance_main_loss", False)
            ),
            "context_loss": bool(
                config["train_erc"].get("class_balance_context_loss", False)
            ),
        },
        "selection_metric_weights": _selection_metric_weights(
            config["train_erc"]
        ),
        "gate": {
            "mode": str(config["model"].get("context_gate_mode", "transition")),
            "transition_power": float(
                config["model"].get("transition_gate_power", 1.0)
            ),
            "helpfulness_power": float(
                config["model"].get("helpfulness_gate_power", 1.0)
            ),
            "helpfulness_prior": float(
                config["model"].get("helpfulness_prior", 0.55)
            ),
            "helpfulness_margin": float(
                config["model"].get("context_helpfulness_margin", 0.0)
            ),
            "helpfulness_loss_weight": float(
                config["train_erc"].get("helpfulness_loss_weight", 0.0)
            ),
        },
        "class_residual": {
            "enabled": str(config["model"].get("logit_fusion", "none"))
            == "current_anchored_class_residual",
            "use_change": bool(
                config["model"].get("residual_use_change", False)
            ),
            "use_trigger": bool(
                config["model"].get("residual_use_trigger", False)
            ),
            "use_vector_gate": bool(
                config["model"].get("residual_use_vector_gate", False)
            ),
            "use_transition_feature": bool(
                config["model"].get(
                    "residual_use_transition_feature", False
                )
            ),
            "gate_init": float(
                config["model"].get("residual_gate_init", 0.5)
            ),
            "scale": float(config["model"].get("residual_scale", 1.0)),
            "l1_weight": float(
                config["train_erc"].get("residual_l1_weight", 0.0)
            ),
        },
        "directed_affect": {
            "enabled": str(config["model"].get("logit_fusion", "none"))
            == "directed_probabilistic_update",
            "variant": str(config["model"].get("directed_variant", "full")),
            "matrix_mode": str(
                config["model"].get("directed_matrix_mode", "dynamic")
            ),
            "matrix_feature_mode": str(
                config["model"].get(
                    "directed_matrix_feature_mode",
                    "current_conditioned",
                )
            ),
            "fusion_mode": str(
                config["model"].get("directed_fusion_mode", "")
            ),
            "prior_mode": str(
                config["model"].get("directed_prior_mode", "speaker")
            ),
            "class_prior": [
                float(value)
                for value in config["model"].get(
                    "directed_class_prior", []
                )
            ],
            "gate_init": float(
                config["model"].get("directed_gate_init", 0.05)
            ),
            "transition_gate_init": float(
                config["model"].get(
                    "directed_transition_gate_init",
                    config["model"].get("directed_gate_init", 0.05),
                )
            ),
            "correction_scale": float(
                config["model"].get("directed_correction_scale", 1.0)
            ),
            "transition_scale": float(
                config["model"].get("directed_transition_scale", 1.0)
            ),
            "detach_belief_history": bool(
                config["model"].get(
                    "directed_detach_belief_history", True
                )
            ),
            "prior_loss_weight": float(
                config["train_erc"].get(
                    "directed_prior_loss_weight", 0.0
                )
            ),
        },
    }
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        seed_checkpoint = base_checkpoint.with_name(f"{base_checkpoint.stem}_seed{seed}.pt")
        if reuse_existing and seed_checkpoint.exists():
            _validate_reusable_seed_checkpoint(seed_checkpoint, config, seed)
            stored = torch.load(
                seed_checkpoint, map_location="cpu", weights_only=False
            )
            train_result = {
                "checkpoint": str(seed_checkpoint),
                "seed": seed,
                "best_validation_weighted_f1": stored.get(
                    "validation_metrics", {}
                ).get("weighted_f1"),
                "best_validation_macro_f1": stored.get(
                    "validation_metrics", {}
                ).get("macro_f1"),
                "best_validation_selection_score": stored.get(
                    "validation_selection_score"
                ),
                "best_epoch": int(stored.get("epoch", 0)),
                "epochs_run": None,
                "selection_metric_weights": stored.get(
                    "selection_metric_weights"
                ),
                "class_balance": stored.get("class_balance"),
                "reused_existing_checkpoint": True,
            }
            print(f"[seed sweep] reusing completed seed={seed}: {seed_checkpoint}")
        else:
            train_result = train_erc(
                config,
                device=device,
                seed=seed,
                checkpoint_path=seed_checkpoint,
            )
            train_result["reused_existing_checkpoint"] = False
        evaluation = evaluate_erc(
            config, device=device, checkpoint_path=seed_checkpoint, save_report=False
        )
        individual_report = report_file.with_name(
            f"{report_file.stem}_seed{seed}{report_file.suffix}"
        )
        if save_individual:
            save_json(evaluation, individual_report)
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(seed_checkpoint),
                "individual_report": str(individual_report)
                if save_individual
                else None,
                "training": train_result,
                "evaluation": evaluation,
            }
        )
        partial_summary = _summarize_seed_runs(runs)
        save_json(
            {
                "status": "in_progress",
                "protocol": protocol,
                "completed_seeds": [run["seed"] for run in runs],
                "summary": partial_summary,
                "runs": runs,
                "weighted_f1_mean": partial_summary["overall_weighted_f1"][
                    "mean"
                ],
                "weighted_f1_std": partial_summary["overall_weighted_f1"]["std"],
                "num_seeds": len(runs),
            },
            report_file,
        )
    summary = _summarize_seed_runs(runs)
    result = {
        "status": "completed",
        "protocol": protocol,
        "completed_seeds": [run["seed"] for run in runs],
        "summary": summary,
        "runs": runs,
        "weighted_f1_mean": summary["overall_weighted_f1"]["mean"],
        "weighted_f1_std": summary["overall_weighted_f1"]["std"],
        "num_seeds": len(runs),
    }
    save_json(result, report_file)
    return result
