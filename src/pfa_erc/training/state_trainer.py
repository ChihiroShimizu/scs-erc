from __future__ import annotations

import copy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..models import EventTransitionMatrixTracker, ProbabilisticEmotionStateTracker
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


def same_speaker_transition_masks(
    speaker_ids: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    known = torch.zeros_like(valid_mask)
    transition = torch.zeros_like(valid_mask)
    for batch_index in range(speaker_ids.shape[0]):
        previous: dict[int, int] = {}
        for raw_index in torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten():
            index = int(raw_index)
            speaker = int(speaker_ids[batch_index, index])
            label = int(labels[batch_index, index])
            if speaker in previous:
                known[batch_index, index] = True
                transition[batch_index, index] = label != previous[speaker]
            previous[speaker] = label
    return known, transition


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
        batch_size=int(config["train_state"].get("batch_size", 8)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=int(config["experiment"].get("num_workers", 0)),
        collate_fn=collate_dialogues,
        pin_memory=device.type == "cuda",
    )


def _validate_store(
    store: dict[str, Any], config: dict[str, Any], num_classes: int
) -> None:
    embedding_dim = int(config["state_model"].get("embedding_dim", store["embedding_dim"]))
    if int(store["embedding_dim"]) != embedding_dim:
        raise ValueError(
            f"Feature dimension {store['embedding_dim']} does not match state_model.embedding_dim={embedding_dim}"
        )
    if str(config["state_model"].get("anchor_source", "cached_logits")) != "cached_logits":
        return
    for split, dialogues in store["splits"].items():
        for dialogue in dialogues:
            length = int(dialogue["embeddings"].shape[0])
            logits = dialogue.get("current_logits")
            if logits is None or tuple(logits.shape) != (length, num_classes):
                raise ValueError(
                    "Cached-logit state experiments require feature extraction from a 3A "
                    f"classifier; invalid current_logits in {split}:{dialogue.get('dialogue_id')}"
                )


def _class_weights(
    dialogues: list[dict[str, Any]],
    num_classes: int,
    train_cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor | None, list[int]]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for dialogue in dialogues:
        labels = torch.as_tensor(dialogue["labels"], dtype=torch.long)
        counts += torch.bincount(labels[labels >= 0], minlength=num_classes)
    strategy = str(train_cfg.get("class_weight_strategy", "none"))
    if strategy == "none":
        return None, counts.tolist()
    if strategy != "inverse_frequency":
        raise ValueError("train_state.class_weight_strategy must be none or inverse_frequency")
    if torch.any(counts == 0):
        raise ValueError("Cannot class-balance when a training class has zero examples")
    power = float(train_cfg.get("class_weight_power", 0.5))
    weights = counts.float().pow(-power)
    weights /= weights.mean()
    weights = weights.clamp(
        min=float(train_cfg.get("class_weight_min", 0.25)),
        max=float(train_cfg.get("class_weight_max", 3.0)),
    )
    return weights.to(device), counts.tolist()


def _build_state_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    architecture = str(config["state_model"].get("architecture", "recurrent"))
    if architecture == "recurrent":
        return ProbabilisticEmotionStateTracker(config, num_classes)
    if architecture == "transition_matrix":
        return EventTransitionMatrixTracker(config, num_classes)
    raise ValueError(
        "state_model.architecture must be recurrent or transition_matrix, "
        f"got {architecture!r}"
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError(
            f"_weighted_mean expects a 1-D value tensor, got {tuple(values.shape)}"
        )
    if weights is None:
        return values.mean()
    if weights.shape != values.shape:
        raise ValueError(
            "_weighted_mean requires weights and values to have the same shape; "
            f"got {tuple(weights.shape)} and {tuple(values.shape)}"
        )
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def _loss(
    logits: torch.Tensor,
    aux: dict[str, torch.Tensor],
    batch: dict[str, Any],
    class_criterion: nn.CrossEntropyLoss,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["mask"] & batch["labels"].ge(0)
    labels = batch["labels"]
    class_loss = class_criterion(logits[valid], labels[valid])
    known, transition = same_speaker_transition_masks(
        batch["speaker_ids"], labels, valid
    )
    known = known & aux["state_known_mask"]
    if known.any():
        prior = aux["prior_emotion_distribution"][known].float()
        known_labels = labels[known]
        gold_probability = prior.gather(-1, known_labels.unsqueeze(-1)).squeeze(-1)
        state_weights = None
        if bool(train_cfg.get("state_nll_class_weighted", False)):
            if class_criterion.weight is None:
                raise ValueError(
                    "state_nll_class_weighted=true requires class_weight_strategy"
                )
            state_weights = class_criterion.weight[known_labels].float()
        state_nll = _weighted_mean(
            -torch.log(gold_probability.clamp_min(1e-8)), state_weights
        )
        one_hot = nn.functional.one_hot(
            known_labels, num_classes=prior.shape[-1]
        ).float()
        state_brier = _weighted_mean(
            ((prior - one_hot) ** 2).sum(dim=-1), state_weights
        )
        transition_loss = nn.functional.binary_cross_entropy_with_logits(
            aux["transition_logits"][known].float(), transition[known].float()
        )
    else:
        state_nll = class_loss.new_zeros(())
        state_brier = class_loss.new_zeros(())
        transition_loss = class_loss.new_zeros(())
    total = (
        class_loss
        + float(train_cfg.get("state_nll_weight", 0.2)) * state_nll
        + float(train_cfg.get("state_brier_weight", 0.0)) * state_brier
        + float(train_cfg.get("transition_loss_weight", 0.1)) * transition_loss
    )
    if not total.requires_grad:
        total = total + aux["context_scale"].sum() * 0.0
    return total, {
        "classification": float(class_loss.detach()),
        "state_nll": float(state_nll.detach()),
        "state_brier": float(state_brier.detach()),
        "transition": float(transition_loss.detach()),
    }


def _subset_metrics(
    labels: list[int], predictions: list[int], id_to_label: dict[int, str]
) -> dict[str, Any]:
    if not labels:
        return {"weighted_f1": None, "macro_f1": None, "accuracy": None, "utterances": 0}
    result = classification_metrics(labels, predictions, id_to_label)
    result["utterances"] = len(labels)
    return result


def _expected_calibration_error(
    probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10
) -> float:
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels).float()
    ece = probabilities.new_zeros(())
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=probabilities.device)
    for index in range(bins):
        selected = (confidence > boundaries[index]) & (confidence <= boundaries[index + 1])
        if selected.any():
            ece += selected.float().mean() * (
                correct[selected].mean() - confidence[selected].mean()
            ).abs()
    return float(ece)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
    use_amp: bool,
    train_cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    all_labels: list[int] = []
    all_predictions: list[int] = []
    anchor_predictions: list[int] = []
    stable_labels: list[int] = []
    stable_predictions: list[int] = []
    transition_labels: list[int] = []
    transition_predictions: list[int] = []
    transition_targets: list[int] = []
    transition_probabilities: list[float] = []
    prior_probabilities: list[torch.Tensor] = []
    prior_labels: list[torch.Tensor] = []
    gates: list[float] = []
    stable_gates: list[float] = []
    transition_gates: list[float] = []
    update_gates: list[float] = []
    stable_update_gates: list[float] = []
    transition_update_gates: list[float] = []
    transition_matrix_diagonal: list[float] = []
    losses: list[float] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = move_to_device(raw_batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp and device.type == "cuda",
            ):
                logits, aux = model(
                    batch["embeddings"],
                    batch["speaker_ids"],
                    batch["mask"],
                    current_logits=batch.get("current_logits"),
                    return_aux=True,
                )
                loss, _ = _loss(logits, aux, batch, criterion, train_cfg)
            losses.append(float(loss))
            valid = batch["mask"] & batch["labels"].ge(0)
            known, transition = same_speaker_transition_masks(
                batch["speaker_ids"], batch["labels"], valid
            )
            stable = known & ~transition
            predictions = logits.argmax(dim=-1)
            anchor = aux["anchor_logits"].argmax(dim=-1)
            gate = aux["influence_gate"].squeeze(-1)
            update_gate = aux.get("state_update_gate")

            all_labels.extend(batch["labels"][valid].cpu().tolist())
            all_predictions.extend(predictions[valid].cpu().tolist())
            anchor_predictions.extend(anchor[valid].cpu().tolist())
            stable_labels.extend(batch["labels"][stable].cpu().tolist())
            stable_predictions.extend(predictions[stable].cpu().tolist())
            transition_labels.extend(batch["labels"][transition].cpu().tolist())
            transition_predictions.extend(predictions[transition].cpu().tolist())
            transition_targets.extend(transition[known].long().cpu().tolist())
            transition_probability = aux.get("transition_probability")
            if transition_probability is None:
                transition_probability = torch.sigmoid(aux["transition_logits"].float())
            transition_probabilities.extend(
                transition_probability[known].float().cpu().tolist()
            )
            prior_probabilities.append(aux["prior_emotion_distribution"][known].float().cpu())
            prior_labels.append(batch["labels"][known].cpu())
            gates.extend(gate[valid].float().cpu().tolist())
            stable_gates.extend(gate[stable].float().cpu().tolist())
            transition_gates.extend(gate[transition].float().cpu().tolist())
            if update_gate is not None:
                update_gate = update_gate.squeeze(-1)
                update_gates.extend(update_gate[valid].float().cpu().tolist())
                stable_update_gates.extend(update_gate[stable].float().cpu().tolist())
                transition_update_gates.extend(
                    update_gate[transition].float().cpu().tolist()
                )
            event_matrix = aux.get("event_transition_matrix")
            if event_matrix is not None:
                diagonal = torch.diagonal(
                    event_matrix.float(), dim1=-2, dim2=-1
                ).mean(dim=-1)
                transition_matrix_diagonal.extend(diagonal[valid].cpu().tolist())

    result = classification_metrics(all_labels, all_predictions, id_to_label)
    result["utterances"] = len(all_labels)
    result["loss"] = mean(losses) if losses else None
    result["anchor"] = classification_metrics(
        all_labels, anchor_predictions, id_to_label
    )
    result["same_speaker_stable"] = _subset_metrics(
        stable_labels, stable_predictions, id_to_label
    )
    result["same_speaker_transition"] = _subset_metrics(
        transition_labels, transition_predictions, id_to_label
    )
    good = sum(
        a != y and p == y
        for y, a, p in zip(all_labels, anchor_predictions, all_predictions)
    )
    bad = sum(
        a == y and p != y
        for y, a, p in zip(all_labels, anchor_predictions, all_predictions)
    )
    result["fusion_flips"] = {"beneficial": good, "harmful": bad, "net": good - bad}

    if prior_probabilities and sum(tensor.shape[0] for tensor in prior_probabilities) > 0:
        probs = torch.cat(prior_probabilities)
        gold = torch.cat(prior_labels)
        one_hot = nn.functional.one_hot(gold, num_classes=len(id_to_label)).float()
        result["emotion_state_forecast"] = {
            "known_turns": int(gold.numel()),
            "nll": float(
                -torch.log(
                    probs.gather(-1, gold.unsqueeze(-1)).clamp_min(1e-8)
                ).mean()
            ),
            "brier": float(((probs - one_hot) ** 2).sum(dim=-1).mean()),
            "ece": _expected_calibration_error(probs, gold),
            "accuracy": float(probs.argmax(dim=-1).eq(gold).float().mean()),
        }
        if class_weights is not None:
            forecast_weights = class_weights.detach().cpu()[gold]
            nll_values = -torch.log(
                probs.gather(-1, gold.unsqueeze(-1))
                .squeeze(-1)
                .clamp_min(1e-8)
            )
            brier_values = ((probs - one_hot) ** 2).sum(dim=-1)
            result["emotion_state_forecast"]["class_balanced_nll"] = float(
                _weighted_mean(nll_values, forecast_weights)
            )
            result["emotion_state_forecast"]["class_balanced_brier"] = float(
                _weighted_mean(brier_values, forecast_weights)
            )
    else:
        result["emotion_state_forecast"] = {"known_turns": 0}

    if transition_targets:
        binary = [int(value >= 0.5) for value in transition_probabilities]
        precision, recall, f1, _ = precision_recall_fscore_support(
            transition_targets, binary, average="binary", zero_division=0
        )
        ap = (
            float(average_precision_score(transition_targets, transition_probabilities))
            if len(set(transition_targets)) > 1
            else None
        )
        result["transition_detector"] = {
            "average_precision": ap,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "target_rate": mean(transition_targets),
            "predicted_rate": mean(binary),
        }
    result["influence_gate"] = {
        "overall_mean": mean(gates) if gates else None,
        "stable_mean": mean(stable_gates) if stable_gates else None,
        "transition_mean": mean(transition_gates) if transition_gates else None,
        "context_scale": float(
            (
                model.context_scale_max
                * torch.sigmoid(model.context_scale_logit)
            ).detach()
        ),
    }
    if update_gates:
        result["state_update_gate"] = {
            "overall_mean": mean(update_gates),
            "stable_mean": mean(stable_update_gates) if stable_update_gates else None,
            "transition_mean": (
                mean(transition_update_gates) if transition_update_gates else None
            ),
        }
    if transition_matrix_diagonal:
        diagonal_mean = mean(transition_matrix_diagonal)
        result["event_transition_matrix"] = {
            "mean_diagonal_probability": diagonal_mean,
            "mean_off_diagonal_mass": 1.0 - diagonal_mean,
        }
    return result


def _selection_score(metrics: dict[str, Any], train_cfg: dict[str, Any]) -> float:
    overall_weight = float(train_cfg.get("selection_overall_macro_weight", 0.5))
    if not 0.0 <= overall_weight <= 1.0:
        raise ValueError("selection_overall_macro_weight must be in [0, 1]")
    transition_macro = metrics["same_speaker_transition"].get("macro_f1")
    if transition_macro is None:
        transition_macro = metrics["macro_f1"]
    return overall_weight * float(metrics["macro_f1"]) + (1.0 - overall_weight) * float(
        transition_macro
    )


def train_probabilistic_state(
    config: dict[str, Any],
    device: str | None = None,
    seed_override: int | None = None,
    checkpoint_override: str | Path | None = None,
) -> dict[str, Any]:
    train_cfg = config["train_state"]
    seed = int(
        seed_override
        if seed_override is not None
        else config["experiment"].get("seed", 42)
    )
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    torch_device = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    num_classes = len(store["label_to_id"])
    _validate_store(store, config, num_classes)
    model = _build_state_model(config, num_classes).to(torch_device)
    train_loader = _loader(store["splits"]["train"], config, True, seed, torch_device)
    validation_loader = _loader(
        store["splits"]["validation"], config, False, seed, torch_device
    )
    class_weights, class_counts = _class_weights(
        store["splits"]["train"], num_classes, train_cfg, torch_device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    epochs = int(train_cfg.get("epochs", 30))
    total_steps = max(epochs * len(train_loader), 1)
    warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.1)) * total_steps)
    scheduler = LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay_lambda(step, warmup_steps, total_steps),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and torch_device.type == "cuda"
    scaler = make_grad_scaler(torch_device, use_amp)
    stopper = EarlyStopping(int(train_cfg.get("patience", 8)), mode="max")
    checkpoint = Path(checkpoint_override or train_cfg["checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    id_to_label = {int(index): label for label, index in store["label_to_id"].items()}
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running: list[float] = []
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, torch_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, aux = model(
                    batch["embeddings"],
                    batch["speaker_ids"],
                    batch["mask"],
                    current_logits=batch.get("current_logits"),
                    return_aux=True,
                )
                loss, _ = _loss(logits, aux, batch, criterion, train_cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running.append(float(loss.detach()))

        validation = _evaluate(
            model, validation_loader, torch_device, id_to_label, use_amp, train_cfg, class_weights
        )
        score = _selection_score(validation, train_cfg)
        transition_macro = validation["same_speaker_transition"]["macro_f1"]
        if transition_macro is None:
            transition_macro = validation["macro_f1"]
        record = {
            "epoch": epoch,
            "train_loss": mean(running),
            "validation_macro_f1": validation["macro_f1"],
            "validation_transition_macro_f1": transition_macro,
            "selection_score": score,
        }
        history.append(record)
        print(
            f"[State-ERC] seed={seed} epoch={epoch:03d} loss={record['train_loss']:.5f} "
            f"val_macro={record['validation_macro_f1']:.4f} "
            f"val_transition_macro={record['validation_transition_macro_f1']:.4f} "
            f"selection={score:.4f}"
        )
        improved, stop = stopper.update(score)
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "selection_score": score,
                    "label_to_id": store["label_to_id"],
                    "embedding_dim": store["embedding_dim"],
                    "variant": config["state_model"].get("variant", "full"),
                    "architecture": config["state_model"].get(
                        "architecture", "recurrent"
                    ),
                    "class_counts": class_counts,
                },
                checkpoint,
            )
        if stop:
            break
    return {
        "checkpoint": str(checkpoint),
        "seed": seed,
        "variant": config["state_model"].get("variant", "full"),
        "best_selection_score": stopper.best,
        "epochs_run": len(history),
        "class_counts": class_counts,
        "history": history,
    }


def evaluate_probabilistic_state(
    config: dict[str, Any],
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    torch_device = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    num_classes = len(store["label_to_id"])
    _validate_store(store, config, num_classes)
    checkpoint_file = Path(checkpoint_path or config["train_state"]["checkpoint"])
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = _build_state_model(config, num_classes).to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    target_split = split or config.get("state_evaluation", {}).get("split", "test")
    loader = _loader(
        store["splits"][target_split], config, False, int(checkpoint.get("seed", 42)), torch_device
    )
    id_to_label = {int(index): label for label, index in store["label_to_id"].items()}
    class_weights, _ = _class_weights(
        store["splits"]["train"],
        num_classes,
        config["train_state"],
        torch_device,
    )
    metrics = _evaluate(
        model,
        loader,
        torch_device,
        id_to_label,
        bool(config["train_state"].get("use_amp", True)),
        config["train_state"],
        class_weights,
    )
    payload = {
        "checkpoint": str(checkpoint_file),
        "split": target_split,
        "seed": int(checkpoint.get("seed", 42)),
        "variant": checkpoint.get("variant", config["state_model"].get("variant")),
        "architecture": checkpoint.get(
            "architecture", config["state_model"].get("architecture", "recurrent")
        ),
        "metrics": metrics,
    }
    report_file = config.get("state_evaluation", {}).get("report_file")
    if report_file:
        save_json(payload, report_file)
    return payload


def train_probabilistic_state_seeds(
    config: dict[str, Any], device: str | None = None
) -> dict[str, Any]:
    seeds = [int(seed) for seed in config.get("state_evaluation", {}).get("seeds", [42, 43, 44])]
    base_checkpoint = Path(config["train_state"]["checkpoint"])
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_seed{seed}{base_checkpoint.suffix}"
        )
        training = train_probabilistic_state(
            config, device=device, seed_override=seed, checkpoint_override=checkpoint
        )
        run_config = copy.deepcopy(config)
        run_config["train_state"]["checkpoint"] = str(checkpoint)
        evaluation = evaluate_probabilistic_state(
            run_config, device=device, checkpoint_path=checkpoint
        )
        runs.append({"training": training, "evaluation": evaluation})

    metric_paths = {
        "overall_weighted_f1": ("weighted_f1",),
        "overall_macro_f1": ("macro_f1",),
        "stable_macro_f1": ("same_speaker_stable", "macro_f1"),
        "transition_macro_f1": ("same_speaker_transition", "macro_f1"),
        "state_nll": ("emotion_state_forecast", "nll"),
        "state_brier": ("emotion_state_forecast", "brier"),
        "state_ece": ("emotion_state_forecast", "ece"),
        "state_class_balanced_nll": (
            "emotion_state_forecast",
            "class_balanced_nll",
        ),
        "state_class_balanced_brier": (
            "emotion_state_forecast",
            "class_balanced_brier",
        ),
        "transition_average_precision": (
            "transition_detector",
            "average_precision",
        ),
        "fusion_gate": ("influence_gate", "overall_mean"),
        "state_update_gate": ("state_update_gate", "overall_mean"),
        "transition_matrix_diagonal": (
            "event_transition_matrix",
            "mean_diagonal_probability",
        ),
    }
    summary: dict[str, Any] = {}
    for name, path in metric_paths.items():
        values: list[float] = []
        for run in runs:
            value: Any = run["evaluation"]["metrics"]
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        summary[name] = {
            "values": values,
            "mean": mean(values) if values else None,
            "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
        }
    label_names = [
        label
        for label, _ in sorted(
            load_feature_store(config["data"]["feature_file"])["label_to_id"].items(),
            key=lambda item: int(item[1]),
        )
    ]
    per_emotion: dict[str, Any] = {}
    sections = {
        "overall": (),
        "stable": ("same_speaker_stable",),
        "transition": ("same_speaker_transition",),
    }
    for section_name, prefix in sections.items():
        per_emotion[section_name] = {}
        for label in label_names:
            values: list[float] = []
            for run in runs:
                metrics: Any = run["evaluation"]["metrics"]
                for key in prefix:
                    metrics = metrics.get(key, {})
                report = metrics.get("classification_report", {})
                if label in report:
                    values.append(float(report[label]["f1-score"]))
            per_emotion[section_name][label] = {
                "values": values,
                "mean": mean(values) if values else None,
                "std": pstdev(values) if len(values) > 1 else (0.0 if values else None),
            }
    payload = {
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
        "per_emotion_f1": per_emotion,
    }
    report = config.get("state_evaluation", {}).get("report_file")
    if report:
        report_path = Path(report)
        save_json(payload, report_path.with_name(f"{report_path.stem}_seeds{report_path.suffix}"))
    return payload
