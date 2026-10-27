from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import HiTFTForecaster
from .common import (
    EarlyStopping,
    ForecastWindowDataset,
    collate_forecast,
    get_device,
    load_feature_store,
    make_grad_scaler,
    move_to_device,
    save_json,
    seed_everything,
)


def _forecast_loss(
    prediction: torch.Tensor, target: torch.Tensor, target_mask: torch.Tensor
) -> torch.Tensor:
    squared_error = (prediction - target).square()
    expanded_mask = target_mask.unsqueeze(-1).to(squared_error.dtype)
    return (squared_error * expanded_mask).sum() / (
        expanded_mask.sum().clamp_min(1.0) * squared_error.shape[-1]
    )


def _zero_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_squared_sum(
    candidate: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> tuple[float, int]:
    squared = (candidate - target).square()
    numerator = float((squared * selected.unsqueeze(-1).to(squared.dtype)).sum().cpu())
    denominator = int(selected.sum()) * squared.shape[-1]
    return numerator, denominator


def _transition_aware_loss(
    prediction: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_config: dict[str, Any],
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = batch["target"]
    target_mask = batch["target_mask"]
    known = target_mask & batch["transition_mask"]
    transition_target = batch["transition_targets"]
    stable_mask = known & transition_target.lt(0.5)
    transition_mask = known & transition_target.ge(0.5)

    main_mse = _forecast_loss(prediction, target, target_mask)
    stable_mse = (
        _forecast_loss(auxiliary["stable_prediction"], target, stable_mask)
        if stable_mask.any()
        else _zero_loss(prediction)
    )
    transition_mse = (
        _forecast_loss(auxiliary["transition_prediction"], target, transition_mask)
        if transition_mask.any()
        else _zero_loss(prediction)
    )
    base_mse = (
        _forecast_loss(auxiliary["base_prediction"], target, target_mask)
        if "base_prediction" in auxiliary
        else _zero_loss(prediction)
    )

    if known.any():
        logits = auxiliary["transition_logits"]
        targets = transition_target.to(logits.dtype)
        binary = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probability = torch.sigmoid(logits)
        probability_true = torch.where(targets.gt(0.5), probability, 1.0 - probability)
        gamma = float(loss_config.get("focal_gamma", 2.0))
        alpha = float(loss_config.get("focal_alpha", 0.5))
        alpha_true = torch.where(
            targets.gt(0.5),
            logits.new_full((), alpha),
            logits.new_full((), 1.0 - alpha),
        )
        focal = alpha_true * (1.0 - probability_true).pow(gamma) * binary
        transition_classification = (focal * known.to(focal.dtype)).sum() / known.sum().clamp_min(1)
    else:
        transition_classification = _zero_loss(prediction)

    if transition_mask.any():
        direction_reference = auxiliary.get("direction_reference")
        if direction_reference is None:
            direction_reference = auxiliary["anchor"].unsqueeze(1)
        target_delta = target - direction_reference
        direction = 1.0 - nn.functional.cosine_similarity(
            auxiliary["transition_delta"], target_delta, dim=-1, eps=1e-8
        )
        direction_loss = (
            direction * transition_mask.to(direction.dtype)
        ).sum() / transition_mask.sum().clamp_min(1)
    else:
        direction_loss = _zero_loss(prediction)

    stable_error = (auxiliary["stable_prediction"] - target).square().mean(dim=-1)
    transition_error = (auxiliary["transition_prediction"] - target).square().mean(dim=-1)
    margin = float(loss_config.get("specialization_margin", 0.002))
    specialization_terms: list[torch.Tensor] = []
    if stable_mask.any():
        specialization_terms.append(
            nn.functional.relu(stable_error - transition_error + margin)[stable_mask].mean()
        )
    if transition_mask.any():
        specialization_terms.append(
            nn.functional.relu(transition_error - stable_error + margin)[transition_mask].mean()
        )
    specialization_loss = (
        torch.stack(specialization_terms).mean()
        if specialization_terms
        else _zero_loss(prediction)
    )

    total = (
        float(loss_config.get("main_mse_weight", 1.0)) * main_mse
        + float(loss_config.get("base_mse_weight", 0.0)) * base_mse
        + float(loss_config.get("stable_mse_weight", 0.5)) * stable_mse
        + float(loss_config.get("transition_mse_weight", 1.0)) * transition_mse
        + float(loss_config.get("classification_weight", 0.1)) * transition_classification
        + float(loss_config.get("direction_weight", 0.05)) * direction_loss
        + float(loss_config.get("specialization_weight", 0.1)) * specialization_loss
    )
    if not return_components:
        return total
    return total, {
        "main_mse": main_mse,
        "base_mse": base_mse,
        "stable_mse": stable_mse,
        "transition_mse": transition_mse,
        "classification": transition_classification,
        "direction": direction_loss,
        "specialization": specialization_loss,
    }


def _run_epoch(
    model: HiTFTForecaster,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    grad_clip: float,
    transition_loss_config: dict[str, Any] | None = None,
) -> tuple[float, dict[str, float]]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    component_totals: dict[str, float] = {}
    for batch in tqdm(loader, desc="HiTFT train" if training else "HiTFT valid", leave=False):
        batch = move_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                transition_aware = model.is_transition_forecaster
                if transition_aware:
                    prediction, auxiliary = model(
                        batch["history"],
                        batch["speaker_ids"],
                        batch["history_mask"],
                        return_aux=True,
                    )
                    loss, components = _transition_aware_loss(
                        prediction,
                        auxiliary,
                        batch,
                        transition_loss_config or {},
                        return_components=True,
                    )
                else:
                    prediction = model(
                        batch["history"], batch["speaker_ids"], batch["history_mask"]
                    )
                    loss = _forecast_loss(prediction, batch["target"], batch["target_mask"])
                    components = {"forecast_mse": loss}
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
        batch_size = batch["history"].shape[0]
        total_loss += float(loss.detach()) * batch_size
        for name, value in components.items():
            component_totals[name] = component_totals.get(name, 0.0) + (
                float(value.detach()) * batch_size
            )
        total_samples += batch_size
    denominator = max(total_samples, 1)
    return total_loss / denominator, {
        name: value / denominator for name, value in component_totals.items()
    }


def _load_warm_start(
    model: HiTFTForecaster,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"HiTFT warm-start checkpoint not found: {path}. "
            "Use the original absolute-forecast HiTFT checkpoint."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=False)
    allowed_missing_prefixes = (
        "stable_residual_head.",
        "transition_residual_head.",
        "transition_logit_head.",
    )
    unexpected_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Warm-start checkpoint is architecture-incompatible. "
            f"Unexpected missing keys: {unexpected_missing}; "
            f"unexpected checkpoint keys: {incompatible.unexpected_keys}"
        )
    return {
        "checkpoint": str(path),
        "source_epoch": checkpoint.get("epoch"),
        "source_validation_mse": checkpoint.get("validation_mse"),
        "initialized_new_keys": list(incompatible.missing_keys),
    }


def train_hitft(config: dict[str, Any], device: str | None = None) -> dict[str, Any]:
    seed = int(config["experiment"].get("seed", 42))
    seed_everything(seed, bool(config["experiment"].get("deterministic", True)))
    device_obj = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    history_steps = int(config["model"].get("history_steps", 8))
    future_steps = int(config["model"].get("future_steps", 4))
    train_data = ForecastWindowDataset(store["splits"]["train"], history_steps, future_steps)
    validation_split = store["splits"].get("validation", store["splits"]["train"])
    validation_data = ForecastWindowDataset(validation_split, history_steps, future_steps)
    if not train_data:
        raise ValueError("No forecast windows found; training dialogues need at least two turns")

    train_config = config["train_hitft"]
    transition_loss_config = train_config.get("transition_loss", {})
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = {
        "batch_size": int(train_config.get("batch_size", 128)),
        "num_workers": int(config["experiment"].get("num_workers", 0)),
        "collate_fn": collate_forecast,
        "pin_memory": device_obj.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_kwargs)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_kwargs)

    model = HiTFTForecaster(config)
    warm_start: dict[str, Any] | None = None
    warm_start_checkpoint = train_config.get("warm_start_checkpoint")
    if warm_start_checkpoint:
        warm_start = _load_warm_start(model, warm_start_checkpoint)
        print(
            "[HiTFT] warm-started backbone/base forecast from "
            f"{warm_start['checkpoint']}"
        )
    model = model.to(device_obj)

    freeze_backbone_epochs = int(train_config.get("freeze_backbone_epochs", 0))
    if freeze_backbone_epochs > 0:
        model.set_transition_backbone_trainable(False)
        print(
            f"[HiTFT] freezing warm-started backbone for {freeze_backbone_epochs} epochs"
        )

    learning_rate = float(train_config.get("learning_rate", 1e-4))
    weight_decay = float(train_config.get("weight_decay", 1e-3))
    if model.forecaster_mode == "transition_residual":
        head_parameters = model.transition_head_parameters()
        head_ids = {id(parameter) for parameter in head_parameters}
        backbone_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in head_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_parameters,
                    "lr": float(train_config.get("backbone_learning_rate", 1e-5)),
                },
                {"params": head_parameters, "lr": learning_rate},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    use_amp = bool(train_config.get("use_amp", True)) and device_obj.type == "cuda"
    scaler = make_grad_scaler(device_obj, use_amp)
    stopper = EarlyStopping(int(train_config.get("patience", 30)), mode="min")
    checkpoint_path = Path(train_config["checkpoint"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(train_config.get("epochs", 400)) + 1):
        if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
            model.set_transition_backbone_trainable(True)
            print("[HiTFT] unfroze backbone for low-learning-rate joint fine-tuning")

        train_loss, train_components = _run_epoch(
            model,
            train_loader,
            device_obj,
            optimizer,
            scaler,
            use_amp,
            float(train_config.get("grad_clip", 1.0)),
            transition_loss_config,
        )
        validation_loss, validation_components = _run_epoch(
            model,
            validation_loader,
            device_obj,
            None,
            scaler,
            use_amp,
            float(train_config.get("grad_clip", 1.0)),
            transition_loss_config,
        )
        transition_aware = model.is_transition_forecaster
        train_key = "train_loss" if transition_aware else "train_mse"
        validation_key = "validation_loss" if transition_aware else "validation_mse"
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            train_key: train_loss,
            validation_key: validation_loss,
        }
        epoch_record.update(
            {f"train_{name}": value for name, value in train_components.items()}
        )
        epoch_record.update(
            {
                f"validation_{name}": value
                for name, value in validation_components.items()
            }
        )
        history.append(epoch_record)
        improved, should_stop = stopper.update(validation_loss)
        prefix = (
            "Residual-Transition-HiTFT"
            if model.forecaster_mode == "transition_residual"
            else "Transition-HiTFT"
            if transition_aware
            else "HiTFT"
        )
        message = (
            f"[{prefix}] epoch={epoch:03d} train={train_loss:.6f} "
            f"validation={validation_loss:.6f}"
        )
        if transition_aware:
            message += (
                f" val_base={validation_components.get('base_mse', 0.0):.6f}"
                f" val_cls={validation_components.get('classification', 0.0):.6f}"
                f" val_direction={validation_components.get('direction', 0.0):.6f}"
            )
        print(message)
        if improved:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    validation_key: validation_loss,
                    "config": config,
                    "embedding_dim": store["embedding_dim"],
                    "forecaster_mode": model.forecaster_mode,
                    "warm_start": warm_start,
                    "validation_components": validation_components,
                },
                checkpoint_path,
            )
        if should_stop:
            break
    result = {
        "checkpoint": str(checkpoint_path),
        "epochs_run": len(history),
        "history": history,
        "warm_start": warm_start,
    }
    best_key = (
        "best_validation_loss"
        if model.is_transition_forecaster
        else "best_validation_mse"
    )
    result[best_key] = stopper.best
    return result


def evaluate_hitft(
    config: dict[str, Any], device: str | None = None, split: str | None = None
) -> dict[str, Any]:
    device_obj = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    split = split or config.get("transition_evaluation", {}).get("split", "test")
    history_steps = int(config["model"].get("history_steps", 8))
    future_steps = int(config["model"].get("future_steps", 4))
    dataset = ForecastWindowDataset(store["splits"][split], history_steps, future_steps)
    if not dataset:
        raise ValueError(f"No forecast windows found in split {split!r}")
    loader = DataLoader(
        dataset,
        batch_size=int(config["train_hitft"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(config["experiment"].get("num_workers", 0)),
        collate_fn=collate_forecast,
        pin_memory=device_obj.type == "cuda",
    )

    model = HiTFTForecaster(config).to(device_obj)
    if not model.is_transition_forecaster:
        raise ValueError("evaluate-hitft requires a transition forecaster mode")
    checkpoint_path = Path(config["train_hitft"]["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    overall_numerator = 0.0
    overall_denominator = 0
    base_numerator = 0.0
    base_denominator = 0
    stable_numerator = 0.0
    stable_denominator = 0
    transition_numerator = 0.0
    transition_denominator = 0
    stable_on_transition_numerator = 0.0
    stable_on_transition_denominator = 0
    transition_on_stable_numerator = 0.0
    transition_on_stable_denominator = 0
    direction_sum = 0.0
    direction_count = 0
    stable_residual_square_sum = 0.0
    transition_residual_square_sum = 0.0
    residual_denominator = 0
    labels: list[int] = []
    probabilities: list[float] = []

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Transition-HiTFT evaluate", leave=False):
            batch = move_to_device(batch, device_obj)
            prediction, auxiliary = model(
                batch["history"],
                batch["speaker_ids"],
                batch["history_mask"],
                return_aux=True,
            )
            target = batch["target"]
            target_mask = batch["target_mask"]
            known = target_mask & batch["transition_mask"]
            transition_target = batch["transition_targets"]
            stable_mask = known & transition_target.lt(0.5)
            transition_mask = known & transition_target.ge(0.5)

            numerator, denominator = _masked_squared_sum(prediction, target, target_mask)
            overall_numerator += numerator
            overall_denominator += denominator
            if "base_prediction" in auxiliary:
                numerator, denominator = _masked_squared_sum(
                    auxiliary["base_prediction"], target, target_mask
                )
                base_numerator += numerator
                base_denominator += denominator
            numerator, denominator = _masked_squared_sum(
                auxiliary["stable_prediction"], target, stable_mask
            )
            stable_numerator += numerator
            stable_denominator += denominator
            numerator, denominator = _masked_squared_sum(
                auxiliary["transition_prediction"], target, transition_mask
            )
            transition_numerator += numerator
            transition_denominator += denominator
            numerator, denominator = _masked_squared_sum(
                auxiliary["stable_prediction"], target, transition_mask
            )
            stable_on_transition_numerator += numerator
            stable_on_transition_denominator += denominator
            numerator, denominator = _masked_squared_sum(
                auxiliary["transition_prediction"], target, stable_mask
            )
            transition_on_stable_numerator += numerator
            transition_on_stable_denominator += denominator

            if transition_mask.any():
                direction_reference = auxiliary.get("direction_reference")
                if direction_reference is None:
                    direction_reference = auxiliary["anchor"].unsqueeze(1)
                target_delta = target - direction_reference
                cosine = nn.functional.cosine_similarity(
                    auxiliary["transition_delta"], target_delta, dim=-1, eps=1e-8
                )
                direction_sum += float((cosine * transition_mask).sum().cpu())
                direction_count += int(transition_mask.sum())

            if "stable_residual" in auxiliary:
                expanded_mask = target_mask.unsqueeze(-1).to(target.dtype)
                stable_residual_square_sum += float(
                    (auxiliary["stable_residual"].square() * expanded_mask).sum().cpu()
                )
                transition_residual_square_sum += float(
                    (
                        auxiliary["transition_residual"].square()
                        * expanded_mask
                    ).sum().cpu()
                )
                residual_denominator += int(target_mask.sum()) * target.shape[-1]

            labels.extend(transition_target[known].int().cpu().tolist())
            probabilities.extend(auxiliary["transition_probability"][known].cpu().tolist())

    if not labels:
        raise ValueError(f"No known same-speaker transition labels found in split {split!r}")
    threshold = float(config.get("transition_evaluation", {}).get("threshold", 0.5))
    predictions = [int(value >= threshold) for value in probabilities]
    average_precision = (
        float(average_precision_score(labels, probabilities)) if len(set(labels)) > 1 else None
    )
    probability_tensor = torch.tensor(probabilities)
    probability_quantiles = probability_tensor.quantile(
        torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    ).tolist()
    result = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "metrics": {
            "forecast_mse": overall_numerator / max(overall_denominator, 1),
            "base_forecast_mse": (
                base_numerator / max(base_denominator, 1)
                if base_denominator
                else None
            ),
            "stable_candidate_mse": stable_numerator / max(stable_denominator, 1),
            "transition_candidate_mse": transition_numerator
            / max(transition_denominator, 1),
            "stable_candidate_on_transition_mse": stable_on_transition_numerator
            / max(stable_on_transition_denominator, 1),
            "transition_candidate_on_stable_mse": transition_on_stable_numerator
            / max(transition_on_stable_denominator, 1),
            "transition_average_precision": average_precision,
            "transition_f1": float(f1_score(labels, predictions, zero_division=0)),
            "transition_precision": float(
                precision_score(labels, predictions, zero_division=0)
            ),
            "transition_recall": float(recall_score(labels, predictions, zero_division=0)),
            "transition_rate": sum(labels) / len(labels),
            "predicted_transition_rate": sum(predictions) / len(predictions),
            "transition_probability_mean": float(probability_tensor.mean()),
            "transition_probability_std": float(probability_tensor.std(unbiased=False)),
            "transition_probability_quantiles": probability_quantiles,
            "transition_delta_cosine": direction_sum / max(direction_count, 1),
            "stable_residual_rms": (
                (stable_residual_square_sum / residual_denominator) ** 0.5
                if residual_denominator
                else None
            ),
            "transition_residual_rms": (
                (transition_residual_square_sum / residual_denominator) ** 0.5
                if residual_denominator
                else None
            ),
            "known_targets": len(labels),
            "stable_targets": len(labels) - sum(labels),
            "transition_targets": sum(labels),
        },
    }
    report_file = config.get("transition_evaluation", {}).get("report_file")
    if report_file:
        save_json(result, report_file)
    return result
