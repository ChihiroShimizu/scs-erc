from __future__ import annotations

import copy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from ..models import BoundaryAwareResidualReranker, ProbabilisticEmotionStateTracker
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
from .state_trainer import (
    _class_weights,
    _subset_metrics,
    _validate_store,
    _weighted_mean,
    same_speaker_transition_masks,
)


def _boundary_loader(
    dialogues: list[dict[str, Any]],
    config: dict[str, Any],
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        DialogueFeatureDataset(dialogues),
        batch_size=int(config["boundary_train"].get("batch_size", 16)),
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=int(config["experiment"].get("num_workers", 0)),
        collate_fn=collate_dialogues,
        pin_memory=device.type == "cuda",
    )


def _load_context_tracker(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> ProbabilisticEmotionStateTracker:
    checkpoint_file = Path(config["boundary_model"]["state_checkpoint"])
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    tracker = ProbabilisticEmotionStateTracker(config, num_classes)
    tracker.load_state_dict(checkpoint["model_state"])
    tracker.requires_grad_(False)
    tracker.eval()
    return tracker.to(device)


def _build_boundary_model(
    config: dict[str, Any], num_classes: int, device: torch.device
) -> BoundaryAwareResidualReranker:
    tracker = _load_context_tracker(config, num_classes, device)
    return BoundaryAwareResidualReranker(config, num_classes, tracker).to(device)


def _sample_weights(
    labels: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
) -> torch.Tensor | None:
    if criterion.weight is None:
        return None
    return criterion.weight[labels].float()


def _boundary_loss(
    logits: torch.Tensor,
    aux: dict[str, torch.Tensor],
    batch: dict[str, Any],
    criterion: nn.CrossEntropyLoss,
    train_cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["mask"] & batch["labels"].ge(0)
    labels = batch["labels"]
    classification = criterion(logits[valid], labels[valid])
    anchor = aux["anchor_logits"]
    anchor_prediction = anchor.argmax(dim=-1)
    anchor_correct = anchor_prediction.eq(labels)

    gold_logits = logits.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    wrong_logits = logits.masked_fill(
        nn.functional.one_hot(
            labels.clamp_min(0), num_classes=logits.shape[-1]
        ).bool(),
        torch.finfo(logits.dtype).min,
    ).max(dim=-1).values
    margin_examples = valid
    if bool(train_cfg.get("margin_on_anchor_wrong_only", True)):
        margin_examples = margin_examples & ~anchor_correct
    if margin_examples.any():
        margin_values = nn.functional.relu(
            float(train_cfg.get("margin_target", 0.5))
            + wrong_logits[margin_examples]
            - gold_logits[margin_examples]
        ).float()
        margin_weights = _sample_weights(labels[margin_examples], criterion)
        margin_loss = _weighted_mean(margin_values, margin_weights)
    else:
        margin_loss = classification.new_zeros(())

    keep_threshold = float(train_cfg.get("keep_margin_threshold", 1.0))
    keep_examples = valid & anchor_correct & aux["anchor_margin"].ge(keep_threshold)
    if keep_examples.any():
        anchor_probability = torch.softmax(anchor[keep_examples].float(), dim=-1)
        fused_log_probability = torch.log_softmax(logits[keep_examples].float(), dim=-1)
        keep_values = nn.functional.kl_div(
            fused_log_probability,
            anchor_probability,
            reduction="none",
        ).sum(dim=-1)
        keep_weights = _sample_weights(labels[keep_examples], criterion)
        keep_loss = _weighted_mean(keep_values, keep_weights)
    else:
        keep_loss = classification.new_zeros(())

    residual_values = aux["boundary_residual"][valid].float().pow(2).mean(dim=-1)
    residual_l2 = (
        residual_values.mean()
        if residual_values.numel()
        else classification.new_zeros(())
    )
    total = (
        classification
        + float(train_cfg.get("margin_loss_weight", 0.0)) * margin_loss
        + float(train_cfg.get("keep_loss_weight", 0.0)) * keep_loss
        + float(train_cfg.get("residual_l2_weight", 0.001)) * residual_l2
    )
    return total, {
        "classification": float(classification.detach()),
        "margin": float(margin_loss.detach()),
        "keep": float(keep_loss.detach()),
        "residual_l2": float(residual_l2.detach()),
    }


def _evaluate_boundary(
    model: BoundaryAwareResidualReranker,
    loader: DataLoader,
    device: torch.device,
    id_to_label: dict[int, str],
    use_amp: bool,
    train_cfg: dict[str, Any],
    class_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    labels_all: list[int] = []
    fused_all: list[int] = []
    anchor_all: list[int] = []
    context_all: list[int] = []
    context_oracle_all: list[int] = []
    topk_oracle_all: list[int] = []
    stable_labels: list[int] = []
    stable_predictions: list[int] = []
    transition_labels: list[int] = []
    transition_predictions: list[int] = []
    gates: list[float] = []
    stable_gates: list[float] = []
    transition_gates: list[float] = []
    margins: list[float] = []
    topk_hits: list[int] = []
    context_nll_help: list[int] = []
    context_direction_help: list[int] = []
    losses: list[float] = []
    margin_groups: dict[str, tuple[list[int], list[int]]] = {
        "small_lt_0_5": ([], []),
        "medium_0_5_to_1_5": ([], []),
        "large_ge_1_5": ([], []),
    }

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
                loss, _ = _boundary_loss(logits, aux, batch, criterion, train_cfg)
            losses.append(float(loss))
            valid = batch["mask"] & batch["labels"].ge(0)
            labels = batch["labels"]
            known, transition = same_speaker_transition_masks(
                batch["speaker_ids"], labels, valid
            )
            stable = known & ~transition
            fused_prediction = logits.argmax(dim=-1)
            anchor_prediction = aux["anchor_logits"].argmax(dim=-1)
            context_prediction = aux["context_candidate_logits"].argmax(dim=-1)
            gate = aux["boundary_gate"].squeeze(-1)
            margin = aux["anchor_margin"]

            context_oracle = anchor_prediction.clone()
            context_can_fix = ~anchor_prediction.eq(labels) & context_prediction.eq(labels)
            context_oracle[context_can_fix] = context_prediction[context_can_fix]
            topk = aux["anchor_logits"].topk(model.top_k, dim=-1).indices
            topk_hit = topk.eq(labels.clamp_min(0).unsqueeze(-1)).any(dim=-1) & valid
            topk_oracle = anchor_prediction.clone()
            topk_oracle[topk_hit] = labels[topk_hit]

            anchor_probability = torch.softmax(aux["anchor_logits"].float(), dim=-1)
            context_probability = torch.softmax(
                aux["context_candidate_logits"].float(), dim=-1
            )
            gold_index = labels.clamp_min(0).unsqueeze(-1)
            anchor_gold = anchor_probability.gather(-1, gold_index).squeeze(-1)
            context_gold = context_probability.gather(-1, gold_index).squeeze(-1)
            context_delta = aux["context_candidate_logits"] - aux["anchor_logits"]
            gold_delta = context_delta.gather(-1, gold_index).squeeze(-1)
            anchor_class_delta = context_delta.gather(
                -1, anchor_prediction.unsqueeze(-1)
            ).squeeze(-1)

            labels_all.extend(labels[valid].cpu().tolist())
            fused_all.extend(fused_prediction[valid].cpu().tolist())
            anchor_all.extend(anchor_prediction[valid].cpu().tolist())
            context_all.extend(context_prediction[valid].cpu().tolist())
            context_oracle_all.extend(context_oracle[valid].cpu().tolist())
            topk_oracle_all.extend(topk_oracle[valid].cpu().tolist())
            stable_labels.extend(labels[stable].cpu().tolist())
            stable_predictions.extend(fused_prediction[stable].cpu().tolist())
            transition_labels.extend(labels[transition].cpu().tolist())
            transition_predictions.extend(fused_prediction[transition].cpu().tolist())
            gates.extend(gate[valid].float().cpu().tolist())
            stable_gates.extend(gate[stable].float().cpu().tolist())
            transition_gates.extend(gate[transition].float().cpu().tolist())
            margins.extend(margin[valid].float().cpu().tolist())
            topk_hits.extend(topk_hit[valid].long().cpu().tolist())
            context_nll_help.extend(
                context_gold[valid].gt(anchor_gold[valid]).long().cpu().tolist()
            )
            context_direction_help.extend(
                gold_delta[valid].gt(anchor_class_delta[valid]).long().cpu().tolist()
            )

            group_masks = {
                "small_lt_0_5": valid & margin.lt(0.5),
                "medium_0_5_to_1_5": valid & margin.ge(0.5) & margin.lt(1.5),
                "large_ge_1_5": valid & margin.ge(1.5),
            }
            for name, selected in group_masks.items():
                margin_groups[name][0].extend(labels[selected].cpu().tolist())
                margin_groups[name][1].extend(fused_prediction[selected].cpu().tolist())

    result = classification_metrics(labels_all, fused_all, id_to_label)
    result["utterances"] = len(labels_all)
    result["loss"] = mean(losses) if losses else None
    result["anchor"] = classification_metrics(labels_all, anchor_all, id_to_label)
    result["context_candidate"] = classification_metrics(
        labels_all, context_all, id_to_label
    )
    result["oracle_anchor_or_context"] = classification_metrics(
        labels_all, context_oracle_all, id_to_label
    )
    result["oracle_anchor_topk"] = classification_metrics(
        labels_all, topk_oracle_all, id_to_label
    )
    result["same_speaker_stable"] = _subset_metrics(
        stable_labels, stable_predictions, id_to_label
    )
    result["same_speaker_transition"] = _subset_metrics(
        transition_labels, transition_predictions, id_to_label
    )
    beneficial = sum(
        anchor != gold and fused == gold
        for gold, anchor, fused in zip(labels_all, anchor_all, fused_all)
    )
    harmful = sum(
        anchor == gold and fused != gold
        for gold, anchor, fused in zip(labels_all, anchor_all, fused_all)
    )
    anchor_errors = sum(anchor != gold for gold, anchor in zip(labels_all, anchor_all))
    anchor_correct = len(labels_all) - anchor_errors
    result["fusion_flips"] = {
        "beneficial": beneficial,
        "harmful": harmful,
        "net": beneficial - harmful,
        "anchor_error_correction_rate": beneficial / max(anchor_errors, 1),
        "anchor_correct_harm_rate": harmful / max(anchor_correct, 1),
    }
    result["boundary_opportunity"] = {
        "topk": model.top_k,
        "anchor_topk_gold_recall": mean(topk_hits) if topk_hits else None,
        "context_nll_help_rate": mean(context_nll_help) if context_nll_help else None,
        "context_direction_help_rate": (
            mean(context_direction_help) if context_direction_help else None
        ),
        "anchor_margin_mean": mean(margins) if margins else None,
    }
    result["boundary_gate"] = {
        "overall_mean": mean(gates) if gates else None,
        "stable_mean": mean(stable_gates) if stable_gates else None,
        "transition_mean": mean(transition_gates) if transition_gates else None,
    }
    result["margin_groups"] = {
        name: _subset_metrics(group_labels, group_predictions, id_to_label)
        for name, (group_labels, group_predictions) in margin_groups.items()
    }
    return result


def _selection_score(metrics: dict[str, Any], train_cfg: dict[str, Any]) -> float:
    score = float(metrics["macro_f1"])
    harmful_rate = float(metrics["fusion_flips"]["anchor_correct_harm_rate"])
    score -= float(train_cfg.get("selection_harmful_flip_penalty", 0.2)) * harmful_rate
    tolerance = float(train_cfg.get("selection_class_tolerance", 0.01))
    regressions: list[float] = []
    fused_report = metrics["classification_report"]
    anchor_report = metrics["anchor"]["classification_report"]
    for label, anchor_values in anchor_report.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        regression = float(anchor_values["f1-score"]) - float(
            fused_report[label]["f1-score"]
        )
        regressions.append(max(0.0, regression - tolerance))
    if regressions:
        score -= float(train_cfg.get("selection_class_regression_penalty", 0.5)) * mean(
            regressions
        )
    return score


def analyze_boundary_oracle(
    config: dict[str, Any], device: str | None = None, split: str | None = None
) -> dict[str, Any]:
    torch_device = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    num_classes = len(store["label_to_id"])
    _validate_store(store, config, num_classes)
    model = _build_boundary_model(config, num_classes, torch_device)
    target_split = split or config.get("boundary_oracle", {}).get(
        "split", "validation"
    )
    loader = _boundary_loader(
        store["splits"][target_split], config, False, 42, torch_device
    )
    id_to_label = {int(index): label for label, index in store["label_to_id"].items()}
    class_weights, _ = _class_weights(
        store["splits"]["train"],
        num_classes,
        config["boundary_train"],
        torch_device,
    )
    metrics = _evaluate_boundary(
        model,
        loader,
        torch_device,
        id_to_label,
        bool(config["boundary_train"].get("use_amp", True)),
        config["boundary_train"],
        class_weights,
    )
    payload = {
        "split": target_split,
        "state_checkpoint": config["boundary_model"]["state_checkpoint"],
        "metrics": metrics,
    }
    report = config.get("boundary_oracle", {}).get("report_file")
    if report:
        save_json(payload, report)
    return payload


def train_boundary_reranker(
    config: dict[str, Any],
    device: str | None = None,
    seed_override: int | None = None,
    checkpoint_override: str | Path | None = None,
) -> dict[str, Any]:
    train_cfg = config["boundary_train"]
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
    model = _build_boundary_model(config, num_classes, torch_device)
    train_loader = _boundary_loader(
        store["splits"]["train"], config, True, seed, torch_device
    )
    validation_loader = _boundary_loader(
        store["splits"]["validation"], config, False, seed, torch_device
    )
    class_weights, class_counts = _class_weights(
        store["splits"]["train"], num_classes, train_cfg, torch_device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-3)),
    )
    epochs = int(train_cfg.get("epochs", 20))
    total_steps = max(epochs * len(train_loader), 1)
    warmup_steps = int(float(train_cfg.get("warmup_ratio", 0.1)) * total_steps)
    scheduler = LambdaLR(
        optimizer,
        lambda step: linear_warmup_decay_lambda(step, warmup_steps, total_steps),
    )
    use_amp = bool(train_cfg.get("use_amp", True)) and torch_device.type == "cuda"
    scaler = make_grad_scaler(torch_device, use_amp)
    stopper = EarlyStopping(int(train_cfg.get("patience", 5)), mode="max")
    checkpoint_file = Path(checkpoint_override or train_cfg["checkpoint"])
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
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
                loss, _ = _boundary_loss(logits, aux, batch, criterion, train_cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, float(train_cfg.get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running.append(float(loss.detach()))

        validation = _evaluate_boundary(
            model,
            validation_loader,
            torch_device,
            id_to_label,
            use_amp,
            train_cfg,
            class_weights,
        )
        score = _selection_score(validation, train_cfg)
        record = {
            "epoch": epoch,
            "train_loss": mean(running),
            "validation_macro_f1": validation["macro_f1"],
            "validation_weighted_f1": validation["weighted_f1"],
            "beneficial_flips": validation["fusion_flips"]["beneficial"],
            "harmful_flips": validation["fusion_flips"]["harmful"],
            "selection_score": score,
        }
        history.append(record)
        print(
            f"[Boundary] seed={seed} epoch={epoch:03d} "
            f"loss={record['train_loss']:.5f} "
            f"macro={record['validation_macro_f1']:.4f} "
            f"flips={record['beneficial_flips']}/{record['harmful_flips']} "
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
                    "variant": config["boundary_model"]["variant"],
                    "label_to_id": store["label_to_id"],
                    "class_counts": class_counts,
                },
                checkpoint_file,
            )
        if stop:
            break
    return {
        "checkpoint": str(checkpoint_file),
        "seed": seed,
        "variant": config["boundary_model"]["variant"],
        "best_selection_score": stopper.best,
        "epochs_run": len(history),
        "history": history,
    }


def evaluate_boundary_reranker(
    config: dict[str, Any],
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    torch_device = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    num_classes = len(store["label_to_id"])
    _validate_store(store, config, num_classes)
    checkpoint_file = Path(checkpoint_path or config["boundary_train"]["checkpoint"])
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model = _build_boundary_model(config, num_classes, torch_device)
    model.load_state_dict(checkpoint["model_state"])
    target_split = split or config.get("boundary_evaluation", {}).get(
        "split", "validation"
    )
    loader = _boundary_loader(
        store["splits"][target_split],
        config,
        False,
        int(checkpoint.get("seed", 42)),
        torch_device,
    )
    class_weights, _ = _class_weights(
        store["splits"]["train"],
        num_classes,
        config["boundary_train"],
        torch_device,
    )
    id_to_label = {int(index): label for label, index in store["label_to_id"].items()}
    metrics = _evaluate_boundary(
        model,
        loader,
        torch_device,
        id_to_label,
        bool(config["boundary_train"].get("use_amp", True)),
        config["boundary_train"],
        class_weights,
    )
    payload = {
        "checkpoint": str(checkpoint_file),
        "split": target_split,
        "seed": int(checkpoint.get("seed", 42)),
        "variant": checkpoint.get("variant", config["boundary_model"]["variant"]),
        "metrics": metrics,
    }
    report = config.get("boundary_evaluation", {}).get("report_file")
    if report:
        save_json(payload, report)
    return payload


def _summarize_runs(
    config: dict[str, Any], seeds: list[int], runs: list[dict[str, Any]]
) -> dict[str, Any]:
    metric_paths = {
        "overall_weighted_f1": ("weighted_f1",),
        "overall_macro_f1": ("macro_f1",),
        "stable_macro_f1": ("same_speaker_stable", "macro_f1"),
        "transition_macro_f1": ("same_speaker_transition", "macro_f1"),
        "beneficial_flips": ("fusion_flips", "beneficial"),
        "harmful_flips": ("fusion_flips", "harmful"),
        "net_flips": ("fusion_flips", "net"),
        "gate_mean": ("boundary_gate", "overall_mean"),
        "topk_gold_recall": ("boundary_opportunity", "anchor_topk_gold_recall"),
        "context_nll_help_rate": ("boundary_opportunity", "context_nll_help_rate"),
        "oracle_context_macro_f1": ("oracle_anchor_or_context", "macro_f1"),
        "oracle_topk_macro_f1": ("oracle_anchor_topk", "macro_f1"),
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
    selection_values = [
        float(run["training"]["best_selection_score"]) for run in runs
    ]
    summary["best_validation_selection_score"] = {
        "values": selection_values,
        "mean": mean(selection_values),
        "std": pstdev(selection_values) if len(selection_values) > 1 else 0.0,
    }

    store = load_feature_store(config["data"]["feature_file"])
    labels = [
        label
        for label, _ in sorted(store["label_to_id"].items(), key=lambda item: item[1])
    ]
    per_emotion: dict[str, Any] = {}
    for label in labels:
        values = [
            float(
                run["evaluation"]["metrics"]["classification_report"][label][
                    "f1-score"
                ]
            )
            for run in runs
        ]
        per_emotion[label] = {
            "values": values,
            "mean": mean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
        }
    return {
        "seeds": seeds,
        "runs": runs,
        "summary": summary,
        "per_emotion_f1": per_emotion,
    }


def train_boundary_reranker_seeds(
    config: dict[str, Any], device: str | None = None
) -> dict[str, Any]:
    seeds = [
        int(seed)
        for seed in config.get("boundary_evaluation", {}).get(
            "seeds", [42, 43, 44]
        )
    ]
    base_checkpoint = Path(config["boundary_train"]["checkpoint"])
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_seed{seed}{base_checkpoint.suffix}"
        )
        training = train_boundary_reranker(
            config,
            device=device,
            seed_override=seed,
            checkpoint_override=checkpoint,
        )
        run_config = copy.deepcopy(config)
        run_config["boundary_train"]["checkpoint"] = str(checkpoint)
        evaluation = evaluate_boundary_reranker(
            run_config, device=device, checkpoint_path=checkpoint
        )
        runs.append({"training": training, "evaluation": evaluation})

    payload = _summarize_runs(config, seeds, runs)
    report = config.get("boundary_evaluation", {}).get("report_file")
    if report:
        report_path = Path(report)
        save_json(
            payload,
            report_path.with_name(f"{report_path.stem}_seeds{report_path.suffix}"),
        )
    return payload


def evaluate_boundary_reranker_seeds(
    config: dict[str, Any],
    device: str | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Evaluate existing seed checkpoints without retraining them."""
    seeds = [
        int(seed)
        for seed in config.get("boundary_evaluation", {}).get(
            "seeds", [42, 43, 44]
        )
    ]
    base_checkpoint = Path(config["boundary_train"]["checkpoint"])
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint_file = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_seed{seed}{base_checkpoint.suffix}"
        )
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Boundary checkpoint for seed {seed} is missing: {checkpoint_file}"
            )
        checkpoint = torch.load(
            checkpoint_file, map_location="cpu", weights_only=False
        )
        run_config = copy.deepcopy(config)
        run_config["boundary_evaluation"].pop("report_file", None)
        evaluation = evaluate_boundary_reranker(
            run_config,
            device=device,
            checkpoint_path=checkpoint_file,
            split=split,
        )
        runs.append(
            {
                "training": {
                    "best_selection_score": checkpoint.get("selection_score", 0.0),
                    "checkpoint": str(checkpoint_file),
                    "seed": seed,
                },
                "evaluation": evaluation,
            }
        )

    payload = _summarize_runs(config, seeds, runs)
    validation_report = Path(config["boundary_evaluation"]["report_file"])
    split_name = validation_report.name.replace("_validation", f"_{split}")
    output = validation_report.with_name(Path(split_name).stem + "_seeds.json")
    save_json(payload, output)
    payload["report_file"] = str(output)
    return payload
