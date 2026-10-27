from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from .common import get_device, load_feature_store, move_to_device, save_json
from .erc_trainer import (
    _build_model,
    _loader,
    _same_speaker_transition_masks,
    _subset_classification_metrics,
    _validate_current_logit_features,
)


SUBSETS = ("overall", "stable", "transition", "unknown")
MODEL_OUTPUTS = ("anchor", "context", "fusion")


def _scalar_summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values) if values else None,
        "std": pstdev(values) if values else None,
        "count": len(values),
    }


def _rate(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(bool(record[key]) for record in records) / len(records)


def _mean(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return sum(float(record[key]) for record in records) / len(records)


def _std(records: list[dict[str, Any]], key: str) -> float | None:
    if not records:
        return None
    return pstdev(float(record[key]) for record in records)


def _classification(
    records: list[dict[str, Any]],
    prediction_key: str,
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    labels = [int(record["label"]) for record in records]
    predictions = [int(record[prediction_key]) for record in records]
    return _subset_classification_metrics(labels, predictions, id_to_label)


def _correction_subset_metrics(
    records: list[dict[str, Any]], id_to_label: dict[int, str]
) -> dict[str, Any]:
    """Summarize correction direction for one evaluation subset."""
    ordered_ids = sorted(id_to_label)
    classification = {
        output: _classification(records, f"{output}_prediction", id_to_label)
        for output in MODEL_OUTPUTS
    }
    labels = [int(record["label"]) for record in records]
    if labels:
        confusion_matrices = {
            output: confusion_matrix(
                labels,
                [int(record[f"{output}_prediction"]) for record in records],
                labels=ordered_ids,
            ).tolist()
            for output in MODEL_OUTPUTS
        }
    else:
        empty_matrix = [[0 for _ in ordered_ids] for _ in ordered_ids]
        confusion_matrices = {
            output: [row.copy() for row in empty_matrix]
            for output in MODEL_OUTPUTS
        }
    per_emotion: dict[str, Any] = {}
    for label_id in ordered_ids:
        emotion = id_to_label[label_id]
        emotion_records = [
            record for record in records if int(record["label"]) == label_id
        ]
        anchor_report = classification["anchor"]["classification_report"].get(
            emotion, {}
        )
        context_report = classification["context"]["classification_report"].get(
            emotion, {}
        )
        fusion_report = classification["fusion"]["classification_report"].get(
            emotion, {}
        )
        context_beneficial = sum(
            bool(record["context_beneficial_flip"]) for record in emotion_records
        )
        context_harmful = sum(
            bool(record["context_harmful_flip"]) for record in emotion_records
        )
        fusion_beneficial = sum(
            bool(record["fusion_beneficial_flip"]) for record in emotion_records
        )
        fusion_harmful = sum(
            bool(record["fusion_harmful_flip"]) for record in emotion_records
        )
        per_emotion[emotion] = {
            "label_id": label_id,
            "support": len(emotion_records),
            "anchor_precision": anchor_report.get("precision"),
            "anchor_recall": anchor_report.get("recall"),
            "anchor_f1": anchor_report.get("f1-score"),
            "context_precision": context_report.get("precision"),
            "context_recall": context_report.get("recall"),
            "context_f1": context_report.get("f1-score"),
            "fusion_precision": fusion_report.get("precision"),
            "fusion_recall": fusion_report.get("recall"),
            "fusion_f1": fusion_report.get("f1-score"),
            "anchor_context_agreement_rate": _rate(
                emotion_records, "anchor_context_agree"
            ),
            "fusion_anchor_agreement_rate": _rate(
                emotion_records, "fusion_anchor_agree"
            ),
            "context_flip_rate": _rate(emotion_records, "context_changed_anchor"),
            "fusion_flip_rate": _rate(emotion_records, "fusion_changed_anchor"),
            "context_beneficial_flips": context_beneficial,
            "context_harmful_flips": context_harmful,
            "context_net_correct_flips": context_beneficial - context_harmful,
            "fusion_beneficial_flips": fusion_beneficial,
            "fusion_harmful_flips": fusion_harmful,
            "fusion_net_correct_flips": fusion_beneficial - fusion_harmful,
            "context_nll_help_rate": _rate(emotion_records, "context_nll_helpful"),
            "fusion_nll_help_rate": _rate(emotion_records, "fusion_nll_helpful"),
            "context_delta_nll_mean": _mean(emotion_records, "context_delta_nll"),
            "fusion_delta_nll_mean": _mean(emotion_records, "fusion_delta_nll"),
            "context_margin_help_rate": _rate(
                emotion_records, "context_margin_helpful"
            ),
            "fusion_margin_help_rate": _rate(
                emotion_records, "fusion_margin_helpful"
            ),
            "context_delta_margin_mean": _mean(
                emotion_records, "context_delta_margin"
            ),
            "fusion_delta_margin_mean": _mean(
                emotion_records, "fusion_delta_margin"
            ),
            "gate_mean": _mean(emotion_records, "gate"),
            "gate_std": _std(emotion_records, "gate"),
            "base_gate_mean": _mean(emotion_records, "base_gate"),
            "transition_probability_mean": _mean(
                emotion_records, "transition_probability"
            ),
        }
    return {
        "utterances": len(records),
        "label_order": [id_to_label[label_id] for label_id in ordered_ids],
        "classification": classification,
        "confusion_matrices": confusion_matrices,
        "per_emotion": per_emotion,
    }


def _gold_margin(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    safe_labels = labels.clamp_min(0)
    gold = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    other = logits.masked_fill(
        torch.nn.functional.one_hot(
            safe_labels, num_classes=logits.shape[-1]
        ).bool(),
        float("-inf"),
    ).max(dim=-1).values
    return gold - other


def _batch_records(
    batch: dict[str, Any],
    fusion_logits: torch.Tensor,
    auxiliary: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    valid = batch["mask"]
    labels = batch["labels"]
    anchor_logits = batch["current_logits"].to(fusion_logits.dtype)
    context_logits = auxiliary["context_logits"]
    known, transition = _same_speaker_transition_masks(
        batch["speaker_ids"], labels, valid
    )
    stable = known & ~transition

    anchor_prediction = anchor_logits.argmax(dim=-1)
    context_prediction = context_logits.argmax(dim=-1)
    fusion_prediction = fusion_logits.argmax(dim=-1)
    safe_labels = labels.clamp_min(0).unsqueeze(-1)
    anchor_nll = -torch.log_softmax(anchor_logits.float(), dim=-1).gather(
        -1, safe_labels
    ).squeeze(-1)
    context_nll = -torch.log_softmax(context_logits.float(), dim=-1).gather(
        -1, safe_labels
    ).squeeze(-1)
    fusion_nll = -torch.log_softmax(fusion_logits.float(), dim=-1).gather(
        -1, safe_labels
    ).squeeze(-1)
    anchor_margin = _gold_margin(anchor_logits.float(), labels)
    context_margin = _gold_margin(context_logits.float(), labels)
    fusion_margin = _gold_margin(fusion_logits.float(), labels)
    gate = auxiliary["context_gate"].squeeze(-1)
    base_gate = auxiliary["base_context_gate"].squeeze(-1)
    transition_probability = auxiliary[
        "predicted_transition_probability"
    ]

    tensors = {
        "label": labels[valid],
        "anchor_prediction": anchor_prediction[valid],
        "context_prediction": context_prediction[valid],
        "fusion_prediction": fusion_prediction[valid],
        "known": known[valid],
        "stable": stable[valid],
        "transition": transition[valid],
        "anchor_nll": anchor_nll[valid],
        "context_nll": context_nll[valid],
        "fusion_nll": fusion_nll[valid],
        "anchor_margin": anchor_margin[valid],
        "context_margin": context_margin[valid],
        "fusion_margin": fusion_margin[valid],
        "gate": gate[valid],
        "base_gate": base_gate[valid],
        "transition_probability": transition_probability[valid],
    }
    cpu = {key: value.detach().cpu().tolist() for key, value in tensors.items()}
    records: list[dict[str, Any]] = []
    for index in range(len(cpu["label"])):
        label = int(cpu["label"][index])
        anchor_pred = int(cpu["anchor_prediction"][index])
        context_pred = int(cpu["context_prediction"][index])
        fusion_pred = int(cpu["fusion_prediction"][index])
        context_delta_nll = float(cpu["anchor_nll"][index]) - float(
            cpu["context_nll"][index]
        )
        fusion_delta_nll = float(cpu["anchor_nll"][index]) - float(
            cpu["fusion_nll"][index]
        )
        context_delta_margin = float(cpu["context_margin"][index]) - float(
            cpu["anchor_margin"][index]
        )
        fusion_delta_margin = float(cpu["fusion_margin"][index]) - float(
            cpu["anchor_margin"][index]
        )
        if bool(cpu["transition"][index]):
            subset = "transition"
        elif bool(cpu["stable"][index]):
            subset = "stable"
        else:
            subset = "unknown"
        records.append(
            {
                "label": label,
                "subset": subset,
                "anchor_prediction": anchor_pred,
                "context_prediction": context_pred,
                "fusion_prediction": fusion_pred,
                "anchor_context_agree": anchor_pred == context_pred,
                "fusion_anchor_agree": fusion_pred == anchor_pred,
                "context_changed_anchor": context_pred != anchor_pred,
                "fusion_changed_anchor": fusion_pred != anchor_pred,
                "context_beneficial_flip": anchor_pred != label
                and context_pred == label,
                "context_harmful_flip": anchor_pred == label and context_pred != label,
                "fusion_beneficial_flip": anchor_pred != label and fusion_pred == label,
                "fusion_harmful_flip": anchor_pred == label and fusion_pred != label,
                "context_nll_helpful": context_delta_nll > 0.0,
                "fusion_nll_helpful": fusion_delta_nll > 0.0,
                "context_delta_nll": context_delta_nll,
                "fusion_delta_nll": fusion_delta_nll,
                "context_margin_helpful": context_delta_margin > 0.0,
                "fusion_margin_helpful": fusion_delta_margin > 0.0,
                "context_delta_margin": context_delta_margin,
                "fusion_delta_margin": fusion_delta_margin,
                "gate": float(cpu["gate"][index]),
                "base_gate": float(cpu["base_gate"][index]),
                "transition_probability": float(
                    cpu["transition_probability"][index]
                ),
            }
        )
    return records


def analyze_correction_checkpoint(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    device: str | None = None,
    split: str | None = None,
    expected_seed: int | None = None,
) -> dict[str, Any]:
    device_obj = get_device(device)
    store = load_feature_store(config["data"]["feature_file"])
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seed = int(checkpoint.get("seed", -1))
    if expected_seed is not None and seed != expected_seed:
        raise ValueError(
            f"4A source checkpoint {checkpoint_path} has seed={seed}, "
            f"expected seed={expected_seed}"
        )
    expected_experiments = {
        str(name)
        for name in config["correction_analysis"].get(
            "compatible_checkpoint_experiments", []
        )
    }
    saved_config = checkpoint.get("config", {})
    saved_experiment = str(saved_config.get("experiment", {}).get("name", ""))
    if expected_experiments and saved_experiment not in expected_experiments:
        raise ValueError(
            f"4A source checkpoint {checkpoint_path} belongs to "
            f"{saved_experiment!r}; expected one of {sorted(expected_experiments)!r}"
        )
    expected_model = config.get("model", {})
    saved_model = saved_config.get("model", {})
    lineage = {
        "logit_fusion": "none",
        "context_gate_mode": "transition",
        "transition_gate_power": 1.0,
        "context_gate_init": 0.1,
        "context_gate_max": 1.0,
    }
    mismatches = []
    for key, default in lineage.items():
        expected = expected_model.get(key, default)
        saved = saved_model.get(key, default)
        if saved != expected:
            mismatches.append(f"{key}: saved={saved!r}, expected={expected!r}")
    if mismatches:
        raise ValueError(
            f"4A source checkpoint {checkpoint_path} has incompatible settings: "
            + "; ".join(mismatches)
        )
    label_to_id = {
        str(key): int(value) for key, value in checkpoint["label_to_id"].items()
    }
    store_labels = {
        str(key): int(value) for key, value in store["label_to_id"].items()
    }
    if store_labels != label_to_id:
        raise ValueError(
            "4A feature-store label mapping does not match the source checkpoint"
        )
    id_to_label = {value: key for key, value in label_to_id.items()}
    model = _build_model(config, len(label_to_id), device_obj)
    if not model.use_current_logit_anchor:
        raise ValueError("Experiment 4A requires a current-logit anchored model")
    _validate_current_logit_features(store, len(label_to_id))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    analysis = config["correction_analysis"]
    split = split or str(analysis.get("split", "validation"))
    loader = _loader(store["splits"][split], config, False, seed, device_obj)
    use_amp = bool(config["train_erc"].get("use_amp", True)) and device_obj.type == "cuda"
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"4A correction seed={seed}", leave=False):
            batch = move_to_device(batch, device_obj)
            with torch.autocast(
                device_type=device_obj.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                fusion_logits, auxiliary = model(
                    batch["embeddings"],
                    batch["speaker_ids"],
                    batch["mask"],
                    current_logits=batch.get("current_logits"),
                    return_aux=True,
                )
            records.extend(_batch_records(batch, fusion_logits, auxiliary))

    subsets = {
        "overall": _correction_subset_metrics(records, id_to_label),
        **{
            name: _correction_subset_metrics(
                [record for record in records if record["subset"] == name],
                id_to_label,
            )
            for name in SUBSETS
            if name != "overall"
        },
    }
    return {
        "checkpoint": str(checkpoint_path),
        "seed": seed,
        "split": split,
        "utterances": len(records),
        "subsets": subsets,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "anchor_precision",
        "anchor_recall",
        "anchor_f1",
        "context_precision",
        "context_recall",
        "context_f1",
        "fusion_precision",
        "fusion_recall",
        "fusion_f1",
        "anchor_context_agreement_rate",
        "fusion_anchor_agreement_rate",
        "context_flip_rate",
        "fusion_flip_rate",
        "context_nll_help_rate",
        "fusion_nll_help_rate",
        "context_delta_nll_mean",
        "fusion_delta_nll_mean",
        "context_margin_help_rate",
        "fusion_margin_help_rate",
        "context_delta_margin_mean",
        "fusion_delta_margin_mean",
        "gate_mean",
        "gate_std",
        "base_gate_mean",
        "transition_probability_mean",
    )
    count_names = (
        "support",
        "context_beneficial_flips",
        "context_harmful_flips",
        "context_net_correct_flips",
        "fusion_beneficial_flips",
        "fusion_harmful_flips",
        "fusion_net_correct_flips",
    )
    summary: dict[str, Any] = {"subsets": {}}
    emotions = list(reports[0]["subsets"]["overall"]["per_emotion"])
    for subset in SUBSETS:
        subset_summary: dict[str, Any] = {"per_emotion": {}}
        for emotion in emotions:
            items = [report["subsets"][subset]["per_emotion"][emotion] for report in reports]
            emotion_summary = {
                metric: _scalar_summary(
                    [float(item[metric]) for item in items if item.get(metric) is not None]
                )
                for metric in metric_names
            }
            emotion_summary.update(
                {
                    metric: _scalar_summary(
                        [float(item[metric]) for item in items if item.get(metric) is not None]
                    )
                    for metric in count_names
                }
            )
            subset_summary["per_emotion"][emotion] = emotion_summary
        for output in MODEL_OUTPUTS:
            metrics = [
                report["subsets"][subset]["classification"][output]
                for report in reports
            ]
            subset_summary[f"{output}_weighted_f1"] = _scalar_summary(
                [float(item["weighted_f1"]) for item in metrics if item["weighted_f1"] is not None]
            )
            subset_summary[f"{output}_macro_f1"] = _scalar_summary(
                [float(item["macro_f1"]) for item in metrics if item["macro_f1"] is not None]
            )
        summary["subsets"][subset] = subset_summary
    return summary


def analyze_correction_sweep(
    config: dict[str, Any],
    device: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    analysis = config["correction_analysis"]
    base_checkpoint = Path(analysis["checkpoint"])
    report_file = Path(analysis["report_file"])
    seeds = [int(seed) for seed in analysis.get("seeds", [42, 43, 44])]
    split = split or str(analysis.get("split", "validation"))
    if not seeds:
        raise ValueError("Experiment 4A requires at least one checkpoint seed")
    if split == "test" and not bool(analysis.get("allow_test", False)):
        raise ValueError(
            "Experiment 4A is validation-only. Set correction_analysis.allow_test=true "
            "only after the analysis protocol has been frozen."
        )
    reports: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = base_checkpoint.with_name(f"{base_checkpoint.stem}_seed{seed}.pt")
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing 4A source checkpoint: {checkpoint}")
        report = analyze_correction_checkpoint(
            config,
            checkpoint,
            device=device,
            split=split,
            expected_seed=seed,
        )
        reports.append(report)
        if bool(analysis.get("save_individual_seed_reports", True)):
            individual = report_file.with_name(
                f"{report_file.stem}_seed{seed}{report_file.suffix}"
            )
            save_json(report, individual)

    result = {
        "status": "completed",
        "experiment": str(config["experiment"]["name"]),
        "split": split,
        "seeds": seeds,
        "source_checkpoint_base": str(base_checkpoint),
        "feature_file": str(config["data"]["feature_file"]),
        "test_used_for_model_selection": split == "test",
        "summary": _aggregate_reports(reports),
        "runs": reports,
    }
    save_json(result, report_file)
    return result
