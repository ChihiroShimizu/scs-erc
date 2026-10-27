from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from ..data import prepare_dataset
from ..features import extract_features
from .calibration import (
    _collect_dialogue_outputs,
    _concatenate,
    _cross_fitted_calibration,
    _fit_temperature,
    _metrics_from_logits,
    _temperature_from_parameters,
)
from .common import (
    get_device,
    load_feature_store,
    save_json,
)
from .erc_trainer import _build_model, evaluate_erc, train_erc
from .roberta_trainer import evaluate_roberta, train_roberta


VARIANTS: dict[str, dict[str, str]] = {
    "final": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "speaker",
    },
    # This control preserves logit-residual geometry and removes only the
    # utterance-conditioned matrix delta.
    "static": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "static_logit_residual",
        "directed_prior_mode": "speaker",
    },
    "uniform": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "uniform",
    },
    "wrong_recipient": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "other_speaker",
    },
}
PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "weighted_f1": ("weighted_f1",),
    "macro_f1": ("macro_f1",),
    "accuracy": ("accuracy",),
    "nll": ("probability_quality", "nll"),
    "brier": ("probability_quality", "brier"),
    "ece": ("probability_quality", "ece"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values) if values else None,
        "std": pstdev(values) if values else None,
        "count": len(values),
    }


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _experiment(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("experiment9")
    if not isinstance(section, dict):
        raise KeyError("Config is missing the experiment9 section")
    return section


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(seed) for seed in _experiment(config).get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "experiment9.seeds must contain at least three distinct seeds"
        )
    return seeds


def _artifact_dir(config: dict[str, Any]) -> Path:
    return Path(_experiment(config)["artifact_dir"])


def _dataset_name(config: dict[str, Any]) -> str:
    return str(config["data"]["dataset"]).lower()


def _seed_config(
    config: dict[str, Any],
    seed: int,
    *,
    variant: str = "final",
    split: str = "validation",
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown Experiment 9 variant: {variant}")
    result = copy.deepcopy(config)
    root = _artifact_dir(result)
    result["experiment"]["seed"] = int(seed)
    result["experiment"]["name"] = (
        f"{_dataset_name(result)}_experiment9_{variant}_seed{seed}"
    )
    anchor_dir = root / "anchors" / f"seed{seed}"
    feature_file = root / "features" / f"seed{seed}.pt"
    result["train_roberta"]["checkpoint"] = str(anchor_dir)
    result["features"]["encoder_checkpoint"] = str(anchor_dir)
    result["features"]["splits"] = ["train", "validation"]
    result["data"]["feature_file"] = str(feature_file)
    result["train_erc"]["checkpoint"] = str(
        root / "checkpoints" / f"{variant}_seed{seed}.pt"
    )
    result["evaluation"]["split"] = split
    result["evaluation"]["seeds"] = [seed]
    result["evaluation"]["report_file"] = str(
        root / split / f"{variant}_seed{seed}.json"
    )
    result["evaluation_roberta"]["split"] = split
    result["evaluation_roberta"]["report_file"] = str(
        root / split / f"anchor_seed{seed}.json"
    )
    result["model"].update(VARIANTS[variant])
    return result


def _test_seed_config(
    config: dict[str, Any], seed: int
) -> dict[str, Any]:
    result = _seed_config(
        config, seed, variant="final", split="test"
    )
    result["data"]["feature_file"] = str(
        _artifact_dir(result) / "test_features" / f"seed{seed}.pt"
    )
    result["features"]["splits"] = ["test"]
    return result


def _checkpoint_matches(
    path: Path,
    config: dict[str, Any],
    seed: int,
    variant: str,
) -> bool:
    if not path.exists():
        return False
    payload = torch.load(path, map_location="cpu", weights_only=False)
    saved = payload.get("config", {})
    saved_model = saved.get("model", {})
    expected = VARIANTS[variant]
    return (
        int(payload.get("seed", -1)) == int(seed)
        and Path(saved.get("data", {}).get("feature_file", "")) ==
        Path(config["data"]["feature_file"])
        and all(saved_model.get(key) == value for key, value in expected.items())
    )


def _aggregate_metric_reports(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = [report["metrics"] for report in reports]
    aggregate = {
        name: _summary(
            [
                float(value)
                for item in metrics
                if (value := _nested(item, path)) is not None
            ]
        )
        for name, path in PRIMARY_METRICS.items()
    }
    first_report = metrics[0]["classification_report"]
    excluded = {"accuracy", "macro avg", "weighted avg"}
    emotions = [
        name
        for name, values in first_report.items()
        if name not in excluded and isinstance(values, dict)
    ]
    aggregate["per_emotion_f1"] = {
        emotion: _summary(
            [
                float(item["classification_report"][emotion]["f1-score"])
                for item in metrics
            ]
        )
        for emotion in emotions
    }
    return aggregate


def _anchor_stage(config: dict[str, Any], device: str | None) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_experiment(config).get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed)
        checkpoint = Path(seed_config["train_roberta"]["checkpoint"])
        state_file = checkpoint / "pfa_erc_training.json"
        if reuse and state_file.exists():
            state = _read_json(state_file)
            if int(state.get("seed", -1)) != seed:
                raise ValueError(
                    f"Anchor checkpoint seed mismatch: {checkpoint}"
                )
            training = {
                "checkpoint": str(checkpoint),
                "seed": seed,
                "reused_existing_checkpoint": True,
                "best_validation_weighted_f1": state.get(
                    "best_validation_metrics", {}
                ).get("weighted_f1"),
            }
        else:
            training = train_roberta(seed_config, device=device, seed=seed)
            training["reused_existing_checkpoint"] = False
        evaluation = evaluate_roberta(
            seed_config,
            device=device,
            checkpoint_path=checkpoint,
            split="validation",
            save_report=False,
        )
        individual = root / "validation" / f"anchor_seed{seed}.json"
        save_json(evaluation, individual)
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "training": training,
                "evaluation": evaluation,
                "individual_report": str(individual),
            }
        )
    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "independent_encoder_training": True,
            "test_used": False,
        },
        "runs": runs,
        "aggregate": _aggregate_metric_reports(
            [run["evaluation"] for run in runs]
        ),
    }
    save_json(payload, root / "validation" / "anchor_seeds.json")
    return payload


def _feature_stage(config: dict[str, Any], device: str | None) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_experiment(config).get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed)
        feature_file = Path(seed_config["data"]["feature_file"])
        if reuse and feature_file.exists():
            store = load_feature_store(feature_file)
            source = Path(
                store.get("feature_config", {}).get(
                    "encoder_checkpoint", ""
                )
            )
            expected = Path(seed_config["features"]["encoder_checkpoint"])
            if source != expected:
                raise ValueError(
                    f"Feature cache lineage mismatch: {feature_file}"
                )
            result = {
                "feature_file": str(feature_file),
                "embedding_dim": int(store["embedding_dim"]),
                "reused_existing_cache": True,
            }
        else:
            result = extract_features(seed_config, device=device)
            result["reused_existing_cache"] = False
        result["seed"] = seed
        runs.append(result)
    payload = {
        "status": "completed",
        "dataset": _dataset_name(config),
        "seeds": _seeds(config),
        "independent_feature_cache_per_encoder_seed": True,
        "runs": runs,
    }
    save_json(payload, root / "validation" / "feature_manifest.json")
    return payload


def _train_variant_stage(
    config: dict[str, Any],
    device: str | None,
    variant: str,
) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_experiment(config).get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed, variant=variant)
        feature_file = Path(seed_config["data"]["feature_file"])
        if not feature_file.exists():
            raise FileNotFoundError(
                f"Missing seed-specific feature cache: {feature_file}"
            )
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        if reuse and _checkpoint_matches(
            checkpoint, seed_config, seed, variant
        ):
            saved = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            training = {
                "checkpoint": str(checkpoint),
                "seed": seed,
                "best_validation_weighted_f1": saved.get(
                    "validation_metrics", {}
                ).get("weighted_f1"),
                "best_validation_macro_f1": saved.get(
                    "validation_metrics", {}
                ).get("macro_f1"),
                "best_validation_selection_score": saved.get(
                    "validation_selection_score"
                ),
                "reused_existing_checkpoint": True,
            }
        else:
            training = train_erc(
                seed_config,
                device=device,
                seed=seed,
                checkpoint_path=checkpoint,
            )
            training["reused_existing_checkpoint"] = False
        evaluation = evaluate_erc(
            seed_config,
            device=device,
            checkpoint_path=checkpoint,
            split="validation",
            save_report=False,
        )
        individual = root / "validation" / f"{variant}_seed{seed}.json"
        save_json(evaluation, individual)
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "feature_file": str(feature_file),
                "training": training,
                "evaluation": evaluation,
                "individual_report": str(individual),
            }
        )
    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "variant": variant,
            "model": VARIANTS[variant],
            "independent_encoder_seed_per_model_seed": True,
            "stable_transition_used_for_training": False,
            "stable_transition_used_for_selection": False,
            "test_used": False,
        },
        "runs": runs,
        "aggregate": _aggregate_metric_reports(
            [run["evaluation"] for run in runs]
        ),
    }
    save_json(payload, root / "validation" / f"{variant}_seeds.json")
    return payload


def _identity_parameters(minimum: float, maximum: float) -> dict[str, Any]:
    fraction = (1.0 - minimum) / (maximum - minimum)
    raw_bias = math.log(fraction / (1.0 - fraction))
    return {
        "method": "scalar",
        "minimum": minimum,
        "maximum": maximum,
        "gate_center": 0.0,
        "raw_bias": raw_bias,
        "raw_slope": None,
        "selected_identity": True,
        "fit_temperature_mean": 1.0,
    }


def _calibration_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    root = _artifact_dir(config)
    settings = _experiment(config).get("calibration", {})
    folds = int(settings.get("folds", 5))
    fold_seed = int(settings.get("fold_seed", 2026))
    minimum = float(settings.get("minimum_temperature", 0.25))
    maximum = float(settings.get("maximum_temperature", 4.0))
    max_iter = int(settings.get("max_iter", 100))
    device_obj = get_device(device)
    provisional: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed, variant="final")
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        records, id_to_label, stored_seed = _collect_dialogue_outputs(
            seed_config, checkpoint, "validation", device_obj
        )
        if stored_seed != seed:
            raise ValueError(f"Checkpoint seed mismatch: {checkpoint}")
        indices = list(range(len(records)))
        logits, labels, gate = _concatenate(records, indices)
        uncalibrated = _metrics_from_logits(
            logits, labels, id_to_label
        )
        cross_fitted = _cross_fitted_calibration(
            records=records,
            id_to_label=id_to_label,
            method="scalar",
            folds=folds,
            fold_seed=fold_seed,
            minimum=minimum,
            maximum=maximum,
            max_iter=max_iter,
            slope_l2=0.0,
        )
        full_parameters = _fit_temperature(
            logits=logits,
            labels=labels,
            gate=gate,
            method="scalar",
            minimum=minimum,
            maximum=maximum,
            max_iter=max_iter,
            slope_l2=0.0,
        )
        improvement = (
            float(uncalibrated["probability_quality"]["nll"])
            - float(
                cross_fitted["metrics"]["probability_quality"]["nll"]
            )
        )
        provisional.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "dialogues": len(records),
                "utterances": int(labels.numel()),
                "uncalibrated": uncalibrated,
                "scalar_cross_fitted": cross_fitted,
                "full_parameters": full_parameters,
                "cross_fitted_nll_improvement": improvement,
            }
        )

    improvements = [
        float(run["cross_fitted_nll_improvement"]) for run in provisional
    ]
    scalar_selected = (
        mean(improvements) > 0.0
        and sum(value > 0.0 for value in improvements) >= 2
    )
    selected_method = "scalar" if scalar_selected else "identity"
    runs: list[dict[str, Any]] = []
    calibrator_dir = root / "calibration"
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    for run in provisional:
        fitted_parameters = run.pop("full_parameters")
        parameters = (
            fitted_parameters
            if scalar_selected
            else _identity_parameters(minimum, maximum)
        )
        calibrator_file = calibrator_dir / (
            f"{selected_method}_seed{run['seed']}.json"
        )
        calibrator = {
            "format_version": "pfa-erc-temperature-v1",
            "method": "scalar",
            "selection": selected_method,
            "model_seed": int(run["seed"]),
            "source_checkpoint": run["checkpoint"],
            "fit_split": "validation",
            "dataset": _dataset_name(config),
            "parameters": parameters,
        }
        save_json(calibrator, calibrator_file)
        run["selected_calibrator"] = str(calibrator_file)
        runs.append(run)

    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "folds": folds,
            "fold_unit": "dialogue",
            "fold_seed": fold_seed,
            "classification_boundary_changed": False,
            "test_used": False,
            "selection_rule": (
                "Use scalar temperature only when cross-fitted mean NLL "
                "improves and at least two of three seeds improve; otherwise "
                "use identity temperature."
            ),
        },
        "selected_method": selected_method,
        "cross_fitted_nll_improvement": _summary(improvements),
        "runs": runs,
        "test_locked": True,
    }
    save_json(payload, root / "validation" / "calibration.json")
    return payload


def _paired_seed_contrast(
    reference: dict[str, Any],
    control: dict[str, Any],
    path: tuple[str, ...],
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    values: list[float] = []
    for reference_run, control_run in zip(
        reference["runs"], control["runs"]
    ):
        reference_value = _nested(
            reference_run["evaluation"]["metrics"], path
        )
        control_value = _nested(
            control_run["evaluation"]["metrics"], path
        )
        if reference_value is None or control_value is None:
            continue
        values.append(
            float(control_value) - float(reference_value)
            if lower_is_better
            else float(reference_value) - float(control_value)
        )
    result = _summary(values)
    result["positive_seeds"] = sum(value > 0.0 for value in values)
    result["negative_seeds"] = sum(value < 0.0 for value in values)
    return result


def _validation_summary(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    anchor_path = root / "validation" / "anchor_seeds.json"
    calibration_path = root / "validation" / "calibration.json"
    variant_paths = {
        variant: root / "validation" / f"{variant}_seeds.json"
        for variant in VARIANTS
    }
    required = [anchor_path, calibration_path, *variant_paths.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experiment 9 validation is incomplete: " + ", ".join(missing)
        )
    anchor = _read_json(anchor_path)
    calibration = _read_json(calibration_path)
    variants = {
        name: _read_json(path) for name, path in variant_paths.items()
    }
    final = variants["final"]
    mechanisms: dict[str, Any] = {}
    for control in ("static", "uniform", "wrong_recipient"):
        mechanisms[control] = {
            "weighted_f1": _paired_seed_contrast(
                final,
                variants[control],
                PRIMARY_METRICS["weighted_f1"],
                lower_is_better=False,
            ),
            "macro_f1": _paired_seed_contrast(
                final,
                variants[control],
                PRIMARY_METRICS["macro_f1"],
                lower_is_better=False,
            ),
            "nll": _paired_seed_contrast(
                final,
                variants[control],
                PRIMARY_METRICS["nll"],
                lower_is_better=True,
            ),
        }
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "protocol": {
            "series": "9A-9D",
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "frozen_method": (
                "dynamic speaker-conditioned logit-residual matrix update"
            ),
            "independent_encoder_seeds": True,
            "stable_transition_used_for_training": False,
            "stable_transition_used_for_selection": False,
            "test_used": False,
            "test_policy": (
                "Validation results are descriptive and do not decide whether "
                "the new dataset is reported. The frozen final test is run "
                "once and reported regardless of direction."
            ),
        },
        "anchor": anchor["aggregate"],
        "variants": {
            name: payload["aggregate"]
            for name, payload in variants.items()
        },
        "paired_mechanism_advantages": mechanisms,
        "calibration": calibration,
        "frozen_for_test": True,
        "test_locked": True,
    }
    save_json(payload, root / "reports" / "validation_summary.json")
    return payload


def _probability_metrics(
    labels: list[int],
    probabilities: list[list[float]],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    probability = torch.tensor(probabilities, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    logits = probability.clamp_min(1e-8).log()
    return _metrics_from_logits(logits, label_tensor, id_to_label)


def _test_seed(
    config: dict[str, Any],
    seed: int,
    device: str | None,
    calibrator_file: Path,
) -> dict[str, Any]:
    seed_config = _test_seed_config(config, seed)
    checkpoint = Path(seed_config["train_erc"]["checkpoint"])
    test_feature_file = Path(seed_config["data"]["feature_file"])
    if test_feature_file.exists():
        store = load_feature_store(test_feature_file)
        if set(store["splits"]) != {"test"}:
            raise ValueError(
                f"Invalid test-only feature cache: {test_feature_file}"
            )
        source = Path(
            store.get("feature_config", {}).get(
                "encoder_checkpoint", ""
            )
        )
        if source != Path(seed_config["features"]["encoder_checkpoint"]):
            raise ValueError(
                f"Test feature lineage mismatch: {test_feature_file}"
            )
    else:
        extract_features(seed_config, device=device)
    calibrator = _read_json(calibrator_file)
    records, id_to_label, stored_seed = _collect_dialogue_outputs(
        seed_config, checkpoint, "test", get_device(device)
    )
    if stored_seed != seed or int(calibrator["model_seed"]) != seed:
        raise ValueError(f"Seed lineage mismatch for test seed {seed}")
    all_logits, all_labels, all_gates = _concatenate(
        records, list(range(len(records)))
    )
    all_anchor = torch.cat(
        [record["anchor_logits"] for record in records], dim=0
    )
    temperature = _temperature_from_parameters(
        all_gates, calibrator["parameters"]
    )
    calibrated_logits = all_logits / temperature.unsqueeze(-1)
    predictions: list[dict[str, Any]] = []
    offset = 0
    for record in records:
        length = int(record["labels"].numel())
        current_temperature = temperature[offset : offset + length]
        final_logits = record["logits"] / current_temperature.unsqueeze(-1)
        predictions.append(
            {
                "dialogue_id": str(record["dialogue_id"]),
                "labels": record["labels"].tolist(),
                "anchor_probabilities": torch.softmax(
                    record["anchor_logits"].float(), dim=-1
                ).tolist(),
                "final_probabilities": torch.softmax(
                    final_logits.float(), dim=-1
                ).tolist(),
            }
        )
        offset += length
    return {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "test_feature_file": str(test_feature_file),
        "test_feature_sha256": _sha256(test_feature_file),
        "calibrator": str(calibrator_file),
        "calibrator_sha256": _sha256(calibrator_file),
        "split": "test",
        "dialogues": len(records),
        "utterances": int(all_labels.numel()),
        "anchor": _metrics_from_logits(
            all_anchor, all_labels, id_to_label
        ),
        "uncalibrated": _metrics_from_logits(
            all_logits, all_labels, id_to_label
        ),
        "final": _metrics_from_logits(
            calibrated_logits, all_labels, id_to_label
        ),
        "temperature_mean": float(temperature.mean()),
        "argmax_changes_from_calibration": int(
            all_logits.argmax(dim=-1)
            .ne(calibrated_logits.argmax(dim=-1))
            .sum()
        ),
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
        "predictions": predictions,
    }


def _metric_from_prediction_records(
    records: list[dict[str, Any]],
    id_to_label: dict[int, str],
    variant: str,
    metric: str,
) -> float:
    labels: list[int] = []
    probabilities: list[list[float]] = []
    key = f"{variant}_probabilities"
    for record in records:
        labels.extend(int(value) for value in record["labels"])
        probabilities.extend(record[key])
    if metric in {"weighted_f1", "macro_f1", "accuracy"}:
        predictions = [
            max(range(len(row)), key=lambda index: row[index])
            for row in probabilities
        ]
        ordered_labels = sorted(id_to_label)
        if metric == "accuracy":
            return float(accuracy_score(labels, predictions))
        return float(
            f1_score(
                labels,
                predictions,
                labels=ordered_labels,
                average=(
                    "weighted" if metric == "weighted_f1" else "macro"
                ),
                zero_division=0,
            )
        )
    return float(
        _probability_metrics(labels, probabilities, id_to_label)[
            "probability_quality"
        ][metric]
    )


def _hierarchical_bootstrap(
    runs: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError(
            "experiment9.bootstrap_samples must be at least 100"
        )
    generator = np.random.default_rng(seed)
    metrics = ("weighted_f1", "macro_f1", "accuracy", "nll", "brier")
    draws: dict[str, list[float]] = {metric: [] for metric in metrics}
    for _ in range(samples):
        sampled_seed_indices = generator.integers(
            0, len(runs), size=len(runs)
        )
        seed_differences: dict[str, list[float]] = {
            metric: [] for metric in metrics
        }
        for seed_index in sampled_seed_indices:
            run = runs[int(seed_index)]
            records = run["predictions"]
            record_indices = generator.integers(
                0, len(records), size=len(records)
            )
            sampled_records = [
                records[int(index)] for index in record_indices
            ]
            id_to_label = {
                int(key): value
                for key, value in run["id_to_label"].items()
            }
            for metric in metrics:
                final = _metric_from_prediction_records(
                    sampled_records, id_to_label, "final", metric
                )
                anchor = _metric_from_prediction_records(
                    sampled_records, id_to_label, "anchor", metric
                )
                # Positive means final is better for every reported metric.
                difference = (
                    anchor - final
                    if metric in {"nll", "brier"}
                    else final - anchor
                )
                seed_differences[metric].append(difference)
        for metric in metrics:
            draws[metric].append(mean(seed_differences[metric]))

    output: dict[str, Any] = {}
    for metric, values in draws.items():
        array = np.asarray(values, dtype=np.float64)
        output[metric] = {
            "mean_advantage": float(array.mean()),
            "ci95": [
                float(np.quantile(array, 0.025)),
                float(np.quantile(array, 0.975)),
            ],
            "probability_positive": float((array > 0.0).mean()),
            "samples": samples,
        }
    return output


def _aggregate_test_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for variant in ("anchor", "uncalibrated", "final"):
        output[variant] = {
            metric: _summary(
                [
                    float(_nested(run[variant], path))
                    for run in runs
                ]
            )
            for metric, path in PRIMARY_METRICS.items()
        }
    output["final_minus_anchor"] = {
        metric: _summary(
            [
                (
                    float(_nested(run["anchor"], path))
                    - float(_nested(run["final"], path))
                    if metric in {"nll", "brier", "ece"}
                    else float(_nested(run["final"], path))
                    - float(_nested(run["anchor"], path))
                )
                for run in runs
            ]
        )
        for metric, path in PRIMARY_METRICS.items()
    }
    reports = [run["final"]["classification_report"] for run in runs]
    anchor_reports = [
        run["anchor"]["classification_report"] for run in runs
    ]
    excluded = {"accuracy", "macro avg", "weighted avg"}
    emotions = [
        emotion
        for emotion, values in reports[0].items()
        if emotion not in excluded and isinstance(values, dict)
    ]
    output["per_emotion_final_minus_anchor_f1"] = {
        emotion: _summary(
            [
                float(final[emotion]["f1-score"])
                - float(anchor[emotion]["f1-score"])
                for final, anchor in zip(reports, anchor_reports)
            ]
        )
        for emotion in emotions
    }
    return output


def _check_final_inputs(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    validation_file = root / "reports" / "validation_summary.json"
    calibration_file = root / "validation" / "calibration.json"
    if not validation_file.exists() or not calibration_file.exists():
        raise FileNotFoundError(
            "Run Experiment 9 validation and calibration before final test"
        )
    validation = _read_json(validation_file)
    calibration = _read_json(calibration_file)
    if (
        validation.get("status") != "completed"
        or validation.get("frozen_for_test") is not True
        or validation.get("protocol", {}).get("test_used") is not False
        or calibration.get("test_locked") is not True
    ):
        raise ValueError("Experiment 9 validation lineage is not frozen")
    inputs: list[dict[str, Any]] = []
    calibrators = {
        int(run["seed"]): Path(run["selected_calibrator"])
        for run in calibration["runs"]
    }
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed, variant="final")
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        feature_file = Path(seed_config["data"]["feature_file"])
        calibrator = calibrators[seed]
        for path in (checkpoint, feature_file, calibrator):
            if not path.exists():
                raise FileNotFoundError(f"Missing frozen input: {path}")
        inputs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "feature_file": str(feature_file),
                "feature_sha256": _sha256(feature_file),
                "calibrator": str(calibrator),
                "calibrator_sha256": _sha256(calibrator),
            }
        )
    return {
        "status": "ready",
        "dataset": _dataset_name(config),
        "seeds": _seeds(config),
        "frozen_variant": VARIANTS["final"],
        "selected_calibration": calibration["selected_method"],
        "validation_summary": str(validation_file),
        "validation_summary_sha256": _sha256(validation_file),
        "calibration_summary": str(calibration_file),
        "calibration_summary_sha256": _sha256(calibration_file),
        "inputs": inputs,
        "test_read": False,
    }


def _final_test(
    config: dict[str, Any],
    device: str | None,
    *,
    confirm: bool,
    resume: bool,
) -> dict[str, Any]:
    if not confirm:
        raise ValueError(
            "Experiment 9 final test requires explicit confirmation"
        )
    ready = _check_final_inputs(config)
    root = _artifact_dir(config)
    report_file = root / "reports" / "final_test.json"
    lock_file = root / "locks" / "final_test.lock.json"
    if report_file.exists():
        raise FileExistsError(
            f"Final test is already complete: {report_file}"
        )
    if lock_file.exists() and not resume:
        raise FileExistsError(
            f"Final-test lock exists: {lock_file}; use resume"
        )
    lock = {
        "status": "running",
        "created_at": _utc_now(),
        "protocol": ready,
    }
    if lock_file.exists():
        lock = _read_json(lock_file)
        if lock.get("protocol") != ready:
            raise ValueError(
                "Existing final-test lock has different frozen inputs"
            )
        lock["status"] = "resuming"
        lock["resumed_at"] = _utc_now()
    _atomic_json(lock, lock_file)
    calibration = _read_json(root / "validation" / "calibration.json")
    calibrators = {
        int(run["seed"]): Path(run["selected_calibrator"])
        for run in calibration["runs"]
    }
    runs: list[dict[str, Any]] = []
    try:
        for seed in _seeds(config):
            individual = root / "test" / f"seed{seed}.json"
            if individual.exists():
                if not resume:
                    raise FileExistsError(
                        f"Unexpected test result: {individual}"
                    )
                run = _read_json(individual)
                expected_checkpoint = Path(
                    _seed_config(
                        config, seed, variant="final"
                    )["train_erc"]["checkpoint"]
                )
                if (
                    int(run.get("seed", -1)) != seed
                    or Path(run.get("checkpoint", ""))
                    != expected_checkpoint
                    or Path(run.get("calibrator", ""))
                    != calibrators[seed]
                    or run.get("split") != "test"
                ):
                    raise ValueError(
                        f"Existing seed-{seed} test result has wrong lineage"
                    )
            else:
                run = _test_seed(
                    config,
                    seed,
                    device,
                    calibrators[seed],
                )
                save_json(run, individual)
            if int(run["argmax_changes_from_calibration"]) != 0:
                raise AssertionError(
                    "Positive temperature changed a class prediction"
                )
            runs.append(run)
            lock["completed_seeds"] = [
                int(item["seed"]) for item in runs
            ]
            _atomic_json(lock, lock_file)
        bootstrap_samples = int(
            _experiment(config).get("bootstrap_samples", 2000)
        )
        bootstrap_seed = int(
            _experiment(config).get("bootstrap_seed", 2029)
        )
        result = {
            "status": "completed",
            "completed_at": _utc_now(),
            "protocol": {
                **ready,
                "test_read": True,
                "one_time_test": True,
                "selection_after_test": False,
                "stable_transition_used": False,
                "bootstrap_unit": "dialogue nested within model seed",
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": bootstrap_seed,
            },
            "aggregate": _aggregate_test_runs(runs),
            "paired_hierarchical_bootstrap": _hierarchical_bootstrap(
                runs, bootstrap_samples, bootstrap_seed
            ),
            "runs": runs,
        }
        save_json(result, report_file)
        lock["status"] = "completed"
        lock["completed_at"] = result["completed_at"]
        lock["report_file"] = str(report_file)
        lock["report_sha256"] = _sha256(report_file)
        _atomic_json(lock, lock_file)
        return result
    except Exception as error:
        lock["status"] = "interrupted_or_failed"
        lock["updated_at"] = _utc_now()
        lock["error_type"] = type(error).__name__
        lock["error"] = str(error)
        _atomic_json(lock, lock_file)
        raise


def _efficiency_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    root = _artifact_dir(config)
    device_obj = get_device(device)
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed, variant="final")
        store = load_feature_store(seed_config["data"]["feature_file"])
        model = _build_model(
            seed_config, len(store["label_to_id"]), device_obj
        )
        total_parameters = sum(
            parameter.numel() for parameter in model.parameters()
        )
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        del model
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()
        if device_obj.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device_obj)
            torch.cuda.synchronize(device_obj)
        started = time.perf_counter()
        records, _, _ = _collect_dialogue_outputs(
            seed_config, checkpoint, "validation", device_obj
        )
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        elapsed = time.perf_counter() - started
        utterances = sum(
            int(record["labels"].numel()) for record in records
        )
        runs.append(
            {
                "seed": seed,
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "validation_dialogues": len(records),
                "validation_utterances": utterances,
                "cold_start_evaluation_seconds": elapsed,
                "utterances_per_second": utterances / max(elapsed, 1e-9),
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated(device_obj))
                    if device_obj.type == "cuda"
                    else None
                ),
            }
        )
    numeric = (
        "total_parameters",
        "trainable_parameters",
        "checkpoint_bytes",
        "cold_start_evaluation_seconds",
        "utterances_per_second",
    )
    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "test_used": False,
            "timing_scope": (
                "cold checkpoint/model load plus validation inference"
            ),
        },
        "runs": runs,
        "aggregate": {
            key: _summary([float(run[key]) for run in runs])
            for key in numeric
        },
    }
    save_json(payload, root / "reports" / "efficiency.json")
    return payload


def run_experiment9(
    config: dict[str, Any],
    *,
    stage: str,
    device: str | None = None,
    confirm_final_test: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the pre-registered cross-dataset Experiment 9 protocol.

    No stage named ``all`` includes the test split. This prevents an
    accidental test read while running validation experiments.
    """
    _seeds(config)
    _artifact_dir(config).mkdir(parents=True, exist_ok=True)
    if stage == "check":
        return {
            "status": "ready",
            "dataset": _dataset_name(config),
            "seeds": _seeds(config),
            "artifact_dir": str(_artifact_dir(config)),
            "variants": VARIANTS,
            "test_read": False,
        }
    if stage == "prepare":
        return prepare_dataset(config)
    if stage == "anchors":
        return _anchor_stage(config, device)
    if stage == "features":
        return _feature_stage(config, device)
    if stage == "final-validation":
        return _train_variant_stage(config, device, "final")
    if stage in {"static", "uniform", "wrong-recipient"}:
        return _train_variant_stage(
            config, device, stage.replace("-", "_")
        )
    if stage == "ablations":
        return {
            name: _train_variant_stage(config, device, name)
            for name in ("static", "uniform", "wrong_recipient")
        }
    if stage == "calibrate":
        return _calibration_stage(config, device)
    if stage == "summarize-validation":
        return _validation_summary(config)
    if stage == "all-validation":
        prepare_dataset(config)
        _anchor_stage(config, device)
        _feature_stage(config, device)
        _train_variant_stage(config, device, "final")
        for variant in ("static", "uniform", "wrong_recipient"):
            _train_variant_stage(config, device, variant)
        _calibration_stage(config, device)
        return _validation_summary(config)
    if stage == "check-final":
        return _check_final_inputs(config)
    if stage == "final-test":
        return _final_test(
            config,
            device,
            confirm=confirm_final_test,
            resume=resume,
        )
    if stage == "efficiency":
        return _efficiency_stage(config, device)
    if stage == "summarize-final":
        path = _artifact_dir(config) / "reports" / "final_test.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing final report: {path}")
        return _read_json(path)
    raise ValueError(f"Unknown Experiment 9 stage: {stage}")
