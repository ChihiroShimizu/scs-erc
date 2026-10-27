from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

from .calibration import (
    _collect_dialogue_outputs,
    _concatenate,
    _cross_fitted_calibration,
    _fit_temperature,
    _metrics_from_logits,
    _temperature_from_parameters,
)
from .common import get_device, save_json
from .experiment9 import (
    _hierarchical_bootstrap,
    _identity_parameters,
    _seed_config,
    _sha256,
)


METRIC_PATHS: dict[str, tuple[str, ...]] = {
    "weighted_f1": ("weighted_f1",),
    "macro_f1": ("macro_f1",),
    "accuracy": ("accuracy",),
    "nll": ("probability_quality", "nll"),
    "brier": ("probability_quality", "brier"),
    "ece": ("probability_quality", "ece"),
}
LOWER_IS_BETTER = {"nll", "brier", "ece"}


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("experiment11")
    if not isinstance(settings, dict):
        raise KeyError("Config is missing the experiment11 section")
    return settings


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(seed) for seed in _settings(config).get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "experiment11.seeds must contain at least three distinct seeds"
        )
    return seeds


def _output_dir(config: dict[str, Any]) -> Path:
    return Path(_settings(config)["artifact_dir"])


def _source_report(config: dict[str, Any]) -> Path:
    return Path(_settings(config)["source_final_report"])


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values) if values else None,
        "std": pstdev(values) if values else None,
        "count": len(values),
    }


def _source_records(
    records: list[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    if source not in {"anchor", "final"}:
        raise ValueError(f"Unknown logit source: {source}")
    converted: list[dict[str, Any]] = []
    for record in records:
        logits = (
            record["anchor_logits"]
            if source == "anchor"
            else record["logits"]
        )
        converted.append(
            {
                "dialogue_id": record["dialogue_id"],
                "logits": logits,
                "labels": record["labels"],
                # Experiment 11 deliberately uses the same scalar-only
                # calibration family for both sources. The gate is therefore
                # an inert placeholder, not an extra feature.
                "gate": torch.zeros(
                    record["labels"].numel(), dtype=torch.float32
                ),
            }
        )
    return converted


def _selection(
    provisional: list[dict[str, Any]], source: str
) -> tuple[str, dict[str, Any]]:
    improvements = [
        float(run[source]["raw"]["probability_quality"]["nll"])
        - float(
            run[source]["scalar_cross_fitted"]["metrics"][
                "probability_quality"
            ]["nll"]
        )
        for run in provisional
    ]
    selected = (
        "scalar"
        if mean(improvements) > 0.0
        and sum(value > 0.0 for value in improvements) >= 2
        else "identity"
    )
    return selected, {
        "selected": selected,
        "cross_fitted_nll_improvement": _summary(improvements),
        "positive_seeds": sum(value > 0.0 for value in improvements),
        "rule": (
            "Select scalar temperature only when mean dialogue-cross-fitted "
            "NLL improves and at least two model seeds improve; otherwise "
            "select identity temperature."
        ),
    }


def _metric_summary(
    runs: list[dict[str, Any]], source: str, variant: str
) -> dict[str, Any]:
    return {
        metric: _summary(
            [
                _nested(run[source][variant], path)
                for run in runs
            ]
        )
        for metric, path in METRIC_PATHS.items()
    }


def _advantage(
    left: dict[str, Any],
    right: dict[str, Any],
    metric: str,
) -> float:
    left_value = _nested(left, METRIC_PATHS[metric])
    right_value = _nested(right, METRIC_PATHS[metric])
    if metric in LOWER_IS_BETTER:
        return left_value - right_value
    return right_value - left_value


def _aggregate_validation(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in ("anchor", "final"):
        output[source] = {
            "raw": _metric_summary(runs, source, "raw"),
            "selected_cross_fitted": _metric_summary(
                runs, source, "selected_cross_fitted"
            ),
        }
    output["calibrated_final_minus_calibrated_anchor"] = {
        metric: _summary(
            [
                _advantage(
                    run["anchor"]["selected_cross_fitted"],
                    run["final"]["selected_cross_fitted"],
                    metric,
                )
                for run in runs
            ]
        )
        for metric in METRIC_PATHS
    }
    output["raw_final_minus_raw_anchor"] = {
        metric: _summary(
            [
                _advantage(
                    run["anchor"]["raw"],
                    run["final"]["raw"],
                    metric,
                )
                for run in runs
            ]
        )
        for metric in METRIC_PATHS
    }
    return output


def _validation_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    settings = _settings(config)
    output_dir = _output_dir(config)
    folds = int(settings.get("folds", 5))
    fold_seed = int(settings.get("fold_seed", 2026))
    minimum = float(settings.get("minimum_temperature", 0.25))
    maximum = float(settings.get("maximum_temperature", 4.0))
    max_iter = int(settings.get("max_iter", 100))
    provisional: list[dict[str, Any]] = []
    device_obj = get_device(device)

    for seed in _seeds(config):
        seed_config = _seed_config(
            config, seed, variant="final", split="validation"
        )
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing frozen final checkpoint: {checkpoint}"
            )
        records, id_to_label, stored_seed = _collect_dialogue_outputs(
            seed_config, checkpoint, "validation", device_obj
        )
        if stored_seed != seed:
            raise ValueError(
                f"Checkpoint {checkpoint} stores seed={stored_seed}, "
                f"expected {seed}"
            )
        seed_run: dict[str, Any] = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "dialogues": len(records),
        }
        for source in ("anchor", "final"):
            source_records = _source_records(records, source)
            logits, labels, gate = _concatenate(
                source_records, list(range(len(source_records)))
            )
            raw = _metrics_from_logits(logits, labels, id_to_label)
            cross_fitted = _cross_fitted_calibration(
                records=source_records,
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
            seed_run[source] = {
                "utterances": int(labels.numel()),
                "raw": raw,
                "scalar_cross_fitted": cross_fitted,
                "full_parameters": full_parameters,
            }
        provisional.append(seed_run)

    selections = {
        source: _selection(provisional, source)
        for source in ("anchor", "final")
    }
    calibrator_dir = output_dir / "calibrators"
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for provisional_run in provisional:
        run = {
            key: value
            for key, value in provisional_run.items()
            if key not in {"anchor", "final"}
        }
        for source in ("anchor", "final"):
            selected = selections[source][0]
            source_payload = provisional_run[source]
            parameters = (
                source_payload["full_parameters"]
                if selected == "scalar"
                else _identity_parameters(minimum, maximum)
            )
            calibrator_file = (
                calibrator_dir
                / f"{source}_{selected}_seed{run['seed']}.json"
            )
            calibrator = {
                "format_version": "pfa-erc-calibration-attribution-v1",
                "method": "scalar",
                "selection": selected,
                "logit_source": source,
                "model_seed": int(run["seed"]),
                "source_checkpoint": run["checkpoint"],
                "fit_split": "validation",
                "dataset": str(config["data"]["dataset"]).lower(),
                "parameters": parameters,
            }
            save_json(calibrator, calibrator_file)
            selected_metrics = (
                source_payload["scalar_cross_fitted"]["metrics"]
                if selected == "scalar"
                else source_payload["raw"]
            )
            run[source] = {
                "raw": source_payload["raw"],
                "scalar_cross_fitted": source_payload[
                    "scalar_cross_fitted"
                ],
                "selected_method": selected,
                "selected_cross_fitted": selected_metrics,
                "calibrator": str(calibrator_file),
                "calibrator_sha256": _sha256(calibrator_file),
            }
        runs.append(run)

    payload = {
        "status": "completed",
        "experiment": "11_fair_calibration_attribution",
        "protocol": {
            "dataset": str(config["data"]["dataset"]).lower(),
            "split": "validation",
            "seeds": _seeds(config),
            "folds": folds,
            "fold_unit": "dialogue",
            "fold_seed": fold_seed,
            "same_calibrator_family": True,
            "same_selection_rule": True,
            "model_retrained": False,
            "classification_boundary_changed": False,
            "test_used": False,
        },
        "selection": {
            source: details
            for source, (_, details) in selections.items()
        },
        "aggregate": _aggregate_validation(runs),
        "runs": runs,
        "test_locked": True,
    }
    report_file = output_dir / "validation_attribution.json"
    save_json(payload, report_file)
    return payload


def _probabilities_to_logits(
    probabilities: torch.Tensor,
) -> torch.Tensor:
    probabilities = probabilities.float()
    if probabilities.ndim != 2:
        raise ValueError("Probabilities must have shape [N, C]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probabilities contain non-finite values")
    if torch.any(probabilities < 0.0):
        raise ValueError("Probabilities must be non-negative")
    normalizer = probabilities.sum(dim=-1, keepdim=True)
    if torch.any(normalizer <= 0.0):
        raise ValueError("Every probability row must have positive mass")
    probabilities = probabilities / normalizer
    return probabilities.clamp_min(1e-12).log()


def _load_calibrator(
    path: Path, source: str, seed: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("format_version")
        != "pfa-erc-calibration-attribution-v1"
        or payload.get("logit_source") != source
        or int(payload.get("model_seed", -1)) != seed
        or payload.get("fit_split") != "validation"
    ):
        raise ValueError(f"Invalid Experiment 11 calibrator: {path}")
    return payload


def _apply_scalar(
    logits: torch.Tensor, calibrator: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    gate = torch.zeros(logits.shape[0], dtype=torch.float32)
    temperature = _temperature_from_parameters(
        gate, calibrator["parameters"]
    )
    return logits / temperature.unsqueeze(-1), temperature


def _metrics_for_prediction_records(
    records: list[dict[str, Any]],
    id_to_label: dict[int, str],
    key: str,
) -> dict[str, Any]:
    labels = torch.tensor(
        [
            int(label)
            for record in records
            for label in record["labels"]
        ],
        dtype=torch.long,
    )
    probabilities = torch.tensor(
        [
            row
            for record in records
            for row in record[key]
        ],
        dtype=torch.float32,
    )
    return _metrics_from_logits(
        _probabilities_to_logits(probabilities),
        labels,
        id_to_label,
    )


def _test_run(
    run: dict[str, Any],
    validation_run: dict[str, Any],
) -> dict[str, Any]:
    seed = int(run["seed"])
    id_to_label = {
        int(key): value for key, value in run["id_to_label"].items()
    }
    anchor_calibrator_path = Path(
        validation_run["anchor"]["calibrator"]
    )
    final_calibrator_path = Path(validation_run["final"]["calibrator"])
    anchor_calibrator = _load_calibrator(
        anchor_calibrator_path, "anchor", seed
    )
    final_calibrator = _load_calibrator(
        final_calibrator_path, "final", seed
    )
    old_final_temperature = float(run.get("temperature_mean", 1.0))
    if old_final_temperature <= 0.0:
        raise ValueError("Stored final temperature must be positive")

    output_records: list[dict[str, Any]] = []
    anchor_temperatures: list[torch.Tensor] = []
    final_temperatures: list[torch.Tensor] = []
    for record in run["predictions"]:
        labels = [int(label) for label in record["labels"]]
        anchor_raw_logits = _probabilities_to_logits(
            torch.tensor(record["anchor_probabilities"])
        )
        # Experiment 9 stores probabilities after its selected scalar
        # calibration. For a scalar T_old,
        # log softmax(z / T_old) * T_old is equivalent to z up to a
        # per-row additive constant, which leaves every softmax unchanged.
        final_raw_logits = (
            _probabilities_to_logits(
                torch.tensor(record["final_probabilities"])
            )
            * old_final_temperature
        )
        anchor_calibrated_logits, anchor_temperature = _apply_scalar(
            anchor_raw_logits, anchor_calibrator
        )
        final_calibrated_logits, final_temperature = _apply_scalar(
            final_raw_logits, final_calibrator
        )
        anchor_temperatures.append(anchor_temperature)
        final_temperatures.append(final_temperature)
        output_records.append(
            {
                "dialogue_id": str(record["dialogue_id"]),
                "labels": labels,
                "anchor_raw_probabilities": torch.softmax(
                    anchor_raw_logits, dim=-1
                ).tolist(),
                "anchor_calibrated_probabilities": torch.softmax(
                    anchor_calibrated_logits, dim=-1
                ).tolist(),
                "final_raw_probabilities": torch.softmax(
                    final_raw_logits, dim=-1
                ).tolist(),
                "final_calibrated_probabilities": torch.softmax(
                    final_calibrated_logits, dim=-1
                ).tolist(),
                # Compatibility keys for the existing hierarchical bootstrap.
                "anchor_probabilities": torch.softmax(
                    anchor_calibrated_logits, dim=-1
                ).tolist(),
                "final_probabilities": torch.softmax(
                    final_calibrated_logits, dim=-1
                ).tolist(),
            }
        )

    metrics = {
        key: _metrics_for_prediction_records(
            output_records, id_to_label, f"{key}_probabilities"
        )
        for key in (
            "anchor_raw",
            "anchor_calibrated",
            "final_raw",
            "final_calibrated",
        )
    }
    return {
        "seed": seed,
        "source_test_report": run.get("checkpoint"),
        "source_old_final_temperature": old_final_temperature,
        "anchor_calibrator": str(anchor_calibrator_path),
        "anchor_calibrator_sha256": _sha256(anchor_calibrator_path),
        "final_calibrator": str(final_calibrator_path),
        "final_calibrator_sha256": _sha256(final_calibrator_path),
        "anchor_temperature_mean": float(
            torch.cat(anchor_temperatures).mean()
        ),
        "final_temperature_mean": float(
            torch.cat(final_temperatures).mean()
        ),
        **metrics,
        "id_to_label": {str(key): value for key, value in id_to_label.items()},
        "predictions": output_records,
    }


def _aggregate_test(runs: list[dict[str, Any]]) -> dict[str, Any]:
    variants = (
        "anchor_raw",
        "anchor_calibrated",
        "final_raw",
        "final_calibrated",
    )
    aggregate = {
        variant: {
            metric: _summary(
                [
                    _nested(run[variant], path)
                    for run in runs
                ]
            )
            for metric, path in METRIC_PATHS.items()
        }
        for variant in variants
    }
    contrasts = {
        "raw_mechanism": ("anchor_raw", "final_raw"),
        "anchor_calibration": ("anchor_raw", "anchor_calibrated"),
        "final_calibration": ("final_raw", "final_calibrated"),
        "calibrated_mechanism": (
            "anchor_calibrated",
            "final_calibrated",
        ),
    }
    aggregate["attribution"] = {
        name: {
            metric: _summary(
                [
                    _advantage(run[left], run[right], metric)
                    for run in runs
                ]
            )
            for metric in METRIC_PATHS
        }
        for name, (left, right) in contrasts.items()
    }
    return aggregate


def _check_final(config: dict[str, Any]) -> dict[str, Any]:
    output_dir = _output_dir(config)
    validation_file = output_dir / "validation_attribution.json"
    source_report = _source_report(config)
    if not validation_file.exists():
        raise FileNotFoundError(
            "Run Experiment 11 validation before final attribution"
        )
    if not source_report.exists():
        raise FileNotFoundError(
            f"Missing already-frozen source test report: {source_report}"
        )
    validation = json.loads(validation_file.read_text(encoding="utf-8"))
    source = json.loads(source_report.read_text(encoding="utf-8"))
    if (
        validation.get("status") != "completed"
        or validation.get("test_locked") is not True
        or validation.get("protocol", {}).get("test_used") is not False
    ):
        raise ValueError("Experiment 11 validation is not frozen")
    if source.get("status") != "completed":
        raise ValueError("Source final-test report is not completed")
    source_seeds = sorted(int(run["seed"]) for run in source["runs"])
    if source_seeds != sorted(_seeds(config)):
        raise ValueError(
            f"Source test seeds {source_seeds} do not match "
            f"Experiment 11 seeds {_seeds(config)}"
        )
    for run in source["runs"]:
        if "predictions" not in run:
            raise ValueError(
                "Source report lacks stored dialogue predictions; "
                "Experiment 11 refuses to rerun or reread the test dataset"
            )
    return {
        "status": "ready",
        "dataset": str(config["data"]["dataset"]).lower(),
        "seeds": _seeds(config),
        "validation_report": str(validation_file),
        "validation_sha256": _sha256(validation_file),
        "source_frozen_test_report": str(source_report),
        "source_frozen_test_sha256": _sha256(source_report),
        "test_dataset_will_be_reread": False,
        "stored_predictions_only": True,
        "post_hoc_sensitivity_analysis": True,
    }


def _final_attribution(
    config: dict[str, Any], *, confirm: bool
) -> dict[str, Any]:
    if not confirm:
        raise ValueError(
            "Experiment 11 requires explicit confirmation before consuming "
            "the stored frozen-test predictions"
        )
    manifest = _check_final(config)
    output_dir = _output_dir(config)
    report_file = output_dir / "final_attribution.json"
    if report_file.exists():
        raise FileExistsError(
            f"Experiment 11 final attribution already exists: {report_file}"
        )
    validation = json.loads(
        (output_dir / "validation_attribution.json").read_text(
            encoding="utf-8"
        )
    )
    source = json.loads(_source_report(config).read_text(encoding="utf-8"))
    validation_runs = {
        int(run["seed"]): run for run in validation["runs"]
    }
    runs = [
        _test_run(run, validation_runs[int(run["seed"])])
        for run in source["runs"]
    ]
    settings = _settings(config)
    bootstrap_samples = int(settings.get("bootstrap_samples", 2000))
    bootstrap_seed = int(settings.get("bootstrap_seed", 2031))
    payload = {
        "status": "completed",
        "experiment": "11_fair_calibration_attribution",
        "protocol": {
            **manifest,
            "analysis_type": "post_hoc_sensitivity_analysis",
            "model_or_calibrator_selected_from_test": False,
            "test_dataset_reread": False,
            "classification_argmax_changed": False,
            "bootstrap_unit": "dialogue nested within model seed",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "aggregate": _aggregate_test(runs),
        "paired_hierarchical_bootstrap": _hierarchical_bootstrap(
            runs, bootstrap_samples, bootstrap_seed
        ),
        "runs": runs,
    }
    save_json(payload, report_file)
    return payload


def run_calibration_attribution(
    config: dict[str, Any],
    *,
    stage: str,
    device: str | None = None,
    confirm_posthoc_test: bool = False,
) -> dict[str, Any]:
    _seeds(config)
    _output_dir(config).mkdir(parents=True, exist_ok=True)
    if stage == "check":
        return {
            "status": "ready",
            "experiment": "11_fair_calibration_attribution",
            "dataset": str(config["data"]["dataset"]).lower(),
            "seeds": _seeds(config),
            "source_artifact_dir": str(
                Path(config["experiment9"]["artifact_dir"])
            ),
            "output_dir": str(_output_dir(config)),
            "source_final_report": str(_source_report(config)),
            "test_used": False,
        }
    if stage == "validation":
        return _validation_stage(config, device)
    if stage == "check-final":
        return _check_final(config)
    if stage == "final-attribution":
        return _final_attribution(
            config, confirm=confirm_posthoc_test
        )
    if stage == "summarize":
        final = _output_dir(config) / "final_attribution.json"
        validation = _output_dir(config) / "validation_attribution.json"
        target = final if final.exists() else validation
        if not target.exists():
            raise FileNotFoundError(
                "No Experiment 11 report exists. Run validation first."
            )
        return json.loads(target.read_text(encoding="utf-8"))
    raise ValueError(f"Unknown Experiment 11 stage: {stage}")
