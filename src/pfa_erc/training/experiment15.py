from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .calibration import (
    _concatenate,
    _dialogue_folds,
    _fit_temperature,
    _metrics_from_logits,
    _temperature_from_parameters,
    _temperature_summary,
)
from .common import get_device, save_json
from .erc_trainer import _probability_quality
from .experiment14 import (
    METRIC_PATHS,
    MODEL_ORDER,
    VARIANTS,
    _anchor_records,
    _collect_outputs,
    _seed_config,
)

CONTROLS = (
    "capacity_direct",
    "contextual",
    "speaker_aware",
    "anchor",
)
LOWER_IS_BETTER = {"nll", "brier", "ece"}
CALIBRATION_STATES = ("raw", "cross_fitted_calibrated")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


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


def _settings(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("experiment15")
    if not isinstance(section, dict):
        raise KeyError("Config is missing the experiment15 section")
    return section


def _artifact_dir(config: dict[str, Any]) -> Path:
    return Path(_settings(config)["artifact_dir"])


def _source_dir(config: dict[str, Any]) -> Path:
    return Path(_settings(config)["source_experiment14_dir"])


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(seed) for seed in _settings(config).get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "experiment15.seeds must contain at least three distinct seeds"
        )
    source_seeds = [
        int(seed)
        for seed in config.get("experiment14", {}).get("seeds", [])
    ]
    if seeds != source_seeds:
        raise ValueError(
            "experiment15.seeds must exactly match experiment14.seeds, "
            f"got {seeds} versus {source_seeds}"
        )
    return seeds


def _metric_value(payload: dict[str, Any], metric: str) -> float:
    value = _nested(payload, METRIC_PATHS[metric])
    if value is None:
        raise KeyError(f"Missing metric {metric}")
    return float(value)


def _advantage(reference: float, control: float, metric: str) -> float:
    return (
        control - reference
        if metric in LOWER_IS_BETTER
        else reference - control
    )


def _check_source_inputs(config: dict[str, Any]) -> dict[str, Any]:
    if str(config["data"]["dataset"]).lower() != "meld":
        raise ValueError("Experiment 15 is the MELD development-only analysis")
    if str(config["evaluation"].get("split", "validation")) != "validation":
        raise ValueError("Experiment 15 may only read the validation split")

    source = _source_dir(config)
    expected_source = Path(config["experiment14"]["artifact_dir"])
    if source.resolve() != expected_source.resolve():
        raise ValueError(
            "experiment15.source_experiment14_dir must match the configured "
            "experiment14 artifact_dir"
        )
    required = [
        source / "reports" / "validation_summary.json",
        source / "reports" / "parameter_audit.json",
        source / "validation" / "anchor_seeds.json",
        *[
            source / "validation" / f"{variant}_seeds.json"
            for variant in VARIANTS
        ],
        *[
            source / "features" / f"seed{seed}.pt"
            for seed in _seeds(config)
        ],
        *[
            source / "checkpoints" / f"{variant}_seed{seed}.pt"
            for seed in _seeds(config)
            for variant in VARIANTS
        ],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experiment 14 validation inputs are incomplete: "
            + ", ".join(missing)
        )

    parameter_audit = _read_json(
        source / "reports" / "parameter_audit.json"
    )
    capacity = parameter_audit.get("capacity_match", {})
    exact_match = all(
        bool(capacity.get(key))
        for key in (
            "exact_total_parameters",
            "exact_trainable_parameters",
            "exact_active_parameter_signature",
        )
    )
    if parameter_audit.get("status") != "passed" or not exact_match:
        raise ValueError("Experiment 14 capacity audit is not valid")

    source_summary = _read_json(
        source / "reports" / "validation_summary.json"
    )
    summary_seeds = [
        int(seed)
        for seed in source_summary.get("protocol", {}).get("seeds", [])
    ]
    if summary_seeds != _seeds(config):
        raise ValueError(
            "Source validation summary seed list does not match Experiment 15"
        )

    manifest = {
        "status": "ready",
        "experiment": 15,
        "dataset": "meld",
        "split": "validation",
        "seeds": _seeds(config),
        "models": list(MODEL_ORDER),
        "source_experiment14_dir": str(source),
        "source_validation_summary": str(
            source / "reports" / "validation_summary.json"
        ),
        "source_validation_summary_sha256": _sha256(
            source / "reports" / "validation_summary.json"
        ),
        "exact_capacity_match": True,
        "analysis_only": True,
        "retraining": False,
        "test_read": False,
        "test_locked": True,
    }
    save_json(manifest, _artifact_dir(config) / "reports" / "input_audit.json")
    return manifest


def _assert_paired_records(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    model_name: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(
            f"{model_name} has {len(candidate)} dialogues; expected "
            f"{len(reference)}"
        )
    seen: set[str] = set()
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left_id = str(left["dialogue_id"])
        right_id = str(right["dialogue_id"])
        if left_id in seen:
            raise ValueError(f"Duplicate dialogue id: {left_id}")
        seen.add(left_id)
        if left_id != right_id:
            raise ValueError(
                f"Dialogue order mismatch for {model_name} at {index}: "
                f"{left_id} versus {right_id}"
            )
        if not torch.equal(left["labels"], right["labels"]):
            raise ValueError(
                f"Label mismatch for {model_name} dialogue {left_id}"
            )


def _cross_fit_model(
    records: list[dict[str, Any]],
    id_to_label: dict[int, str],
    assignments: list[list[int]],
    *,
    minimum: float,
    maximum: float,
    max_iter: int,
) -> tuple[dict[str, Any], list[list[list[float]]], list[list[list[float]]]]:
    all_indices = set(range(len(records)))
    temperatures: list[torch.Tensor | None] = [None] * len(records)
    fold_reports: list[dict[str, Any]] = []
    for fold_index, held_out in enumerate(assignments):
        training = sorted(all_indices - set(held_out))
        train_logits, train_labels, train_gate = _concatenate(
            records, training
        )
        parameters = _fit_temperature(
            train_logits,
            train_labels,
            train_gate,
            method="scalar",
            minimum=minimum,
            maximum=maximum,
            max_iter=max_iter,
            slope_l2=0.0,
        )
        held_logits, held_labels, held_gate = _concatenate(records, held_out)
        held_temperature = _temperature_from_parameters(
            held_gate.float(), parameters
        )
        held_metrics = _metrics_from_logits(
            held_logits,
            held_labels,
            id_to_label,
            held_temperature,
        )
        offset = 0
        for record_index in held_out:
            length = int(records[record_index]["labels"].numel())
            temperatures[record_index] = held_temperature[
                offset : offset + length
            ]
            offset += length
        fold_reports.append(
            {
                "fold": fold_index,
                "fit_dialogues": len(training),
                "held_out_dialogues": len(held_out),
                "held_out_utterances": int(held_labels.numel()),
                "held_out_dialogue_ids": [
                    str(records[index]["dialogue_id"])
                    for index in held_out
                ],
                "fit_parameters": parameters,
                "held_out_probability_quality": held_metrics[
                    "probability_quality"
                ],
            }
        )
    if any(value is None for value in temperatures):
        raise AssertionError("Every validation dialogue must be held out once")

    indices = list(range(len(records)))
    logits, labels, gate = _concatenate(records, indices)
    joined_temperature = torch.cat(
        [value for value in temperatures if value is not None], dim=0
    )
    calibrated_logits = logits / joined_temperature.unsqueeze(-1)
    argmax_changes = int(
        logits.argmax(dim=-1)
        .ne(calibrated_logits.argmax(dim=-1))
        .sum()
    )
    if argmax_changes != 0:
        raise AssertionError("Positive scalar temperature changed an argmax")

    deployment_parameters = _fit_temperature(
        logits,
        labels,
        gate,
        method="scalar",
        minimum=minimum,
        maximum=maximum,
        max_iter=max_iter,
        slope_l2=0.0,
    )
    raw_metrics = _metrics_from_logits(logits, labels, id_to_label)
    calibrated_metrics = _metrics_from_logits(
        logits,
        labels,
        id_to_label,
        joined_temperature,
    )

    raw_probabilities: list[list[list[float]]] = []
    calibrated_probabilities: list[list[list[float]]] = []
    for index, record in enumerate(records):
        record_logits = record["logits"].float()
        record_temperature = temperatures[index]
        assert record_temperature is not None
        raw_probabilities.append(
            torch.softmax(record_logits, dim=-1).tolist()
        )
        calibrated_probabilities.append(
            torch.softmax(
                record_logits / record_temperature.unsqueeze(-1),
                dim=-1,
            ).tolist()
        )

    report = {
        "raw": raw_metrics,
        "cross_fitted_calibrated": calibrated_metrics,
        "temperature": _temperature_summary(joined_temperature),
        "cross_fitted_nll_improvement": (
            _metric_value(raw_metrics, "nll")
            - _metric_value(calibrated_metrics, "nll")
        ),
        "argmax_changes": argmax_changes,
        "folds": fold_reports,
        "deployment_parameters": deployment_parameters,
    }
    return report, raw_probabilities, calibrated_probabilities


def _collect_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    settings = _settings(config)
    calibration = settings.get("calibration", {})
    folds = int(calibration.get("folds", 5))
    fold_seed = int(calibration.get("fold_seed", 2015))
    minimum = float(calibration.get("minimum_temperature", 0.25))
    maximum = float(calibration.get("maximum_temperature", 4.0))
    max_iter = int(calibration.get("max_iter", 100))

    role_config = _seed_config(
        config, seed, variant="roletrans", split="validation"
    )
    role_checkpoint = Path(role_config["train_erc"]["checkpoint"])
    role_records, id_to_label, stored_seed = _collect_outputs(
        role_config,
        role_checkpoint,
        "validation",
        device,
        "roletrans",
    )
    if stored_seed != seed:
        raise ValueError(
            f"RoleTrans checkpoint stores seed {stored_seed}, expected {seed}"
        )
    records_by_model: dict[str, list[dict[str, Any]]] = {
        "roletrans": role_records,
        "anchor": _anchor_records(role_records),
    }
    checkpoints: dict[str, str] = {
        "anchor": str(
            Path(role_config["features"]["encoder_checkpoint"])
        ),
        "roletrans": str(role_checkpoint),
    }
    for model_name in ("contextual", "speaker_aware", "capacity_direct"):
        model_config = _seed_config(
            config, seed, variant=model_name, split="validation"
        )
        checkpoint = Path(model_config["train_erc"]["checkpoint"])
        records, labels, candidate_seed = _collect_outputs(
            model_config,
            checkpoint,
            "validation",
            device,
            model_name,
        )
        if candidate_seed != seed:
            raise ValueError(
                f"{model_name} checkpoint stores seed {candidate_seed}, "
                f"expected {seed}"
            )
        if labels != id_to_label:
            raise ValueError(f"{model_name} label map differs from RoleTrans")
        _assert_paired_records(
            role_records, records, model_name=model_name
        )
        records_by_model[model_name] = records
        checkpoints[model_name] = str(checkpoint)
    _assert_paired_records(
        role_records, records_by_model["anchor"], model_name="anchor"
    )

    assignments = _dialogue_folds(role_records, folds, fold_seed)
    held_out_ids = [
        [str(role_records[index]["dialogue_id"]) for index in fold]
        for fold in assignments
    ]
    model_reports: dict[str, Any] = {}
    probabilities: dict[
        str, tuple[list[list[list[float]]], list[list[list[float]]]]
    ] = {}
    for model_name in MODEL_ORDER:
        report, raw, calibrated = _cross_fit_model(
            records_by_model[model_name],
            id_to_label,
            assignments,
            minimum=minimum,
            maximum=maximum,
            max_iter=max_iter,
        )
        report["source_checkpoint"] = checkpoints[model_name]
        model_reports[model_name] = report
        probabilities[model_name] = (raw, calibrated)

    dialogues: list[dict[str, Any]] = []
    for index, reference in enumerate(role_records):
        dialogue = {
            "dialogue_id": str(reference["dialogue_id"]),
            "labels": [int(value) for value in reference["labels"].tolist()],
            "models": {},
        }
        for model_name in MODEL_ORDER:
            raw, calibrated = probabilities[model_name]
            dialogue["models"][model_name] = {
                "raw_probabilities": raw[index],
                "cross_fitted_calibrated_probabilities": calibrated[index],
            }
        dialogues.append(dialogue)

    root = _artifact_dir(config)
    prediction_file = (
        root / "validation" / f"predictions_seed{seed}.json"
    )
    prediction_payload = {
        "format_version": "pfa-erc-experiment15-predictions-v1",
        "experiment": 15,
        "dataset": "meld",
        "split": "validation",
        "seed": seed,
        "id_to_label": {
            str(key): value for key, value in id_to_label.items()
        },
        "dialogue_folds": held_out_ids,
        "dialogue_fold_fingerprint": _fingerprint(held_out_ids),
        "dialogues": dialogues,
        "test_read": False,
    }
    save_json(prediction_payload, prediction_file)

    calibration_file = (
        root / "validation" / f"calibration_seed{seed}.json"
    )
    calibration_payload = {
        "status": "completed",
        "experiment": 15,
        "seed": seed,
        "protocol": {
            "method": "forced symmetric scalar temperature",
            "fit_scheme": "dialogue-level cross-fitting",
            "folds": folds,
            "fold_seed": fold_seed,
            "same_folds_for_all_models": True,
            "positive_temperature_preserves_argmax": True,
            "model_selection_or_retuning": False,
            "test_read": False,
        },
        "models": model_reports,
        "prediction_file": str(prediction_file),
        "prediction_file_sha256": _sha256(prediction_file),
    }
    save_json(calibration_payload, calibration_file)
    return calibration_payload


def _load_or_collect_seed(
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    path = _artifact_dir(config) / "validation" / f"calibration_seed{seed}.json"
    reuse = bool(_settings(config).get("reuse_existing", True))
    if reuse and path.exists():
        payload = _read_json(path)
        prediction_file = Path(payload.get("prediction_file", ""))
        if (
            int(payload.get("seed", -1)) == seed
            and prediction_file.exists()
            and _sha256(prediction_file)
            == payload.get("prediction_file_sha256")
        ):
            return payload
    return _collect_seed(config, seed, device)


def _aggregate_models(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        output[model_name] = {}
        for state in CALIBRATION_STATES:
            output[model_name][state] = {
                metric: _summary(
                    [
                        _metric_value(run["models"][model_name][state], metric)
                        for run in runs
                    ]
                )
                for metric in METRIC_PATHS
            }
        output[model_name]["temperature_mean"] = _summary(
            [
                float(run["models"][model_name]["temperature"]["mean"])
                for run in runs
            ]
        )
        output[model_name]["cross_fitted_nll_improvement"] = _summary(
            [
                float(
                    run["models"][model_name][
                        "cross_fitted_nll_improvement"
                    ]
                )
                for run in runs
            ]
        )
        output[model_name]["argmax_changes"] = _summary(
            [
                float(run["models"][model_name]["argmax_changes"])
                for run in runs
            ]
        )
    return output


def _paired_seed_advantages(
    runs: list[dict[str, Any]],
    *,
    state: str,
    reference: str = "roletrans",
) -> dict[str, Any]:
    if state not in CALIBRATION_STATES:
        raise ValueError(f"Unknown calibration state: {state}")
    seed_values = [int(run["seed"]) for run in runs]
    if len(seed_values) != len(set(seed_values)):
        raise ValueError("Duplicate model seeds in paired comparison")
    output: dict[str, Any] = {}
    tolerance = 1e-12
    for control in CONTROLS:
        metrics: dict[str, Any] = {}
        for metric in METRIC_PATHS:
            per_seed: list[dict[str, Any]] = []
            values: list[float] = []
            for run in sorted(runs, key=lambda item: int(item["seed"])):
                seed = int(run["seed"])
                reference_value = _metric_value(
                    run["models"][reference][state], metric
                )
                control_value = _metric_value(
                    run["models"][control][state], metric
                )
                value = _advantage(
                    reference_value, control_value, metric
                )
                values.append(value)
                per_seed.append(
                    {
                        "seed": seed,
                        "reference": reference_value,
                        "control": control_value,
                        "advantage": value,
                    }
                )
            positive = sum(value > tolerance for value in values)
            negative = sum(value < -tolerance for value in values)
            tied = len(values) - positive - negative
            if positive + negative + tied != len(values):
                raise AssertionError("Seed direction accounting failed")
            metrics[metric] = {
                **_summary(values),
                "advantage_definition": (
                    "control-reference"
                    if metric in LOWER_IS_BETTER
                    else "reference-control"
                ),
                "higher_advantage_is_always_better": True,
                "per_seed": per_seed,
                "positive_seeds": positive,
                "negative_seeds": negative,
                "tied_seeds": tied,
                "total_seeds": len(values),
                "direction": (
                    f"+{positive}/-{negative}/={tied} of {len(values)}"
                ),
                "seed_identity_checked": True,
            }
        output[control] = {
            "reference": reference,
            "control": control,
            "calibration_state": state,
            "metrics": metrics,
        }
    return output


def _symmetric_calibration_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    audit = _check_source_inputs(config)
    device_obj = get_device(device)
    runs = [
        _load_or_collect_seed(config, seed, device_obj)
        for seed in _seeds(config)
    ]
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "experiment": 15,
        "protocol": {
            "dataset": "meld",
            "split": "validation",
            "seeds": _seeds(config),
            "models": list(MODEL_ORDER),
            "calibration": (
                "the same scalar-temperature dialogue cross-fitting "
                "procedure is forced for every model"
            ),
            "no_model_specific_identity_selection": True,
            "classification_argmax_must_remain_unchanged": True,
            "seed_pairing": "explicit seed identity, never positional zip",
            "test_read": False,
            "test_locked": True,
        },
        "input_audit": audit,
        "aggregate": _aggregate_models(runs),
        "paired_seed_directions": {
            state: _paired_seed_advantages(runs, state=state)
            for state in CALIBRATION_STATES
        },
        "runs": [
            {
                "seed": int(run["seed"]),
                "calibration_file": str(
                    _artifact_dir(config)
                    / "validation"
                    / f"calibration_seed{int(run['seed'])}.json"
                ),
                "prediction_file": run["prediction_file"],
                "prediction_file_sha256": run[
                    "prediction_file_sha256"
                ],
            }
            for run in runs
        ],
        "test_locked": True,
    }
    save_json(
        payload,
        _artifact_dir(config)
        / "reports"
        / "symmetric_calibration.json",
    )
    return payload


def _probability_key(state: str) -> str:
    if state == "raw":
        return "raw_probabilities"
    if state == "cross_fitted_calibrated":
        return "cross_fitted_calibrated_probabilities"
    raise ValueError(f"Unknown calibration state: {state}")


def _metric_bundle(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    state: str,
    num_classes: int,
) -> dict[str, float]:
    labels: list[int] = []
    probabilities: list[list[float]] = []
    key = _probability_key(state)
    for record in records:
        labels.extend(int(value) for value in record["labels"])
        probabilities.extend(record["models"][model_name][key])
    predictions = [
        max(range(len(row)), key=lambda index: row[index])
        for row in probabilities
    ]
    quality = _probability_quality(
        labels, probabilities, num_classes
    )
    return {
        "weighted_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(num_classes)),
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "nll": float(quality["nll"]),
        "brier": float(quality["brier"]),
        "ece": float(quality["ece"]),
    }


def _load_prediction_runs(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    calibration_file = (
        _artifact_dir(config)
        / "reports"
        / "symmetric_calibration.json"
    )
    if not calibration_file.exists():
        raise FileNotFoundError(
            "Run Experiment 15 symmetric calibration before bootstrap"
        )
    calibration = _read_json(calibration_file)
    if calibration.get("protocol", {}).get("test_read") is not False:
        raise ValueError("Experiment 15 calibration manifest is not validation-only")
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        path = (
            _artifact_dir(config)
            / "validation"
            / f"predictions_seed{seed}.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction cache: {path}")
        payload = _read_json(path)
        if int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Prediction seed mismatch: {path}")
        if payload.get("split") != "validation":
            raise ValueError(f"Non-validation prediction cache: {path}")
        runs.append(payload)
    return runs


def _paired_dialogue_bootstrap(
    runs: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError(
            "experiment15.bootstrap_samples must be at least 100"
        )
    if len(runs) < 3:
        raise ValueError("Paired bootstrap requires at least three model seeds")

    canonical_ids = [
        str(record["dialogue_id"]) for record in runs[0]["dialogues"]
    ]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise ValueError("Duplicate dialogue IDs in prediction cache")
    canonical_labels = {
        str(record["dialogue_id"]): record["labels"]
        for record in runs[0]["dialogues"]
    }
    run_maps: list[dict[str, dict[str, Any]]] = []
    id_to_label = runs[0]["id_to_label"]
    for run in runs:
        if run["id_to_label"] != id_to_label:
            raise ValueError("Label maps differ across model seeds")
        mapping = {
            str(record["dialogue_id"]): record
            for record in run["dialogues"]
        }
        if set(mapping) != set(canonical_ids):
            raise ValueError("Dialogue sets differ across model seeds")
        for dialogue_id in canonical_ids:
            if mapping[dialogue_id]["labels"] != canonical_labels[dialogue_id]:
                raise ValueError(
                    f"Labels differ across seeds for dialogue {dialogue_id}"
                )
        run_maps.append(mapping)

    num_classes = len(id_to_label)
    generator = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for state_index, state in enumerate(CALIBRATION_STATES):
        point_metrics: dict[str, dict[str, float]] = {}
        for model_name in MODEL_ORDER:
            per_seed = [
                _metric_bundle(
                    [mapping[item] for item in canonical_ids],
                    model_name=model_name,
                    state=state,
                    num_classes=num_classes,
                )
                for mapping in run_maps
            ]
            point_metrics[model_name] = {
                metric: mean([result[metric] for result in per_seed])
                for metric in METRIC_PATHS
            }

        draws = {
            control: {metric: [] for metric in METRIC_PATHS}
            for control in CONTROLS
        }
        state_generator = np.random.default_rng(
            generator.integers(0, 2**32 - 1) + state_index
        )
        for _ in range(samples):
            indices = state_generator.integers(
                0, len(canonical_ids), size=len(canonical_ids)
            )
            sampled_ids = [canonical_ids[int(index)] for index in indices]
            averaged: dict[str, dict[str, float]] = {}
            for model_name in MODEL_ORDER:
                per_seed = [
                    _metric_bundle(
                        [mapping[item] for item in sampled_ids],
                        model_name=model_name,
                        state=state,
                        num_classes=num_classes,
                    )
                    for mapping in run_maps
                ]
                averaged[model_name] = {
                    metric: mean(
                        [result[metric] for result in per_seed]
                    )
                    for metric in METRIC_PATHS
                }
            for control in CONTROLS:
                for metric in METRIC_PATHS:
                    draws[control][metric].append(
                        _advantage(
                            averaged["roletrans"][metric],
                            averaged[control][metric],
                            metric,
                        )
                    )

        controls: dict[str, Any] = {}
        for control in CONTROLS:
            controls[control] = {}
            for metric, values in draws[control].items():
                array = np.asarray(values, dtype=np.float64)
                probability_positive = float((array > 0.0).mean())
                lower_tail = float((array <= 0.0).mean())
                upper_tail = float((array >= 0.0).mean())
                point = _advantage(
                    point_metrics["roletrans"][metric],
                    point_metrics[control][metric],
                    metric,
                )
                controls[control][metric] = {
                    "point_estimate_advantage": point,
                    "bootstrap_mean_advantage": float(array.mean()),
                    "ci95": [
                        float(np.quantile(array, 0.025)),
                        float(np.quantile(array, 0.975)),
                    ],
                    "probability_positive": probability_positive,
                    "two_sided_tail_probability": min(
                        1.0,
                        2.0
                        * min(
                            lower_tail,
                            upper_tail,
                        ),
                    ),
                    "advantage_definition": (
                        "control-reference"
                        if metric in LOWER_IS_BETTER
                        else "reference-control"
                    ),
                    "samples": samples,
                }
        output[state] = {
            "point_metrics": point_metrics,
            "comparisons": controls,
        }
    return {
        "protocol": {
            "unit": "complete dialogue",
            "pairing": (
                "the same sampled dialogue IDs are used for every model "
                "and every fixed training seed"
            ),
            "model_seeds": [int(run["seed"]) for run in runs],
            "model_seeds_resampled": False,
            "seed_variability_reported_separately": True,
            "dialogues": len(canonical_ids),
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "primary_comparison": "roletrans versus capacity_direct",
            "secondary_comparisons": [
                "roletrans versus contextual",
                "roletrans versus speaker_aware",
                "roletrans versus anchor",
            ],
            "test_read": False,
        },
        "states": output,
    }


def _bootstrap_stage(config: dict[str, Any]) -> dict[str, Any]:
    _check_source_inputs(config)
    runs = _load_prediction_runs(config)
    settings = _settings(config)
    result = _paired_dialogue_bootstrap(
        runs,
        samples=int(settings.get("bootstrap_samples", 2000)),
        seed=int(settings.get("bootstrap_seed", 2015)),
    )
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "experiment": 15,
        **result,
        "test_locked": True,
    }
    save_json(
        payload,
        _artifact_dir(config) / "reports" / "paired_bootstrap.json",
    )
    return payload


def _summary_stage(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    calibration_file = root / "reports" / "symmetric_calibration.json"
    bootstrap_file = root / "reports" / "paired_bootstrap.json"
    for path in (calibration_file, bootstrap_file):
        if not path.exists():
            raise FileNotFoundError(f"Missing Experiment 15 report: {path}")
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "experiment": 15,
        "title": (
            "Symmetric calibration and paired dialogue-level bootstrap"
        ),
        "protocol": {
            "analysis_only": True,
            "models_retrained": False,
            "model_selection_changed": False,
            "validation_split_only": True,
            "test_read": False,
            "test_locked": True,
        },
        "symmetric_calibration": _read_json(calibration_file),
        "paired_dialogue_bootstrap": _read_json(bootstrap_file),
    }
    save_json(payload, root / "reports" / "validation_summary.json")
    return payload


def run_experiment15(
    config: dict[str, Any],
    *,
    stage: str,
    device: str | None = None,
) -> dict[str, Any]:
    """Run validation-only statistical closure for Experiment 14."""

    _seeds(config)
    _artifact_dir(config).mkdir(parents=True, exist_ok=True)
    if stage in {"test", "final-test", "check-final", "summarize-final"}:
        raise ValueError(
            "Experiment 15 is validation-only and refuses every test stage"
        )
    if stage == "check":
        return _check_source_inputs(config)
    if stage == "symmetric-calibration":
        return _symmetric_calibration_stage(config, device)
    if stage == "paired-bootstrap":
        return _bootstrap_stage(config)
    if stage == "summarize":
        return _summary_stage(config)
    if stage == "all-validation":
        _symmetric_calibration_stage(config, device)
        _bootstrap_stage(config)
        return _summary_stage(config)
    raise ValueError(f"Unknown Experiment 15 stage: {stage}")
