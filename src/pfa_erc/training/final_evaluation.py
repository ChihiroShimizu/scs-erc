from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .calibration import evaluate_calibrated_erc


FINAL_METHOD = "scalar"
METRIC_PATHS = {
    "weighted_f1": ("weighted_f1",),
    "macro_f1": ("macro_f1",),
    "accuracy": ("accuracy",),
    "nll": ("probability_quality", "nll"),
    "brier": ("probability_quality", "brier"),
    "ece": ("probability_quality", "ece"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return {
        "values": values,
        "mean": mean(values),
        "std": pstdev(values),
        "count": len(values),
    }


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> float:
    value: Any = payload
    for key in path:
        value = value[key]
    return float(value)


def _classification_labels(run: dict[str, Any]) -> list[str]:
    report = run["calibrated"]["classification_report"]
    return [
        str(label)
        for label, metrics in report.items()
        if isinstance(metrics, dict)
        and "support" in metrics
        and label not in {"macro avg", "weighted avg"}
    ]


def _aggregate_final_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("Final evaluation requires at least one seed")
    aggregate: dict[str, Any] = {}
    for variant in ("anchor", "uncalibrated", "calibrated"):
        aggregate[variant] = {
            metric: _summary(
                [_nested(run[variant], path) for run in runs]
            )
            for metric, path in METRIC_PATHS.items()
        }
    aggregate["temperature_mean"] = _summary(
        [float(run["temperature"]["mean"]) for run in runs]
    )
    aggregate["argmax_changes"] = _summary(
        [float(run["argmax_changes"]) for run in runs]
    )
    aggregate["calibration_improvement"] = {
        "nll": _summary(
            [
                _nested(run["uncalibrated"], METRIC_PATHS["nll"])
                - _nested(run["calibrated"], METRIC_PATHS["nll"])
                for run in runs
            ]
        ),
        "brier": _summary(
            [
                _nested(run["uncalibrated"], METRIC_PATHS["brier"])
                - _nested(run["calibrated"], METRIC_PATHS["brier"])
                for run in runs
            ]
        ),
        "ece": _summary(
            [
                _nested(run["uncalibrated"], METRIC_PATHS["ece"])
                - _nested(run["calibrated"], METRIC_PATHS["ece"])
                for run in runs
            ]
        ),
    }
    aggregate["final_minus_anchor"] = {
        metric: _summary(
            [
                _nested(run["calibrated"], METRIC_PATHS[metric])
                - _nested(run["anchor"], METRIC_PATHS[metric])
                for run in runs
            ]
        )
        for metric in ("weighted_f1", "macro_f1", "accuracy")
    }
    return aggregate


def _aggregate_per_emotion(
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    labels = _classification_labels(runs[0])
    output: dict[str, Any] = {}
    for label in labels:
        output[label] = {}
        for variant in ("anchor", "uncalibrated", "calibrated"):
            output[label][variant] = {}
            for metric in ("precision", "recall", "f1-score", "support"):
                values = [
                    float(
                        run[variant]["classification_report"][label][metric]
                    )
                    for run in runs
                ]
                output[label][variant][metric] = _summary(values)
        output[label]["final_minus_anchor_f1"] = _summary(
            [
                float(
                    run["calibrated"]["classification_report"][label][
                        "f1-score"
                    ]
                )
                - float(
                    run["anchor"]["classification_report"][label]["f1-score"]
                )
                for run in runs
            ]
        )
    return output


def _assert_boundary_preserved(run: dict[str, Any]) -> None:
    if int(run["argmax_changes"]) != 0:
        raise AssertionError("Scalar calibration changed a test argmax")
    for metric in ("weighted_f1", "macro_f1", "accuracy"):
        before = float(run["uncalibrated"][metric])
        after = float(run["calibrated"][metric])
        if abs(before - after) > 1e-12:
            raise AssertionError(
                f"Scalar calibration changed {metric}: {before} -> {after}"
            )


def _input_manifest(
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    final = config.get("final_evaluation", {})
    method = str(final.get("method", ""))
    split = str(final.get("split", ""))
    seeds = [int(seed) for seed in final.get("seeds", [])]
    if method != FINAL_METHOD:
        raise ValueError(
            "Experiment 8E is frozen to the 8D-selected scalar calibrator"
        )
    if split != "test":
        raise ValueError("Experiment 8E must evaluate exactly split=test")
    if seeds != [42, 43, 44]:
        raise ValueError("Experiment 8E is frozen to model seeds [42, 43, 44]")
    if not bool(config.get("calibration", {}).get("allow_test", False)):
        raise ValueError("Experiment 8E requires calibration.allow_test=true")

    validation_report = Path(final["validation_report_file"])
    if not validation_report.exists():
        raise FileNotFoundError(
            f"Missing completed 8D validation report: {validation_report}"
        )
    validation = _read_json(validation_report)
    if validation.get("status") != "completed":
        raise ValueError("Experiment 8D validation is not complete")
    if validation.get("recommended_validation_method") != FINAL_METHOD:
        raise ValueError(
            "Experiment 8D did not select the frozen scalar calibrator"
        )
    protocol = validation.get("protocol", {})
    if (
        protocol.get("split") != "validation"
        or bool(protocol.get("test_used"))
        or validation.get("completed_seeds") != seeds
        or validation.get("test_locked") is not True
    ):
        raise ValueError("Invalid or test-contaminated 8D selection report")
    if not validation.get("eligibility", {}).get(FINAL_METHOD, {}).get(
        "eligible", False
    ):
        raise ValueError("The selected scalar calibrator is not eligible")

    checkpoint_base = Path(final["checkpoint"])
    calibrator_dir = Path(final["calibrator_dir"])
    inputs: list[dict[str, Any]] = []
    selected = validation.get("selected_calibrators", {})
    for seed in seeds:
        checkpoint = checkpoint_base.with_name(
            f"{checkpoint_base.stem}_seed{seed}.pt"
        )
        calibrator = calibrator_dir / (
            f"meld_logit_residual_8d_{FINAL_METHOD}_seed{seed}.json"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing frozen 8C-4 checkpoint: {checkpoint}"
            )
        if not calibrator.exists():
            raise FileNotFoundError(
                f"Missing frozen 8D scalar calibrator: {calibrator}"
            )
        calibrator_payload = _read_json(calibrator)
        if (
            calibrator_payload.get("format_version")
            != "pfa-erc-temperature-v1"
            or calibrator_payload.get("method") != FINAL_METHOD
            or int(calibrator_payload.get("model_seed", -1)) != seed
            or calibrator_payload.get("fit_split") != "validation"
        ):
            raise ValueError(f"Invalid frozen calibrator: {calibrator}")
        selected_path = Path(str(selected.get(str(seed), "")))
        if selected_path.name != calibrator.name:
            raise ValueError(
                f"8D selected a different seed-{seed} calibrator"
            )
        source_name = Path(
            str(calibrator_payload.get("source_checkpoint", ""))
        ).name
        if source_name != checkpoint.name:
            raise ValueError(
                f"{calibrator} was fitted for {source_name}, not "
                f"{checkpoint.name}"
            )
        inputs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "calibrator": str(calibrator),
                "calibrator_sha256": _sha256(calibrator),
            }
        )

    manifest = {
        "dataset": "MELD",
        "split": split,
        "method": method,
        "seeds": seeds,
        "selection_split": "validation",
        "validation_report": str(validation_report),
        "validation_report_sha256": _sha256(validation_report),
        "model_retrained": False,
        "stable_transition_used": False,
        "classification_boundary_changed": False,
    }
    return manifest, inputs


def _validate_resumed_run(
    run: dict[str, Any], expected: dict[str, Any]
) -> None:
    if (
        int(run.get("seed", -1)) != int(expected["seed"])
        or run.get("split") != "test"
        or Path(str(run.get("checkpoint", ""))).name
        != Path(expected["checkpoint"]).name
        or Path(str(run.get("calibrator", ""))).name
        != Path(expected["calibrator"]).name
    ):
        raise ValueError(
            f"Existing seed-{expected['seed']} result does not match 8E"
        )
    _assert_boundary_preserved(run)


def evaluate_final_calibrated_seed_sweep(
    config: dict[str, Any],
    device: str | None = None,
    *,
    confirm: bool = False,
    resume: bool = False,
    check_only: bool = False,
) -> dict[str, Any]:
    final = config.get("final_evaluation", {})
    manifest, inputs = _input_manifest(config)
    if check_only:
        return {
            "status": "ready",
            "protocol": manifest,
            "inputs": inputs,
            "test_read": False,
            "message": (
                "Frozen 8C-4 checkpoints, 8D scalar calibrators, and the "
                "validation-only selection report are consistent."
            ),
        }
    if not confirm:
        raise ValueError(
            "Final test requires --confirm-final-test. This is a one-time "
            "evaluation of the already frozen method."
        )

    report_file = Path(final["report_file"])
    lock_file = Path(final["lock_file"])
    individual_dir = Path(final["individual_report_dir"])
    if report_file.exists():
        raise FileExistsError(
            f"Final test is already complete: {report_file}. "
            "Use the summarize mode; do not rerun it."
        )
    individual_dir.mkdir(parents=True, exist_ok=True)
    lock_payload = {
        "format_version": "pfa-erc-final-test-lock-v1",
        "status": "running",
        "created_at": _utc_now(),
        "protocol": manifest,
        "inputs": inputs,
    }
    if lock_file.exists():
        if not resume:
            raise FileExistsError(
                f"An 8E lock already exists: {lock_file}. If a previous run "
                "was interrupted, use --resume instead of starting over."
            )
        existing_lock = _read_json(lock_file)
        if existing_lock.get("protocol") != manifest:
            raise ValueError("Existing 8E lock has a different protocol")
        lock_payload = existing_lock
        lock_payload["status"] = "resuming"
        lock_payload["resumed_at"] = _utc_now()
        _atomic_save_json(lock_payload, lock_file)
    else:
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with lock_file.open("x", encoding="utf-8") as handle:
                json.dump(
                    lock_payload,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                handle.write("\n")
        except FileExistsError as error:
            raise FileExistsError(
                f"Another final-test process created {lock_file}"
            ) from error

    runs: list[dict[str, Any]] = []
    try:
        for item in inputs:
            seed = int(item["seed"])
            individual = individual_dir / f"seed{seed}_test.json"
            if individual.exists():
                if not resume:
                    raise FileExistsError(
                        f"Unexpected existing test result: {individual}"
                    )
                run = _read_json(individual)
                _validate_resumed_run(run, item)
            else:
                run = evaluate_calibrated_erc(
                    config,
                    calibrator_path=item["calibrator"],
                    checkpoint_path=item["checkpoint"],
                    device=device,
                    split="test",
                )
                _validate_resumed_run(run, item)
                run["frozen_method"] = "8C-4 logit residual + 8D-1 scalar"
                run["test_evaluation"] = "one-time final"
                _atomic_save_json(run, individual)
            runs.append(run)

        result = {
            "status": "completed",
            "completed_at": _utc_now(),
            "protocol": {
                **manifest,
                "test_used": True,
                "test_locked_after_completion": True,
                "selection_after_test": False,
                "individual_seed_reports": str(individual_dir),
            },
            "inputs": inputs,
            "completed_seeds": [int(run["seed"]) for run in runs],
            "aggregate": _aggregate_final_runs(runs),
            "per_emotion": _aggregate_per_emotion(runs),
            "runs": runs,
        }
        _atomic_save_json(result, report_file)
        lock_payload["status"] = "completed"
        lock_payload["completed_at"] = result["completed_at"]
        lock_payload["report_file"] = str(report_file)
        lock_payload["report_sha256"] = _sha256(report_file)
        _atomic_save_json(lock_payload, lock_file)
        return result
    except Exception as error:
        lock_payload["status"] = "interrupted_or_failed"
        lock_payload["updated_at"] = _utc_now()
        lock_payload["completed_seeds"] = [
            int(run["seed"]) for run in runs
        ]
        lock_payload["error_type"] = type(error).__name__
        lock_payload["error"] = str(error)
        _atomic_save_json(lock_payload, lock_file)
        raise
