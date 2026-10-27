from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch
from torch import nn

from .common import (
    classification_metrics,
    get_device,
    load_feature_store,
    move_to_device,
    save_json,
)
from .erc_trainer import (
    _build_model,
    _loader,
    _probability_quality,
    _validate_current_logit_features,
)


VALID_METHODS = {"scalar", "gate"}


def _logit(probability: float) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("Expected a probability strictly between zero and one")
    return math.log(probability / (1.0 - probability))


class BoundedTemperature(nn.Module):
    """A positive post-hoc temperature that cannot change the argmax class."""

    def __init__(
        self,
        method: str,
        minimum: float,
        maximum: float,
        gate_center: float,
    ) -> None:
        super().__init__()
        if method not in VALID_METHODS:
            raise ValueError(
                f"Calibration method must be one of {sorted(VALID_METHODS)}"
            )
        if not 0.0 < minimum < 1.0 < maximum:
            raise ValueError(
                "Temperature bounds must satisfy 0 < minimum < 1 < maximum"
            )
        self.method = method
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.register_buffer(
            "gate_center", torch.tensor(float(gate_center), dtype=torch.float64)
        )
        initial_fraction = (1.0 - minimum) / (maximum - minimum)
        self.raw_bias = nn.Parameter(
            torch.tensor(_logit(initial_fraction), dtype=torch.float64)
        )
        if method == "gate":
            self.raw_slope = nn.Parameter(torch.zeros((), dtype=torch.float64))
        else:
            self.register_parameter("raw_slope", None)

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        gate = gate.to(dtype=self.raw_bias.dtype)
        raw = self.raw_bias
        if self.raw_slope is not None:
            raw = raw + self.raw_slope * (gate - self.gate_center)
        return self.minimum + (
            self.maximum - self.minimum
        ) * torch.sigmoid(raw)

    def export_parameters(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "gate_center": float(self.gate_center),
            "raw_bias": float(self.raw_bias.detach()),
            "raw_slope": (
                None
                if self.raw_slope is None
                else float(self.raw_slope.detach())
            ),
        }


def _temperature_from_parameters(
    gate: torch.Tensor, parameters: dict[str, Any]
) -> torch.Tensor:
    minimum = float(parameters["minimum"])
    maximum = float(parameters["maximum"])
    raw = torch.full_like(gate, float(parameters["raw_bias"]))
    slope = parameters.get("raw_slope")
    if slope is not None:
        raw = raw + float(slope) * (
            gate - float(parameters["gate_center"])
        )
    return minimum + (maximum - minimum) * torch.sigmoid(raw)


def _fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gate: torch.Tensor,
    method: str,
    minimum: float,
    maximum: float,
    max_iter: int,
    slope_l2: float,
) -> dict[str, Any]:
    if logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("Expected logits [N,K] and labels [N]")
    if logits.shape[0] != labels.numel() or gate.numel() != labels.numel():
        raise ValueError("Calibration logits, labels, and gates must align")
    if labels.numel() == 0:
        raise ValueError("Cannot fit a calibrator without observations")
    if max_iter <= 0 or slope_l2 < 0.0:
        raise ValueError("Invalid calibration optimizer settings")

    logits = logits.detach().to(device="cpu", dtype=torch.float64)
    labels = labels.detach().to(device="cpu", dtype=torch.long)
    gate = gate.detach().to(device="cpu", dtype=torch.float64)
    model = BoundedTemperature(
        method=method,
        minimum=minimum,
        maximum=maximum,
        gate_center=float(gate.mean()),
    )
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=0.5,
        max_iter=max_iter,
        tolerance_grad=1e-10,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    def objective() -> torch.Tensor:
        temperature = model(gate)
        loss = nn.functional.cross_entropy(
            logits / temperature.unsqueeze(-1), labels
        )
        if model.raw_slope is not None and slope_l2 > 0.0:
            loss = loss + slope_l2 * model.raw_slope.square()
        return loss

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = objective()
        loss.backward()
        return loss

    before = float(nn.functional.cross_entropy(logits, labels))
    optimizer.step(closure)
    after = float(objective().detach())
    parameters = model.export_parameters()
    fitted_temperature = model(gate).detach()
    parameters["fit_nll_before"] = before
    parameters["fit_objective_after"] = after
    parameters["fit_temperature_mean"] = float(fitted_temperature.mean())
    parameters["fit_temperature_min"] = float(fitted_temperature.min())
    parameters["fit_temperature_max"] = float(fitted_temperature.max())
    parameters["fit_utterances"] = int(labels.numel())
    return parameters


def _dialogue_folds(
    records: list[dict[str, Any]], folds: int, seed: int
) -> list[list[int]]:
    if folds < 2 or folds > len(records):
        raise ValueError(
            f"calibration.folds must be between 2 and {len(records)}"
        )
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    # Greedy assignment balances utterance counts while keeping complete
    # dialogues together. The initial shuffle randomizes equal-length ties.
    indices.sort(key=lambda index: int(records[index]["labels"].numel()), reverse=True)
    assignments: list[list[int]] = [[] for _ in range(folds)]
    loads = [0 for _ in range(folds)]
    for index in indices:
        destination = min(range(folds), key=lambda fold: (loads[fold], fold))
        assignments[destination].append(index)
        loads[destination] += int(records[index]["labels"].numel())
    return assignments


def _concatenate(
    records: list[dict[str, Any]], indices: list[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat([records[index]["logits"] for index in indices], dim=0),
        torch.cat([records[index]["labels"] for index in indices], dim=0),
        torch.cat([records[index]["gate"] for index in indices], dim=0),
    )


def _temperature_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().float().flatten()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "quantiles": [
            float(value)
            for value in torch.quantile(
                values, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
            )
        ],
    }


def _metrics_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    id_to_label: dict[int, str],
    temperature: torch.Tensor | None = None,
) -> dict[str, Any]:
    logits = logits.float()
    labels = labels.long()
    if temperature is None:
        temperature = torch.ones(logits.shape[0], dtype=logits.dtype)
    temperature = temperature.float().flatten()
    if temperature.numel() != logits.shape[0] or torch.any(temperature <= 0):
        raise ValueError("Every calibration temperature must be positive")
    calibrated_logits = logits / temperature.unsqueeze(-1)
    predictions = calibrated_logits.argmax(dim=-1)
    probabilities = torch.softmax(calibrated_logits, dim=-1)
    metrics = classification_metrics(
        labels.tolist(), predictions.tolist(), id_to_label
    )
    metrics["probability_quality"] = _probability_quality(
        labels.tolist(), probabilities.tolist(), len(id_to_label)
    )
    metrics["utterances"] = int(labels.numel())
    return metrics


def _collect_dialogue_outputs(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[int, str], int]:
    store = load_feature_store(config["data"]["feature_file"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_model = checkpoint.get("config", {}).get("model", {})
    expected_lineage = {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "speaker",
    }
    lineage_mismatches = {
        key: saved_model.get(key)
        for key, expected in expected_lineage.items()
        if saved_model.get(key) != expected
    }
    if lineage_mismatches:
        raise ValueError(
            f"{checkpoint_path} is not an 8C-4 checkpoint: "
            f"{lineage_mismatches}"
        )
    label_to_id = {
        str(key): int(value) for key, value in checkpoint["label_to_id"].items()
    }
    id_to_label = {value: key for key, value in label_to_id.items()}
    model = _build_model(config, len(label_to_id), device)
    if not model.use_directed_update:
        raise ValueError("Experiment 8D requires a directed-affect checkpoint")
    assert model.directed_affect_updater is not None
    if model.directed_affect_updater.matrix_mode != "logit_residual":
        raise ValueError("Experiment 8D requires the 8C-4 logit-residual model")
    _validate_current_logit_features(store, len(label_to_id))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    seed = int(checkpoint.get("seed", 42))
    loader = _loader(store["splits"][split], config, False, seed, device)
    use_amp = (
        bool(config["train_erc"].get("use_amp", True))
        and device.type == "cuda"
    )
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in loader:
            batch = move_to_device(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits, auxiliary = model(
                    batch["embeddings"],
                    batch["speaker_ids"],
                    batch["mask"],
                    current_logits=batch["current_logits"],
                    return_aux=True,
                )
            gate = auxiliary["transition_gate"].squeeze(-1)
            for row, dialogue_id in enumerate(batch["dialogue_ids"]):
                length = int(batch["mask"][row].sum())
                records.append(
                    {
                        "dialogue_id": str(dialogue_id),
                        "logits": logits[row, :length].float().cpu(),
                        "anchor_logits": batch["current_logits"][
                            row, :length
                        ].float().cpu(),
                        "labels": batch["labels"][row, :length].long().cpu(),
                        "gate": gate[row, :length].float().cpu(),
                    }
                )
    return records, id_to_label, seed


def _cross_fitted_calibration(
    records: list[dict[str, Any]],
    id_to_label: dict[int, str],
    method: str,
    folds: int,
    fold_seed: int,
    minimum: float,
    maximum: float,
    max_iter: int,
    slope_l2: float,
) -> dict[str, Any]:
    assignments = _dialogue_folds(records, folds, fold_seed)
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
            method,
            minimum,
            maximum,
            max_iter,
            slope_l2,
        )
        _, held_labels, held_gate = _concatenate(records, held_out)
        held_temperature = _temperature_from_parameters(
            held_gate.float(), parameters
        )
        offset = 0
        for record_index in held_out:
            length = int(records[record_index]["labels"].numel())
            temperatures[record_index] = held_temperature[offset : offset + length]
            offset += length
        fold_reports.append(
            {
                "fold": fold_index,
                "fit_dialogues": len(training),
                "held_out_dialogues": len(held_out),
                "held_out_utterances": int(held_labels.numel()),
                "parameters": parameters,
            }
        )
    if any(value is None for value in temperatures):
        raise AssertionError("Every validation dialogue must be held out once")

    indices = list(range(len(records)))
    logits, labels, _ = _concatenate(records, indices)
    joined_temperature = torch.cat(
        [value for value in temperatures if value is not None], dim=0
    )
    original_predictions = logits.argmax(dim=-1)
    calibrated_predictions = (
        logits / joined_temperature.unsqueeze(-1)
    ).argmax(dim=-1)
    argmax_changes = int(
        original_predictions.ne(calibrated_predictions).sum()
    )
    if argmax_changes != 0:
        raise AssertionError("Positive temperature scaling changed an argmax")
    return {
        "method": method,
        "metrics": _metrics_from_logits(
            logits, labels, id_to_label, joined_temperature
        ),
        "temperature": _temperature_summary(joined_temperature),
        "argmax_changes": argmax_changes,
        "folds": fold_reports,
    }


def _scalar_summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": mean(values),
        "std": pstdev(values),
        "count": len(values),
    }


def _aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    variants = ("uncalibrated", "scalar", "gate")
    paths = {
        "weighted_f1": ("weighted_f1",),
        "macro_f1": ("macro_f1",),
        "accuracy": ("accuracy",),
        "nll": ("probability_quality", "nll"),
        "brier": ("probability_quality", "brier"),
        "ece": ("probability_quality", "ece"),
    }
    output: dict[str, Any] = {}
    for variant in variants:
        output[variant] = {}
        for metric, path in paths.items():
            values: list[float] = []
            for run in runs:
                payload: Any = run[variant]["metrics"]
                for key in path:
                    payload = payload[key]
                values.append(float(payload))
            output[variant][metric] = _scalar_summary(values)
        output[variant]["argmax_changes"] = _scalar_summary(
            [float(run[variant].get("argmax_changes", 0)) for run in runs]
        )
        if variant != "uncalibrated":
            output[variant]["temperature_mean"] = _scalar_summary(
                [float(run[variant]["temperature"]["mean"]) for run in runs]
            )
    return output


def _select_method(
    aggregate: dict[str, Any], runs: list[dict[str, Any]]
) -> tuple[str | None, dict[str, Any]]:
    checks: dict[str, Any] = {}
    eligible: list[str] = []
    baseline_nll = aggregate["uncalibrated"]["nll"]["mean"]
    for method in ("scalar", "gate"):
        improvements = [
            float(run["uncalibrated"]["metrics"]["probability_quality"]["nll"])
            - float(run[method]["metrics"]["probability_quality"]["nll"])
            for run in runs
        ]
        method_checks = {
            "argmax_unchanged": all(
                int(run[method]["argmax_changes"]) == 0 for run in runs
            ),
            "mean_nll_improved": (
                aggregate[method]["nll"]["mean"] < baseline_nll
            ),
            "nll_improved_on_at_least_2_seeds": (
                sum(value > 0.0 for value in improvements) >= 2
            ),
            "nll_improvement": _scalar_summary(improvements),
        }
        method_checks["eligible"] = all(
            value
            for key, value in method_checks.items()
            if key != "nll_improvement"
        )
        checks[method] = method_checks
        if method_checks["eligible"]:
            eligible.append(method)
    if not eligible:
        return None, checks
    if "scalar" in eligible and "gate" in eligible:
        scalar_nll = aggregate["scalar"]["nll"]["mean"]
        gate_nll = aggregate["gate"]["nll"]["mean"]
        gate_seed_wins = sum(
            run["gate"]["metrics"]["probability_quality"]["nll"]
            < run["scalar"]["metrics"]["probability_quality"]["nll"]
            for run in runs
        )
        # Prefer the simpler scalar calibrator unless conditioning has a
        # material and seed-consistent out-of-fold advantage.
        if scalar_nll - gate_nll >= 0.002 and gate_seed_wins >= 2:
            return "gate", checks
        return "scalar", checks
    return eligible[0], checks


def calibrate_erc_seed_sweep(
    config: dict[str, Any], device: str | None = None
) -> dict[str, Any]:
    calibration = config.get("calibration", {})
    split = str(calibration.get("split", "validation"))
    if split != "validation":
        raise ValueError("Experiment 8D calibration must use validation")
    seeds = [int(seed) for seed in calibration.get("seeds", [42, 43, 44])]
    folds = int(calibration.get("folds", 5))
    fold_seed = int(calibration.get("fold_seed", 2026))
    minimum = float(calibration.get("minimum_temperature", 0.25))
    maximum = float(calibration.get("maximum_temperature", 4.0))
    max_iter = int(calibration.get("max_iter", 100))
    slope_l2 = float(calibration.get("gate_slope_l2", 0.01))
    report_file = Path(calibration["report_file"])
    calibrator_dir = Path(calibration["calibrator_dir"])
    calibrator_dir.mkdir(parents=True, exist_ok=True)
    device_obj = get_device(device)
    base_checkpoint = Path(config["train_erc"]["checkpoint"])
    methods = ["scalar", "gate"]
    runs: list[dict[str, Any]] = []

    for seed in seeds:
        checkpoint = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_seed{seed}.pt"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing 8C-4 seed checkpoint: {checkpoint}"
            )
        records, id_to_label, checkpoint_seed = _collect_dialogue_outputs(
            config, checkpoint, split, device_obj
        )
        if checkpoint_seed != seed:
            raise ValueError(
                f"{checkpoint} stores seed={checkpoint_seed}, expected {seed}"
            )
        indices = list(range(len(records)))
        logits, labels, gate = _concatenate(records, indices)
        uncalibrated = {
            "method": "identity",
            "metrics": _metrics_from_logits(
                logits, labels, id_to_label, None
            ),
            "temperature": _temperature_summary(
                torch.ones(labels.numel())
            ),
            "argmax_changes": 0,
        }
        run: dict[str, Any] = {
            "seed": seed,
            "checkpoint": str(checkpoint),
            "dialogues": len(records),
            "utterances": int(labels.numel()),
            "uncalibrated": uncalibrated,
        }
        full_fit: dict[str, Any] = {}
        for method in methods:
            run[method] = _cross_fitted_calibration(
                records=records,
                id_to_label=id_to_label,
                method=method,
                folds=folds,
                fold_seed=fold_seed,
                minimum=minimum,
                maximum=maximum,
                max_iter=max_iter,
                slope_l2=slope_l2,
            )
            parameters = _fit_temperature(
                logits,
                labels,
                gate,
                method,
                minimum,
                maximum,
                max_iter,
                slope_l2,
            )
            calibrator_file = calibrator_dir / (
                f"meld_logit_residual_8d_{method}_seed{seed}.json"
            )
            calibrator = {
                "format_version": "pfa-erc-temperature-v1",
                "method": method,
                "model_seed": seed,
                "source_checkpoint": str(checkpoint),
                "fit_split": split,
                "fit_dialogues": len(records),
                "fit_utterances": int(labels.numel()),
                "parameters": parameters,
            }
            save_json(calibrator, calibrator_file)
            full_fit[method] = {
                "calibrator_file": str(calibrator_file),
                "parameters": parameters,
            }
        run["full_validation_fit"] = full_fit
        runs.append(run)
        save_json(
            {
                "status": "in_progress",
                "protocol": {
                    "split": split,
                    "seeds": seeds,
                    "folds": folds,
                    "fold_unit": "dialogue",
                    "fold_seed": fold_seed,
                    "gate_feature": "8C-4 transition_gate",
                    "test_used": False,
                },
                "completed_seeds": [item["seed"] for item in runs],
                "runs": runs,
            },
            report_file,
        )

    aggregate = _aggregate_runs(runs)
    recommended, eligibility = _select_method(aggregate, runs)
    result = {
        "status": "completed",
        "protocol": {
            "dataset": "MELD",
            "split": split,
            "seeds": seeds,
            "folds": folds,
            "fold_unit": "dialogue",
            "fold_seed": fold_seed,
            "gate_feature": "8C-4 transition_gate",
            "test_used": False,
            "model_retrained": False,
            "classification_boundary_changed": False,
            "stable_transition_used": False,
            "selection_rule": (
                "Require zero argmax changes and lower out-of-fold NLL on at "
                "least two seeds. Prefer scalar temperature unless the "
                "gate-conditioned method lowers mean NLL by at least 0.002 "
                "and wins on at least two seeds."
            ),
        },
        "completed_seeds": seeds,
        "aggregate": aggregate,
        "eligibility": eligibility,
        "recommended_validation_method": recommended,
        "selected_calibrators": (
            {
                str(run["seed"]): run["full_validation_fit"][recommended][
                    "calibrator_file"
                ]
                for run in runs
            }
            if recommended is not None
            else {}
        ),
        "runs": runs,
        "test_locked": True,
    }
    save_json(result, report_file)
    return result


def evaluate_calibrated_erc(
    config: dict[str, Any],
    calibrator_path: str | Path,
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    calibration = config.get("calibration", {})
    split = str(split or calibration.get("split", "validation"))
    if split == "test" and not bool(calibration.get("allow_test", False)):
        raise ValueError(
            "Test is locked. Freeze the 8D method and explicitly set "
            "calibration.allow_test=true for the one-time final evaluation."
        )
    calibrator_path = Path(calibrator_path)
    calibrator = json.loads(calibrator_path.read_text(encoding="utf-8"))
    if calibrator.get("format_version") != "pfa-erc-temperature-v1":
        raise ValueError(f"Invalid calibrator: {calibrator_path}")
    checkpoint = Path(
        checkpoint_path or calibrator["source_checkpoint"]
    )
    records, id_to_label, seed = _collect_dialogue_outputs(
        config, checkpoint, split, get_device(device)
    )
    logits, labels, gate = _concatenate(
        records, list(range(len(records)))
    )
    anchor_logits = torch.cat(
        [record["anchor_logits"] for record in records], dim=0
    )
    temperature = _temperature_from_parameters(
        gate, calibrator["parameters"]
    )
    calibrated = _metrics_from_logits(
        logits, labels, id_to_label, temperature
    )
    uncalibrated = _metrics_from_logits(logits, labels, id_to_label)
    argmax_changes = int(
        logits.argmax(dim=-1)
        .ne((logits / temperature.unsqueeze(-1)).argmax(dim=-1))
        .sum()
    )
    return {
        "checkpoint": str(checkpoint),
        "calibrator": str(calibrator_path),
        "split": split,
        "seed": seed,
        "anchor": _metrics_from_logits(
            anchor_logits, labels, id_to_label
        ),
        "uncalibrated": uncalibrated,
        "calibrated": calibrated,
        "temperature": _temperature_summary(temperature),
        "argmax_changes": argmax_changes,
    }
