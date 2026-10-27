from __future__ import annotations

import copy
import hashlib
import json
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
    _concatenate,
    _cross_fitted_calibration,
    _fit_temperature,
    _metrics_from_logits,
    _temperature_from_parameters,
)
from .common import (
    get_device,
    load_feature_store,
    move_to_device,
    save_json,
)
from .erc_trainer import (
    _build_model,
    _loader,
    _validate_current_logit_features,
    evaluate_erc,
    train_erc,
)
from .roberta_trainer import evaluate_roberta, train_roberta


# All four heads consume the same independently fine-tuned RoBERTa features.
# The first two differ only in how a K x K output is used.  Their module graph
# and active trainable parameter set are exactly the same.
VARIANTS: dict[str, dict[str, Any]] = {
    "roletrans": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "speaker",
        "directed_fusion_mode": "logit_residual",
        "directed_freeze_unused_parameters": True,
    },
    "capacity_direct": {
        "directed_variant": "dynamic_only",
        "directed_matrix_mode": "direct_residual",
        "directed_prior_mode": "speaker",
        "directed_fusion_mode": "logit_residual",
        "directed_freeze_unused_parameters": True,
    },
    "contextual": {
        "directed_variant": "global_context",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "speaker",
        "directed_fusion_mode": "logit_residual",
        "directed_freeze_unused_parameters": True,
    },
    "speaker_aware": {
        "directed_variant": "dual_channel",
        "directed_matrix_mode": "logit_residual",
        "directed_prior_mode": "speaker",
        "directed_fusion_mode": "logit_residual",
        "directed_freeze_unused_parameters": True,
    },
}

MODEL_ORDER = (
    "anchor",
    "contextual",
    "speaker_aware",
    "capacity_direct",
    "roletrans",
)
METRIC_PATHS: dict[str, tuple[str, ...]] = {
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


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
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


def _directory_sha256(path: Path) -> str:
    if not path.is_dir():
        raise FileNotFoundError(f"Expected checkpoint directory: {path}")
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"Empty checkpoint directory: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_fingerprint(payload: Any) -> str:
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
    section = config.get("experiment14")
    if not isinstance(section, dict):
        raise KeyError("Config is missing the experiment14 section")
    return section


def _artifact_dir(config: dict[str, Any]) -> Path:
    return Path(_settings(config)["artifact_dir"])


def _seeds(config: dict[str, Any]) -> list[int]:
    seeds = [int(seed) for seed in _settings(config).get("seeds", [])]
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError(
            "experiment14.seeds must contain at least three distinct seeds"
        )
    return seeds


def _dataset_name(config: dict[str, Any]) -> str:
    return str(config["data"]["dataset"]).lower()


def _seed_config(
    config: dict[str, Any],
    seed: int,
    *,
    variant: str = "roletrans",
    split: str = "validation",
) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise KeyError(f"Unknown Experiment 14 variant: {variant}")
    result = copy.deepcopy(config)
    root = _artifact_dir(result)
    result["experiment"]["seed"] = int(seed)
    result["experiment"]["name"] = (
        f"{_dataset_name(result)}_experiment14_{variant}_seed{seed}"
    )
    anchor_dir = root / "anchors" / f"seed{seed}"
    result["train_roberta"]["checkpoint"] = str(anchor_dir)
    result["features"]["encoder_checkpoint"] = str(anchor_dir)
    result["features"]["splits"] = ["train", "validation"]
    result["data"]["feature_file"] = str(
        root / "features" / f"seed{seed}.pt"
    )
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
    config: dict[str, Any], seed: int, variant: str
) -> dict[str, Any]:
    result = _seed_config(
        config, seed, variant=variant, split="test"
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
    saved_model = payload.get("config", {}).get("model", {})
    return (
        int(payload.get("seed", -1)) == int(seed)
        and Path(
            payload.get("config", {})
            .get("data", {})
            .get("feature_file", "")
        )
        == Path(config["data"]["feature_file"])
        and all(
            saved_model.get(key) == value
            for key, value in VARIANTS[variant].items()
        )
    )


def _finite_checkpoint(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    nonfinite: list[str] = []
    for name, value in payload.get("model_state", {}).items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            nonfinite.append(f"model_state.{name}")
    history = payload.get("history", [])
    for index, epoch in enumerate(history):
        if not isinstance(epoch, dict):
            continue
        for name, value in epoch.items():
            if isinstance(value, (float, int)) and not np.isfinite(value):
                nonfinite.append(f"history[{index}].{name}")
    if nonfinite:
        raise ValueError(
            f"Non-finite checkpoint/history at {path}: {nonfinite[:20]}"
        )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "epoch": int(payload.get("epoch", -1)),
        "finite": True,
    }


def _assert_finite_numbers(payload: Any, prefix: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _assert_finite_numbers(value, f"{prefix}.{key}")
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _assert_finite_numbers(value, f"{prefix}[{index}]")
    elif isinstance(payload, (float, int)) and not isinstance(payload, bool):
        if not np.isfinite(payload):
            raise ValueError(f"Non-finite value at {prefix}: {payload}")


def _finite_anchor_checkpoint(
    checkpoint: Path,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    state_file = checkpoint / "pfa_erc_training.json"
    if not state_file.exists():
        raise FileNotFoundError(f"Missing Anchor state: {state_file}")
    state = _read_json(state_file)
    _assert_finite_numbers(state, "anchor_state")
    _assert_finite_numbers(evaluation, "anchor_evaluation")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _directory_sha256(checkpoint),
        "best_epoch": int(state.get("best_epoch", -1)),
        "finite_state_and_evaluation": True,
    }


def _aggregate_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [report["metrics"] for report in reports]
    output = {
        name: _summary(
            [
                float(value)
                for item in metrics
                if (value := _nested(item, path)) is not None
            ]
        )
        for name, path in METRIC_PATHS.items()
    }
    classification_report = metrics[0]["classification_report"]
    excluded = {"accuracy", "macro avg", "weighted avg"}
    emotions = [
        name
        for name, values in classification_report.items()
        if name not in excluded and isinstance(values, dict)
    ]
    output["per_emotion_f1"] = {
        emotion: _summary(
            [
                float(item["classification_report"][emotion]["f1-score"])
                for item in metrics
            ]
        )
        for emotion in emotions
    }
    return output


def _anchor_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_settings(config).get("reuse_existing", True))
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
        audit = _finite_anchor_checkpoint(checkpoint, evaluation)
        individual = root / "validation" / f"anchor_seed{seed}.json"
        save_json(evaluation, individual)
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "training": training,
                "checkpoint_audit": audit,
                "evaluation": evaluation,
            }
        )
    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "independent_full_anchor_seed": True,
            "test_used": False,
        },
        "runs": runs,
        "aggregate": _aggregate_reports(
            [run["evaluation"] for run in runs]
        ),
    }
    save_json(payload, root / "validation" / "anchor_seeds.json")
    return payload


def _feature_stage(
    config: dict[str, Any], device: str | None
) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_settings(config).get("reuse_existing", True))
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
                    f"Feature lineage mismatch: {feature_file}"
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
        "independent_feature_cache_per_anchor_seed": True,
        "splits": ["train", "validation"],
        "test_used": False,
        "runs": runs,
    }
    save_json(payload, root / "validation" / "feature_manifest.json")
    return payload


def _train_variant(
    config: dict[str, Any],
    device: str | None,
    variant: str,
) -> dict[str, Any]:
    root = _artifact_dir(config)
    reuse = bool(_settings(config).get("reuse_existing", True))
    runs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        seed_config = _seed_config(config, seed, variant=variant)
        feature_file = Path(seed_config["data"]["feature_file"])
        if not feature_file.exists():
            raise FileNotFoundError(
                f"Missing seed feature cache: {feature_file}"
            )
        checkpoint = Path(seed_config["train_erc"]["checkpoint"])
        if reuse and _checkpoint_matches(
            checkpoint, seed_config, seed, variant
        ):
            training = {
                "checkpoint": str(checkpoint),
                "seed": seed,
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
        audit = _finite_checkpoint(checkpoint)
        evaluation = evaluate_erc(
            seed_config,
            device=device,
            checkpoint_path=checkpoint,
            split="validation",
            save_report=False,
        )
        _assert_finite_numbers(evaluation, f"{variant}.evaluation.seed{seed}")
        individual = root / "validation" / f"{variant}_seed{seed}.json"
        save_json(evaluation, individual)
        runs.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "feature_file": str(feature_file),
                "training": training,
                "checkpoint_audit": audit,
                "evaluation": evaluation,
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
            "same_anchor_seed_pairing": True,
            "test_used": False,
        },
        "runs": runs,
        "aggregate": _aggregate_reports(
            [run["evaluation"] for run in runs]
        ),
    }
    save_json(payload, root / "validation" / f"{variant}_seeds.json")
    return payload


def _parameter_audit(config: dict[str, Any]) -> dict[str, Any]:
    label_file = Path(config["data"]["processed_dir"]) / "label_map.json"
    if not label_file.exists():
        raise FileNotFoundError(
            f"Prepare the dataset before parameter audit: {label_file}"
        )
    num_classes = len(_read_json(label_file))
    models: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_config = _seed_config(
            config, _seeds(config)[0], variant=variant
        )
        model = _build_model(
            variant_config, num_classes, torch.device("cpu")
        )
        active = {
            name: int(parameter.numel())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        models[variant] = {
            "total_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "trainable_parameters": int(sum(active.values())),
            "active_parameter_names": sorted(active),
            "active_parameter_signature": _json_fingerprint(active),
        }
    role = models["roletrans"]
    control = models["capacity_direct"]
    exact = (
        role["total_parameters"] == control["total_parameters"]
        and role["trainable_parameters"] == control["trainable_parameters"]
        and role["active_parameter_signature"]
        == control["active_parameter_signature"]
    )
    if not exact:
        raise AssertionError(
            "RoleTrans and capacity_direct are not exactly capacity matched"
        )
    payload = {
        "status": "passed",
        "dataset": _dataset_name(config),
        "models": models,
        "capacity_match": {
            "reference": "roletrans",
            "control": "capacity_direct",
            "exact_total_parameters": True,
            "exact_trainable_parameters": True,
            "exact_active_parameter_signature": True,
            "control_has_recursive_belief_transition": False,
        },
        "test_used": False,
    }
    save_json(
        payload, _artifact_dir(config) / "reports" / "parameter_audit.json"
    )
    return payload


def _collect_outputs(
    config: dict[str, Any],
    checkpoint_path: Path,
    split: str,
    device: torch.device,
    variant: str,
) -> tuple[list[dict[str, Any]], dict[int, str], int]:
    store = load_feature_store(config["data"]["feature_file"])
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    saved_model = checkpoint.get("config", {}).get("model", {})
    mismatches = {
        key: {"expected": value, "stored": saved_model.get(key)}
        for key, value in VARIANTS[variant].items()
        if saved_model.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"Checkpoint variant mismatch at {checkpoint_path}: {mismatches}"
        )
    label_to_id = {
        str(key): int(value)
        for key, value in checkpoint["label_to_id"].items()
    }
    id_to_label = {value: key for key, value in label_to_id.items()}
    model = _build_model(config, len(label_to_id), device)
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
            gate = (
                auxiliary["transition_gate"]
                if variant in {"roletrans", "capacity_direct"}
                else auxiliary["context_gate"]
            ).squeeze(-1)
            for row, dialogue_id in enumerate(batch["dialogue_ids"]):
                length = int(batch["mask"][row].sum())
                records.append(
                    {
                        "dialogue_id": str(dialogue_id),
                        "logits": logits[row, :length].float().cpu(),
                        "anchor_logits": batch["current_logits"][
                            row, :length
                        ].float().cpu(),
                        "labels": batch["labels"][
                            row, :length
                        ].long().cpu(),
                        "gate": gate[row, :length].float().cpu(),
                    }
                )
    return records, id_to_label, seed


def _anchor_records(
    records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            **record,
            "logits": record["anchor_logits"],
            "gate": torch.zeros_like(record["gate"]),
        }
        for record in records
    ]


def _identity_temperature(
    minimum: float, maximum: float
) -> dict[str, Any]:
    fraction = (1.0 - minimum) / (maximum - minimum)
    raw_bias = float(np.log(fraction / (1.0 - fraction)))
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
    settings = _settings(config).get("calibration", {})
    folds = int(settings.get("folds", 5))
    fold_seed = int(settings.get("fold_seed", 2014))
    minimum = float(settings.get("minimum_temperature", 0.25))
    maximum = float(settings.get("maximum_temperature", 4.0))
    max_iter = int(settings.get("max_iter", 100))
    device_obj = get_device(device)
    model_runs: dict[str, list[dict[str, Any]]] = {
        name: [] for name in MODEL_ORDER
    }
    for seed in _seeds(config):
        cached_role_records: list[dict[str, Any]] | None = None
        cached_labels: dict[int, str] | None = None
        for model_name in MODEL_ORDER:
            source_variant = (
                "roletrans" if model_name == "anchor" else model_name
            )
            seed_config = _seed_config(
                config, seed, variant=source_variant
            )
            checkpoint = Path(seed_config["train_erc"]["checkpoint"])
            if model_name == "anchor" and cached_role_records is not None:
                records = _anchor_records(cached_role_records)
                assert cached_labels is not None
                id_to_label = cached_labels
                stored_seed = seed
            else:
                records, id_to_label, stored_seed = _collect_outputs(
                    seed_config,
                    checkpoint,
                    "validation",
                    device_obj,
                    source_variant,
                )
                if source_variant == "roletrans":
                    cached_role_records = records
                    cached_labels = id_to_label
                if model_name == "anchor":
                    records = _anchor_records(records)
            if stored_seed != seed:
                raise ValueError(
                    f"Checkpoint seed mismatch for {model_name} seed {seed}"
                )
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
            fitted = _fit_temperature(
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
            model_runs[model_name].append(
                {
                    "seed": seed,
                    "source_variant": source_variant,
                    "source_checkpoint": str(checkpoint),
                    "uncalibrated": uncalibrated,
                    "cross_fitted": cross_fitted,
                    "fitted_parameters": fitted,
                    "cross_fitted_nll_improvement": improvement,
                }
            )

    outputs: dict[str, Any] = {}
    calibrator_dir = root / "calibration"
    for model_name, provisional in model_runs.items():
        improvements = [
            float(run["cross_fitted_nll_improvement"])
            for run in provisional
        ]
        positive_required = max(2, (len(improvements) + 1) // 2)
        selected = (
            mean(improvements) > 0.0
            and sum(value > 0.0 for value in improvements)
            >= positive_required
        )
        selected_method = "scalar" if selected else "identity"
        final_runs: list[dict[str, Any]] = []
        for run in provisional:
            fitted = run.pop("fitted_parameters")
            parameters = (
                fitted
                if selected
                else _identity_temperature(minimum, maximum)
            )
            calibrator_file = (
                calibrator_dir
                / f"{model_name}_{selected_method}_seed{run['seed']}.json"
            )
            calibrator = {
                "format_version": "pfa-erc-temperature-v1",
                "experiment": 14,
                "model_name": model_name,
                "selection": selected_method,
                "model_seed": int(run["seed"]),
                "source_checkpoint": run["source_checkpoint"],
                "fit_split": "validation",
                "dataset": _dataset_name(config),
                "parameters": parameters,
            }
            save_json(calibrator, calibrator_file)
            run["selected_calibrator"] = str(calibrator_file)
            final_runs.append(run)
        outputs[model_name] = {
            "selected_method": selected_method,
            "cross_fitted_nll_improvement": _summary(improvements),
            "positive_improvement_seeds": sum(
                value > 0.0 for value in improvements
            ),
            "runs": final_runs,
        }
    payload = {
        "status": "completed",
        "protocol": {
            "dataset": _dataset_name(config),
            "split": "validation",
            "seeds": _seeds(config),
            "fold_unit": "dialogue",
            "folds": folds,
            "fold_seed": fold_seed,
            "positive_temperature_preserves_argmax": True,
            "test_used": False,
        },
        "models": outputs,
        "test_locked": True,
    }
    save_json(payload, root / "validation" / "calibration.json")
    return payload


def _paired_validation_contrast(
    reference: dict[str, Any],
    control: dict[str, Any],
) -> dict[str, Any]:
    def by_seed(payload: dict[str, Any], name: str) -> dict[int, dict[str, Any]]:
        output: dict[int, dict[str, Any]] = {}
        for run in payload["runs"]:
            seed = int(run["seed"])
            if seed in output:
                raise ValueError(f"Duplicate {name} seed in paired contrast: {seed}")
            output[seed] = run
        return output

    reference_by_seed = by_seed(reference, "reference")
    control_by_seed = by_seed(control, "control")
    if set(reference_by_seed) != set(control_by_seed):
        raise ValueError(
            "Paired validation contrasts require identical seed identities: "
            f"reference={sorted(reference_by_seed)}, "
            f"control={sorted(control_by_seed)}"
        )

    output: dict[str, Any] = {}
    for metric, path in METRIC_PATHS.items():
        values: list[float] = []
        per_seed: list[dict[str, Any]] = []
        for seed in sorted(reference_by_seed):
            reference_run = reference_by_seed[seed]
            control_run = control_by_seed[seed]
            reference_value = _nested(
                reference_run["evaluation"]["metrics"], path
            )
            control_value = _nested(
                control_run["evaluation"]["metrics"], path
            )
            if reference_value is None or control_value is None:
                continue
            difference = (
                float(control_value) - float(reference_value)
                if metric in {"nll", "brier", "ece"}
                else float(reference_value) - float(control_value)
            )
            values.append(difference)
            per_seed.append(
                {
                    "seed": seed,
                    "reference": float(reference_value),
                    "control": float(control_value),
                    "advantage": difference,
                }
            )
        tolerance = 1e-12
        positive = sum(value > tolerance for value in values)
        negative = sum(value < -tolerance for value in values)
        tied = len(values) - positive - negative
        output[metric] = {
            **_summary(values),
            "advantage_definition": (
                "control-reference"
                if metric in {"nll", "brier", "ece"}
                else "reference-control"
            ),
            "per_seed": per_seed,
            "positive_seeds": positive,
            "negative_seeds": negative,
            "tied_seeds": tied,
            "total_seeds": len(values),
            "direction": f"+{positive}/-{negative}/={tied} of {len(values)}",
            "seed_identity_checked": True,
        }
    return output


def _validation_summary(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    required = [
        root / "validation" / "anchor_seeds.json",
        root / "validation" / "calibration.json",
        root / "reports" / "parameter_audit.json",
        *[
            root / "validation" / f"{variant}_seeds.json"
            for variant in VARIANTS
        ],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Experiment 14 validation is incomplete: " + ", ".join(missing)
        )
    anchor = _read_json(required[0])
    calibration = _read_json(required[1])
    parameter_audit = _read_json(required[2])
    variants = {
        variant: _read_json(
            root / "validation" / f"{variant}_seeds.json"
        )
        for variant in VARIANTS
    }
    roletrans = variants["roletrans"]
    comparisons = {
        control: _paired_validation_contrast(
            roletrans, variants[control]
        )
        for control in (
            "capacity_direct",
            "contextual",
            "speaker_aware",
        )
    }
    payload = {
        "status": "completed",
        "completed_at": _utc_now(),
        "protocol": {
            "series": "14A-14C reviewer closure",
            "dataset": _dataset_name(config),
            "test_session": config["data"].get(
                "iemocap_test_session"
            ),
            "split": "validation",
            "seeds": _seeds(config),
            "independent_full_anchor_seed": True,
            "same_backbone_and_feature_cache_within_seed": True,
            "capacity_control_pre_registered": True,
            "test_used": False,
            "test_policy": (
                "All variants and calibration rules are frozen before the "
                "Session-5 test cache is created. The final test is reported "
                "once regardless of effect direction."
            ),
        },
        "anchor": anchor["aggregate"],
        "variants": {
            name: result["aggregate"]
            for name, result in variants.items()
        },
        "roletrans_paired_advantages": comparisons,
        "parameter_audit": parameter_audit,
        "calibration": calibration,
        "frozen_for_test": True,
        "test_locked": True,
    }
    save_json(payload, root / "reports" / "validation_summary.json")
    return payload


def _calibrator_map(
    calibration: dict[str, Any]
) -> dict[tuple[str, int], Path]:
    output: dict[tuple[str, int], Path] = {}
    for model_name, payload in calibration["models"].items():
        for run in payload["runs"]:
            output[(model_name, int(run["seed"]))] = Path(
                run["selected_calibrator"]
            )
    return output


def _freeze_before_test(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    validation = root / "reports" / "validation_summary.json"
    calibration = root / "validation" / "calibration.json"
    parameter_audit = root / "reports" / "parameter_audit.json"
    for path in (validation, calibration, parameter_audit):
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen validation input: {path}")
    if _read_json(parameter_audit).get("status") != "passed":
        raise ValueError("Capacity parameter audit did not pass")
    if not bool(config["data"].get("iemocap_defer_test", False)):
        raise ValueError(
            "Experiment 14 requires data.iemocap_defer_test=true"
        )
    processed_test = Path(config["data"]["processed_dir"]) / "test.jsonl"
    if not processed_test.exists():
        raise FileNotFoundError(
            f"Missing deferred test placeholder: {processed_test}"
        )
    if processed_test.read_text(encoding="utf-8").strip():
        raise ValueError(
            "The processed Experiment-14 test split is not empty; "
            "use a fresh processed_dir before claiming an unseen test"
        )
    forbidden = [
        root / "reports" / "final_test.json",
        root / "locks" / "final_test.lock.json",
    ]
    forbidden.extend((root / "test").glob("*.json"))
    forbidden.extend((root / "test_features").glob("*.pt"))
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise FileExistsError(
            "The confirmatory test is not pristine; found: "
            + ", ".join(existing)
        )
    calibration_payload = _read_json(calibration)
    calibrators = _calibrator_map(calibration_payload)
    inputs: list[dict[str, Any]] = []
    for seed in _seeds(config):
        feature_file = Path(
            _seed_config(config, seed)["data"]["feature_file"]
        )
        for variant in VARIANTS:
            checkpoint = Path(
                _seed_config(
                    config, seed, variant=variant
                )["train_erc"]["checkpoint"]
            )
            calibrator = calibrators[(variant, seed)]
            for path in (feature_file, checkpoint, calibrator):
                if not path.exists():
                    raise FileNotFoundError(f"Missing frozen input: {path}")
            inputs.append(
                {
                    "seed": seed,
                    "model_name": variant,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha256(checkpoint),
                    "validation_feature_file": str(feature_file),
                    "validation_feature_sha256": _sha256(feature_file),
                    "calibrator": str(calibrator),
                    "calibrator_sha256": _sha256(calibrator),
                }
            )
        anchor_calibrator = calibrators[("anchor", seed)]
        anchor_checkpoint = Path(
            _seed_config(
                config, seed
            )["train_roberta"]["checkpoint"]
        )
        inputs.append(
            {
                "seed": seed,
                "model_name": "anchor",
                "anchor_checkpoint": str(anchor_checkpoint),
                "anchor_checkpoint_sha256": _directory_sha256(
                    anchor_checkpoint
                ),
                "calibrator": str(anchor_calibrator),
                "calibrator_sha256": _sha256(anchor_calibrator),
            }
        )
    payload = {
        "status": "ready",
        "experiment": 14,
        "dataset": _dataset_name(config),
        "test_session": config["data"].get("iemocap_test_session"),
        "seeds": _seeds(config),
        "variants": VARIANTS,
        "validation_summary": str(validation),
        "validation_summary_sha256": _sha256(validation),
        "calibration_summary": str(calibration),
        "calibration_summary_sha256": _sha256(calibration),
        "parameter_audit": str(parameter_audit),
        "parameter_audit_sha256": _sha256(parameter_audit),
        "inputs": inputs,
        "test_cache_created": False,
        "processed_test_was_empty": True,
        "test_read": False,
    }
    payload["input_fingerprint"] = _json_fingerprint(payload)
    freeze_file = root / "locks" / "frozen_before_test.json"
    _atomic_json(payload, freeze_file)
    return payload


def _verify_frozen_inputs(config: dict[str, Any]) -> dict[str, Any]:
    root = _artifact_dir(config)
    freeze_file = root / "locks" / "frozen_before_test.json"
    if not freeze_file.exists():
        raise FileNotFoundError(
            "Run Experiment 14 check-final before the final test"
        )
    frozen = _read_json(freeze_file)
    if (
        frozen.get("status") != "ready"
        or frozen.get("test_read") is not False
        or frozen.get("dataset") != _dataset_name(config)
        or frozen.get("seeds") != _seeds(config)
    ):
        raise ValueError("Frozen Experiment 14 manifest is invalid")
    fingerprint_payload = dict(frozen)
    stored_fingerprint = fingerprint_payload.pop(
        "input_fingerprint", None
    )
    if stored_fingerprint != _json_fingerprint(fingerprint_payload):
        raise ValueError("Frozen Experiment 14 manifest was modified")
    for path_key, hash_key in (
        ("validation_summary", "validation_summary_sha256"),
        ("calibration_summary", "calibration_summary_sha256"),
        ("parameter_audit", "parameter_audit_sha256"),
    ):
        path = Path(frozen[path_key])
        if not path.exists() or _sha256(path) != frozen[hash_key]:
            raise ValueError(f"Frozen protocol input changed: {path}")
    for item in frozen["inputs"]:
        for path_key, hash_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("validation_feature_file", "validation_feature_sha256"),
            ("calibrator", "calibrator_sha256"),
        ):
            if path_key not in item:
                continue
            path = Path(item[path_key])
            if not path.exists() or _sha256(path) != item[hash_key]:
                raise ValueError(f"Frozen input changed: {path}")
        if "anchor_checkpoint" in item:
            anchor_checkpoint = Path(item["anchor_checkpoint"])
            if (
                _directory_sha256(anchor_checkpoint)
                != item["anchor_checkpoint_sha256"]
            ):
                raise ValueError(
                    f"Frozen Anchor changed: {anchor_checkpoint}"
                )
    return frozen


def _prepare_locked_test(config: dict[str, Any]) -> dict[str, Any]:
    """Parse the held-out IEMOCAP session only after input freezing."""

    if _dataset_name(config) != "iemocap":
        raise ValueError(
            "Experiment 14 locked-test preparation currently requires IEMOCAP"
        )
    test_config = copy.deepcopy(config)
    test_config["data"]["iemocap_defer_test"] = False
    stats = prepare_dataset(test_config)
    test_stats = stats.get("splits", {}).get("test", {})
    if int(test_stats.get("dialogues", 0)) <= 0:
        raise ValueError("The locked IEMOCAP test split is empty")
    return stats


def _metric_from_probabilities(
    labels: list[int],
    probabilities: list[list[float]],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    probability = torch.tensor(probabilities, dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return _metrics_from_logits(
        probability.clamp_min(1e-8).log(),
        label_tensor,
        id_to_label,
    )


def _test_seed(
    config: dict[str, Any],
    seed: int,
    device: str | None,
    calibrators: dict[tuple[str, int], Path],
) -> dict[str, Any]:
    device_obj = get_device(device)
    test_feature_file = (
        _artifact_dir(config) / "test_features" / f"seed{seed}.pt"
    )
    role_test_config = _test_seed_config(
        config, seed, "roletrans"
    )
    if test_feature_file.exists():
        store = load_feature_store(test_feature_file)
        if set(store["splits"]) != {"test"}:
            raise ValueError(
                f"Expected test-only cache: {test_feature_file}"
            )
    else:
        extract_features(role_test_config, device=device)
    all_records: dict[str, list[dict[str, Any]]] = {}
    id_to_label: dict[int, str] | None = None
    for variant in VARIANTS:
        variant_config = _test_seed_config(config, seed, variant)
        checkpoint = Path(variant_config["train_erc"]["checkpoint"])
        records, labels, stored_seed = _collect_outputs(
            variant_config,
            checkpoint,
            "test",
            device_obj,
            variant,
        )
        if stored_seed != seed:
            raise ValueError(
                f"Test checkpoint seed mismatch for {variant}"
            )
        if id_to_label is None:
            id_to_label = labels
        elif labels != id_to_label:
            raise ValueError("Label maps differ across test variants")
        all_records[variant] = records
    assert id_to_label is not None
    all_records["anchor"] = _anchor_records(all_records["roletrans"])

    results: dict[str, Any] = {}
    prediction_records: list[dict[str, Any]] = []
    reference = all_records["roletrans"]
    for index, record in enumerate(reference):
        prediction_records.append(
            {
                "dialogue_id": record["dialogue_id"],
                "labels": record["labels"].tolist(),
            }
        )
    for model_name in MODEL_ORDER:
        records = all_records[model_name]
        if len(records) != len(reference):
            raise ValueError("Test dialogue counts differ across variants")
        logits, labels, gate = _concatenate(
            records, list(range(len(records)))
        )
        calibrator_file = calibrators[(model_name, seed)]
        calibrator = _read_json(calibrator_file)
        temperature = _temperature_from_parameters(
            gate, calibrator["parameters"]
        )
        calibrated_logits = logits / temperature.unsqueeze(-1)
        results[model_name] = {
            "raw": _metrics_from_logits(
                logits, labels, id_to_label
            ),
            "calibrated": _metrics_from_logits(
                calibrated_logits, labels, id_to_label
            ),
            "calibrator": str(calibrator_file),
            "calibrator_sha256": _sha256(calibrator_file),
            "temperature_mean": float(temperature.mean()),
            "argmax_changes": int(
                logits.argmax(dim=-1)
                .ne(calibrated_logits.argmax(dim=-1))
                .sum()
            ),
        }
        offset = 0
        for index, record in enumerate(records):
            length = int(record["labels"].numel())
            if (
                record["dialogue_id"]
                != prediction_records[index]["dialogue_id"]
                or record["labels"].tolist()
                != prediction_records[index]["labels"]
            ):
                raise ValueError(
                    "Test records are not paired across variants"
                )
            probability = torch.softmax(
                calibrated_logits[offset : offset + length], dim=-1
            )
            prediction_records[index][
                f"{model_name}_probabilities"
            ] = probability.tolist()
            offset += length
    return {
        "seed": seed,
        "split": "test",
        "test_feature_file": str(test_feature_file),
        "test_feature_sha256": _sha256(test_feature_file),
        "models": results,
        "id_to_label": {
            str(key): value for key, value in id_to_label.items()
        },
        "predictions": prediction_records,
    }


def _aggregate_test(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for model_name in MODEL_ORDER:
        output[model_name] = {}
        for calibration in ("raw", "calibrated"):
            output[model_name][calibration] = {
                metric: _summary(
                    [
                        float(
                            _nested(
                                run["models"][model_name][calibration],
                                path,
                            )
                        )
                        for run in runs
                    ]
                )
                for metric, path in METRIC_PATHS.items()
            }
        output[model_name]["temperature_mean"] = _summary(
            [
                float(
                    run["models"][model_name]["temperature_mean"]
                )
                for run in runs
            ]
        )
        first_report = runs[0]["models"][model_name]["calibrated"][
            "classification_report"
        ]
        excluded = {"accuracy", "macro avg", "weighted avg"}
        emotions = [
            emotion
            for emotion, values in first_report.items()
            if emotion not in excluded and isinstance(values, dict)
        ]
        output[model_name]["per_emotion"] = {
            emotion: {
                metric: _summary(
                    [
                        float(
                            run["models"][model_name]["calibrated"][
                                "classification_report"
                            ][emotion][metric]
                        )
                        for run in runs
                    ]
                )
                for metric in ("precision", "recall", "f1-score")
            }
            for emotion in emotions
        }
    return output


def _bootstrap_metric(
    records: list[dict[str, Any]],
    id_to_label: dict[int, str],
    model_name: str,
    metric: str,
) -> float:
    labels: list[int] = []
    probabilities: list[list[float]] = []
    for record in records:
        labels.extend(int(value) for value in record["labels"])
        probabilities.extend(
            record[f"{model_name}_probabilities"]
        )
    if metric in {"weighted_f1", "macro_f1", "accuracy"}:
        predictions = [
            max(range(len(row)), key=lambda index: row[index])
            for row in probabilities
        ]
        if metric == "accuracy":
            return float(accuracy_score(labels, predictions))
        return float(
            f1_score(
                labels,
                predictions,
                labels=sorted(id_to_label),
                average=(
                    "weighted"
                    if metric == "weighted_f1"
                    else "macro"
                ),
                zero_division=0,
            )
        )
    metrics = _metric_from_probabilities(
        labels, probabilities, id_to_label
    )
    return float(_nested(metrics, METRIC_PATHS[metric]))


def _hierarchical_bootstrap(
    runs: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError(
            "experiment14.bootstrap_samples must be at least 100"
        )
    generator = np.random.default_rng(seed)
    controls = (
        "capacity_direct",
        "contextual",
        "speaker_aware",
        "anchor",
    )
    metrics = tuple(METRIC_PATHS)
    draws = {
        control: {metric: [] for metric in metrics}
        for control in controls
    }
    for _ in range(samples):
        sampled_seed_indices = generator.integers(
            0, len(runs), size=len(runs)
        )
        per_control = {
            control: {metric: [] for metric in metrics}
            for control in controls
        }
        for seed_index in sampled_seed_indices:
            run = runs[int(seed_index)]
            source = run["predictions"]
            indices = generator.integers(
                0, len(source), size=len(source)
            )
            records = [source[int(index)] for index in indices]
            labels = {
                int(key): value
                for key, value in run["id_to_label"].items()
            }
            for metric in metrics:
                role = _bootstrap_metric(
                    records, labels, "roletrans", metric
                )
                for control in controls:
                    baseline = _bootstrap_metric(
                        records, labels, control, metric
                    )
                    difference = (
                        baseline - role
                        if metric in {"nll", "brier", "ece"}
                        else role - baseline
                    )
                    per_control[control][metric].append(difference)
        for control in controls:
            for metric in metrics:
                draws[control][metric].append(
                    mean(per_control[control][metric])
                )
    output: dict[str, Any] = {}
    for control in controls:
        output[control] = {}
        for metric, values in draws[control].items():
            array = np.asarray(values, dtype=np.float64)
            output[control][metric] = {
                "mean_advantage": float(array.mean()),
                "ci95": [
                    float(np.quantile(array, 0.025)),
                    float(np.quantile(array, 0.975)),
                ],
                "probability_positive": float((array > 0.0).mean()),
                "samples": samples,
            }
    return output


def _final_test(
    config: dict[str, Any],
    device: str | None,
    *,
    confirm: bool,
    resume: bool,
) -> dict[str, Any]:
    if not confirm:
        raise ValueError(
            "Experiment 14 final test requires explicit confirmation"
        )
    frozen = _verify_frozen_inputs(config)
    root = _artifact_dir(config)
    report_file = root / "reports" / "final_test.json"
    lock_file = root / "locks" / "final_test.lock.json"
    if report_file.exists():
        raise FileExistsError(
            f"Final test is already complete: {report_file}"
        )
    if lock_file.exists() and not resume:
        raise FileExistsError(
            f"Final-test lock exists: {lock_file}; use --resume"
        )
    lock = {
        "status": "running",
        "created_at": _utc_now(),
        "frozen_input_fingerprint": frozen["input_fingerprint"],
        "completed_seeds": [],
    }
    if lock_file.exists():
        lock = _read_json(lock_file)
        if (
            lock.get("frozen_input_fingerprint")
            != frozen["input_fingerprint"]
        ):
            raise ValueError("Frozen inputs changed before resume")
        lock["status"] = "resuming"
        lock["resumed_at"] = _utc_now()
    _atomic_json(lock, lock_file)
    test_stats = _prepare_locked_test(config)
    lock["test_prepared_at"] = _utc_now()
    lock["test_split_statistics"] = test_stats["splits"]["test"]
    _atomic_json(lock, lock_file)
    calibration = _read_json(
        root / "validation" / "calibration.json"
    )
    calibrators = _calibrator_map(calibration)
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
                if int(run.get("seed", -1)) != seed:
                    raise ValueError(
                        f"Wrong seed in existing report: {individual}"
                    )
            else:
                run = _test_seed(
                    config, seed, device, calibrators
                )
                save_json(run, individual)
            for model_name in MODEL_ORDER:
                if int(
                    run["models"][model_name]["argmax_changes"]
                ) != 0:
                    raise AssertionError(
                        "Positive temperature changed an argmax"
                    )
            runs.append(run)
            lock["completed_seeds"] = [
                int(item["seed"]) for item in runs
            ]
            _atomic_json(lock, lock_file)
        samples = int(
            _settings(config).get("bootstrap_samples", 2000)
        )
        bootstrap_seed = int(
            _settings(config).get("bootstrap_seed", 2014)
        )
        result = {
            "status": "completed",
            "completed_at": _utc_now(),
            "protocol": {
                **frozen,
                "test_read": True,
                "one_time_confirmatory_test": True,
                "selection_after_test": False,
                "all_variants_reported": True,
                "bootstrap_unit": (
                    "dialogue nested within complete model seed"
                ),
                "bootstrap_samples": samples,
                "bootstrap_seed": bootstrap_seed,
            },
            "aggregate": _aggregate_test(runs),
            "paired_hierarchical_bootstrap": _hierarchical_bootstrap(
                runs, samples, bootstrap_seed
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


def run_experiment14(
    config: dict[str, Any],
    *,
    stage: str,
    device: str | None = None,
    confirm_final_test: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the reviewer-closure protocol without an accidental test read."""

    _seeds(config)
    _artifact_dir(config).mkdir(parents=True, exist_ok=True)
    if stage == "check":
        return {
            "status": "ready",
            "experiment": 14,
            "dataset": _dataset_name(config),
            "seeds": _seeds(config),
            "variants": VARIANTS,
            "test_read": False,
        }
    if stage == "prepare":
        return prepare_dataset(config)
    if stage == "anchors":
        return _anchor_stage(config, device)
    if stage == "features":
        return _feature_stage(config, device)
    if stage == "parameter-audit":
        return _parameter_audit(config)
    stage_to_variant = {
        "roletrans": "roletrans",
        "capacity-direct": "capacity_direct",
        "contextual": "contextual",
        "speaker-aware": "speaker_aware",
    }
    if stage in stage_to_variant:
        return _train_variant(
            config, device, stage_to_variant[stage]
        )
    if stage == "baselines":
        return {
            name: _train_variant(config, device, name)
            for name in (
                "capacity_direct",
                "contextual",
                "speaker_aware",
            )
        }
    if stage == "calibrate":
        return _calibration_stage(config, device)
    if stage == "summarize-validation":
        return _validation_summary(config)
    if stage == "all-validation":
        prepare_dataset(config)
        _anchor_stage(config, device)
        _feature_stage(config, device)
        _parameter_audit(config)
        for variant in VARIANTS:
            _train_variant(config, device, variant)
        _calibration_stage(config, device)
        return _validation_summary(config)
    if stage == "check-final":
        return _freeze_before_test(config)
    if stage == "final-test":
        return _final_test(
            config,
            device,
            confirm=confirm_final_test,
            resume=resume,
        )
    if stage == "summarize-final":
        report = _artifact_dir(config) / "reports" / "final_test.json"
        if not report.exists():
            raise FileNotFoundError(f"Missing final report: {report}")
        return _read_json(report)
    raise ValueError(f"Unknown Experiment 14 stage: {stage}")
