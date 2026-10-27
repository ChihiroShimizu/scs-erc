from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, recursively resolving its optional ``_base_`` entry."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    if "_base_" in current:
        base_path = (path.parent / current["_base_"]).resolve()
        current = _deep_merge(load_config(base_path), current)
    current["_config_path"] = str(path)
    return current


def apply_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply command-line overrides such as ``model.future_steps=6``."""
    result = copy.deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}; expected key=value")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor: dict[str, Any] = result
        keys = dotted_key.split(".")
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = value
    return result


def resolve_project_paths(config: dict[str, Any], project_root: str | Path) -> dict[str, Any]:
    """Resolve known artifact/data paths relative to the project root."""
    result = copy.deepcopy(config)
    root = Path(project_root).resolve()
    path_keys = {
        "data": ["raw_dir", "processed_dir", "feature_file"],
        "features": ["audit_report_file", "encoder_checkpoint"],
        "train_hitft": ["checkpoint", "warm_start_checkpoint"],
        "train_erc": ["checkpoint"],
        "train_roberta": ["checkpoint"],
        "evaluation": ["report_file"],
        "evaluation_roberta": ["report_file"],
        "transition_evaluation": ["report_file"],
        "correction_analysis": ["checkpoint", "report_file"],
        "train_state": ["checkpoint"],
        "state_evaluation": ["report_file"],
        "boundary_model": ["state_checkpoint"],
        "boundary_train": ["checkpoint"],
        "boundary_evaluation": ["report_file"],
        "boundary_oracle": ["report_file"],
        "influence_audit": [
            "checkpoint",
            "report_file",
            "records_file",
            "summary_file",
        ],
        "calibration": ["report_file", "calibrator_dir"],
        "final_evaluation": [
            "validation_report_file",
            "checkpoint",
            "calibrator_dir",
            "report_file",
            "individual_report_dir",
            "lock_file",
        ],
        "experiment9": ["artifact_dir"],
        "experiment11": ["artifact_dir", "source_final_report"],
        "experiment13": [
            "artifact_dir",
            "emotioncue_train_file",
            "feature_file_pattern",
            "test_feature_file_pattern",
        ],
        "experiment14": ["artifact_dir"],
        "experiment15": ["artifact_dir", "source_experiment14_dir"],
    }
    for section, keys in path_keys.items():
        for key in keys:
            value = result.get(section, {}).get(key)
            if value and not Path(value).is_absolute():
                result[section][key] = str(root / value)
    output_dir = result.get("experiment", {}).get("output_dir")
    if output_dir and not Path(output_dir).is_absolute():
        result["experiment"]["output_dir"] = str(root / output_dir)
    result["_project_root"] = str(root)
    return result
