from __future__ import annotations

import copy
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from .common import save_json
from .influence_audit import audit_context_influence


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _series(values: list[Any]) -> dict[str, Any]:
    numeric = [float(value) for value in values if value is not None]
    positive = sum(value > 0.0 for value in numeric)
    negative = sum(value < 0.0 for value in numeric)
    zero = len(numeric) - positive - negative
    return {
        "values": numeric,
        "mean": mean(numeric) if numeric else None,
        "std": pstdev(numeric) if numeric else None,
        "count": len(numeric),
        "positive_seeds": positive,
        "negative_seeds": negative,
        "zero_seeds": zero,
        "all_same_nonzero_direction": bool(numeric)
        and (positive == len(numeric) or negative == len(numeric)),
    }


def _ci_direction(interval: Any) -> str:
    if not isinstance(interval, list) or len(interval) != 2:
        return "unavailable"
    lower, upper = interval
    if lower is None or upper is None:
        return "unavailable"
    if float(lower) > 0.0:
        return "positive"
    if float(upper) < 0.0:
        return "negative"
    return "crosses_zero"


def _aggregate_effect_entries(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    scalar_fields = (
        "mean_delta_nll",
        "median_delta_nll",
        "helpful_rate",
        "harmful_rate",
        "neutral_rate",
        "unique_context_helpful_flip_rate",
        "unique_context_harmful_flip_rate",
    )
    output = {
        field: _series([entry.get(field) for entry in entries])
        for field in scalar_fields
    }
    ci_directions = [
        _ci_direction(entry.get("mean_delta_nll_95ci"))
        for entry in entries
    ]
    output["within_seed_cluster_ci"] = {
        "directions": ci_directions,
        "positive": ci_directions.count("positive"),
        "negative": ci_directions.count("negative"),
        "crosses_zero": ci_directions.count("crosses_zero"),
        "unavailable": ci_directions.count("unavailable"),
    }
    output["counts"] = {
        "counterfactuals": _series([entry.get("count") for entry in entries]),
        "unique_targets": _series(
            [entry.get("unique_targets") for entry in entries]
        ),
    }
    return output


def _aggregate_contrasts(
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    differences = _series([entry.get("mean_difference") for entry in entries])
    ci_directions = [
        _ci_direction(entry.get("cluster_bootstrap_95ci"))
        for entry in entries
    ]
    return {
        "definition": entries[0].get("definition") if entries else None,
        "mean_difference": differences,
        "within_seed_cluster_ci": {
            "directions": ci_directions,
            "positive": ci_directions.count("positive"),
            "negative": ci_directions.count("negative"),
            "crosses_zero": ci_directions.count("crosses_zero"),
            "unavailable": ci_directions.count("unavailable"),
        },
    }


def _aggregate_seed_results(
    results: list[dict[str, Any]],
    seeds: list[int],
    sampling_seed: int,
    bootstrap_seed: int,
    split: str,
) -> dict[str, Any]:
    baseline_paths = {
        "overall_weighted_f1": ("baseline", "all", "weighted_f1"),
        "overall_macro_f1": ("baseline", "all", "macro_f1"),
        "overall_accuracy": ("baseline", "all", "accuracy"),
        "stable_weighted_f1": ("baseline", "stable", "weighted_f1"),
        "stable_macro_f1": ("baseline", "stable", "macro_f1"),
        "transition_weighted_f1": (
            "baseline",
            "transition",
            "weighted_f1",
        ),
        "transition_macro_f1": ("baseline", "transition", "macro_f1"),
    }
    baseline = {
        name: _series([_nested(result, path) for result in results])
        for name, path in baseline_paths.items()
    }

    interventions = sorted(
        {
            intervention
            for result in results
            for intervention in result.get("effects", {})
        }
    )
    effects: dict[str, Any] = {}
    group_dimensions = (
        "same_speaker",
        "gap_bucket",
        "target_transition_group",
        "target_emotion",
    )
    for intervention in interventions:
        analyses = [
            result["effects"][intervention]
            for result in results
            if intervention in result.get("effects", {})
        ]
        effect_summary: dict[str, Any] = {
            "overall": _aggregate_effect_entries(
                [analysis["overall"] for analysis in analyses]
            )
        }
        for dimension in group_dimensions:
            group_names = sorted(
                {
                    group
                    for analysis in analyses
                    for group in analysis.get(dimension, {})
                }
            )
            effect_summary[dimension] = {
                group: _aggregate_effect_entries(
                    [
                        analysis[dimension][group]
                        for analysis in analyses
                        if group in analysis.get(dimension, {})
                    ]
                )
                for group in group_names
            }
        contrast_names = sorted(
            {
                name
                for analysis in analyses
                for name in analysis.get("contrasts", {})
            }
        )
        effect_summary["contrasts"] = {
            name: _aggregate_contrasts(
                [
                    analysis["contrasts"][name]
                    for analysis in analyses
                    if name in analysis.get("contrasts", {})
                ]
            )
            for name in contrast_names
        }
        effects[intervention] = effect_summary

    same_speaker_checks: dict[str, Any] = {}
    for intervention in interventions:
        contrast = effects[intervention]["contrasts"].get(
            "same_minus_different_speaker", {}
        )
        series = contrast.get("mean_difference", {})
        same_speaker_checks[intervention] = {
            "all_seeds_positive": (
                series.get("count") == len(seeds)
                and series.get("positive_seeds") == len(seeds)
            ),
            "positive_cluster_ci_seeds": contrast.get(
                "within_seed_cluster_ci", {}
            ).get("positive", 0),
        }
    return {
        "status": "completed" if len(results) == len(seeds) else "in_progress",
        "protocol": {
            "model_seeds": seeds,
            "completed_seeds": [int(result["checkpoint_seed"]) for result in results],
            "sampling_seed": sampling_seed,
            "bootstrap_seed": bootstrap_seed,
            "same_counterfactual_sampling_across_model_seeds": True,
            "development_split": split,
        },
        "baseline": baseline,
        "effects": effects,
        "decision_checks": {
            "same_speaker_contrast": same_speaker_checks,
            "same_speaker_positive_for_both_interventions": bool(
                same_speaker_checks
            )
            and all(
                check["all_seeds_positive"]
                for check in same_speaker_checks.values()
            ),
        },
    }


def audit_context_influence_seeds(
    config: dict[str, Any],
    device: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Audit independently trained ERC checkpoints with fixed interventions."""
    audit = config.get("influence_audit", {})
    seeds = [
        int(seed)
        for seed in audit.get(
            "model_seeds",
            config.get("evaluation", {}).get("seeds", [42, 43, 44]),
        )
    ]
    if len(seeds) < 2:
        raise ValueError("8A seed audit requires at least two model seeds")
    sampling_seed = int(audit.get("sampling_seed", 2026))
    bootstrap_seed = int(audit.get("bootstrap_seed", 2027))
    reuse_existing = bool(audit.get("reuse_existing_reports", True))

    base_checkpoint = Path(
        audit.get("checkpoint") or config["train_erc"]["checkpoint"]
    )
    base_report = Path(audit["report_file"])
    base_records = Path(audit["records_file"])
    summary_file = Path(
        audit.get(
            "summary_file",
            base_report.with_name(f"{base_report.stem}_summary.json"),
        )
    )
    selected_split = str(split or audit.get("split", "validation"))

    results: list[dict[str, Any]] = []
    run_references: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = base_checkpoint.with_name(
            f"{base_checkpoint.stem}_seed{seed}{base_checkpoint.suffix}"
        )
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing independently trained checkpoint: {checkpoint}. "
                "Run the 8A three-seed training stage first."
            )
        report = base_report.with_name(
            f"{base_report.stem}_seed{seed}{base_report.suffix}"
        )
        records = base_records.with_name(
            f"{base_records.stem}_seed{seed}{base_records.suffix}"
        )
        if reuse_existing and report.exists() and records.exists():
            result = json.loads(report.read_text(encoding="utf-8"))
            if int(result.get("checkpoint_seed", -1)) != seed:
                raise ValueError(
                    f"Existing report {report} has checkpoint_seed="
                    f"{result.get('checkpoint_seed')}, expected {seed}"
                )
            if str(result.get("split")) != selected_split:
                raise ValueError(
                    f"Existing report {report} used split={result.get('split')!r}, "
                    f"expected {selected_split!r}"
                )
            if int(result.get("sampling_seed", -1)) != sampling_seed:
                raise ValueError(
                    f"Existing report {report} used sampling_seed="
                    f"{result.get('sampling_seed')}, expected {sampling_seed}"
                )
            if int(result.get("bootstrap_seed", -1)) != bootstrap_seed:
                raise ValueError(
                    f"Existing report {report} used bootstrap_seed="
                    f"{result.get('bootstrap_seed')}, expected {bootstrap_seed}"
                )
            print(f"[8A seed audit] reusing seed={seed}: {report}")
        else:
            seed_config = copy.deepcopy(config)
            seed_config.setdefault("influence_audit", {}).update(
                {
                    "checkpoint": str(checkpoint),
                    "report_file": str(report),
                    "records_file": str(records),
                    "sampling_seed": sampling_seed,
                    "bootstrap_seed": bootstrap_seed,
                }
            )
            result = audit_context_influence(
                seed_config,
                device=device,
                checkpoint_path=checkpoint,
                split=selected_split,
            )
        results.append(result)
        run_references.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "report_file": str(report),
                "records_file": str(records),
            }
        )
        partial = _aggregate_seed_results(
            results,
            seeds,
            sampling_seed,
            bootstrap_seed,
            selected_split,
        )
        partial["runs"] = run_references
        save_json(partial, summary_file)

    final = _aggregate_seed_results(
        results,
        seeds,
        sampling_seed,
        bootstrap_seed,
        selected_split,
    )
    final["runs"] = run_references
    final["summary_file"] = str(summary_file)
    save_json(final, summary_file)
    return final


__all__ = ["audit_context_influence_seeds"]
