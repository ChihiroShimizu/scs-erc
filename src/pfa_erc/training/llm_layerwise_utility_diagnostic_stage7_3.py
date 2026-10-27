"""Stage 7.3: layerwise utility learnability and oracle READ-budget frontier.

The frozen Stage 7.1 emotion model is never updated.  Stage 7.3 reuses the
dialogue-disjoint Stage 7.2 partitions and complete current/recent-3 scores,
caches frozen hidden states at several depths, audits utility stability across
seeds, trains equal-capacity probes, and evaluates matched READ budgets.

There is deliberately no MELD-test implementation and no reinforcement
learning path in this module.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from sklearn.metrics import average_precision_score

from pfa_erc.data import read_jsonl
from pfa_erc.training.llm_current_only_gate_stage7_1 import AdaptiveLLMEngine
from pfa_erc.training.llm_early_utility_gate_stage7_2 import (
    VERSION as STAGE7_2_SOURCE_VERSION,
    EarlyBinaryGateHead,
    EarlyFeatureExtractor,
    _last_non_padding_positions,
    _metrics_from_scores,
    _mix_scores,
    _stage7_1_adapter,
    _threshold_diagnostics,
    _tree_sha256,
    counterfactual_utility_target,
    layer_equivalent_cost_ratio,
    locate_decoder_layers,
    teacher_read_label,
    teacher_sample_weight,
)
from pfa_erc.training.llm_evidence_supervision_stage7 import (
    LABELS,
    Target,
    _fit_temperature,
    _metrics,
    _project_root,
    _read_jsonl,
    _read_yaml,
    _resolve,
    _sha256,
    _softmax,
    _target_from_serialized,
    _targets,
    _write_json,
    _write_jsonl,
)
from pfa_erc.training.stage7_3_diagnostics_core import (
    ThresholdRule,
    apply_budget_threshold,
    capacity_matched_pair_bottleneck,
    dialogue_bootstrap_counts,
    exact_random_mask,
    exact_top_k_mask,
    fit_budget_threshold,
    layer_number,
    nll_gain,
    normalise_layer_specs,
    paired_dialogue_ap_bootstrap,
    paired_dialogue_bootstrap,
    probability_entropy,
    utility_stability_report,
)


VERSION = "1.0.1"
EXPERIMENT = "llm_layerwise_utility_diagnostic_stage7_3"
BASE_OBJECTIVES = ("semantic", "utility_v1", "utility_rank")
CONTROL_SUFFIX = "_shuffled"
PAIR_LAYER = "pair_aware_final"


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("llm_layerwise_utility_diagnostic_stage7_3")
    if not isinstance(value, dict):
        raise KeyError(
            "Missing llm_layerwise_utility_diagnostic_stage7_3 configuration section"
        )
    return value


def _paths(
    root: Path, section: dict[str, Any], run_name: str = "validation"
) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get(
            "artifact_dir", "artifacts/llm_layerwise_utility_stage7_3/meld"
        ),
    )
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "cache": base / "cache",
        "probes": base / "probes",
        "scores": base / "scores",
        "reports": base / "reports",
        "preflight": base / "reports" / "preflight.json",
        "utility_audit": base / "reports" / "utility_seed_stability.json",
        "utility_examples": base / "reports" / "utility_examples.jsonl",
        "layer_probe_metrics": base / "reports" / "layer_probe_metrics.jsonl",
        "validation_predictions": base
        / "reports"
        / "validation_predictions.jsonl",
        "summary": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
        "bootstrap_indices": base
        / "reports"
        / "paired_dialogue_bootstrap_indices.npz",
        "decision": base / "reports" / "frozen_decision.json",
        "frontier_csv": base / "reports" / "budget_frontier.csv",
        "runtime": base / "reports" / "runtime_profile.json",
        "runtime_breakdown_csv": base / "reports" / "runtime_breakdown.csv",
        "lock_manifest": base / "lock_manifest.json",
        "manifest": base / "run_manifest.json",
    }


def _stage7_2_base(root: Path, section: dict[str, Any]) -> Path:
    return _resolve(
        root,
        section.get(
            "stage7_2_artifact_dir",
            "artifacts/llm_early_utility_gate_stage7_2/meld",
        ),
    )


def _source_cache(root: Path, section: dict[str, Any], seed: int) -> Path:
    return _stage7_2_base(root, section) / "cache" / f"seed{int(seed)}"


def _source_prepared(root: Path, section: dict[str, Any]) -> Path:
    return _stage7_2_base(root, section) / "prepared"


def _cache_dir(paths: dict[str, Path], seed: int) -> Path:
    return paths["cache"] / f"seed{int(seed)}"


def _probe_dir(
    paths: dict[str, Path], seed: int, layer: str, objective: str
) -> Path:
    return paths["probes"] / f"seed{int(seed)}" / layer / objective


def _load_targets(path: Path) -> list[Target]:
    return [_target_from_serialized(item) for item in _read_jsonl(path)]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _combine_prediction_files(
    path: Path, seed_files: Sequence[tuple[int, Path]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for seed, source in seed_files:
            with source.open("r", encoding="utf-8") as source_handle:
                for line in source_handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["seed"] = int(seed)
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _configured_layers(section: dict[str, Any]) -> list[str]:
    return normalise_layer_specs(
        section.get("probe_layers", ["embedding", 2, 4, 8, 16, "final"])
    )


def preflight(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if str(section.get("dataset", "meld")).lower() != "meld":
        raise ValueError("Stage 7.3 accepts MELD only")
    if section.get("test_locked") is not True:
        raise ValueError("Stage 7.3 requires test_locked: true")
    if STAGE7_2_SOURCE_VERSION != "1.0.1":
        raise ValueError(
            "Stage 7.3 requires the Stage 7.2 padding-safe source v1.0.1; "
            f"found {STAGE7_2_SOURCE_VERSION}"
        )
    layers = _configured_layers(section)
    if "layer_2" not in layers:
        raise ValueError("probe_layers must retain layer 2 as the Stage 7.2 anchor")
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    if seeds != [42, 43, 44] and run_name == "validation":
        raise ValueError("Scientific Stage 7.3 validation requires seeds [42, 43, 44]")
    stage7_2 = _stage7_2_base(root, section)
    decision_path = stage7_2 / "reports" / "frozen_validation_decision.json"
    required = [
        decision_path,
        _source_prepared(root, section) / "semantic_train_targets.jsonl",
        _source_prepared(root, section) / "utility_train_targets.jsonl",
        _source_prepared(root, section) / "threshold_calibration_targets.jsonl",
        _source_prepared(root, section) / "teacher_train_annotations.jsonl",
        _source_prepared(root, section) / "teacher_threshold_annotations.jsonl",
        _resolve(root, section["processed_dir"]) / "validation.jsonl",
    ]
    for seed in seeds:
        source = _source_cache(root, section, seed)
        required.extend(
            [
                source / "cache_complete.json",
                source / "utility_train_metadata.json",
                source / "threshold_calibration_metadata.json",
                source / "validation_metadata.json",
                source / "semantic_train_features.safetensors",
                _stage7_1_adapter(root, section, seed) / "training_complete.json",
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing completed Stage 7.2 dependency: " + ", ".join(missing)
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "stop_and_diagnose_early_utility_gate":
        raise ValueError(
            "Stage 7.3 is preregistered for the frozen Stage 7.2 stop decision; "
            f"found {decision.get('status')!r}"
        )
    if decision.get("test_locked") is not True:
        raise ValueError("The inherited Stage 7.2 decision does not preserve the test lock")
    budgets = [float(value) for value in section.get("read_budgets", [])]
    if not budgets or any(value < 0.0 or value > 1.0 for value in budgets):
        raise ValueError("read_budgets must be a non-empty list inside [0, 1]")
    if budgets != sorted(set(budgets)):
        raise ValueError("read_budgets must be unique and increasing")
    stage7_2_states = {}
    for seed in seeds:
        state = json.loads(
            (_source_cache(root, section, seed) / "cache_complete.json").read_text(
                encoding="utf-8"
            )
        )
        if state.get("test_data_read") is not False:
            raise ValueError(f"Stage 7.2 cache seed={seed} does not prove test exclusion")
        stage7_2_states[str(seed)] = {
            "version": state.get("version"),
            "adapter_tree_sha256": state.get("adapter_tree_sha256"),
            "groups": state.get("groups"),
        }
    result = {
        "status": "ready",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "stage7_2_source_version": STAGE7_2_SOURCE_VERSION,
        "stage7_2_frozen_decision": decision["status"],
        "model_name": section["model_name"],
        "probe_layers": layers,
        "read_budgets": budgets,
        "seeds": seeds,
        "stage7_2_caches": stage7_2_states,
        "emotion_base_frozen": True,
        "reinforcement_learning_used": False,
        "test_locked": True,
    }
    _write_json(paths["preflight"], result)
    manifest = {
        **result,
        "config": str(config_path.expanduser().resolve()),
        "config_sha256": _sha256(config_path.expanduser().resolve()),
        "implementation": str(Path(__file__).resolve()),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["lock_manifest"], manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@dataclass
class LayerwiseFeatureResult:
    features: dict[str, torch.Tensor]
    token_counts: list[int]
    elapsed_seconds: float


class LayerwiseFeatureExtractor:
    """Capture selected current-only hidden states in one frozen full forward."""

    def __init__(self, model: torch.nn.Module, tokenizer: Any, max_input_tokens: int):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = locate_decoder_layers(model)
        self.total_layers = len(self.layers)
        self.max_input_tokens = int(max_input_tokens)

    def _module(self, name: str) -> torch.nn.Module:
        if name == "embedding":
            module = self.model.get_input_embeddings()
            if module is None:
                raise RuntimeError("Qwen model does not expose input embeddings")
            return module
        if name == "final":
            for path in (
                "base_model.model.model.norm",
                "base_model.model.model.model.norm",
                "model.model.norm",
                "model.norm",
            ):
                module: Any = self.model
                for component in path.split("."):
                    if not hasattr(module, component):
                        module = None
                        break
                    module = getattr(module, component)
                if isinstance(module, torch.nn.Module):
                    return module
            raise RuntimeError("Qwen model does not expose its final hidden-state norm")
        number = layer_number(name, self.total_layers)
        return self.layers[number - 1]

    def extract(
        self,
        prompts: Sequence[str],
        layer_specs: Sequence[str],
        *,
        batch_size: int,
    ) -> LayerwiseFeatureResult:
        names = normalise_layer_specs(layer_specs, self.total_layers)
        collected: dict[str, list[torch.Tensor]] = {name: [] for name in names}
        token_counts: list[int] = []
        self.model.eval()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for start in range(0, len(prompts), int(batch_size)):
            encoded = self.tokenizer(
                list(prompts[start : start + int(batch_size)]),
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            token_counts.extend(
                int(value) for value in encoded["attention_mask"].sum(dim=1)
            )
            encoded = {
                key: value.to("cuda", non_blocking=True)
                for key, value in encoded.items()
            }
            last = _last_non_padding_positions(encoded["attention_mask"])
            rows = torch.arange(len(last), device="cuda")
            captured: dict[str, torch.Tensor] = {}
            handles = []

            def make_hook(name: str):
                def hook(
                    _module: torch.nn.Module,
                    _inputs: tuple[Any, ...],
                    output: Any,
                ) -> None:
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    if not isinstance(hidden, torch.Tensor):
                        raise TypeError(f"Layer {name} did not return a tensor")
                    captured[name] = hidden[rows, last].detach().to(
                        device="cpu", dtype=torch.float16
                    )

                return hook

            for name in names:
                handles.append(self._module(name).register_forward_hook(make_hook(name)))
            try:
                with torch.inference_mode():
                    self.model(**encoded, use_cache=False, output_hidden_states=False)
            finally:
                for handle in handles:
                    handle.remove()
            missing = [name for name in names if name not in captured]
            if missing:
                raise RuntimeError(f"Layerwise hooks did not execute: {missing}")
            for name in names:
                collected[name].append(captured[name])
            del encoded, captured
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        hidden_size = int(getattr(self.model.config, "hidden_size", 0))
        result = {
            name: (
                torch.cat(values, dim=0).contiguous()
                if values
                else torch.empty((0, hidden_size), dtype=torch.float16)
            )
            for name, values in collected.items()
        }
        return LayerwiseFeatureResult(result, token_counts, elapsed)


def _feature_compatibility(
    new_layer2: torch.Tensor, legacy_path: Path, maximum_rows: int = 256
) -> dict[str, Any]:
    legacy = load_file(str(legacy_path), device="cpu")["features"]
    if tuple(legacy.shape) != tuple(new_layer2.shape):
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "new_shape": list(new_layer2.shape),
            "legacy_shape": list(legacy.shape),
        }
    count = min(len(legacy), int(maximum_rows))
    left = new_layer2[:count].float()
    right = legacy[:count].float()
    cosine = F.cosine_similarity(left, right, dim=1)
    difference = (left - right).abs()
    result = {
        "passed": bool(
            float(cosine.mean()) >= 0.999
            and float(difference.mean()) <= 0.01
        ),
        "rows_checked": count,
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
        "mean_absolute_difference": float(difference.mean()),
        "maximum_absolute_difference": float(difference.max()),
    }
    del legacy, left, right, cosine, difference
    return result


def cache_seed(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    max_targets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    if not paths["preflight"].exists():
        preflight(config_path, run_name)
    output = _cache_dir(paths, seed)
    done = output / "cache_complete.json"
    if done.exists() and not overwrite:
        result = json.loads(done.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    if output.exists() and overwrite:
        backup = output.with_name(
            output.name + ".backup_" + time.strftime("%Y%m%dT%H%M%S")
        )
        shutil.move(str(output), str(backup))
    prepared = _source_prepared(root, section)
    groups = {
        "semantic_train": _load_targets(prepared / "semantic_train_targets.jsonl"),
        "utility_train": _load_targets(prepared / "utility_train_targets.jsonl"),
        "threshold_calibration": _load_targets(
            prepared / "threshold_calibration_targets.jsonl"
        ),
        "validation": _targets(
            read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl"),
            max_history=3,
        ),
    }
    if max_targets is not None:
        for name in groups:
            groups[name] = groups[name][: int(max_targets)]
    adapter = _stage7_1_adapter(root, section, seed)
    engine = AdaptiveLLMEngine(section, adapter, with_gate=False)
    extractor = LayerwiseFeatureExtractor(
        engine.model,
        engine.tokenizer,
        int(section.get("max_sequence_tokens", 768)),
    )
    layers = normalise_layer_specs(_configured_layers(section), extractor.total_layers)
    output.mkdir(parents=True, exist_ok=True)
    group_state: dict[str, Any] = {}
    compatibility: dict[str, Any] | None = None
    batch_size = int(section.get("feature_batch_size", 4))
    source = _source_cache(root, section, seed)
    for name, targets in groups.items():
        source_metadata = json.loads(
            (source / f"{name}_metadata.json").read_text(encoding="utf-8")
        )
        expected_keys = [target.key for target in targets]
        if source_metadata["keys"][: len(expected_keys)] != expected_keys:
            raise ValueError(f"Stage 7.2 and Stage 7.3 keys differ for {name}")
        print(
            f"[Stage7.3] seed={seed} current features {name}: {len(targets)}",
            flush=True,
        )
        current = extractor.extract(
            engine.prompts(targets, "current"), layers, batch_size=batch_size
        )
        tensors = dict(current.features)
        recent_seconds = 0.0
        if name != "semantic_train":
            print(
                f"[Stage7.3] seed={seed} recent final {name}: {len(targets)}",
                flush=True,
            )
            recent = extractor.extract(
                engine.prompts(targets, "recent3"), ["final"], batch_size=batch_size
            )
            tensors["recent_final"] = recent.features["final"]
            recent_seconds = recent.elapsed_seconds
            del recent
        feature_path = output / f"{name}_features.safetensors"
        save_file(tensors, str(feature_path))
        if name == "semantic_train" and "layer_2" in tensors:
            compatibility = _feature_compatibility(
                tensors["layer_2"], source / "semantic_train_features.safetensors"
            )
            compatibility["tokenizer_padding_side"] = str(
                getattr(engine.tokenizer, "padding_side", "unknown")
            )
            compatibility["stage7_2_cache_version"] = json.loads(
                (source / "cache_complete.json").read_text(encoding="utf-8")
            ).get("version")
        group_state[name] = {
            "targets": len(targets),
            "feature_file": str(feature_path),
            "feature_sha256": _sha256(feature_path),
            "current_feature_seconds": current.elapsed_seconds,
            "recent_final_seconds": recent_seconds,
            "current_token_count": int(sum(current.token_counts)),
            "source_metadata": str(source / f"{name}_metadata.json"),
            "source_metadata_sha256": _sha256(source / f"{name}_metadata.json"),
        }
        del current, tensors
    if compatibility is None:
        raise RuntimeError("Layer-2 compatibility audit was not executed")
    result = {
        "status": "complete" if compatibility["passed"] else "incompatible",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "adapter": str(adapter),
        "adapter_tree_sha256": _tree_sha256(adapter),
        "probe_layers": layers,
        "total_decoder_layers": extractor.total_layers,
        "hidden_size": int(getattr(engine.model.config, "hidden_size", 0)),
        "groups": group_state,
        "stage7_2_layer2_feature_compatibility": compatibility,
        "emotion_base_frozen": True,
        "test_data_read": False,
    }
    state_path = done if compatibility["passed"] else output / "cache_incompatible.json"
    _write_json(state_path, result)
    del engine, extractor
    gc.collect()
    torch.cuda.empty_cache()
    if not compatibility["passed"] and bool(
        section.get("require_stage7_2_feature_compatibility", True)
    ):
        raise RuntimeError(
            "Stage 7.2 layer-2 features do not match the padding-safe extractor. "
            "Keep MELD test locked and regenerate Stage 7.2 caches/results with "
            "llm_early_utility_gate_stage7_2.py v1.0.1 before Stage 7.3. "
            f"Audit: {state_path}"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _utility_row(
    section: dict[str, Any], metadata: dict[str, Any], index: int, labels: Sequence[str]
) -> dict[str, Any]:
    return counterfactual_utility_target(
        metadata["current_scores"][index],
        metadata["recent_scores"][index],
        labels.index(metadata["gold"][index]),
        int(metadata["current_tokens"][index]),
        int(metadata["recent_tokens"][index]),
        bool(metadata["history_available"][index]),
        cost_lambda=float(section.get("utility_cost_lambda", 0.05)),
        read_margin=float(section.get("utility_read_margin", 0.0)),
        weight_scale=float(section.get("utility_weight_scale", 0.20)),
        maximum_weight=float(section.get("utility_maximum_weight", 3.0)),
    )


def _target_details(targets: Sequence[Target]) -> dict[str, dict[str, Any]]:
    result = {}
    for target in targets:
        previous_speaker = target.history[-1].speaker if target.history else None
        result[target.key] = {
            "dialogue_id": target.dialogue_id,
            "turn_index": int(target.turn_index),
            "speaker": target.speaker,
            "history_turns": len(target.history),
            "speaker_switch": bool(
                previous_speaker is not None and previous_speaker != target.speaker
            ),
        }
    return result


def _stratified_utility_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    entropy = np.asarray(
        [float(row["mean_current_entropy"]) for row in rows], dtype=np.float64
    )
    entropy_edges = np.quantile(entropy, [0.25, 0.50, 0.75]) if len(entropy) else []

    def entropy_band(value: float) -> str:
        return "Q" + str(int(np.searchsorted(entropy_edges, value, side="right")) + 1)

    groupings = {
        "emotion": [str(row["gold"]) for row in rows],
        "history_turns": [str(int(row.get("history_turns", 0))) for row in rows],
        "speaker_switch": [str(bool(row.get("speaker_switch", False))) for row in rows],
        "current_entropy_quartile": [
            entropy_band(float(row["mean_current_entropy"])) for row in rows
        ],
    }
    result: dict[str, Any] = {}
    for name, categories in groupings.items():
        values: dict[str, Any] = {}
        for category in sorted(set(categories)):
            selected = np.asarray(
                [
                    float(row["consensus_nll_gain"])
                    for row, value in zip(rows, categories)
                    if value == category
                ],
                dtype=np.float64,
            )
            values[category] = {
                "examples": int(len(selected)),
                "mean_nll_gain": float(selected.mean()),
                "median_nll_gain": float(np.median(selected)),
                "positive_rate": float(np.mean(selected > 0.0)),
                "near_zero_0.05_rate": float(np.mean(np.abs(selected) < 0.05)),
            }
        result[name] = values
    result["current_entropy_quartile_edges"] = [
        float(value) for value in entropy_edges
    ]
    return result


def audit_utility_targets(
    config_path: Path,
    run_name: str = "validation",
    *,
    seeds_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = (
        [int(value) for value in seeds_override]
        if seeds_override is not None
        else [int(value) for value in section.get("seeds", [42, 43, 44])]
    )
    for seed in seeds:
        if not (_cache_dir(paths, seed) / "cache_complete.json").exists():
            raise FileNotFoundError(f"Run Stage 7.3 cache-seed first for seed={seed}")
    labels = list(section.get("labels", LABELS))
    groups = ("utility_train", "threshold_calibration", "validation")
    prepared = _source_prepared(root, section)
    target_sets = {
        "utility_train": _load_targets(prepared / "utility_train_targets.jsonl"),
        "threshold_calibration": _load_targets(
            prepared / "threshold_calibration_targets.jsonl"
        ),
        "validation": _targets(
            read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl"),
            max_history=3,
        ),
    }
    all_rows: list[dict[str, Any]] = []
    group_reports: dict[str, Any] = {}
    for group in groups:
        metadata_by_seed = {
            seed: json.loads(
                (
                    _source_cache(root, section, seed) / f"{group}_metadata.json"
                ).read_text(encoding="utf-8")
            )
            for seed in seeds
        }
        reference = metadata_by_seed[seeds[0]]
        limit = len(target_sets[group])
        if run_name != "validation":
            limit = int(
                json.loads(
                    (_cache_dir(paths, seeds[0]) / "cache_complete.json").read_text(
                        encoding="utf-8"
                    )
                )["groups"][group]["targets"]
            )
        keys = reference["keys"][:limit]
        for seed, metadata in metadata_by_seed.items():
            if metadata["keys"][:limit] != keys:
                raise ValueError(f"Utility metadata keys differ for seed={seed}, {group}")
            if metadata["gold"][:limit] != reference["gold"][:limit]:
                raise ValueError(f"Gold labels differ for seed={seed}, {group}")
        details = _target_details(target_sets[group][:limit])
        delta_matrix: list[list[float]] = [[] for _ in seeds]
        label_matrix: list[list[int]] = [[] for _ in seeds]
        group_rows: list[dict[str, Any]] = []
        for index, key in enumerate(keys):
            seed_rows = {
                seed: _utility_row(section, metadata_by_seed[seed], index, labels)
                for seed in seeds
            }
            deltas = [float(seed_rows[seed]["nll_gain"]) for seed in seeds]
            utility_v1 = [int(seed_rows[seed]["label"]) for seed in seeds]
            for seed_index, (delta, label) in enumerate(zip(deltas, utility_v1)):
                delta_matrix[seed_index].append(delta)
                label_matrix[seed_index].append(label)
            correctness_gain = [
                int(seed_rows[seed]["recent_correct"])
                - int(seed_rows[seed]["current_correct"])
                for seed in seeds
            ]
            current_entropy = [
                probability_entropy(metadata_by_seed[seed]["current_scores"][index])
                for seed in seeds
            ]
            row = {
                "group": group,
                "key": key,
                **details.get(key, {}),
                "gold": reference["gold"][index],
                "history_available": bool(reference["history_available"][index]),
                "nll_gain_by_seed": {
                    str(seed): value for seed, value in zip(seeds, deltas)
                },
                "utility_v1_by_seed": {
                    str(seed): value for seed, value in zip(seeds, utility_v1)
                },
                "sample_weight_by_seed": {
                    str(seed): float(seed_rows[seed]["sample_weight"])
                    for seed in seeds
                },
                "rule_by_seed": {
                    str(seed): str(seed_rows[seed]["rule"]) for seed in seeds
                },
                "correctness_gain_by_seed": {
                    str(seed): value for seed, value in zip(seeds, correctness_gain)
                },
                "current_entropy_by_seed": {
                    str(seed): value for seed, value in zip(seeds, current_entropy)
                },
                "mean_current_entropy": float(np.mean(current_entropy)),
                "consensus_nll_gain": float(np.median(deltas)),
                "consensus_positive": int(float(np.median(deltas)) > 0.0),
                "consensus_correctness_gain": float(np.mean(correctness_gain)),
            }
            group_rows.append(row)
            all_rows.append(row)
        report = utility_stability_report(
            delta_matrix,
            label_matrix,
            seeds,
            near_zero_bands=section.get("near_zero_bands", [0.02, 0.05, 0.10]),
            minimum_spearman=float(section.get("minimum_seed_spearman", 0.40)),
            minimum_sign_agreement=float(
                section.get("minimum_seed_sign_agreement", 0.70)
            ),
        )
        report["gold_distribution"] = dict(Counter(row["gold"] for row in group_rows))
        report["current_wrong_recent_correct_rate"] = float(
            np.mean(
                [
                    value > 0
                    for row in group_rows
                    for value in row["correctness_gain_by_seed"].values()
                ]
            )
        )
        report["current_correct_recent_wrong_rate"] = float(
            np.mean(
                [
                    value < 0
                    for row in group_rows
                    for value in row["correctness_gain_by_seed"].values()
                ]
            )
        )
        report["stratified_utility"] = _stratified_utility_summary(group_rows)
        group_reports[group] = report
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seeds": seeds,
        "groups": group_reports,
        "target_reliability_pass": group_reports["utility_train"][
            "target_reliability_pass"
        ],
        "target_definition": (
            "continuous consensus is median seed [NLL(current)-NLL(recent3)]; "
            "Stage 7.2 utility_v1 is retained separately"
        ),
        "emotion_base_frozen": True,
        "validation_used_for_target_training": False,
        "test_data_read": False,
        "test_locked": True,
    }
    _write_jsonl(paths["utility_examples"], all_rows)
    _write_json(paths["utility_audit"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@dataclass
class ProbeRows:
    features: torch.Tensor
    targets: torch.Tensor
    weights: torch.Tensor


def _robust_rank_transform(values: Sequence[float]) -> tuple[np.ndarray, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    center = float(np.median(array))
    mad = float(np.median(np.abs(array - center)))
    scale = max(1.4826 * mad, float(array.std()), 1e-6)
    transformed = np.clip((array - center) / scale, -5.0, 5.0)
    return transformed.astype(np.float32), {"center": center, "scale": scale, "clip": 5.0}


def _shuffle_rows(
    targets: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(len(targets))
    changed = float(np.mean(permutation != np.arange(len(targets))))
    return targets[permutation], weights[permutation], changed


def _train_probe(
    section: dict[str, Any],
    rows: ProbeRows,
    output_dir: Path,
    *,
    objective: str,
    initialization_seed: int,
    bottleneck: int,
    target_transform: dict[str, float] | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    random.seed(initialization_seed)
    np.random.seed(initialization_seed)
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = EarlyBinaryGateHead(
        int(rows.features.shape[1]),
        int(bottleneck),
        float(section.get("probe_dropout", 0.10)),
    ).to(device)
    steps = int(max_steps_override or section.get("probe_max_steps", 500))
    batch_size = min(int(section.get("probe_batch_size", 64)), len(rows.features))
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(section.get("probe_learning_rate", 3e-4)),
        weight_decay=float(section.get("probe_weight_decay", 0.01)),
    )
    from transformers import get_cosine_schedule_with_warmup

    warmup = min(int(section.get("probe_warmup_steps", 25)), max(steps - 1, 0))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, steps)
    generator = torch.Generator().manual_seed(int(initialization_seed))
    losses: list[float] = []
    head.train()
    started = time.time()
    for step in range(steps):
        indexes = torch.randint(
            0, len(rows.features), (batch_size,), generator=generator
        )
        x = rows.features.index_select(0, indexes).to(device, non_blocking=True)
        y = rows.targets.index_select(0, indexes).to(device, non_blocking=True)
        weight = rows.weights.index_select(0, indexes).to(device, non_blocking=True)
        prediction = head(x)
        if objective.startswith("utility_rank"):
            raw = F.smooth_l1_loss(prediction, y, reduction="none", beta=1.0)
        else:
            raw = F.binary_cross_entropy_with_logits(prediction, y, reduction="none")
        loss = (raw * weight).sum() / weight.sum().clamp_min(1e-8)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite Stage 7.3 loss: {objective}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            head.parameters(), float(section.get("probe_max_grad_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach().cpu()))
        interval = int(section.get("log_every_steps", 50))
        if step == 0 or (step + 1) % interval == 0:
            print(
                f"[Stage7.3] {output_dir.parent.name}/{objective} "
                f"step={step + 1}/{steps} loss={np.mean(losses[-interval:]):.6f}",
                flush=True,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        {key: value.detach().cpu() for key, value in head.state_dict().items()},
        str(output_dir / "probe.safetensors"),
    )
    config = {
        "hidden_size": int(rows.features.shape[1]),
        "bottleneck": int(bottleneck),
        "dropout": float(section.get("probe_dropout", 0.10)),
        "objective": objective,
        "target_transform": target_transform,
        "classes": ["SKIP", "READ"] if not objective.startswith("utility_rank") else None,
    }
    _write_json(output_dir / "probe_config.json", config)
    return {
        "status": "complete",
        "objective": objective,
        "examples": len(rows.features),
        "optimizer_steps": steps,
        "initialization_seed": int(initialization_seed),
        "trainable_parameters": int(sum(value.numel() for value in head.parameters())),
        "mean_loss": float(np.mean(losses)),
        "finite": bool(np.isfinite(losses).all()),
        "elapsed_seconds": time.time() - started,
        "emotion_base_frozen": True,
        "test_data_read": False,
    }


def _audit_rows_by_group(paths: dict[str, Path]) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _read_jsonl(paths["utility_examples"]):
        result.setdefault(str(row["group"]), {})[str(row["key"])] = row
    return result


def train_probes(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    overwrite: bool = False,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    cache = _cache_dir(paths, seed)
    if not (cache / "cache_complete.json").exists():
        raise FileNotFoundError(f"Run cache-seed first for seed={seed}")
    if not paths["utility_audit"].exists():
        raise FileNotFoundError("Run audit-targets before training probes")
    state = json.loads((cache / "cache_complete.json").read_text(encoding="utf-8"))
    layers = list(state["probe_layers"])
    hidden_size = int(state["hidden_size"])
    base_bottleneck = int(section.get("probe_bottleneck", 256))
    semantic_store = load_file(
        str(cache / "semantic_train_features.safetensors"), device="cpu"
    )
    utility_store = load_file(
        str(cache / "utility_train_features.safetensors"), device="cpu"
    )
    source = _source_cache(root, section, seed)
    semantic_metadata = json.loads(
        (source / "semantic_train_metadata.json").read_text(encoding="utf-8")
    )
    utility_metadata = json.loads(
        (source / "utility_train_metadata.json").read_text(encoding="utf-8")
    )
    semantic_count = int(state["groups"]["semantic_train"]["targets"])
    utility_count = int(state["groups"]["utility_train"]["targets"])
    semantic_keys = semantic_metadata["keys"][:semantic_count]
    utility_keys = utility_metadata["keys"][:utility_count]
    teacher = {
        f"{row['dialogue_id']}::{int(row['target_turn_id'])}": row
        for row in _read_jsonl(
            _source_prepared(root, section) / "teacher_train_annotations.jsonl"
        )
    }
    semantic_targets = np.asarray(
        [teacher_read_label(str(teacher[key]["context_need"])) for key in semantic_keys],
        dtype=np.float32,
    )
    semantic_weights = np.asarray(
        [
            teacher_sample_weight(str(teacher[key]["context_need"]))
            for key in semantic_keys
        ],
        dtype=np.float32,
    )
    audit_rows = _audit_rows_by_group(paths)
    utility_rows = [audit_rows["utility_train"][key] for key in utility_keys]
    utility_targets = np.asarray(
        [row["utility_v1_by_seed"][str(seed)] for row in utility_rows],
        dtype=np.float32,
    )
    utility_weights = np.asarray(
        [row["sample_weight_by_seed"][str(seed)] for row in utility_rows],
        dtype=np.float32,
    )
    rank_raw = np.asarray(
        [row["consensus_nll_gain"] for row in utility_rows], dtype=np.float64
    )
    rank_targets, rank_transform = _robust_rank_transform(rank_raw)
    rank_weights = np.ones(len(rank_targets), dtype=np.float32)
    target_sets = {
        "semantic": (semantic_targets, semantic_weights),
        "utility_v1": (utility_targets, utility_weights),
        "utility_rank": (rank_targets, rank_weights),
    }
    training: dict[str, Any] = {}

    def train_layer_objective(
        layer: str,
        objective: str,
        features: torch.Tensor,
        targets: np.ndarray,
        weights: np.ndarray,
        bottleneck: int,
    ) -> None:
        for shuffled in (False, True):
            name = objective + (CONTROL_SUFFIX if shuffled else "")
            output = _probe_dir(paths, seed, layer, name)
            complete = output / "training_complete.json"
            if complete.exists() and not overwrite:
                training[f"{layer}/{name}"] = json.loads(
                    complete.read_text(encoding="utf-8")
                )
                continue
            if output.exists() and overwrite:
                backup = output.with_name(
                    output.name + ".backup_" + time.strftime("%Y%m%dT%H%M%S")
                )
                shutil.move(str(output), str(backup))
            current_targets = targets
            current_weights = weights
            changed = 0.0
            if shuffled:
                current_targets, current_weights, changed = _shuffle_rows(
                    targets,
                    weights,
                    int(section.get("shuffle_control_seed", 7373))
                    + seed
                    + sum(ord(value) for value in f"{layer}/{objective}"),
                )
            layer_seed_offset = (
                int(state["total_decoder_layers"]) + 1
                if layer == PAIR_LAYER
                else layer_number(layer, int(state["total_decoder_layers"]))
            )
            initialization_seed = (
                seed
                + {"semantic": 101, "utility_v1": 202, "utility_rank": 303}[
                    objective
                ]
                + 1009 * layer_seed_offset
            )
            probe_rows = ProbeRows(
                features=features.float(),
                targets=torch.tensor(current_targets, dtype=torch.float32),
                weights=torch.tensor(current_weights, dtype=torch.float32),
            )
            trained = _train_probe(
                section,
                probe_rows,
                output,
                objective=name,
                initialization_seed=initialization_seed,
                bottleneck=bottleneck,
                target_transform=rank_transform if objective == "utility_rank" else None,
                max_steps_override=max_steps_override,
            )
            trained["shuffled_target"] = shuffled
            trained["shuffled_changed_rate"] = changed
            trained["layer"] = layer
            _write_json(complete, trained)
            training[f"{layer}/{name}"] = trained
            del probe_rows

    for layer in layers:
        for objective in BASE_OBJECTIVES:
            if objective == "semantic":
                features = semantic_store[layer][:semantic_count]
            else:
                features = utility_store[layer][:utility_count]
            targets, weights = target_sets[objective]
            train_layer_objective(
                layer,
                objective,
                features,
                targets,
                weights,
                base_bottleneck,
            )
    pair_features = torch.cat(
        [
            utility_store["final"][:utility_count],
            utility_store["recent_final"][:utility_count],
        ],
        dim=1,
    )
    pair_bottleneck = capacity_matched_pair_bottleneck(
        hidden_size, base_bottleneck
    )
    train_layer_objective(
        PAIR_LAYER,
        "utility_rank",
        pair_features,
        rank_targets,
        rank_weights,
        pair_bottleneck,
    )
    base_parameters = training[f"final/utility_rank"]["trainable_parameters"]
    pair_parameters = training[f"{PAIR_LAYER}/utility_rank"]["trainable_parameters"]
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "probe_layers": layers,
        "training": training,
        "rank_target_transform": rank_transform,
        "pair_bottleneck": pair_bottleneck,
        "pair_parameter_count_no_greater_than_single": pair_parameters
        <= base_parameters,
        "all_training_finite": all(value["finite"] for value in training.values()),
        "all_emotion_bases_frozen": all(
            value["emotion_base_frozen"] for value in training.values()
        ),
        "validation_features_loaded": False,
        "test_data_read": False,
    }
    _write_json(paths["reports"] / f"seed{seed}_probe_training.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _load_probe(path: Path, device: torch.device) -> tuple[EarlyBinaryGateHead, dict[str, Any]]:
    config = json.loads((path / "probe_config.json").read_text(encoding="utf-8"))
    head = EarlyBinaryGateHead(
        int(config["hidden_size"]),
        int(config["bottleneck"]),
        float(config.get("dropout", 0.10)),
    ).to(device)
    head.load_state_dict(load_file(str(path / "probe.safetensors"), device=str(device)))
    head.eval()
    return head, config


def _probe_scores(
    head: EarlyBinaryGateHead,
    features: torch.Tensor,
    objective: str,
    *,
    batch_size: int = 512,
) -> np.ndarray:
    device = next(head.parameters()).device
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), int(batch_size)):
            values = features[start : start + int(batch_size)].to(
                device, dtype=torch.float32, non_blocking=True
            )
            score = head(values)
            if not objective.startswith("utility_rank"):
                score = torch.sigmoid(score)
            output.append(score.float().cpu().numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=np.float64)


def _candidate_specs(layers: Sequence[str]) -> list[tuple[str, str]]:
    return [(layer, objective) for layer in layers for objective in BASE_OBJECTIVES] + [
        (PAIR_LAYER, "utility_rank")
    ]


def _is_deployable_probe_layer(layer: str) -> bool:
    """Only pre-final current-only states can support compute-saving routing."""
    return layer not in {"final", PAIR_LAYER}


def _theoretical_cost_ratio(
    metadata: dict[str, Any], read_mask: np.ndarray, prefix_layers: int, total_layers: int
) -> float | None:
    if total_layers <= 0:
        raise ValueError("total_layers must be positive")
    if prefix_layers < 0 or prefix_layers >= total_layers:
        return None
    if prefix_layers == 0:
        baseline = float(sum(metadata["recent_tokens"]))
        adaptive = float(
            sum(
                recent if read else current
                for current, recent, read in zip(
                    metadata["current_tokens"],
                    metadata["recent_tokens"],
                    read_mask.tolist(),
                )
            )
        )
        return adaptive / baseline if baseline else 0.0
    return layer_equivalent_cost_ratio(
        metadata["current_tokens"],
        metadata["recent_tokens"],
        read_mask.tolist(),
        prefix_layers,
        total_layers,
    )


def _adaptive_metric_row(
    calibration: dict[str, Any],
    validation: dict[str, Any],
    calibration_mask: np.ndarray,
    validation_mask: np.ndarray,
    labels: Sequence[str],
) -> dict[str, Any]:
    calibration_scores = _mix_scores(
        calibration["current_scores"],
        calibration["recent_scores"],
        calibration_mask.tolist(),
    )
    validation_scores = _mix_scores(
        validation["current_scores"],
        validation["recent_scores"],
        validation_mask.tolist(),
    )
    metric = _metrics_with_probabilities(
        calibration_scores,
        validation_scores,
        calibration["gold"],
        validation["gold"],
        labels,
    )
    metric["context_use_rate"] = float(validation_mask.mean())
    history = np.asarray(validation["history_available"], dtype=bool)
    metric["history_eligible_read_rate"] = (
        float(validation_mask[history].mean()) if history.any() else 0.0
    )
    return metric


def _metrics_with_probabilities(
    calibration_scores: Sequence[Sequence[float]],
    validation_scores: Sequence[Sequence[float]],
    calibration_gold: Sequence[str],
    validation_gold: Sequence[str],
    labels: Sequence[str],
) -> dict[str, Any]:
    """Return Stage 7 metrics plus the calibrated probabilities they summarize."""
    temperature = _fit_temperature(
        calibration_scores, [labels.index(value) for value in calibration_gold]
    )
    metric = _metrics(validation_scores, validation_gold, labels, temperature)
    metric["temperature"] = float(temperature)
    metric["probabilities"] = [
        np.asarray(_softmax(row, temperature), dtype=np.float64).tolist()
        for row in validation_scores
    ]
    metric["predictions"] = [
        int(np.argmax(probability)) for probability in metric["probabilities"]
    ]
    return metric


def _prediction_record(
    prediction_id: str,
    kind: str,
    metadata: dict[str, Any],
    metric: dict[str, Any],
    read_mask: Sequence[bool],
    route_scores: Sequence[float] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    probabilities = np.asarray(metric["probabilities"], dtype=np.float64)
    predictions = metric.get("predictions")
    if predictions is None:
        predictions = probabilities.argmax(axis=1).astype(int).tolist()
    record = {
        "prediction_id": prediction_id,
        "kind": kind,
        **extra,
        "keys": list(metadata["keys"]),
        "gold": list(metadata["gold"]),
        "dialogue_ids": [key.rsplit("::", 1)[0] for key in metadata["keys"]],
        "read_mask": [bool(value) for value in read_mask],
        "route_scores": (
            [float(value) for value in route_scores]
            if route_scores is not None
            else None
        ),
        "probabilities": probabilities.tolist(),
        "predictions": list(predictions),
    }
    if prediction_id == "base_current":
        record["raw_scores"] = metadata["current_scores"]
        record["input_tokens"] = metadata["current_tokens"]
    elif prediction_id == "base_recent3":
        record["raw_scores"] = metadata["recent_scores"]
        record["input_tokens"] = metadata["recent_tokens"]
    return record


def evaluate_seed(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    output_path = paths["scores"] / f"seed{seed}_validation.json"
    prediction_path = paths["scores"] / f"seed{seed}_validation_predictions.jsonl"
    if output_path.exists() and prediction_path.exists() and not overwrite:
        result = json.loads(output_path.read_text(encoding="utf-8"))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    cache = _cache_dir(paths, seed)
    cache_state_path = cache / "cache_complete.json"
    training_path = paths["reports"] / f"seed{seed}_probe_training.json"
    if not cache_state_path.exists() or not training_path.exists():
        raise FileNotFoundError(f"Missing Stage 7.3 cache/probes for seed={seed}")
    state = json.loads(cache_state_path.read_text(encoding="utf-8"))
    layers = list(state["probe_layers"])
    total_layers = int(state["total_decoder_layers"])
    calibration_store = load_file(
        str(cache / "threshold_calibration_features.safetensors"), device="cpu"
    )
    validation_store = load_file(
        str(cache / "validation_features.safetensors"), device="cpu"
    )
    source = _source_cache(root, section, seed)
    calibration = json.loads(
        (source / "threshold_calibration_metadata.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (source / "validation_metadata.json").read_text(encoding="utf-8")
    )
    calibration_count = int(state["groups"]["threshold_calibration"]["targets"])
    validation_count = int(state["groups"]["validation"]["targets"])
    calibration = {
        key: (value[:calibration_count] if isinstance(value, list) else value)
        for key, value in calibration.items()
    }
    validation = {
        key: (value[:validation_count] if isinstance(value, list) else value)
        for key, value in validation.items()
    }
    labels = list(section.get("labels", LABELS))
    audit = _audit_rows_by_group(paths)
    calibration_targets = [
        audit["threshold_calibration"][key] for key in calibration["keys"]
    ]
    validation_targets = [audit["validation"][key] for key in validation["keys"]]
    consensus_calibration = np.asarray(
        [row["consensus_nll_gain"] for row in calibration_targets], dtype=np.float64
    )
    consensus_validation = np.asarray(
        [row["consensus_nll_gain"] for row in validation_targets], dtype=np.float64
    )
    correctness_calibration = np.asarray(
        [row["consensus_correctness_gain"] for row in calibration_targets],
        dtype=np.float64,
    )
    correctness_validation = np.asarray(
        [row["consensus_correctness_gain"] for row in validation_targets],
        dtype=np.float64,
    )
    consensus_positive_calibration = (consensus_calibration > 0.0).astype(np.int64)
    consensus_positive_validation = (consensus_validation > 0.0).astype(np.int64)
    utility_v1_calibration = np.asarray(
        [row["utility_v1_by_seed"][str(seed)] for row in calibration_targets],
        dtype=np.int64,
    )
    utility_v1_validation = np.asarray(
        [row["utility_v1_by_seed"][str(seed)] for row in validation_targets],
        dtype=np.int64,
    )
    history_calibration = np.asarray(calibration["history_available"], dtype=bool)
    history_validation = np.asarray(validation["history_available"], dtype=bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for layer, objective in _candidate_specs(layers):
        for control in (False, True):
            name = objective + (CONTROL_SUFFIX if control else "")
            head, _ = _load_probe(_probe_dir(paths, seed, layer, name), device)
            if layer == PAIR_LAYER:
                calibration_features = torch.cat(
                    [
                        calibration_store["final"][:calibration_count],
                        calibration_store["recent_final"][:calibration_count],
                    ],
                    dim=1,
                )
                validation_features = torch.cat(
                    [
                        validation_store["final"][:validation_count],
                        validation_store["recent_final"][:validation_count],
                    ],
                    dim=1,
                )
            else:
                calibration_features = calibration_store[layer][:calibration_count]
                validation_features = validation_store[layer][:validation_count]
            scores[(layer, name)] = (
                _probe_scores(head, calibration_features, name),
                _probe_scores(head, validation_features, name),
            )
            del head, calibration_features, validation_features

    teacher_calibration = _read_jsonl(
        _source_prepared(root, section) / "teacher_threshold_annotations.jsonl"
    )
    current_temperature = _fit_temperature(
        calibration["current_scores"],
        [labels.index(value) for value in calibration["gold"]],
    )
    entropy_calibration = np.asarray(
        [
            probability_entropy(row, current_temperature)
            for row in calibration["current_scores"]
        ],
        dtype=np.float64,
    )
    entropy_validation = np.asarray(
        [
            probability_entropy(row, current_temperature)
            for row in validation["current_scores"]
        ],
        dtype=np.float64,
    )
    correctness_score_calibration = 2.0 * correctness_calibration + 0.01 * np.tanh(
        consensus_calibration
    )
    correctness_score_validation = 2.0 * correctness_validation + 0.01 * np.tanh(
        consensus_validation
    )
    budgets = [float(value) for value in section["read_budgets"]]
    predictions: list[dict[str, Any]] = []
    frontier: dict[str, Any] = {}

    current_metric = _metrics_with_probabilities(
        calibration["current_scores"],
        validation["current_scores"],
        calibration["gold"],
        validation["gold"],
        labels,
    )
    recent_metric = _metrics_with_probabilities(
        calibration["recent_scores"],
        validation["recent_scores"],
        calibration["gold"],
        validation["gold"],
        labels,
    )
    no_read = np.zeros(validation_count, dtype=bool)
    all_history = history_validation.copy()
    predictions.append(
        _prediction_record(
            "base_current",
            "baseline",
            validation,
            current_metric,
            no_read,
            None,
            {"seed": int(seed)},
        )
    )
    predictions.append(
        _prediction_record(
            "base_recent3",
            "baseline",
            validation,
            recent_metric,
            all_history,
            None,
            {"seed": int(seed)},
        )
    )

    def add_frontier(
        *,
        layer: str,
        objective: str,
        budget: float,
        cal_score: np.ndarray,
        val_score: np.ndarray,
        deployable: bool,
        oracle: bool = False,
        shuffled_score: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        if oracle:
            cal_mask = exact_top_k_mask(
                cal_score,
                history_calibration,
                int(np.rint(budget * len(cal_score))),
                calibration["keys"],
            )
            val_mask = exact_top_k_mask(
                val_score,
                history_validation,
                int(np.rint(budget * len(val_score))),
                validation["keys"],
            )
            rule = None
        else:
            rule = fit_budget_threshold(cal_score, history_calibration, budget)
            salt = int(seed) + int(round(budget * 10000)) + sum(
                ord(value) for value in f"{layer}/{objective}"
            )
            cal_mask = apply_budget_threshold(
                cal_score,
                history_calibration,
                calibration["keys"],
                rule,
                salt=salt,
            )
            val_mask = apply_budget_threshold(
                val_score,
                history_validation,
                validation["keys"],
                rule,
                salt=salt,
            )
        candidate_id = f"{layer}__{objective}__b{budget:.4f}"
        metric = _adaptive_metric_row(
            calibration, validation, cal_mask, val_mask, labels
        )
        prefix = (
            layer_number(layer, total_layers)
            if deployable and layer != PAIR_LAYER
            else -1
        )
        cost_ratio = _theoretical_cost_ratio(
            validation, val_mask, prefix, total_layers
        )
        utility_ap = (
            float(average_precision_score(consensus_positive_validation, val_score))
            if len(np.unique(consensus_positive_validation)) > 1
            else None
        )
        diagnostics = _threshold_diagnostics(
            cal_score,
            cal_mask,
            calibration,
            teacher_calibration,
            utility_v1_calibration,
        )
        common = {
            "seed": int(seed),
            "layer": layer,
            "objective": objective,
            "requested_read_budget": budget,
            "threshold_rule": rule.to_dict() if rule is not None else None,
            "deployable": deployable,
            "oracle": oracle,
            "utility_consensus_ap": utility_ap,
            "utility_consensus_prevalence": float(
                consensus_positive_validation.mean()
            ),
            "layer_equivalent_cost_ratio": cost_ratio,
            "layer_equivalent_token_reduction": (
                1.0 - cost_ratio if cost_ratio is not None else None
            ),
            "calibration_diagnostics": diagnostics,
        }
        frontier[candidate_id] = {
            "candidate": {
                **{
                    key: value
                    for key, value in metric.items()
                    if key not in {"probabilities", "predictions"}
                },
                **common,
            }
        }
        predictions.append(
            _prediction_record(
                candidate_id,
                "candidate",
                validation,
                metric,
                val_mask,
                val_score,
                common,
            )
        )
        if shuffled_score is None:
            return
        shuffled_cal, shuffled_val = shuffled_score
        shuffled_cal_mask = exact_top_k_mask(
            shuffled_cal,
            history_calibration,
            int(cal_mask.sum()),
            calibration["keys"],
        )
        shuffled_val_mask = exact_top_k_mask(
            shuffled_val,
            history_validation,
            int(val_mask.sum()),
            validation["keys"],
        )
        random_cal_mask = exact_random_mask(
            history_calibration,
            int(cal_mask.sum()),
            calibration["keys"],
            salt=int(section.get("matched_random_seed", 7381)) + seed,
        )
        random_val_mask = exact_random_mask(
            history_validation,
            int(val_mask.sum()),
            validation["keys"],
            salt=int(section.get("matched_random_seed", 7381)) + seed + 1,
        )
        for control_name, control_cal, control_val, control_score in (
            (
                "matched_shuffled",
                shuffled_cal_mask,
                shuffled_val_mask,
                shuffled_val,
            ),
            ("matched_random", random_cal_mask, random_val_mask, None),
        ):
            control_id = f"{candidate_id}__{control_name}"
            control_metric = _adaptive_metric_row(
                calibration, validation, control_cal, control_val, labels
            )
            frontier[candidate_id][control_name] = {
                key: value
                for key, value in control_metric.items()
                if key not in {"probabilities", "predictions"}
            }
            predictions.append(
                _prediction_record(
                    control_id,
                    control_name,
                    validation,
                    control_metric,
                    control_val,
                    control_score,
                    {
                        "seed": int(seed),
                        "parent_candidate_id": candidate_id,
                        "layer": layer,
                        "objective": objective,
                        "requested_read_budget": budget,
                    },
                )
            )
        shuffled_ap = (
            float(
                average_precision_score(
                    consensus_positive_validation, shuffled_val
                )
            )
            if len(np.unique(consensus_positive_validation)) > 1
            else None
        )
        frontier[candidate_id]["candidate"]["matched_shuffled_utility_ap"] = shuffled_ap
        frontier[candidate_id]["candidate"]["utility_ap_advantage_over_shuffled"] = (
            utility_ap - shuffled_ap
            if utility_ap is not None and shuffled_ap is not None
            else None
        )

    for layer, objective in _candidate_specs(layers):
        cal_score, val_score = scores[(layer, objective)]
        shuffled = scores[(layer, objective + CONTROL_SUFFIX)]
        for budget in budgets:
            add_frontier(
                layer=layer,
                objective=objective,
                budget=budget,
                cal_score=cal_score,
                val_score=val_score,
                deployable=_is_deployable_probe_layer(layer),
                shuffled_score=shuffled,
            )
    for objective, cal_score, val_score in (
        ("full_pass_entropy", entropy_calibration, entropy_validation),
        ("oracle_nll", consensus_calibration, consensus_validation),
        (
            "oracle_correctness",
            correctness_score_calibration,
            correctness_score_validation,
        ),
    ):
        for budget in budgets:
            add_frontier(
                layer="final" if objective == "full_pass_entropy" else "oracle",
                objective=objective,
                budget=budget,
                cal_score=cal_score,
                val_score=val_score,
                deployable=False,
                oracle=objective.startswith("oracle_"),
            )
    _write_jsonl(prediction_path, predictions)
    flat_rows: list[dict[str, Any]] = []
    for candidate_id, row in frontier.items():
        candidate = row["candidate"]
        flat_rows.append(
            {
                "seed": seed,
                "candidate_id": candidate_id,
                "layer": candidate["layer"],
                "objective": candidate["objective"],
                "budget": candidate["requested_read_budget"],
                "context_use_rate": candidate["context_use_rate"],
                "weighted_f1": candidate["weighted_f1"],
                "macro_f1": candidate["macro_f1"],
                "nll": candidate["nll"],
                "brier": candidate["brier"],
                "ece": candidate["ece"],
                "utility_ap": candidate["utility_consensus_ap"],
                "utility_ap_advantage_over_shuffled": candidate.get(
                    "utility_ap_advantage_over_shuffled"
                ),
                "layer_equivalent_token_reduction": candidate.get(
                    "layer_equivalent_token_reduction"
                ),
            }
        )
    result = {
        "status": "evaluated",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "split": "validation",
        "targets": validation_count,
        "dialogues": len({key.rsplit("::", 1)[0] for key in validation["keys"]}),
        "labels": labels,
        "baselines": {
            "base_current": {
                key: value
                for key, value in current_metric.items()
                if key not in {"probabilities", "predictions"}
            },
            "base_recent3": {
                key: value
                for key, value in recent_metric.items()
                if key not in {"probabilities", "predictions"}
            },
        },
        "frontier": frontier,
        "prediction_file": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "architecture_audit": {
            "emotion_base_frozen": True,
            "layerwise_features_from_current_only": True,
            "pair_aware_marked_non_deployable": True,
            "final_current_only_marked_non_deployable": True,
            "non_deployable_layers_excluded_from_cost_and_policy_gates": True,
            "oracle_uses_gold_for_diagnostics_only": True,
            "thresholds_fit_on_train_calibration_only": True,
        },
        "validation_outcomes_used_for_threshold_selection": False,
        "test_data_read": False,
        "test_locked": True,
    }
    _write_json(output_path, result)
    _write_csv(paths["scores"] / f"seed{seed}_budget_frontier.csv", flat_rows)
    print(f"Stage 7.3 seed {seed} validation written to {output_path}")
    del calibration_store, validation_store
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


@dataclass
class RuntimeProductionResult:
    elapsed_seconds: float
    prefix_seconds: float
    head_seconds: float
    packing_seconds: float
    current_seconds: float
    recent_seconds: float
    read_count: int
    target_count: int
    raw_input_tokens: int
    layer_equivalent_tokens: float


class ProfiledLayerRouter:
    """Real two-stage execution used only for the systems feasibility audit."""

    def __init__(
        self,
        engine: AdaptiveLLMEngine,
        section: dict[str, Any],
        head_path: Path,
        layer: int,
    ):
        self.engine = engine
        self.extractor = EarlyFeatureExtractor(
            engine.model,
            engine.tokenizer,
            int(layer),
            int(section.get("max_sequence_tokens", 768)),
        )
        self.head, self.head_config = _load_probe(head_path, torch.device("cuda"))

    def route_and_score(
        self,
        targets: Sequence[Target],
        rule: ThresholdRule,
        batch_size: int,
        *,
        salt: int,
    ) -> RuntimeProductionResult:
        feature = self.extractor.extract(
            self.engine.prompts(targets, "current"),
            batch_size=int(batch_size),
            keep_on_device=True,
        )
        torch.cuda.synchronize()
        head_started = time.perf_counter()
        with torch.inference_mode():
            score = self.head(feature.features.float())
            if not str(self.head_config["objective"]).startswith("utility_rank"):
                score = torch.sigmoid(score)
        torch.cuda.synchronize()
        route_score = score.float().cpu().numpy()
        head_seconds = time.perf_counter() - head_started
        packing_started = time.perf_counter()
        read = apply_budget_threshold(
            route_score,
            [bool(target.history) for target in targets],
            [target.key for target in targets],
            rule,
            salt=int(salt),
        )
        current_indexes = np.flatnonzero(~read).tolist()
        recent_indexes = np.flatnonzero(read).tolist()
        current_targets = [targets[index] for index in current_indexes]
        recent_targets = [targets[index] for index in recent_indexes]
        packing_seconds = time.perf_counter() - packing_started
        current_seconds = 0.0
        recent_seconds = 0.0
        full_tokens = 0
        if current_targets:
            current = self.engine.forward_targets(
                current_targets,
                "current",
                need_gate=False,
                batch_size=int(batch_size),
            )
            current_seconds = float(current.elapsed_seconds)
            full_tokens += int(sum(current.token_counts))
        if recent_targets:
            recent = self.engine.forward_targets(
                recent_targets,
                "recent3",
                need_gate=False,
                batch_size=int(batch_size),
            )
            recent_seconds = float(recent.elapsed_seconds)
            full_tokens += int(sum(recent.token_counts))
        gate_tokens = int(sum(feature.token_counts))
        fraction = self.extractor.early_layers / self.extractor.total_layers
        return RuntimeProductionResult(
            elapsed_seconds=float(
                feature.elapsed_seconds
                + head_seconds
                + packing_seconds
                + current_seconds
                + recent_seconds
            ),
            prefix_seconds=float(feature.elapsed_seconds),
            head_seconds=float(head_seconds),
            packing_seconds=float(packing_seconds),
            current_seconds=current_seconds,
            recent_seconds=recent_seconds,
            read_count=int(read.sum()),
            target_count=len(targets),
            raw_input_tokens=gate_tokens + full_tokens,
            layer_equivalent_tokens=gate_tokens * fraction + full_tokens,
        )


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def profile_runtime(
    config_path: Path,
    run_name: str = "validation",
    *,
    seed_override: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 7.3 runtime profiling requires CUDA")
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seed = int(seed_override or section.get("runtime_seed", 42))
    evaluation_path = paths["scores"] / f"seed{seed}_validation.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(f"Run evaluate-seed first for runtime seed={seed}")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    candidate_specs: list[tuple[int, str, float]] = []
    auto_selected = bool(section.get("runtime_profile_auto_select", True))
    if auto_selected:
        available_evaluations = [evaluation]
        for other_seed in [
            int(value) for value in section.get("seeds", [42, 43, 44])
        ]:
            if other_seed == seed:
                continue
            other_path = paths["scores"] / f"seed{other_seed}_validation.json"
            if other_path.exists():
                available_evaluations.append(
                    json.loads(other_path.read_text(encoding="utf-8"))
                )
        ranked: list[tuple[tuple[Any, ...], tuple[int, str, float]]] = []
        for candidate_id, item in evaluation["frontier"].items():
            row = item["candidate"]
            layer_name = str(row["layer"])
            objective = str(row["objective"])
            budget = float(row["requested_read_budget"])
            if (
                not layer_name.startswith("layer_")
                or not layer_name[6:].isdigit()
                or objective not in {"utility_v1", "utility_rank"}
                or budget > float(section.get("maximum_policy_read_rate", 0.50))
            ):
                continue
            rows = [
                value["frontier"][candidate_id]["candidate"]
                for value in available_evaluations
            ]
            signal_like = all(
                value.get("utility_ap_advantage_over_shuffled") is not None
                and float(value["utility_ap_advantage_over_shuffled"])
                >= float(section.get("minimum_utility_ap_advantage", 0.05))
                and float(value["calibration_diagnostics"]["required_recall"])
                >= float(section.get("minimum_required_recall", 0.90))
                and float(value["calibration_diagnostics"]["useful_recall"])
                >= float(section.get("minimum_useful_recall", 0.85))
                and float(
                    value["calibration_diagnostics"][
                        "current_sufficient_skip_rate"
                    ]
                )
                >= float(section.get("minimum_current_skip_rate", 0.25))
                for value in rows
            )
            noninferior = _all_seed_noninferior(
                rows,
                available_evaluations,
                f1_margin=float(
                    section.get("policy_f1_noninferiority_margin", 0.005)
                ),
                nll_margin=float(
                    section.get("policy_nll_noninferiority_margin", 0.015)
                ),
            )
            quality = float(
                np.mean(
                    [
                        0.5
                        * (float(value["weighted_f1"]) + float(value["macro_f1"]))
                        for value in rows
                    ]
                )
            )
            layer = int(layer_name[6:])
            ranked.append(
                (
                    (
                        int(signal_like and noninferior),
                        int(noninferior),
                        quality,
                        -layer,
                        -budget,
                    ),
                    (layer, objective, budget),
                )
            )
        ranked.sort(reverse=True)
        maximum_profiles = int(section.get("runtime_profile_max_candidates", 3))
        candidate_specs = [value for _, value in ranked[:maximum_profiles]]
    else:
        candidate_specs = [
            (int(layer), str(objective), float(budget))
            for layer in section.get("runtime_profile_layers", [2, 8, 16])
            for objective in section.get(
                "runtime_profile_objectives", ["utility_rank"]
            )
            for budget in section.get("runtime_profile_budgets", [0.3802, 0.4725])
        ]
    if not candidate_specs:
        raise RuntimeError("No numeric current-only candidate is available to profile")
    all_targets = _targets(
        read_jsonl(_resolve(root, section["processed_dir"]) / "validation.jsonl"),
        max_history=3,
    )
    maximum = int(section.get("runtime_max_targets", 128))
    if 0 < maximum < len(all_targets):
        indexes = np.linspace(0, len(all_targets) - 1, maximum).round().astype(int)
        targets = [all_targets[int(index)] for index in indexes]
        sampling = "fixed evenly-spaced validation targets"
    else:
        targets = all_targets
        sampling = "all validation targets"
    engine = AdaptiveLLMEngine(
        section,
        _stage7_1_adapter(root, section, seed),
        with_gate=False,
    )
    batch_sizes = [int(value) for value in section.get("benchmark_batch_sizes", [1, 4])]
    repeats = int(section.get("runtime_repeats", 1))
    profile: dict[str, Any] = {}
    trained_layers = [
        str(value)
        for value in json.loads(
            (_cache_dir(paths, seed) / "cache_complete.json").read_text(
                encoding="utf-8"
            )
        )["probe_layers"]
    ]
    for layer, objective, budget in candidate_specs:
        if f"layer_{layer}" not in trained_layers:
            raise ValueError(f"Runtime layer {layer} was not trained")
        router = ProfiledLayerRouter(
            engine,
            section,
            _probe_dir(paths, seed, f"layer_{layer}", objective),
            layer,
        )
        candidate_id = f"layer_{layer}__{objective}__b{budget:.4f}"
        if candidate_id not in evaluation["frontier"]:
            raise KeyError(f"Missing evaluated runtime candidate {candidate_id}")
        rule = ThresholdRule(
            **evaluation["frontier"][candidate_id]["candidate"]["threshold_rule"]
        )
        for batch_size in batch_sizes:
                key = f"{candidate_id}__batch{batch_size}"
                warmup_targets = targets[: min(len(targets), max(batch_size, 1))]
                router.route_and_score(
                    warmup_targets,
                    rule,
                    batch_size,
                    salt=seed + layer,
                )
                engine.forward_targets(
                    warmup_targets,
                    "recent3",
                    need_gate=False,
                    batch_size=batch_size,
                )
                baseline_latencies: list[float] = []
                routed_latencies: list[float] = []
                breakdown: dict[str, list[float]] = {
                    "prefix": [],
                    "head": [],
                    "packing": [],
                    "current_tail": [],
                    "recent_tail": [],
                }
                read_counts = []
                equivalent_tokens = []
                raw_tokens = []
                baseline_tokens = []
                torch.cuda.reset_peak_memory_stats()
                for repeat in range(repeats):
                    for start in range(0, len(targets), batch_size):
                        batch = targets[start : start + batch_size]
                        baseline = engine.forward_targets(
                            batch,
                            "recent3",
                            need_gate=False,
                            batch_size=batch_size,
                        )
                        routed = router.route_and_score(
                            batch,
                            rule,
                            batch_size,
                            salt=seed + layer + repeat,
                        )
                        baseline_latencies.append(float(baseline.elapsed_seconds))
                        routed_latencies.append(routed.elapsed_seconds)
                        breakdown["prefix"].append(routed.prefix_seconds)
                        breakdown["head"].append(routed.head_seconds)
                        breakdown["packing"].append(routed.packing_seconds)
                        breakdown["current_tail"].append(routed.current_seconds)
                        breakdown["recent_tail"].append(routed.recent_seconds)
                        read_counts.append(routed.read_count)
                        equivalent_tokens.append(routed.layer_equivalent_tokens)
                        raw_tokens.append(routed.raw_input_tokens)
                        baseline_tokens.append(int(sum(baseline.token_counts)))
                total_examples = len(targets) * repeats
                baseline_total = float(sum(baseline_latencies))
                routed_total = float(sum(routed_latencies))
                baseline_summary = _percentiles(baseline_latencies)
                routed_summary = _percentiles(routed_latencies)
                equivalent = float(sum(equivalent_tokens))
                baseline_token_total = float(sum(baseline_tokens))
                profile[key] = {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "layer": layer,
                    "objective": objective,
                    "requested_read_budget": budget,
                    "batch_size": batch_size,
                    "targets_per_repeat": len(targets),
                    "repeats": repeats,
                    "sampling": sampling,
                    "baseline_latency_seconds": baseline_summary,
                    "routed_latency_seconds": routed_summary,
                    "throughput_examples_per_second": {
                        "always_recent3": total_examples / baseline_total,
                        "routed": total_examples / routed_total,
                    },
                    "speedup_vs_always_recent3": baseline_total / routed_total,
                    "p95_speedup_vs_always_recent3": baseline_summary["p95"]
                    / routed_summary["p95"],
                    "context_use_rate": float(sum(read_counts) / total_examples),
                    "component_latency_seconds": {
                        name: _percentiles(values) for name, values in breakdown.items()
                    },
                    "raw_input_tokens": int(sum(raw_tokens)),
                    "layer_equivalent_tokens": equivalent,
                    "always_recent3_input_tokens": int(baseline_token_total),
                    "layer_equivalent_token_reduction": (
                        1.0 - equivalent / baseline_token_total
                        if baseline_token_total
                        else 0.0
                    ),
                    "peak_allocated_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                    "measurement_scope": (
                        "tokenization, padding-safe early prefix, probe, route packing, "
                        "selected complete path and CUDA synchronization; model load excluded"
                    ),
                }
        del router
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "runtime_seed": seed,
        "candidate_selection": (
            "automatic frozen validation-rule shortlist"
            if auto_selected
            else "explicit configuration cross-product"
        ),
        "candidate_specs": [
            {"layer": layer, "objective": objective, "budget": budget}
            for layer, objective, budget in candidate_specs
        ],
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "profiles": profile,
        "online_latency_profiled": True,
        "offline_regrouped_upper_bound_claimed": False,
        "test_data_read": False,
        "test_locked": True,
    }
    _write_json(paths["runtime"], result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _mean_sd(values: Sequence[float | None]) -> dict[str, float | None]:
    array = np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )
    if not len(array):
        return {"mean": None, "sd": None, "minimum": None, "maximum": None}
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["prediction_id"])] = row
    return result


def _ensemble_prediction(
    indexes: Sequence[dict[str, dict[str, Any]]], prediction_id: str
) -> dict[str, Any]:
    rows = [index[prediction_id] for index in indexes]
    reference = rows[0]
    for row in rows[1:]:
        for key in ("keys", "gold", "dialogue_ids"):
            if row[key] != reference[key]:
                raise ValueError(
                    f"Seed prediction alignment differs for {prediction_id}: {key}"
                )
    probability = np.mean(
        np.asarray([row["probabilities"] for row in rows], dtype=np.float64), axis=0
    )
    route_rows = [row.get("route_scores") for row in rows]
    route_score = None
    if all(value is not None for value in route_rows):
        route_score = np.mean(np.asarray(route_rows, dtype=np.float64), axis=0)
    return {
        "keys": list(reference["keys"]),
        "gold": list(reference["gold"]),
        "dialogue_ids": list(reference["dialogue_ids"]),
        "probabilities": probability,
        "route_scores": route_score,
    }


def _all_seed_noninferior(
    candidate_rows: Sequence[dict[str, Any]],
    evaluations: Sequence[dict[str, Any]],
    *,
    f1_margin: float,
    nll_margin: float,
) -> bool:
    return all(
        float(candidate["weighted_f1"])
        >= float(evaluation["baselines"]["base_recent3"]["weighted_f1"])
        - float(f1_margin)
        and float(candidate["macro_f1"])
        >= float(evaluation["baselines"]["base_recent3"]["macro_f1"])
        - float(f1_margin)
        and float(candidate["nll"])
        <= float(evaluation["baselines"]["base_recent3"]["nll"])
        + float(nll_margin)
        for candidate, evaluation in zip(candidate_rows, evaluations)
    )


def _co_primary_superiority(comparison: dict[str, Any]) -> bool:
    weighted = comparison["weighted_f1"]
    macro = comparison["macro_f1"]
    return bool(
        (
            float(weighted["ci_low"]) > 0.0
            and float(macro["ci_low"]) >= 0.0
        )
        or (
            float(macro["ci_low"]) > 0.0
            and float(weighted["ci_low"]) >= 0.0
        )
    )


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    """Aggregate all frozen validation results and emit exactly one decision."""
    if run_name != "validation":
        raise ValueError("Scientific report is available for validation only")
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    if seeds != [42, 43, 44]:
        raise ValueError("Stage 7.3 report requires frozen seeds [42, 43, 44]")
    required = [paths["utility_audit"]]
    required.extend(paths["scores"] / f"seed{seed}_validation.json" for seed in seeds)
    required.extend(
        paths["scores"] / f"seed{seed}_validation_predictions.jsonl"
        for seed in seeds
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage 7.3 report input: " + ", ".join(missing))
    evaluations = [
        json.loads(
            (paths["scores"] / f"seed{seed}_validation.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in seeds
    ]
    prediction_indexes = [
        _prediction_index(
            paths["scores"] / f"seed{seed}_validation_predictions.jsonl"
        )
        for seed in seeds
    ]
    candidate_ids = list(evaluations[0]["frontier"])
    if any(list(value["frontier"]) != candidate_ids for value in evaluations[1:]):
        raise ValueError("Candidate frontiers differ across seeds")
    labels = list(evaluations[0]["labels"])
    metric_names = (
        "context_use_rate",
        "weighted_f1",
        "macro_f1",
        "nll",
        "brier",
        "ece",
        "utility_consensus_ap",
        "utility_ap_advantage_over_shuffled",
        "layer_equivalent_token_reduction",
    )
    aggregate: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        rows = [value["frontier"][candidate_id]["candidate"] for value in evaluations]
        first = rows[0]
        metrics = {
            name: _mean_sd([row.get(name) for row in rows]) for name in metric_names
        }
        aggregate[candidate_id] = {
            "candidate_id": candidate_id,
            "layer": first["layer"],
            "objective": first["objective"],
            "requested_read_budget": float(first["requested_read_budget"]),
            "deployable": bool(first["deployable"]),
            "oracle": bool(first["oracle"]),
            "metrics": metrics,
            "per_seed": {
                str(seed): row for seed, row in zip(seeds, rows)
            },
        }
        flat = {
            "candidate_id": candidate_id,
            "layer": first["layer"],
            "objective": first["objective"],
            "requested_read_budget": first["requested_read_budget"],
            "deployable": first["deployable"],
            "oracle": first["oracle"],
        }
        for name, value in metrics.items():
            flat[f"{name}_mean"] = value["mean"]
            flat[f"{name}_sd"] = value["sd"]
        csv_rows.append(flat)
    _write_csv(paths["frontier_csv"], csv_rows)
    probe_metric_rows: list[dict[str, Any]] = []
    seen_probes: set[tuple[str, str]] = set()
    for candidate_id in candidate_ids:
        item = aggregate[candidate_id]
        key = (str(item["layer"]), str(item["objective"]))
        if key in seen_probes or item["objective"] not in BASE_OBJECTIVES:
            continue
        seen_probes.add(key)
        rows = [value["frontier"][candidate_id]["candidate"] for value in evaluations]
        probe_metric_rows.append(
            {
                "layer": item["layer"],
                "objective": item["objective"],
                "deployable": item["deployable"],
                "utility_consensus_ap": _mean_sd(
                    [row.get("utility_consensus_ap") for row in rows]
                ),
                "matched_shuffled_utility_ap": _mean_sd(
                    [row.get("matched_shuffled_utility_ap") for row in rows]
                ),
                "utility_ap_advantage_over_shuffled": _mean_sd(
                    [
                        row.get("utility_ap_advantage_over_shuffled")
                        for row in rows
                    ]
                ),
                "semantic_context_ap": _mean_sd(
                    [
                        row["calibration_diagnostics"].get(
                            "context_average_precision"
                        )
                        for row in rows
                    ]
                ),
                "utility_v1_ap": _mean_sd(
                    [
                        row["calibration_diagnostics"].get(
                            "utility_average_precision"
                        )
                        for row in rows
                    ]
                ),
                "seeds": seeds,
                "validation_labels_used_for_probe_training": False,
                "test_data_read": False,
            }
        )
    _write_jsonl(paths["layer_probe_metrics"], probe_metric_rows)
    _combine_prediction_files(
        paths["validation_predictions"],
        [
            (
                seed,
                paths["scores"] / f"seed{seed}_validation_predictions.jsonl",
            )
            for seed in seeds
        ],
    )

    baseline_aggregate = {
        baseline: {
            metric: _mean_sd(
                [
                    evaluation["baselines"][baseline].get(metric)
                    for evaluation in evaluations
                ]
            )
            for metric in ("weighted_f1", "macro_f1", "nll", "brier", "ece")
        }
        for baseline in ("base_current", "base_recent3")
    }
    bootstrap_samples = int(section.get("bootstrap_samples", 10000))
    if bootstrap_samples < 10000:
        raise ValueError("Scientific Stage 7.3 requires at least 10,000 bootstraps")
    bootstrap_seed = int(section.get("bootstrap_seed", 7390))
    base = _ensemble_prediction(prediction_indexes, "base_recent3")
    names, dialogue_index, counts = dialogue_bootstrap_counts(
        base["dialogue_ids"], bootstrap_samples, bootstrap_seed
    )
    paths["bootstrap_indices"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["bootstrap_indices"],
        dialogue_names=np.asarray(names, dtype=str),
        dialogue_index=dialogue_index.astype(np.int32),
        dialogue_counts=np.rint(counts).astype(np.int16),
        bootstrap_seed=np.asarray([bootstrap_seed], dtype=np.int64),
    )
    shared_bootstrap = (names, dialogue_index, counts)
    gold_index = [labels.index(value) for value in base["gold"]]
    maximum_budget = float(section.get("maximum_policy_read_rate", 0.50))
    f1_margin = float(section.get("policy_f1_noninferiority_margin", 0.005))
    nll_margin = float(section.get("policy_nll_noninferiority_margin", 0.015))
    minimum_ap_advantage = float(section.get("minimum_utility_ap_advantage", 0.05))
    minimum_required = float(section.get("minimum_required_recall", 0.90))
    minimum_useful = float(section.get("minimum_useful_recall", 0.85))
    minimum_skip = float(section.get("minimum_current_skip_rate", 0.25))

    audit_rows = _audit_rows_by_group(paths)
    consensus_target = np.asarray(
        [
            int(audit_rows["validation"][key]["consensus_positive"])
            for key in base["keys"]
        ],
        dtype=np.int64,
    )
    ap_bootstrap: dict[str, Any] = {}
    ap_representatives: dict[str, str] = {}
    for candidate_id in candidate_ids:
        item = aggregate[candidate_id]
        if item["objective"] not in {"utility_v1", "utility_rank"}:
            continue
        key = f"{item['layer']}/{item['objective']}"
        ap_representatives.setdefault(key, candidate_id)
    for key, candidate_id in ap_representatives.items():
        candidate = _ensemble_prediction(prediction_indexes, candidate_id)
        shuffled = _ensemble_prediction(
            prediction_indexes, candidate_id + "__matched_shuffled"
        )
        if candidate["route_scores"] is None or shuffled["route_scores"] is None:
            raise ValueError(f"Missing route scores for AP bootstrap: {candidate_id}")
        ap_bootstrap[key] = paired_dialogue_ap_bootstrap(
            consensus_target,
            candidate["route_scores"],
            shuffled["route_scores"],
            base["dialogue_ids"],
            samples=bootstrap_samples,
            seed=bootstrap_seed,
            precomputed=shared_bootstrap,
        )

    deterministic_policy_candidates: list[str] = []
    for candidate_id in candidate_ids:
        item = aggregate[candidate_id]
        rows = [value["frontier"][candidate_id]["candidate"] for value in evaluations]
        learned = (
            item["deployable"]
            and item["objective"] in BASE_OBJECTIVES
            and item["layer"] != PAIR_LAYER
        )
        within_budget = all(
            float(row["context_use_rate"]) <= maximum_budget for row in rows
        )
        if learned and within_budget and _all_seed_noninferior(
            rows,
            evaluations,
            f1_margin=f1_margin,
            nll_margin=nll_margin,
        ):
            deterministic_policy_candidates.append(candidate_id)

    policy_bootstrap: dict[str, Any] = {}
    for candidate_id in deterministic_policy_candidates:
        candidate = _ensemble_prediction(prediction_indexes, candidate_id)
        comparisons: dict[str, Any] = {}
        for control in ("matched_shuffled", "matched_random"):
            control_prediction = _ensemble_prediction(
                prediction_indexes, candidate_id + "__" + control
            )
            comparisons[f"vs_{control}"] = paired_dialogue_bootstrap(
                gold_index,
                candidate["probabilities"],
                control_prediction["probabilities"],
                base["dialogue_ids"],
                samples=bootstrap_samples,
                seed=bootstrap_seed,
                precomputed=shared_bootstrap,
            )
        policy_bootstrap[candidate_id] = comparisons

    signal_candidates: list[str] = []
    signal_checks: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        item = aggregate[candidate_id]
        if not item["deployable"] or item["objective"] not in {
            "utility_v1",
            "utility_rank",
        }:
            continue
        rows = [value["frontier"][candidate_id]["candidate"] for value in evaluations]
        ap_advantages = [
            row.get("utility_ap_advantage_over_shuffled") for row in rows
        ]
        ap_key = f"{item['layer']}/{item['objective']}"
        diagnostics = [row["calibration_diagnostics"] for row in rows]
        checks = {
            "ap_advantage_at_least_minimum_all_seeds": all(
                value is not None and float(value) >= minimum_ap_advantage
                for value in ap_advantages
            ),
            "ap_advantage_positive_all_seeds": all(
                value is not None and float(value) > 0.0 for value in ap_advantages
            ),
            "ap_bootstrap_ci_excludes_zero": float(
                ap_bootstrap[ap_key]["ci_low"]
            )
            > 0.0,
            "required_recall_all_seeds": all(
                float(value["required_recall"]) >= minimum_required
                for value in diagnostics
            ),
            "useful_recall_all_seeds": all(
                float(value["useful_recall"]) >= minimum_useful
                for value in diagnostics
            ),
            "current_skip_all_seeds": all(
                float(value["current_sufficient_skip_rate"]) >= minimum_skip
                for value in diagnostics
            ),
        }
        checks["passed"] = all(checks.values())
        signal_checks[candidate_id] = checks
        if checks["passed"]:
            signal_candidates.append(candidate_id)

    policy_candidates: list[str] = []
    policy_checks: dict[str, Any] = {}
    for candidate_id in deterministic_policy_candidates:
        comparisons = policy_bootstrap[candidate_id]
        checks = {
            "actual_read_rate_at_most_maximum_all_seeds": True,
            "all_seed_noninferiority": True,
            "superior_to_matched_shuffled": _co_primary_superiority(
                comparisons["vs_matched_shuffled"]
            ),
            "superior_to_matched_random": _co_primary_superiority(
                comparisons["vs_matched_random"]
            ),
        }
        checks["passed"] = all(checks.values())
        policy_checks[candidate_id] = checks
        if checks["passed"]:
            policy_candidates.append(candidate_id)

    oracle_candidates: list[str] = []
    for candidate_id in candidate_ids:
        item = aggregate[candidate_id]
        if not item["oracle"] or item["requested_read_budget"] > maximum_budget:
            continue
        rows = [value["frontier"][candidate_id]["candidate"] for value in evaluations]
        if all(float(row["context_use_rate"]) <= maximum_budget for row in rows) and (
            _all_seed_noninferior(
                rows,
                evaluations,
                f1_margin=f1_margin,
                nll_margin=nll_margin,
            )
        ):
            oracle_candidates.append(candidate_id)

    pair_key = f"{PAIR_LAYER}/utility_rank"
    pair_candidates = [
        candidate_id
        for candidate_id in candidate_ids
        if aggregate[candidate_id]["layer"] == PAIR_LAYER
    ]
    pair_advantages = [
        value["frontier"][pair_candidates[0]]["candidate"].get(
            "utility_ap_advantage_over_shuffled"
        )
        for value in evaluations
    ] if pair_candidates else []
    pair_aware_strong = bool(
        pair_candidates
        and all(
            value is not None and float(value) >= minimum_ap_advantage
            for value in pair_advantages
        )
        and float(ap_bootstrap[pair_key]["ci_low"]) > 0.0
    )

    runtime = (
        json.loads(paths["runtime"].read_text(encoding="utf-8"))
        if paths["runtime"].exists()
        else None
    )
    runtime_rows: list[dict[str, Any]] = []
    if runtime is not None:
        for profile_id, value in runtime.get("profiles", {}).items():
            row = {
                "profile_id": profile_id,
                "candidate_id": value["candidate_id"],
                "seed": value["seed"],
                "layer": value["layer"],
                "objective": value["objective"],
                "requested_read_budget": value["requested_read_budget"],
                "batch_size": value["batch_size"],
                "speedup_vs_always_recent3": value[
                    "speedup_vs_always_recent3"
                ],
                "p95_speedup_vs_always_recent3": value[
                    "p95_speedup_vs_always_recent3"
                ],
                "context_use_rate": value["context_use_rate"],
                "layer_equivalent_token_reduction": value[
                    "layer_equivalent_token_reduction"
                ],
                "peak_allocated_memory_bytes": value[
                    "peak_allocated_memory_bytes"
                ],
            }
            for component, metrics in value["component_latency_seconds"].items():
                for statistic in ("mean", "p50", "p95"):
                    row[f"{component}_{statistic}_seconds"] = metrics[statistic]
            runtime_rows.append(row)
    _write_csv(paths["runtime_breakdown_csv"], runtime_rows)
    minimum_speedup_1 = float(section.get("minimum_speedup_batch1", 1.05))
    minimum_speedup_4 = float(section.get("minimum_speedup_batch4", 1.03))
    minimum_token_reduction = float(
        section.get("minimum_layer_equivalent_token_reduction", 0.15)
    )
    systems_checks: dict[str, Any] = {}
    systems_candidates: list[str] = []
    if runtime is not None:
        grouped: dict[str, dict[int, dict[str, Any]]] = {}
        for value in runtime.get("profiles", {}).values():
            grouped.setdefault(str(value["candidate_id"]), {})[
                int(value["batch_size"])
            ] = value
        for candidate_id, batches in grouped.items():
            checks = {
                "batch1_speedup": 1 in batches
                and float(batches[1]["speedup_vs_always_recent3"])
                >= minimum_speedup_1,
                "batch4_speedup": 4 in batches
                and float(batches[4]["speedup_vs_always_recent3"])
                >= minimum_speedup_4,
                "p95_no_regression": all(
                    float(value["p95_speedup_vs_always_recent3"]) >= 1.0
                    for value in batches.values()
                    if int(value["batch_size"]) in {1, 4}
                )
                and {1, 4}.issubset(batches),
                "token_reduction": all(
                    float(value["layer_equivalent_token_reduction"])
                    >= minimum_token_reduction
                    for value in batches.values()
                    if int(value["batch_size"]) in {1, 4}
                )
                and {1, 4}.issubset(batches),
                "all_overheads_included": all(
                    "tokenization" in str(value.get("measurement_scope", ""))
                    and "packing" in str(value.get("measurement_scope", ""))
                    for value in batches.values()
                ),
            }
            checks["passed"] = all(checks.values())
            systems_checks[candidate_id] = checks
            if checks["passed"]:
                systems_candidates.append(candidate_id)

    target_audit = json.loads(paths["utility_audit"].read_text(encoding="utf-8"))
    target_reliability_pass = bool(target_audit["target_reliability_pass"])
    oracle_viability_pass = bool(oracle_candidates)
    signal_pass = bool(signal_candidates)
    policy_pass = bool(policy_candidates)
    signal_policy = set(signal_candidates) & set(policy_candidates)
    systems_pass = bool(signal_policy & set(systems_candidates))
    full_pass = sorted(signal_policy & set(systems_candidates))

    def selection_key(candidate_id: str) -> tuple[float, float, float]:
        item = aggregate[candidate_id]
        quality = 0.5 * (
            float(item["metrics"]["weighted_f1"]["mean"])
            + float(item["metrics"]["macro_f1"]["mean"])
        )
        return (
            quality,
            -float(item["metrics"]["context_use_rate"]["mean"]),
            -float(item["requested_read_budget"]),
        )

    selected_id: str | None = None
    if full_pass:
        selected_id = max(full_pass, key=selection_key)
    elif signal_policy:
        selected_id = max(signal_policy, key=selection_key)

    entropy_ap = max(
        (
            float(item["metrics"]["utility_consensus_ap"]["mean"])
            for item in aggregate.values()
            if item["objective"] == "full_pass_entropy"
            and item["metrics"]["utility_consensus_ap"]["mean"] is not None
        ),
        default=-math.inf,
    )
    learned_ap = max(
        (
            float(item["metrics"]["utility_consensus_ap"]["mean"])
            for item in aggregate.values()
            if item["deployable"]
            and item["objective"] in {"utility_v1", "utility_rank"}
            and item["metrics"]["utility_consensus_ap"]["mean"] is not None
        ),
        default=-math.inf,
    )
    entropy_best_ranking = entropy_ap >= learned_ap

    if not target_reliability_pass:
        decision_label = "stop_target_unreliable"
    elif not oracle_viability_pass:
        decision_label = "stop_selective_recent3_under_current_model"
    elif full_pass:
        decision_label = "permit_stage_7_4_contextual_bandit"
    elif signal_policy:
        decision_label = "continue_to_router_systems_redesign"
    elif pair_aware_strong and not signal_pass:
        decision_label = "stop_current_only_information_limit"
    elif entropy_best_ranking or not signal_pass or not policy_pass:
        decision_label = "keep_entropy_as_offline_pareto_reference"
    else:
        decision_label = "continue_to_router_systems_redesign"
    rl_permitted = decision_label == "permit_stage_7_4_contextual_bandit"
    selected = aggregate.get(selected_id) if selected_id is not None else None
    decision = {
        "stage": "7.3",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "decision": decision_label,
        "test_locked": True,
        "target_reliability_pass": target_reliability_pass,
        "oracle_viability_pass": oracle_viability_pass,
        "signal_pass": signal_pass,
        "policy_pass": policy_pass,
        "systems_pass": systems_pass,
        "jointly_passing_candidates": full_pass,
        "selected_candidate_id": selected_id,
        "selected_layer": selected["layer"] if selected is not None else None,
        "selected_read_budget": (
            selected["requested_read_budget"] if selected is not None else None
        ),
        "rl_permitted": rl_permitted,
        "reinforcement_learning_used": False,
        "test_data_read": False,
    }
    bootstrap_result = {
        "unit": "dialogue",
        "samples": bootstrap_samples,
        "seed": bootstrap_seed,
        "shared_index_file": str(paths["bootstrap_indices"]),
        "shared_index_sha256": _sha256(paths["bootstrap_indices"]),
        "ap_vs_matched_shuffled": ap_bootstrap,
        "policy_comparisons": policy_bootstrap,
        "test_data_read": False,
    }
    checks = {
        "target": target_audit,
        "signal": signal_checks,
        "policy": policy_checks,
        "oracle_viable_candidates": oracle_candidates,
        "pair_aware_strong": pair_aware_strong,
        "systems": systems_checks,
        "runtime_profile_present": runtime is not None,
        "entropy_best_ranking": entropy_best_ranking,
    }
    result = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seeds": seeds,
        "baselines": baseline_aggregate,
        "frontier": aggregate,
        "checks": checks,
        "decision": decision,
        "bootstrap_file": str(paths["bootstrap"]),
        "frontier_csv": str(paths["frontier_csv"]),
        "layer_probe_metrics": str(paths["layer_probe_metrics"]),
        "validation_predictions": str(paths["validation_predictions"]),
        "runtime_profile": str(paths["runtime"]) if runtime is not None else None,
        "runtime_breakdown_csv": str(paths["runtime_breakdown_csv"]),
        "test_data_read": False,
        "test_locked": True,
    }
    _write_json(paths["bootstrap"], bootstrap_result)
    _write_json(paths["decision"], decision)
    _write_json(paths["summary"], result)
    recent = baseline_aggregate["base_recent3"]
    lines = [
        "# Stage 7.3 validation summary",
        "",
        f"- Frozen decision: `{decision_label}`",
        f"- RL permitted: `{str(rl_permitted).lower()}`",
        "- MELD test: **hard-locked and unread**",
        "- Always recent-3 wF1: "
        f"{recent['weighted_f1']['mean']:.4f} ± {recent['weighted_f1']['sd']:.4f}",
        "- Always recent-3 macro F1: "
        f"{recent['macro_f1']['mean']:.4f} ± {recent['macro_f1']['sd']:.4f}",
        "",
        "## Frozen gates",
        "",
        "| Gate | Pass |",
        "|---|---:|",
        f"| Target reliability | {target_reliability_pass} |",
        f"| Oracle viability | {oracle_viability_pass} |",
        f"| Current-only signal | {signal_pass} |",
        f"| Learned policy | {policy_pass} |",
        f"| Runtime systems | {systems_pass} |",
        "",
        f"Selected candidate: `{selected_id}`" if selected_id else "Selected candidate: none",
        "",
        "See `budget_frontier.csv`, `paired_dialogue_bootstrap.json`, and "
        "`runtime_profile.json` for the complete evidence.",
    ]
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return result


def status(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    root = _project_root(config_path)
    section = _section(_read_yaml(config_path))
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    result = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "run_name": run_name,
        "preflight": paths["preflight"].exists(),
        "cache_complete": {
            str(seed): (_cache_dir(paths, seed) / "cache_complete.json").exists()
            for seed in seeds
        },
        "utility_audit": paths["utility_audit"].exists(),
        "probe_training": {
            str(seed): (
                paths["reports"] / f"seed{seed}_probe_training.json"
            ).exists()
            for seed in seeds
        },
        "evaluation": {
            str(seed): (
                paths["scores"] / f"seed{seed}_validation.json"
            ).exists()
            for seed in seeds
        },
        "runtime_profile": paths["runtime"].exists(),
        "report": paths["summary"].exists(),
        "frozen_decision": (
            json.loads(paths["decision"].read_text(encoding="utf-8"))
            if paths["decision"].exists()
            else None
        ),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def smoke(config_path: Path, seed: int, maximum: int = 8) -> dict[str, Any]:
    run_name = f"smoke_seed{int(seed)}"
    preflight(config_path, run_name)
    cache_seed(
        config_path,
        seed,
        run_name,
        max_targets=int(maximum),
        overwrite=True,
    )
    audit_utility_targets(config_path, run_name, seeds_override=[seed])
    train_probes(
        config_path,
        seed,
        run_name,
        overwrite=True,
        max_steps_override=3,
    )
    evaluation = evaluate_seed(config_path, seed, run_name, overwrite=True)
    result = {
        "status": "smoke_complete",
        "seed": int(seed),
        "maximum_targets_per_group": int(maximum),
        "validation_targets": evaluation["targets"],
        "scientific_result": False,
        "test_data_read": False,
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def final_test_locked() -> None:
    raise RuntimeError(
        "MELD test is hard-locked in Stage 7.3. No final-test implementation exists."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    cache = subparsers.add_parser("cache-seed")
    cache.add_argument("--seed", type=int, required=True)
    cache.add_argument("--max-targets", type=int)
    cache.add_argument("--overwrite", action="store_true")
    subparsers.add_parser("audit-targets")
    train = subparsers.add_parser("train-probes")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--overwrite", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--overwrite", action="store_true")
    runtime_parser = subparsers.add_parser("profile-runtime")
    runtime_parser.add_argument("--seed", type=int)
    subparsers.add_parser("report")
    subparsers.add_parser("status")
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--seed", type=int, default=42)
    smoke_parser.add_argument("--max-targets", type=int, default=8)
    subparsers.add_parser("final-test")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if args.command == "preflight":
        preflight(config_path)
    elif args.command == "cache-seed":
        cache_seed(
            config_path,
            args.seed,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
        )
    elif args.command == "audit-targets":
        audit_utility_targets(config_path)
    elif args.command == "train-probes":
        train_probes(
            config_path,
            args.seed,
            overwrite=args.overwrite,
            max_steps_override=args.max_steps,
        )
    elif args.command == "evaluate":
        evaluate_seed(config_path, args.seed, overwrite=args.overwrite)
    elif args.command == "profile-runtime":
        profile_runtime(config_path, seed_override=args.seed)
    elif args.command == "report":
        report(config_path)
    elif args.command == "status":
        status(config_path)
    elif args.command == "smoke":
        smoke(config_path, args.seed, args.max_targets)
    elif args.command == "final-test":
        final_test_locked()
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
