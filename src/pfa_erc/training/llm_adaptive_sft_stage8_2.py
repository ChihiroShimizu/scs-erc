"""Stage 8.2: full-context adaptive fast/slow SFT for MELD ERC.

The model always receives the complete causal dialogue through the target turn.
There is no context gate, recent-k window, retrieval action, or future access.
Stage 8.2 distils a sparse *reasoning-strategy* policy from cached FAST/SLOW
counterfactuals produced by three frozen Stage 7 adapters, continues LoRA SFT,
and evaluates the learned single-model policy on MELD validation.  MELD test is
hard locked.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import re
import shutil
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import precision_recall_fscore_support

from pfa_erc.data import read_jsonl
from pfa_erc.training.llm_full_context_fast_slow_stage8 import (
    LABELS,
    FullContextGenerationEngine,
    FullContextTarget,
    _aggregate,
    _project_root,
    _read_jsonl,
    _render_prompt,
    _resolve,
    _sha256_bytes,
    _sha256_file,
    _stable_hash,
    _write_json,
    _write_jsonl,
    audit_response_semantics,
    build_full_context_prompt,
    build_full_context_targets,
    classification_and_cost_metrics,
    oracle_budget_frontier,
    paired_dialogue_bootstrap,
    render_causal_dialogue,
    response_prefix_through_emotion,
)


VERSION = "1.0.0"
EXPERIMENT = "llm_adaptive_sft_stage8_2"
PROTOCOL_VERSION = "full-causal-dialogue-adaptive-sft-v1"
POLICY_VARIANTS = ("adaptive_sft", "always_fast_sft")
EVALUATION_POLICIES = ("adaptive", "adaptive_prompt_forced_fast")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("llm_adaptive_sft_stage8_2")
    if not isinstance(value, dict):
        raise KeyError("Missing llm_adaptive_sft_stage8_2 configuration section")
    return value


def _paths(root: Path, section: dict[str, Any], run_name: str) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get("artifact_dir", "artifacts/llm_adaptive_sft_stage8_2/meld"),
    )
    if run_name != "validation":
        base = base / run_name
    reports = base / "reports"
    datasets = base / "datasets"
    return {
        "base": base,
        "counterfactuals": base / "counterfactuals" / "train",
        "datasets": datasets,
        "utility_labels": datasets / "train_utility_labels.jsonl",
        "policy_train": datasets / "policy_train.jsonl",
        "policy_calibration": datasets / "policy_calibration.jsonl",
        "dataset_summary": reports / "dataset_summary.json",
        "adapters": base / "adapters",
        "predictions": base / "predictions" / "validation",
        "reports": reports,
        "manifest": reports / "lock_manifest.json",
        "summary": reports / "validation_summary.json",
        "summary_md": reports / "validation_summary.md",
        "bootstrap": reports / "paired_dialogue_bootstrap.json",
        "decision": reports / "frozen_decision.json",
    }


def _stage7_adapter_dir(root: Path, section: dict[str, Any], seed: int) -> Path:
    base = _resolve(
        root,
        section.get(
            "stage7_adapter_root",
            "artifacts/llm_evidence_supervision_stage7/meld/adapters",
        ),
    )
    variant = str(section.get("source_adapter_variant", "full_supervised"))
    return base / f"seed{seed}" / variant


def _stage8_2_adapter_dir(
    paths: dict[str, Path], seed: int, variant: str
) -> Path:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown Stage 8.2 policy variant: {variant}")
    return paths["adapters"] / f"seed{seed}" / variant


def _stage8_1_base(root: Path, section: dict[str, Any]) -> Path:
    return _resolve(
        root,
        section.get(
            "stage8_1_artifact_dir",
            "artifacts/llm_full_context_fast_slow_stage8/meld",
        ),
    )


def _stage8_1_prediction_path(
    root: Path, section: dict[str, Any], seed: int, mode: str
) -> Path:
    variant = str(section.get("source_adapter_variant", "full_supervised"))
    return (
        _stage8_1_base(root, section)
        / "predictions"
        / f"seed{seed}_{variant}_{mode}.jsonl"
    )


def _counterfactual_path(paths: dict[str, Path], seed: int, mode: str) -> Path:
    return paths["counterfactuals"] / f"seed{seed}_{mode}.jsonl"


def _validation_prediction_path(
    paths: dict[str, Path], seed: int, variant: str, policy: str
) -> Path:
    return paths["predictions"] / f"seed{seed}_{variant}_{policy}.jsonl"


def _load_targets(
    root: Path, section: dict[str, Any], split: str
) -> list[FullContextTarget]:
    if split not in {"train", "validation"}:
        raise ValueError("Stage 8.2 may read only MELD train or validation")
    path = _resolve(root, section["processed_dir"]) / f"{split}.jsonl"
    if "test" in path.name.lower() or not path.exists():
        raise FileNotFoundError(f"Missing safe MELD {split} file: {path}")
    return build_full_context_targets(read_jsonl(path))


def build_adaptive_sft_prompt(
    target: FullContextTarget,
    labels: Sequence[str],
    slow_reasoning_word_limit: int = 80,
) -> str:
    """One-model policy prompt; every strategy sees identical full context."""
    dialogue = render_causal_dialogue(target)
    return (
        "Task: classify the target speaker's expressed emotion. The complete causal "
        "dialogue stream is already supplied. Never retrieve, omit, or select context.\n\n"
        f"Complete causal dialogue stream:\n{dialogue}\n\n"
        f"Allowed labels: {', '.join(labels)}.\n"
        "First choose the response strategy. Use SLOW only when explicit deliberation "
        "is likely to correct or disambiguate a direct emotion prediction; merely having "
        "earlier dialogue is not a reason to choose SLOW. Otherwise use FAST.\n"
        "Return exactly one of:\n"
        "<FAST>\n<EMOTION>label</EMOTION>\n\n"
        "or\n\n"
        "<SLOW>\n"
        f"<REASONING>reasoning under {slow_reasoning_word_limit} words</REASONING>\n"
        "<EMOTION>label</EMOTION>"
    )


def extract_reasoning_text(text: str, word_limit: int = 80) -> str:
    """Recover a rationale from exact or malformed Stage 8.1 SLOW output."""
    match = re.search(
        r"<\s*REASONING\s*>(.*?)(?:</\s*REASONING\s*>|<\s*EMOTION\s*>)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        value = match.group(1)
    else:
        value = re.sub(
            r"<\s*EMOTION\s*>.*?</\s*EMOTION\s*>",
            " ",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        value = re.sub(
            r"</?\s*(?:SLOW|SLW|FAST|REASONING)\s*>",
            " ",
            value,
            flags=re.IGNORECASE,
        )
    value = re.sub(r"\s+", " ", value).strip(" \n\t:;-")
    words = value.split()
    return " ".join(words[:word_limit]).strip()


def _render_user_prompt(engine: FullContextGenerationEngine, user_prompt: str) -> str:
    return _render_prompt(
        engine.tokenizer,
        user_prompt,
        enable_thinking=bool(engine.section.get("enable_thinking", False)),
    )


def _generate_records(
    engine: FullContextGenerationEngine,
    targets: Sequence[FullContextTarget],
    labels: Sequence[str],
    output_path: Path,
    *,
    seed: int,
    split: str,
    source: str,
    policy: str,
    prompt_builder: Callable[[FullContextTarget], str],
    expected_mode: str,
    max_new_tokens: int,
    batch_size: int,
    overwrite: bool,
    forced_response_prefix: str = "",
) -> list[dict[str, Any]]:
    if overwrite and output_path.exists():
        output_path.unlink()
    cached = _read_jsonl(output_path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in cached:
        if row.get("experiment") != EXPERIMENT or row.get("version") != VERSION:
            raise ValueError(f"Incompatible cached Stage 8.2 record in {output_path}")
        if row.get("seed") != seed or row.get("split") != split:
            raise ValueError(f"Cached Stage 8.2 seed/split mismatch in {output_path}")
        if row.get("source") != source or row.get("policy") != policy:
            raise ValueError(f"Cached Stage 8.2 source/policy mismatch in {output_path}")
        key = str(row["key"])
        if key in by_key:
            raise ValueError(f"Duplicate cached Stage 8.2 key: {key}")
        by_key[key] = row
    target_keys = {target.key for target in targets}
    extras = set(by_key) - target_keys
    if extras:
        raise ValueError(f"Cached predictions contain unexpected keys: {sorted(extras)[:3]}")
    for target in targets:
        if target.key not in by_key:
            continue
        expected_rendered = (
            _render_user_prompt(engine, prompt_builder(target)) + forced_response_prefix
        )
        if by_key[target.key].get("prompt_sha256") != _sha256_bytes(expected_rendered):
            raise ValueError(
                f"Cached Stage 8.2 prompt differs for {target.key}; use --overwrite"
            )

    missing = [target for target in targets if target.key not in by_key]
    log_every = max(1, int(engine.section.get("log_every_batches", 25)))
    for start in range(0, len(missing), batch_size):
        batch_targets = missing[start : start + batch_size]
        user_prompts = [prompt_builder(target) for target in batch_targets]
        rendered = [
            _render_user_prompt(engine, value) + forced_response_prefix
            for value in user_prompts
        ]
        generated_rows, timing = engine.generate_batch(rendered, max_new_tokens)
        new_rows: list[dict[str, Any]] = []
        for target, rendered_prompt, generated in zip(
            batch_targets, rendered, generated_rows
        ):
            response = forced_response_prefix + str(generated["response"])
            audit = audit_response_semantics(response, labels, expected_mode)
            prefix = response_prefix_through_emotion(response, labels)
            prefix_tokens = len(
                engine.tokenizer.encode(forced_response_prefix, add_special_tokens=False)
            )
            completion_tokens = int(generated["completion_tokens"]) + prefix_tokens
            tokens_to_emotion = (
                len(engine.tokenizer.encode(prefix, add_special_tokens=False))
                if prefix is not None
                else completion_tokens
            )
            dialogue = render_causal_dialogue(target)
            row = {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "split": split,
                "source": source,
                "policy": policy,
                "mode": expected_mode,
                "key": target.key,
                "dialogue_id": target.dialogue_id,
                "turn_index": target.turn_index,
                "utterance_id": target.utterance_id,
                "gold": target.gold,
                "history_turn_count": len(target.history),
                "causal_turn_count": len(target.history) + 1,
                "dialogue_sha256": _sha256_bytes(dialogue),
                "prompt_sha256": _sha256_bytes(rendered_prompt),
                "dialogue_tokens": len(
                    engine.tokenizer.encode(dialogue, add_special_tokens=False)
                ),
                "response": response,
                "prompt_tokens": int(generated["prompt_tokens"]) - prefix_tokens,
                "completion_tokens": completion_tokens,
                "forced_prefix_tokens": prefix_tokens,
                "tokens_to_emotion": tokens_to_emotion,
                **audit,
                **timing,
                "test_data_read": False,
            }
            new_rows.append(row)
            by_key[target.key] = row
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        batch_number = start // batch_size + 1
        total_batches = max(1, (len(missing) + batch_size - 1) // batch_size)
        if (
            batch_number == 1
            or batch_number == total_batches
            or batch_number % log_every == 0
        ):
            print(
                f"[Stage8.2] split={split} seed={seed} source={source} "
                f"policy={policy} completed={min(start + len(batch_targets), len(missing))}/"
                f"{len(missing)} cached={len(cached)}",
                flush=True,
            )
    ordered = [by_key[target.key] for target in targets]
    _write_jsonl(output_path, ordered)
    return ordered


def preflight(
    config_path: Path,
    run_name: str = "validation",
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    if str(section.get("dataset", "")).lower() != "meld":
        raise ValueError("Stage 8.2 currently accepts MELD only")
    if section.get("context_policy") != "full_causal_dialogue":
        raise ValueError("context_policy must be exactly full_causal_dialogue")
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    processed = _resolve(root, section["processed_dir"])
    train_targets = _load_targets(root, section, "train")
    validation_targets = _load_targets(root, section, "validation")
    requested = [int(value) for value in (seeds or section.get("seeds", [42, 43, 44]))]
    missing: list[str] = []
    adapter_hashes: dict[str, str] = {}
    for seed in requested:
        adapter = _stage7_adapter_dir(root, section, seed)
        cfg = adapter / "adapter_config.json"
        if not cfg.exists():
            missing.append(str(adapter))
        else:
            adapter_hashes[str(seed)] = _sha256_file(cfg)
        for mode in ("fast", "slow"):
            baseline = _stage8_1_prediction_path(root, section, seed, mode)
            if not baseline.exists():
                missing.append(str(baseline))
    if missing:
        raise FileNotFoundError("Missing Stage 8 prerequisites:\n" + "\n".join(missing))
    manifest = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MELD",
        "training_split": "train",
        "evaluation_split": "validation",
        "context_policy": "full_causal_dialogue",
        "context_gate_used": False,
        "context_truncation_used": False,
        "future_turns_visible": False,
        "validation_used_for_sft": False,
        "test_locked": True,
        "seeds": requested,
        "train_targets": len(train_targets),
        "validation_targets": len(validation_targets),
        "stage7_adapter_config_sha256": adapter_hashes,
        "config_sha256": _sha256_file(config_path),
        "train_sha256": _sha256_file(processed / "train.jsonl"),
        "validation_sha256": _sha256_file(processed / "validation.jsonl"),
    }
    _write_json(paths["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def collect_train_counterfactuals(
    config_path: Path,
    seed: int,
    run_name: str = "validation",
    *,
    max_targets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    targets = _load_targets(root, section, "train")
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    labels = [str(value) for value in section.get("labels", LABELS)]
    engine = FullContextGenerationEngine(section, _stage7_adapter_dir(root, section, seed))
    batch_size = int(section.get("counterfactual_batch_size", 4))
    results: dict[str, Any] = {}
    for mode in ("fast", "slow"):
        rows = _generate_records(
            engine,
            targets,
            labels,
            _counterfactual_path(paths, seed, mode),
            seed=seed,
            split="train",
            source="frozen_stage7_full_supervised",
            policy=f"counterfactual_{mode}",
            prompt_builder=lambda target, selected=mode: build_full_context_prompt(
                target,
                labels,
                selected,
                int(section.get("slow_reasoning_word_limit", 80)),
            ),
            expected_mode=mode,
            max_new_tokens=int(
                section.get("max_new_tokens", {}).get(
                    mode, 24 if mode == "fast" else 160
                )
            ),
            batch_size=batch_size,
            overwrite=overwrite,
        )
        results[mode] = classification_and_cost_metrics(rows, labels)
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    output = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "split": "train",
        "targets": len(targets),
        "modes": results,
        "validation_used": False,
        "test_locked": True,
    }
    _write_json(paths["reports"] / f"counterfactual_seed{seed}.json", output)
    return output


def _aligned_counterfactual_maps(
    per_seed: dict[int, dict[str, Sequence[dict[str, Any]]]]
) -> tuple[list[str], dict[int, dict[str, dict[str, dict[str, Any]]]]]:
    indexed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    key_set: set[str] | None = None
    for seed, modes in per_seed.items():
        indexed[seed] = {}
        for mode in ("fast", "slow"):
            mapping = {str(row["key"]): row for row in modes[mode]}
            if len(mapping) != len(modes[mode]):
                raise ValueError(f"Duplicate counterfactual key for seed={seed} mode={mode}")
            indexed[seed][mode] = mapping
            key_set = set(mapping) if key_set is None else key_set
            if set(mapping) != key_set:
                raise ValueError("Counterfactual seeds/modes do not contain identical keys")
    return sorted(key_set or []), indexed


def consensus_utility_rows(
    per_seed: dict[int, dict[str, Sequence[dict[str, Any]]]],
    targets: Sequence[FullContextTarget],
    labels: Sequence[str],
    *,
    minimum_benefit_seeds: int,
    maximum_harm_seeds: int,
    reasoning_word_limit: int,
) -> list[dict[str, Any]]:
    """Create seed-consensus utility labels without using validation for SFT."""
    keys, indexed = _aligned_counterfactual_maps(per_seed)
    target_map = {target.key: target for target in targets}
    if set(keys) != set(target_map):
        raise ValueError("Counterfactual rows and processed targets are not identical")
    result: list[dict[str, Any]] = []
    for key in keys:
        target = target_map[key]
        outcomes: dict[str, str] = {}
        benefit_count = 0
        harm_count = 0
        teachers: list[tuple[int, int, str]] = []
        dialogue_hash: str | None = None
        for seed in sorted(indexed):
            fast = indexed[seed]["fast"][key]
            slow = indexed[seed]["slow"][key]
            if fast["gold"] != target.gold or slow["gold"] != target.gold:
                raise ValueError(f"Gold mismatch for {key}")
            if fast["dialogue_sha256"] != slow["dialogue_sha256"]:
                raise ValueError(f"Full-context invariant failed for {key}")
            dialogue_hash = dialogue_hash or str(fast["dialogue_sha256"])
            if dialogue_hash != str(fast["dialogue_sha256"]):
                raise ValueError(f"Dialogue differs across seeds for {key}")
            fast_correct = fast.get("prediction") == target.gold
            slow_correct = slow.get("prediction") == target.gold
            state = f"{int(fast_correct)}{int(slow_correct)}"
            outcomes[str(seed)] = state
            benefit_count += int(not fast_correct and slow_correct)
            harm_count += int(fast_correct and not slow_correct)
            if slow_correct:
                reasoning = extract_reasoning_text(
                    str(slow.get("response", "")), reasoning_word_limit
                )
                if len(reasoning.split()) >= 4:
                    teachers.append(
                        (int(slow.get("completion_tokens", 10**9)), seed, reasoning)
                    )
        robust_slow = (
            benefit_count >= minimum_benefit_seeds
            and harm_count <= maximum_harm_seeds
        )
        conflicting = benefit_count > 0 and harm_count > 0
        teacher = min(teachers)[2] if robust_slow and teachers else ""
        if robust_slow and teacher:
            policy_target = "slow"
            include_in_sft = True
        elif conflicting or (robust_slow and not teacher):
            policy_target = "uncertain"
            include_in_sft = False
        else:
            policy_target = "fast"
            include_in_sft = True
        adaptive_response = (
            f"<SLOW>\n<REASONING>{teacher}</REASONING>\n"
            f"<EMOTION>{target.gold}</EMOTION>"
            if policy_target == "slow"
            else f"<FAST>\n<EMOTION>{target.gold}</EMOTION>"
        )
        result.append(
            {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "key": key,
                "dialogue_id": target.dialogue_id,
                "turn_index": target.turn_index,
                "utterance_id": target.utterance_id,
                "gold": target.gold,
                "dialogue_sha256": dialogue_hash,
                "seed_outcomes_fast_slow_correct": outcomes,
                "benefit_count": benefit_count,
                "harm_count": harm_count,
                "policy_target": policy_target,
                "include_in_sft": include_in_sft,
                "teacher_reasoning": teacher or None,
                "user_prompt": build_adaptive_sft_prompt(
                    target, labels, reasoning_word_limit
                ),
                "adaptive_response": adaptive_response,
                "always_fast_response": f"<FAST>\n<EMOTION>{target.gold}</EMOTION>",
                "test_data_read": False,
            }
        )
    return result


def split_policy_rows(
    rows: Sequence[dict[str, Any]], calibration_fraction: float, split_seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between zero and one")
    train: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    boundary = int(calibration_fraction * 1_000_000)
    assignment: dict[str, str] = {}
    for row in rows:
        if not row.get("include_in_sft"):
            continue
        dialogue_id = str(row["dialogue_id"])
        bucket = _stable_hash(dialogue_id, split_seed) % 1_000_000
        split = "calibration" if bucket < boundary else "train"
        prior = assignment.setdefault(dialogue_id, split)
        if prior != split:
            raise AssertionError("Dialogue-disjoint split assignment changed")
        value = {**row, "policy_split": split}
        (calibration if split == "calibration" else train).append(value)
    if not train or not calibration:
        raise ValueError("Policy train/calibration split produced an empty partition")
    if {row["dialogue_id"] for row in train} & {
        row["dialogue_id"] for row in calibration
    }:
        raise AssertionError("Policy split is not dialogue-disjoint")
    return train, calibration


def build_policy_dataset(
    config_path: Path,
    run_name: str = "validation",
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    labels = [str(value) for value in section.get("labels", LABELS)]
    per_seed: dict[int, dict[str, Sequence[dict[str, Any]]]] = {}
    for seed in seeds:
        per_seed[seed] = {
            mode: _read_jsonl(_counterfactual_path(paths, seed, mode))
            for mode in ("fast", "slow")
        }
        if not per_seed[seed]["fast"] or not per_seed[seed]["slow"]:
            raise FileNotFoundError(
                f"Run collect for seed {seed} before building the policy dataset"
            )
    targets = _load_targets(root, section, "train")
    rows = consensus_utility_rows(
        per_seed,
        targets,
        labels,
        minimum_benefit_seeds=int(section.get("minimum_benefit_seeds", 2)),
        maximum_harm_seeds=int(section.get("maximum_harm_seeds", 0)),
        reasoning_word_limit=int(section.get("teacher_reasoning_word_limit", 80)),
    )
    train, calibration = split_policy_rows(
        rows,
        float(section.get("policy_calibration_fraction", 0.10)),
        int(section.get("policy_split_seed", 8230)),
    )
    _write_jsonl(paths["utility_labels"], rows)
    _write_jsonl(paths["policy_train"], train)
    _write_jsonl(paths["policy_calibration"], calibration)
    target_counts = Counter(str(row["policy_target"]) for row in rows)
    train_counts = Counter(str(row["policy_target"]) for row in train)
    calibration_counts = Counter(str(row["policy_target"]) for row in calibration)
    per_emotion_slow = Counter(
        str(row["gold"]) for row in rows if row["policy_target"] == "slow"
    )
    summary = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "source_split": "train",
        "validation_used_for_sft": False,
        "targets": len(rows),
        "target_counts": dict(target_counts),
        "robust_slow_rate": target_counts.get("slow", 0) / max(1, len(rows)),
        "policy_train_records": len(train),
        "policy_calibration_records": len(calibration),
        "policy_train_counts": dict(train_counts),
        "policy_calibration_counts": dict(calibration_counts),
        "slow_targets_per_emotion": dict(per_emotion_slow),
        "minimum_benefit_seeds": int(section.get("minimum_benefit_seeds", 2)),
        "maximum_harm_seeds": int(section.get("maximum_harm_seeds", 0)),
        "dialogue_disjoint_policy_split": True,
        "utility_labels_sha256": _sha256_file(paths["utility_labels"]),
        "policy_train_sha256": _sha256_file(paths["policy_train"]),
        "policy_calibration_sha256": _sha256_file(paths["policy_calibration"]),
        "test_locked": True,
    }
    _write_json(paths["dataset_summary"], summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def tokenize_sft_record(
    record: dict[str, Any],
    tokenizer: Any,
    variant: str,
    *,
    enable_thinking: bool,
    max_prompt_tokens: int,
    max_sequence_tokens: int,
    strategy_token_weight: float,
) -> dict[str, Any]:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown SFT variant: {variant}")
    response_key = (
        "adaptive_response" if variant == "adaptive_sft" else "always_fast_response"
    )
    response = str(record[response_key])
    rendered = _render_prompt(
        tokenizer,
        str(record["user_prompt"]),
        enable_thinking=enable_thinking,
    )
    prompt_ids = list(tokenizer.encode(rendered, add_special_tokens=False))
    response_ids = list(tokenizer.encode(response, add_special_tokens=False))
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not response_ids or response_ids[-1] != int(eos_id)):
        response_ids.append(int(eos_id))
    input_ids = prompt_ids + response_ids
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError(
            f"Full causal prompt has {len(prompt_ids)} tokens, exceeding "
            f"max_prompt_tokens={max_prompt_tokens}; truncation is forbidden"
        )
    if len(input_ids) > max_sequence_tokens:
        raise ValueError(
            f"Full SFT sequence has {len(input_ids)} tokens, exceeding "
            f"max_sequence_tokens={max_sequence_tokens}; increase the limit explicitly"
        )
    labels = [-100] * len(prompt_ids) + response_ids
    loss_weights = [0.0] * len(prompt_ids) + [1.0] * len(response_ids)
    strategy = "<SLOW>" if response.lstrip().upper().startswith("<SLOW>") else "<FAST>"
    strategy_ids = list(tokenizer.encode(strategy, add_special_tokens=False))
    for index in range(min(len(strategy_ids), len(response_ids))):
        loss_weights[len(prompt_ids) + index] = float(strategy_token_weight)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_weights": loss_weights,
    }


class WeightedSFTCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(item["input_ids"]) for item in features)
        input_ids: list[list[int]] = []
        attention: list[list[int]] = []
        labels: list[list[int]] = []
        weights: list[list[float]] = []
        for item in features:
            padding = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
            weights.append(item["loss_weights"] + [0.0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "loss_weights": torch.tensor(weights, dtype=torch.float32),
        }


class _ListDataset(torch.utils.data.Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def _build_weighted_trainer_class(base: Any) -> Any:
    class WeightedTrainer(base):  # type: ignore[misc, valid-type]
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            weights = inputs.pop("loss_weights")
            labels = inputs["labels"]
            outputs = model(**inputs)
            logits = outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_weights = weights[:, 1:].to(shift_logits.device).contiguous()
            valid = shift_labels.ne(-100)
            safe_labels = shift_labels.masked_fill(~valid, 0)
            token_loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                safe_labels.view(-1),
                reduction="none",
            ).view_as(safe_labels)
            effective_weights = shift_weights * valid.to(shift_weights.dtype)
            loss = (token_loss * effective_weights).sum() / effective_weights.sum().clamp_min(1.0)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def train_policy_adapter(
    config_path: Path,
    seed: int,
    variant: str,
    run_name: str = "validation",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown policy variant: {variant}")
    try:
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Stage 8.2 training requires transformers, accelerate and peft"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 8.2 SFT requires CUDA")
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    output_adapter = _stage8_2_adapter_dir(paths, seed, variant)
    training_dir = paths["base"] / "training" / f"seed{seed}" / variant
    if (output_adapter / "adapter_config.json").exists() and not overwrite:
        metadata_path = output_adapter / "stage8_2_training_metadata.json"
        value = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {"status": "cached_adapter", "adapter": str(output_adapter)}
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return value
    if overwrite and output_adapter.exists():
        shutil.rmtree(output_adapter)
    if overwrite and training_dir.exists():
        shutil.rmtree(training_dir)
    raw_train = _read_jsonl(paths["policy_train"])
    raw_calibration = _read_jsonl(paths["policy_calibration"])
    if not raw_train or not raw_calibration:
        raise FileNotFoundError("Run build-dataset before Stage 8.2 SFT")
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(
        section["model_name"],
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        local_files_only=bool(section.get("local_files_only", False)),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    max_sequence_tokens = int(section.get("max_sequence_tokens", 4352))
    strategy_weight = float(section.get("strategy_token_weight", 8.0))
    tokenize = lambda row: tokenize_sft_record(
        row,
        tokenizer,
        variant,
        enable_thinking=bool(section.get("enable_thinking", False)),
        max_prompt_tokens=int(section.get("max_prompt_tokens", 4096)),
        max_sequence_tokens=max_sequence_tokens,
        strategy_token_weight=strategy_weight,
    )
    train_rows = [tokenize(row) for row in raw_train]
    calibration_rows = [tokenize(row) for row in raw_calibration]
    dtype = (
        torch.bfloat16
        if str(section.get("dtype", "bfloat16")) == "bfloat16"
        else torch.float16
    )
    base = AutoModelForCausalLM.from_pretrained(
        section["model_name"],
        torch_dtype=dtype,
        trust_remote_code=bool(section.get("trust_remote_code", False)),
        local_files_only=bool(section.get("local_files_only", False)),
        low_cpu_mem_usage=True,
    )
    source_adapter = _stage7_adapter_dir(root, section, seed)
    model = PeftModel.from_pretrained(base, str(source_adapter), is_trainable=True)
    model.config.use_cache = False
    if bool(section.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    argument_values: dict[str, Any] = {
        "output_dir": str(training_dir),
        "num_train_epochs": float(section.get("num_train_epochs", 1.0)),
        "per_device_train_batch_size": int(section.get("train_batch_size", 1)),
        "per_device_eval_batch_size": int(section.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(section.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(section.get("learning_rate", 5e-5)),
        "warmup_ratio": float(section.get("warmup_ratio", 0.03)),
        "weight_decay": float(section.get("weight_decay", 0.0)),
        "logging_steps": int(section.get("logging_steps", 10)),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "bf16": dtype == torch.bfloat16,
        "fp16": dtype == torch.float16,
        "report_to": [],
        "remove_unused_columns": False,
        "gradient_checkpointing": bool(section.get("gradient_checkpointing", True)),
        "seed": seed,
        "data_seed": int(section.get("policy_split_seed", 8230)) + seed,
        "dataloader_num_workers": 0,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        argument_values["eval_strategy"] = "epoch"
    else:
        argument_values["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**argument_values)
    WeightedTrainer = _build_weighted_trainer_class(Trainer)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=_ListDataset(train_rows),
        eval_dataset=_ListDataset(calibration_rows),
        data_collator=WeightedSFTCollator(int(tokenizer.pad_token_id)),
    )
    resume = any(training_dir.glob("checkpoint-*")) and not overwrite
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    output_adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_adapter, safe_serialization=True)
    tokenizer.save_pretrained(output_adapter)
    metadata = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "variant": variant,
        "initial_adapter": str(source_adapter),
        "train_records": len(train_rows),
        "calibration_records": len(calibration_rows),
        "strategy_token_weight": strategy_weight,
        "policy_train_sha256": _sha256_file(paths["policy_train"]),
        "policy_calibration_sha256": _sha256_file(paths["policy_calibration"]),
        "train_runtime": float(result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(result.metrics.get("train_loss", float("nan"))),
        "validation_used_for_sft": False,
        "full_context_only": True,
        "test_locked": True,
    }
    _write_json(output_adapter / "stage8_2_training_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    del trainer, model, base
    gc.collect()
    torch.cuda.empty_cache()
    return metadata


def _validation_consensus_rows(
    root: Path,
    section: dict[str, Any],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    per_seed = {
        seed: {
            mode: _read_jsonl(_stage8_1_prediction_path(root, section, seed, mode))
            for mode in ("fast", "slow")
        }
        for seed in seeds
    }
    return consensus_utility_rows(
        per_seed,
        _load_targets(root, section, "validation"),
        labels,
        minimum_benefit_seeds=int(section.get("minimum_benefit_seeds", 2)),
        maximum_harm_seeds=int(section.get("maximum_harm_seeds", 0)),
        reasoning_word_limit=int(section.get("teacher_reasoning_word_limit", 80)),
    )


def strategy_selection_metrics(
    predictions: Sequence[dict[str, Any]],
    utility_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    truth_by_key = {
        str(row["key"]): row.get("policy_target") == "slow" for row in utility_rows
    }
    predicted_by_key = {
        str(row["key"]): row.get("semantic_strategy") == "slow"
        for row in predictions
    }
    if set(truth_by_key) != set(predicted_by_key):
        raise ValueError("Strategy predictions and consensus utility labels are not aligned")
    keys = sorted(truth_by_key)
    truth = [truth_by_key[key] for key in keys]
    predicted = [predicted_by_key[key] for key in keys]
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=[True],
        average="macro",
        zero_division=0,
    )
    tp = sum(a and b for a, b in zip(truth, predicted))
    fp = sum((not a) and b for a, b in zip(truth, predicted))
    fn = sum(a and (not b) for a, b in zip(truth, predicted))
    return {
        "targets": len(keys),
        "consensus_slow_targets": int(sum(truth)),
        "consensus_slow_rate": float(np.mean(truth)),
        "selected_slow_targets": int(sum(predicted)),
        "selected_slow_rate": float(np.mean(predicted)),
        "slow_precision": float(precision),
        "slow_recall": float(recall),
        "slow_f1": float(f1),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def evaluate_policy_seed(
    config_path: Path,
    seed: int,
    variant: str,
    run_name: str = "validation",
    *,
    policy: str = "adaptive",
    max_targets: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown policy variant: {variant}")
    if policy not in EVALUATION_POLICIES:
        raise ValueError(f"Unknown evaluation policy: {policy}")
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    labels = [str(value) for value in section.get("labels", LABELS)]
    targets = _load_targets(root, section, "validation")
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    adapter = _stage8_2_adapter_dir(paths, seed, variant)
    engine = FullContextGenerationEngine(section, adapter)
    forced_prefix = "<FAST>\n" if policy == "adaptive_prompt_forced_fast" else ""
    max_tokens = int(
        section.get("max_new_tokens", {}).get(
            "fast" if forced_prefix else "adaptive", 24 if forced_prefix else 160
        )
    )
    rows = _generate_records(
        engine,
        targets,
        labels,
        _validation_prediction_path(paths, seed, variant, policy),
        seed=seed,
        split="validation",
        source=variant,
        policy=policy,
        prompt_builder=lambda target: build_adaptive_sft_prompt(
            target,
            labels,
            int(section.get("slow_reasoning_word_limit", 80)),
        ),
        expected_mode="adaptive",
        max_new_tokens=max_tokens,
        batch_size=int(section.get("evaluation_batch_size", 4)),
        overwrite=overwrite,
        forced_response_prefix=forced_prefix,
    )
    metrics = classification_and_cost_metrics(rows, labels)
    utility = _validation_consensus_rows(root, section, labels)
    if max_targets is not None:
        selected = {target.key for target in targets}
        utility = [row for row in utility if row["key"] in selected]
    strategy_metrics = strategy_selection_metrics(rows, utility)
    result = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "variant": variant,
        "policy": policy,
        "split": "validation",
        "targets": len(rows),
        "classification_and_cost": metrics,
        "strategy_selection": strategy_metrics,
        "context_policy": "full_causal_dialogue",
        "test_locked": True,
    }
    _write_json(
        paths["reports"] / f"seed{seed}_{variant}_{policy}_evaluation.json",
        result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def matched_random_records(
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
    adaptive: Sequence[dict[str, Any]],
    random_seed: int,
) -> list[dict[str, Any]]:
    fast_map = {str(row["key"]): row for row in fast}
    slow_map = {str(row["key"]): row for row in slow}
    adaptive_map = {str(row["key"]): row for row in adaptive}
    if set(fast_map) != set(slow_map) or set(fast_map) != set(adaptive_map):
        raise ValueError("Matched-random inputs are not aligned")
    keys = sorted(fast_map)
    slow_count = sum(
        adaptive_map[key].get("semantic_strategy") == "slow" for key in keys
    )
    ranked = sorted(keys, key=lambda key: (_stable_hash(key, random_seed), key))
    selected = set(ranked[:slow_count])
    result: list[dict[str, Any]] = []
    for key in keys:
        source = slow_map[key] if key in selected else fast_map[key]
        result.append(
            {
                **source,
                "mode": "adaptive",
                "policy": "matched_random",
                "strategy": "slow" if key in selected else "fast",
                "semantic_strategy": "slow" if key in selected else "fast",
            }
        )
    return result


def _runtime_subset(
    targets: Sequence[FullContextTarget], count: int, seed: int
) -> list[FullContextTarget]:
    ranked = sorted(targets, key=lambda target: (_stable_hash(target.key, seed), target.key))
    return ranked[: min(len(ranked), count)]


def runtime_profile_seed(
    config_path: Path,
    seed: int,
    variant: str = "adaptive_sft",
    run_name: str = "validation",
    *,
    max_targets: int | None = None,
    repeats_override: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    output = paths["reports"] / f"runtime_profile_seed{seed}_{variant}.json"
    if output.exists() and not overwrite:
        value = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return value
    labels = [str(value) for value in section.get("labels", LABELS)]
    count = int(max_targets or section.get("runtime_profile_targets", 128))
    targets = _runtime_subset(
        _load_targets(root, section, "validation"),
        count,
        int(section.get("runtime_subset_seed", 8130)),
    )
    engine = FullContextGenerationEngine(
        section, _stage8_2_adapter_dir(paths, seed, variant)
    )
    rendered = [
        _render_user_prompt(
            engine,
            build_adaptive_sft_prompt(
                target,
                labels,
                int(section.get("slow_reasoning_word_limit", 80)),
            ),
        )
        for target in targets
    ]
    repeats = int(repeats_override or section.get("runtime_repeats", 3))
    warmups = int(section.get("runtime_warmup_batches", 2))
    max_new_tokens = int(section.get("max_new_tokens", {}).get("adaptive", 160))
    result: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "variant": variant,
        "targets": len(targets),
        "repeats": repeats,
        "isolated_fixed_workload": True,
        "batch_sizes": {},
        "test_locked": True,
    }
    for batch_size in [int(value) for value in section.get("runtime_batch_sizes", [1, 4])]:
        for warmup in range(warmups):
            start = (warmup * batch_size) % len(rendered)
            engine.generate_batch(
                rendered[start : start + batch_size] or rendered[:batch_size],
                max_new_tokens,
            )
        batch_latency: list[float] = []
        amortized: list[float] = []
        completions: list[int] = []
        to_emotion: list[int] = []
        strategies: list[str | None] = []
        examples = 0
        total_seconds = 0.0
        peak_memory = 0
        for _ in range(repeats):
            for start in range(0, len(rendered), batch_size):
                outputs, timing = engine.generate_batch(
                    rendered[start : start + batch_size], max_new_tokens
                )
                batch_latency.append(float(timing["batch_latency_ms"]))
                amortized.extend(
                    [float(timing["amortized_latency_ms"])] * len(outputs)
                )
                for generated in outputs:
                    response = str(generated["response"])
                    audit = audit_response_semantics(response, labels, "adaptive")
                    strategies.append(audit["semantic_strategy"])
                    completions.append(int(generated["completion_tokens"]))
                    prefix = response_prefix_through_emotion(response, labels)
                    to_emotion.append(
                        len(engine.tokenizer.encode(prefix, add_special_tokens=False))
                        if prefix is not None
                        else int(generated["completion_tokens"])
                    )
                examples += len(outputs)
                total_seconds += float(timing["batch_latency_ms"]) / 1000.0
                peak_memory = max(peak_memory, int(timing["peak_cuda_memory_bytes"]))
        key = f"batch{batch_size}"
        result["batch_sizes"][key] = {
            "examples": examples,
            "batches": len(batch_latency),
            "examples_per_second": examples / total_seconds,
            "mean_batch_latency_ms": float(np.mean(batch_latency)),
            "p95_batch_latency_ms": float(np.quantile(batch_latency, 0.95)),
            "mean_amortized_latency_ms": float(np.mean(amortized)),
            "p95_amortized_latency_ms": float(np.quantile(amortized, 0.95)),
            "mean_completion_tokens": float(np.mean(completions)),
            "mean_tokens_to_emotion": float(np.mean(to_emotion)),
            "p95_tokens_to_emotion": float(np.quantile(to_emotion, 0.95)),
            "semantic_slow_rate": float(np.mean([value == "slow" for value in strategies])),
            "peak_cuda_memory_bytes": peak_memory,
        }
    baseline_runtime = (
        _stage8_1_base(root, section)
        / "reports"
        / f"runtime_profile_seed{seed}.json"
    )
    if baseline_runtime.exists():
        baseline = json.loads(baseline_runtime.read_text(encoding="utf-8"))
        for key, values in result["batch_sizes"].items():
            slow_eps = float(baseline["modes"]["slow"][key]["examples_per_second"])
            values["speedup_vs_stage8_1_slow"] = (
                float(values["examples_per_second"]) / slow_eps
            )
    _write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return result


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    labels = [str(value) for value in section.get("labels", LABELS)]
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    method_rows: dict[int, dict[str, list[dict[str, Any]]]] = {}
    evaluations: dict[int, dict[str, Any]] = {}
    bootstrap: dict[str, Any] = {}
    oracle_values: list[float] = []
    oracle_rates: list[float] = []
    for seed in seeds:
        fast = _read_jsonl(_stage8_1_prediction_path(root, section, seed, "fast"))
        slow = _read_jsonl(_stage8_1_prediction_path(root, section, seed, "slow"))
        adaptive = _read_jsonl(
            _validation_prediction_path(paths, seed, "adaptive_sft", "adaptive")
        )
        forced = _read_jsonl(
            _validation_prediction_path(
                paths, seed, "adaptive_sft", "adaptive_prompt_forced_fast"
            )
        )
        always_fast = _read_jsonl(
            _validation_prediction_path(paths, seed, "always_fast_sft", "adaptive")
        )
        if not all((fast, slow, adaptive, forced, always_fast)):
            raise FileNotFoundError(
                f"Missing Stage 8.2 validation predictions for seed {seed}"
            )
        matched = matched_random_records(
            fast,
            slow,
            adaptive,
            int(section.get("matched_random_seed", 8232)) + seed,
        )
        frontier = oracle_budget_frontier(
            fast,
            slow,
            labels,
            [1.0],
            int(section.get("oracle_tie_seed", 8233)) + seed,
        )[0]
        oracle_values.append(float(frontier["weighted_f1"]))
        oracle_rates.append(float(frontier["actual_slow_rate"]))
        method_rows[seed] = {
            "stage8_1_fast": fast,
            "stage8_1_slow": slow,
            "adaptive_sft": adaptive,
            "always_fast_sft": always_fast,
            "adaptive_prompt_forced_fast": forced,
            "matched_random": matched,
        }
        evaluation_path = (
            paths["reports"] / f"seed{seed}_adaptive_sft_adaptive_evaluation.json"
        )
        if not evaluation_path.exists():
            raise FileNotFoundError(evaluation_path)
        evaluations[seed] = json.loads(evaluation_path.read_text(encoding="utf-8"))
        bootstrap[f"seed{seed}_adaptive_sft_minus_stage8_1_fast"] = (
            paired_dialogue_bootstrap(
                fast,
                adaptive,
                labels,
                int(section.get("bootstrap_samples", 2000)),
                int(section.get("bootstrap_seed", 8234)) + seed,
            )
        )
        bootstrap[f"seed{seed}_adaptive_sft_minus_matched_random"] = (
            paired_dialogue_bootstrap(
                matched,
                adaptive,
                labels,
                int(section.get("bootstrap_samples", 2000)),
                int(section.get("bootstrap_seed", 8234)) + 100 + seed,
            )
        )
        bootstrap[f"seed{seed}_always_fast_sft_minus_stage8_1_fast"] = (
            paired_dialogue_bootstrap(
                fast,
                always_fast,
                labels,
                int(section.get("bootstrap_samples", 2000)),
                int(section.get("bootstrap_seed", 8234)) + 200 + seed,
            )
        )

    metric_names = (
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "parse_success_rate",
        "strict_format_success_rate",
        "strategy_tag_success_rate",
        "semantic_slow_rate",
        "reasoning_present_rate",
        "reasoning_before_emotion_rate",
        "mean_completion_tokens",
        "mean_tokens_to_emotion",
        "p95_tokens_to_emotion",
    )
    methods = list(method_rows[seeds[0]])
    aggregates: dict[str, Any] = {}
    for method in methods:
        aggregates[method] = {}
        for metric in metric_names:
            aggregates[method][metric] = _aggregate(
                [
                    float(
                        classification_and_cost_metrics(
                            method_rows[seed][method], labels
                        )[metric]
                    )
                    for seed in seeds
                ]
            )
    strategy_aggregates = {
        metric: _aggregate(
            [
                float(evaluations[seed]["strategy_selection"][metric])
                for seed in seeds
            ]
        )
        for metric in (
            "consensus_slow_rate",
            "selected_slow_rate",
            "slow_precision",
            "slow_recall",
            "slow_f1",
        )
    }
    runtime_profiles: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = paths["reports"] / f"runtime_profile_seed{seed}_adaptive_sft.json"
        if path.exists():
            runtime_profiles[seed] = json.loads(path.read_text(encoding="utf-8"))
    runtime_aggregates: dict[str, Any] = {}
    if len(runtime_profiles) == len(seeds):
        for batch_size in [int(value) for value in section.get("runtime_batch_sizes", [1, 4])]:
            key = f"batch{batch_size}"
            runtime_aggregates[key] = {
                metric: _aggregate(
                    [float(runtime_profiles[seed]["batch_sizes"][key][metric]) for seed in seeds]
                )
                for metric in (
                    "examples_per_second",
                    "mean_amortized_latency_ms",
                    "p95_amortized_latency_ms",
                    "mean_completion_tokens",
                    "mean_tokens_to_emotion",
                    "semantic_slow_rate",
                    "speedup_vs_stage8_1_slow",
                )
            }

    fast_wf1 = float(aggregates["stage8_1_fast"]["weighted_f1"]["mean"])
    adaptive_wf1 = float(aggregates["adaptive_sft"]["weighted_f1"]["mean"])
    matched_wf1 = float(aggregates["matched_random"]["weighted_f1"]["mean"])
    oracle_wf1 = float(np.mean(oracle_values))
    oracle_gap = max(1e-12, oracle_wf1 - fast_wf1)
    oracle_gap_closure = (adaptive_wf1 - fast_wf1) / oracle_gap
    slow_rate = float(aggregates["adaptive_sft"]["semantic_slow_rate"]["mean"])
    format_rate = float(
        aggregates["adaptive_sft"]["strict_format_success_rate"]["mean"]
    )
    parse_rate = float(aggregates["adaptive_sft"]["parse_success_rate"]["mean"])
    completion_tokens = float(
        aggregates["adaptive_sft"]["mean_completion_tokens"]["mean"]
    )
    tolerance = float(section.get("classification_tolerance", 0.003))
    checks = {
        "quality_noninferior_to_fast": adaptive_wf1 >= fast_wf1 - tolerance,
        "selective_value_over_matched_random": adaptive_wf1
        >= matched_wf1 + float(section.get("minimum_gain_over_matched_random", 0.0)),
        "minimum_oracle_gap_closure": oracle_gap_closure
        >= float(section.get("minimum_oracle_gap_closure", 0.10)),
        "calibrated_slow_rate": float(section.get("minimum_slow_rate", 0.05))
        <= slow_rate
        <= float(section.get("maximum_slow_rate", 0.12)),
        "parse_success": parse_rate >= float(section.get("minimum_parse_success", 1.0)),
        "strict_format_success": format_rate
        >= float(section.get("minimum_strict_format_success", 0.99)),
        "completion_token_budget": completion_tokens
        <= float(section.get("maximum_mean_completion_tokens", 20.0)),
    }
    if runtime_aggregates:
        checks["runtime_speedup"] = (
            float(runtime_aggregates["batch1"]["speedup_vs_stage8_1_slow"]["mean"])
            >= float(section.get("minimum_batch1_speedup", 3.0))
            and float(runtime_aggregates["batch4"]["speedup_vs_stage8_1_slow"]["mean"])
            >= float(section.get("minimum_batch4_speedup", 3.0))
        )
    passed = all(checks.values()) and bool(runtime_aggregates)
    status_value = (
        "stage8_2_sft_established_ready_for_preregistered_stage8_3"
        if passed
        else "revise_stage8_2_sft_rl_remains_blocked"
    )
    decision = {
        "status": status_value,
        "fast_weighted_f1": fast_wf1,
        "adaptive_sft_weighted_f1": adaptive_wf1,
        "matched_random_weighted_f1": matched_wf1,
        "oracle_weighted_f1": oracle_wf1,
        "oracle_realized_slow_rate": float(np.mean(oracle_rates)),
        "oracle_gap_closure": oracle_gap_closure,
        "adaptive_sft_semantic_slow_rate": slow_rate,
        "adaptive_sft_mean_completion_tokens": completion_tokens,
        "checks": checks,
        "rl_status": (
            "eligible_for_separately_preregistered_comparison; not_run"
            if passed
            else "not_permitted; adaptive_sft has not passed all frozen checks"
        ),
        "test_locked": True,
    }
    final = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "dataset": "MELD",
        "split": "validation",
        "seeds": seeds,
        "context_policy": "full_causal_dialogue",
        "aggregates": aggregates,
        "strategy_selection": strategy_aggregates,
        "runtime": runtime_aggregates,
        "decision": decision,
        "test_locked": True,
    }
    _write_json(paths["summary"], final)
    _write_json(paths["bootstrap"], bootstrap)
    _write_json(paths["decision"], decision)
    lines = [
        "# Stage 8.2 — Full-context Adaptive SFT",
        "",
        "> MELD validation-only. All methods receive the complete causal dialogue. "
        "No context gate, recent-k truncation, validation training, or test access.",
        "",
        "| Method | wF1 | Macro F1 | Accuracy | Completion tokens | Semantic SLOW "
        "| Strict format |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        values = aggregates[method]
        lines.append(
            f"| {method} | {values['weighted_f1']['mean']:.4f} ± "
            f"{values['weighted_f1']['std']:.4f} "
            f"| {values['macro_f1']['mean']:.4f} ± {values['macro_f1']['std']:.4f} "
            f"| {values['accuracy']['mean']:.4f} ± {values['accuracy']['std']:.4f} "
            f"| {values['mean_completion_tokens']['mean']:.2f} "
            f"| {values['semantic_slow_rate']['mean']:.2%} "
            f"| {values['strict_format_success_rate']['mean']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"- Status: `{status_value}`",
            f"- Adaptive SFT vs fixed FAST wF1: {adaptive_wf1 - fast_wf1:+.4f}",
            f"- Adaptive SFT vs matched random wF1: {adaptive_wf1 - matched_wf1:+.4f}",
            f"- Oracle gap closure: {oracle_gap_closure:.2%}",
            f"- Semantic SLOW rate: {slow_rate:.2%}",
            f"- Mean completion tokens: {completion_tokens:.2f}",
            f"- RL status: `{decision['rl_status']}`",
            "- MELD test remains locked.",
        ]
    )
    paths["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return final


def status(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    section = _section(_read_yaml(config_path))
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    value = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "preflight": paths["manifest"].exists(),
        "counterfactuals": {
            str(seed): {
                mode: _counterfactual_path(paths, seed, mode).exists()
                for mode in ("fast", "slow")
            }
            for seed in seeds
        },
        "dataset": {
            "utility_labels": paths["utility_labels"].exists(),
            "policy_train": paths["policy_train"].exists(),
            "policy_calibration": paths["policy_calibration"].exists(),
        },
        "adapters": {
            str(seed): {
                variant: (
                    _stage8_2_adapter_dir(paths, seed, variant) / "adapter_config.json"
                ).exists()
                for variant in POLICY_VARIANTS
            }
            for seed in seeds
        },
        "predictions": {
            str(seed): {
                f"{variant}:{policy}": _validation_prediction_path(
                    paths, seed, variant, policy
                ).exists()
                for variant, policy in (
                    ("adaptive_sft", "adaptive"),
                    ("adaptive_sft", "adaptive_prompt_forced_fast"),
                    ("always_fast_sft", "adaptive"),
                )
            }
            for seed in seeds
        },
        "runtime_profiles": {
            str(seed): (
                paths["reports"] / f"runtime_profile_seed{seed}_adaptive_sft.json"
            ).exists()
            for seed in seeds
        },
        "final_report": paths["summary"].exists(),
        "test_locked": True,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def final_test_locked() -> None:
    raise RuntimeError(
        "Stage 8.2 cannot read MELD test. SFT must pass every frozen validation "
        "criterion before a separate hash-locked test patch can be created."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    commands = parser.add_subparsers(dest="command", required=True)

    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--seeds", nargs="*", type=int)

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--seed", type=int, required=True)
    collect_parser.add_argument("--max-targets", type=int)
    collect_parser.add_argument("--overwrite", action="store_true")

    commands.add_parser("build-dataset")

    train_parser = commands.add_parser("train")
    train_parser.add_argument("--seed", type=int, required=True)
    train_parser.add_argument("--variant", choices=POLICY_VARIANTS, required=True)
    train_parser.add_argument("--overwrite", action="store_true")

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--variant", choices=POLICY_VARIANTS, required=True)
    evaluate_parser.add_argument(
        "--policy", choices=EVALUATION_POLICIES, default="adaptive"
    )
    evaluate_parser.add_argument("--max-targets", type=int)
    evaluate_parser.add_argument("--overwrite", action="store_true")

    profile_parser = commands.add_parser("profile")
    profile_parser.add_argument("--seed", type=int, required=True)
    profile_parser.add_argument(
        "--variant", choices=POLICY_VARIANTS, default="adaptive_sft"
    )
    profile_parser.add_argument("--max-targets", type=int)
    profile_parser.add_argument("--repeats", type=int)
    profile_parser.add_argument("--overwrite", action="store_true")

    commands.add_parser("report")
    commands.add_parser("status")
    commands.add_parser("final-test")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.config, args.run_name, args.seeds)
    elif args.command == "collect":
        collect_train_counterfactuals(
            args.config,
            args.seed,
            args.run_name,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
        )
    elif args.command == "build-dataset":
        build_policy_dataset(args.config, args.run_name)
    elif args.command == "train":
        train_policy_adapter(
            args.config,
            args.seed,
            args.variant,
            args.run_name,
            overwrite=args.overwrite,
        )
    elif args.command == "evaluate":
        evaluate_policy_seed(
            args.config,
            args.seed,
            args.variant,
            args.run_name,
            policy=args.policy,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
        )
    elif args.command == "profile":
        runtime_profile_seed(
            args.config,
            args.seed,
            args.variant,
            args.run_name,
            max_targets=args.max_targets,
            repeats_override=args.repeats,
            overwrite=args.overwrite,
        )
    elif args.command == "report":
        report(args.config, args.run_name)
    elif args.command == "status":
        status(args.config, args.run_name)
    elif args.command == "final-test":
        final_test_locked()


if __name__ == "__main__":
    main()
