"""Stage 8.2R: post-SFT counterfactual and strategy-margin audit.

This is a validation-only diagnostic revision of Stage 8.2.  It does not train
another adapter.  Every condition receives the same complete causal dialogue.
The routing threshold is fixed from the dialogue-disjoint MELD-train
calibration partition before it is applied to MELD validation.  MELD test is
hard locked.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from pfa_erc.data import read_jsonl
from pfa_erc.training.llm_adaptive_sft_stage8_2 import (
    POLICY_VARIANTS,
    build_adaptive_sft_prompt,
)
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
    build_full_context_targets,
    classification_and_cost_metrics,
    oracle_budget_frontier,
    paired_dialogue_bootstrap,
    render_causal_dialogue,
    response_prefix_through_emotion,
)


EXPERIMENT = "llm_adaptive_sft_stage8_2r"
VERSION = "1.0.1"
PROTOCOL_VERSION = "full-causal-dialogue-post-sft-counterfactual-v1"
AUDITED_VARIANT = "adaptive_sft"
CONTROL_VARIANT = "always_fast_sft"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("llm_adaptive_sft_stage8_2r")
    if not isinstance(value, dict):
        raise KeyError("Missing llm_adaptive_sft_stage8_2r configuration section")
    return value


def _paths(root: Path, section: dict[str, Any], run_name: str) -> dict[str, Path]:
    base = _resolve(
        root,
        section.get("artifact_dir", "artifacts/llm_adaptive_sft_stage8_2r/meld"),
    )
    if run_name != "validation":
        base = base / run_name
    reports = base / "reports"
    predictions = base / "predictions" / "validation"
    scores = base / "strategy_scores"
    return {
        "base": base,
        "reports": reports,
        "predictions": predictions,
        "scores": scores,
        "manifest": reports / "lock_manifest.json",
        "summary": reports / "validation_summary.json",
        "summary_md": reports / "validation_summary.md",
        "decision": reports / "frozen_decision.json",
        "bootstrap": reports / "paired_dialogue_bootstrap.json",
    }


def _stage8_2_base(root: Path, section: dict[str, Any]) -> Path:
    return _resolve(
        root,
        section.get(
            "stage8_2_artifact_dir",
            "artifacts/llm_adaptive_sft_stage8_2/meld",
        ),
    )


def _adapter_dir(
    root: Path, section: dict[str, Any], seed: int, variant: str
) -> Path:
    if variant not in POLICY_VARIANTS:
        raise ValueError(f"Unknown Stage 8.2 variant: {variant}")
    return _stage8_2_base(root, section) / "adapters" / f"seed{seed}" / variant


def _stage8_2_prediction_path(
    root: Path, section: dict[str, Any], seed: int, variant: str, policy: str
) -> Path:
    return (
        _stage8_2_base(root, section)
        / "predictions"
        / "validation"
        / f"seed{seed}_{variant}_{policy}.jsonl"
    )


def _calibration_path(root: Path, section: dict[str, Any]) -> Path:
    return _stage8_2_base(root, section) / "datasets" / "policy_calibration.jsonl"


def _policy_train_path(root: Path, section: dict[str, Any]) -> Path:
    return _stage8_2_base(root, section) / "datasets" / "policy_train.jsonl"


def _fixed_slow_path(paths: dict[str, Path], seed: int) -> Path:
    return paths["predictions"] / f"seed{seed}_{AUDITED_VARIANT}_fixed_slow.jsonl"


def _routed_path(paths: dict[str, Path], seed: int, policy: str) -> Path:
    return paths["predictions"] / f"seed{seed}_{AUDITED_VARIANT}_{policy}.jsonl"


def _score_path(paths: dict[str, Path], seed: int, split: str) -> Path:
    return paths["scores"] / f"seed{seed}_{split}_strategy_margin.jsonl"


def _load_validation_targets(
    root: Path, section: dict[str, Any]
) -> list[FullContextTarget]:
    path = _resolve(root, section["processed_dir"]) / "validation.jsonl"
    if "test" in path.name.lower() or not path.exists():
        raise FileNotFoundError(f"Missing safe MELD validation file: {path}")
    return build_full_context_targets(read_jsonl(path))


def strategy_tag_spec(tokenizer: Any) -> dict[str, Any]:
    """Return the first token position at which FAST and SLOW tags diverge."""
    fast_ids = [int(value) for value in tokenizer.encode("<FAST>", add_special_tokens=False)]
    slow_ids = [int(value) for value in tokenizer.encode("<SLOW>", add_special_tokens=False)]
    if not fast_ids or not slow_ids or fast_ids == slow_ids:
        raise ValueError("FAST/SLOW tags must tokenize to distinct non-empty sequences")
    common = 0
    for fast_id, slow_id in zip(fast_ids, slow_ids):
        if fast_id != slow_id:
            break
        common += 1
    if common >= min(len(fast_ids), len(slow_ids)):
        raise ValueError("One strategy tag token sequence is a prefix of the other")
    return {
        "fast_ids": fast_ids,
        "slow_ids": slow_ids,
        "common_prefix_ids": fast_ids[:common],
        "common_prefix_length": common,
        "fast_divergent_token_id": fast_ids[common],
        "slow_divergent_token_id": slow_ids[common],
        "fast_divergent_token_text": tokenizer.decode([fast_ids[common]]),
        "slow_divergent_token_text": tokenizer.decode([slow_ids[common]]),
        "score": "slow_logit_minus_fast_logit_at_first_divergent_strategy_token",
    }


def _tag_margin_batch(
    engine: FullContextGenerationEngine,
    rendered_prompts: Sequence[str],
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score one pairwise strategy decision without generating a full response."""
    common_ids = [int(value) for value in spec["common_prefix_ids"]]
    sequences = [
        [int(token) for token in engine.tokenizer.encode(prompt, add_special_tokens=False)]
        + common_ids
        for prompt in rendered_prompts
    ]
    too_long = [len(value) for value in sequences if len(value) > engine.max_prompt_tokens]
    if too_long:
        raise ValueError(
            "Full causal prompt plus common strategy prefix exceeds max_prompt_tokens; "
            f"largest={max(too_long)} limit={engine.max_prompt_tokens}"
        )
    width = max(len(value) for value in sequences)
    pad_id = int(engine.tokenizer.pad_token_id)
    input_ids = []
    attention = []
    for value in sequences:
        padding = width - len(value)
        input_ids.append([pad_id] * padding + value)
        attention.append([0] * padding + [1] * len(value))
    device = next(engine.model.parameters()).device
    encoded = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }
    with torch.inference_mode():
        generated = engine.model.generate(
            **encoded,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=pad_id,
            eos_token_id=int(engine.tokenizer.eos_token_id),
        )
    if not generated.scores:
        raise RuntimeError("Transformers generate returned no strategy logits")
    logits = generated.scores[0].float()
    fast_id = int(spec["fast_divergent_token_id"])
    slow_id = int(spec["slow_divergent_token_id"])
    margins = logits[:, slow_id] - logits[:, fast_id]
    greedy_ids = generated.sequences[:, width]
    return [
        {
            "strategy_margin": float(margins[index].item()),
            "pairwise_strategy": (
                "slow" if float(margins[index].item()) >= 0.0 else "fast"
            ),
            "greedy_next_token_id": int(greedy_ids[index].item()),
            "greedy_next_token_text": engine.tokenizer.decode(
                [int(greedy_ids[index].item())]
            ),
        }
        for index in range(len(rendered_prompts))
    ]


def _score_items(
    engine: FullContextGenerationEngine,
    items: Sequence[dict[str, Any]],
    output_path: Path,
    *,
    seed: int,
    split: str,
    batch_size: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = strategy_tag_spec(engine.tokenizer)
    rendered_by_key = {
        str(item["key"]): _render_prompt(
            engine.tokenizer,
            str(item["user_prompt"]),
            enable_thinking=bool(engine.section.get("enable_thinking", False)),
        )
        for item in items
    }
    if overwrite and output_path.exists():
        output_path.unlink()
    cached = _read_jsonl(output_path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in cached:
        if row.get("experiment") != EXPERIMENT or row.get("version") != VERSION:
            raise ValueError(f"Incompatible cached strategy score in {output_path}")
        if int(row.get("seed", -1)) != seed or row.get("split") != split:
            raise ValueError(f"Cached score seed/split mismatch in {output_path}")
        key = str(row["key"])
        if row.get("prompt_sha256") != _sha256_bytes(rendered_by_key[key]):
            raise ValueError(f"Cached prompt changed for {key}; rerun with --overwrite")
        by_key[key] = row
    expected_keys = [str(item["key"]) for item in items]
    if set(by_key) - set(expected_keys):
        raise ValueError(f"Cached scores contain unexpected keys in {output_path}")
    missing = [item for item in items if str(item["key"]) not in by_key]
    log_every = max(1, int(engine.section.get("log_every_batches", 25)))
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        rendered = [rendered_by_key[str(item["key"])] for item in batch]
        scored = _tag_margin_batch(engine, rendered, spec)
        new_rows = []
        for item, prompt, score in zip(batch, rendered, scored):
            row = {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "split": split,
                "key": str(item["key"]),
                "dialogue_id": str(item["dialogue_id"]),
                "gold": str(item["gold"]),
                "policy_target": item.get("policy_target"),
                "prompt_sha256": _sha256_bytes(prompt),
                **score,
                "test_data_read": False,
            }
            by_key[str(item["key"])] = row
            new_rows.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        batch_number = start // batch_size + 1
        total_batches = max(1, math.ceil(len(missing) / batch_size))
        if batch_number == 1 or batch_number == total_batches or batch_number % log_every == 0:
            print(
                f"[Stage8.2R] score split={split} seed={seed} "
                f"completed={min(start + len(batch), len(missing))}/{len(missing)} "
                f"cached={len(cached)}",
                flush=True,
            )
    ordered = [by_key[key] for key in expected_keys]
    _write_jsonl(output_path, ordered)
    return ordered, spec


def ranking_metrics(
    score_rows: Sequence[dict[str, Any]], truth_by_key: dict[str, bool]
) -> dict[str, Any]:
    keys = [str(row["key"]) for row in score_rows]
    if set(keys) != set(truth_by_key) or len(keys) != len(set(keys)):
        raise ValueError("Strategy scores and binary truth labels are not aligned")
    truth = np.asarray([bool(truth_by_key[key]) for key in keys], dtype=np.int64)
    score = np.asarray(
        [float(row["strategy_margin"]) for row in score_rows], dtype=np.float64
    )
    positives = int(truth.sum())
    prevalence = positives / max(1, len(truth))
    if positives == 0:
        average_precision = 0.0
        roc_auc = 0.5
    elif positives == len(truth):
        average_precision = 1.0
        roc_auc = 0.5
    else:
        average_precision = float(average_precision_score(truth, score))
        roc_auc = float(roc_auc_score(truth, score))
    ranked = np.argsort(-score, kind="stable")
    at_rate: dict[str, Any] = {}
    for rate in (0.05, 0.10):
        count = max(1, int(math.ceil(rate * len(truth))))
        selected = ranked[:count]
        tp = int(truth[selected].sum())
        precision = tp / count
        recall = tp / positives if positives else 0.0
        at_rate[f"top_{int(rate * 100)}pct"] = {
            "selected": count,
            "true_positive": tp,
            "precision": precision,
            "recall": recall,
            "precision_lift_over_prevalence": precision - prevalence,
        }
    positive_scores = score[truth == 1]
    negative_scores = score[truth == 0]
    return {
        "records": len(truth),
        "positives": positives,
        "prevalence": prevalence,
        "average_precision": average_precision,
        "average_precision_lift_over_prevalence": average_precision - prevalence,
        "roc_auc": roc_auc,
        "positive_margin_mean": (
            float(positive_scores.mean()) if len(positive_scores) else None
        ),
        "positive_margin_median": (
            float(np.median(positive_scores)) if len(positive_scores) else None
        ),
        "negative_margin_mean": (
            float(negative_scores.mean()) if len(negative_scores) else None
        ),
        "negative_margin_median": (
            float(np.median(negative_scores)) if len(negative_scores) else None
        ),
        "precision_at_rate": at_rate,
    }


def normalized_strategy_weight_audit(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    strategy_token_weight: float,
) -> dict[str, Any]:
    """Audit the effective tag mass under Stage 8.2 batch-size-one loss.

    Stage 8.2 normalizes each microbatch loss by the sum of token weights. With
    a device batch size of one, long SLOW rationales therefore dilute their tag
    supervision more than short FAST responses even though both tags use the
    same nominal multiplier.
    """
    by_class: dict[str, list[dict[str, float]]] = {"fast": [], "slow": []}
    eos_extra = 1 if tokenizer.eos_token_id is not None else 0
    for row in rows:
        strategy = str(row.get("policy_target", "")).lower()
        if strategy not in by_class:
            continue
        response = str(row["adaptive_response"])
        tag = "<SLOW>" if strategy == "slow" else "<FAST>"
        response_tokens = len(tokenizer.encode(response, add_special_tokens=False)) + eos_extra
        tag_tokens = len(tokenizer.encode(tag, add_special_tokens=False))
        total_weight = response_tokens + (strategy_token_weight - 1.0) * tag_tokens
        tag_mass = strategy_token_weight * tag_tokens / max(total_weight, 1e-12)
        by_class[strategy].append(
            {
                "response_tokens": float(response_tokens),
                "tag_tokens": float(tag_tokens),
                "normalized_tag_mass": float(tag_mass),
            }
        )
    result: dict[str, Any] = {
        "strategy_token_weight_applies_to_both_classes": True,
        "device_train_batch_size": 1,
        "per_record_loss_normalization": True,
        "classes": {},
    }
    for strategy, values in by_class.items():
        if not values:
            raise ValueError(f"No {strategy.upper()} rows available for weight audit")
        result["classes"][strategy] = {
            "records": len(values),
            "mean_response_tokens": float(
                np.mean([value["response_tokens"] for value in values])
            ),
            "mean_tag_tokens": float(
                np.mean([value["tag_tokens"] for value in values])
            ),
            "mean_normalized_tag_mass_per_record": float(
                np.mean([value["normalized_tag_mass"] for value in values])
            ),
            "aggregate_normalized_tag_mass": float(
                np.sum([value["normalized_tag_mass"] for value in values])
            ),
        }
    fast_mass = float(result["classes"]["fast"]["aggregate_normalized_tag_mass"])
    slow_mass = float(result["classes"]["slow"]["aggregate_normalized_tag_mass"])
    result["fast_to_slow_aggregate_tag_mass_ratio"] = fast_mass / max(
        slow_mass, 1e-12
    )
    return result


def calibration_threshold(
    score_rows: Sequence[dict[str, Any]], truth_by_key: dict[str, bool]
) -> dict[str, Any]:
    """Match the held-out train-calibration SLOW prevalence without val tuning."""
    if {str(row["key"]) for row in score_rows} != set(truth_by_key):
        raise ValueError("Calibration scores and labels are not aligned")
    scores = sorted(
        [float(row["strategy_margin"]) for row in score_rows], reverse=True
    )
    target_count = sum(bool(value) for value in truth_by_key.values())
    if target_count <= 0:
        threshold = float("inf")
    elif target_count >= len(scores):
        threshold = float("-inf")
    else:
        upper = scores[target_count - 1]
        lower = scores[target_count]
        threshold = (upper + lower) / 2.0 if upper > lower else upper
    selected = sum(float(row["strategy_margin"]) >= threshold for row in score_rows)
    return {
        "threshold": float(threshold),
        "comparison": "strategy_margin >= threshold",
        "source_split": "train_policy_calibration",
        "target_slow_count": int(target_count),
        "target_slow_rate": target_count / max(1, len(scores)),
        "selected_slow_count": int(selected),
        "selected_slow_rate": selected / max(1, len(scores)),
        "validation_labels_used": False,
        "test_used": False,
    }


def counterfactual_states(
    fast: Sequence[dict[str, Any]], slow: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    fast_map = {str(row["key"]): row for row in fast}
    slow_map = {str(row["key"]): row for row in slow}
    if set(fast_map) != set(slow_map):
        raise ValueError("FAST/SLOW counterfactual keys are not aligned")
    counts: Counter[str] = Counter()
    for key in fast_map:
        if fast_map[key].get("gold") != slow_map[key].get("gold"):
            raise ValueError(f"FAST/SLOW gold mismatch for {key}")
        fast_correct = fast_map[key].get("prediction") == fast_map[key].get("gold")
        slow_correct = slow_map[key].get("prediction") == slow_map[key].get("gold")
        counts[f"{int(fast_correct)}{int(slow_correct)}"] += 1
    total = max(1, len(fast_map))
    return {
        "records": len(fast_map),
        "state_counts": {state: int(counts.get(state, 0)) for state in ("00", "01", "10", "11")},
        "benefit_count": int(counts.get("01", 0)),
        "benefit_rate": counts.get("01", 0) / total,
        "harm_count": int(counts.get("10", 0)),
        "harm_rate": counts.get("10", 0) / total,
    }


def mix_records(
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
    selected_slow_keys: set[str],
    policy: str,
) -> list[dict[str, Any]]:
    fast_map = {str(row["key"]): row for row in fast}
    slow_map = {str(row["key"]): row for row in slow}
    if set(fast_map) != set(slow_map):
        raise ValueError("Cannot route unaligned FAST/SLOW records")
    if selected_slow_keys - set(fast_map):
        raise ValueError("SLOW selection contains unknown keys")
    result = []
    for key in sorted(fast_map):
        selected = key in selected_slow_keys
        source = slow_map[key] if selected else fast_map[key]
        result.append(
            {
                **source,
                "experiment": EXPERIMENT,
                "version": VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "mode": "adaptive",
                "policy": policy,
                "routing_strategy": "slow" if selected else "fast",
                "test_data_read": False,
            }
        )
    return result


def matched_random_records(
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
    slow_count: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    keys = sorted(str(row["key"]) for row in fast)
    ranked = sorted(keys, key=lambda key: (_stable_hash(key, random_seed), key))
    return mix_records(fast, slow, set(ranked[:slow_count]), "matched_random")


def _benefit_truth(
    fast: Sequence[dict[str, Any]], slow: Sequence[dict[str, Any]]
) -> dict[str, bool]:
    slow_map = {str(row["key"]): row for row in slow}
    result: dict[str, bool] = {}
    for row in fast:
        key = str(row["key"])
        other = slow_map[key]
        result[key] = bool(
            row.get("prediction") != row.get("gold")
            and other.get("prediction") == other.get("gold")
        )
    return result


def selection_audit(
    selected_keys: set[str],
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    fast_map = {str(row["key"]): row for row in fast}
    slow_map = {str(row["key"]): row for row in slow}
    benefit = 0
    harm = 0
    for key in selected_keys:
        fast_correct = fast_map[key].get("prediction") == fast_map[key].get("gold")
        slow_correct = slow_map[key].get("prediction") == slow_map[key].get("gold")
        benefit += int(not fast_correct and slow_correct)
        harm += int(fast_correct and not slow_correct)
    count = len(selected_keys)
    return {
        "selected": count,
        "selected_rate": count / max(1, len(fast_map)),
        "benefit_selected": benefit,
        "harm_selected": harm,
        "benefit_precision": benefit / max(1, count),
        "harm_rate_among_selected": harm / max(1, count),
        "net_correctness_gain": benefit - harm,
    }


def _generate_fixed_slow(
    engine: FullContextGenerationEngine,
    targets: Sequence[FullContextTarget],
    labels: Sequence[str],
    output_path: Path,
    *,
    seed: int,
    batch_size: int,
    max_new_tokens: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    prefix = "<SLOW>\n"
    if overwrite and output_path.exists():
        output_path.unlink()
    cached = _read_jsonl(output_path)
    by_key: dict[str, dict[str, Any]] = {}
    for row in cached:
        if row.get("experiment") != EXPERIMENT or row.get("version") != VERSION:
            raise ValueError(f"Incompatible fixed-SLOW cache in {output_path}")
        if int(row.get("seed", -1)) != seed:
            raise ValueError(f"Fixed-SLOW seed mismatch in {output_path}")
        by_key[str(row["key"])] = row
    target_keys = {target.key for target in targets}
    if set(by_key) - target_keys:
        raise ValueError(f"Fixed-SLOW cache contains unexpected keys in {output_path}")
    missing = [target for target in targets if target.key not in by_key]
    log_every = max(1, int(engine.section.get("log_every_batches", 25)))
    prefix_tokens = len(engine.tokenizer.encode(prefix, add_special_tokens=False))
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        prompts = [
            build_adaptive_sft_prompt(
                target,
                labels,
                int(engine.section.get("slow_reasoning_word_limit", 80)),
            )
            for target in batch
        ]
        rendered = [
            _render_prompt(
                engine.tokenizer,
                prompt,
                enable_thinking=bool(engine.section.get("enable_thinking", False)),
            )
            + prefix
            for prompt in prompts
        ]
        generated_rows, timing = engine.generate_batch(rendered, max_new_tokens)
        new_rows = []
        for target, rendered_prompt, generated in zip(batch, rendered, generated_rows):
            response = prefix + str(generated["response"])
            audit = audit_response_semantics(response, labels, "slow")
            emotion_prefix = response_prefix_through_emotion(response, labels)
            completion_tokens = int(generated["completion_tokens"]) + prefix_tokens
            dialogue = render_causal_dialogue(target)
            row = {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "split": "validation",
                "source": AUDITED_VARIANT,
                "policy": "fixed_slow",
                "mode": "slow",
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
                "tokens_to_emotion": (
                    len(engine.tokenizer.encode(emotion_prefix, add_special_tokens=False))
                    if emotion_prefix is not None
                    else completion_tokens
                ),
                **audit,
                **timing,
                "test_data_read": False,
            }
            by_key[target.key] = row
            new_rows.append(row)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
        batch_number = start // batch_size + 1
        total_batches = max(1, math.ceil(len(missing) / batch_size))
        if batch_number == 1 or batch_number == total_batches or batch_number % log_every == 0:
            print(
                f"[Stage8.2R] fixed_slow seed={seed} "
                f"completed={min(start + len(batch), len(missing))}/{len(missing)} "
                f"cached={len(cached)}",
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
        raise ValueError("Stage 8.2R currently accepts MELD only")
    if section.get("context_policy") != "full_causal_dialogue":
        raise ValueError("context_policy must be exactly full_causal_dialogue")
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    requested = [int(value) for value in (seeds or section.get("seeds", [42, 43, 44]))]
    processed = _resolve(root, section["processed_dir"])
    validation = processed / "validation.jsonl"
    calibration = _calibration_path(root, section)
    policy_train = _policy_train_path(root, section)
    required = [validation, calibration, policy_train]
    adapter_hashes: dict[str, str] = {}
    prediction_hashes: dict[str, str] = {}
    for seed in requested:
        for variant in (AUDITED_VARIANT, CONTROL_VARIANT):
            adapter_config = _adapter_dir(root, section, seed, variant) / "adapter_config.json"
            required.append(adapter_config)
            if adapter_config.exists():
                adapter_hashes[f"seed{seed}:{variant}"] = _sha256_file(adapter_config)
        for variant, policy in (
            (AUDITED_VARIANT, "adaptive_prompt_forced_fast"),
            (CONTROL_VARIANT, "adaptive"),
        ):
            prediction = _stage8_2_prediction_path(
                root, section, seed, variant, policy
            )
            required.append(prediction)
            if prediction.exists():
                prediction_hashes[f"seed{seed}:{variant}:{policy}"] = _sha256_file(
                    prediction
                )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Stage 8.2R prerequisites:\n" + "\n".join(missing)
        )
    if any("test" in path.name.lower() for path in required):
        raise RuntimeError("Stage 8.2R preflight refuses every test-named input")
    calibration_rows = _read_jsonl(calibration)
    validation_targets = _load_validation_targets(root, section)
    actual_calibration_hash = _sha256_file(calibration)
    actual_policy_train_hash = _sha256_file(policy_train)
    actual_validation_hash = _sha256_file(validation)
    expected_hashes = {
        "policy_calibration": section.get("expected_policy_calibration_sha256"),
        "policy_train": section.get("expected_policy_train_sha256"),
        "validation": section.get("expected_validation_sha256"),
    }
    actual_hashes = {
        "policy_calibration": actual_calibration_hash,
        "policy_train": actual_policy_train_hash,
        "validation": actual_validation_hash,
    }
    for name, expected in expected_hashes.items():
        if expected is not None and str(expected) != actual_hashes[name]:
            raise ValueError(
                f"Frozen {name} SHA-256 mismatch: actual={actual_hashes[name]} "
                f"expected={expected}"
            )
    manifest = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MELD",
        "calibration_source": "train_dialogue_disjoint_policy_calibration",
        "evaluation_split": "validation",
        "context_policy": "full_causal_dialogue",
        "context_gate_used": False,
        "context_truncation_used": False,
        "validation_used_for_threshold": False,
        "validation_used_for_training": False,
        "test_locked": True,
        "seeds": requested,
        "calibration_records": len(calibration_rows),
        "validation_targets": len(validation_targets),
        "adapter_config_sha256": adapter_hashes,
        "stage8_2_prediction_sha256": prediction_hashes,
        "policy_calibration_sha256": actual_calibration_hash,
        "policy_train_sha256": actual_policy_train_hash,
        "validation_sha256": actual_validation_hash,
        "config_sha256": _sha256_file(config_path),
    }
    _write_json(paths["manifest"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def diagnose_seed(
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
    labels = [str(value) for value in section.get("labels", LABELS)]
    calibration_rows = _read_jsonl(_calibration_path(root, section))
    if not calibration_rows:
        raise FileNotFoundError("Missing Stage 8.2 policy calibration records")
    calibration_items = [
        {
            "key": row["key"],
            "dialogue_id": row["dialogue_id"],
            "gold": row["gold"],
            "policy_target": row["policy_target"],
            "user_prompt": row["user_prompt"],
        }
        for row in calibration_rows
    ]
    targets = _load_validation_targets(root, section)
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    validation_items = [
        {
            "key": target.key,
            "dialogue_id": target.dialogue_id,
            "gold": target.gold,
            "policy_target": None,
            "user_prompt": build_adaptive_sft_prompt(
                target,
                labels,
                int(section.get("slow_reasoning_word_limit", 80)),
            ),
        }
        for target in targets
    ]
    engine = FullContextGenerationEngine(
        section, _adapter_dir(root, section, seed, AUDITED_VARIANT)
    )
    weight_audit = normalized_strategy_weight_audit(
        _read_jsonl(_policy_train_path(root, section)),
        engine.tokenizer,
        float(section.get("stage8_2_strategy_token_weight", 8.0)),
    )
    score_batch_size = int(section.get("score_batch_size", 4))
    calibration_scores, tag_spec = _score_items(
        engine,
        calibration_items,
        _score_path(paths, seed, "train_calibration"),
        seed=seed,
        split="train_calibration",
        batch_size=score_batch_size,
        overwrite=overwrite,
    )
    calibration_truth = {
        str(row["key"]): str(row["policy_target"]).lower() == "slow"
        for row in calibration_rows
    }
    calibration_ranking = ranking_metrics(calibration_scores, calibration_truth)
    threshold = calibration_threshold(calibration_scores, calibration_truth)
    validation_scores, validation_tag_spec = _score_items(
        engine,
        validation_items,
        _score_path(paths, seed, "validation"),
        seed=seed,
        split="validation",
        batch_size=score_batch_size,
        overwrite=overwrite,
    )
    if tag_spec != validation_tag_spec:
        raise AssertionError("Strategy tokenizer specification changed within one run")
    fixed_slow = _generate_fixed_slow(
        engine,
        targets,
        labels,
        _fixed_slow_path(paths, seed),
        seed=seed,
        batch_size=int(section.get("generation_batch_size", 4)),
        max_new_tokens=int(section.get("max_new_tokens", {}).get("slow", 160)),
        overwrite=overwrite,
    )
    del engine
    gc.collect()
    torch.cuda.empty_cache()

    fixed_fast = _read_jsonl(
        _stage8_2_prediction_path(
            root,
            section,
            seed,
            AUDITED_VARIANT,
            "adaptive_prompt_forced_fast",
        )
    )
    always_fast = _read_jsonl(
        _stage8_2_prediction_path(
            root, section, seed, CONTROL_VARIANT, "adaptive"
        )
    )
    selected_keys = {
        str(row["key"])
        for row in validation_scores
        if float(row["strategy_margin"]) >= float(threshold["threshold"])
    }
    expected_keys = {target.key for target in targets}
    fixed_fast = [row for row in fixed_fast if str(row["key"]) in expected_keys]
    always_fast = [row for row in always_fast if str(row["key"]) in expected_keys]
    if not fixed_fast or len(fixed_fast) != len(targets) or len(always_fast) != len(targets):
        raise ValueError(
            "Stage 8.2 fixed-FAST controls do not match the requested validation workload"
        )
    routed = mix_records(fixed_fast, fixed_slow, selected_keys, "calibrated_margin")
    matched = matched_random_records(
        fixed_fast,
        fixed_slow,
        len(selected_keys),
        int(section.get("matched_random_seed", 8242)) + seed,
    )
    _write_jsonl(_routed_path(paths, seed, "calibrated_margin"), routed)
    _write_jsonl(_routed_path(paths, seed, "matched_random"), matched)
    states = counterfactual_states(fixed_fast, fixed_slow)
    benefit_truth = _benefit_truth(fixed_fast, fixed_slow)
    validation_ranking = ranking_metrics(validation_scores, benefit_truth)
    selection = selection_audit(selected_keys, fixed_fast, fixed_slow)
    oracle = oracle_budget_frontier(
        fixed_fast,
        fixed_slow,
        labels,
        [1.0],
        int(section.get("oracle_tie_seed", 8243)) + seed,
    )[0]
    methods = {
        "adaptive_sft_fixed_fast": fixed_fast,
        "adaptive_sft_fixed_slow": fixed_slow,
        "adaptive_sft_calibrated": routed,
        "matched_random": matched,
        "always_fast_sft": always_fast,
    }
    metrics = {
        name: classification_and_cost_metrics(rows, labels)
        for name, rows in methods.items()
    }
    bootstraps = {
        "calibrated_minus_same_model_fixed_fast": paired_dialogue_bootstrap(
            fixed_fast,
            routed,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 8244)) + seed,
        ),
        "calibrated_minus_matched_random": paired_dialogue_bootstrap(
            matched,
            routed,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 8244)) + 100 + seed,
        ),
        "calibrated_minus_always_fast_sft": paired_dialogue_bootstrap(
            always_fast,
            routed,
            labels,
            int(section.get("bootstrap_samples", 2000)),
            int(section.get("bootstrap_seed", 8244)) + 200 + seed,
        ),
    }
    result = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "split": "validation",
        "targets": len(targets),
        "threshold": threshold,
        "strategy_tag_tokenization": tag_spec,
        "stage8_2_normalized_strategy_weight_audit": weight_audit,
        "calibration_ranking": calibration_ranking,
        "post_sft_counterfactual_states": states,
        "validation_benefit_ranking": validation_ranking,
        "selection_audit": selection,
        "methods": metrics,
        "post_sft_oracle": oracle,
        "bootstrap": bootstraps,
        "validation_used_for_threshold": False,
        "validation_used_for_training": False,
        "test_locked": True,
    }
    _write_json(paths["reports"] / f"seed{seed}_diagnostic.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _mean_seed_metric(
    seed_reports: dict[int, dict[str, Any]], method: str, metric: str
) -> dict[str, Any]:
    return _aggregate(
        [
            float(seed_reports[seed]["methods"][method][metric])
            for seed in sorted(seed_reports)
        ]
    )


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    section = _section(_read_yaml(config_path))
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    seed_reports: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        path = paths["reports"] / f"seed{seed}_diagnostic.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing Stage 8.2R diagnostic: {path}")
        seed_reports[seed] = json.loads(path.read_text(encoding="utf-8"))
    weight_audit = seed_reports[seeds[0]][
        "stage8_2_normalized_strategy_weight_audit"
    ]
    for seed in seeds[1:]:
        if (
            seed_reports[seed]["stage8_2_normalized_strategy_weight_audit"]
            != weight_audit
        ):
            raise ValueError("Stage 8.2 loss-weight audit differs across seeds")
    methods = [
        "adaptive_sft_fixed_fast",
        "adaptive_sft_fixed_slow",
        "adaptive_sft_calibrated",
        "matched_random",
        "always_fast_sft",
    ]
    metric_names = (
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "mean_completion_tokens",
        "mean_tokens_to_emotion",
        "parse_success_rate",
        "strict_format_success_rate",
    )
    aggregates = {
        method: {
            metric: _mean_seed_metric(seed_reports, method, metric)
            for metric in metric_names
        }
        for method in methods
    }
    oracle = {
        metric: _aggregate(
            [float(seed_reports[seed]["post_sft_oracle"][metric]) for seed in seeds]
        )
        for metric in ("weighted_f1", "macro_f1", "accuracy", "actual_slow_rate", "mean_completion_tokens")
    }
    diagnostics = {
        "calibration_prevalence": _aggregate(
            [float(seed_reports[seed]["calibration_ranking"]["prevalence"]) for seed in seeds]
        ),
        "calibration_average_precision": _aggregate(
            [float(seed_reports[seed]["calibration_ranking"]["average_precision"]) for seed in seeds]
        ),
        "calibration_average_precision_lift": _aggregate(
            [float(seed_reports[seed]["calibration_ranking"]["average_precision_lift_over_prevalence"]) for seed in seeds]
        ),
        "validation_benefit_prevalence": _aggregate(
            [float(seed_reports[seed]["validation_benefit_ranking"]["prevalence"]) for seed in seeds]
        ),
        "validation_benefit_average_precision": _aggregate(
            [float(seed_reports[seed]["validation_benefit_ranking"]["average_precision"]) for seed in seeds]
        ),
        "validation_benefit_average_precision_lift": _aggregate(
            [float(seed_reports[seed]["validation_benefit_ranking"]["average_precision_lift_over_prevalence"]) for seed in seeds]
        ),
        "selected_slow_rate": _aggregate(
            [float(seed_reports[seed]["selection_audit"]["selected_rate"]) for seed in seeds]
        ),
        "selected_benefit_precision": _aggregate(
            [float(seed_reports[seed]["selection_audit"]["benefit_precision"]) for seed in seeds]
        ),
        "selected_harm_rate": _aggregate(
            [float(seed_reports[seed]["selection_audit"]["harm_rate_among_selected"]) for seed in seeds]
        ),
    }
    fixed_fast_wf1 = float(aggregates["adaptive_sft_fixed_fast"]["weighted_f1"]["mean"])
    calibrated_wf1 = float(aggregates["adaptive_sft_calibrated"]["weighted_f1"]["mean"])
    random_wf1 = float(aggregates["matched_random"]["weighted_f1"]["mean"])
    always_fast_wf1 = float(aggregates["always_fast_sft"]["weighted_f1"]["mean"])
    oracle_wf1 = float(oracle["weighted_f1"]["mean"])
    oracle_gain = oracle_wf1 - fixed_fast_wf1
    calibrated_gain = calibrated_wf1 - fixed_fast_wf1
    validation_ap_lift = float(
        diagnostics["validation_benefit_average_precision_lift"]["mean"]
    )
    calibration_ap_lift = float(
        diagnostics["calibration_average_precision_lift"]["mean"]
    )
    slow_rate = float(diagnostics["selected_slow_rate"]["mean"])
    checks = {
        "post_sft_oracle_exists": oracle_gain
        >= float(section.get("minimum_post_sft_oracle_gain", 0.003)),
        "calibration_labels_rankable": calibration_ap_lift
        >= float(section.get("minimum_calibration_ap_lift", 0.02)),
        "validation_benefit_rankable": validation_ap_lift
        >= float(section.get("minimum_validation_ap_lift", 0.02)),
        "calibrated_policy_beats_fixed_fast": calibrated_gain
        >= float(section.get("minimum_calibrated_gain", 0.001)),
        "calibrated_policy_beats_matched_random": calibrated_wf1
        >= random_wf1 + float(section.get("minimum_gain_over_matched_random", 0.001)),
        "calibrated_slow_rate": float(section.get("minimum_slow_rate", 0.03))
        <= slow_rate
        <= float(section.get("maximum_slow_rate", 0.12)),
    }
    if not checks["post_sft_oracle_exists"]:
        status_value = "stop_adaptive_route_adopt_always_fast_sft"
    elif not (
        checks["calibration_labels_rankable"]
        and checks["validation_benefit_rankable"]
    ):
        status_value = "revise_supervision_strategy_not_rankable"
    elif not (
        checks["calibrated_policy_beats_fixed_fast"]
        and checks["calibrated_policy_beats_matched_random"]
        and checks["calibrated_slow_rate"]
    ):
        status_value = "proceed_to_strategy_only_sft_rl_blocked"
    else:
        status_value = "adaptive_threshold_quality_policy_established_rl_still_blocked"
    decision = {
        "status": status_value,
        "adaptive_sft_fixed_fast_weighted_f1": fixed_fast_wf1,
        "adaptive_sft_calibrated_weighted_f1": calibrated_wf1,
        "matched_random_weighted_f1": random_wf1,
        "always_fast_sft_weighted_f1": always_fast_wf1,
        "post_sft_oracle_weighted_f1": oracle_wf1,
        "post_sft_oracle_gain": oracle_gain,
        "calibrated_gain_over_same_model_fixed_fast": calibrated_gain,
        "calibrated_gain_over_matched_random": calibrated_wf1 - random_wf1,
        "calibrated_minus_always_fast_sft": calibrated_wf1 - always_fast_wf1,
        "selected_slow_rate": slow_rate,
        "stage8_2_fast_to_slow_aggregate_tag_mass_ratio": float(
            weight_audit["fast_to_slow_aggregate_tag_mass_ratio"]
        ),
        "checks": checks,
        "rl_status": "not_permitted; Stage 8.2R is diagnostic and test remains locked",
        "runtime_status": (
            "not_claimed; a fused margin-and-generation implementation with KV-cache "
            "reuse must be profiled in a separately preregistered stage"
        ),
        "validation_used_for_threshold": False,
        "test_locked": True,
    }
    bootstrap = {
        f"seed{seed}": seed_reports[seed]["bootstrap"] for seed in seeds
    }
    final = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "dataset": "MELD",
        "split": "validation",
        "context_policy": "full_causal_dialogue",
        "seeds": seeds,
        "aggregates": aggregates,
        "post_sft_oracle": oracle,
        "strategy_diagnostics": diagnostics,
        "stage8_2_normalized_strategy_weight_audit": weight_audit,
        "decision": decision,
        "test_locked": True,
    }
    _write_json(paths["summary"], final)
    _write_json(paths["decision"], decision)
    _write_json(paths["bootstrap"], bootstrap)
    lines = [
        "# Stage 8.2R — Post-SFT Counterfactual & Strategy-Margin Audit",
        "",
        "> MELD validation-only. The SLOW threshold is fixed on the dialogue-disjoint "
        "MELD-train calibration partition. Validation is never used for training or "
        "threshold selection. Every method receives the same complete causal dialogue.",
        "",
        "| Method | wF1 | Macro F1 | Accuracy | Completion tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        values = aggregates[method]
        lines.append(
            f"| {method} | {values['weighted_f1']['mean']:.4f} ± "
            f"{values['weighted_f1']['std']:.4f} | "
            f"{values['macro_f1']['mean']:.4f} ± {values['macro_f1']['std']:.4f} | "
            f"{values['accuracy']['mean']:.4f} ± {values['accuracy']['std']:.4f} | "
            f"{values['mean_completion_tokens']['mean']:.2f} |"
        )
    lines.extend(
        [
            f"| post_sft_oracle | {oracle['weighted_f1']['mean']:.4f} ± "
            f"{oracle['weighted_f1']['std']:.4f} | "
            f"{oracle['macro_f1']['mean']:.4f} ± {oracle['macro_f1']['std']:.4f} | "
            f"{oracle['accuracy']['mean']:.4f} ± {oracle['accuracy']['std']:.4f} | "
            f"{oracle['mean_completion_tokens']['mean']:.2f} |",
            "",
            "## Frozen decision",
            "",
            f"- Status: `{status_value}`",
            f"- Post-SFT oracle gain over same-model FAST: {oracle_gain:+.4f}",
            f"- Calibrated policy gain over same-model FAST: {calibrated_gain:+.4f}",
            f"- Calibrated policy gain over matched random: {calibrated_wf1 - random_wf1:+.4f}",
            f"- Selected SLOW rate: {slow_rate:.2%}",
            "- Stage 8.2 normalized FAST:SLOW strategy-tag mass ratio: "
            f"{weight_audit['fast_to_slow_aggregate_tag_mass_ratio']:.3f}:1",
            f"- Calibration AP lift: {calibration_ap_lift:+.4f}",
            f"- Validation benefit AP lift: {validation_ap_lift:+.4f}",
            f"- RL status: `{decision['rl_status']}`",
            f"- Runtime status: `{decision['runtime_status']}`",
            "- MELD test remains locked.",
        ]
    )
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
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
        "seeds": {
            str(seed): {
                "calibration_scores": _score_path(paths, seed, "train_calibration").exists(),
                "validation_scores": _score_path(paths, seed, "validation").exists(),
                "fixed_slow": _fixed_slow_path(paths, seed).exists(),
                "calibrated_policy": _routed_path(paths, seed, "calibrated_margin").exists(),
                "diagnostic": (paths["reports"] / f"seed{seed}_diagnostic.json").exists(),
            }
            for seed in seeds
        },
        "final_report": paths["summary"].exists(),
        "test_locked": True,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def final_test_locked() -> None:
    raise RuntimeError(
        "Stage 8.2R is validation-only and cannot read MELD test. A separate "
        "preregistered patch would be required after the adaptive policy is established."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--seeds", nargs="*", type=int)
    diagnose_parser = commands.add_parser("diagnose")
    diagnose_parser.add_argument("--seed", type=int, required=True)
    diagnose_parser.add_argument("--max-targets", type=int)
    diagnose_parser.add_argument("--overwrite", action="store_true")
    commands.add_parser("report")
    commands.add_parser("status")
    commands.add_parser("final-test")
    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.config, args.run_name, args.seeds)
    elif args.command == "diagnose":
        diagnose_seed(
            args.config,
            args.seed,
            args.run_name,
            max_targets=args.max_targets,
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
