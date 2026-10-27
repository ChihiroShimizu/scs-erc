from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from pfa_erc.data import Dialogue, Utterance, read_jsonl


STAGE1_VERSION = "1.0.0"


@dataclass(frozen=True)
class TargetSpec:
    dialogue_id: str
    turn_index: int
    utterance_id: str
    speaker: str
    text: str
    gold: str
    history: tuple[Utterance, ...]

    @property
    def key(self) -> str:
        return f"{self.dialogue_id}::{self.turn_index}::{self.utterance_id}"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _append_jsonl(path: Path, payloads: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}") from error
    return records


def build_prompt(
    target: TargetSpec,
    labels: Sequence[str],
    history: Sequence[Utterance],
    prompt_preamble: str,
) -> str:
    """Build a causal prompt; only turns before the target are accepted as history."""
    history_lines = [f"[{i + 1}] {turn.speaker}: {turn.text}" for i, turn in enumerate(history)]
    history_block = "\n".join(history_lines) if history_lines else "[no prior dialogue provided]"
    label_text = ", ".join(labels)
    return (
        f"{prompt_preamble.strip()}\n\n"
        f"Dialogue history (earlier turns only):\n{history_block}\n\n"
        f"Target utterance:\n{target.speaker}: {target.text}\n\n"
        "Classify the target speaker's emotion. Use only the information shown. "
        f"Return exactly one label from: {label_text}.\n"
        "Emotion:"
    )


def make_target_specs(
    dialogue: Dialogue,
    max_history_turns: int,
    include_first_turn: bool = True,
) -> list[TargetSpec]:
    specs: list[TargetSpec] = []
    for index, turn in enumerate(dialogue.utterances):
        if index == 0 and not include_first_turn:
            continue
        start = max(0, index - max_history_turns)
        specs.append(
            TargetSpec(
                dialogue_id=dialogue.dialogue_id,
                turn_index=index,
                utterance_id=turn.utterance_id,
                speaker=turn.speaker,
                text=turn.text,
                gold=turn.emotion,
                history=tuple(dialogue.utterances[start:index]),
            )
        )
    return specs


def classify_utility(
    delta_nll: float,
    current_prediction_correct: bool,
    epsilon: float,
) -> str:
    """Assign a diagnostic group without using an LLM-generated pseudo-label."""
    if delta_nll > epsilon:
        return "helpful_context"
    if delta_nll < -epsilon:
        return "harmful_context"
    return "no_context_needed" if current_prediction_correct else "uncertain"


def _softmax(scores: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = np.asarray(scores, dtype=np.float64) / temperature
    values -= np.max(values)
    exp = np.exp(values)
    return exp / exp.sum()


def _prediction_payload(scores: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    probabilities = _softmax(scores)
    prediction_index = int(np.argmax(probabilities))
    return {
        "scores": [float(value) for value in scores],
        "probabilities": [float(value) for value in probabilities],
        "prediction": labels[prediction_index],
        "confidence": float(probabilities[prediction_index]),
    }


def _nll(probabilities: Sequence[float], gold_index: int) -> float:
    return float(-math.log(max(float(probabilities[gold_index]), 1e-12)))


class CausalLabelScorer:
    """Score fixed label continuations with a causal language model.

    A one-token fast path is used when every label has exactly one continuation token.
    Otherwise, full label sequences are scored and reduced by mean token log-probability.
    """

    def __init__(self, config: dict[str, Any], labels: Sequence[str]) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("transformers is required; run python3 -m pip install -e .") from error

        self.labels = list(labels)
        self.model_name = str(config["model_name"])
        self.max_input_tokens = int(config.get("max_input_tokens", 1024))
        self.batch_size = int(config.get("batch_size", 2))
        self.score_reduction = str(config.get("score_reduction", "mean"))
        if self.score_reduction not in {"mean", "sum"}:
            raise ValueError("score_reduction must be 'mean' or 'sum'")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = str(config.get("dtype", "bfloat16"))
        if self.device.type == "cpu":
            dtype = torch.float32
        elif dtype_name == "float16":
            dtype = torch.float16
        elif dtype_name == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        trust_remote_code = bool(config.get("trust_remote_code", False))
        local_files_only = bool(config.get("local_files_only", False))
        print(f"Loading tokenizer: {self.model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = "left"

        print(f"Loading causal LM on {self.device}: {self.model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.continuations = [f" {label}" for label in self.labels]
        self.label_token_ids = [
            self.tokenizer.encode(text, add_special_tokens=False) for text in self.continuations
        ]
        if any(not ids for ids in self.label_token_ids):
            raise ValueError("At least one emotion label has an empty tokenization")
        self.fast_single_token = all(len(ids) == 1 for ids in self.label_token_ids)
        token_lengths = {label: len(ids) for label, ids in zip(self.labels, self.label_token_ids)}
        print(
            f"Label token lengths: {token_lengths}; fast_single_token={self.fast_single_token}",
            flush=True,
        )
        if bool(config.get("require_single_token_labels", False)) and not self.fast_single_token:
            raise ValueError(
                "This tokenizer does not encode every label as one token. "
                "Set require_single_token_labels=false to use the documented multi-token fallback."
            )

    @property
    def audit(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_input_tokens": self.max_input_tokens,
            "batch_size": self.batch_size,
            "score_reduction": self.score_reduction,
            "label_continuations": dict(zip(self.labels, self.continuations)),
            "label_token_lengths": {
                label: len(ids) for label, ids in zip(self.labels, self.label_token_ids)
            },
            "fast_single_token": self.fast_single_token,
        }

    def score_prompts(self, prompts: Sequence[str]) -> list[list[float]]:
        if not prompts:
            return []
        if self.fast_single_token:
            return self._score_single_token(prompts)
        return self._score_multi_token(prompts)

    def _score_single_token(self, prompts: Sequence[str]) -> list[list[float]]:
        results: list[list[float]] = []
        label_ids = torch.tensor(
            [ids[0] for ids in self.label_token_ids], dtype=torch.long, device=self.device
        )
        for start in range(0, len(prompts), self.batch_size):
            batch_prompts = list(prompts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            last_indices = encoded["attention_mask"].sum(dim=1) - 1
            with torch.inference_mode():
                logits = self.model(**encoded).logits
            row_indices = torch.arange(logits.shape[0], device=self.device)
            next_logits = logits[row_indices, last_indices]
            selected = next_logits.index_select(dim=-1, index=label_ids).float().cpu().numpy()
            results.extend([[float(value) for value in row] for row in selected])
            del logits, next_logits, encoded
        return results

    def _score_multi_token(self, prompts: Sequence[str]) -> list[list[float]]:
        pairs: list[tuple[int, int, list[int], int]] = []
        reserve = max(len(ids) for ids in self.label_token_ids)
        for prompt_index, prompt in enumerate(prompts):
            prompt_ids = self.tokenizer.encode(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=max(2, self.max_input_tokens - reserve),
            )
            for label_index, label_ids in enumerate(self.label_token_ids):
                pairs.append((prompt_index, label_index, prompt_ids + label_ids, len(prompt_ids)))

        output = np.zeros((len(prompts), len(self.labels)), dtype=np.float64)
        pad_id = int(self.tokenizer.pad_token_id)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            max_length = max(len(item[2]) for item in batch)
            input_ids = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
            for row, (_, _, ids, _) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = 1
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
            for row, (prompt_index, label_index, ids, prompt_length) in enumerate(batch):
                token_scores: list[torch.Tensor] = []
                for position in range(prompt_length, len(ids)):
                    token_scores.append(log_probs[row, position - 1, ids[position]])
                stacked = torch.stack(token_scores)
                score = stacked.mean() if self.score_reduction == "mean" else stacked.sum()
                output[prompt_index, label_index] = float(score.cpu())
            del logits, log_probs, input_ids, attention_mask
        return output.tolist()


def _record_for_target(
    target: TargetSpec,
    labels: Sequence[str],
    scores: Sequence[Sequence[float]],
    utility_epsilon: float,
    deletion_epsilon: float,
) -> dict[str, Any]:
    if len(scores) != 2 + len(target.history):
        raise ValueError("Expected current, full, and one deletion score per history turn")
    label_to_id = {label: index for index, label in enumerate(labels)}
    if target.gold not in label_to_id:
        raise ValueError(f"Unknown gold label {target.gold!r}")
    gold_index = label_to_id[target.gold]

    current = _prediction_payload(scores[0], labels)
    full = _prediction_payload(scores[1], labels)
    current["nll"] = _nll(current["probabilities"], gold_index)
    full["nll"] = _nll(full["probabilities"], gold_index)
    delta_nll = float(current["nll"] - full["nll"])

    deletions: list[dict[str, Any]] = []
    for history_index, (source, source_scores) in enumerate(zip(target.history, scores[2:])):
        deleted = _prediction_payload(source_scores, labels)
        deleted_nll = _nll(deleted["probabilities"], gold_index)
        source_delta = float(deleted_nll - full["nll"])
        gap = len(target.history) - history_index
        deletions.append(
            {
                "history_index": history_index,
                "source_utterance_id": source.utterance_id,
                "source_speaker": source.speaker,
                "same_speaker": source.speaker == target.speaker,
                "turn_gap": gap,
                "delta_nll": source_delta,
                "effect": (
                    "helpful_turn"
                    if source_delta > deletion_epsilon
                    else "harmful_turn"
                    if source_delta < -deletion_epsilon
                    else "negligible_turn"
                ),
                "deleted_prediction": deleted["prediction"],
                "deleted_nll": deleted_nll,
            }
        )

    return {
        "key": target.key,
        "dialogue_id": target.dialogue_id,
        "turn_index": target.turn_index,
        "utterance_id": target.utterance_id,
        "speaker": target.speaker,
        "gold": target.gold,
        "history_count": len(target.history),
        "current": current,
        "full": full,
        "context_utility_delta_nll": delta_nll,
        "utility_class": classify_utility(
            delta_nll,
            current_prediction_correct=current["prediction"] == target.gold,
            epsilon=utility_epsilon,
        ),
        "deletions": deletions,
    }


def _metric_bundle(
    records: Sequence[dict[str, Any]], branch: str, labels: Sequence[str]
) -> dict[str, Any]:
    label_to_id = {label: index for index, label in enumerate(labels)}
    gold = [label_to_id[record["gold"]] for record in records]
    predicted = [label_to_id[record[branch]["prediction"]] for record in records]
    probabilities = np.asarray([record[branch]["probabilities"] for record in records])
    one_hot = np.eye(len(labels), dtype=np.float64)[gold]
    nll_values = [-math.log(max(probabilities[i, target], 1e-12)) for i, target in enumerate(gold)]
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predicted) == np.asarray(gold)
    ordered_ids = list(range(len(labels)))
    per_emotion = f1_score(
        gold, predicted, labels=ordered_ids, average=None, zero_division=0
    )
    return {
        "weighted_f1": float(
            f1_score(
                gold,
                predicted,
                labels=ordered_ids,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                gold,
                predicted,
                labels=ordered_ids,
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(gold, predicted)),
        "nll": float(np.mean(nll_values)),
        "brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "ece": _ece(confidence, correct, bins=15),
        "per_emotion_f1": {
            label: float(value) for label, value in zip(labels, per_emotion)
        },
    }


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int) -> float:
    total = len(confidence)
    result = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins + 1)[:-1], np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if not np.any(selected):
            continue
        result += float(selected.mean()) * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return result if total else float("nan")


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std(ddof=0)) if len(array) else None,
        "count": int(len(array)),
    }


def _percentile_ci(values: Sequence[float]) -> list[float | None]:
    if not values:
        return [None, None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _cluster_bootstrap(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_dialogue[record["dialogue_id"]].append(record)
    ids = sorted(by_dialogue)
    if not ids or samples <= 0:
        return {}
    rng = random.Random(seed)
    utility_values: list[float] = []
    weighted_values: list[float] = []
    macro_values: list[float] = []
    nll_values: list[float] = []
    for _ in range(samples):
        sampled: list[dict[str, Any]] = []
        for dialogue_id in (rng.choice(ids) for _ in ids):
            sampled.extend(by_dialogue[dialogue_id])
        eligible = [record for record in sampled if record["history_count"] > 0]
        utility_values.append(float(np.mean([record["context_utility_delta_nll"] for record in eligible])))
        current = _metric_bundle(sampled, "current", labels)
        full = _metric_bundle(sampled, "full", labels)
        weighted_values.append(full["weighted_f1"] - current["weighted_f1"])
        macro_values.append(full["macro_f1"] - current["macro_f1"])
        nll_values.append(current["nll"] - full["nll"])
    return {
        "samples": samples,
        "seed": seed,
        "dialogues": len(ids),
        "mean_context_utility": {
            "ci95": _percentile_ci(utility_values),
            "probability_positive": float(np.mean(np.asarray(utility_values) > 0)),
        },
        "full_minus_current_weighted_f1": {
            "ci95": _percentile_ci(weighted_values),
            "probability_positive": float(np.mean(np.asarray(weighted_values) > 0)),
        },
        "full_minus_current_macro_f1": {
            "ci95": _percentile_ci(macro_values),
            "probability_positive": float(np.mean(np.asarray(macro_values) > 0)),
        },
        "current_minus_full_nll": {
            "ci95": _percentile_ci(nll_values),
            "probability_positive": float(np.mean(np.asarray(nll_values) > 0)),
        },
    }


def _deletion_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    items = [
        {
            **item,
            "target_key": record["key"],
            "target_emotion": record["gold"],
        }
        for record in records
        for item in record.get("deletions", [])
    ]
    if not items:
        return {"count": 0}

    def summarize(group: Sequence[dict[str, Any]]) -> dict[str, Any]:
        values = [float(item["delta_nll"]) for item in group]
        return {
            **_mean_std(values),
            "median": float(np.median(values)) if values else None,
            "positive_rate": float(np.mean(np.asarray(values) > 0)) if values else None,
            "negative_rate": float(np.mean(np.asarray(values) < 0)) if values else None,
        }

    by_gap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        gap = int(item["turn_gap"])
        bucket = "1" if gap == 1 else "2-3" if gap <= 3 else "4-7" if gap <= 7 else "8+"
        by_gap[bucket].append(item)
    helpful = [item for item in items if item["effect"] == "helpful_turn"]
    harmful = [item for item in items if item["effect"] == "harmful_turn"]
    negligible = [item for item in items if item["effect"] == "negligible_turn"]
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_target[item["target_key"]].append(item)
    oracle_top1 = [max(group, key=lambda item: item["delta_nll"])["delta_nll"] for group in by_target.values()]
    random_expectation = [
        float(np.mean([item["delta_nll"] for item in group])) for group in by_target.values()
    ]
    by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_emotion[item["target_emotion"]].append(item)
    return {
        "count": len(items),
        "overall": summarize(items),
        "same_speaker": summarize([item for item in items if item["same_speaker"]]),
        "different_speaker": summarize([item for item in items if not item["same_speaker"]]),
        "by_turn_gap": {key: summarize(by_gap.get(key, [])) for key in ("1", "2-3", "4-7", "8+")},
        "by_target_emotion": {
            emotion: summarize(group) for emotion, group in sorted(by_emotion.items())
        },
        "effect_counts": {
            "helpful_turn": len(helpful),
            "harmful_turn": len(harmful),
            "negligible_turn": len(negligible),
            "helpful_rate": len(helpful) / len(items),
            "harmful_rate": len(harmful) / len(items),
            "negligible_rate": len(negligible) / len(items),
        },
        "heterogeneity": {
            "q10": float(np.quantile([item["delta_nll"] for item in items], 0.10)),
            "q50": float(np.quantile([item["delta_nll"] for item in items], 0.50)),
            "q90": float(np.quantile([item["delta_nll"] for item in items], 0.90)),
        },
        "oracle_selection_diagnostic": {
            "targets": len(by_target),
            "oracle_top1_mean_delta_nll": float(np.mean(oracle_top1)),
            "uniform_random_expected_mean_delta_nll": float(np.mean(random_expectation)),
            "oracle_top1_minus_random": float(
                np.mean(np.asarray(oracle_top1) - np.asarray(random_expectation))
            ),
            "warning": (
                "Top-1 is selected using observed deletion effects and is therefore an oracle "
                "upper bound, not a trained selector result."
            ),
        },
    }


def summarize_records(
    records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    utility_epsilon: float,
    deletion_epsilon: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if not records:
        raise ValueError("No records available to summarize")
    eligible = [record for record in records if record["history_count"] > 0]
    utility = [float(record["context_utility_delta_nll"]) for record in eligible]
    groups: dict[str, int] = defaultdict(int)
    for record in eligible:
        groups[record["utility_class"]] += 1
    current = _metric_bundle(records, "current", labels)
    full = _metric_bundle(records, "full", labels)
    utility_by_emotion: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        utility_by_emotion[record["gold"]].append(record)
    deletion = _deletion_summary(records)
    q10 = deletion.get("heterogeneity", {}).get("q10")
    q90 = deletion.get("heterogeneity", {}).get("q90")
    effect_counts = deletion.get("effect_counts", {})
    context_heterogeneous = (
        groups.get("helpful_context", 0) / max(len(eligible), 1) >= 0.05
        and groups.get("harmful_context", 0) / max(len(eligible), 1) >= 0.05
    )
    sparse_turn_signal = (
        q10 is not None
        and q90 is not None
        and q90 - q10 >= 2 * deletion_epsilon
        and effect_counts.get("helpful_rate", 0.0) >= 0.05
    )
    return {
        "experiment": "LLM_context_utility_stage1",
        "version": STAGE1_VERSION,
        "split": "validation",
        "labels": list(labels),
        "targets": len(records),
        "history_eligible_targets": len(eligible),
        "dialogues": len({record["dialogue_id"] for record in records}),
        "thresholds": {
            "utility_epsilon": utility_epsilon,
            "deletion_epsilon": deletion_epsilon,
        },
        "current_only": current,
        "full_causal_history": full,
        "paired_advantage_full_over_current": {
            "weighted_f1": full["weighted_f1"] - current["weighted_f1"],
            "macro_f1": full["macro_f1"] - current["macro_f1"],
            "accuracy": full["accuracy"] - current["accuracy"],
            "nll": current["nll"] - full["nll"],
            "brier": current["brier"] - full["brier"],
            "ece": current["ece"] - full["ece"],
        },
        "target_context_utility": {
            **_mean_std(utility),
            "median": float(np.median(utility)) if utility else None,
            "group_counts": dict(groups),
            "group_rates": {key: value / max(len(eligible), 1) for key, value in groups.items()},
            "by_emotion": {
                emotion: {
                    **_mean_std(
                        [float(record["context_utility_delta_nll"]) for record in group]
                    ),
                    "helpful_rate": float(
                        np.mean([record["utility_class"] == "helpful_context" for record in group])
                    ),
                    "harmful_rate": float(
                        np.mean([record["utility_class"] == "harmful_context" for record in group])
                    ),
                }
                for emotion, group in sorted(utility_by_emotion.items())
            },
        },
        "turn_deletion": deletion,
        "cluster_bootstrap": _cluster_bootstrap(
            records, labels, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "stage1_decision": {
            "context_effect_is_heterogeneous": context_heterogeneous,
            "individual_turn_effect_is_sparse": sparse_turn_signal,
            "proceed_to_selective_memory_stage": context_heterogeneous and sparse_turn_signal,
            "note": (
                "This operational gate tests whether selective read/write memory is motivated. "
                "It is not a test-set claim and must not be tuned on MELD test."
            ),
        },
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    current = summary["current_only"]
    full = summary["full_causal_history"]
    advantage = summary["paired_advantage_full_over_current"]
    utility = summary["target_context_utility"]
    deletion = summary["turn_deletion"]
    lines = [
        "# Stage 1 — LLM Context Utility Diagnostic",
        "",
        "> Validation-only diagnostic. The MELD test split remains locked.",
        "",
        f"Targets: **{summary['targets']}**; history-eligible targets: **{summary['history_eligible_targets']}**; dialogues: **{summary['dialogues']}**.",
        "",
        "## Current-only vs full causal history",
        "",
        "| Input | weighted F1 | macro F1 | accuracy | NLL | Brier | ECE |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Current only | {current['weighted_f1']:.4f} | {current['macro_f1']:.4f} | {current['accuracy']:.4f} | {current['nll']:.4f} | {current['brier']:.4f} | {current['ece']:.4f} |",
        f"| Full causal history | {full['weighted_f1']:.4f} | {full['macro_f1']:.4f} | {full['accuracy']:.4f} | {full['nll']:.4f} | {full['brier']:.4f} | {full['ece']:.4f} |",
        f"| Full advantage | {advantage['weighted_f1']:+.4f} | {advantage['macro_f1']:+.4f} | {advantage['accuracy']:+.4f} | {advantage['nll']:+.4f} | {advantage['brier']:+.4f} | {advantage['ece']:+.4f} |",
        "",
        "For NLL/Brier/ECE, a positive advantage means full history reduced the error.",
        "",
        "## Target-level context utility",
        "",
        f"Mean utility `NLL(current)-NLL(history)` = **{utility['mean']:.4f}** (median {utility['median']:.4f}).",
        "",
        "| Group | Count | Rate |",
        "|---|---:|---:|",
    ]
    for key in ("helpful_context", "harmful_context", "no_context_needed", "uncertain"):
        count = utility["group_counts"].get(key, 0)
        rate = utility["group_rates"].get(key, 0.0)
        lines.append(f"| {key} | {count} | {rate:.1%} |")
    lines.extend(
        [
            "",
            "## Leave-one-turn-out intervention",
            "",
            f"Counterfactual deletions: **{deletion.get('count', 0)}**.",
        ]
    )
    if deletion.get("count", 0):
        effects = deletion["effect_counts"]
        heterogeneity = deletion["heterogeneity"]
        lines.extend(
            [
                f"Helpful / harmful / negligible turn rates: **{effects['helpful_rate']:.1%} / {effects['harmful_rate']:.1%} / {effects['negligible_rate']:.1%}**.",
                f"Deletion effect quantiles (q10/q50/q90): **{heterogeneity['q10']:.4f} / {heterogeneity['q50']:.4f} / {heterogeneity['q90']:.4f}**.",
                "",
                "Positive deletion delta means the removed historical turn had helped the gold emotion; negative means it had harmed it.",
            ]
        )
    decision = summary["stage1_decision"]
    lines.extend(
        [
            "",
            "## Pre-registered decision gate",
            "",
            f"- Heterogeneous context effect: **{decision['context_effect_is_heterogeneous']}**",
            f"- Sparse individual-turn effect: **{decision['individual_turn_effect_is_sparse']}**",
            f"- Proceed to selective role-memory stage: **{decision['proceed_to_selective_memory_stage']}**",
            "",
            decision["note"],
            "",
        ]
    )
    return "\n".join(lines)


def _experiment_paths(root: Path, config: dict[str, Any], run_name: str) -> dict[str, Path]:
    section = config["llm_context_utility"]
    base = _resolve_path(root, section.get("artifact_dir", "artifacts/llm_stage1/meld"))
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "records": base / "records" / "validation.jsonl",
        "manifest": base / "run_manifest.json",
        "summary_json": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
    }


def _load_context(config_path: Path) -> tuple[Path, dict[str, Any], list[str], list[Dialogue]]:
    root = config_path.resolve().parent.parent
    config = _read_yaml(config_path)
    section = config.get("llm_context_utility", {})
    if section.get("split", "validation") != "validation":
        raise ValueError("Stage 1 is validation-only; refusing to read the test split")
    labels = list(section.get("labels", []))
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("llm_context_utility.labels must contain unique emotion labels")
    processed_dir = _resolve_path(root, config["data"]["processed_dir"])
    validation_file = processed_dir / "validation.jsonl"
    if not validation_file.exists():
        raise FileNotFoundError(
            f"Missing {validation_file}. Prepare MELD first; the test split is not used by Stage 1."
        )
    dialogues = read_jsonl(validation_file)
    unknown = sorted(
        {turn.emotion for dialogue in dialogues for turn in dialogue.utterances} - set(labels)
    )
    if unknown:
        raise ValueError(f"Validation data contains labels absent from config: {unknown}")
    return root, config, labels, dialogues


def check(config_path: Path) -> dict[str, Any]:
    root, config, labels, dialogues = _load_context(config_path)
    section = config["llm_context_utility"]
    result = {
        "status": "ready",
        "stage": "validation_only",
        "project_root": str(root),
        "model_name": section["model_name"],
        "labels": labels,
        "dialogues": len(dialogues),
        "utterances": sum(len(dialogue.utterances) for dialogue in dialogues),
        "history_eligible_targets": sum(max(len(dialogue.utterances) - 1, 0) for dialogue in dialogues),
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "test_locked": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(
    config_path: Path,
    run_name: str,
    max_dialogues: int | None,
    max_targets: int | None,
    overwrite: bool,
    model_override: str | None,
) -> dict[str, Any]:
    root, config, labels, dialogues = _load_context(config_path)
    section = config["llm_context_utility"]
    if model_override:
        section["model_name"] = model_override
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]
    paths = _experiment_paths(root, config, run_name)
    if overwrite and paths["records"].exists():
        paths["records"].unlink()
    existing = _read_jsonl_dicts(paths["records"])
    completed = {record["key"] for record in existing}

    scorer = CausalLabelScorer(section, labels)
    manifest = {
        "experiment": "LLM_context_utility_stage1",
        "version": STAGE1_VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path.resolve()),
        "run_name": run_name,
        "limits": {"max_dialogues": max_dialogues, "max_targets": max_targets},
        "scorer": scorer.audit,
        "thresholds": {
            "utility_epsilon": float(section.get("utility_epsilon", 0.02)),
            "deletion_epsilon": float(section.get("deletion_epsilon", 0.01)),
        },
    }
    _write_json(paths["manifest"], manifest)

    max_history = int(section.get("max_history_turns", 12))
    preamble = str(
        section.get(
            "prompt_preamble",
            "You are a careful dialogue emotion classifier. Do not invent missing context.",
        )
    )
    utility_epsilon = float(section.get("utility_epsilon", 0.02))
    deletion_epsilon = float(section.get("deletion_epsilon", 0.01))
    processed_targets = len(existing)
    stop = False
    progress = tqdm(dialogues, desc="Stage 1 dialogues", unit="dialogue")
    for dialogue in progress:
        pending: list[TargetSpec] = []
        for target in make_target_specs(dialogue, max_history_turns=max_history):
            if target.key in completed:
                continue
            if max_targets is not None and processed_targets + len(pending) >= max_targets:
                stop = True
                break
            pending.append(target)
        if not pending:
            if stop:
                break
            continue

        prompts: list[str] = []
        spans: list[tuple[int, int]] = []
        for target in pending:
            start = len(prompts)
            prompts.append(build_prompt(target, labels, [], preamble))
            prompts.append(build_prompt(target, labels, target.history, preamble))
            for deletion_index in range(len(target.history)):
                deleted_history = target.history[:deletion_index] + target.history[deletion_index + 1 :]
                prompts.append(build_prompt(target, labels, deleted_history, preamble))
            spans.append((start, len(prompts)))
        prompt_scores = scorer.score_prompts(prompts)
        new_records = [
            _record_for_target(
                target,
                labels,
                prompt_scores[start:end],
                utility_epsilon,
                deletion_epsilon,
            )
            for target, (start, end) in zip(pending, spans)
        ]
        _append_jsonl(paths["records"], new_records)
        processed_targets += len(new_records)
        progress.set_postfix(targets=processed_targets)
        if stop:
            break

    return summarize(config_path, run_name)


def summarize(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config, labels, _ = _load_context(config_path)
    section = config["llm_context_utility"]
    paths = _experiment_paths(root, config, run_name)
    records = _read_jsonl_dicts(paths["records"])
    summary = summarize_records(
        records,
        labels=labels,
        utility_epsilon=float(section.get("utility_epsilon", 0.02)),
        deletion_epsilon=float(section.get("deletion_epsilon", 0.01)),
        bootstrap_samples=int(section.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(section.get("bootstrap_seed", 2026)),
    )
    summary["artifacts"] = {key: str(value) for key, value in paths.items()}
    if paths["manifest"].exists():
        summary["manifest_sha256"] = _sha256(paths["manifest"])
    _write_json(paths["summary_json"], summary)
    paths["summary_md"].parent.mkdir(parents=True, exist_ok=True)
    paths["summary_md"].write_text(_markdown_report(summary), encoding="utf-8")
    print(_markdown_report(summary))
    print(f"\nJSON report: {paths['summary_json']}")
    print(f"Markdown report: {paths['summary_md']}")
    return summary


def verify(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config, _, _ = _load_context(config_path)
    paths = _experiment_paths(root, config, run_name)
    missing = [str(path) for key, path in paths.items() if key != "base" and not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Stage 1 artifact(s): {missing}")
    summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    checks = {
        "validation_only": summary.get("split") == "validation",
        "has_targets": int(summary.get("targets", 0)) > 0,
        "has_history_targets": int(summary.get("history_eligible_targets", 0)) > 0,
        "has_deletions": int(summary.get("turn_deletion", {}).get("count", 0)) > 0,
        "manifest_locks_test": json.loads(paths["manifest"].read_text(encoding="utf-8")).get("test_locked") is True,
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 1 verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation-only LLM context-utility Stage 1")
    parser.add_argument("stage", choices=("check", "smoke", "run", "summarize", "verify"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--model-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.stage == "check":
        check(arguments.config)
    elif arguments.stage == "smoke":
        run(
            arguments.config,
            run_name="smoke",
            max_dialogues=arguments.max_dialogues or 2,
            max_targets=arguments.max_targets or 12,
            overwrite=arguments.overwrite,
            model_override=arguments.model_name,
        )
    elif arguments.stage == "run":
        run(
            arguments.config,
            run_name=arguments.run_name,
            max_dialogues=arguments.max_dialogues,
            max_targets=arguments.max_targets,
            overwrite=arguments.overwrite,
            model_override=arguments.model_name,
        )
    elif arguments.stage == "summarize":
        summarize(arguments.config, arguments.run_name)
    else:
        verify(arguments.config, arguments.run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
