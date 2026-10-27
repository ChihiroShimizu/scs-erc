"""Pure CPU protocol and statistics for Stage 8C.1 (current vs previous 3).

No model, routing, training, annotation loader, or test-set access lives here.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

VERSION = "1.0.0"
EXPERIMENT = "llm_context_budget_stage8c_1"
PROTOCOL = "current-vs-previous3-direct-emotion-v1"
LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
VIEWS = ("fast", "slow")
SPLITS = ("calibration", "validation")
SYSTEM = (
    "You are a dialogue emotion classifier. Use only the supplied text. "
    "Return exactly one emotion label and no explanation."
)
INVALID = "__invalid__"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def stable_hash(value: str, seed: int) -> int:
    # Intentionally identical to Stage 7's dialogue split implementation.
    return int.from_bytes(hashlib.sha256(f"{seed}:{value}".encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Turn:
    index: int
    speaker: str
    text: str


@dataclass(frozen=True)
class Target:
    dialogue_id: str
    turn_index: int
    utterance_id: str
    speaker: str
    text: str
    gold: str
    history: tuple[Turn, ...]

    @property
    def key(self) -> str:
        return f"{self.dialogue_id}::{self.turn_index}"

    @property
    def identity_sha256(self) -> str:
        # Shared identity across both views; includes gold ONLY for cache checking.
        return digest({"key": self.key, "utterance_id": self.utterance_id,
                       "speaker": self.speaker, "text": self.text, "gold": self.gold,
                       "history": [(x.index, x.speaker, x.text) for x in self.history]})


def build_targets(dialogues: Sequence[Any]) -> list[Target]:
    result = []
    seen_dialogues = set()
    for dialogue in dialogues:
        did = str(dialogue.dialogue_id)
        if did in seen_dialogues:
            raise ValueError(f"Duplicate dialogue ID: {did}")
        seen_dialogues.add(did)
        for i, turn in enumerate(dialogue.utterances):
            if str(turn.emotion) not in LABELS:
                raise ValueError(f"Unknown gold label at {did}::{i}: {turn.emotion}")
            result.append(Target(
                did, i, str(turn.utterance_id), str(turn.speaker), str(turn.text),
                str(turn.emotion),
                tuple(Turn(j, str(dialogue.utterances[j].speaker),
                           str(dialogue.utterances[j].text))
                      for j in range(max(0, i - 3), i)),
            ))
    if not result:
        raise ValueError("Empty target set")
    return result


def split_dialogues(ids: Sequence[str], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    unique = sorted(set(ids), key=lambda x: (stable_hash(x, seed), x))
    if len(unique) < 2 or not 0 < fraction < 1:
        raise ValueError("A dialogue-disjoint split requires >=2 dialogues and 0<fraction<1")
    count = max(1, min(len(unique) - 1, int(round(len(unique) * fraction))))
    return set(unique[count:]), set(unique[:count])


def build_prompt(target: Target, view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"Unknown context view: {view}")
    expected = list(range(max(0, target.turn_index - 3), target.turn_index))
    if [x.index for x in target.history] != expected:
        raise ValueError(f"Noncausal or incomplete previous-3 window: {target.key}")
    history = target.history if view == "slow" else ()
    evidence = "\n".join(f"[{x.index}] {x.speaker}: {x.text}" for x in history)
    if not evidence:
        evidence = "[no earlier turns supplied]"
    # The ONLY experimental change is evidence content, including the empty marker.
    # FAST/SLOW strategy tags, gold, annotation fields and future turns are absent.
    return (
        "Task: classify the target speaker's expressed emotion. "
        "Use the earlier evidence only when it helps interpret the target.\n"
        f"Earlier evidence:\n{evidence}\n\n"
        f"Target [{target.turn_index}] {target.speaker}: {target.text}\n\n"
        f"Return exactly one label from: {', '.join(LABELS)}.\nEmotion:"
    )


def parse_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    strict = stripped in LABELS
    # Allow only capitalization / one trailing period, never label extraction from CoT.
    normalized = stripped.lower()
    if normalized.endswith("."):
        normalized = normalized[:-1]
    prediction = normalized if normalized in LABELS else INVALID
    return {"prediction": prediction, "parse_success": prediction != INVALID,
            "strict_format_success": strict}


def check_pair(fast: Sequence[dict], slow: Sequence[dict]) -> None:
    if not fast or len(fast) != len(slow):
        raise ValueError("Empty or unequal paired prediction sets")
    if len({x["key"] for x in fast}) != len(fast):
        raise ValueError("Duplicate prediction key")
    for f, s in zip(fast, slow):
        for field in ("key", "dialogue_id", "gold", "target_sha256"):
            if f[field] != s[field]:
                raise ValueError(f"Pair mismatch in {field}: {f['key']}")


def confusion(records: Sequence[dict], weights: Sequence[int] | None = None) -> np.ndarray:
    # Final column retains invalid outputs as false negatives; never drop records.
    matrix = np.zeros((len(LABELS), len(LABELS) + 1), dtype=np.float64)
    index = {label: i for i, label in enumerate(LABELS)}
    for i, row in enumerate(records):
        if row["gold"] not in index:
            raise ValueError("Unexpected gold label")
        matrix[index[row["gold"]], index.get(row["prediction"], len(LABELS))] += (
            1 if weights is None else weights[i])
    return matrix


def scores_from_confusion(matrix: np.ndarray) -> dict[str, float]:
    support = matrix.sum(axis=1)
    tp = np.diag(matrix[:, :len(LABELS)])
    predicted = matrix[:, :len(LABELS)].sum(axis=0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(len(LABELS)),
                   where=(support + predicted) != 0)
    total = support.sum()
    if total <= 0:
        raise ValueError("Cannot score an empty sample")
    return {"weighted_f1": float(np.dot(f1, support) / total),
            "macro_f1": float(np.mean(f1)), "accuracy": float(tp.sum() / total)}


def metrics(records: Sequence[dict]) -> dict[str, Any]:
    result = scores_from_confusion(confusion(records))
    result["targets"] = len(records)
    for field in ("prompt_tokens", "completion_tokens", "generated_tokens_including_eos",
                  "parse_success", "strict_format_success"):
        result[f"mean_{field}" if "tokens" in field else field] = float(
            np.mean([r[field] for r in records]))
    positions = [r["tokens_to_emotion"] for r in records if r["tokens_to_emotion"] is not None]
    result["mean_tokens_to_emotion_valid_outputs_only"] = float(np.mean(positions)) if positions else None
    result["invalid_outputs"] = sum(r["prediction"] == INVALID for r in records)
    return result


def paired_diagnostic(fast: Sequence[dict], slow: Sequence[dict]) -> dict[str, Any]:
    check_pair(fast, slow)
    counts = Counter({"both_correct": 0, "history_helps": 0, "history_harms": 0, "both_wrong": 0})
    by_emotion = {label: Counter() for label in LABELS}
    by_history = {str(i): Counter() for i in range(4)}
    examples = []
    for f, s in zip(fast, slow):
        a, b = f["prediction"] == f["gold"], s["prediction"] == s["gold"]
        kind = "both_correct" if a and b else "history_helps" if b else "history_harms" if a else "both_wrong"
        counts[kind] += 1
        by_emotion[f["gold"]][kind] += 1
        by_history[str(f["available_history_turns"])][kind] += 1
        if f["available_history_turns"] == 0 and f["response"] != s["response"]:
            raise ValueError("Identical first-turn prompts produced different outputs; audit pairing/backend")
        examples.append({"key": f["key"], "dialogue_id": f["dialogue_id"], "gold": f["gold"],
                         "fast_prediction": f["prediction"], "slow_prediction": s["prediction"],
                         "type": kind, "available_history_turns": f["available_history_turns"]})
    n = len(fast)
    return {"counts": dict(counts), "rates": {k: v / n for k, v in counts.items()},
            "net_accuracy_change": (counts["history_helps"] - counts["history_harms"]) / n,
            "by_gold_emotion": {k: dict(v) for k, v in by_emotion.items()},
            "by_available_history_turns": {k: dict(v) for k, v in by_history.items()},
            "cases": examples}


def oracle_and_random(fast: Sequence[dict], slow: Sequence[dict], budgets: Sequence[float],
                      random_repeats: int, seed: int) -> list[dict]:
    check_pair(fast, slow)
    if random_repeats < 1 or any(not 0 <= b <= 1 for b in budgets):
        raise ValueError("Invalid oracle budgets or random repeat count")
    helpful = [i for i, (f, s) in enumerate(zip(fast, slow))
               if f["prediction"] != f["gold"] and s["prediction"] == s["gold"]]
    helpful.sort(key=lambda i: (stable_hash(fast[i]["key"], seed), fast[i]["key"]))
    result = []
    for budget in budgets:
        selected = set(helpful[:min(len(helpful), math.floor(budget * len(fast)))])
        mixed = [s if i in selected else f for i, (f, s) in enumerate(zip(fast, slow))]
        random_metrics = []
        # Independent but reproducible random selections; same realized call count as oracle.
        rng = random.Random(seed)
        for _ in range(random_repeats):
            picks = set(rng.sample(range(len(fast)), len(selected)))
            random_metrics.append(metrics([s if i in picks else f for i, (f, s) in enumerate(zip(fast, slow))]))
        result.append({"budget": float(budget), "realized_slow_rate": len(selected) / len(fast),
                       "selected_slow_count": len(selected), "correctness_oracle": metrics(mixed),
                       "matched_random_repeats": random_repeats,
                       "matched_random_weighted_f1_mean": float(np.mean([x["weighted_f1"] for x in random_metrics])),
                       "matched_random_weighted_f1_std": float(np.std([x["weighted_f1"] for x in random_metrics], ddof=1)) if random_repeats > 1 else 0.0,
                       "cost_scope": "offline_one_selected_call; routing/escalation_cost_not_included",
                       "deployable_policy": False,
                       "oracle_scope": "gold-assisted_correctness_oracle; not_a_wF1_optimized_upper_bound"})
    return result


def paired_bootstrap(pairs: Sequence[tuple[Sequence[dict], Sequence[dict]]],
                     samples: int, seed: int) -> dict[str, Any]:
    if samples < 2 or not pairs:
        raise ValueError("Need >=2 bootstrap samples and >=1 paired seed")
    reference = [(r["key"], r["dialogue_id"], r["gold"], r["target_sha256"]) for r in pairs[0][0]]
    ids = sorted({x[1] for x in reference})
    matrices = []
    for fast, slow in pairs:
        check_pair(fast, slow)
        if [(r["key"], r["dialogue_id"], r["gold"], r["target_sha256"]) for r in fast] != reference:
            raise ValueError("Bootstrap requires identical ordered targets across adapter seeds")
        views = []
        for records in (fast, slow):
            groups = defaultdict(list)
            for row in records:
                groups[row["dialogue_id"]].append(row)
            views.append(np.stack([confusion(groups[did]) for did in ids]))
        matrices.append(views)
    matrices = np.asarray(matrices)  # seed, view, dialogue, gold, prediction
    observed = float(np.mean([scores_from_confusion(x[1].sum(axis=0))["weighted_f1"] -
                              scores_from_confusion(x[0].sum(axis=0))["weighted_f1"] for x in matrices]))
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        weights = np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids))
        sample = np.einsum("svdij,d->svij", matrices, weights)
        values.append(np.mean([scores_from_confusion(x[1])["weighted_f1"] -
                               scores_from_confusion(x[0])["weighted_f1"] for x in sample]))
    return {"comparison": "slow_minus_fast", "mean_seed_weighted_f1_difference": observed,
            "ci95": [float(x) for x in np.quantile(values, [.025, .975])],
            "samples": samples, "dialogues": len(ids), "adapter_seeds": len(pairs),
            "resampling_unit": "dialogue; same resample across views and fixed adapter seeds",
            "scope": "development_validation_uncertainty; not_independent_test_confirmation"}


def aggregate(values: Sequence[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def summarize_runtime(batches: Sequence[dict]) -> dict[str, Any]:
    if not batches:
        raise ValueError("No runtime batches")
    examples = sum(b["examples"] for b in batches)
    total_ms = sum(b["e2e_ms"] for b in batches)
    return {
        "examples_including_repeats": examples, "batches": len(batches),
        "mean_batch_emotion_response_latency_ms": float(np.mean([b["e2e_ms"] for b in batches])),
        "p50_batch_emotion_response_latency_ms": float(np.quantile([b["e2e_ms"] for b in batches], .5)),
        "p95_batch_emotion_response_latency_ms": float(np.quantile([b["e2e_ms"] for b in batches], .95)),
        "mean_model_generate_batch_ms": float(np.mean([b["model_generate_ms"] for b in batches])),
        "amortized_ms_per_example": total_ms / examples,
        "examples_per_second": examples * 1000 / total_ms,
        "mean_prompt_tokens": sum(b["prompt_tokens_sum"] for b in batches) / examples,
        "mean_completion_tokens": sum(b["completion_tokens_sum"] for b in batches) / examples,
        "mean_generated_tokens_including_eos": sum(b["generated_tokens_sum"] for b in batches) / examples,
        "padded_prompt_tokens_total": sum(b["padded_prompt_tokens"] for b in batches),
        "peak_cuda_allocated_bytes": max(b["peak_cuda_allocated_bytes"] for b in batches),
        "peak_cuda_reserved_bytes": max(b["peak_cuda_reserved_bytes"] for b in batches),
        "scope": "prompt_render+tokenize+H2D+generate_through_EOS+decode_parse; excludes_load_queue_network",
        "latency_warning": "batch>1 amortized_ms is throughput cost, not individual response latency",
    }
