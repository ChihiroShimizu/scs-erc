"""Pure numerical helpers for the Stage 7.3 routing diagnostic.

This module intentionally has no PyTorch or project-local imports.  Keeping the
statistics and budget logic independent makes them testable on a CPU-only
machine and prevents validation labels from entering threshold selection.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, cohen_kappa_score


@dataclass(frozen=True)
class ThresholdRule:
    """A calibration-only score threshold with deterministic boundary ties."""

    cutoff: float
    tie_fraction: float
    requested_rate: float
    target_count: int
    eligible_count: int
    total_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def log_probability(scores: Sequence[float], gold_index: int) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("scores must be a non-empty one-dimensional sequence")
    if not 0 <= int(gold_index) < len(values):
        raise IndexError("gold_index is outside the score vector")
    shifted = values - float(values.max())
    return float(shifted[int(gold_index)] - np.log(np.exp(shifted).sum()))


def nll_gain(
    current_scores: Sequence[float],
    recent_scores: Sequence[float],
    gold_index: int,
) -> float:
    """Return NLL(current) - NLL(recent3); positive means context helped."""
    return float(
        -log_probability(current_scores, gold_index)
        + log_probability(recent_scores, gold_index)
    )


def probability_entropy(scores: Sequence[float], temperature: float = 1.0) -> float:
    values = np.asarray(scores, dtype=np.float64) / max(float(temperature), 1e-8)
    values -= values.max()
    probability = np.exp(values)
    probability /= probability.sum()
    return -float(np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0))))


def normalise_layer_specs(
    values: Sequence[str | int], total_layers: int | None = None
) -> list[str]:
    """Canonicalise probe locations without silently dropping duplicates."""
    result: list[str] = []
    for raw in values:
        if isinstance(raw, bool):
            raise ValueError("Boolean values are not valid layer specifications")
        if isinstance(raw, int):
            if raw <= 0:
                raise ValueError("Numeric decoder layers are one-indexed and positive")
            if total_layers is not None and raw > int(total_layers):
                raise ValueError(f"Layer {raw} exceeds total_layers={total_layers}")
            value = f"layer_{raw}"
        else:
            text = str(raw).strip().lower()
            if text in {"embedding", "final"}:
                value = text
            elif text.startswith("layer_") and text[6:].isdigit():
                number = int(text[6:])
                if number <= 0:
                    raise ValueError("Numeric decoder layers are one-indexed and positive")
                if total_layers is not None and number > int(total_layers):
                    raise ValueError(
                        f"Layer {number} exceeds total_layers={total_layers}"
                    )
                value = f"layer_{number}"
            elif text.isdigit():
                number = int(text)
                if number <= 0:
                    raise ValueError("Numeric decoder layers are one-indexed and positive")
                if total_layers is not None and number > int(total_layers):
                    raise ValueError(
                        f"Layer {number} exceeds total_layers={total_layers}"
                    )
                value = f"layer_{number}"
            else:
                raise ValueError(f"Unknown layer specification: {raw!r}")
        if value in result:
            raise ValueError(f"Duplicate layer specification: {value}")
        result.append(value)
    if not result:
        raise ValueError("At least one probe layer is required")
    return result


def layer_number(layer: str, total_layers: int) -> int:
    if layer == "embedding":
        return 0
    if layer == "final":
        return int(total_layers)
    if layer.startswith("layer_") and layer[6:].isdigit():
        value = int(layer[6:])
        if not 0 < value <= int(total_layers):
            raise ValueError(f"Invalid layer {layer} for total_layers={total_layers}")
        return value
    raise ValueError(f"Unknown canonical layer: {layer}")


def _hash_fraction(key: str, salt: int) -> float:
    digest = hashlib.sha256(f"{int(salt)}::{key}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], "big", signed=False)
    return integer / float(2**64)


def fit_budget_threshold(
    scores: Sequence[float],
    history_available: Sequence[bool],
    requested_rate: float,
) -> ThresholdRule:
    """Fit a READ threshold using only calibration scores and availability."""
    value = np.asarray(scores, dtype=np.float64)
    history = np.asarray(history_available, dtype=bool)
    if value.ndim != 1 or history.ndim != 1 or len(value) != len(history):
        raise ValueError("scores and history_available must be aligned vectors")
    if not np.isfinite(value).all():
        raise ValueError("routing scores must be finite")
    rate = float(requested_rate)
    if not 0.0 <= rate <= 1.0:
        raise ValueError("requested_rate must lie in [0, 1]")
    eligible = value[history]
    target = min(int(np.rint(rate * len(value))), int(len(eligible)))
    if target <= 0 or not len(eligible):
        return ThresholdRule(math.inf, 0.0, rate, 0, len(eligible), len(value))
    if target >= len(eligible):
        return ThresholdRule(-math.inf, 1.0, rate, len(eligible), len(eligible), len(value))
    ordered = np.sort(eligible)[::-1]
    boundary = float(ordered[target - 1])
    greater = int(np.sum(eligible > boundary))
    tied = int(np.sum(eligible == boundary))
    tie_fraction = (target - greater) / tied if tied else 0.0
    return ThresholdRule(
        boundary,
        float(np.clip(tie_fraction, 0.0, 1.0)),
        rate,
        target,
        len(eligible),
        len(value),
    )


def apply_budget_threshold(
    scores: Sequence[float],
    history_available: Sequence[bool],
    keys: Sequence[str],
    rule: ThresholdRule,
    *,
    salt: int = 0,
) -> np.ndarray:
    """Apply a frozen calibration threshold; validation labels are not inputs."""
    value = np.asarray(scores, dtype=np.float64)
    history = np.asarray(history_available, dtype=bool)
    if len(value) != len(history) or len(value) != len(keys):
        raise ValueError("scores, history_available and keys must be aligned")
    if math.isinf(rule.cutoff):
        return history.copy() if rule.cutoff < 0 else np.zeros(len(value), dtype=bool)
    read = (value > rule.cutoff) & history
    tied = np.isclose(value, rule.cutoff, rtol=0.0, atol=1e-12) & history
    if rule.tie_fraction >= 1.0:
        read |= tied
    elif rule.tie_fraction > 0.0:
        for index in np.flatnonzero(tied):
            if _hash_fraction(str(keys[index]), int(salt)) < rule.tie_fraction:
                read[index] = True
    return read


def exact_top_k_mask(
    scores: Sequence[float],
    history_available: Sequence[bool],
    count: int,
    keys: Sequence[str] | None = None,
) -> np.ndarray:
    value = np.asarray(scores, dtype=np.float64)
    history = np.asarray(history_available, dtype=bool)
    if len(value) != len(history):
        raise ValueError("scores and history_available must be aligned")
    if keys is None:
        keys = [str(index) for index in range(len(value))]
    if len(keys) != len(value):
        raise ValueError("keys must align with scores")
    candidates = np.flatnonzero(history).tolist()
    requested = min(max(int(count), 0), len(candidates))
    candidates.sort(key=lambda index: (-float(value[index]), str(keys[index])))
    result = np.zeros(len(value), dtype=bool)
    if requested:
        result[np.asarray(candidates[:requested], dtype=np.int64)] = True
    return result


def exact_random_mask(
    history_available: Sequence[bool],
    count: int,
    keys: Sequence[str],
    *,
    salt: int,
) -> np.ndarray:
    history = np.asarray(history_available, dtype=bool)
    if len(history) != len(keys):
        raise ValueError("history_available and keys must be aligned")
    candidates = np.flatnonzero(history).tolist()
    candidates.sort(key=lambda index: (_hash_fraction(str(keys[index]), salt), str(keys[index])))
    requested = min(max(int(count), 0), len(candidates))
    result = np.zeros(len(history), dtype=bool)
    if requested:
        result[np.asarray(candidates[:requested], dtype=np.int64)] = True
    return result


def _safe_average_precision(target: np.ndarray, score: np.ndarray) -> float | None:
    if len(target) == 0 or len(np.unique(target)) < 2:
        return None
    return float(average_precision_score(target, score))


def utility_stability_report(
    deltas: Sequence[Sequence[float]],
    binary_labels: Sequence[Sequence[int]],
    seeds: Sequence[int],
    *,
    near_zero_bands: Sequence[float] = (0.02, 0.05, 0.10),
    minimum_spearman: float = 0.40,
    minimum_sign_agreement: float = 0.70,
) -> dict[str, Any]:
    values = np.asarray(deltas, dtype=np.float64)
    labels = np.asarray(binary_labels, dtype=np.int64)
    if values.ndim != 2 or labels.shape != values.shape:
        raise ValueError("deltas and binary_labels must have shape [seed, example]")
    if values.shape[0] != len(seeds):
        raise ValueError("seed count does not match utility matrices")
    if not np.isfinite(values).all():
        raise ValueError("utility deltas must be finite")
    consensus = np.median(values, axis=0)
    consensus_positive = (consensus > 0.0).astype(np.int64)
    pairwise: list[dict[str, Any]] = []
    for left, right in combinations(range(len(seeds)), 2):
        correlation = spearmanr(values[left], values[right]).statistic
        kappa = cohen_kappa_score(labels[left], labels[right])
        sign: dict[str, Any] = {}
        for band in near_zero_bands:
            mask = (np.abs(values[left]) >= float(band)) & (
                np.abs(values[right]) >= float(band)
            )
            sign[f"{float(band):.4f}"] = {
                "examples": int(mask.sum()),
                "agreement": (
                    float(
                        np.mean(
                            np.sign(values[left, mask])
                            == np.sign(values[right, mask])
                        )
                    )
                    if mask.any()
                    else None
                ),
            }
        pairwise.append(
            {
                "left_seed": int(seeds[left]),
                "right_seed": int(seeds[right]),
                "spearman": float(correlation) if np.isfinite(correlation) else None,
                "binary_kappa": float(kappa) if np.isfinite(kappa) else None,
                "sign_agreement": sign,
            }
        )
    near_zero = {
        f"{float(band):.4f}": float(np.mean(np.abs(consensus) < float(band)))
        for band in near_zero_bands
    }
    correlations = [
        float(item["spearman"])
        for item in pairwise
        if item["spearman"] is not None
    ]
    band_key = f"{0.05:.4f}"
    agreements = [
        float(item["sign_agreement"][band_key]["agreement"])
        for item in pairwise
        if band_key in item["sign_agreement"]
        and item["sign_agreement"][band_key]["agreement"] is not None
    ]
    median_spearman = float(np.median(correlations)) if correlations else None
    median_agreement = float(np.median(agreements)) if agreements else None
    unreliable = bool(
        len(seeds) >= 2
        and (
            median_spearman is None
            or median_spearman < float(minimum_spearman)
        )
        and (
            median_agreement is None
            or median_agreement < float(minimum_sign_agreement)
        )
    )
    per_seed = []
    for index, seed in enumerate(seeds):
        per_seed.append(
            {
                "seed": int(seed),
                "mean": float(values[index].mean()),
                "median": float(np.median(values[index])),
                "std": float(values[index].std()),
                "positive_rate": float(np.mean(values[index] > 0.0)),
                "utility_v1_positive_rate": float(labels[index].mean()),
                "ap_against_consensus_positive": _safe_average_precision(
                    consensus_positive, values[index]
                ),
            }
        )
    return {
        "seeds": [int(seed) for seed in seeds],
        "examples": int(values.shape[1]),
        "consensus": {
            "definition": "median seed NLL gain",
            "mean": float(consensus.mean()),
            "median": float(np.median(consensus)),
            "std": float(consensus.std()),
            "minimum": float(consensus.min()) if len(consensus) else None,
            "maximum": float(consensus.max()) if len(consensus) else None,
            "positive_rate": float(consensus_positive.mean()) if len(consensus) else None,
            "near_zero_rate": near_zero,
            "quantiles": {
                str(value): float(np.quantile(consensus, value))
                for value in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
            }
            if len(consensus)
            else {},
        },
        "per_seed": per_seed,
        "pairwise": pairwise,
        "median_pairwise_spearman": median_spearman,
        "median_sign_agreement_outside_0.05": median_agreement,
        "failure_rule": {
            "minimum_spearman": float(minimum_spearman),
            "minimum_sign_agreement": float(minimum_sign_agreement),
            "requires_both_failures": True,
        },
        "target_reliability_pass": not unreliable if len(seeds) >= 2 else None,
        "single_seed_smoke_only": len(seeds) < 2,
    }


def capacity_matched_pair_bottleneck(hidden_size: int, base_bottleneck: int) -> int:
    """Choose a 2H-input bottleneck with no more parameters than the H-input head."""
    hidden = int(hidden_size)
    base = int(base_bottleneck)
    if hidden <= 0 or base <= 0:
        raise ValueError("hidden_size and base_bottleneck must be positive")
    base_parameters = hidden * base + 2 * hidden + 2 * base + 1
    pair = max(1, int((base_parameters - 4 * hidden - 1) // (2 * hidden + 2)))
    return pair


def _confusion_by_dialogue(
    gold: np.ndarray,
    prediction: np.ndarray,
    dialogue_index: np.ndarray,
    dialogue_count: int,
    class_count: int,
) -> np.ndarray:
    result = np.zeros((dialogue_count, class_count, class_count), dtype=np.float64)
    for true, pred, dialogue in zip(gold, prediction, dialogue_index):
        result[int(dialogue), int(true), int(pred)] += 1.0
    return result


def _f1_from_confusion(confusion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    true_positive = np.diagonal(confusion, axis1=1, axis2=2)
    support = confusion.sum(axis=2)
    predicted = confusion.sum(axis=1)
    denominator = 2.0 * true_positive + (predicted - true_positive) + (
        support - true_positive
    )
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    total = support.sum(axis=1)
    weighted = np.divide(
        (per_class * support).sum(axis=1),
        total,
        out=np.zeros_like(total),
        where=total > 0,
    )
    macro = per_class.mean(axis=1)
    return weighted, macro


def dialogue_bootstrap_counts(
    dialogue_ids: Sequence[str], samples: int, seed: int
) -> tuple[list[str], np.ndarray, np.ndarray]:
    names = list(dict.fromkeys(map(str, dialogue_ids)))
    mapping = {name: index for index, name in enumerate(names)}
    indexes = np.asarray([mapping[str(value)] for value in dialogue_ids], dtype=np.int64)
    if not names:
        raise ValueError("dialogue_ids must not be empty")
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(
        len(names), np.full(len(names), 1.0 / len(names)), size=int(samples)
    ).astype(np.float64)
    return names, indexes, counts


def _difference_summary(observed: float, values: np.ndarray) -> dict[str, float]:
    return {
        "observed_difference": float(observed),
        "bootstrap_mean": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "probability_positive": float(np.mean(values > 0.0)),
    }


def paired_dialogue_bootstrap(
    gold: Sequence[int],
    candidate_probabilities: Sequence[Sequence[float]],
    control_probabilities: Sequence[Sequence[float]],
    dialogue_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
    precomputed: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    truth = np.asarray(gold, dtype=np.int64)
    candidate = np.asarray(candidate_probabilities, dtype=np.float64)
    control = np.asarray(control_probabilities, dtype=np.float64)
    if candidate.shape != control.shape or candidate.ndim != 2:
        raise ValueError("candidate/control probabilities must share shape [N, C]")
    if len(truth) != len(candidate) or len(truth) != len(dialogue_ids):
        raise ValueError("bootstrap inputs are not aligned")
    if precomputed is None:
        names, dialogue_index, counts = dialogue_bootstrap_counts(
            dialogue_ids, samples, seed
        )
    else:
        names, dialogue_index, counts = precomputed
        if counts.shape[0] != int(samples):
            raise ValueError("precomputed bootstrap sample count differs")
    classes = candidate.shape[1]
    candidate_prediction = candidate.argmax(axis=1)
    control_prediction = control.argmax(axis=1)
    candidate_confusion = _confusion_by_dialogue(
        truth, candidate_prediction, dialogue_index, len(names), classes
    )
    control_confusion = _confusion_by_dialogue(
        truth, control_prediction, dialogue_index, len(names), classes
    )
    candidate_boot = counts @ candidate_confusion.reshape(len(names), -1)
    control_boot = counts @ control_confusion.reshape(len(names), -1)
    candidate_boot = candidate_boot.reshape(-1, classes, classes)
    control_boot = control_boot.reshape(-1, classes, classes)
    candidate_weighted, candidate_macro = _f1_from_confusion(candidate_boot)
    control_weighted, control_macro = _f1_from_confusion(control_boot)
    observed_candidate = candidate_confusion.sum(axis=0, keepdims=True)
    observed_control = control_confusion.sum(axis=0, keepdims=True)
    observed_candidate_w, observed_candidate_m = _f1_from_confusion(observed_candidate)
    observed_control_w, observed_control_m = _f1_from_confusion(observed_control)

    clipped_candidate = np.clip(candidate, 1e-12, 1.0)
    clipped_control = np.clip(control, 1e-12, 1.0)
    rows = np.arange(len(truth))
    candidate_nll = -np.log(clipped_candidate[rows, truth])
    control_nll = -np.log(clipped_control[rows, truth])
    one_hot = np.eye(classes, dtype=np.float64)[truth]
    candidate_brier = np.square(candidate - one_hot).sum(axis=1)
    control_brier = np.square(control - one_hot).sum(axis=1)

    def grouped_sum(values: np.ndarray) -> np.ndarray:
        result = np.zeros(len(names), dtype=np.float64)
        np.add.at(result, dialogue_index, values)
        return result

    dialogue_sizes = grouped_sum(np.ones(len(truth), dtype=np.float64))
    bootstrap_sizes = counts @ dialogue_sizes
    candidate_nll_boot = (counts @ grouped_sum(candidate_nll)) / bootstrap_sizes
    control_nll_boot = (counts @ grouped_sum(control_nll)) / bootstrap_sizes
    candidate_brier_boot = (counts @ grouped_sum(candidate_brier)) / bootstrap_sizes
    control_brier_boot = (counts @ grouped_sum(control_brier)) / bootstrap_sizes
    return {
        "unit": "dialogue",
        "samples": int(samples),
        "seed": int(seed),
        "dialogues": len(names),
        "weighted_f1": _difference_summary(
            float(observed_candidate_w[0] - observed_control_w[0]),
            candidate_weighted - control_weighted,
        ),
        "macro_f1": _difference_summary(
            float(observed_candidate_m[0] - observed_control_m[0]),
            candidate_macro - control_macro,
        ),
        "nll_improvement": _difference_summary(
            float(control_nll.mean() - candidate_nll.mean()),
            control_nll_boot - candidate_nll_boot,
        ),
        "brier_improvement": _difference_summary(
            float(control_brier.mean() - candidate_brier.mean()),
            control_brier_boot - candidate_brier_boot,
        ),
    }


def paired_dialogue_ap_bootstrap(
    target: Sequence[int],
    candidate_scores: Sequence[float],
    control_scores: Sequence[float],
    dialogue_ids: Sequence[str],
    *,
    samples: int,
    seed: int,
    precomputed: tuple[list[str], np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    truth = np.asarray(target, dtype=np.int64)
    candidate = np.asarray(candidate_scores, dtype=np.float64)
    control = np.asarray(control_scores, dtype=np.float64)
    if not (len(truth) == len(candidate) == len(control) == len(dialogue_ids)):
        raise ValueError("AP bootstrap inputs are not aligned")
    if len(np.unique(truth)) < 2:
        raise ValueError("AP bootstrap requires both target classes")
    if precomputed is None:
        names, dialogue_index, counts = dialogue_bootstrap_counts(
            dialogue_ids, samples, seed
        )
    else:
        names, dialogue_index, counts = precomputed
    differences = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        weights = counts[index, dialogue_index]
        if float(weights[truth == 1].sum()) == 0 or float(weights[truth == 0].sum()) == 0:
            differences[index] = np.nan
            continue
        differences[index] = average_precision_score(
            truth, candidate, sample_weight=weights
        ) - average_precision_score(truth, control, sample_weight=weights)
    differences = differences[np.isfinite(differences)]
    observed = average_precision_score(truth, candidate) - average_precision_score(
        truth, control
    )
    result = _difference_summary(float(observed), differences)
    result.update(
        {
            "unit": "dialogue",
            "requested_samples": int(samples),
            "valid_samples": int(len(differences)),
            "seed": int(seed),
            "dialogues": len(names),
        }
    )
    return result
