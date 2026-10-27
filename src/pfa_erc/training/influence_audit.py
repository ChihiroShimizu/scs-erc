from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .common import (
    classification_metrics,
    collate_dialogues,
    get_device,
    load_feature_store,
    move_to_device,
    save_json,
    seed_everything,
)
from .erc_trainer import _build_model, _validate_current_logit_features


def _gap_bucket(gap: int) -> str:
    if gap <= 1:
        return "1"
    if gap <= 3:
        return "2-3"
    if gap <= 7:
        return "4-7"
    return "8+"


def _effect_class(delta_nll: float, threshold: float) -> str:
    if delta_nll > threshold:
        return "helpful"
    if delta_nll < -threshold:
        return "harmful"
    return "neutral"


def _transition_groups(dialogue: dict[str, Any]) -> list[str]:
    """Build evaluation-only same-speaker stable/transition labels."""
    previous_label: dict[int, int] = {}
    groups: list[str] = []
    for speaker_value, label_value in zip(
        dialogue["speaker_ids"].tolist(),
        dialogue["labels"].tolist(),
    ):
        speaker = int(speaker_value)
        label = int(label_value)
        if speaker not in previous_label:
            groups.append("unknown")
        elif previous_label[speaker] == label:
            groups.append("stable")
        else:
            groups.append("transition")
        previous_label[speaker] = label
    return groups


def _select_turns(value: Any, indices: list[int]) -> Any:
    if torch.is_tensor(value):
        index = torch.tensor(indices, dtype=torch.long)
        return value.index_select(0, index)
    if isinstance(value, list):
        return [value[index] for index in indices]
    return value


def _dialogue_prefix(
    dialogue: dict[str, Any],
    target_turn: int,
    removed_turn: int | None = None,
) -> dict[str, Any]:
    indices = list(range(target_turn + 1))
    if removed_turn is not None:
        if not 0 <= removed_turn < target_turn:
            raise ValueError("removed_turn must be a history position before target_turn")
        indices.remove(removed_turn)
    result: dict[str, Any] = {}
    turn_fields = {
        "embeddings",
        "speaker_ids",
        "speaker_names",
        "labels",
        "label_names",
        "texts",
        "utterance_ids",
        "current_logits",
    }
    for key, value in dialogue.items():
        result[key] = _select_turns(value, indices) if key in turn_fields else value
    return result


def _replacement_prefix(
    dialogue: dict[str, Any],
    target_turn: int,
    history_turn: int,
    donor_dialogue: dict[str, Any],
    donor_turn: int,
) -> dict[str, Any]:
    if not 0 <= history_turn < target_turn:
        raise ValueError("history_turn must be a history position before target_turn")
    result = _dialogue_prefix(dialogue, target_turn)
    result["embeddings"] = result["embeddings"].clone()
    result["embeddings"][history_turn] = donor_dialogue["embeddings"][donor_turn]
    if "current_logits" in result:
        if "current_logits" not in donor_dialogue:
            raise ValueError("Replacement donor is missing current_logits")
        result["current_logits"] = result["current_logits"].clone()
        result["current_logits"][history_turn] = donor_dialogue["current_logits"][
            donor_turn
        ]
    # Speaker identity and sequence length stay unchanged. This isolates the
    # utterance-content contribution from speaker-change and distance effects.
    return result


def _cluster_bootstrap_mean_ci(
    records: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[float | None]:
    if not records or samples <= 0:
        return [None, None]
    grouped: dict[str, list[float]] = defaultdict(list)
    for index, record in enumerate(records):
        cluster = str(record.get("target_dialogue_id", index))
        grouped[cluster].append(float(record["delta_nll"]))
    clusters = sorted(grouped)
    if len(clusters) == 1:
        value = float(np.mean(grouped[clusters[0]]))
        return [value, value]
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        values = [
            value
            for cluster in selected
            for value in grouped[str(cluster)]
        ]
        draws.append(float(np.mean(values)))
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return [float(lower), float(upper)]


def _cluster_bootstrap_contrast(
    records: list[dict[str, Any]],
    field: str,
    first_value: Any,
    second_value: Any,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    first = [record for record in records if record[field] == first_value]
    second = [record for record in records if record[field] == second_value]
    result = {
        "definition": f"mean({first_value}) - mean({second_value})",
        "first_count": len(first),
        "second_count": len(second),
        "mean_difference": None,
        "cluster_bootstrap_95ci": [None, None],
    }
    if not first or not second:
        return result
    result["mean_difference"] = float(
        np.mean([record["delta_nll"] for record in first])
        - np.mean([record["delta_nll"] for record in second])
    )
    if samples <= 0:
        return result

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[str(record.get("target_dialogue_id", index))].append(record)
    clusters = sorted(grouped)
    if len(clusters) == 1:
        result["cluster_bootstrap_95ci"] = [
            result["mean_difference"],
            result["mean_difference"],
        ]
        return result

    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(samples):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sampled = [
            record
            for cluster in selected
            for record in grouped[str(cluster)]
        ]
        sampled_first = [
            float(record["delta_nll"])
            for record in sampled
            if record[field] == first_value
        ]
        sampled_second = [
            float(record["delta_nll"])
            for record in sampled
            if record[field] == second_value
        ]
        if sampled_first and sampled_second:
            differences.append(
                float(np.mean(sampled_first) - np.mean(sampled_second))
            )
    if differences:
        lower, upper = np.quantile(differences, [0.025, 0.975])
        result["cluster_bootstrap_95ci"] = [float(lower), float(upper)]
    return result


def _summarize_records(
    records: list[dict[str, Any]],
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "unique_targets": 0,
            "mean_delta_nll": None,
            "median_delta_nll": None,
            "std_delta_nll": None,
            "mean_delta_nll_95ci": [None, None],
            "helpful_rate": None,
            "harmful_rate": None,
            "neutral_rate": None,
            "context_helpful_flips": 0,
            "context_harmful_flips": 0,
            "net_context_flips": 0,
            "unique_context_helpful_flip_targets": 0,
            "unique_context_harmful_flip_targets": 0,
            "unique_context_helpful_flip_rate": None,
            "unique_context_harmful_flip_rate": None,
        }
    values = [float(record["delta_nll"]) for record in records]
    helpful = sum(value > threshold for value in values)
    harmful = sum(value < -threshold for value in values)
    neutral = len(values) - helpful - harmful
    helpful_flips = sum(
        bool(record["full_correct"]) and not bool(record["intervened_correct"])
        for record in records
    )
    harmful_flips = sum(
        not bool(record["full_correct"]) and bool(record["intervened_correct"])
        for record in records
    )
    target_keys = {
        (str(record.get("target_dialogue_id")), int(record.get("target_turn", -1)))
        for record in records
    }
    helpful_flip_targets = {
        (str(record.get("target_dialogue_id")), int(record.get("target_turn", -1)))
        for record in records
        if bool(record["full_correct"]) and not bool(record["intervened_correct"])
    }
    harmful_flip_targets = {
        (str(record.get("target_dialogue_id")), int(record.get("target_turn", -1)))
        for record in records
        if not bool(record["full_correct"]) and bool(record["intervened_correct"])
    }
    return {
        "count": len(records),
        "unique_targets": len(target_keys),
        "mean_delta_nll": float(np.mean(values)),
        "median_delta_nll": float(np.median(values)),
        "std_delta_nll": float(np.std(values)),
        "mean_delta_nll_95ci": _cluster_bootstrap_mean_ci(
            records, bootstrap_samples, seed
        ),
        "helpful_rate": helpful / len(values),
        "harmful_rate": harmful / len(values),
        "neutral_rate": neutral / len(values),
        "context_helpful_flips": helpful_flips,
        "context_harmful_flips": harmful_flips,
        "net_context_flips": helpful_flips - harmful_flips,
        "unique_context_helpful_flip_targets": len(helpful_flip_targets),
        "unique_context_harmful_flip_targets": len(harmful_flip_targets),
        "unique_context_helpful_flip_rate": len(helpful_flip_targets)
        / len(target_keys),
        "unique_context_harmful_flip_rate": len(harmful_flip_targets)
        / len(target_keys),
    }


def _grouped_summary(
    records: list[dict[str, Any]],
    field: str,
    threshold: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(record)
    return {
        key: _summarize_records(
            values,
            threshold,
            bootstrap_samples,
            seed + index + 1,
        )
        for index, (key, values) in enumerate(sorted(grouped.items()))
    }


def _forward_target_logits(
    model: torch.nn.Module,
    samples: list[dict[str, Any]],
    batch_size: int,
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(samples), batch_size):
        batch = move_to_device(
            collate_dialogues(samples[start : start + batch_size]),
            device,
        )
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(
                batch["embeddings"],
                batch["speaker_ids"],
                batch["mask"],
                labels=None,
                teacher_forcing=0.0,
                current_logits=batch.get("current_logits"),
            )
        if isinstance(logits, tuple):
            logits = logits[0]
        target_positions = batch["mask"].long().sum(dim=1) - 1
        row_indices = torch.arange(logits.shape[0], device=logits.device)
        outputs.append(logits[row_indices, target_positions].float().cpu())
    return torch.cat(outputs, dim=0)


def _baseline_logits(
    model: torch.nn.Module,
    dialogue: dict[str, Any],
    device: torch.device,
    use_amp: bool,
) -> torch.Tensor:
    batch = move_to_device(collate_dialogues([dialogue]), device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_amp,
    ):
        logits = model(
            batch["embeddings"],
            batch["speaker_ids"],
            batch["mask"],
            labels=None,
            teacher_forcing=0.0,
            current_logits=batch.get("current_logits"),
        )
    if isinstance(logits, tuple):
        logits = logits[0]
    length = int(dialogue["embeddings"].shape[0])
    return logits[0, :length].float().cpu()


def _donor_pool(
    dialogues: list[dict[str, Any]],
) -> list[tuple[int, int]]:
    return [
        (dialogue_index, turn_index)
        for dialogue_index, dialogue in enumerate(dialogues)
        for turn_index in range(int(dialogue["embeddings"].shape[0]))
    ]


def _choose_donor(
    pool: list[tuple[int, int]],
    target_dialogue_index: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    if not pool:
        raise ValueError("Cannot run replacement audit with an empty donor pool")
    start = int(rng.integers(0, len(pool)))
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate[0] != target_dialogue_index:
            return candidate
    raise ValueError("Replacement audit requires at least two dialogues")


def _baseline_subset_metrics(
    labels: list[int],
    predictions: list[int],
    groups: list[str],
    id_to_label: dict[int, str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group_name in ("all", "stable", "transition", "unknown"):
        indices = (
            list(range(len(labels)))
            if group_name == "all"
            else [index for index, value in enumerate(groups) if value == group_name]
        )
        if not indices:
            output[group_name] = {"utterances": 0}
            continue
        metrics = classification_metrics(
            [labels[index] for index in indices],
            [predictions[index] for index in indices],
            id_to_label,
        )
        metrics["utterances"] = len(indices)
        output[group_name] = metrics
    return output


def audit_context_influence(
    config: dict[str, Any],
    device: str | None = None,
    checkpoint_path: str | Path | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Run deletion/replacement counterfactuals over observed dialogue history.

    Positive ``delta_nll`` means that the original history turn helped the gold
    prediction. Negative values mean removing/replacing it improved the target
    prediction, which is evidence of harmful context propagation.
    """
    audit = config.get("influence_audit", {})
    default_seed = int(audit.get("seed", config["experiment"].get("seed", 42)))
    sampling_seed = int(audit.get("sampling_seed", default_seed))
    bootstrap_seed = int(audit.get("bootstrap_seed", sampling_seed))
    seed_everything(
        sampling_seed,
        bool(config["experiment"].get("deterministic", True)),
    )
    device_obj = get_device(device)
    use_amp = bool(audit.get("use_amp", device_obj.type == "cuda")) and (
        device_obj.type == "cuda"
    )
    selected_split = str(split or audit.get("split", "validation"))
    if selected_split == "test" and not bool(audit.get("allow_test", False)):
        raise ValueError(
            "8A defaults to validation-only model development. Set "
            "influence_audit.allow_test=true explicitly for the final locked run."
        )

    store = load_feature_store(config["data"]["feature_file"])
    if selected_split not in store["splits"]:
        raise KeyError(f"Feature store has no split {selected_split!r}")
    dialogues = list(store["splits"][selected_split])
    max_dialogues = int(audit.get("max_dialogues", 0))
    if max_dialogues > 0:
        dialogues = dialogues[:max_dialogues]
    if len(dialogues) < 1:
        raise ValueError(f"No dialogues found for split {selected_split!r}")

    checkpoint_file = Path(
        checkpoint_path
        or audit.get("checkpoint")
        or config["train_erc"]["checkpoint"]
    )
    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    checkpoint_seed = int(checkpoint.get("seed", default_seed))
    label_to_id = {
        str(key): int(value)
        for key, value in checkpoint.get(
            "label_to_id", store["label_to_id"]
        ).items()
    }
    id_to_label = {value: key for key, value in label_to_id.items()}
    model_config = checkpoint.get("config", config)
    if bool(model_config["model"].get("use_hitft", False)):
        raise ValueError(
            "8A is designed to isolate observed-history influence. Use a no-HiTFT "
            "checkpoint such as Experiment 3B."
        )
    model = _build_model(model_config, len(label_to_id), device_obj)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if getattr(model, "use_current_logit_anchor", False):
        _validate_current_logit_features(store, len(label_to_id))

    interventions: list[str] = []
    if bool(audit.get("include_deletion", True)):
        interventions.append("deletion")
    if bool(audit.get("include_replacement", True)):
        interventions.append("replacement")
    if not interventions:
        raise ValueError("Enable at least one 8A intervention")

    batch_size = int(audit.get("batch_size", 64))
    max_history_turns = int(audit.get("max_history_turns", 12))
    min_target_turn = max(1, int(audit.get("min_target_turn", 1)))
    max_interventions = int(audit.get("max_interventions", 0))
    threshold = float(audit.get("effect_threshold", 0.01))
    bootstrap_samples = int(audit.get("bootstrap_samples", 500))
    top_k = int(audit.get("top_k", 50))
    rng = np.random.default_rng(sampling_seed)
    donors = _donor_pool(dialogues)

    records: list[dict[str, Any]] = []
    baseline_labels: list[int] = []
    baseline_predictions: list[int] = []
    baseline_groups: list[str] = []
    seen_targets: set[tuple[str, int]] = set()

    stop = False
    for dialogue_index, dialogue in enumerate(
        tqdm(dialogues, desc=f"8A influence audit ({selected_split})")
    ):
        length = int(dialogue["embeddings"].shape[0])
        if length <= min_target_turn:
            continue
        full_logits = _baseline_logits(model, dialogue, device_obj, use_amp)
        full_log_probabilities = torch.log_softmax(full_logits, dim=-1)
        full_probabilities = torch.softmax(full_logits, dim=-1)
        full_predictions = full_logits.argmax(dim=-1)
        transition_groups = _transition_groups(dialogue)
        utterance_ids = dialogue.get("utterance_ids", list(range(length)))
        dialogue_id = str(dialogue.get("dialogue_id", dialogue_index))

        samples: list[dict[str, Any]] = []
        metadata: list[dict[str, Any]] = []
        for target_turn in range(min_target_turn, length):
            target_key = (dialogue_id, target_turn)
            if target_key not in seen_targets:
                seen_targets.add(target_key)
                baseline_labels.append(int(dialogue["labels"][target_turn]))
                baseline_predictions.append(int(full_predictions[target_turn]))
                baseline_groups.append(transition_groups[target_turn])
            start = max(0, target_turn - max_history_turns)
            for history_turn in range(start, target_turn):
                for intervention in interventions:
                    donor_dialogue_index: int | None = None
                    donor_turn: int | None = None
                    if intervention == "deletion":
                        sample = _dialogue_prefix(
                            dialogue,
                            target_turn,
                            removed_turn=history_turn,
                        )
                    else:
                        donor_dialogue_index, donor_turn = _choose_donor(
                            donors,
                            dialogue_index,
                            rng,
                        )
                        sample = _replacement_prefix(
                            dialogue,
                            target_turn,
                            history_turn,
                            dialogues[donor_dialogue_index],
                            donor_turn,
                        )
                    samples.append(sample)
                    metadata.append(
                        {
                            "intervention": intervention,
                            "target_dialogue_id": dialogue_id,
                            "target_utterance_id": str(utterance_ids[target_turn]),
                            "target_turn": target_turn,
                            "history_utterance_id": str(utterance_ids[history_turn]),
                            "history_turn": history_turn,
                            "gap": target_turn - history_turn,
                            "gap_bucket": _gap_bucket(target_turn - history_turn),
                            "same_speaker": bool(
                                dialogue["speaker_ids"][history_turn]
                                == dialogue["speaker_ids"][target_turn]
                            ),
                            "target_emotion": id_to_label[
                                int(dialogue["labels"][target_turn])
                            ],
                            "target_transition_group": transition_groups[target_turn],
                            "donor_dialogue_id": (
                                None
                                if donor_dialogue_index is None
                                else str(
                                    dialogues[donor_dialogue_index].get(
                                        "dialogue_id", donor_dialogue_index
                                    )
                                )
                            ),
                            "donor_turn": donor_turn,
                        }
                    )
                    if max_interventions > 0 and (
                        len(records) + len(samples) >= max_interventions
                    ):
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break

        if samples:
            counterfactual_logits = _forward_target_logits(
                model,
                samples,
                batch_size,
                device_obj,
                use_amp,
            )
            counterfactual_log_probabilities = torch.log_softmax(
                counterfactual_logits, dim=-1
            )
            counterfactual_probabilities = torch.softmax(
                counterfactual_logits, dim=-1
            )
            counterfactual_predictions = counterfactual_logits.argmax(dim=-1)
            for row, meta in enumerate(metadata):
                target_turn = int(meta["target_turn"])
                gold = int(dialogue["labels"][target_turn])
                full_nll = -float(full_log_probabilities[target_turn, gold])
                intervened_nll = -float(
                    counterfactual_log_probabilities[row, gold]
                )
                delta_nll = intervened_nll - full_nll
                full_prediction = int(full_predictions[target_turn])
                intervened_prediction = int(counterfactual_predictions[row])
                record = {
                    **meta,
                    "gold_label": gold,
                    "full_prediction": id_to_label[full_prediction],
                    "intervened_prediction": id_to_label[intervened_prediction],
                    "full_correct": full_prediction == gold,
                    "intervened_correct": intervened_prediction == gold,
                    "full_gold_probability": float(
                        full_probabilities[target_turn, gold]
                    ),
                    "intervened_gold_probability": float(
                        counterfactual_probabilities[row, gold]
                    ),
                    "full_nll": full_nll,
                    "intervened_nll": intervened_nll,
                    "delta_nll": delta_nll,
                    "effect": _effect_class(delta_nll, threshold),
                }
                records.append(record)
        if stop:
            break

    if not records:
        raise ValueError("8A produced no counterfactual records")

    by_intervention: dict[str, Any] = {}
    for intervention in interventions:
        subset = [
            record
            for record in records
            if record["intervention"] == intervention
        ]
        by_intervention[intervention] = {
            "overall": _summarize_records(
                subset,
                threshold,
                bootstrap_samples,
                bootstrap_seed,
            ),
            "same_speaker": _grouped_summary(
                subset,
                "same_speaker",
                threshold,
                bootstrap_samples,
                bootstrap_seed + 100,
            ),
            "gap_bucket": _grouped_summary(
                subset,
                "gap_bucket",
                threshold,
                bootstrap_samples,
                bootstrap_seed + 200,
            ),
            "target_transition_group": _grouped_summary(
                subset,
                "target_transition_group",
                threshold,
                bootstrap_samples,
                bootstrap_seed + 300,
            ),
            "target_emotion": _grouped_summary(
                subset,
                "target_emotion",
                threshold,
                bootstrap_samples,
                bootstrap_seed + 400,
            ),
            "contrasts": {
                "same_minus_different_speaker": _cluster_bootstrap_contrast(
                    subset,
                    "same_speaker",
                    True,
                    False,
                    bootstrap_samples,
                    bootstrap_seed + 500,
                ),
                "gap_1_minus_8_plus": _cluster_bootstrap_contrast(
                    subset,
                    "gap_bucket",
                    "1",
                    "8+",
                    bootstrap_samples,
                    bootstrap_seed + 600,
                ),
                "transition_minus_stable": _cluster_bootstrap_contrast(
                    subset,
                    "target_transition_group",
                    "transition",
                    "stable",
                    bootstrap_samples,
                    bootstrap_seed + 700,
                ),
            },
        }

    sorted_by_effect = sorted(records, key=lambda item: float(item["delta_nll"]))
    result = {
        "experiment": "8A Context Influence Audit",
        "research_question": (
            "Which observed history turns help or harm a target emotion prediction, "
            "and does influence depend on speaker and interaction distance?"
        ),
        "split": selected_split,
        "seed": checkpoint_seed,
        "checkpoint_seed": checkpoint_seed,
        "sampling_seed": sampling_seed,
        "bootstrap_seed": bootstrap_seed,
        "checkpoint": str(checkpoint_file),
        "feature_file": str(config["data"]["feature_file"]),
        "settings": {
            "interventions": interventions,
            "max_dialogues": max_dialogues,
            "max_history_turns": max_history_turns,
            "min_target_turn": min_target_turn,
            "max_interventions": max_interventions,
            "effect_threshold": threshold,
            "bootstrap_samples": bootstrap_samples,
            "confidence_interval_unit": "Dialogue_ID cluster bootstrap",
            "sampling_seed": sampling_seed,
            "bootstrap_seed": bootstrap_seed,
            "no_gold_labels_passed_to_model": True,
            "replacement_preserves_original_speaker": True,
        },
        "counts": {
            "dialogues": len(dialogues),
            "eligible_targets": len(baseline_labels),
            "counterfactuals": len(records),
        },
        "baseline": _baseline_subset_metrics(
            baseline_labels,
            baseline_predictions,
            baseline_groups,
            id_to_label,
        ),
        "effects": by_intervention,
        "top_harmful_context": sorted_by_effect[:top_k],
        "top_helpful_context": list(reversed(sorted_by_effect[-top_k:])),
        "interpretation": {
            "positive_delta_nll": (
                "The original history turn helped the gold prediction."
            ),
            "negative_delta_nll": (
                "The original history turn harmed the gold prediction."
            ),
            "context_helpful_flip": (
                "Full history was correct but the intervention became wrong."
            ),
            "context_harmful_flip": (
                "Full history was wrong but the intervention became correct."
            ),
        },
    }

    report_file = Path(
        audit.get(
            "report_file",
            Path(config["experiment"]["output_dir"])
            / "results"
            / f"{config['experiment']['name']}_{selected_split}.json",
        )
    )
    records_file = Path(
        audit.get(
            "records_file",
            report_file.with_name(f"{report_file.stem}_records.jsonl"),
        )
    )
    save_json(result, report_file)
    records_file.parent.mkdir(parents=True, exist_ok=True)
    with records_file.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "report_file": str(report_file),
        "records_file": str(records_file),
        **result,
    }


__all__ = [
    "audit_context_influence",
    "_dialogue_prefix",
    "_effect_class",
    "_gap_bucket",
    "_replacement_prefix",
    "_summarize_records",
    "_transition_groups",
]
