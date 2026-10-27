from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from pfa_erc.data import Dialogue, Utterance, read_jsonl
from pfa_erc.training.llm_context_utility_stage1 import (
    CausalLabelScorer,
    TargetSpec,
    _append_jsonl,
    _metric_bundle,
    _read_jsonl_dicts,
    _write_json,
    make_target_specs,
)
from pfa_erc.training.llm_selective_context_stage2 import (
    _calibrated_payload,
    _gold_nll,
)


STAGE3_VERSION = "1.3.0"

VARIANT_LABELS = {
    "current_only": "3A current-only",
    "always_full": "3B always-full history",
    "entropy_router": "2C frozen entropy router",
    "evidence_only": "3C LLM-selected evidence",
    "speaker_memory": "3D speaker-write memory",
    "role_memory": "3E affected-role memory",
    "oracle_current_full": "Oracle current/full",
}

PLANNER_SYSTEM_PROMPT = """You are the memory controller for causal emotion recognition in a
dialogue.
Use only the earlier turns and the current utterance supplied by the user. Never assume future
speakers or future events. Do not output an emotion label for the target utterance. Decide whether
earlier evidence is needed, identify concise evidence, and decide which already-seen roles may have
their ongoing state changed by the current utterance. Return one JSON object only."""


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _root_and_config(config_path: Path) -> tuple[Path, dict[str, Any]]:
    path = config_path.expanduser().resolve()
    return path.parent.parent, _read_yaml(path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _section(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("llm_role_memory_stage3")
    if not isinstance(section, dict):
        raise KeyError("Missing llm_role_memory_stage3 configuration section")
    return section


def _labels(config: dict[str, Any]) -> list[str]:
    labels = list(_section(config).get("labels", []))
    if len(labels) < 2 or len(labels) != len(set(labels)):
        raise ValueError("llm_role_memory_stage3.labels must contain unique labels")
    return labels


def _paths(root: Path, config: dict[str, Any], run_name: str) -> dict[str, Path]:
    section = _section(config)
    base = _resolve(root, section.get("artifact_dir", "artifacts/llm_role_memory_stage3/meld"))
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "stage1_records": _resolve(root, section["stage1_validation_records"]),
        "stage2_models": _resolve(root, section["stage2_model_file"]),
        "manifest": base / "run_manifest.json",
        "report_json": base / "reports" / "validation_summary.json",
        "report_md": base / "reports" / "validation_summary.md",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
        "decision": base / "reports" / "frozen_validation_decision.json",
    }


def _plan_path(paths: dict[str, Path], seed: int) -> Path:
    return paths["base"] / "plans" / f"planner_seed{seed}.jsonl"


def _score_path(paths: dict[str, Path], seed: int) -> Path:
    return paths["base"] / "scores" / f"predictions_seed{seed}.jsonl"


def _load_validation(root: Path, config: dict[str, Any]) -> list[Dialogue]:
    path = _resolve(root, config["data"]["processed_dir"]) / "validation.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing processed validation split: {path}")
    return read_jsonl(path)


def _targets(dialogue: Dialogue, max_history_turns: int) -> list[TargetSpec]:
    return make_target_specs(dialogue, max_history_turns=max_history_turns, include_first_turn=True)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first balanced JSON object without accepting trailing prose as JSON."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return value if isinstance(value, dict) else None
    return None


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if not math.isfinite(number):
        number = default
    return min(max(number, 0.0), 1.0)


def _canonical_role(value: Any, allowed_roles: Sequence[str]) -> str | None:
    raw = str(value).strip()
    mapping = {role.casefold(): role for role in allowed_roles}
    return mapping.get(raw.casefold())


def normalize_plan(
    raw: dict[str, Any] | None,
    *,
    allowed_roles: Sequence[str],
    history_turn_ids: Sequence[int],
    max_selected_turns: int,
    max_state_characters: int,
) -> dict[str, Any]:
    """Validate an LLM plan and remove every reference outside causal input."""
    parse_success = isinstance(raw, dict) and "reasoning_mode" in raw
    raw = raw or {}
    history_set = set(int(value) for value in history_turn_ids)
    mode = str(raw.get("reasoning_mode", "fast")).strip().lower()
    if mode not in {"fast", "memory"}:
        mode = "fast"

    evidence_source = raw.get("evidence", raw.get("evidence_turn_ids", []))
    evidence_items: list[dict[str, Any]] = []
    if isinstance(evidence_source, list):
        for item in evidence_source:
            if isinstance(item, dict):
                turn_value = item.get("turn_id", item.get("id"))
                confidence = _bounded_float(item.get("confidence", 1.0), 1.0)
            else:
                turn_value = item
                confidence = 1.0
            try:
                turn_id = int(turn_value)
            except (TypeError, ValueError):
                continue
            if turn_id not in history_set or any(x["turn_id"] == turn_id for x in evidence_items):
                continue
            evidence_items.append({"turn_id": turn_id, "confidence": confidence})
    evidence_items.sort(key=lambda item: (-item["confidence"], item["turn_id"]))
    evidence_items = evidence_items[:max_selected_turns]
    evidence_items.sort(key=lambda item: item["turn_id"])

    affected: list[str] = []
    raw_affected = raw.get("affected_roles", [])
    if isinstance(raw_affected, list):
        for value in raw_affected:
            role = _canonical_role(value, allowed_roles)
            if role is not None and role not in affected:
                affected.append(role)

    updates: list[dict[str, Any]] = []
    raw_updates = raw.get("updates", [])
    if isinstance(raw_updates, list):
        for item in raw_updates:
            if not isinstance(item, dict):
                continue
            role = _canonical_role(item.get("role"), allowed_roles)
            action = str(item.get("action", "keep")).strip().lower()
            if role is None or action not in {"keep", "update", "forget"}:
                continue
            state = " ".join(str(item.get("state", "")).split())[:max_state_characters]
            update_evidence: list[int] = []
            values = item.get("evidence_turn_ids", [])
            if isinstance(values, list):
                for value in values:
                    try:
                        turn_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if turn_id in history_set and turn_id not in update_evidence:
                        update_evidence.append(turn_id)
            if action == "update" and not state:
                continue
            updates.append(
                {
                    "role": role,
                    "action": action,
                    "state": state,
                    "evidence_turn_ids": sorted(update_evidence),
                    "confidence": _bounded_float(item.get("confidence", 0.5), 0.5),
                }
            )
            if role not in affected and action in {"update", "forget"}:
                affected.append(role)

    if mode == "fast":
        evidence_items = []
    return {
        "parse_success": parse_success,
        "reasoning_mode": mode,
        "context_confidence": _bounded_float(raw.get("context_confidence", 0.0)),
        "evidence": evidence_items,
        "evidence_turn_ids": [item["turn_id"] for item in evidence_items],
        "affected_roles": affected,
        "updates": updates,
    }


def build_planner_prompt(target: TargetSpec, max_selected_turns: int) -> str:
    start = target.turn_index - len(target.history)
    history_lines = [
        f"[{start + offset}] {turn.speaker}: {turn.text}"
        for offset, turn in enumerate(target.history)
    ]
    history = "\n".join(history_lines) if history_lines else "[no earlier turns]"
    roles = []
    for turn in (*target.history, Utterance(target.utterance_id, target.speaker, target.text, "")):
        if turn.speaker not in roles:
            roles.append(turn.speaker)
    role_text = ", ".join(roles)
    return f"""Earlier causal dialogue:
{history}

Current turn [{target.turn_index}]:
{target.speaker}: {target.text}

Already-seen roles: {role_text}

Return JSON with this schema:
{{
  "reasoning_mode": "fast" or "memory",
  "context_confidence": number from 0 to 1,
  "evidence": [{{"turn_id": earlier integer, "confidence": number from 0 to 1}}],
  "affected_roles": [role names copied exactly from Already-seen roles],
  "updates": [
    {{
      "role": "an already-seen role",
      "action": "keep" or "update" or "forget",
      "state": "short event-grounded state, without assigning the target emotion label",
      "evidence_turn_ids": [earlier integer IDs],
      "confidence": number from 0 to 1
    }}
  ]
}}

Rules:
- Choose fast when the current utterance alone is sufficient; otherwise choose memory.
- Select at most {max_selected_turns} earlier evidence turns.
- A current utterance may affect a listener who is not speaking now.
- Update only an already-seen role that is plausibly affected by the current utterance.
- Do not assume an absent or uninvolved role is affected. A non-speaking role needs direct address,
  an explicit reference, or clear participation in the same local interaction.
- Use forget only when an earlier state is explicitly resolved or contradicted, not merely because
  the topic changes.
- Do not predict or print the final emotion label. Output JSON only."""


def _history_by_absolute_id(target: TargetSpec) -> dict[int, Utterance]:
    start = target.turn_index - len(target.history)
    return {start + offset: turn for offset, turn in enumerate(target.history)}


def _selected_turns(target: TargetSpec, plan: dict[str, Any]) -> list[tuple[int, Utterance]]:
    history = _history_by_absolute_id(target)
    return [
        (turn_id, history[turn_id])
        for turn_id in plan["evidence_turn_ids"]
        if turn_id in history
    ]


def _memory_snapshot(
    memory: dict[str, list[dict[str, Any]]], roles: Iterable[str], max_roles: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in roles:
        if role not in memory or any(item["role"] == role for item in output):
            continue
        output.append({"role": role, "entries": [dict(entry) for entry in memory[role]]})
        if len(output) >= max_roles:
            break
    return output


def _render_memory(snapshot: Sequence[dict[str, Any]]) -> str:
    lines: list[str] = []
    for role_state in snapshot:
        role = role_state["role"]
        for entry in role_state["entries"]:
            evidence = (
                ",".join(str(value) for value in entry.get("evidence_turn_ids", []))
                or "none"
            )
            lines.append(
                f"- {role}: {entry['state']} (source turn {entry['source_turn_id']}; "
                f"evidence {evidence}; confidence {entry['confidence']:.2f})"
            )
    return "\n".join(lines) if lines else "[no retained role memory]"


def build_prediction_prompt(
    target: TargetSpec,
    labels: Sequence[str],
    selected: Sequence[tuple[int, Utterance]],
    memory_snapshot: Sequence[dict[str, Any]],
    prompt_preamble: str,
) -> str:
    evidence = (
        "\n".join(f"[{turn_id}] {turn.speaker}: {turn.text}" for turn_id, turn in selected)
        if selected
        else "[no raw evidence selected]"
    )
    return (
        f"{prompt_preamble.strip()}\n\n"
        f"Role memory before the target turn:\n{_render_memory(memory_snapshot)}\n\n"
        f"Selected earlier evidence:\n{evidence}\n\n"
        f"Target utterance:\n{target.speaker}: {target.text}\n\n"
        "Classify the target speaker's emotion using only the information shown. "
        f"Return exactly one label from: {', '.join(labels)}.\nEmotion:"
    )


def apply_plan_to_memory(
    memory: dict[str, list[dict[str, Any]]],
    plan: dict[str, Any],
    *,
    source_turn_id: int,
    source_speaker: str,
    speaker_only: bool,
    max_entries_per_role: int,
) -> None:
    """Apply planner actions after prediction; no current-turn update is read by its prediction."""
    for update in plan.get("updates", []):
        role = update["role"]
        if speaker_only and role != source_speaker:
            continue
        action = update["action"]
        if action == "forget":
            memory.pop(role, None)
        elif action == "update":
            memory.setdefault(role, []).append(
                {
                    "state": update["state"],
                    "evidence_turn_ids": list(update.get("evidence_turn_ids", [])),
                    "confidence": float(update.get("confidence", 0.5)),
                    "source_turn_id": int(source_turn_id),
                    "source_speaker": source_speaker,
                }
            )
            memory[role] = memory[role][-max_entries_per_role:]


class StructuredPlanner:
    def __init__(self, config: dict[str, Any]) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "transformers is required; run python3 -m pip install -e ."
            ) from error

        self.model_name = str(config["model_name"])
        self.batch_size = int(config.get("planner_batch_size", 2))
        self.max_input_tokens = int(config.get("max_input_tokens", 1024))
        self.max_new_tokens = int(config.get("planner_max_new_tokens", 220))
        self.do_sample = bool(config.get("planner_do_sample", True))
        self.temperature = float(config.get("planner_temperature", 0.2))
        self.top_p = float(config.get("planner_top_p", 0.9))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype_name = str(config.get("dtype", "bfloat16"))
        dtype = torch.float32
        if self.device.type == "cuda" and dtype_name == "float16":
            dtype = torch.float16
        elif self.device.type == "cuda" and dtype_name == "bfloat16":
            dtype = torch.bfloat16

        print(f"Loading Stage 3 planner tokenizer: {self.model_name}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=bool(config.get("local_files_only", False)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        print(f"Loading Stage 3 planner on {self.device}: {self.model_name}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=bool(config.get("trust_remote_code", False)),
            local_files_only=bool(config.get("local_files_only", False)),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def audit(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature if self.do_sample else None,
            "top_p": self.top_p if self.do_sample else None,
        }

    def _render(self, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"System: {PLANNER_SYSTEM_PROMPT}\nUser: {user_prompt}\nAssistant:"

    def generate(self, prompts: Sequence[str], seed: int) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(prompts), self.batch_size):
            rendered = [self._render(prompt) for prompt in prompts[start : start + self.batch_size]]
            encoded = self.tokenizer(
                rendered,
                padding=True,
                truncation=True,
                max_length=self.max_input_tokens,
                return_tensors="pt",
            )
            input_tokens = encoded["attention_mask"].sum(dim=1).tolist()
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            torch.manual_seed(seed + start)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + start)
            generation_args: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": self.do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if self.do_sample:
                generation_args.update(temperature=self.temperature, top_p=self.top_p)
            with torch.inference_mode():
                output = self.model.generate(**encoded, **generation_args)
            prompt_width = encoded["input_ids"].shape[1]
            generated = output[:, prompt_width:]
            for row, tokens in enumerate(generated):
                text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
                results.append(
                    {
                        "text": text,
                        "input_tokens": int(input_tokens[row]),
                        "output_tokens": int((tokens != self.tokenizer.pad_token_id).sum().item()),
                    }
                )
            del output, generated, encoded
        return results


def _seen_roles(target: TargetSpec) -> list[str]:
    roles: list[str] = []
    for turn in target.history:
        if turn.speaker not in roles:
            roles.append(turn.speaker)
    if target.speaker not in roles:
        roles.append(target.speaker)
    return roles


def check(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    labels = _labels(config)
    dialogues = _load_validation(root, config)
    paths = _paths(root, config, run_name)
    missing = [
        str(path)
        for path in (paths["stage1_records"], paths["stage2_models"])
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing prerequisite artifact(s): " + ", ".join(missing))
    validation_keys = {
        target.key
        for dialogue in dialogues
        for target in _targets(
            dialogue, int(section.get("max_history_turns", 12))
        )
    }
    stage1_records = _read_jsonl_dicts(paths["stage1_records"])
    stage1_keys = {record["key"] for record in stage1_records}
    if len(stage1_records) != len(stage1_keys):
        raise ValueError("Stage 1 validation records contain duplicate target keys")
    if stage1_keys != validation_keys:
        raise ValueError(
            "Stage 1 records do not exactly match the processed MELD validation targets: "
            f"missing={len(validation_keys - stage1_keys)}, "
            f"unexpected={len(stage1_keys - validation_keys)}"
        )
    stage2_models = json.loads(paths["stage2_models"].read_text(encoding="utf-8"))
    if "temperature" not in stage2_models or "uncertainty_router" not in stage2_models:
        raise ValueError("Stage 2 model file lacks frozen temperature/router parameters")
    if "test" in str(paths["base"]).lower():
        raise ValueError("Stage 3 artifact path must not contain 'test'")
    result = {
        "status": "ready",
        "experiment": "Stage3_LLM_native_selective_role_memory",
        "version": STAGE3_VERSION,
        "split": "validation",
        "test_locked": True,
        "model_name": section["model_name"],
        "labels": labels,
        "planner_seeds": list(section.get("planner_seeds", [42, 43, 44])),
        "dialogues": len(dialogues),
        "targets": sum(len(dialogue.utterances) for dialogue in dialogues),
        "stage1_targets": len(stage1_records),
        "cuda_available": torch.cuda.is_available(),
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "prerequisites": {
            "stage1_records": str(paths["stage1_records"]),
            "stage2_models": str(paths["stage2_models"]),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run_plans(
    config_path: Path,
    run_name: str,
    max_dialogues: int | None,
    max_targets: int | None,
    overwrite: bool,
    seeds_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    dialogues = _load_validation(root, config)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]
    paths = _paths(root, config, run_name)
    seeds = list(seeds_override or section.get("planner_seeds", [42, 43, 44]))
    max_history = int(section.get("max_history_turns", 12))
    max_selected = int(section.get("max_selected_turns", 3))
    max_state_chars = int(section.get("max_state_characters", 240))
    planner = StructuredPlanner(section)
    manifest = {
        "experiment": "Stage3_LLM_native_selective_role_memory",
        "version": STAGE3_VERSION,
        "split": "validation",
        "test_locked": True,
        "config": str(config_path.resolve()),
        "run_name": run_name,
        "planner": planner.audit,
        "planner_seeds": seeds,
        "limits": {"max_dialogues": max_dialogues, "max_targets": max_targets},
    }
    _write_json(paths["manifest"], manifest)

    all_targets = [target for dialogue in dialogues for target in _targets(dialogue, max_history)]
    if max_targets is not None:
        all_targets = all_targets[:max_targets]
    counts: dict[str, int] = {}
    for seed in seeds:
        plan_path = _plan_path(paths, seed)
        if overwrite and plan_path.exists():
            plan_path.unlink()
        existing = _read_jsonl_dicts(plan_path)
        completed = {record["key"] for record in existing}
        pending = [target for target in all_targets if target.key not in completed]
        progress = tqdm(total=len(pending), desc=f"Stage 3 planner seed={seed}", unit="target")
        for start in range(0, len(pending), planner.batch_size):
            batch = pending[start : start + planner.batch_size]
            generated = planner.generate(
                [build_planner_prompt(target, max_selected) for target in batch], seed=seed
            )
            output: list[dict[str, Any]] = []
            for target, generation in zip(batch, generated):
                history_ids = list(_history_by_absolute_id(target))
                plan = normalize_plan(
                    _extract_json_object(generation["text"]),
                    allowed_roles=_seen_roles(target),
                    history_turn_ids=history_ids,
                    max_selected_turns=max_selected,
                    max_state_characters=max_state_chars,
                )
                output.append(
                    {
                        "key": target.key,
                        "dialogue_id": target.dialogue_id,
                        "turn_index": target.turn_index,
                        "utterance_id": target.utterance_id,
                        "speaker": target.speaker,
                        "history_count": len(target.history),
                        "plan": plan,
                        "raw_response": generation["text"],
                        "input_tokens": generation["input_tokens"],
                        "output_tokens": generation["output_tokens"],
                    }
                )
            _append_jsonl(plan_path, output)
            progress.update(len(output))
        progress.close()
        counts[str(seed)] = len(_read_jsonl_dicts(plan_path))
    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"status": "plans_complete", "counts": counts, "test_locked": True}
    print(json.dumps(result, indent=2))
    return result


def _prompt_roles(target: TargetSpec, selected: Sequence[tuple[int, Utterance]]) -> list[str]:
    roles = [target.speaker]
    for _, turn in selected:
        if turn.speaker not in roles:
            roles.append(turn.speaker)
    return roles


def run_scores(
    config_path: Path,
    run_name: str,
    max_dialogues: int | None,
    max_targets: int | None,
    overwrite: bool,
    seeds_override: Sequence[int] | None = None,
) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    labels = _labels(config)
    dialogues = _load_validation(root, config)
    if max_dialogues is not None:
        dialogues = dialogues[:max_dialogues]
    paths = _paths(root, config, run_name)
    stage1 = {record["key"]: record for record in _read_jsonl_dicts(paths["stage1_records"])}
    seeds = list(seeds_override or section.get("planner_seeds", [42, 43, 44]))
    max_history = int(section.get("max_history_turns", 12))
    max_entries = int(section.get("max_memory_entries_per_role", 3))
    max_roles = int(section.get("max_memory_roles", 3))
    preamble = str(
        section.get(
            "prediction_prompt_preamble",
            "You are a careful dialogue emotion classifier.",
        )
    )
    scorer = CausalLabelScorer(section, labels)
    manifest = (
        json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if paths["manifest"].exists()
        else {}
    )
    manifest["scorer"] = scorer.audit
    _write_json(paths["manifest"], manifest)
    counts: dict[str, int] = {}

    for seed in seeds:
        plans = {record["key"]: record for record in _read_jsonl_dicts(_plan_path(paths, seed))}
        score_path = _score_path(paths, seed)
        if overwrite and score_path.exists():
            score_path.unlink()
        existing_records = _read_jsonl_dicts(score_path)
        completed = {record["key"] for record in existing_records}
        processed = len(existing_records)
        stop = False
        progress = tqdm(dialogues, desc=f"Stage 3 scoring seed={seed}", unit="dialogue")
        for dialogue in progress:
            speaker_memory: dict[str, list[dict[str, Any]]] = {}
            role_memory: dict[str, list[dict[str, Any]]] = {}
            prompts: list[str] = []
            pending: list[dict[str, Any]] = []
            prompt_index: dict[str, int] = {}
            for target in _targets(dialogue, max_history):
                if (
                    target.key not in completed
                    and max_targets is not None
                    and processed + len(pending) >= max_targets
                ):
                    stop = True
                    break
                plan_record = plans.get(target.key)
                if plan_record is None:
                    raise ValueError(f"Missing planner output for {target.key} seed={seed}")
                plan = plan_record["plan"]
                if target.key not in completed:
                    base = stage1.get(target.key)
                    if base is None:
                        raise ValueError(f"Missing Stage 1 score record for {target.key}")
                    selected = _selected_turns(target, plan)
                    route_memory = plan["reasoning_mode"] == "memory" and bool(
                        target.history
                    )
                    roles = _prompt_roles(target, selected)
                    snapshots = {
                        "evidence_only": [],
                        "speaker_memory": _memory_snapshot(speaker_memory, roles, max_roles),
                        "role_memory": _memory_snapshot(role_memory, roles, max_roles),
                    }
                    branch_scores: dict[str, Any] = {}
                    branch_uses: dict[str, bool] = {}
                    prompt_words: dict[str, int] = {}
                    for branch in ("evidence_only", "speaker_memory", "role_memory"):
                        snapshot = snapshots[branch]
                        use = route_memory and bool(selected or snapshot)
                        branch_uses[branch] = use
                        if not use:
                            branch_scores[branch] = list(base["current"]["scores"])
                            prompt_words[branch] = 0
                            continue
                        prompt = build_prediction_prompt(
                            target, labels, selected, snapshot, preamble
                        )
                        if prompt not in prompt_index:
                            prompt_index[prompt] = len(prompts)
                            prompts.append(prompt)
                        branch_scores[branch] = {"prompt_index": prompt_index[prompt]}
                        prompt_words[branch] = len(prompt.split())
                    pending.append(
                        {
                            "target": target,
                            "base": base,
                            "plan": plan,
                            "branch_scores": branch_scores,
                            "branch_uses": branch_uses,
                            "prompt_words": prompt_words,
                            "selected_turn_ids": [turn_id for turn_id, _ in selected],
                            "speaker_memory_entries": sum(
                                len(item["entries"])
                                for item in snapshots["speaker_memory"]
                            ),
                            "role_memory_entries": sum(
                                len(item["entries"])
                                for item in snapshots["role_memory"]
                            ),
                            "planner_input_tokens": int(
                                plan_record.get("input_tokens", 0)
                            ),
                            "planner_output_tokens": int(
                                plan_record.get("output_tokens", 0)
                            ),
                        }
                    )
                apply_plan_to_memory(
                    speaker_memory,
                    plan,
                    source_turn_id=target.turn_index,
                    source_speaker=target.speaker,
                    speaker_only=True,
                    max_entries_per_role=max_entries,
                )
                apply_plan_to_memory(
                    role_memory,
                    plan,
                    source_turn_id=target.turn_index,
                    source_speaker=target.speaker,
                    speaker_only=False,
                    max_entries_per_role=max_entries,
                )
                if stop:
                    break

            prompt_scores = scorer.score_prompts(prompts)
            output: list[dict[str, Any]] = []
            for item in pending:
                resolved: dict[str, list[float]] = {}
                for branch, value in item["branch_scores"].items():
                    resolved[branch] = (
                        prompt_scores[value["prompt_index"]]
                        if isinstance(value, dict)
                        else value
                    )
                target = item["target"]
                output.append(
                    {
                        "key": target.key,
                        "dialogue_id": target.dialogue_id,
                        "turn_index": target.turn_index,
                        "speaker": target.speaker,
                        "gold": target.gold,
                        "history_count": len(target.history),
                        "current_scores": list(item["base"]["current"]["scores"]),
                        "full_scores": list(item["base"]["full"]["scores"]),
                        "branch_scores": resolved,
                        "branch_uses_context": item["branch_uses"],
                        "planner": item["plan"],
                        "selected_turn_ids": item["selected_turn_ids"],
                        "speaker_memory_entries": item["speaker_memory_entries"],
                        "role_memory_entries": item["role_memory_entries"],
                        "prediction_prompt_words": item["prompt_words"],
                        "planner_input_tokens": item["planner_input_tokens"],
                        "planner_output_tokens": item["planner_output_tokens"],
                    }
                )
            if output:
                _append_jsonl(score_path, output)
                processed += len(output)
                progress.set_postfix(targets=processed)
            if stop:
                break
        progress.close()
        counts[str(seed)] = len(_read_jsonl_dicts(score_path))
    del scorer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = {"status": "scores_complete", "counts": counts, "test_locked": True}
    print(json.dumps(result, indent=2))
    return result


def _entropy(probabilities: Sequence[float]) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    return float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))))


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()) if len(array) else None,
        "std": float(array.std(ddof=0)) if len(array) else None,
        "count": int(len(array)),
    }


def _calibrated_records(
    raw_records: Sequence[dict[str, Any]],
    labels: Sequence[str],
    temperature: float,
    entropy_threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw in raw_records:
        current = _calibrated_payload(raw["current_scores"], labels, temperature)
        full = _calibrated_payload(raw["full_scores"], labels, temperature)
        payloads = {
            "current_only": current,
            "always_full": full,
            "entropy_router": (
                full
                if raw["history_count"] > 0
                and _entropy(current["probabilities"]) / math.log(len(labels)) >= entropy_threshold
                else current
            ),
            "oracle_current_full": (
                full
                if _gold_nll(full, raw["gold"], labels) < _gold_nll(current, raw["gold"], labels)
                else current
            ),
        }
        for branch in ("evidence_only", "speaker_memory", "role_memory"):
            payloads[branch] = _calibrated_payload(
                raw["branch_scores"][branch], labels, temperature
            )
        record: dict[str, Any] = {
            "key": raw["key"],
            "dialogue_id": raw["dialogue_id"],
            "turn_index": raw["turn_index"],
            "speaker": raw["speaker"],
            "gold": raw["gold"],
            "history_count": raw["history_count"],
            "planner": raw["planner"],
            "selected_turn_ids": raw["selected_turn_ids"],
            "branch_uses_context": raw["branch_uses_context"],
            "speaker_memory_entries": raw["speaker_memory_entries"],
            "role_memory_entries": raw["role_memory_entries"],
        }
        for branch, payload in payloads.items():
            copied = dict(payload)
            copied["nll"] = _gold_nll(copied, raw["gold"], labels)
            record[branch] = copied
        output.append(record)
    return output


def _flip_stats(records: Sequence[dict[str, Any]], branch: str) -> dict[str, int]:
    good = bad = 0
    for record in records:
        current_correct = record["current_only"]["prediction"] == record["gold"]
        branch_correct = record[branch]["prediction"] == record["gold"]
        good += int(not current_correct and branch_correct)
        bad += int(current_correct and not branch_correct)
    return {"good": good, "bad": bad, "net": good - bad}


def _planner_diagnostics(
    records: Sequence[dict[str, Any]],
    stage1_by_key: dict[str, dict[str, Any]],
    utility_epsilon: float,
    deletion_epsilon: float,
) -> dict[str, Any]:
    eligible = [record for record in records if record["history_count"] > 0]
    parse_success = [bool(record["planner"]["parse_success"]) for record in records]
    context_scores: list[float] = []
    context_truth: list[int] = []
    evidence_scores: list[float] = []
    evidence_truth: list[int] = []
    non_speaker_updates = total_updates = 0
    for record in eligible:
        base = stage1_by_key[record["key"]]
        context_scores.append(float(record["planner"]["context_confidence"]))
        context_truth.append(int(float(base["context_utility_delta_nll"]) > utility_epsilon))
        selected = {
            item["turn_id"]: float(item["confidence"])
            for item in record["planner"].get("evidence", [])
        }
        start = int(record.get("turn_index", base["turn_index"])) - int(
            record["history_count"]
        )
        for deletion in base.get("deletions", []):
            turn_id = start + int(deletion["history_index"])
            evidence_scores.append(selected.get(turn_id, 0.0))
            evidence_truth.append(int(float(deletion["delta_nll"]) > deletion_epsilon))
        for update in record["planner"].get("updates", []):
            if update["action"] not in {"update", "forget"}:
                continue
            total_updates += 1
            non_speaker_updates += int(update["role"] != base["speaker"])
    route_ap = (
        float(average_precision_score(context_truth, context_scores))
        if len(set(context_truth)) == 2
        else None
    )
    evidence_ap = (
        float(average_precision_score(evidence_truth, evidence_scores))
        if len(set(evidence_truth)) == 2
        else None
    )
    return {
        "targets": len(records),
        "history_eligible": len(eligible),
        "parse_success_rate": float(np.mean(parse_success)) if parse_success else None,
        "memory_reasoning_rate": float(
            np.mean([record["planner"]["reasoning_mode"] == "memory" for record in eligible])
        )
        if eligible
        else None,
        "mean_selected_turns": float(
            np.mean([len(record["selected_turn_ids"]) for record in eligible])
        )
        if eligible
        else None,
        "mean_speaker_memory_entries": float(
            np.mean([record["speaker_memory_entries"] for record in records])
        ),
        "mean_role_memory_entries": float(
            np.mean([record["role_memory_entries"] for record in records])
        ),
        "mean_planner_input_tokens": float(
            np.mean([record.get("planner_input_tokens", 0) for record in records])
        ),
        "mean_planner_output_tokens": float(
            np.mean([record.get("planner_output_tokens", 0) for record in records])
        ),
        "mean_role_memory_prompt_words": float(
            np.mean(
                [
                    record.get("prediction_prompt_words", {}).get("role_memory", 0)
                    for record in records
                ]
            )
        ),
        "non_speaker_update_rate": non_speaker_updates / max(total_updates, 1),
        "updates": total_updates,
        "context_need_average_precision": route_ap,
        "evidence_average_precision": evidence_ap,
        "context_use_rate": float(
            np.mean([record["branch_uses_context"]["role_memory"] for record in eligible])
        )
        if eligible
        else None,
    }


def _aggregate(
    seed_reports: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    metric_names = ("weighted_f1", "macro_f1", "accuracy", "nll", "brier", "ece")
    variants: dict[str, Any] = {}
    for variant, label in VARIANT_LABELS.items():
        metrics = [report["metrics"][variant] for report in seed_reports]
        variants[variant] = {
            "label": label,
            **{name: _mean_std([item[name] for item in metrics]) for name in metric_names},
            "per_emotion_f1": {
                emotion: _mean_std([item["per_emotion_f1"][emotion] for item in metrics])
                for emotion in labels
            },
            "flips_vs_current": {
                name: _mean_std([report["flips"][variant][name] for report in seed_reports])
                for name in ("good", "bad", "net")
            }
            if variant != "current_only"
            else None,
        }
    diagnostic_names = (
        "parse_success_rate",
        "memory_reasoning_rate",
        "mean_selected_turns",
        "mean_speaker_memory_entries",
        "mean_role_memory_entries",
        "mean_planner_input_tokens",
        "mean_planner_output_tokens",
        "mean_role_memory_prompt_words",
        "non_speaker_update_rate",
        "context_need_average_precision",
        "evidence_average_precision",
        "context_use_rate",
    )
    diagnostics = {
        name: _mean_std(
            [
                report["diagnostics"][name]
                for report in seed_reports
                if report["diagnostics"][name] is not None
            ]
        )
        for name in diagnostic_names
    }
    return {"variants": variants, "planner_diagnostics": diagnostics}


def _bootstrap(
    seed_records: Sequence[list[dict[str, Any]]],
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    controls = ("entropy_router", "speaker_memory")
    dialogue_ids = sorted({record["dialogue_id"] for record in seed_records[0]})
    indexed: list[dict[str, list[dict[str, Any]]]] = []
    for records in seed_records:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[record["dialogue_id"]].append(record)
        indexed.append(grouped)
    rng = random.Random(seed)
    values = {
        control: {metric: [] for metric in ("weighted_f1", "macro_f1", "nll", "brier")}
        for control in controls
    }
    for _ in range(samples):
        sampled = [rng.choice(dialogue_ids) for _ in dialogue_ids]
        for control in controls:
            deltas = {metric: [] for metric in values[control]}
            for grouped in indexed:
                rows = [record for dialogue_id in sampled for record in grouped[dialogue_id]]
                proposed_metrics = _metric_bundle(rows, "role_memory", labels)
                control_metrics = _metric_bundle(rows, control, labels)
                for metric in deltas:
                    delta = (
                        control_metrics[metric] - proposed_metrics[metric]
                        if metric in {"nll", "brier"}
                        else proposed_metrics[metric] - control_metrics[metric]
                    )
                    deltas[metric].append(float(delta))
            for metric in values[control]:
                values[control][metric].append(float(np.mean(deltas[metric])))
    return {
        "method": "paired dialogue bootstrap averaged over fixed planner seeds",
        "proposed": "3E affected-role memory",
        "dialogues": len(dialogue_ids),
        "samples": samples,
        "seed": seed,
        "controls": {
            control: {
                metric: {
                    "mean": float(np.mean(items)),
                    "ci95": [float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))],
                    "probability_positive": float(np.mean(np.asarray(items) > 0.0)),
                }
                for metric, items in metrics.items()
            }
            for control, metrics in values.items()
        },
    }


def _format_optional(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Stage 3 — LLM-Native Selective Role Memory",
        "",
        "> Development-validation experiment. The MELD test split is inaccessible to this module.",
        "",
        "| Variant | wF1 | macro F1 | NLL | Brier | ECE | good/bad/net flips |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["variants"].values():
        flips = item.get("flips_vs_current")
        flip_text = "n/a"
        if flips:
            flip_text = (
                f"{flips['good']['mean']:.1f}/{flips['bad']['mean']:.1f}/{flips['net']['mean']:.1f}"
            )
        lines.append(
            f"| {item['label']} | {item['weighted_f1']['mean']:.4f} "
            f"± {item['weighted_f1']['std']:.4f} "
            f"| {item['macro_f1']['mean']:.4f} ± {item['macro_f1']['std']:.4f} "
            f"| {item['nll']['mean']:.4f} ± {item['nll']['std']:.4f} "
            f"| {item['brier']['mean']:.4f} ± {item['brier']['std']:.4f} "
            f"| {item['ece']['mean']:.4f} ± {item['ece']['std']:.4f} | {flip_text} |"
        )
    diagnostics = report["planner_diagnostics"]
    context_ap = _format_optional(
        diagnostics["context_need_average_precision"]["mean"], ".4f"
    )
    evidence_ap = _format_optional(
        diagnostics["evidence_average_precision"]["mean"], ".4f"
    )
    lines.extend(
        [
            "",
            "## LLM memory-controller diagnostics",
            "",
            f"- JSON parse success: **{diagnostics['parse_success_rate']['mean']:.1%}**",
            f"- Memory-reasoning rate: **{diagnostics['memory_reasoning_rate']['mean']:.1%}**",
            f"- Context-use rate: **{diagnostics['context_use_rate']['mean']:.1%}**",
            f"- Mean selected evidence turns: **{diagnostics['mean_selected_turns']['mean']:.3f}**",
            f"- Non-speaker update rate: **{diagnostics['non_speaker_update_rate']['mean']:.1%}**",
            f"- Context-need AP: **{context_ap}**",
            f"- Evidence AP: **{evidence_ap}**",
            f"- Mean planner input/output tokens: "
            f"**{diagnostics['mean_planner_input_tokens']['mean']:.1f} / "
            f"{diagnostics['mean_planner_output_tokens']['mean']:.1f}**",
            f"- Mean role-memory prediction prompt words: "
            f"**{diagnostics['mean_role_memory_prompt_words']['mean']:.1f}**",
            "",
            "## Frozen decision",
            "",
            f"- Status: **{report['decision']['status']}**",
            f"- Advance to cross-dataset replication: **{report['decision']['eligible']}**",
            f"- MELD test locked: **{report['test_locked']}**",
            "",
            "The speaker-memory and affected-role-memory variants share the same LLM plans "
            "and evidence. Their only difference is whether a turn may write to a "
            "non-speaking recipient's memory.",
            "",
            "## Validation per-emotion F1",
            "",
            "| Emotion | Current | Entropy router | Evidence | Speaker memory | Role memory |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for emotion in report["labels"]:
        values = [
            report["variants"][key]["per_emotion_f1"][emotion]["mean"]
            for key in (
                "current_only",
                "entropy_router",
                "evidence_only",
                "speaker_memory",
                "role_memory",
            )
        ]
        lines.append(f"| {emotion} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return "\n".join(lines) + "\n"


def summarize(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    labels = _labels(config)
    paths = _paths(root, config, run_name)
    stage1_records = _read_jsonl_dicts(paths["stage1_records"])
    stage1_by_key = {record["key"]: record for record in stage1_records}
    stage2_models = json.loads(paths["stage2_models"].read_text(encoding="utf-8"))
    temperature = float(stage2_models["temperature"]["temperature"])
    entropy_threshold = float(stage2_models["uncertainty_router"]["threshold"])
    seeds = list(section.get("planner_seeds", [42, 43, 44]))
    seed_reports: list[dict[str, Any]] = []
    seed_records: list[list[dict[str, Any]]] = []
    reference_keys: set[str] | None = None
    for seed in seeds:
        raw = _read_jsonl_dicts(_score_path(paths, seed))
        if not raw:
            raise ValueError(f"No Stage 3 scores found for planner seed {seed}")
        keys = {record["key"] for record in raw}
        if len(keys) != len(raw):
            raise ValueError(f"Duplicate Stage 3 score keys for planner seed {seed}")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"Planner seed {seed} does not cover the same target set")
        if run_name == "validation" and keys != set(stage1_by_key):
            raise ValueError(
                f"Incomplete Stage 3 validation scores for seed {seed}: "
                f"found={len(keys)}, expected={len(stage1_by_key)}"
            )
        records = _calibrated_records(raw, labels, temperature, entropy_threshold)
        seed_records.append(records)
        metrics = {variant: _metric_bundle(records, variant, labels) for variant in VARIANT_LABELS}
        flips = {
            variant: _flip_stats(records, variant)
            for variant in VARIANT_LABELS
            if variant != "current_only"
        }
        diagnostics = _planner_diagnostics(
            raw,
            stage1_by_key,
            utility_epsilon=float(section.get("utility_epsilon", 0.02)),
            deletion_epsilon=float(section.get("deletion_epsilon", 0.01)),
        )
        seed_reports.append(
            {"seed": seed, "metrics": metrics, "flips": flips, "diagnostics": diagnostics}
        )

    aggregate = _aggregate(seed_reports, labels)
    role = aggregate["variants"]["role_memory"]
    entropy = aggregate["variants"]["entropy_router"]
    speaker = aggregate["variants"]["speaker_memory"]
    diagnostics = aggregate["planner_diagnostics"]
    tolerance = float(section.get("classification_tolerance", 0.003))
    role_macro_seed = [
        report["metrics"]["role_memory"]["macro_f1"] for report in seed_reports
    ]
    speaker_macro_seed = [
        report["metrics"]["speaker_memory"]["macro_f1"] for report in seed_reports
    ]
    checks = {
        "macro_within_entropy_tolerance": role["macro_f1"]["mean"]
        >= entropy["macro_f1"]["mean"] - tolerance,
        "weighted_f1_within_entropy_tolerance": role["weighted_f1"]["mean"]
        >= entropy["weighted_f1"]["mean"] - tolerance,
        "nll_within_entropy_tolerance": role["nll"]["mean"]
        <= entropy["nll"]["mean"] + float(section.get("max_nll_increase", 0.02)),
        "role_memory_beats_speaker_memory": role["macro_f1"]["mean"] > speaker["macro_f1"]["mean"],
        "positive_role_memory_seeds": sum(
            a > b for a, b in zip(role_macro_seed, speaker_macro_seed)
        )
        >= int(section.get("minimum_positive_role_seeds", 2)),
        "planner_parse_success": diagnostics["parse_success_rate"]["mean"]
        >= float(section.get("minimum_parse_success", 0.95)),
        "bounded_context_use": diagnostics["context_use_rate"]["mean"]
        <= float(section.get("maximum_context_use_rate", 0.50)),
    }
    eligible = all(checks.values())
    decision = {
        "status": "advance_to_cross_dataset" if eligible else "stop_and_diagnose_memory",
        "eligible": eligible,
        "checks": checks,
        "selection_basis": "pre-registered MELD development-validation gate",
        "next_action": (
            "Freeze prompts and replicate on EmoryNLP before any test access."
            if eligible
            else "Keep MELD test locked; inspect parse failures, route decisions, "
            "and non-speaker writes."
        ),
        "test_locked": True,
    }
    bootstrap = _bootstrap(
        seed_records,
        labels,
        samples=int(section.get("bootstrap_samples", 2000)),
        seed=int(section.get("bootstrap_seed", 2030)),
    )
    report = {
        "experiment": "Stage3_LLM_native_selective_role_memory",
        "version": STAGE3_VERSION,
        "split": "validation",
        "test_locked": True,
        "labels": labels,
        "planner_seeds": seeds,
        "temperature": temperature,
        "entropy_threshold": entropy_threshold,
        "targets_per_seed": [len(records) for records in seed_records],
        **aggregate,
        "seed_reports": seed_reports,
        "paired_dialogue_bootstrap": bootstrap,
        "decision": decision,
    }
    _write_json(paths["report_json"], report)
    _write_json(paths["bootstrap"], bootstrap)
    _write_json(paths["decision"], decision)
    paths["report_md"].parent.mkdir(parents=True, exist_ok=True)
    markdown = _markdown(report)
    paths["report_md"].write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"JSON report: {paths['report_json']}")
    return report


def verify(config_path: Path, run_name: str) -> dict[str, Any]:
    root, config = _root_and_config(config_path)
    section = _section(config)
    paths = _paths(root, config, run_name)
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    seeds = list(section.get("planner_seeds", [42, 43, 44]))
    checks = {
        "validation_only": report.get("split") == "validation",
        "test_locked": report.get("test_locked") is True,
        "all_planner_seeds": report.get("planner_seeds") == seeds,
        "all_score_files": all(_score_path(paths, seed).exists() for seed in seeds),
        "all_plan_files": all(_plan_path(paths, seed).exists() for seed in seeds),
        "finite_metrics": all(
            math.isfinite(float(report["variants"][variant][metric]["mean"]))
            for variant in VARIANT_LABELS
            for metric in ("weighted_f1", "macro_f1", "nll", "brier", "ece")
        ),
        "no_test_path": "test" not in str(paths["base"]).lower(),
    }
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks}
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise RuntimeError("Stage 3 verification failed")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3 LLM-native selective role memory")
    parser.add_argument(
        "stage",
        choices=("check", "plan", "score", "summarize", "verify", "all-validation"),
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--max-dialogues", type=int)
    parser.add_argument("--max-targets", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "check":
        check(args.config, args.run_name)
    elif args.stage == "plan":
        run_plans(
            args.config,
            args.run_name,
            args.max_dialogues,
            args.max_targets,
            args.overwrite,
            args.seeds,
        )
    elif args.stage == "score":
        run_scores(
            args.config,
            args.run_name,
            args.max_dialogues,
            args.max_targets,
            args.overwrite,
            args.seeds,
        )
    elif args.stage == "summarize":
        summarize(args.config, args.run_name)
    elif args.stage == "verify":
        verify(args.config, args.run_name)
    else:
        check(args.config, args.run_name)
        run_plans(
            args.config,
            args.run_name,
            args.max_dialogues,
            args.max_targets,
            args.overwrite,
            args.seeds,
        )
        run_scores(
            args.config,
            args.run_name,
            args.max_dialogues,
            args.max_targets,
            args.overwrite,
            args.seeds,
        )
        summarize(args.config, args.run_name)
        verify(args.config, args.run_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
