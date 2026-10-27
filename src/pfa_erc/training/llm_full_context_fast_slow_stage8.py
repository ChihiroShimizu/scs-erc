"""Stage 8.1: full-context fast/slow thinking audit for MELD ERC.

Every condition receives the same complete *causal* dialogue stream: all turns
from the beginning of the dialogue through the target turn, and no future
turns.  The experiment changes only the response policy:

* FAST: directly emit the emotion label.
* SLOW: emit an explicit bounded reasoning trace, then the emotion label.
* ADAPTIVE: choose FAST or SLOW inside the same generation.

There is no context gate, no recent-k truncation, and no SKIP/READ action.  The
Stage 7 adapters are frozen.  This stage measures the counterfactual quality,
completion-token cost, and real wall-clock cost required before any SFT/RL
decision.  MELD test is deliberately inaccessible.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from pfa_erc.data import Dialogue, Utterance, read_jsonl


VERSION = "1.0.0"
EXPERIMENT = "llm_full_context_fast_slow_stage8"
LABELS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
MODES = ("fast", "slow", "adaptive")
PROTOCOL_VERSION = "full-causal-dialogue-fast-slow-v1"

COMMON_SYSTEM = (
    "You are a dialogue emotion classifier. You always receive the complete causal "
    "dialogue stream through the target utterance. Never use future turns or outside "
    "story knowledge. Follow the requested response protocol exactly."
)


@dataclass(frozen=True)
class FullContextTarget:
    dialogue_id: str
    turn_index: int
    utterance_id: str
    speaker: str
    text: str
    gold: str
    history: tuple[Utterance, ...]

    @property
    def key(self) -> str:
        return f"{self.dialogue_id}::{self.turn_index}"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            records.append(item)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON encode {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _project_root(config_path: Path) -> Path:
    path = config_path.expanduser().resolve()
    return path.parent.parent if path.parent.name == "configs" else Path.cwd().resolve()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _section(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get("llm_full_context_fast_slow_stage8")
    if not isinstance(value, dict):
        raise KeyError("Missing llm_full_context_fast_slow_stage8 configuration section")
    return value


def _paths(root: Path, section: dict[str, Any], run_name: str) -> dict[str, Path]:
    base = _resolve(root, section.get("artifact_dir", "artifacts/llm_full_context_fast_slow_stage8/meld"))
    if run_name != "validation":
        base = base / run_name
    return {
        "base": base,
        "predictions": base / "predictions",
        "reports": base / "reports",
        "manifest": base / "reports" / "lock_manifest.json",
        "preflight": base / "reports" / "preflight.json",
        "summary": base / "reports" / "validation_summary.json",
        "summary_md": base / "reports" / "validation_summary.md",
        "cost_csv": base / "reports" / "cost_quality.csv",
        "bootstrap": base / "reports" / "paired_dialogue_bootstrap.json",
        "decision": base / "reports" / "frozen_decision.json",
    }


def _adapter_dir(root: Path, section: dict[str, Any], seed: int, variant: str) -> Path:
    adapter_root = _resolve(
        root,
        section.get(
            "stage7_adapter_root",
            "artifacts/llm_evidence_supervision_stage7/meld/adapters",
        ),
    )
    return adapter_root / f"seed{seed}" / variant


def build_full_context_targets(dialogues: Sequence[Dialogue]) -> list[FullContextTarget]:
    """Build targets with all earlier turns; never a recent-k slice."""
    result: list[FullContextTarget] = []
    for dialogue in dialogues:
        for turn_index, turn in enumerate(dialogue.utterances):
            result.append(
                FullContextTarget(
                    dialogue_id=str(dialogue.dialogue_id),
                    turn_index=turn_index,
                    utterance_id=str(turn.utterance_id),
                    speaker=str(turn.speaker),
                    text=str(turn.text),
                    gold=str(turn.emotion),
                    history=tuple(dialogue.utterances[:turn_index]),
                )
            )
    if len({target.key for target in result}) != len(result):
        raise ValueError("Duplicate dialogue/turn keys in processed MELD data")
    return result


def render_causal_dialogue(target: FullContextTarget) -> str:
    lines = [
        f"[{index}] {turn.speaker}: {turn.text}"
        for index, turn in enumerate(target.history)
    ]
    lines.append(f"[TARGET {target.turn_index}] {target.speaker}: {target.text}")
    return "\n".join(lines)


def build_full_context_prompt(
    target: FullContextTarget,
    labels: Sequence[str],
    mode: str,
    slow_reasoning_word_limit: int = 80,
) -> str:
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode}; expected one of {MODES}")
    dialogue = render_causal_dialogue(target)
    common = (
        "Task: classify the target speaker's expressed emotion. The complete causal "
        "dialogue stream is already supplied; do not request, omit, select, or retrieve "
        "context.\n\n"
        f"Complete causal dialogue stream:\n{dialogue}\n\n"
        f"Allowed labels: {', '.join(labels)}.\n"
    )
    if mode == "fast":
        policy = (
            "Use FAST thinking: decide directly and do not provide reasoning. Return exactly:\n"
            "<FAST>\n<EMOTION>label</EMOTION>"
        )
    elif mode == "slow":
        policy = (
            "Use SLOW thinking: explicitly analyze speaker intent, relevant earlier events, "
            "emotion transition, and possible ambiguity. Keep the reasoning under "
            f"{slow_reasoning_word_limit} words. Put the final emotion last. Return exactly:\n"
            "<SLOW>\n<REASONING>reasoning</REASONING>\n<EMOTION>label</EMOTION>"
        )
    else:
        policy = (
            "Choose the reasoning strategy yourself. Choose FAST when the emotion is clear "
            "without multi-step deliberation. Choose SLOW only when explicit deliberation is "
            "likely to change or disambiguate the emotion label. Seeing earlier turns is not "
            "itself a reason to choose SLOW. Return exactly one of:\n"
            "<FAST>\n<EMOTION>label</EMOTION>\n\n"
            "or\n\n"
            "<SLOW>\n"
            f"<REASONING>reasoning under {slow_reasoning_word_limit} words</REASONING>\n"
            "<EMOTION>label</EMOTION>"
        )
    return common + policy


def _extract_emotion(text: str, labels: Sequence[str]) -> tuple[str | None, bool]:
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    tagged = re.search(
        rf"<EMOTION>\s*({label_pattern})\s*</EMOTION>",
        text,
        flags=re.IGNORECASE,
    )
    if tagged:
        return tagged.group(1).lower(), True
    json_style = re.search(
        rf'["\']emotion["\']\s*:\s*["\']({label_pattern})["\']',
        text,
        flags=re.IGNORECASE,
    )
    if json_style:
        return json_style.group(1).lower(), False
    standalone = re.findall(rf"\b({label_pattern})\b", text, flags=re.IGNORECASE)
    if standalone:
        return standalone[-1].lower(), False
    return None, False


def response_prefix_through_emotion(text: str, labels: Sequence[str]) -> str | None:
    """Return the generated prefix ending at the emotion label token text."""
    label_pattern = "|".join(re.escape(label) for label in sorted(labels, key=len, reverse=True))
    tagged = re.search(
        rf"<EMOTION>\s*({label_pattern})",
        text,
        flags=re.IGNORECASE,
    )
    if tagged:
        return text[: tagged.end()]
    json_style = re.search(
        rf'["\']emotion["\']\s*:\s*["\']({label_pattern})',
        text,
        flags=re.IGNORECASE,
    )
    if json_style:
        return text[: json_style.end()]
    standalone = list(re.finditer(rf"\b({label_pattern})\b", text, flags=re.IGNORECASE))
    return text[: standalone[-1].end()] if standalone else None


def parse_response(text: str, labels: Sequence[str], expected_mode: str) -> dict[str, Any]:
    emotion, emotion_tag_valid = _extract_emotion(text, labels)
    has_fast = bool(re.search(r"<FAST>", text, flags=re.IGNORECASE))
    has_slow = bool(re.search(r"<SLOW>", text, flags=re.IGNORECASE))
    strategy: str | None
    if has_fast and not has_slow:
        strategy = "fast"
    elif has_slow and not has_fast:
        strategy = "slow"
    else:
        strategy = None
    reasoning_match = re.search(
        r"<REASONING>(.*?)</REASONING>", text, flags=re.IGNORECASE | re.DOTALL
    )
    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    strategy_valid = strategy is not None and (
        expected_mode == "adaptive" or strategy == expected_mode
    )
    reasoning_valid = (
        (strategy == "slow" and bool(reasoning_match))
        or (strategy == "fast" and not reasoning)
    )
    return {
        "prediction": emotion,
        "strategy": strategy,
        "reasoning": reasoning,
        "emotion_tag_valid": emotion_tag_valid,
        "strategy_tag_valid": strategy_valid,
        "format_valid": bool(emotion_tag_valid and strategy_valid and reasoning_valid),
    }


def audit_response_semantics(
    text: str,
    labels: Sequence[str],
    expected_mode: str,
) -> dict[str, Any]:
    """Separate semantic slow thinking from exact response-schema compliance.

    Stage 8.1 originally reported an exact ``<SLOW>`` tag rate as "Slow rate".
    The frozen raw outputs show that a response can contain a substantive
    rationale while missing or misspelling one XML wrapper.  This audit keeps
    both facts instead of treating malformed XML as absence of reasoning.
    """
    parsed = parse_response(text, labels, expected_mode)
    exact_fast = bool(re.search(r"<\s*FAST\s*>", text, flags=re.IGNORECASE))
    exact_slow = bool(re.search(r"<\s*SLOW\s*>", text, flags=re.IGNORECASE))
    tolerant_slow = bool(
        re.search(r"<\s*(?:SLOW|SLW)\s*>", text, flags=re.IGNORECASE)
    )
    reasoning_tag_open = bool(
        re.search(r"<\s*REASONING\s*>", text, flags=re.IGNORECASE)
    )
    reasoning_tag_close = bool(
        re.search(r"</\s*REASONING\s*>", text, flags=re.IGNORECASE)
    )

    # Strip the emotion field, then inspect material on either side of it. Some
    # malformed SLOW outputs put the emotion first and the rationale afterwards.
    label_pattern = "|".join(re.escape(label) for label in labels)
    rationale_material = re.sub(
        rf"<\s*EMOTION\s*>\s*(?:{label_pattern})\s*</\s*EMOTION\s*>",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    rationale_material = re.sub(
        r"</?\s*(?:FAST|SLOW|SLW|REASONING)\s*>",
        " ",
        rationale_material,
        flags=re.IGNORECASE,
    )
    rationale_material = re.sub(
        rf"(?:the\s+)?(?:final\s+)?emotion\s*(?:is|:)\s*(?:{label_pattern})",
        " ",
        rationale_material,
        flags=re.IGNORECASE,
    )
    lexical_words = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", rationale_material
    )
    reasoning_present = bool(reasoning_tag_open or len(lexical_words) >= 4)
    emotion_start = re.search(r"<\s*EMOTION\s*>", text, flags=re.IGNORECASE)
    material_before_emotion = text[: emotion_start.start()] if emotion_start else text
    material_before_emotion = re.sub(
        r"</?\s*(?:FAST|SLOW|SLW|REASONING)\s*>",
        " ",
        material_before_emotion,
        flags=re.IGNORECASE,
    )
    material_before_emotion = re.sub(
        rf"(?:the\s+)?(?:final\s+)?emotion\s*(?:is|:)\s*(?:{label_pattern})",
        " ",
        material_before_emotion,
        flags=re.IGNORECASE,
    )
    before_words = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u4e00-\u9fff]", material_before_emotion
    )
    reasoning_open_match = re.search(
        r"<\s*REASONING\s*>", text, flags=re.IGNORECASE
    )
    reasoning_tag_before_emotion = bool(
        reasoning_open_match
        and (emotion_start is None or reasoning_open_match.start() < emotion_start.start())
    )
    reasoning_before_emotion = bool(
        reasoning_tag_before_emotion or len(before_words) >= 4
    )
    if reasoning_present or tolerant_slow:
        semantic_strategy: str | None = "slow"
    elif exact_fast:
        semantic_strategy = "fast"
    else:
        semantic_strategy = parsed["strategy"]
    expected_tag_success = (
        (expected_mode == "fast" and exact_fast and not exact_slow)
        or (expected_mode == "slow" and exact_slow and not exact_fast)
        or (expected_mode == "adaptive" and exact_fast != exact_slow)
    )
    return {
        **parsed,
        "strict_format_valid": bool(parsed["format_valid"]),
        "exact_fast_tag": exact_fast,
        "exact_slow_tag": exact_slow,
        "tolerant_slow_tag": tolerant_slow,
        "strategy_tag_success": bool(expected_tag_success),
        "reasoning_tag_open": reasoning_tag_open,
        "reasoning_tag_close": reasoning_tag_close,
        "reasoning_present": reasoning_present,
        "reasoning_before_emotion": reasoning_before_emotion,
        "emotion_before_reasoning": bool(
            reasoning_present and not reasoning_before_emotion
        ),
        "semantic_strategy": semantic_strategy,
        "direct_label_response": bool(parsed["prediction"] and not reasoning_present),
    }


def _apply_chat_template(tokenizer: Any, messages: Sequence[dict[str, str]], enable_thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )


def _render_prompt(tokenizer: Any, user_prompt: str, enable_thinking: bool) -> str:
    messages = [
        {"role": "system", "content": COMMON_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return _apply_chat_template(tokenizer, messages, enable_thinking)
    return f"system: {COMMON_SYSTEM}\nuser: {user_prompt}\nassistant:"


def _completion_ids(row: Sequence[int], eos_id: int | None, pad_id: int | None) -> list[int]:
    result: list[int] = []
    for token in row:
        value = int(token)
        if eos_id is not None and value == int(eos_id):
            break
        if pad_id is not None and eos_id != pad_id and value == int(pad_id):
            break
        result.append(value)
    return result


class FullContextGenerationEngine:
    def __init__(self, section: dict[str, Any], adapter_path: Path):
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Stage 8 requires transformers, accelerate and peft; run the setup script"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("Stage 8 generation requires CUDA")
        if not (adapter_path / "adapter_config.json").exists():
            raise FileNotFoundError(f"Missing frozen Stage 7 adapter: {adapter_path}")
        self.section = section
        self.adapter_path = adapter_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            section["model_name"],
            trust_remote_code=bool(section.get("trust_remote_code", False)),
            local_files_only=bool(section.get("local_files_only", False)),
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
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
        self.model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
        self.model.to("cuda")
        self.model.eval()
        self.model.config.use_cache = True
        self.max_prompt_tokens = int(section.get("max_prompt_tokens", 4096))

    def render(self, target: FullContextTarget, labels: Sequence[str], mode: str) -> str:
        prompt = build_full_context_prompt(
            target,
            labels,
            mode,
            slow_reasoning_word_limit=int(self.section.get("slow_reasoning_word_limit", 80)),
        )
        return _render_prompt(
            self.tokenizer,
            prompt,
            enable_thinking=bool(self.section.get("enable_thinking", False)),
        )

    def generate_batch(
        self, rendered_prompts: Sequence[str], max_new_tokens: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        encoded = self.tokenizer(
            list(rendered_prompts),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
        too_long = [int(value) for value in prompt_lengths if int(value) > self.max_prompt_tokens]
        if too_long:
            raise ValueError(
                "Complete dialogue prompt exceeds max_prompt_tokens; Stage 8 refuses to "
                f"truncate full context. Largest observed={max(too_long)}, "
                f"limit={self.max_prompt_tokens}. Increase max_prompt_tokens explicitly."
            )
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                use_cache=True,
                pad_token_id=int(self.tokenizer.pad_token_id),
                eos_token_id=int(self.tokenizer.eos_token_id),
            )
        torch.cuda.synchronize()
        elapsed_seconds = time.perf_counter() - started
        peak_memory = int(torch.cuda.max_memory_allocated())
        outputs: list[dict[str, Any]] = []
        for prompt_tokens, row in zip(prompt_lengths, generated[:, input_width:].tolist()):
            ids = _completion_ids(
                row,
                self.tokenizer.eos_token_id,
                self.tokenizer.pad_token_id,
            )
            outputs.append(
                {
                    "response": self.tokenizer.decode(ids, skip_special_tokens=True).strip(),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": len(ids),
                }
            )
        batch = len(rendered_prompts)
        timing = {
            "batch_size": batch,
            "batch_latency_ms": elapsed_seconds * 1000.0,
            "amortized_latency_ms": elapsed_seconds * 1000.0 / max(1, batch),
            "completion_tokens_per_second": (
                sum(item["completion_tokens"] for item in outputs) / elapsed_seconds
                if elapsed_seconds > 0
                else None
            ),
            "peak_cuda_memory_bytes": peak_memory,
        }
        del encoded, generated
        return outputs, timing


def _prediction_path(paths: dict[str, Path], seed: int, adapter: str, mode: str) -> Path:
    return paths["predictions"] / f"seed{seed}_{adapter}_{mode}.jsonl"


def _max_new_tokens(section: dict[str, Any], mode: str) -> int:
    defaults = {"fast": 24, "slow": 160, "adaptive": 160}
    configured = section.get("max_new_tokens", {})
    return int(configured.get(mode, defaults[mode]))


def _validate_cached_record(
    record: dict[str, Any], seed: int, adapter: str, mode: str
) -> None:
    expected = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "seed": seed,
        "adapter_variant": adapter,
        "mode": mode,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(
                f"Cached Stage 8 prediction has {key}={record.get(key)!r}, "
                f"expected {value!r}; rerun with --overwrite"
            )


def classification_and_cost_metrics(
    records: Sequence[dict[str, Any]], labels: Sequence[str]
) -> dict[str, Any]:
    if not records:
        raise ValueError("Cannot score an empty prediction set")
    gold = [str(record["gold"]) for record in records]
    predicted = [
        str(record.get("prediction"))
        if record.get("prediction") in labels
        else "__invalid__"
        for record in records
    ]
    completion = np.asarray(
        [float(record.get("completion_tokens", 0)) for record in records], dtype=np.float64
    )
    to_emotion = np.asarray(
        [
            float(
                record.get("tokens_to_emotion")
                if record.get("tokens_to_emotion") is not None
                else record.get("completion_tokens", 0)
            )
            for record in records
        ],
        dtype=np.float64,
    )
    prompt = np.asarray(
        [float(record.get("prompt_tokens", 0)) for record in records], dtype=np.float64
    )
    dialogue_tokens = np.asarray(
        [float(record.get("dialogue_tokens", 0)) for record in records], dtype=np.float64
    )
    latency = np.asarray(
        [float(record.get("amortized_latency_ms", 0.0)) for record in records],
        dtype=np.float64,
    )
    audits = [
        audit_response_semantics(
            str(record.get("response", "")),
            labels,
            str(record.get("mode"))
            if str(record.get("mode")) in MODES
            else "adaptive",
        )
        for record in records
    ]
    strategies = [audit["strategy"] for audit in audits]
    per_emotion = {
        label: float(
            f1_score(gold, predicted, labels=[label], average="macro", zero_division=0)
        )
        for label in labels
    }
    return {
        "records": len(records),
        "weighted_f1": float(
            f1_score(gold, predicted, labels=list(labels), average="weighted", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(gold, predicted, labels=list(labels), average="macro", zero_division=0)
        ),
        "accuracy": float(accuracy_score(gold, predicted)),
        "parse_success_rate": float(np.mean([value != "__invalid__" for value in predicted])),
        # Backward-compatible alias: this remains the strict XML/schema rate.
        "format_success_rate": float(
            np.mean([bool(audit["strict_format_valid"]) for audit in audits])
        ),
        "strict_format_success_rate": float(
            np.mean([bool(audit["strict_format_valid"]) for audit in audits])
        ),
        "emotion_tag_success_rate": float(
            np.mean([bool(audit["emotion_tag_valid"]) for audit in audits])
        ),
        "strategy_tag_success_rate": float(
            np.mean([bool(audit["strategy_tag_success"]) for audit in audits])
        ),
        "slow_strategy_rate": float(np.mean([value == "slow" for value in strategies])),
        "semantic_slow_rate": float(
            np.mean([audit["semantic_strategy"] == "slow" for audit in audits])
        ),
        "reasoning_present_rate": float(
            np.mean([bool(audit["reasoning_present"]) for audit in audits])
        ),
        "reasoning_before_emotion_rate": float(
            np.mean([bool(audit["reasoning_before_emotion"]) for audit in audits])
        ),
        "emotion_before_reasoning_rate": float(
            np.mean([bool(audit["emotion_before_reasoning"]) for audit in audits])
        ),
        "direct_label_response_rate": float(
            np.mean([bool(audit["direct_label_response"]) for audit in audits])
        ),
        "mean_dialogue_tokens": float(dialogue_tokens.mean()),
        "mean_prompt_tokens": float(prompt.mean()),
        "mean_completion_tokens": float(completion.mean()),
        "p50_completion_tokens": float(np.quantile(completion, 0.50)),
        "p95_completion_tokens": float(np.quantile(completion, 0.95)),
        "mean_tokens_to_emotion": float(to_emotion.mean()),
        "p50_tokens_to_emotion": float(np.quantile(to_emotion, 0.50)),
        "p95_tokens_to_emotion": float(np.quantile(to_emotion, 0.95)),
        "mean_total_tokens": float((prompt + completion).mean()),
        "mean_amortized_latency_ms": float(latency.mean()),
        "p50_amortized_latency_ms": float(np.quantile(latency, 0.50)),
        "p95_amortized_latency_ms": float(np.quantile(latency, 0.95)),
        "per_emotion_f1": per_emotion,
    }


def _load_validation_targets(root: Path, section: dict[str, Any]) -> list[FullContextTarget]:
    processed = _resolve(root, section["processed_dir"])
    validation_path = processed / "validation.jsonl"
    if "test" in validation_path.name.lower() or not validation_path.exists():
        raise FileNotFoundError(f"Missing safe MELD validation file: {validation_path}")
    return build_full_context_targets(read_jsonl(validation_path))


def _run_mode(
    engine: FullContextGenerationEngine,
    targets: Sequence[FullContextTarget],
    labels: Sequence[str],
    mode: str,
    seed: int,
    adapter: str,
    output_path: Path,
    section: dict[str, Any],
    *,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if overwrite and output_path.exists():
        output_path.unlink()
    cached = _read_jsonl(output_path)
    by_key: dict[str, dict[str, Any]] = {}
    for record in cached:
        _validate_cached_record(record, seed, adapter, mode)
        key = str(record["key"])
        if key in by_key:
            raise ValueError(f"Duplicate cached Stage 8 key: {key}")
        by_key[key] = record
    target_keys = {target.key for target in targets}
    extras = set(by_key) - target_keys
    if extras:
        raise ValueError(f"Cached predictions contain keys outside this run: {sorted(extras)[:3]}")
    for target in targets:
        if target.key not in by_key:
            continue
        expected_prompt_hash = _sha256_bytes(engine.render(target, labels, mode))
        if by_key[target.key].get("prompt_sha256") != expected_prompt_hash:
            raise ValueError(
                f"Cached prompt protocol differs for {target.key}; rerun with --overwrite"
            )
    missing = [target for target in targets if target.key not in by_key]
    batch_size = int(section.get("evaluation_batch_size", 4))
    log_every_batches = max(1, int(section.get("log_every_batches", 25)))
    for start in range(0, len(missing), batch_size):
        batch_targets = missing[start : start + batch_size]
        rendered = [engine.render(target, labels, mode) for target in batch_targets]
        outputs, timing = engine.generate_batch(rendered, _max_new_tokens(section, mode))
        new_records: list[dict[str, Any]] = []
        for target, rendered_prompt, generated in zip(batch_targets, rendered, outputs):
            parsed = audit_response_semantics(generated["response"], labels, mode)
            prefix = response_prefix_through_emotion(generated["response"], labels)
            tokens_to_emotion = (
                len(engine.tokenizer.encode(prefix, add_special_tokens=False))
                if prefix is not None
                else None
            )
            dialogue_text = render_causal_dialogue(target)
            record = {
                "experiment": EXPERIMENT,
                "version": VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "seed": seed,
                "adapter_variant": adapter,
                "mode": mode,
                "split": "validation",
                "key": target.key,
                "dialogue_id": target.dialogue_id,
                "turn_index": target.turn_index,
                "utterance_id": target.utterance_id,
                "gold": target.gold,
                "causal_turn_count": len(target.history) + 1,
                "history_turn_count": len(target.history),
                "dialogue_sha256": _sha256_bytes(dialogue_text),
                "prompt_sha256": _sha256_bytes(rendered_prompt),
                "dialogue_tokens": len(
                    engine.tokenizer.encode(dialogue_text, add_special_tokens=False)
                ),
                **generated,
                "tokens_to_emotion": tokens_to_emotion,
                **parsed,
                **timing,
                "test_data_read": False,
            }
            new_records.append(record)
            by_key[target.key] = record
        _append_jsonl(output_path, new_records)
        batch_number = start // batch_size + 1
        total_batches = max(1, (len(missing) + batch_size - 1) // batch_size)
        if batch_number == 1 or batch_number == total_batches or batch_number % log_every_batches == 0:
            print(
                f"[Stage8] seed={seed} adapter={adapter} mode={mode} "
                f"completed={min(start + len(batch_targets), len(missing))}/{len(missing)} "
                f"cached={len(cached)}",
                flush=True,
            )
    ordered = [by_key[target.key] for target in targets]
    _write_jsonl(output_path, ordered)
    return ordered, classification_and_cost_metrics(ordered, labels)


def _aligned_records(
    left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    left_map = {str(record["key"]): record for record in left}
    right_map = {str(record["key"]): record for record in right}
    if set(left_map) != set(right_map):
        raise ValueError("Stage 8 variants do not contain identical target keys")
    keys = sorted(left_map)
    for key in keys:
        if left_map[key]["gold"] != right_map[key]["gold"]:
            raise ValueError(f"Gold mismatch for {key}")
        if left_map[key]["dialogue_sha256"] != right_map[key]["dialogue_sha256"]:
            raise ValueError(
                f"Full-context invariant failed for {key}: dialogue differs between modes"
            )
    return [left_map[key] for key in keys], [right_map[key] for key in keys]


def random_matched_records(
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
    adaptive: Sequence[dict[str, Any]],
    random_seed: int,
) -> list[dict[str, Any]]:
    fast_rows, slow_rows = _aligned_records(fast, slow)
    _, adaptive_rows = _aligned_records(fast_rows, adaptive)
    slow_count = sum(record.get("strategy") == "slow" for record in adaptive_rows)
    order = sorted(
        range(len(fast_rows)),
        key=lambda index: (_stable_hash(str(fast_rows[index]["key"]), random_seed), index),
    )
    selected = set(order[:slow_count])
    result: list[dict[str, Any]] = []
    for index, (fast_record, slow_record) in enumerate(zip(fast_rows, slow_rows)):
        source = slow_record if index in selected else fast_record
        result.append({**source, "mode": "random_matched", "strategy": "slow" if index in selected else "fast"})
    return result


def oracle_budget_frontier(
    fast: Sequence[dict[str, Any]],
    slow: Sequence[dict[str, Any]],
    labels: Sequence[str],
    budgets: Sequence[float],
    stable_seed: int,
) -> list[dict[str, Any]]:
    fast_rows, slow_rows = _aligned_records(fast, slow)
    candidates: list[tuple[int, float, int, int]] = []
    for index, (fast_record, slow_record) in enumerate(zip(fast_rows, slow_rows)):
        fast_correct = int(fast_record.get("prediction") == fast_record["gold"])
        slow_correct = int(slow_record.get("prediction") == slow_record["gold"])
        gain = slow_correct - fast_correct
        extra_tokens = float(slow_record["completion_tokens"]) - float(fast_record["completion_tokens"])
        candidates.append(
            (
                -gain,
                extra_tokens,
                _stable_hash(str(fast_record["key"]), stable_seed),
                index,
            )
        )
    ranked_positive = [item[3] for item in sorted(candidates) if item[0] < 0]
    frontier: list[dict[str, Any]] = []
    for budget in budgets:
        if not 0.0 <= float(budget) <= 1.0:
            raise ValueError(f"Oracle slow budget must be within [0,1], got {budget}")
        capacity = int(np.floor(float(budget) * len(fast_rows) + 1e-12))
        selected = set(ranked_positive[:capacity])
        mixed: list[dict[str, Any]] = []
        for index, (fast_record, slow_record) in enumerate(zip(fast_rows, slow_rows)):
            source = slow_record if index in selected else fast_record
            mixed.append({**source, "mode": "oracle_at_most_budget"})
        metrics = classification_and_cost_metrics(mixed, labels)
        frontier.append(
            {
                "budget": float(budget),
                "actual_slow_rate": len(selected) / len(fast_rows),
                **metrics,
            }
        )
    return frontier


def _derived_comparisons(
    records: dict[str, list[dict[str, Any]]],
    labels: Sequence[str],
    section: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    random_rows = random_matched_records(
        records["fast"],
        records["slow"],
        records["adaptive"],
        int(section.get("matched_random_seed", 8128)) + seed,
    )
    frontier = oracle_budget_frontier(
        records["fast"],
        records["slow"],
        labels,
        [float(value) for value in section.get("oracle_slow_budgets", [0, 0.1, 0.25, 0.5, 0.75, 1.0])],
        int(section.get("oracle_tie_seed", 8129)) + seed,
    )
    return {
        "random_matched": classification_and_cost_metrics(random_rows, labels),
        "oracle_frontier": frontier,
        "full_context_invariant_verified": True,
        "oracle_uses_validation_gold_for_upper_bound_only": True,
    }


def preflight(
    config_path: Path,
    run_name: str = "validation",
    seeds: Sequence[int] | None = None,
) -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    if str(section.get("dataset", "")).lower() != "meld":
        raise ValueError("Stage 8.1 currently accepts MELD only")
    if section.get("context_policy") != "full_causal_dialogue":
        raise ValueError("context_policy must be exactly full_causal_dialogue")
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    processed = _resolve(root, section["processed_dir"])
    train_path = processed / "train.jsonl"
    validation_path = processed / "validation.jsonl"
    for path in (train_path, validation_path):
        if not path.exists():
            raise FileNotFoundError(path)
        if "test" in path.name.lower():
            raise ValueError(f"Unsafe split path: {path}")
    validation_dialogues = read_jsonl(validation_path)
    targets = build_full_context_targets(validation_dialogues)
    requested_seeds = list(seeds or section.get("seeds", [42, 43, 44]))
    variants = [str(value) for value in section.get("adapter_variants", ["full_supervised"])]
    missing_adapters: list[str] = []
    adapter_hashes: dict[str, str] = {}
    for seed in requested_seeds:
        for variant in variants:
            adapter = _adapter_dir(root, section, int(seed), variant)
            config_file = adapter / "adapter_config.json"
            if not config_file.exists():
                missing_adapters.append(str(adapter))
            else:
                adapter_hashes[f"seed{seed}:{variant}"] = _sha256_file(config_file)
    if missing_adapters:
        raise FileNotFoundError(
            "Missing frozen Stage 7 adapters:\n" + "\n".join(missing_adapters)
        )
    manifest = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "MELD",
        "split": "validation",
        "context_policy": "full_causal_dialogue",
        "history_window": "all_causal_turns",
        "future_turns_visible": False,
        "context_gate_used": False,
        "context_truncation_used": False,
        "teacher_annotations_visible_at_inference": False,
        "stage7_adapters_frozen": True,
        "seeds": requested_seeds,
        "adapter_variants": variants,
        "validation_dialogues": len(validation_dialogues),
        "validation_targets": len(targets),
        "max_causal_turns": max(len(target.history) + 1 for target in targets),
        "config_sha256": _sha256_file(config_path),
        "validation_sha256": _sha256_file(validation_path),
        "adapter_config_sha256": adapter_hashes,
        "test_locked": True,
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["preflight"], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def evaluate_seed(
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
    if not paths["manifest"].exists():
        preflight(config_path, run_name, [seed])
    targets = _load_validation_targets(root, section)
    if max_targets is not None:
        targets = targets[: int(max_targets)]
    labels = [str(value) for value in section.get("labels", LABELS)]
    variants = [str(value) for value in section.get("adapter_variants", ["full_supervised"])]
    result: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": int(seed),
        "split": "validation",
        "targets": len(targets),
        "context_policy": "full_causal_dialogue",
        "test_locked": True,
        "adapters": {},
    }
    for adapter_variant in variants:
        adapter_path = _adapter_dir(root, section, seed, adapter_variant)
        engine = FullContextGenerationEngine(section, adapter_path)
        adapter_records: dict[str, list[dict[str, Any]]] = {}
        mode_metrics: dict[str, Any] = {}
        for mode in MODES:
            rows, metrics = _run_mode(
                engine,
                targets,
                labels,
                mode,
                seed,
                adapter_variant,
                _prediction_path(paths, seed, adapter_variant, mode),
                section,
                overwrite=overwrite,
            )
            adapter_records[mode] = rows
            mode_metrics[mode] = metrics
        result["adapters"][adapter_variant] = {
            "modes": mode_metrics,
            "derived": _derived_comparisons(
                adapter_records, labels, section, seed
            ),
        }
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    output = paths["reports"] / f"seed{seed}_evaluation.json"
    _write_json(output, result)
    print(f"Stage 8.1 seed {seed} evaluation written to {output}")
    return result


def _runtime_subset(
    targets: Sequence[FullContextTarget], count: int, seed: int
) -> list[FullContextTarget]:
    ordered = sorted(targets, key=lambda target: (_stable_hash(target.key, seed), target.key))
    return ordered[: min(len(ordered), count)]


def runtime_profile_seed(
    config_path: Path,
    seed: int,
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
    output = paths["reports"] / f"runtime_profile_seed{seed}.json"
    if output.exists() and not overwrite:
        value = json.loads(output.read_text(encoding="utf-8"))
        print(json.dumps(value, ensure_ascii=False, indent=2))
        return value
    all_targets = _load_validation_targets(root, section)
    requested = int(max_targets or section.get("runtime_profile_targets", 128))
    targets = _runtime_subset(
        all_targets, requested, int(section.get("runtime_subset_seed", 8130))
    )
    labels = [str(value) for value in section.get("labels", LABELS)]
    adapter_variant = str(section.get("runtime_adapter_variant", "full_supervised"))
    engine = FullContextGenerationEngine(
        section, _adapter_dir(root, section, seed, adapter_variant)
    )
    repeats = int(repeats_override or section.get("runtime_repeats", 3))
    warmup_batches = int(section.get("runtime_warmup_batches", 2))
    batch_sizes = [int(value) for value in section.get("runtime_batch_sizes", [1, 4])]
    profile: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "seed": seed,
        "adapter_variant": adapter_variant,
        "targets": len(targets),
        "repeats": repeats,
        "context_policy": "full_causal_dialogue",
        "isolated_fixed_workload": True,
        "modes": {},
        "test_locked": True,
    }
    for batch_size in batch_sizes:
        for mode in MODES:
            rendered = [engine.render(target, labels, mode) for target in targets]
            for warmup_index in range(warmup_batches):
                start = (warmup_index * batch_size) % len(rendered)
                warmup = rendered[start : start + batch_size] or rendered[:batch_size]
                engine.generate_batch(warmup, _max_new_tokens(section, mode))
            batch_latencies: list[float] = []
            amortized_latencies: list[float] = []
            completion_lengths: list[int] = []
            tokens_to_emotion: list[int] = []
            total_examples = 0
            total_seconds = 0.0
            peak_memory = 0
            for _ in range(repeats):
                for start in range(0, len(rendered), batch_size):
                    outputs, timing = engine.generate_batch(
                        rendered[start : start + batch_size],
                        _max_new_tokens(section, mode),
                    )
                    batch_latencies.append(float(timing["batch_latency_ms"]))
                    amortized_latencies.extend(
                        [float(timing["amortized_latency_ms"])] * len(outputs)
                    )
                    completion_lengths.extend(
                        int(item["completion_tokens"]) for item in outputs
                    )
                    for item in outputs:
                        prefix = response_prefix_through_emotion(item["response"], labels)
                        tokens_to_emotion.append(
                            len(engine.tokenizer.encode(prefix, add_special_tokens=False))
                            if prefix is not None
                            else int(item["completion_tokens"])
                        )
                    total_examples += len(outputs)
                    total_seconds += float(timing["batch_latency_ms"]) / 1000.0
                    peak_memory = max(peak_memory, int(timing["peak_cuda_memory_bytes"]))
            key = f"batch{batch_size}"
            profile["modes"].setdefault(mode, {})[key] = {
                "examples": total_examples,
                "batches": len(batch_latencies),
                "examples_per_second": total_examples / total_seconds,
                "mean_batch_latency_ms": float(np.mean(batch_latencies)),
                "p50_batch_latency_ms": float(np.quantile(batch_latencies, 0.50)),
                "p95_batch_latency_ms": float(np.quantile(batch_latencies, 0.95)),
                "mean_amortized_latency_ms": float(np.mean(amortized_latencies)),
                "p95_amortized_latency_ms": float(np.quantile(amortized_latencies, 0.95)),
                "mean_completion_tokens": float(np.mean(completion_lengths)),
                "mean_tokens_to_emotion": float(np.mean(tokens_to_emotion)),
                "p95_tokens_to_emotion": float(np.quantile(tokens_to_emotion, 0.95)),
                "peak_cuda_memory_bytes": peak_memory,
            }
    for batch_size in batch_sizes:
        key = f"batch{batch_size}"
        slow_eps = profile["modes"]["slow"][key]["examples_per_second"]
        for mode in ("fast", "adaptive"):
            profile["modes"][mode][key]["speedup_vs_slow"] = (
                profile["modes"][mode][key]["examples_per_second"] / slow_eps
            )
    _write_json(output, profile)
    print(f"Stage 8.1 runtime profile written to {output}")
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return profile


def _metric_from_rows(rows: Sequence[dict[str, Any]], metric: str, labels: Sequence[str]) -> float:
    scored = classification_and_cost_metrics(rows, labels)
    return float(scored[metric])


def paired_dialogue_bootstrap(
    left: Sequence[dict[str, Any]],
    right: Sequence[dict[str, Any]],
    labels: Sequence[str],
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left_rows, right_rows = _aligned_records(left, right)
    left_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    right_by_dialogue: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for left_record, right_record in zip(left_rows, right_rows):
        dialogue_id = str(left_record["dialogue_id"])
        left_by_dialogue[dialogue_id].append(left_record)
        right_by_dialogue[dialogue_id].append(right_record)
    dialogue_ids = sorted(left_by_dialogue)
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {
        "weighted_f1": [],
        "macro_f1": [],
        "accuracy": [],
        "mean_completion_tokens": [],
        "mean_amortized_latency_ms": [],
    }
    for _ in range(samples):
        selected = [rng.choice(dialogue_ids) for _ in dialogue_ids]
        left_sample = [record for dialogue_id in selected for record in left_by_dialogue[dialogue_id]]
        right_sample = [record for dialogue_id in selected for record in right_by_dialogue[dialogue_id]]
        left_metrics = classification_and_cost_metrics(left_sample, labels)
        right_metrics = classification_and_cost_metrics(right_sample, labels)
        for metric in distributions:
            distributions[metric].append(float(right_metrics[metric] - left_metrics[metric]))
    result: dict[str, Any] = {"samples": samples, "unit": "dialogue", "difference": "right-minus-left"}
    for metric, values in distributions.items():
        array = np.asarray(values, dtype=np.float64)
        result[metric] = {
            "mean": float(array.mean()),
            "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
            "two_sided_p": float(min(1.0, 2.0 * min(np.mean(array <= 0), np.mean(array >= 0)))),
        }
    return result


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    return {
        "values": [float(value) for value in values],
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "count": len(values),
    }


def _load_mode_records(
    paths: dict[str, Path], seed: int, adapter: str
) -> dict[str, list[dict[str, Any]]]:
    result = {
        mode: _read_jsonl(_prediction_path(paths, seed, adapter, mode)) for mode in MODES
    }
    missing = [mode for mode, rows in result.items() if not rows]
    if missing:
        raise FileNotFoundError(
            f"Missing Stage 8 predictions seed={seed} adapter={adapter}: {missing}"
        )
    return result


def report(config_path: Path, run_name: str = "validation") -> dict[str, Any]:
    config = _read_yaml(config_path)
    section = _section(config)
    root = _project_root(config_path)
    paths = _paths(root, section, run_name)
    seeds = [int(value) for value in section.get("seeds", [42, 43, 44])]
    adapters = [str(value) for value in section.get("adapter_variants", ["full_supervised"])]
    labels = [str(value) for value in section.get("labels", LABELS)]
    preflight(config_path, run_name, seeds)
    summaries: dict[int, dict[str, Any]] = {}
    profiles: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        summary_path = paths["reports"] / f"seed{seed}_evaluation.json"
        runtime_path = paths["reports"] / f"runtime_profile_seed{seed}.json"
        if not summary_path.exists() or not runtime_path.exists():
            raise FileNotFoundError(
                f"Run evaluate and profile for seed {seed} before report"
            )
        summaries[seed] = json.loads(summary_path.read_text(encoding="utf-8"))
        profiles[seed] = json.loads(runtime_path.read_text(encoding="utf-8"))

    # Recompute response metrics from frozen JSONL so reports created before the
    # semantic/format audit split can be corrected without GPU generation.
    refreshed_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        refreshed_metrics[seed] = {}
        for adapter in adapters:
            refreshed_metrics[seed][adapter] = {
                mode: classification_and_cost_metrics(
                    _read_jsonl(_prediction_path(paths, seed, adapter, mode)), labels
                )
                for mode in MODES
            }

    metric_names = (
        "weighted_f1",
        "macro_f1",
        "accuracy",
        "parse_success_rate",
        "format_success_rate",
        "strict_format_success_rate",
        "emotion_tag_success_rate",
        "strategy_tag_success_rate",
        "slow_strategy_rate",
        "semantic_slow_rate",
        "reasoning_present_rate",
        "reasoning_before_emotion_rate",
        "emotion_before_reasoning_rate",
        "direct_label_response_rate",
        "mean_dialogue_tokens",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_tokens_to_emotion",
        "p95_tokens_to_emotion",
        "p95_completion_tokens",
        "mean_total_tokens",
        "mean_amortized_latency_ms",
    )
    aggregates: dict[str, Any] = {}
    cost_rows: list[dict[str, Any]] = []
    bootstrap_result: dict[str, Any] = {}
    for adapter in adapters:
        aggregates[adapter] = {}
        bootstrap_result[adapter] = {}
        for mode in MODES:
            aggregates[adapter][mode] = {
                metric: _aggregate(
                    [
                        float(refreshed_metrics[seed][adapter][mode][metric])
                        for seed in seeds
                    ]
                )
                for metric in metric_names
            }
            for seed in seeds:
                row = {
                    "seed": seed,
                    "adapter": adapter,
                    "mode": mode,
                }
                row.update(
                    {
                        metric: refreshed_metrics[seed][adapter][mode][metric]
                        for metric in metric_names
                    }
                )
                cost_rows.append(row)
        for seed in seeds:
            rows = _load_mode_records(paths, seed, adapter)
            for left, right in (("fast", "slow"), ("fast", "adaptive"), ("slow", "adaptive")):
                name = f"seed{seed}_{right}_minus_{left}"
                bootstrap_result[adapter][name] = paired_dialogue_bootstrap(
                    rows[left],
                    rows[right],
                    labels,
                    int(section.get("bootstrap_samples", 2000)),
                    int(section.get("bootstrap_seed", 8131)) + seed,
                )

    runtime_aggregates: dict[str, Any] = {}
    for mode in MODES:
        runtime_aggregates[mode] = {}
        for batch_size in [int(value) for value in section.get("runtime_batch_sizes", [1, 4])]:
            key = f"batch{batch_size}"
            metric_keys = (
                "examples_per_second",
                "mean_batch_latency_ms",
                "p95_batch_latency_ms",
                "mean_amortized_latency_ms",
                "p95_amortized_latency_ms",
                "mean_completion_tokens",
                "mean_tokens_to_emotion",
                "p95_tokens_to_emotion",
            )
            runtime_aggregates[mode][key] = {
                metric: _aggregate(
                    [float(profiles[seed]["modes"][mode][key][metric]) for seed in seeds]
                )
                for metric in metric_keys
            }
            if mode in {"fast", "adaptive"}:
                runtime_aggregates[mode][key]["speedup_vs_slow"] = _aggregate(
                    [float(profiles[seed]["modes"][mode][key]["speedup_vs_slow"]) for seed in seeds]
                )

    primary = str(section.get("primary_adapter_variant", "full_supervised"))
    fast_wf1 = aggregates[primary]["fast"]["weighted_f1"]["mean"]
    slow_wf1 = aggregates[primary]["slow"]["weighted_f1"]["mean"]
    adaptive_wf1 = aggregates[primary]["adaptive"]["weighted_f1"]["mean"]
    slow_tokens = aggregates[primary]["slow"]["mean_completion_tokens"]["mean"]
    adaptive_tokens = aggregates[primary]["adaptive"]["mean_completion_tokens"]["mean"]
    token_reduction = 1.0 - adaptive_tokens / max(1e-12, slow_tokens)
    oracle_best_values: list[float] = []
    oracle_best_actual_rates: list[float] = []
    oracle_budget_limit = float(section.get("maximum_oracle_slow_rate", 0.50))
    for seed in seeds:
        frontier = summaries[seed]["adapters"][primary]["derived"]["oracle_frontier"]
        eligible = [row for row in frontier if float(row["actual_slow_rate"]) <= oracle_budget_limit]
        best = max(
            eligible,
            key=lambda row: (float(row["weighted_f1"]), -float(row["actual_slow_rate"])),
        )
        oracle_best_values.append(float(best["weighted_f1"]))
        oracle_best_actual_rates.append(float(best["actual_slow_rate"]))
    oracle_gain = float(np.mean(oracle_best_values)) - fast_wf1
    oracle_realized_slow_rate = float(np.mean(oracle_best_actual_rates))
    batch1_speedup = runtime_aggregates["adaptive"]["batch1"]["speedup_vs_slow"]["mean"]
    batch4_speedup = runtime_aggregates["adaptive"]["batch4"]["speedup_vs_slow"]["mean"]

    tolerance = float(section.get("classification_tolerance", 0.003))
    oracle_pass = oracle_gain >= float(section.get("minimum_oracle_wf1_gain", 0.003))
    adaptive_quality_pass = adaptive_wf1 >= slow_wf1 - tolerance
    token_pass = token_reduction >= float(section.get("minimum_completion_token_reduction", 0.15))
    runtime_pass = (
        batch1_speedup >= float(section.get("minimum_batch1_speedup", 1.05))
        and batch4_speedup >= float(section.get("minimum_batch4_speedup", 1.03))
    )
    if not oracle_pass:
        status_value = "stop_slow_reasoning_no_selective_value"
    elif not adaptive_quality_pass:
        status_value = "proceed_to_stage8_2_adaptive_sft"
    elif not token_pass:
        status_value = "proceed_to_stage8_2_cost_aware_sft"
    elif not runtime_pass:
        status_value = "token_saving_without_wallclock_speedup"
    else:
        status_value = "adaptive_prompt_baseline_viable_train_sft_before_rl"
    decision = {
        "status": status_value,
        "primary_adapter": primary,
        "fast_weighted_f1": fast_wf1,
        "slow_weighted_f1": slow_wf1,
        "adaptive_weighted_f1": adaptive_wf1,
        "oracle_gain_at_or_below_budget": oracle_gain,
        "maximum_oracle_slow_rate": oracle_budget_limit,
        "oracle_realized_slow_rate_at_best_point": oracle_realized_slow_rate,
        "adaptive_completion_token_reduction_vs_slow": token_reduction,
        "adaptive_batch1_speedup_vs_slow": batch1_speedup,
        "adaptive_batch4_speedup_vs_slow": batch4_speedup,
        "checks": {
            "oracle_selective_value": oracle_pass,
            "adaptive_quality_noninferiority": adaptive_quality_pass,
            "completion_token_reduction": token_pass,
            "measured_runtime_speedup": runtime_pass,
        },
        "rl_status": "not_permitted_by_stage8_1; establish adaptive SFT before GRPO",
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
        "runtime": runtime_aggregates,
        "decision": decision,
        "test_locked": True,
    }
    _write_json(paths["summary"], final)
    _write_json(paths["bootstrap"], bootstrap_result)
    _write_json(paths["decision"], decision)
    paths["cost_csv"].parent.mkdir(parents=True, exist_ok=True)
    with paths["cost_csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cost_rows[0]))
        writer.writeheader()
        writer.writerows(cost_rows)

    lines = [
        "# Stage 8.1 — Full-context Fast/Slow Thinking",
        "",
        "> MELD validation-only. Every method sees the same complete causal dialogue. No gate, no recent-k truncation, and no test access.",
        "",
        "| Adapter | Mode | wF1 | Macro F1 | Accuracy | Completion tokens | `<SLOW>` tag | Reasoning before label | Semantic slow | Strict format |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for adapter in adapters:
        for mode in MODES:
            values = aggregates[adapter][mode]
            lines.append(
                f"| {adapter} | {mode} | {values['weighted_f1']['mean']:.4f} ± {values['weighted_f1']['std']:.4f} "
                f"| {values['macro_f1']['mean']:.4f} ± {values['macro_f1']['std']:.4f} "
                f"| {values['accuracy']['mean']:.4f} ± {values['accuracy']['std']:.4f} "
                f"| {values['mean_completion_tokens']['mean']:.2f} "
                f"| {values['slow_strategy_rate']['mean']:.2%} "
                f"| {values['reasoning_before_emotion_rate']['mean']:.2%} "
                f"| {values['semantic_slow_rate']['mean']:.2%} "
                f"| {values['strict_format_success_rate']['mean']:.2%} |"
            )
    lines.extend(
        [
            "",
            "## Frozen decision",
            "",
            f"- Status: `{status_value}`",
            f"- Oracle wF1 gain with a preregistered cap of {oracle_budget_limit:.0%}: {oracle_gain:+.4f}",
            f"- Realized Slow rate at the best eligible oracle point: {oracle_realized_slow_rate:.2%}",
            f"- Adaptive completion-token reduction vs slow: {token_reduction:.2%}",
            f"- Adaptive runtime speedup vs slow: batch1 {batch1_speedup:.3f}×; batch4 {batch4_speedup:.3f}×",
            "- RL remains blocked until an adaptive SFT policy is established.",
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
    adapters = [str(value) for value in section.get("adapter_variants", ["full_supervised"])]
    value = {
        "experiment": EXPERIMENT,
        "version": VERSION,
        "run_name": run_name,
        "preflight": paths["preflight"].exists(),
        "predictions": {
            str(seed): {
                adapter: {
                    mode: _prediction_path(paths, seed, adapter, mode).exists()
                    for mode in MODES
                }
                for adapter in adapters
            }
            for seed in seeds
        },
        "evaluations": {
            str(seed): (paths["reports"] / f"seed{seed}_evaluation.json").exists()
            for seed in seeds
        },
        "runtime_profiles": {
            str(seed): (paths["reports"] / f"runtime_profile_seed{seed}.json").exists()
            for seed in seeds
        },
        "final_report": paths["summary"].exists(),
        "test_locked": True,
    }
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return value


def final_test_locked() -> None:
    raise RuntimeError(
        "Stage 8.1 cannot read MELD test. Finish the preregistered validation audit, "
        "freeze the next method, and create a separate hash-locked test patch."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--seeds", type=int, nargs="*")

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--seed", type=int, required=True)
    evaluate_parser.add_argument("--max-targets", type=int)
    evaluate_parser.add_argument("--overwrite", action="store_true")

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--seed", type=int, required=True)
    profile_parser.add_argument("--max-targets", type=int)
    profile_parser.add_argument("--repeats", type=int)
    profile_parser.add_argument("--overwrite", action="store_true")

    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument("--seed", type=int, default=42)
    subparsers.add_parser("report")
    subparsers.add_parser("status")
    subparsers.add_parser("final-test")

    args = parser.parse_args(argv)
    if args.command == "preflight":
        preflight(args.config, args.run_name, args.seeds)
    elif args.command == "evaluate":
        evaluate_seed(
            args.config,
            args.seed,
            args.run_name,
            max_targets=args.max_targets,
            overwrite=args.overwrite,
        )
    elif args.command == "profile":
        runtime_profile_seed(
            args.config,
            args.seed,
            args.run_name,
            max_targets=args.max_targets,
            repeats_override=args.repeats,
            overwrite=args.overwrite,
        )
    elif args.command == "smoke":
        smoke_name = f"smoke_seed{args.seed}"
        preflight(args.config, smoke_name, [args.seed])
        evaluate_seed(
            args.config,
            args.seed,
            smoke_name,
            max_targets=8,
            overwrite=True,
        )
        runtime_profile_seed(
            args.config,
            args.seed,
            smoke_name,
            max_targets=8,
            repeats_override=1,
            overwrite=True,
        )
        print(json.dumps({"status": "smoke_passed", "test_locked": True}, indent=2))
    elif args.command == "report":
        report(args.config, args.run_name)
    elif args.command == "status":
        status(args.config, args.run_name)
    elif args.command == "final-test":
        final_test_locked()


if __name__ == "__main__":
    main()
