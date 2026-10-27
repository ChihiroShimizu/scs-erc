"""Stage 8C.1 CLI: frozen Stage7 adapters, current-only vs current+previous-3.

This is a paired input-context audit, not a FAST/SLOW tag or reasoning-trace
experiment. Nothing in this module trains a model, fits a router, or opens test.
GPU libraries are imported only for inference, keeping preflight/tests usable
without loading torch or model weights.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import gc
import importlib.metadata
import io
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pfa_erc.training.stage8c_context_core import (
    EXPERIMENT, LABELS, PROTOCOL, SPLITS, SYSTEM, VERSION, VIEWS,
    Target, aggregate, build_prompt, build_targets, digest, metrics,
    oracle_and_random, paired_bootstrap, paired_diagnostic, parse_response,
    split_dialogues, stable_hash, summarize_runtime,
)


def log(message: str) -> None:
    print(f"[Stage8C.1 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for n, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected object at {path}:{n}")
                rows.append(row)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def sha_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_data_file(processed: Path, name: str) -> Path:
    if name not in ("train.jsonl", "validation.jsonl"):
        raise ValueError("Stage8C.1 accepts only train.jsonl and validation.jsonl; test is locked")
    path = (processed / name).resolve(strict=True)
    if path.name != name or any(part.lower() in ("test", "testing") for part in path.parts):
        raise ValueError(f"Refusing a renamed/test data path for {name}: {path}")
    return path


def read_meld(path: Path) -> list[Target]:
    # Reuse the repository's processed MELD schema; no speculative JSON conversion.
    try:
        from pfa_erc.data import read_jsonl as read_dialogues
    except ImportError as exc:
        raise RuntimeError("Install this patch into the existing pfa_erc_reproduction project (missing pfa_erc.data)") from exc
    return build_targets(read_dialogues(path))


class Run:
    def __init__(self, config_path: Path, run_name: str, limit: int | None):
        self.config_path = config_path.expanduser().resolve(strict=True)
        if self.config_path.parent.name != "configs":
            raise ValueError("Keep the YAML in the project's configs/ directory")
        self.root = self.config_path.parent.parent
        whole = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.cfg = whole["llm_context_budget_stage8c"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_name):
            raise ValueError("run-name must contain only letters, digits, '_' or '-'")
        self.smoke = run_name.startswith("smoke")
        self.limit = 16 if self.smoke and limit is None else limit
        if self.limit is not None and (not self.smoke or self.limit < 1):
            raise ValueError("Partial targets are allowed only in a smoke* run")
        if self.cfg["context_policy"] != "current_vs_previous3" or self.cfg["history_turns"] != 3:
            raise ValueError("Frozen protocol requires exactly previous 3 causal turns")
        if tuple(self.cfg["labels"]) != LABELS or self.cfg["dataset"].lower() != "meld":
            raise ValueError("Frozen MELD label schema mismatch")
        if self.cfg["evaluation_batch_size"] != 1:
            raise ValueError("Quality evaluation uses batch=1 to avoid cross-view padding confounds")
        if self.cfg["adapter_variants"] != ["emotion_base", "full_supervised"]:
            raise ValueError("This stage freezes Stage7 emotion_base and full_supervised")
        if self.cfg["seeds"] != [42, 43, 44] or self.cfg["primary_adapter_variant"] != "full_supervised":
            raise ValueError("Frozen comparison requires seeds 42/43/44 and primary full_supervised")
        if self.cfg["dtype"] not in ("bfloat16", "float16"):
            raise ValueError("Unsupported dtype")
        for field in ("log_every_targets", "runtime_targets", "runtime_repeats", "runtime_warmup_batches", "matched_random_repeats"):
            if not isinstance(self.cfg[field], int) or self.cfg[field] < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.cfg["runtime_batch_sizes"] != [1, 4] or self.cfg["bootstrap_samples"] < 2:
            raise ValueError("Profile batch sizes must be [1,4] and bootstrap_samples >=2")
        budgets = self.cfg["oracle_slow_budgets"]
        if not budgets or budgets != sorted(set(budgets)) or budgets[0] != 0 or budgets[-1] != 1:
            raise ValueError("Oracle budgets must increase from zero to one")
        if not 0 <= self.cfg["minimum_calibration_oracle_gain"] <= 1 or not 0 <= self.cfg["minimum_strict_format"] <= 1:
            raise ValueError("Invalid diagnostic planning thresholds")
        if not self.cfg.get("test_locked", False):
            raise ValueError("MELD test must remain locked")
        if self.cfg["max_new_tokens"] < 2 or self.cfg["max_prompt_tokens"] < 1:
            raise ValueError("Invalid token budget")
        self.name = run_name
        self.base = self.resolve(self.cfg["artifact_dir"]) / run_name
        expected_base = (self.root / "artifacts/llm_context_budget_stage8c/meld").resolve()
        if self.base.resolve().parent != expected_base:
            raise ValueError("Use the separate Stage8C artifact directory; old experiments are preserved")
        self.reports = self.base / "reports"
        self.predictions = self.base / "predictions"
        self.manifest: dict = {}
        self.targets: dict[str, list[Target]] = {}

    def resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    def adapter(self, seed: int, variant: str) -> Path:
        if seed not in self.cfg["seeds"] or variant not in self.cfg["adapter_variants"]:
            raise ValueError("Seed/adapter is outside the frozen configuration")
        return self.resolve(self.cfg["stage7_adapter_root"]) / f"seed{seed}" / variant

    def preflight(self) -> dict:
        log("Preflight: reading train/validation and checking inherited Stage7 split; no GPU model load")
        processed = self.resolve(self.cfg["processed_dir"])
        train_path = require_data_file(processed, "train.jsonl")
        validation_path = require_data_file(processed, "validation.jsonl")
        train, validation = read_meld(train_path), read_meld(validation_path)
        if len(train) != self.cfg["expected_train_targets"] or len(validation) != self.cfg["expected_validation_targets"]:
            raise ValueError(f"MELD target count mismatch: train={len(train)}, validation={len(validation)}")
        stage7_config_path = self.resolve(self.cfg["stage7_config"])
        stage7_cfg = yaml.safe_load(stage7_config_path.read_text(encoding="utf-8"))["llm_evidence_supervision_stage7"]
        if (int(stage7_cfg["split_seed"]) != self.cfg["split_seed"] or
                float(stage7_cfg["calibration_dialogue_fraction"]) != self.cfg["calibration_dialogue_fraction"] or
                stage7_cfg["model_name"] != self.cfg["model_name"]):
            raise ValueError("Stage7 backbone or train/calibration split differs from this config")
        train_ids, calibration_ids = split_dialogues(
            [t.dialogue_id for t in train], self.cfg["calibration_dialogue_fraction"], self.cfg["split_seed"])
        calibration = [t for t in train if t.dialogue_id in calibration_ids]
        val_ids = {t.dialogue_id for t in validation}
        if val_ids & (train_ids | calibration_ids):
            raise ValueError("Train/validation dialogue IDs overlap")
        old_cal_path = self.resolve(self.cfg["stage7_calibration_targets"])
        old_cal = read_jsonl(old_cal_path)
        old_map = {str(x["key"]): x for x in old_cal}
        if len(old_map) != len(old_cal) or set(old_map) != {t.key for t in calibration}:
            raise ValueError("Calibration keys do not match Stage7's held-out prepared partition")
        for t in calibration:
            old = old_map[t.key]
            if (str(old["gold"]), str(old["text"]), str(old["speaker"])) != (t.gold, t.text, t.speaker):
                raise ValueError(f"Stage7 calibration content mismatch: {t.key}")
        self.targets = {"calibration": calibration, "validation": validation}
        if self.smoke:
            self.targets = {k: sorted(v, key=lambda t: (stable_hash(t.key, 8301), t.key))[:self.limit]
                            for k, v in self.targets.items()}
        # Verify invariants before tokenization or GPU allocation.
        for targets in self.targets.values():
            for target in targets:
                f, s = build_prompt(target, "fast"), build_prompt(target, "slow")
                if target.turn_index == 0 and f != s:
                    raise ValueError("First-turn prompts must be identical")
        adapter_hashes = {}
        for seed in self.cfg["seeds"]:
            for variant in self.cfg["adapter_variants"]:
                directory = self.adapter(seed, variant)
                if not (directory / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"Missing Stage7 adapter: {directory}")
                weights = list(directory.glob("adapter_model*.safetensors")) + list(directory.glob("adapter_model*.bin"))
                if not weights or any(p.stat().st_size == 0 for p in weights):
                    raise FileNotFoundError(f"Missing/empty adapter weights: {directory}")
                adapter_cfg = read_json(directory / "adapter_config.json")
                if str(adapter_cfg.get("peft_type", "")).upper() != "LORA":
                    raise ValueError(f"Expected LoRA adapter: {directory}")
                files = sorted(p for p in directory.iterdir() if p.is_file() and
                               (p.suffix in (".json", ".safetensors", ".bin", ".txt", ".model")))
                log(f"Hashing frozen checkpoint seed={seed} adapter={variant}")
                adapter_hashes[f"{seed}:{variant}"] = {p.name: sha_file(p) for p in files}
        source = {p.name: sha_file(p) for p in (Path(__file__), Path(__file__).with_name("stage8c_context_core.py"))}
        identity = {
            "experiment": EXPERIMENT, "version": VERSION, "protocol": PROTOCOL,
            "run_name": self.name, "smoke": self.smoke, "limit": self.limit,
            "config_sha256": sha_file(self.config_path), "implementation_sha256": source,
            "train_sha256": sha_file(train_path), "validation_sha256": sha_file(validation_path),
            "stage7_config_sha256": sha_file(stage7_config_path),
            "stage7_calibration_sha256": sha_file(old_cal_path), "adapter_sha256": adapter_hashes,
            "ordered_target_sha256": {k: digest([t.identity_sha256 for t in v]) for k, v in self.targets.items()},
            "test_locked": True,
        }
        fingerprint = digest(identity)
        manifest = dict(identity, fingerprint=fingerprint,
                        target_counts={k: len(v) for k, v in self.targets.items()},
                        dialogue_counts={k: len({t.dialogue_id for t in v}) for k, v in self.targets.items()},
                        trained_here=False, calibration_source="unchanged_Stage7_MELD_train_holdout",
                        future_turns_visible=False, teacher_annotations_visible=False,
                        new_policy_training=False, classification_output="one_bare_emotion_label")
        path = self.reports / "manifest.json"
        if path.exists() and read_json(path).get("fingerprint") != fingerprint:
            raise ValueError("Existing run fingerprint changed. Preserve that run and choose a new RUN_NAME.")
        atomic_json(path, manifest)
        self.manifest = manifest
        log(f"Preflight passed: {manifest['target_counts']}; token lengths checked before each generation")
        return manifest

    def prediction_path(self, seed: int, variant: str, split: str, view: str) -> Path:
        if split not in SPLITS or view not in VIEWS:
            raise ValueError("Only calibration/validation and fast/slow are supported; test is locked")
        return self.predictions / f"seed{seed}_{variant}_{split}_{view}.jsonl"

    def checked_rows(self, path: Path, seed: int, variant: str, split: str, view: str,
                     complete: bool) -> list[dict]:
        rows = read_jsonl(path)
        targets = self.targets[split]
        if len(rows) > len(targets) or (complete and len(rows) != len(targets)):
            raise ValueError(f"Wrong prediction count in {path}")
        backend = read_json(self.reports / "backend_manifest.json")
        backend_hash = digest(backend)
        for row, target in zip(rows, targets):
            expected = {"fingerprint": self.manifest["fingerprint"], "backend_sha256": backend_hash,
                        "seed": seed, "adapter_variant": variant, "split": split, "view": view,
                        "key": target.key, "target_sha256": target.identity_sha256,
                        "dialogue_id": target.dialogue_id, "gold": target.gold,
                        "available_history_turns": len(target.history),
                        "supplied_history_turns": len(target.history) if view == "slow" else 0,
                        "user_prompt": build_prompt(target, view), "system_prompt": SYSTEM}
            if any(row.get(k) != value for k, value in expected.items()):
                raise ValueError(f"Stale/misaligned cached prediction at {path}: {target.key}")
            parsed = parse_response(row["response"])
            if any(row[k] != parsed[k] for k in parsed):
                raise ValueError(f"Cached parsing mismatch: {target.key}")
            for field in ("prompt_tokens", "completion_tokens", "generated_tokens_including_eos"):
                if not isinstance(row.get(field), int) or row[field] < 0:
                    raise ValueError(f"Invalid cached token count: {target.key}")
        if complete:
            done = read_json(path.with_suffix(".done.json"))
            if done != {"fingerprint": self.manifest["fingerprint"], "backend_sha256": backend_hash,
                        "records": len(rows), "sha256": sha_file(path)}:
                raise ValueError(f"Missing/stale completion marker for {path}")
        return rows

    def completed(self, seed: int, variant: str, split: str, view: str) -> bool:
        path = self.prediction_path(seed, variant, split, view)
        if path.exists() and path.with_suffix(".done.json").exists():
            self.checked_rows(path, seed, variant, split, view, True)
            return True
        return False


@contextmanager
def run_lock(run: Run):
    run.base.mkdir(parents=True, exist_ok=True)
    with (run.base / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another Stage8C process owns this run; use one launcher per RUN_NAME") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class Engine:
    def __init__(self, run: Run, seed: int, variant: str):
        log(f"Loading tokenizer/model: seed={seed}, adapter={variant}. First load may take minutes.")
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU via GPU_ID / CUDA_VISIBLE_DEVICES")
        self.torch, self.run = torch, run
        self.seed, self.variant = seed, variant
        self.adapter_path = run.adapter(seed, variant)
        cfg = run.cfg
        common = {"local_files_only": cfg["local_files_only"], "trust_remote_code": False}
        tokenizer_source = self.adapter_path if (self.adapter_path / "tokenizer_config.json").exists() else cfg["model_name"]
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), **common)
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer needs an EOS token")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        source_config = AutoConfig.from_pretrained(cfg["model_name"], revision=cfg["model_revision"], **common)
        revision = getattr(source_config, "_commit_hash", None)
        if revision is None and not Path(cfg["model_name"]).exists():
            raise ValueError("Cannot resolve the remote backbone commit; pin model_revision explicitly")
        dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This config requires BF16-capable CUDA")
        backend = {
            "model_name": cfg["model_name"], "resolved_model_revision": revision,
            "tokenizer_sha256": digest({"vocab": self.tokenizer.get_vocab(),
                                        "template": self.tokenizer.chat_template,
                                        "special_tokens": self.tokenizer.special_tokens_map}),
            "dtype": cfg["dtype"], "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "peft_version": importlib.metadata.version("peft"),
            "cuda_version": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
            "decoding": "greedy; num_beams=1; no_forced_emotion; no_strategy_tags",
        }
        backend_path = run.reports / "backend_manifest.json"
        if backend_path.exists() and read_json(backend_path) != backend:
            raise ValueError("Backend/tokenizer differs from the existing run; choose a new RUN_NAME")
        log(f"Backbone revision: {revision}; loading frozen LoRA weights on {backend['gpu_name']}")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        base = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"], revision=revision or cfg["model_revision"],
            torch_dtype=dtype, low_cpu_mem_usage=True, **common)
        self.model = PeftModel.from_pretrained(base, str(self.adapter_path), is_trainable=False)
        self.model.to("cuda").eval()
        self.model.config.use_cache = True
        if any(p.requires_grad for p in self.model.parameters()):
            raise ValueError("A supposedly frozen adapter has trainable parameters")
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=cfg["max_new_tokens"], use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        atomic_json(backend_path, backend)
        self.backend_sha256 = digest(backend)
        log("Model ready. Both views use the same checkpoint, prompt template and generation settings.")

    def render(self, target: Target, view: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(target, view)}],
            tokenize=False, add_generation_prompt=True,
        )

    def generate(self, targets: list[Target], view: str) -> tuple[list[dict], dict]:
        torch = self.torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        prompts = [self.render(t, view) for t in targets]
        encoded = self.tokenizer(prompts, padding=True, truncation=False,
                                 add_special_tokens=False, return_tensors="pt")
        lengths = [int(x) for x in encoded["attention_mask"].sum(dim=1).tolist()]
        if max(lengths) > self.run.cfg["max_prompt_tokens"]:
            raise ValueError(f"Prompt too long ({max(lengths)} tokens): refusing to truncate target or previous-3")
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and max(lengths) + self.run.cfg["max_new_tokens"] > max_positions:
            raise ValueError("Prompt+completion exceeds the model context limit")
        encoded = {k: v.to("cuda") for k, v in encoded.items()}
        width = int(encoded["input_ids"].shape[1])
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model.generate(**encoded, generation_config=self.generation)
        torch.cuda.synchronize()
        model_ms = (time.perf_counter() - model_started) * 1000
        rows = outputs[:, width:].tolist()
        predictions = []
        eos = int(self.tokenizer.eos_token_id)
        for length, raw_ids in zip(lengths, rows):
            stopped = eos in raw_ids
            eos_index = raw_ids.index(eos) if stopped else len(raw_ids)
            ids = raw_ids[:eos_index]
            response = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
            parsed = parse_response(response)
            token_position = None
            if parsed["parse_success"]:
                for i in range(1, len(ids) + 1):
                    prefix = self.tokenizer.decode(ids[:i], skip_special_tokens=True)
                    if parse_response(prefix)["prediction"] == parsed["prediction"]:
                        token_position = i
                        break
            predictions.append(dict(
                response=response, prompt_tokens=length, completion_tokens=len(ids),
                generated_tokens_including_eos=len(ids) + int(stopped),
                tokens_to_emotion=token_position, stopped_on_eos=stopped,
                hit_max_new_tokens=not stopped, **parsed,
            ))
        elapsed_ms = (time.perf_counter() - started) * 1000
        timing = {
            "examples": len(targets), "e2e_ms": elapsed_ms, "model_generate_ms": model_ms,
            "prompt_tokens_sum": sum(lengths), "padded_prompt_tokens": width * len(targets),
            "completion_tokens_sum": sum(x["completion_tokens"] for x in predictions),
            "generated_tokens_sum": sum(x["generated_tokens_including_eos"] for x in predictions),
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        del encoded, outputs
        return predictions, timing

    def close(self) -> None:
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def load_partial(run: Run, path: Path, seed: int, variant: str, split: str, view: str) -> list[dict]:
    if not path.exists():
        return []
    # A killed process can leave only its last JSONL line incomplete. Keep all
    # complete preceding records; never accept a corrupt interior line.
    lines = path.read_bytes().splitlines()
    for i, line in enumerate(lines):
        try:
            json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i != len(lines) - 1:
                raise ValueError(f"Corrupt interior cache line: {path}:{i+1}")
            path.write_bytes(b"\n".join(lines[:i]) + (b"\n" if i else b""))
            log(f"Recovered interrupted trailing record in {path.name}")
    return run.checked_rows(path, seed, variant, split, view, False)


def predict(run: Run, seed: int, variant: str, splits: list[str]) -> None:
    todo = [(split, view) for split in splits for view in VIEWS if not run.completed(seed, variant, split, view)]
    if not todo:
        log(f"All predictions already complete and verified: seed={seed}, {variant}")
        return
    engine = Engine(run, seed, variant)
    try:
        for split, view in todo:
            path = run.prediction_path(seed, variant, split, view)
            partial = path.with_suffix(".partial.jsonl")
            # If an interruption occurred after rename but before .done creation,
            # verify and recover that completed file rather than rerun it.
            source = partial if partial.exists() else path
            cached = load_partial(run, source, seed, variant, split, view) if source.exists() else []
            if source != partial and source.exists():
                source.replace(partial)
            targets = run.targets[split]
            partial.parent.mkdir(parents=True, exist_ok=True)
            log(f"{split} {variant} seed={seed} {view}: resume {len(cached)}/{len(targets)}")
            with partial.open("a", encoding="utf-8") as handle:
                for i in range(len(cached), len(targets)):
                    target = targets[i]
                    outputs, timing = engine.generate([target], view)
                    row = dict(outputs[0], fingerprint=run.manifest["fingerprint"],
                               backend_sha256=engine.backend_sha256, seed=seed,
                               adapter_variant=variant, split=split, view=view,
                               key=target.key, dialogue_id=target.dialogue_id, gold=target.gold,
                               target_sha256=target.identity_sha256, available_history_turns=len(target.history),
                               supplied_history_turns=len(target.history) if view == "slow" else 0,
                               user_prompt=build_prompt(target, view), system_prompt=SYSTEM,
                               routine_generation_timing=timing,
                               timing_scope="routine_nonisolated; use dedicated runtime profile for claims")
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    if i == len(cached) or (i + 1) % run.cfg["log_every_targets"] == 0 or i + 1 == len(targets):
                        log(f"{split} {variant} seed={seed} {view}: {i+1}/{len(targets)}")
            partial.replace(path)
            atomic_json(path.with_suffix(".done.json"), {
                "fingerprint": run.manifest["fingerprint"], "backend_sha256": engine.backend_sha256,
                "records": len(targets), "sha256": sha_file(path),
            })
        for split in splits:
            evaluate_one(run, seed, variant, split)
    finally:
        engine.close()


def read_pair(run: Run, seed: int, variant: str, split: str) -> tuple[list[dict], list[dict]]:
    return tuple(run.checked_rows(run.prediction_path(seed, variant, split, view), seed, variant, split, view, True)
                 for view in VIEWS)


def evaluate_one(run: Run, seed: int, variant: str, split: str) -> dict:
    fast, slow = read_pair(run, seed, variant, split)
    paired = paired_diagnostic(fast, slow)
    cases = paired.pop("cases")
    result = {
        "fingerprint": run.manifest["fingerprint"], "seed": seed, "adapter_variant": variant,
        "split": split, "fast": metrics(fast), "slow": metrics(slow), "paired": paired,
        "hindsight_diagnostics": oracle_and_random(fast, slow, run.cfg["oracle_slow_budgets"],
                                                    run.cfg["matched_random_repeats"], run.cfg["analysis_seed"]),
        "policy_trained": False, "test_locked": True,
    }
    stem = f"seed{seed}_{variant}_{split}"
    atomic_json(run.reports / f"{stem}.json", result)
    write_jsonl(run.reports / f"{stem}_paired_cases.jsonl", cases)
    log(f"{stem}: wF1 FAST={result['fast']['weighted_f1']:.4f}, SLOW={result['slow']['weighted_f1']:.4f}; "
        f"helps={paired['counts']['history_helps']}, harms={paired['counts']['history_harms']}")
    return result


def gpu_snapshot(engine: Engine) -> dict:
    # Record observed compute-process exclusivity; never stop another process.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    props = engine.torch.cuda.get_device_properties(0)
    gpu_uuid = getattr(props, "uuid", None)
    if isinstance(gpu_uuid, bytes):
        gpu_uuid = gpu_uuid.decode()
    if gpu_uuid and not str(gpu_uuid).startswith(("GPU-", "MIG-")):
        gpu_uuid = "GPU-" + str(gpu_uuid)
    try:
        if not gpu_uuid:
            gpu_uuid = subprocess.check_output(
                ["nvidia-smi", "-i", visible or "0", "--query-gpu=uuid", "--format=csv,noheader"],
                text=True, timeout=15).strip()
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=15)
        processes = []
        for row in csv.reader(io.StringIO(output)):
            if len(row) >= 4 and row[0].strip() == str(gpu_uuid):
                processes.append({"pid": int(row[1].strip()), "process": row[2].strip(),
                                  "gpu_memory_mib": row[3].strip()})
        # Own process must be visible, otherwise container PID namespaces or an
        # unsupported UUID mapping prevent us from asserting exclusivity.
        own_visible = any(p["pid"] == os.getpid() for p in processes)
        others = [p for p in processes if p["pid"] != os.getpid()]
        result = {"time_unix": time.time(), "gpu_uuid": str(gpu_uuid), "processes": processes,
                  "own_process_visible": own_visible, "exclusive_observed": own_visible and not others}
    except (subprocess.SubprocessError, OSError, ValueError) as exc:
        result = {"time_unix": time.time(), "exclusive_observed": False, "error": str(exc)}
    if engine.run.cfg["require_exclusive_gpu_for_profile"] and not result["exclusive_observed"]:
        raise RuntimeError("Runtime profiling needs an idle GPU with only this process visible. "
                           "Quality predictions are preserved. Re-run profile-all on an available GPU. "
                           f"Observed: {json.dumps(result, ensure_ascii=False)}")
    return result


def profile(run: Run, seed: int) -> None:
    variant = run.cfg["primary_adapter_variant"]
    fast, slow = read_pair(run, seed, variant, "validation")
    path = run.reports / f"seed{seed}_{variant}_runtime.json"
    if path.exists():
        existing = read_json(path)
        if (existing["fingerprint"] != run.manifest["fingerprint"] or
                existing["backend_sha256"] != digest(read_json(run.reports / "backend_manifest.json"))):
            raise ValueError("Stale runtime report")
        log(f"Verified existing runtime profile seed={seed}; skipping")
        return
    target_map = {t.key: t for t in run.targets["validation"]}
    keys = sorted(target_map, key=lambda key: (stable_hash(key, run.cfg["runtime_subset_seed"]), key))
    keys = keys[:min(len(keys), run.cfg["runtime_targets"])]
    reference = {"fast": {r["key"]: r for r in fast}, "slow": {r["key"]: r for r in slow}}
    engine = Engine(run, seed, variant)
    try:
        snapshots = [gpu_snapshot(engine)]
        all_batches = []
        batch_sizes = run.cfg["runtime_batch_sizes"]
        for batch_size in batch_sizes:
            batches = [keys[i:i+batch_size] for i in range(0, len(keys), batch_size)]
            for view in VIEWS:
                for i in range(run.cfg["runtime_warmup_batches"]):
                    warm = [target_map[k] for k in batches[i % len(batches)]]
                    engine.generate(warm, view)
            for repeat in range(run.cfg["runtime_repeats"]):
                snapshots.append(gpu_snapshot(engine))
                for bi, batch_keys in enumerate(batches):
                    order = VIEWS if (repeat + bi + seed) % 2 == 0 else tuple(reversed(VIEWS))
                    for view in order:
                        outputs, timing = engine.generate([target_map[k] for k in batch_keys], view)
                        all_batches.append(dict(timing, batch_size_requested=batch_size, view=view,
                                                repeat=repeat, batch_index=bi, keys=batch_keys,
                                                predictions=[r["prediction"] for r in outputs],
                                                responses=[r["response"] for r in outputs],
                                                reference_prediction_agreement=sum(
                                                    row["prediction"] == reference[view][key]["prediction"]
                                                    for row, key in zip(outputs, batch_keys)),
                                                invalid_outputs=sum(not r["parse_success"] for r in outputs)))
                    if (bi + 1) % 32 == 0:
                        snapshots.append(gpu_snapshot(engine))
                        log(f"Profile seed={seed}, batch={batch_size}, repeat={repeat+1}: {bi+1}/{len(batches)} pairs")
                log(f"Profile seed={seed}, batch={batch_size}, repeat={repeat+1} finished")
        snapshots.append(gpu_snapshot(engine))
        summary = {}
        for batch_size in batch_sizes:
            views = {}
            for view in VIEWS:
                group = [b for b in all_batches if b["view"] == view and b["batch_size_requested"] == batch_size]
                views[view] = summarize_runtime(group)
                views[view]["prediction_agreement_with_quality_batch1"] = (
                    sum(x["reference_prediction_agreement"] for x in group) / sum(x["examples"] for x in group))
                views[view]["invalid_output_rate"] = sum(x["invalid_outputs"] for x in group) / sum(x["examples"] for x in group)
            # Same workload, repeats and batches. No mixture/oracle speed is inferred.
            views["observed_slow_over_fast_time_ratio"] = (
                views["slow"]["amortized_ms_per_example"] / views["fast"]["amortized_ms_per_example"])
            summary[f"batch{batch_size}"] = views
        atomic_json(path, {
            "fingerprint": run.manifest["fingerprint"], "backend_sha256": engine.backend_sha256,
            "seed": seed, "adapter_variant": variant, "split": "validation",
            "subset_keys": keys, "subset_selection_uses_gold": False,
            "counterbalanced_view_order": True, "warmup_excluded": True,
            "exclusive_gpu_observed_at_checks": all(x["exclusive_observed"] for x in snapshots),
            "exclusivity_scope": "periodic_compute_process_checks; not_a_guarantee_between_checks",
            "gpu_snapshots": snapshots, "summary": summary, "raw_batches": all_batches,
            "time_to_first_token_measured": False,
            "prefill_and_decode_measured_separately": False,
            "adaptive_pipeline_profiled": False, "test_locked": True,
        })
        log(f"Runtime profile saved: {path}")
    finally:
        engine.close()


def report(run: Run) -> dict:
    if run.smoke:
        raise ValueError("Smoke results cannot produce a formal multi-seed report")
    evaluations = {}
    aggregate_results = {}
    for split in SPLITS:
        evaluations[split], aggregate_results[split] = {}, {}
        for variant in run.cfg["adapter_variants"]:
            results = [evaluate_one(run, seed, variant, split) for seed in run.cfg["seeds"]]
            evaluations[split][variant] = results
            entry = {view: {metric: aggregate([r[view][metric] for r in results])
                            for metric in ("weighted_f1", "macro_f1", "accuracy", "mean_prompt_tokens",
                                           "mean_completion_tokens", "mean_generated_tokens_including_eos",
                                           "parse_success", "strict_format_success")}
                     for view in VIEWS}
            entry["paired_rates"] = {key: aggregate([r["paired"]["rates"][key] for r in results])
                                     for key in results[0]["paired"]["rates"]}
            entry["paired_wf1_bootstrap"] = paired_bootstrap(
                [read_pair(run, seed, variant, split) for seed in run.cfg["seeds"]],
                run.cfg["bootstrap_samples"], run.cfg["analysis_seed"])
            entry["hindsight_diagnostics"] = []
            for i, budget in enumerate(run.cfg["oracle_slow_budgets"]):
                points = [r["hindsight_diagnostics"][i] for r in results]
                entry["hindsight_diagnostics"].append({
                    "budget": budget, "realized_slow_rate": aggregate([p["realized_slow_rate"] for p in points]),
                    "oracle_weighted_f1": aggregate([p["correctness_oracle"]["weighted_f1"] for p in points]),
                    "matched_random_weighted_f1": aggregate([p["matched_random_weighted_f1_mean"] for p in points]),
                    "deployable": False,
                })
            aggregate_results[split][variant] = entry
    runtime = {}
    for seed in run.cfg["seeds"]:
        path = run.reports / f"seed{seed}_{run.cfg['primary_adapter_variant']}_runtime.json"
        if path.exists():
            value = read_json(path)
            if value["fingerprint"] != run.manifest["fingerprint"] or value["backend_sha256"] != digest(read_json(run.reports / "backend_manifest.json")):
                raise ValueError("Runtime/source manifest mismatch")
            runtime[str(seed)] = {k: v for k, v in value.items() if k not in ("raw_batches", "gpu_snapshots")}
    primary = aggregate_results["calibration"][run.cfg["primary_adapter_variant"]]
    oracle_gain = primary["hindsight_diagnostics"][-1]["oracle_weighted_f1"]["mean"] - primary["fast"]["weighted_f1"]["mean"]
    format_ok = all(primary[v]["strict_format_success"]["mean"] >= run.cfg["minimum_strict_format"] for v in VIEWS)
    if not format_ok:
        suggestion = "repair_direct_label_protocol_before_additional_training"
    elif oracle_gain >= run.cfg["minimum_calibration_oracle_gain"]:
        suggestion = "context_selection_is_a_candidate_for_a_new_train_only_stage; learnability_not_established"
    else:
        suggestion = "do_not_train_a_router_yet; inspect_view_training_mismatch_and_paired_errors"
    runtime_exclusive = bool(runtime) and all(r["exclusive_gpu_observed_at_checks"] for r in runtime.values())
    decision = {
        "status": "paired_context_audit_complete" if len(runtime) == len(run.cfg["seeds"]) else "quality_complete_runtime_pending",
        "recommendation_basis": "Stage7_train_calibration_only; validation_not_used_to_fit_any_policy",
        "calibration_correctness_oracle_gain": oracle_gain, "next_experiment_suggestion": suggestion,
        "training_performed": False, "adaptive_policy_established": False, "rl_performed": False,
        "old_stage8_results_reused_as_context_utility_labels": False,
        "runtime_status": ("pending" if len(runtime) != len(run.cfg["seeds"]) else
                           "fixed_paths_profiled_with_observed_exclusivity" if runtime_exclusive else
                           "shared_gpu_diagnostic_only; no_controlled_speed_claim"),
        "test_locked": True,
    }
    output = {"experiment": EXPERIMENT, "version": VERSION, "fingerprint": run.manifest["fingerprint"],
              "protocol": PROTOCOL, "results": aggregate_results, "runtime": runtime,
              "decision": decision, "test_locked": True}
    atomic_json(run.reports / "summary.json", output)
    atomic_json(run.reports / "decision.json", decision)
    with (run.reports / "quality_cost.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["split", "adapter", "seed", "view", "weighted_f1", "macro_f1", "accuracy",
                  "mean_prompt_tokens", "mean_completion_tokens", "parse_success", "strict_format_success"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, variants in evaluations.items():
            for variant, rows in variants.items():
                for row in rows:
                    for view in VIEWS:
                        writer.writerow(dict(split=split, adapter=variant, seed=row["seed"], view=view,
                                             **{k: row[view][k] for k in fields[4:]}))
    lines = ["# Stage 8C.1 — 当前句与最近三句的成对上下文实验", "",
             "> FAST=当前句；SLOW=当前句+此前最多3个话语轮次。两者均只输出一个情绪标签。", "",
             "同一 adapter 内只改变历史内容；Stage7 两种 adapter 冻结。本阶段没有新增训练或可部署自适应策略。", "",
             "| Adapter | View | validation wF1 | Macro F1 | Accuracy | Input tokens | Output tokens | Strict format |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for variant in run.cfg["adapter_variants"]:
        for view in VIEWS:
            m = aggregate_results["validation"][variant][view]
            lines.append(f"| {variant} | {view} | {m['weighted_f1']['mean']:.4f} ± {m['weighted_f1']['std']:.4f} | "
                         f"{m['macro_f1']['mean']:.4f} | {m['accuracy']['mean']:.4f} | "
                         f"{m['mean_prompt_tokens']['mean']:.2f} | {m['mean_completion_tokens']['mean']:.2f} | "
                         f"{100*m['strict_format_success']['mean']:.2f}% |")
    for variant in run.cfg["adapter_variants"]:
        entry = aggregate_results["validation"][variant]
        rates, ci = entry["paired_rates"], entry["paired_wf1_bootstrap"]
        lines.extend(["", f"{variant}: 历史纠错 {rates['history_helps']['mean']:.2%}；历史引错 "
                      f"{rates['history_harms']['mean']:.2%}。SLOW−FAST wF1={ci['mean_seed_weighted_f1_difference']:+.4f}，"
                      f"成对 dialogue bootstrap 95% CI [{ci['ci95'][0]:+.4f}, {ci['ci95'][1]:+.4f}]。", ""])
    if runtime:
        lines.extend(["", "| Seed | Batch | FAST response ms | SLOW response ms | SLOW/FAST time | Agreement FAST/SLOW |",
                      "|---|---:|---:|---:|---:|---:|"])
        for seed, value in runtime.items():
            for batch, data in value["summary"].items():
                f, s = data["fast"], data["slow"]
                lines.append(f"| {seed} | {batch} | {f['mean_batch_emotion_response_latency_ms']:.2f} | "
                             f"{s['mean_batch_emotion_response_latency_ms']:.2f} | "
                             f"{data['observed_slow_over_fast_time_ratio']:.3f} | "
                             f"{f['prediction_agreement_with_quality_batch1']:.2%}/{s['prediction_agreement_with_quality_batch1']:.2%} |")
    lines.extend(["", f"状态：`{decision['status']}`。", "",
                  f"计时状态：`{decision['runtime_status']}`。", "",
                  "延迟包含提示构建、分词、传输、生成至结束和解析；不含模型加载、网络和排队。batch4 表中为整批返回延迟，不能除以4后称为单条响应延迟。",
                  "", "Oracle 和 matched random 仅由已完成的两条路径离线混合，没有测得自适应流程速度；先跑 FAST 再升级 SLOW 时必须计入两次调用。",
                  "", "Stage7 的 adapter 训练历史不同，跨 adapter 差异不能单独归因于额外标注。此处直接生成标签也不能与 Stage7 候选标签打分结果当作同协议横比。",
                  "", "这次 validation 仍是开发集检查；正确率 oracle 使用 gold，仅表示纠错机会，不能证明模型能够提前识别这些样本。MELD test 保持锁定。", ""])
    (run.reports / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"Report saved: {run.reports / 'summary.md'}")
    return decision


def status(run: Run) -> dict:
    return {"experiment": EXPERIMENT, "run_name": run.name, "smoke": run.smoke,
            "predictions": {str(seed): {variant: {split: {view: run.completed(seed, variant, split, view)
                            for view in VIEWS} for split in SPLITS} for variant in run.cfg["adapter_variants"]}
                            for seed in run.cfg["seeds"]},
            "runtime": {str(seed): (run.reports / f"seed{seed}_{run.cfg['primary_adapter_variant']}_runtime.json").exists()
                        for seed in run.cfg["seeds"]},
            "report": (run.reports / "summary.json").exists(), "test_locked": True}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-name", default="validation")
    parser.add_argument("--limit", type=int, default=None, help="smoke* runs only")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "report", "status", "final-test"):
        sub.add_parser(name)
    p = sub.add_parser("predict")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--adapter", choices=("emotion_base", "full_supervised"), required=True)
    p.add_argument("--split", choices=(*SPLITS, "both"), default="both")
    p = sub.add_parser("profile")
    p.add_argument("--seed", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "final-test":
        raise SystemExit("MELD test is locked. Stage8C.1 implements no test-set read or evaluation.")
    run = Run(args.config, args.run_name, args.limit)
    with run_lock(run):
        run.preflight()
        if args.command == "preflight":
            print(json.dumps(run.manifest, ensure_ascii=False, indent=2))
        elif args.command == "predict":
            predict(run, args.seed, args.adapter, list(SPLITS) if args.split == "both" else [args.split])
        elif args.command == "profile":
            profile(run, args.seed)
        elif args.command == "report":
            print(json.dumps(report(run), ensure_ascii=False, indent=2))
        elif args.command == "status":
            print(json.dumps(status(run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
