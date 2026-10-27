from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import apply_overrides, load_config, resolve_project_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pfa-erc", description="Independent PFA-ERC / HiTFT reproduction"
    )
    parser.add_argument("--config", default="configs/meld.yaml", help="YAML experiment config")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Repeatable override, e.g. --set model.future_steps=6",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0, ...")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Convert a raw dataset to canonical JSONL")
    subparsers.add_parser(
        "audit-context",
        help="Audit whether tokenizer truncation removes the current utterance",
    )
    subparsers.add_parser("extract", help="Cache contextual RoBERTa CLS embeddings")
    subparsers.add_parser("toy-features", help="Create random offline features for a smoke test")
    subparsers.add_parser("train-hitft", help="Pretrain the pseudo-future predictor")
    evaluate_hitft = subparsers.add_parser(
        "evaluate-hitft", help="Evaluate transition-aware future forecasting"
    )
    evaluate_hitft.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    subparsers.add_parser("train-erc", help="Freeze HiTFT and train the ERC model")
    subparsers.add_parser("train-seeds", help="Train/evaluate all configured random seeds")
    subparsers.add_parser(
        "train-state", help="Train the causal probabilistic emotion-state tracker"
    )
    subparsers.add_parser(
        "train-state-seeds", help="Train/evaluate the state tracker over configured seeds"
    )
    evaluate_state = subparsers.add_parser(
        "evaluate-state", help="Evaluate a probabilistic emotion-state checkpoint"
    )
    evaluate_state.add_argument("--checkpoint", default=None)
    evaluate_state.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    boundary_oracle = subparsers.add_parser(
        "analyze-boundary-oracle",
        help="Measure boundary margins and Anchor/Context oracle upper bounds",
    )
    boundary_oracle.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    subparsers.add_parser(
        "train-boundary", help="Train a boundary-aware residual reranker"
    )
    subparsers.add_parser(
        "train-boundary-seeds", help="Train/evaluate boundary rerankers over seeds"
    )
    evaluate_boundary = subparsers.add_parser(
        "evaluate-boundary", help="Evaluate a boundary-aware reranker checkpoint"
    )
    evaluate_boundary.add_argument("--checkpoint", default=None)
    evaluate_boundary.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    evaluate_boundary_seeds = subparsers.add_parser(
        "evaluate-boundary-seeds",
        help="Evaluate existing boundary seed checkpoints without retraining",
    )
    evaluate_boundary_seeds.add_argument(
        "--split", default="test", choices=["train", "validation", "test"]
    )
    subparsers.add_parser(
        "train-roberta",
        help="Fine-tune RoBERTa on isolated current utterances (Experiment 3A)",
    )
    evaluate_roberta = subparsers.add_parser(
        "evaluate-roberta",
        help="Evaluate an Experiment 3A RoBERTa checkpoint",
    )
    evaluate_roberta.add_argument("--checkpoint", default=None)
    evaluate_roberta.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    evaluate = subparsers.add_parser("evaluate", help="Evaluate a saved PFA-ERC checkpoint")
    evaluate.add_argument("--checkpoint", default=None)
    evaluate.add_argument("--split", default=None, choices=["train", "validation", "test"])
    analyze_corrections = subparsers.add_parser(
        "analyze-corrections",
        help="Compare anchor, context, and fusion corrections per emotion (4A)",
    )
    analyze_corrections.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    influence_audit = subparsers.add_parser(
        "audit-influence",
        help="Run 8A history-turn deletion/replacement influence analysis",
    )
    influence_audit.add_argument("--checkpoint", default=None)
    influence_audit.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    influence_seed_audit = subparsers.add_parser(
        "audit-influence-seeds",
        help="Run 8A over independently trained model-seed checkpoints",
    )
    influence_seed_audit.add_argument(
        "--split", default=None, choices=["train", "validation", "test"]
    )
    subparsers.add_parser(
        "calibrate-erc",
        help="Cross-fit post-hoc temperatures for frozen ERC seed checkpoints",
    )
    evaluate_calibrated = subparsers.add_parser(
        "evaluate-calibrated",
        help="Evaluate a frozen ERC checkpoint with a saved temperature",
    )
    evaluate_calibrated.add_argument("--checkpoint", default=None)
    evaluate_calibrated.add_argument("--calibrator", required=True)
    evaluate_calibrated.add_argument(
        "--split", default=None, choices=["validation", "test"]
    )
    evaluate_final = subparsers.add_parser(
        "evaluate-final-calibrated",
        help="Run the frozen one-time 8E test over all model seeds",
    )
    evaluate_final.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Confirm that model selection is frozen and test may be read",
    )
    evaluate_final.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted 8E run using completed seed reports",
    )
    evaluate_final.add_argument(
        "--check-only",
        action="store_true",
        help="Validate frozen inputs without loading the test split",
    )
    experiment9 = subparsers.add_parser(
        "experiment9",
        help="Run the pre-registered cross-dataset Experiment 9 protocol",
    )
    experiment9.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "prepare",
            "anchors",
            "features",
            "final-validation",
            "static",
            "uniform",
            "wrong-recipient",
            "ablations",
            "calibrate",
            "summarize-validation",
            "all-validation",
            "check-final",
            "final-test",
            "efficiency",
            "summarize-final",
        ],
    )
    experiment9.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Confirm the one-time frozen test read",
    )
    experiment9.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previously interrupted final test",
    )
    experiment14 = subparsers.add_parser(
        "experiment14",
        help="Run the reviewer-closure capacity/baseline/test protocol",
    )
    experiment14.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "prepare",
            "anchors",
            "features",
            "parameter-audit",
            "roletrans",
            "capacity-direct",
            "contextual",
            "speaker-aware",
            "baselines",
            "calibrate",
            "summarize-validation",
            "all-validation",
            "check-final",
            "final-test",
            "summarize-final",
        ],
    )
    experiment14.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Confirm the one-time frozen IEMOCAP Session-5 test read",
    )
    experiment14.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted Experiment 14 final test",
    )
    experiment15 = subparsers.add_parser(
        "experiment15",
        help=(
            "Run validation-only symmetric calibration and paired "
            "dialogue bootstrap"
        ),
    )
    experiment15.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "symmetric-calibration",
            "paired-bootstrap",
            "all-validation",
            "summarize",
        ],
    )
    calibration_attribution = subparsers.add_parser(
        "calibration-attribution",
        help="Run Experiment 11 symmetric Anchor/Final calibration controls",
    )
    calibration_attribution.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "validation",
            "check-final",
            "final-attribution",
            "summarize",
        ],
    )
    calibration_attribution.add_argument(
        "--confirm-posthoc-test",
        action="store_true",
        help="Confirm use of stored frozen-test predictions for sensitivity analysis",
    )
    experiment13 = subparsers.add_parser(
        "experiment13",
        help="Run the pre-registered context-utility routing protocol",
    )
    experiment13.add_argument(
        "--stage",
        required=True,
        choices=[
            "check",
            "train-teachers",
            "build-train-labels",
            "build-validation-labels",
            "train-routers",
            "evaluate-validation",
            "summarize-validation",
            "all-validation",
            "check-final",
            "final-test",
            "summarize-final",
        ],
    )
    experiment13.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Confirm the one-time frozen Experiment 13 test read",
    )
    experiment13.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted identical Experiment 13 final test",
    )
    subparsers.add_parser("run", help="Run prepare, extract, HiTFT, ERC, and evaluation")
    subparsers.add_parser("inspect", help="Print resolved config and model parameter counts")
    return parser


def _resolved_config(arguments: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(arguments.config).expanduser().resolve()
    project_root = config_path.parent.parent
    config = apply_overrides(load_config(config_path), arguments.overrides)
    return resolve_project_paths(config, project_root)


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _inspect(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    from .models import HiTFTForecaster, PFAERC

    hitft = HiTFTForecaster(config)
    pfa = PFAERC(config, num_classes=7, hitft=hitft if config["model"]["use_hitft"] else None)

    def count(module: torch.nn.Module, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in module.parameters()
            if not trainable_only or parameter.requires_grad
        )

    return {
        "config": config,
        "parameters": {
            "hitft_total": count(hitft),
            "pfa_erc_total_including_frozen_hitft": count(pfa),
            "pfa_erc_trainable": count(pfa, trainable_only=True),
        },
    }


def main() -> None:
    arguments = _parser().parse_args()
    config = _resolved_config(arguments)
    if arguments.command == "prepare":
        from .data import prepare_dataset

        _print(prepare_dataset(config))
    elif arguments.command == "audit-context":
        from .features import audit_context_truncation

        _print(audit_context_truncation(config))
    elif arguments.command == "extract":
        from .features import extract_features

        _print(extract_features(config, device=arguments.device))
    elif arguments.command == "toy-features":
        from .features import make_toy_feature_store

        _print(make_toy_feature_store(config))
    elif arguments.command == "train-hitft":
        from .training import train_hitft

        _print(train_hitft(config, device=arguments.device))
    elif arguments.command == "evaluate-hitft":
        from .training import evaluate_hitft

        _print(evaluate_hitft(config, device=arguments.device, split=arguments.split))
    elif arguments.command == "train-erc":
        from .training import train_erc

        _print(train_erc(config, device=arguments.device))
    elif arguments.command == "train-seeds":
        from .training import train_seed_sweep

        _print(train_seed_sweep(config, device=arguments.device))
    elif arguments.command == "train-state":
        from .training import train_probabilistic_state

        _print(train_probabilistic_state(config, device=arguments.device))
    elif arguments.command == "train-state-seeds":
        from .training import train_probabilistic_state_seeds

        _print(train_probabilistic_state_seeds(config, device=arguments.device))
    elif arguments.command == "evaluate-state":
        from .training import evaluate_probabilistic_state

        _print(
            evaluate_probabilistic_state(
                config,
                device=arguments.device,
                checkpoint_path=arguments.checkpoint,
                split=arguments.split,
            )
        )
    elif arguments.command == "analyze-boundary-oracle":
        from .training import analyze_boundary_oracle

        _print(
            analyze_boundary_oracle(
                config,
                device=arguments.device,
                split=arguments.split,
            )
        )
    elif arguments.command == "train-boundary":
        from .training import train_boundary_reranker

        _print(train_boundary_reranker(config, device=arguments.device))
    elif arguments.command == "train-boundary-seeds":
        from .training import train_boundary_reranker_seeds

        _print(train_boundary_reranker_seeds(config, device=arguments.device))
    elif arguments.command == "evaluate-boundary":
        from .training import evaluate_boundary_reranker

        _print(
            evaluate_boundary_reranker(
                config,
                device=arguments.device,
                checkpoint_path=arguments.checkpoint,
                split=arguments.split,
            )
        )
    elif arguments.command == "evaluate-boundary-seeds":
        from .training import evaluate_boundary_reranker_seeds

        _print(
            evaluate_boundary_reranker_seeds(
                config,
                device=arguments.device,
                split=arguments.split,
            )
        )
    elif arguments.command == "train-roberta":
        from .training import train_roberta

        _print(train_roberta(config, device=arguments.device))
    elif arguments.command == "evaluate-roberta":
        from .training import evaluate_roberta

        _print(
            evaluate_roberta(
                config,
                device=arguments.device,
                checkpoint_path=arguments.checkpoint,
                split=arguments.split,
            )
        )
    elif arguments.command == "evaluate":
        from .training import evaluate_erc

        _print(
            evaluate_erc(
                config,
                device=arguments.device,
                checkpoint_path=arguments.checkpoint,
                split=arguments.split,
            )
        )
    elif arguments.command == "analyze-corrections":
        from .training import analyze_correction_sweep

        _print(
            analyze_correction_sweep(
                config,
                device=arguments.device,
                split=arguments.split,
            )
        )
    elif arguments.command == "audit-influence":
        from .training import audit_context_influence

        _print(
            audit_context_influence(
                config,
                device=arguments.device,
                checkpoint_path=arguments.checkpoint,
                split=arguments.split,
            )
        )
    elif arguments.command == "audit-influence-seeds":
        from .training import audit_context_influence_seeds

        _print(
            audit_context_influence_seeds(
                config,
                device=arguments.device,
                split=arguments.split,
            )
        )
    elif arguments.command == "calibrate-erc":
        from .training import calibrate_erc_seed_sweep

        _print(calibrate_erc_seed_sweep(config, device=arguments.device))
    elif arguments.command == "evaluate-calibrated":
        from .training import evaluate_calibrated_erc

        _print(
            evaluate_calibrated_erc(
                config,
                calibrator_path=arguments.calibrator,
                checkpoint_path=arguments.checkpoint,
                device=arguments.device,
                split=arguments.split,
            )
        )
    elif arguments.command == "evaluate-final-calibrated":
        from .training import evaluate_final_calibrated_seed_sweep

        _print(
            evaluate_final_calibrated_seed_sweep(
                config,
                device=arguments.device,
                confirm=arguments.confirm_final_test,
                resume=arguments.resume,
                check_only=arguments.check_only,
            )
        )
    elif arguments.command == "experiment9":
        from .training import run_experiment9

        _print(
            run_experiment9(
                config,
                stage=arguments.stage,
                device=arguments.device,
                confirm_final_test=arguments.confirm_final_test,
                resume=arguments.resume,
            )
        )
    elif arguments.command == "experiment14":
        from .training import run_experiment14

        _print(
            run_experiment14(
                config,
                stage=arguments.stage,
                device=arguments.device,
                confirm_final_test=arguments.confirm_final_test,
                resume=arguments.resume,
            )
        )
    elif arguments.command == "experiment15":
        from .training import run_experiment15

        _print(
            run_experiment15(
                config,
                stage=arguments.stage,
                device=arguments.device,
            )
        )
    elif arguments.command == "calibration-attribution":
        from .training import run_calibration_attribution

        _print(
            run_calibration_attribution(
                config,
                stage=arguments.stage,
                device=arguments.device,
                confirm_posthoc_test=arguments.confirm_posthoc_test,
            )
        )
    elif arguments.command == "experiment13":
        from .training import run_experiment13

        _print(
            run_experiment13(
                config,
                stage=arguments.stage,
                device=arguments.device,
                confirm_final_test=arguments.confirm_final_test,
                resume=arguments.resume,
            )
        )
    elif arguments.command == "run":
        from .data import prepare_dataset
        from .features import extract_features
        from .training import evaluate_erc, train_erc, train_hitft

        result = {
            "prepare": prepare_dataset(config),
            "extract": extract_features(config, device=arguments.device),
            "hitft": train_hitft(config, device=arguments.device),
            "erc": train_erc(config, device=arguments.device),
            "evaluation": evaluate_erc(config, device=arguments.device),
        }
        _print(result)
    elif arguments.command == "inspect":
        _print(_inspect(config))
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
