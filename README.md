# SCS: Context Supervision for Emotion Recognition in Conversation

Research code for studying how additional annotations can supervise the use of dialogue history in LLM-based emotion recognition in conversation (ERC).

The central question is **when recent context helps, and which previous utterance provides useful evidence**. The method uses structured supervision for context need, evidence selection, and evidence attributes. The current experiment package focuses on MELD with 1,500 annotated training-set targets.

## Method

The implementation uses **Qwen2.5-7B-Instruct with LoRA** and a two-step inference procedure:

1. **Context planning:** Given the current utterance and up to three preceding utterance turns, predict whether history is needed and select at most one evidence turn. A full plan also describes the selected evidence through structured attributes.
2. **Emotion classification:** Score candidate emotion-label continuations using the current utterance, the selected evidence, and the predicted attributes, depending on the experimental condition.

The planner sees the three-turn candidate window even when it ultimately selects no history. Context selection therefore controls the evidence supplied to the classifier; it does not eliminate history processing by the planner.

Only causal history is used. Teacher annotations supervise training and are not supplied during development-set inference. The method does not train free-text chain-of-thought reasoning.

## Code structure

| Path | Purpose |
|---|---|
| `src/pfa_erc/training/llm_evidence_supervision_stage7.py` | Core training, prompts, context planning, and emotion scoring |
| `MELD_Stage7R_run1500_v1/run_stage7r.py` | Experiment orchestration and input/version checks |
| `MELD_Stage7R_run1500_v1/config.yaml` | Model, data, training, and evaluation settings |
| `MELD_Stage7R_run1500_v1/data_checks.py` | Annotation and source-input validation |
| `MELD_Stage7R_run1500_v1/statistics_report.py` | Paired statistical comparisons and reports |
| `MELD_Stage7R_run1500_v1/run.sh` | Command-line launcher with logs and exit status |
| `MELD_Stage7R_run1500_v1/launch_tmux.sh` | tmux launcher that exits when the task finishes |
| `MELD_Stage7R_run1500_v1/tests/` | Experiment checks |

## Experimental conditions

| Condition | Description |
|---|---|
| Current only | Emotion classification using the current utterance |
| Always recent-3 | Classification using the current utterance and all available preceding three turns |
| Uncertainty-based selection | Current-only prediction, with recent history used according to a train-calibrated uncertainty threshold |
| Label-only continuation | Additional training on emotion labels without structured context supervision |
| Evidence supervision | Supervision for context need and evidence selection |
| Full supervision (SCS) | Evidence supervision plus structured evidence attributes |
| Shuffled supervision | Control with shuffled additional annotations |

The legacy code name `generic_reasoning` refers to the **label-only continuation** control; it does not indicate free-text reasoning supervision. The `full_no_impact` condition uses the full adapter but hides structured attributes from the classifier.

Additional mechanism comparisons hold the full adapter and classifier prompt fixed while varying only the history supplied: current only, recent-3, or selected evidence.

## Data and evaluation

- **Dataset:** MELD; seven emotion categories.
- **Additional annotations:** 1,200 targets for supervised training and 300 held-out targets for teacher-agreement diagnostics. Both sets come from official MELD training dialogues and are dialogue-disjoint.
- **Development evaluation:** 1,109 utterances from 114 official validation dialogues.
- **Training seeds:** 42, 43, 44, 45, and 46; the data split is held fixed.
- **Metrics:** Weighted F1, macro F1, accuracy, and per-emotion performance.
- **Statistics:** Paired dialogue bootstrap for confidence intervals and a paired permutation test for the primary comparison. Results average seed-specific metrics rather than forming a probability ensemble.

The official test set remains locked. Development results are not test-set benchmark scores. Teacher self-audit measures internal consistency, not independently verified human annotation accuracy.

## Requirements and data preparation

The experiments use Python 3.12, a CUDA-enabled PyTorch environment, and packages including Transformers, PEFT, Accelerate, NumPy, scikit-learn, and PyYAML. GPU training and inference require the Qwen2.5-7B-Instruct model weights. tmux is optional.

The commands below assume the existing project environment and processed MELD data are available. Run them from the repository root, with `src/` and `MELD_Stage7R_run1500_v1/` at the same level. A core-source archive alone does not include a complete installation environment, model weights, or the original dataset.

Required inputs are:

- `data/processed/meld/train.jsonl` and `data/processed/meld/validation.jsonl`, in the project's dialogue format.
- Annotation files under `MELD_Stage7R_run1500_v1/data/`, as specified in the configuration.
- Shared project modules, including `pfa_erc.data`.

Obtain MELD from its original distribution and follow its terms of use. Input validation checks annotation text, causal history, dialogue splits, and file hashes before training.

## Running the MELD experiment

Activate the existing environment, check inputs, and prepare training records:

```bash
source .venv-server/bin/activate

bash MELD_Stage7R_run1500_v1/run.sh test
bash MELD_Stage7R_run1500_v1/run.sh preflight
bash MELD_Stage7R_run1500_v1/run.sh prepare
```

Run a small smoke test before the full experiment. Replace `0` with an available physical GPU index:

```bash
GPU_ID=0 bash MELD_Stage7R_run1500_v1/run.sh smoke --seed 42

GPU_ID=0 SESSION_NAME=scs_meld \
  bash MELD_Stage7R_run1500_v1/launch_tmux.sh run
```

The full run trains and evaluates all five configured seeds. The tmux session closes automatically on completion or failure; logs and exit codes remain available.

Inspect progress and export results:

```bash
bash MELD_Stage7R_run1500_v1/run.sh status

cat artifacts/llm_evidence_supervision_stage7r1500/meld/reports/validation_summary.md

bash MELD_Stage7R_run1500_v1/run.sh export
```

Outputs are saved under `artifacts/llm_evidence_supervision_stage7r1500/meld/`; execution logs are saved under `artifacts/logs/`.

## Documentation

See [the experiment guide](MELD_Stage7R_run1500_v1/README.md) for operational details and [the protocol](MELD_Stage7R_run1500_v1/EXPERIMENT_PROTOCOL.md) for the experimental design.
