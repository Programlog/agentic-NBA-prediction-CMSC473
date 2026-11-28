# Final Demo and Execution Readiness

**Project:** NBA Shot Prediction with Computer Vision and Agentic AI

**Audit date:** November 27, 2026

**Scope:** Command-line interface accessibility, existing demonstration materials, committed H1/H2 visualizations, external execution requirements, and final presentation readiness.

## 1. Purpose

This document establishes which project components are ready for a lightweight demonstration and which require additional datasets, models, credentials, or computing resources.

The goal is to provide a practical guide for demonstrating the completed research without unnecessarily rerunning computationally expensive experiments or modifying frozen results.

This audit distinguishes between:

- Commands whose help interfaces were successfully executed.
- Existing notebooks and visualizations whose paths were verified.
- Full experimental pipelines that were not executed during this audit.

A working command-line help interface does not establish that the corresponding experimental workflow can execute successfully from beginning to end.

## 2. Verified Environment

The demo-readiness checks were performed in a local Windows environment using:

- Python 3.11.4
- An existing project virtual environment at `.venv`
- The `nba_shots` Python package
- The Git feature branch `feature/final-demo-readiness`

The project defines its supported Python version in `pyproject.toml` as:

`>=3.11,<3.13`

The package exposes the following console entry point:

```toml
[project.scripts]
nba-shots = "nba_shots.cli:main"
```

The preceding reproducibility audit also verified dependency-lock consistency using `uv lock --check`.

A completely fresh environment installation was not performed as part of this demo-readiness audit.

## 3. Command-Line Interface Verification

The main command-line help interface was successfully executed using:

```powershell
.\.venv\Scripts\python.exe -m nba_shots --help
```

The command displayed the project's available subcommands, including:

- `download`
- `prepare`
- `train`
- `cross-validate`
- `baseline`
- `evaluate`
- `compare`
- `collect-recent`
- `make-manifest`
- `prepare-cohort`
- `fetch-videos`
- `extract-cv`
- `annotate`
- `collect-basketevent`
- `prepare-basketevent`
- `predict-llm`
- `report`

The following individual help interfaces were also successfully executed:

```powershell
.\.venv\Scripts\python.exe -m nba_shots baseline --help

.\.venv\Scripts\python.exe -m nba_shots evaluate --help

.\.venv\Scripts\python.exe -m nba_shots report --help

.\.venv\Scripts\python.exe -m nba_shots predict-llm --help
```

All four commands displayed their corresponding argument descriptions.

### Verified argument requirements

| Command | Verified arguments | Purpose |
|---|---|---|
| `baseline` | `--input`, `--output`, `--config` | Prepare and train tabular baseline models |
| `evaluate` | `--run`, `--shots`, `--split`, `--final` | Evaluate saved models |
| `report` | `--predictions`, `--output`, `--selection` | Combine prediction outputs into reports |
| `predict-llm` | `--shots`, `--cv`, `--output`, `--model`, `--split`, `--final`, `--limit`, `--seed`, `--games`, `--conditions`, `--num-predict`, `--ollama-url` | Run local LLM prediction conditions |

The help output confirms that:

- `evaluate` requires `--run` and `--shots`.
- `report` requires `--predictions` and `--output`.
- `predict-llm` requires `--shots` and `--output`.
- `baseline` accepts optional input, output, and configuration paths.

The CLI help screens worked successfully. The underlying training, evaluation, reporting, and LLM inference operations were not executed during this audit.

## 4. Existing Demonstration Materials

The repository contains an existing demonstration notebook:

`notebooks/demo.ipynb`

The file was confirmed to exist locally and is tracked by Git.

This notebook provides a starting point for demonstrating the project's tabular modeling workflow.

However, the notebook was not executed during this audit. Its current outputs, execution order, and external data dependencies were not independently verified.

Therefore, the notebook should be described as an available demonstration resource rather than a fully validated, immediately executable demonstration.

## 5. H1: Computer Vision Demonstration

### Existing results

The following H1 visualizations were verified to exist:

- `reports/basketevent_h1_test/h1_effect_estimates.png`
- `reports/basketevent_h1_accuracy/reliability.png`

These figures provide a lightweight way to demonstrate the project's H1 results without retraining the models.

The supporting documentation includes:

- `docs/h1_results.md`
- `docs/h1_test_protocol.md`
- `docs/h1_experiments.md`
- `docs/h1_measurement_check.md`

### Recommended H1 demonstration

A lightweight demonstration can use the existing committed artifacts to explain:

1. The research question comparing tabular XGBoost against XGBoost with video-derived box features.
2. How the additional features were derived from broadcast-video tracking boxes.
3. The difference between baseline and CV-enhanced prediction quality.
4. The held-out Brier-score comparison and confidence interval.
5. The reliability diagram and why classification accuracy is not the primary evaluation metric.
6. The limitations of release detection and box-derived physical proxies.

### Key H1 findings

The frozen held-out test used 1,546 shots.

| Model | Brier score | Accuracy |
|---|---:|---:|
| Constant baseline | 0.2075 | 70.70% |
| Logistic regression | 0.2009 | 70.89% |
| XGBoost | 0.2010 | 70.70% |
| XGBoost + CV boxes | 0.2003 | 70.76% |

The CV-enhanced XGBoost model achieved a small reduction in Brier score compared with standard XGBoost.

The held-out confidence interval for the difference included zero, so this result does not establish a statistically significant improvement on the held-out test.

The accuracy values were very similar across models. Because most shots were misses, predicting nearly every shot as a miss already achieved approximately 70.7% accuracy.

These results should be presented as evidence of a small improvement in probabilistic prediction quality, not a large increase in classification accuracy.

### Requirements for full H1 execution

A full computer-vision demonstration may require:

- BasketEvent 2025–26 source data.
- Hugging Face authentication for data collection.
- Available broadcast-video clips and accompanying tracking information.
- Computer-vision dependencies and model weights.
- Appropriate processing resources.

Raw data and generated video artifacts may be excluded from Git.

The following commands are documented project interfaces, but their full execution was not verified in this audit:

```powershell
.\.venv\Scripts\python.exe -m nba_shots collect-basketevent --help

.\.venv\Scripts\python.exe -m nba_shots prepare-basketevent --help

.\.venv\Scripts\python.exe -m nba_shots extract-cv --help
```

For actual reproduction instructions, consult `docs/h1_results.md`.

The recommended final demonstration uses the existing committed figures instead of attempting a fresh video-processing run during the presentation.

## 6. H2: Agentic AI Demonstration

### Existing results

The following frozen H2 visualizations were verified to exist:

- `experiments/h2/02_final_test_100/report/figures/h2_1_brier_all_models.png`
- `experiments/h2/02_final_test_100/report/figures/h2_2_effect_intervals.png`

Additional H2 visualizations are tracked under:

- `experiments/h2/02_final_test_100/report/figures/`
- `experiments/h2/04_brier_followup/results/figures/`
- `experiments/h2/05_stronger_model/results/figures/`
- `experiments/h2/06_shooter_history/results/figures/`
- `experiments/h2/07_prompt_tuning/results/figures/`

These materials allow the project to demonstrate the frozen multi-agent experiment and later exploratory comparisons without invoking an LLM.

### Recommended H2 demonstration

A lightweight presentation can:

1. Explain the single-agent and multi-agent LLM conditions.
2. Explain the budget-matching procedure.
3. Show the frozen 100-shot held-out Brier-score comparison.
4. Show the paired confidence interval for the primary hypothesis.
5. Compare the LLM conditions with XGBoost.
6. Discuss overconfidence and the exploratory follow-up results.

### Key H2 findings

The frozen H2 test used 100 shots from 50 games.

| Model | Brier score |
|---|---:|
| XGBoost | 0.2056 |
| Logistic regression | 0.2198 |
| Constant baseline | 0.2448 |
| Multi-agent LLM | 0.2483 |
| Budget-matched single-agent LLM | 0.2735 |

The multi-agent LLM outperformed the budget-matched single-agent system on the frozen held-out comparison.

The paired Brier difference was approximately -0.0252, with a 95% game-bootstrap interval of [-0.0441, -0.0073].

However, XGBoost achieved the lowest Brier score among these models.

This supports a multi-agent advantage over the matched LLM control under the frozen configuration, not superiority over traditional machine learning.

### Requirements for full H2 execution

Full H2 inference may require:

- The original Kaggle 2014–15 NBA shot-log data.
- A local Ollama installation and running service.
- The specified local model, `qwen3:4b-instruct`.
- Appropriate model files and computational resources.
- The original frozen H2 runner and configuration.

The primary frozen H2 experiment used an earlier implementation associated with commit `64ee854`.

Later agent implementations should not be assumed equivalent to that frozen system.

The following help command was successfully verified:

```powershell
.\.venv\Scripts\python.exe -m nba_shots predict-llm --help
```

This confirms that the interface is available. It does not confirm that Ollama inference was successfully executed.

For reproduction instructions, consult:

`experiments/h2/02_final_test_100/README.md`

## 7. Existing README Instructions

The README documents the following setup and baseline commands:

```powershell
uv sync --frozen

uv run --frozen nba-shots baseline --output artifacts/baseline
```

It also documents optional dependencies and CLI discovery:

```powershell
uv sync --frozen --all-extras

uv run --frozen nba-shots --help
```

These commands are documented entry points. Their complete execution was not verified during this demo-readiness audit.

The README also provides links to:

- H1 reproduction instructions.
- H2 reproduction instructions.
- General experiment instructions.
- Initial tabular baseline results.
- The demonstration notebook.

The existing README remains the primary source of setup instructions.

## 8. Recommended Lightweight Demo Sequence

The following demonstration sequence relies on existing research artifacts instead of running expensive models.

### Part A: Project overview

Introduce the overall research question:

Can computer-vision features and LLM-based agent collaboration improve NBA shot-make probability estimation?

Explain that H1 and H2 are distinct studies rather than one unified comparison.

### Part B: Code and CLI

From the project root, demonstrate the available command-line interface:

```powershell
.\.venv\Scripts\python.exe -m nba_shots --help
```

Explain how the CLI separates data preparation, model training, evaluation, computer vision, and local LLM inference.

### Part C: H1 results

Open:

`reports/basketevent_h1_test/h1_effect_estimates.png`

Then open:

`reports/basketevent_h1_accuracy/reliability.png`

Explain the small reduction in Brier score and the uncertainty in the held-out confidence interval.

### Part D: H2 results

Open:

`experiments/h2/02_final_test_100/report/figures/h2_1_brier_all_models.png`

Then open:

`experiments/h2/02_final_test_100/report/figures/h2_2_effect_intervals.png`

Explain the multi-agent versus budget-matched single-agent comparison and why XGBoost remained stronger.

### Part E: Limitations

Conclude by acknowledging:

- H1 and H2 use different shot cohorts.
- CV-derived box measurements are not precise physical shooting measurements.
- Release timing and tracking reliability remain limited.
- The multi-agent advantage does not generalize automatically to every LLM configuration.
- Full raw-data reproduction was not verified during this audit.

This sequence provides a presentation-ready walkthrough without requiring live model inference.

## 9. Demo-Readiness Checklist

| Component | Verified status |
|---|---|
| Main CLI help interface | Passed |
| Baseline CLI help | Passed |
| Evaluate CLI help | Passed |
| Report CLI help | Passed |
| LLM CLI help | Passed |
| Demo notebook path | Exists |
| H1 effect-estimate figure | Exists |
| H1 reliability diagram | Exists |
| H2 Brier-score figure | Exists |
| H2 effect-interval figure | Exists |
| Full baseline training run | Not executed |
| Fresh H1 video-processing run | Not executed |
| Fresh H2 Ollama inference run | Not executed |
| Demo notebook execution | Not verified |
| Fresh dependency installation | Not verified |
| Final presentation rehearsal | Not performed |

## 10. Conclusion

The repository contains a functioning command-line help interface, an existing demo notebook, and committed visualizations for its frozen H1 and H2 research results.

These materials are sufficient to prepare a lightweight, artifact-based demonstration of the project's research findings.

However, full pipeline execution, notebook execution, and fresh-environment setup were not verified during this audit.

The recommended final demonstration should prioritize the existing validated research artifacts and clearly distinguish them from workflows that require additional external data, credentials, or inference resources.

No frozen experiments, models, predictions, detector thresholds, or scientific results were modified as part of this audit.