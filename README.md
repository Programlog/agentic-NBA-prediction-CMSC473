# NBA Shot Prediction

CMSC473 project comparing tabular shot prediction, release-time video features, and
single versus multi-agent LLMs using free tools.

Tabular baselines, video annotation/pose-extraction tools, and local Ollama/CrewAI
agent pipelines are implemented. The **final H2 held-out experiment is complete**:
multi-agent orchestration improved Brier score over a call-budget-matched
single-agent control on 100 test shots, although XGBoost remained the stronger
shot predictor. See [final H2 results](docs/experiments/llm_final_results.md)
and [initial baseline results](reports/day1.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11.
On macOS, install XGBoost's runtime with `brew install libomp`.

```bash
uv sync --frozen
uv run --frozen nba-shots baseline --output artifacts/baseline
```

Downloads the [Kaggle shot logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs),
cleans the data, creates chronological game splits, and trains constant, logistic
regression, and XGBoost models. Evaluates validation data only.

Use a new output directory for each run. To supply a downloaded CSV, add
`--input /path/to/shot_logs.csv`. Runs save prepared data, splits, models, predictions,
metrics, and calibration plots. Data and generated artifacts are ignored by Git.

## Experiments

Configure features and training in `configs/default.json`. Splits keep whole games
and dates together; preprocessing uses training data only and excludes outcome fields.
Reserve test evaluation (`--split test --final`) until configurations and prompts are frozen.

Install optional CV and agent dependencies:

```bash
uv sync --frozen --all-extras
.venv/bin/nba-shots --help
```

### Video Annotation Tool

Use the browser tool to inspect exact decoded frames, set release frames (`R`),
run YOLO detection, and tag the shooter and contest defender. Prepare the manifest
using the [experiment instructions](docs/experiments.md) first:

```bash
uv run nba-shots annotate --manifest data/video/manifest.csv --videos data/video/clips
```

See [experiment instructions](docs/experiments.md) for video collection, release
annotations, pose extraction, local LLMs, and comparisons on matching shots.


## H2: Single-agent vs. multi-agent LLMs (completed)

**Research question:** With the same pre-release shot features and number of
model calls, does a specialized multi-agent pipeline predict shot success
better than a single-analyst LLM?

The [pre-registered final protocol](docs/experiments/llm_final_protocol.md)
was frozen before test inference. The November 19, 2026 evaluation used
**100 held-out Kaggle 2014–15 shots from 50 games** (two shots per game), a
deterministic seed of 42, and local Ollama running `qwen3:4b-instruct`.
Player-history and computer-vision features were not included in this
test. The three LLM conditions completed **300/300 predictions with zero
failures**.

### Held-out model results

All models were scored on the **same 100 shots**. Lower Brier score and log
loss are better; higher ROC-AUC and accuracy are better.

| Model | Brier score ↓ | Log loss ↓ | ROC-AUC ↑ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: |
| XGBoost (reference) | **0.2056** | **0.5990** | **0.7414** | **0.740** |
| Logistic regression (reference) | 0.2198 | 0.6299 | 0.6810 | 0.670 |
| Constant-probability baseline | 0.2448 | 0.6827 | 0.5000 | 0.580 |
| **Multi-agent LLM** (`llm_multi`) | **0.2483** | **0.6821** | **0.6874** | **0.510** |
| Budget-matched single agent (`llm_single_budget`) | 0.2735 | 0.7451 | 0.6636 | 0.530 |
| One-call single agent (`llm_single`) | 0.2931 | 0.7915 | 0.6537 | 0.440 |

### Primary hypothesis test

The primary H2 comparison is multi-agent versus the **budget-matched**
single agent, not multi-agent versus XGBoost. Both primary conditions used
**300 LLM calls** across the 100 shots (three calls per shot in this
configuration).

| Paired comparison | Brier difference (multi − control) | 95% game-bootstrap CI | Conclusion |
| --- | ---: | --- | --- |
| Multi-agent vs. budget-matched single agent (**primary**) | **−0.0252** | **[−0.0441, −0.0073]** | Supports H2 |
| Multi-agent vs. one-call single agent (secondary) | −0.0448 | [−0.0699, −0.0212] | Descriptive improvement |

Intervals use **1,000 paired bootstrap resamples of games** with seed 42.
A negative Brier difference favors multi-agent. Because the primary
interval excludes zero, the result meets the protocol's decision rule
for H2.

**Interpretation and limitations:** Specializing and aggregating agents
helped relative to the LLM controls, but **none of the LLMs outperformed
XGBoost or logistic regression** on this dataset. The multi-agent LLM also
had a slightly worse Brier score than the constant baseline. All LLM
conditions overpredicted the probability of a made shot; reduced
overconfidence accounts for much of the multi-agent improvement.
The study used one small local model and only 100 held-out shots, so
its effect size should not be generalized without further replication.

The earlier [30-shot validation pilot](docs/experiments/llm_30shot_validation.md)
did **not** show the same benefit; the final held-out evaluation is the
pre-registered H2 result. The model digest used in the validation pilot
could not be verified against the final test model digest.

### Reproduce and inspect H2

- [Frozen protocol and configuration](docs/experiments/llm_final_protocol.md)
- [Detailed final results and limitations](docs/experiments/llm_final_results.md)
- [Full generated report](reports/llm_final_test_100/report.md)
- [Per-model evaluation metrics (CSV)](reports/llm_final_test_100/combined_metrics.csv)
- [Paired bootstrap comparisons (JSON)](reports/llm_final_test_100/paired_comparisons.json)
- [Per-shot predictions (CSV)](reports/llm_final_test_100/predictions.csv)
- [Final execution script](scripts/run_llm_final.sh)

The final runner checks the frozen implementation and prepared-data
configuration before using `--split test --final`. Its documented
setup expects the recorded model digest and prepared data artifacts;
running it with different inputs or code is a **new experiment**, not
a reproduction of the frozen run.

### Follow-up: disagreement-aware aggregation

An additional *validation-only* experiment investigates whether giving
the final multi-agent aggregator explicit specialist-disagreement
statistics (mean, range, and standard deviation) improves predictions
relative to the original aggregator. This is **new, exploratory work**:
it is not part of the frozen H2 hypothesis test, and no performance
claim is made here until a completed validation analysis is available.

## Checks

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

Use `uv run --frozen --all-extras pytest -q` to include the CrewAI integration tests.

## Resources

- [Baseline results](reports/day1.md)
- [Demo notebook](notebooks/demo.ipynb)
- [Team roadmap](docs/roadmap.md)
- [Troubleshooting](docs/troubleshooting.md)
