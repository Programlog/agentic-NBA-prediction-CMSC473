# NBA Shot Prediction

## NBA Shot Prediction with Agentic AI and Computer Vision

### Problem

NBA shot success depends on a variety of factors, such as shot location,
defender proximity, game context, and player history or statistics. These
factors are often captured in traditional NBA shot data. However, useful
information may still be missed, including shooting mechanics, release
angle, defender positioning, and contest height, which can be investigated
by analyzing video clips.

Since individual shot outcomes are random by nature, our goal is **not to
perfectly predict whether every shot goes in**. Instead, we aim to
**estimate shot quality and shot-success probability at the instant of
release**.

We analyze the Computer Vision and Agentic AI systems separately to
determine whether improvements come from additional computer-vision
features, the architecture of the agentic system, or both.

### Research Questions

- **H1 — Computer Vision:** Do shot-mechanic and defender features derived
  from computer vision improve NBA shot-success predictions beyond
  traditional NBA statistics?
- **H2 — Agentic AI:** When utilizing the same features, does a multi-agent
  LLM pipeline improve NBA shot-success predictions compared with
  conventional supervised-learning models such as XGBoost and a single
  LLM using the same available inputs?

### Approach

- **Data:** Link shot logs, player statistics, tracking data, and video
  clips by game ID, event number, and game clock where data is available.
- **Baseline models:** Use logistic regression and XGBoost on traditional
  shot-log features such as shot distance, defender distance, shot clock,
  and game clock.
- **Computer Vision:** Use YOLO detection and YOLO-Pose to investigate
  additional shooting-mechanics and defender-contest features from video
  frames around the release. Extracted image measurements are proxies,
  not automatically verified physical release angles or heights.
- **Agentic system:** Specialized agents analyze the available features
  from different perspectives, including game context, player history,
  and shot mechanics/defense. Their outputs feed into a prediction agent,
  which produces the final shot-success probability. Agents with missing
  inputs are skipped.
- **Evaluation:** Compare models on held-out games using Brier score,
  log loss, accuracy, and ROC-AUC, with emphasis on probability
  calibration and analyzing failure cases.

These describe the project's overall approach, **not a claim that every
planned data source, video feature, or H1 experiment has been completed**.

### Current Status

The tabular baseline, video-annotation and pose-extraction tools, and
local Ollama/CrewAI agent pipeline have been implemented. The **final
held-out H2 comparison is complete**: the multi-agent system improved on
the budget-matched single-LLM baseline, but did not outperform XGBoost.
See [final H2 results](docs/experiments/llm_final_results.md) and
[initial baseline results](reports/day1.md).

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

**H2: Agents add value beyond the features themselves.** Specialized
agents separately analyze the available shot features from different
perspectives, including game context and shot mechanics/defense. A
prediction agent takes the outputs from these specialized agents to
produce a final shot-success probability.

The proposal calls for comparing the multi-agent system against both
XGBoost on the same features and a single LLM using the same inputs.
For the final held-out H2 decision, the team selected the
**call-budget-matched single-LLM comparison** as the primary test;
XGBoost was included as a reference rather than a criterion for
supporting H2.

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

The next experiment builds on the proposal's idea that specialized
agents analyze the available information from different perspectives
before a prediction agent generates the final shot-success probability.
It tests whether explicitly measuring disagreements among those
specialized agents (mean, range, and standard deviation of their
estimates) helps the prediction agent interpret conflicting evidence.

A completed **4-shot validation pilot** reported a Brier score of
**0.2657** for disagreement-aware aggregation versus **0.2767** for
the original multi-agent aggregator (a 0.0110 absolute, 3.98% relative
reduction), with all 16 predictions completing successfully. However,
accuracy dropped from 50% to 25%, and four shots are far too few for
a reliable performance conclusion. See the
[4-shot pilot report](docs/experiments/adaptive_aggregation_pilot_4.md)
for setup, all condition metrics, and limitations.

A follow-up **10-shot validation run is currently incomplete**: 28 of 40
planned condition predictions were saved successfully, covering **7 matched
shots from 4 games**. On those seven shots, the disagreement-aware
aggregator achieved **0.196086 Brier** versus **0.209529** for the original
aggregator (paired difference **−0.013443**, approximately **6.42% lower**).
These are preliminary, unconfirmed findings from an incomplete run, not
a final 10-shot comparison. See the
[partial validation report](docs/experiments/adaptive_aggregation_validation_10_partial.md).

These exploratory runs do **not** change the frozen held-out H2 results.

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
