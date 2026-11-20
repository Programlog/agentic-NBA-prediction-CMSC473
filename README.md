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
See [final H2 results](experiments/h2/02_final_test_100/README.md) and
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


### H1: Computer-vision experiments (ongoing)

The exploratory H1 pipeline has tested BasketEvent clip matching, YOLO pose
extraction, CoTracker3 point tracking, frame-by-frame visual audits, and
translation-only camera-motion compensation. Results include incomplete/failed
pose and video-retrieval attempts as well as successful tracking runs.

On a **10-shot, single-game** camera-motion pilot, **10/10 clips processed** and
all camera estimates passed the algorithm's internal validity checks. Mean
shooter displacement changed from **0.02473 to 0.02814** after correction;
mean defender displacement changed from **0.04035 to 0.04395** across six
annotated defenders. These changes **do not demonstrate improved accuracy**.
No held-out H1 predictive comparison is complete.

- [Chronological H1 experiment log (including failures)](docs/experiments/H1_EXPERIMENT_LOG.md)
- [Detailed camera-motion pilot and 10-shot results](docs/experiments/h1_camera_motion_results.md)

## H2: Single-agent vs. multi-agent LLMs (completed)

All H2 protocols, scripts, results and follow-ups are in
[`experiments/h2/`](experiments/h2/README.md), one folder per experiment.

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

The [pre-registered final protocol](experiments/h2/02_final_test_100/protocol.md)
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

The earlier [30-shot validation pilot](experiments/h2/01_validation_pilot_30/README.md)
did **not** show the same benefit; the final held-out evaluation is the
pre-registered H2 result. The model digest used in the validation pilot
could not be verified against the final test model digest.

### Reproduce and inspect H2

- [Frozen protocol and configuration](experiments/h2/02_final_test_100/protocol.md)
- [Detailed final results and limitations](experiments/h2/02_final_test_100/README.md)
- [Full generated report](experiments/h2/02_final_test_100/report/report.md)
- [Per-model evaluation metrics (CSV)](experiments/h2/02_final_test_100/report/combined_metrics.csv)
- [Paired bootstrap comparisons (JSON)](experiments/h2/02_final_test_100/report/paired_comparisons.json)
- [Per-shot predictions (CSV)](experiments/h2/02_final_test_100/report/predictions.csv)
- [Final execution script](experiments/h2/02_final_test_100/run_llm_final.sh)
- [Result figures](experiments/h2/02_final_test_100/report/figures/) (regenerate with `experiments/h2/02_final_test_100/make_figures.py`)

The final runner checks the frozen implementation and prepared-data
configuration before using `--split test --final`. Its documented
setup expects the recorded model digest and prepared data artifacts;
running it with different inputs or code is a **new experiment**, not
a reproduction of the frozen run.

The frozen run was re-executed on November 19, 2026 and reproduced all 600
committed predictions exactly (see
[Reproduction](experiments/h2/02_final_test_100/README.md#reproduction)). Use commit
`64ee854` to rerun it; later `agents.py` changes intentionally fail the runner's
frozen-hash check.

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
[4-shot pilot report](experiments/h2/03_disagreement_aggregation/pilot_4_shots.md)
for setup, all condition metrics, and limitations.

A follow-up **10-shot validation run is currently incomplete**: 28 of 40
planned condition predictions were saved successfully, covering **7 matched
shots from 4 games**. On those seven shots, the disagreement-aware
aggregator achieved **0.196086 Brier** versus **0.209529** for the original
aggregator (paired difference **−0.013443**, approximately **6.42% lower**).
These are preliminary, unconfirmed findings from an incomplete run, not
a final 10-shot comparison. See the
[partial validation report](experiments/h2/03_disagreement_aggregation/validation_10_partial.md).

### Follow-up: fixing LLM overconfidence (Brier score)

The LLM conditions' poor Brier scores come mainly from **overconfidence**:
on test, `llm_multi` predicted an average make probability of 0.60 when 42%
of shots were made, even though its ROC-AUC (0.687) matched logistic
regression's. A validation-only follow-up (100 shots, 50 games, same model,
seed and data) tested two fixes:

- **Anchoring** each agent with training-split make rates for the shot's
  distance and defender spacing lowered validation Brier from **0.273 to
  0.238** (−0.035, 95% CI [−0.067, −0.004]). That beats logistic regression
  (0.245) and the constant baseline (0.247), close to XGBoost (0.232).
- **Recalibrating** the frozen test predictions with a map fitted only on
  validation lowered `llm_multi` from **0.248 to 0.227**. With **both** arms
  recalibrated, the multi-agent and budget-matched single LLM are **tied**
  (+0.001 [−0.006, +0.010]), so the published H2 advantage appears to come
  mostly from lower overconfidence. Both still trail XGBoost (0.206).

Anchoring has not yet been applied to the single-agent control, so it
cannot yet be credited to the multi-agent design. See the
[follow-up report](experiments/h2/04_brier_followup/README.md).

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
