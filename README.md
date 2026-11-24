# NBA Shot Prediction with Computer Vision and Agentic AI

CMSC473 research project investigating whether video-derived features and multi-agent LLMs improve NBA shot-make probability estimates.

## Research questions and results

| Hypothesis | Comparison | Main finding |
| --- | --- | --- |
| **H1 — Computer vision** | XGBoost with vs. without video-derived features | Small improvement across development and held-out games; held-out confidence interval includes zero. |
| **H2 — Agentic AI** | Multi-agent vs. single-agent LLMs with matched inputs and call budgets; XGBoost reference | Frozen small-model test supports H2 against the budget-matched LLM, but not against XGBoost. Later stronger-model and calibration experiments limit the generality of the advantage. |

Both primary held-out evaluations are complete. Follow-up experiments are reported separately from frozen results.

## H1 — Computer vision

**Dataset:** BasketEvent 2025–26, matched to NBA shot records. The frozen cohort contains unblocked jump shots with detected releases: 7,726 development shots across 184 games and 1,546 test shots across 34 later games.

**Models:** Identical constrained XGBoost designs using traditional shot features (including earlier-date player history), with or without nine box-derived measurements: release height, jump, ball rise, shooter speed, defender distance, nearby defender count, closing speed, defender position relative to basket, and arms-up proxy.

### Held-out results (November 23, 2026)

| Model | Brier ↓ | Log loss ↓ | ROC-AUC ↑ |
| --- | ---: | ---: | ---: |
| Constant | 0.2075 | 0.6056 | 0.5000 |
| Logistic regression | 0.2009 | 0.5903 | **0.6100** |
| XGBoost | 0.2010 | 0.5904 | 0.6048 |
| XGBoost + CV boxes | **0.2003** | **0.5888** | 0.6089 |

| Evaluation | Brier difference (CV − baseline) | 95% game-bootstrap CI |
| --- | ---: | --- |
| Development cross-validation | −0.00093 | [−0.00169, −0.00008] |
| Frozen held-out test (six seeds averaged) | −0.00067 | [−0.00237, +0.00096] |

The improvement is consistent in direction but not statistically significant on the held-out test alone. Release height drives much of the gain, yet agrees only weakly with independent pose measurements within jump shots (Spearman 0.32). The cohort over-represents misses. Box features are image-based proxies, not calibrated physical measurements.

[Full H1 results](docs/h1_results.md) · [Frozen protocol](docs/h1_test_protocol.md) · [Test metrics](reports/basketevent_h1_test/summary.json) · [Predictions](reports/basketevent_h1_test/test_predictions_seed_mean.csv) · [Effect estimates](reports/basketevent_h1_test/h1_effect_estimates.png)

### H1 experiment index

The following index covers the numbered experiments in the [full H1 log](docs/h1_experiments.md), including unsuccessful approaches.

| # | Experiment | Stage |
| --- | --- | --- |
| 01 | [Original comparison on one validation split (runs 1 and 2)](docs/h1_experiments.md#1-original-comparison-on-one-validation-split-runs-1-and-2) | Development / measurement |
| 02 | [Dropping shots with no detected release (commit 40cd4fd)](docs/h1_experiments.md#2-dropping-shots-with-no-detected-release-commit-40cd4fd) | Development / measurement |
| 03 | [Camera perspective and two more defender features (commit a682b75)](docs/h1_experiments.md#3-camera-perspective-and-two-more-defender-features-commit-a682b75) | Development / measurement |
| 04 | [Blind review of the labels (25 clips)](docs/h1_experiments.md#4-blind-review-of-the-labels-25-clips) | Development / measurement |
| 05 | [Checking which baseline features are filled](docs/h1_experiments.md#5-checking-which-baseline-features-are-filled) | Development / measurement |
| 06 | [Box features alone](docs/h1_experiments.md#6-box-features-alone) | Development / measurement |
| 07 | [Dataset composition](docs/h1_experiments.md#7-dataset-composition) | Development / measurement |
| 08 | [Stronger tabular baseline (run 3)](docs/h1_experiments.md#8-stronger-tabular-baseline-run-3) | Development / measurement |
| 09 | [Jump shots only (run 3)](docs/h1_experiments.md#9-jump-shots-only-run-3) | Development / measurement |
| 10 | [Open-shot validity check](docs/h1_experiments.md#10-open-shot-validity-check) | Development / measurement |
| 11 | [Glitch filter for player boxes (cohort v4, run 4)](docs/h1_experiments.md#11-glitch-filter-for-player-boxes-cohort-v4-run-4) | Development / measurement |
| 12 | [Date-blocked cross-validation and a constrained model (primary test)](docs/h1_experiments.md#12-date-blocked-cross-validation-and-a-constrained-model-primary-test) | Development / measurement |
| 13 | [Defender features only](docs/h1_experiments.md#13-defender-features-only) | Development / measurement |
| 14 | [Feature ablation](docs/h1_experiments.md#14-feature-ablation) | Development / measurement |
| 15 | [Robustness checks](docs/h1_experiments.md#15-robustness-checks) | Development / measurement |
| 16 | [Blocked shots](docs/h1_experiments.md#16-blocked-shots) | Development / measurement |
| 17 | [CoTracker3 point tracking (pilot, 25 reviewed clips)](docs/h1_experiments.md#17-cotracker3-point-tracking-pilot-25-reviewed-clips) | Development / measurement |
| 18 | [YOLO pose wrists (pilot, 25 reviewed clips)](docs/h1_experiments.md#18-yolo-pose-wrists-pilot-25-reviewed-clips) | Development / measurement |
| 19 | [Looking for more data](docs/h1_experiments.md#19-looking-for-more-data) | Development / measurement |
| 20 | [Improvement round 1: measuring release height better](docs/h1_experiments.md#20-improvement-round-1-measuring-release-height-better) | Development / measurement |
| 21 | [Improvement round 2: more posture from the boxes](docs/h1_experiments.md#21-improvement-round-2-more-posture-from-the-boxes) | Development / measurement |
| 22 | [Rebuilt cohort and seed sensitivity](docs/h1_experiments.md#22-rebuilt-cohort-and-seed-sensitivity) | Development / measurement |
| 23 | [Measurement check against video](docs/h1_experiments.md#23-measurement-check-against-video) | Development / measurement |
| 24 | [Test split (scored once)](docs/h1_experiments.md#24-test-split-scored-once) | Frozen test |
| 25 | [What the box model learned (exploratory, after the test run)](docs/h1_experiments.md#25-what-the-box-model-learned-exploratory-after-the-test-run) | Exploratory follow-up |
| 26 | [Release-angle pilot (exploratory, after the test run)](docs/h1_experiments.md#26-release-angle-pilot-exploratory-after-the-test-run) | Exploratory follow-up |
| 27 | [Failure cases (exploratory, after the test run)](docs/h1_experiments.md#27-failure-cases-exploratory-after-the-test-run) | Exploratory follow-up |

### H1 video pipeline and supporting studies

These separate reports document early feasibility work, annotation, and feature-quality checks; they are not additional confirmatory predictive tests.

| Study | Report |
| --- | --- |
| NBA video retrieval, BasketEvent pilot, YOLO11m/11x, CoTracker3 and visual audits | [Early experiment timeline (including failures)](docs/experiments/H1_EXPERIMENT_LOG.md) |
| Assisted annotation | [Guide](docs/experiments/h1_assisted_annotation_guide.md) |
| Automatic annotation iterations | [V1](docs/experiments/h1_auto_annotation_v1.md) · [V2](docs/experiments/h1_auto_annotation_v2.md) · [V3](docs/experiments/h1_auto_annotation_v3.md) · [Visual review](docs/experiments/h1_auto_annotation_visual_review_v1.md) |
| Manual event and video verification | [Workflow](docs/h1_manual_event_review_workflow.md) · [Feasibility audit](docs/h1_manual_video_feasibility_audit.md) · [Quality gate](docs/h1_video_verification_quality_gate.md) |
| Multi-game data expansion | [Report](docs/experiments/h1_multigame_dataset_expansion.md) |
| Point tracking and frame-by-frame audit | [Tracking guide](docs/experiments/h1_motion_tracking_guide.md) · [Visual audit V2](docs/experiments/h1_tracking_visual_audit_v2.md) |
| Camera-motion compensation (2- and 10-shot pilots) | [Pilot](docs/experiments/h1_camera_motion_pilot.md) · [Results](docs/experiments/h1_camera_motion_results.md) |
| Feature engineering rounds | [Protocols and results](docs/h1_improvement_round.md) |
| Independent YOLO / CoTracker3 measurement checks | [Protocol and results](docs/h1_measurement_check.md) |
| Release-angle measurement pilot (November 24) | [Report](docs/h1_release_angle_pilot.md) |
| Failure-case analysis (November 24) | [Report](docs/h1_failure_cases.md) · [Video review](reports/basketevent_h1_failure_cases/review.csv) |

The release-angle pilot measured 4,779 of 7,726 development shots and passed its three planned measurement checks (median image-based angle 47.1°, split-half Spearman 0.75). It is not a calibrated physical angle and has not been added to the frozen predictive model.

## H2 — Agentic AI

**Dataset:** Kaggle 2014–15 NBA shot logs. The frozen November 19, 2026 test used 100 held-out shots from 50 games, identical inputs for all models, and local Ollama `qwen3:4b-instruct`. Neither CV nor player-history features were included in this primary test.

### Held-out results

| Model | Brier ↓ | Log loss ↓ | ROC-AUC ↑ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | **0.2056** | **0.5990** | **0.7414** | **0.740** |
| Logistic regression | 0.2198 | 0.6299 | 0.6810 | 0.670 |
| Constant | 0.2448 | 0.6827 | 0.5000 | 0.580 |
| Multi-agent LLM | 0.2483 | 0.6821 | 0.6874 | 0.510 |
| Budget-matched single LLM | 0.2735 | 0.7451 | 0.6636 | 0.530 |
| One-call single LLM | 0.2931 | 0.7915 | 0.6537 | 0.440 |

Primary paired Brier difference (multi − budget-matched single): **−0.0252**, 95% game-bootstrap CI **[−0.0441, −0.0073]**. Both used three calls per shot. This supports the frozen H2 decision against the matched LLM control, but no LLM beat XGBoost. Later exploratory results suggest the benefit is largely model- and calibration-dependent.

[Full H2 results](experiments/h2/02_final_test_100/README.md) · [Frozen protocol](experiments/h2/02_final_test_100/protocol.md) · [Metrics](experiments/h2/02_final_test_100/report/combined_metrics.csv) · [Predictions](experiments/h2/02_final_test_100/report/predictions.csv) · [Figures](experiments/h2/02_final_test_100/report/figures/)

### H2 experiment index

| # | Experiment | Status / finding |
| --- | --- | --- |
| 01 | [30-shot validation pilot](experiments/h2/01_validation_pilot_30/README.md) | Complete; no multi-agent advantage |
| 02 | [100-shot held-out test](experiments/h2/02_final_test_100/README.md) | Frozen primary test; multi-agent beats matched single LLM |
| 03 | Disagreement-aware aggregation: [4-shot pilot](experiments/h2/03_disagreement_aggregation/pilot_4_shots.md), [partial 10-shot run](experiments/h2/03_disagreement_aggregation/validation_10_partial.md) | Pilot complete; follow-up partial (7 matched shots); inconclusive |
| 04 | [Overconfidence: anchoring and recalibration](experiments/h2/04_brier_followup/README.md) | Complete, exploratory; calibration removes most multi-agent advantage |
| 05 | [Stronger model: GPT-6 Luna](experiments/h2/05_stronger_model/README.md) | Complete, exploratory; multi and single tie |
| 06 | [Shooter history and player-name contamination](experiments/h2/06_shooter_history/README.md) | Complete, exploratory; small history benefit, XGBoost remains best |
| 07 | [Prompt tuning, similar shots, and XGBoost as a tool](experiments/h2/07_prompt_tuning/README.md) | Complete, exploratory; 1,000 test shots, tool-assisted LLM ties XGBoost largely by copying it |

[Complete H2 experiment directory and reproduction index](experiments/h2/README.md). Only experiment 02 determines the frozen H2 decision.

## Repository structure

| Path | Contents |
| --- | --- |
| [src/nba_shots/](src/nba_shots/) | Data processing, supervised models, vision, annotation, agents and evaluation |
| [configs/](configs/) | Dataset, feature and model configurations |
| [scripts/](scripts/) | Dataset collection, audits and H1 experiment runners |
| [docs/](docs/) | Research protocols, H1 reports and technical documentation |
| [reports/](reports/) | Baseline and H1 metrics, predictions, figures and experiment outputs |
| [experiments/h2/](experiments/h2/) | H2 protocols, scripts and results |
| [notebooks/](notebooks/) | Demo notebook |
| [tests/](tests/) | Automated tests |

## Setup and reproduction

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/getting-started/installation/). On macOS, install XGBoost's runtime with `brew install libomp`.

```bash
uv sync --frozen
uv run --frozen nba-shots baseline --output artifacts/baseline
```

The baseline command downloads the Kaggle shot logs and evaluates validation data only. Use a new output directory per run; add `--input /path/to/shot_logs.csv` for a local CSV.

For optional vision and agent dependencies:

```bash
uv sync --frozen --all-extras
uv run --frozen nba-shots --help
```

- [H1 reproduction instructions](docs/h1_results.md#reproduce) (BasketEvent collection requires Hugging Face authentication).
- [H2 reproduction instructions](experiments/h2/02_final_test_100/README.md#reproduction) (frozen H2 runner uses commit `64ee854`; later agent changes are not equivalent).
- [General experiment instructions](docs/experiments.md), including the video annotation tool.
- [Initial tabular baseline results](reports/day1.md) and [demo notebook](notebooks/demo.ipynb).

Keep whole games and dates together, fit preprocessing on training data only, and reserve test evaluation for frozen configurations. Raw data and many generated artifacts are ignored by Git; early pilot reports may rely on local outputs. H1 and H2 use different cohorts, so their absolute metrics should not be directly compared.

### Checks

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

Use `uv run --frozen --all-extras pytest -q` to include optional integration tests.

## Additional documentation

[Project requirements](docs/PRD.md) · [Technical specification](docs/SPEC.md) · [Roadmap](docs/roadmap.md) · [Troubleshooting](docs/troubleshooting.md)
