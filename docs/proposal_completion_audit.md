# Proposal Completion Audit

**Project:** NBA Shot Prediction with Agentic AI and Computer Vision
**Audit date:** 2026-11-25
**Baseline:** main at c01507d (merged PR #30)
**Scope:** Original written proposal, 15-slide proposal presentation,
and existing H1/H2 research evidence.

## Purpose

This document distinguishes completed implementation, completed experiments
with limitations, changes from the original proposal, and work that has not
yet been verified.

An unsuccessful hypothesis is not an unfinished experiment. Likewise,
a working implementation is not proof that every proposed measurement
was successfully obtained or independently validated.

## 1. Completed core research

### Traditional ML baselines

Status: Complete.

The repository contains logistic regression and XGBoost baselines,
data preprocessing, held-out evaluations, and associated metrics.

Evidence:
- README.md
- docs/h1_results.md
- reports/basketevent_h1_test/summary.json
- experiments/h2/02_final_test_100/README.md

### H1: Computer vision features

Status: Primary experiment complete; measurement limitations remain.

The frozen H1 study compared XGBoost with and without nine
video-box-derived measurements on the same shot cohort.

Development cross-validation:
- 7,726 shots across 184 games.
- Brier difference (CV minus baseline): approximately -0.00093.
- 95% interval: [-0.00169, -0.00008].

Frozen held-out test:
- 1,546 shots across 34 games.
- XGBoost Brier: 0.2010.
- XGBoost + boxes Brier: 0.2003.
- Brier difference: approximately -0.00067.
- 95% interval: [-0.00237, +0.00096].

Interpretation:
The added features produced a small improvement in Brier score.
The held-out interval includes zero, so the held-out test alone
does not establish a statistically significant improvement.

Evidence:
- docs/h1_results.md
- docs/h1_test_protocol.md
- docs/h1_measurement_check.md

### H2: Agentic AI

Status: Primary experiment complete.

The frozen H2 study compared a multi-agent LLM with a
budget-matched single-agent LLM and XGBoost on held-out shots.

Frozen 100-shot test:
- XGBoost Brier: 0.2056.
- Multi-agent Brier: 0.2483.
- Budget-matched single-agent Brier: 0.2735.
- Multi-agent minus single-agent Brier: approximately -0.0252.

The frozen experiment favored the multi-agent architecture over
the matched single-agent control, but XGBoost performed better.

Later exploratory experiments with stronger models, calibration,
and tool-assisted prompting substantially qualified the initial
multi-agent advantage.

Evidence:
- experiments/h2/02_final_test_100/README.md
- experiments/h2/05_stronger_model/README.md
- experiments/h2/07_prompt_tuning/README.md

### Validation and failure analysis

Status: Substantially complete.

The repository contains unit tests, video measurement checks,
H1 failure-case investigations, robustness experiments, and
H2 calibration and model-comparison studies.

A failed measurement check is a reported research result,
not evidence that no validation occurred.

Evidence:
- docs/h1_measurement_check.md
- docs/h1_failure_cases.md
- docs/h1_box_error_round.md
- experiments/h2/README.md

## 2. Deviations and limitations requiring disclosure

### Proposed versus implemented CV measurements

The proposal named release angle, elbow angle, defender
contest height, defender positioning, and related mechanics.

The final H1 predictive experiment primarily used box-derived
features, including release height, jump, ball rise, shooter
speed, and defender-location or contest proxies.

An approximate image-based release-angle measurement passed
its exploratory checks but was not added to the frozen model.

Box-derived measurements are not calibrated physical
measurements of true shooting mechanics or 3D contest distance.

Evidence:
- docs/h1_results.md
- docs/h1_release_angle_pilot.md

### Video-event matching and release reliability

Video-linked shots and box trajectories were successfully
used in the H1 study.

However, release-frame timing and player identification can
be wrong. Independent pose agreement for box-derived release
height was weak within jump shots (Spearman rho approximately
0.32).

These limitations restrict how confidently the features can
be interpreted as physical release mechanics.

Evidence:
- docs/h1_measurement_check.md
- docs/h1_failure_cases.md
- docs/h1_manual_video_feasibility_audit.md

### Separate H1 and H2 evaluation cohorts

The proposal describes both separate hypothesis tests and
comparisons involving traditional XGBoost, CV-enhanced
XGBoost, a single LLM, and the multi-agent system.

H1 used BasketEvent/NBA 2025-26 data.
H2 used Kaggle 2014-15 shot logs.

Both primary comparisons were completed separately.
The repository does not demonstrate a unified four-model
evaluation on identical shots with matched available inputs.

Decision needed:
Determine whether the separate controlled experiments satisfy
the intended final scope. Do not directly rank H1 and H2 scores
across different datasets.

Evidence:
- README.md
- docs/h1_results.md
- experiments/h2/README.md

## 3. Items requiring final verification

### Computational and inference costs

Status: Partially documented; comprehensive comparison unverified.

H2 exploratory experiments record API and inference expenses,
but the proposal also calls for comparing costs alongside
predictive quality.

Verify whether the final report includes comparable cost and
resource information for the evaluated systems.

Do not invent GPU-hour estimates, API charges, or equivalent
hardware costs that were not actually measured.

Evidence:
- experiments/h2/07_prompt_tuning/README.md
- experiments/h2/README.md

### Reproducibility

Status: Instructions and recorded results exist; latest local
end-to-end reproduction not yet independently verified.

Checks:
- Confirm the current source and dependency lock file.
- Run the project test suite without modifying frozen results.
- Confirm committed result paths and figure references exist.
- Identify external datasets, credentials, and ignored artifacts
  needed for reproduction.
- Confirm demonstration commands work on the intended machine.

Evidence:
- README.md
- docs/h1_results.md
- experiments/h2/README.md

### Final written report, slides, and demo

Status: Not verified in this audit.

Confirm submission readiness and consistency with the frozen
experimental results, including negative findings and
measurement limitations.

## 4. Priorities

1. Verify the latest checkout, tests, recorded results, and demo.
2. Confirm whether a unified four-model evaluation is required.
3. Ensure the final materials explain changed CV measurements.
4. Confirm computational costs are described accurately.
5. Include release-timing and measurement limitations.

Optional further work:
- Independently adjudicate selected video release annotations.
- Explore release-angle features with development data only.
- Investigate event-selection failures using a fresh audit.

The frozen test sets must not be repeatedly reused to choose
or tune new model designs.

## 5. Audit conclusion

The core H1 and H2 implementations and primary experiments
are complete.

Remaining concerns concern fidelity to the proposal,
measurement reliability, comparability across datasets,
reproducibility, and final communication.

These concerns should not be represented as entirely missing
implementations or as proof that the research hypotheses
were confirmed.
