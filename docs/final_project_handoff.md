# Final Project Handoff and Submission Readiness

**Project:** NBA Shot Prediction with Computer Vision and Agentic AI

**Course:** CMSC473 — Machine Learning Capstone

**Date:** November 28, 2026

**Status:** Primary H1/H2 experiments complete; final academic submission and end-to-end reproduction require separate verification.

## 1. Purpose

This document provides a final handoff for the project's code, experiments, research findings, supporting artifacts, reproducibility instructions, and known limitations.

It consolidates existing documentation without changing the original experiments or claiming that unverified work has been completed.

The repository contains two separate primary research studies:

- **H1:** Does adding computer-vision-derived features improve traditional NBA shot-make probability prediction?
- **H2:** Does multi-agent LLM orchestration improve predictions compared with a budget-matched single-agent LLM and traditional supervised-learning baselines?

Both primary experiments have been completed and reported. However, their datasets and experimental conditions differ, so their numerical results must not be combined into a single model leaderboard.

## 2. Research Deliverables

| Deliverable | Final status | Primary evidence |
| --- | --- | --- |
| Logistic regression baseline | Completed | `README.md` |
| XGBoost baseline | Completed | `README.md` |
| Video-linked shot and box-feature processing | Implemented, with measurement limitations | `docs/h1_results.md` |
| CV-enhanced XGBoost evaluation | Completed | `docs/h1_results.md` |
| Multi-agent LLM prediction system | Completed | `experiments/h2/02_final_test_100/README.md` |
| Budget-matched single-agent LLM comparison | Completed | `experiments/h2/02_final_test_100/README.md` |
| Frozen held-out H1 evaluation | Completed | `docs/h1_test_protocol.md` |
| Frozen held-out H2 evaluation | Completed | `experiments/h2/02_final_test_100/protocol.md` |
| Calibration, robustness, and failure analysis | Substantially completed | `docs/h1_experiments.md`, `experiments/h2/README.md` |
| Unified four-model experiment on identical shots | Not demonstrated | `docs/proposal_completion_audit.md` |
| Complete independent reproduction from raw data | Not established | `docs/final_reproducibility_audit.md` |
| Final submitted report, slides, and live demo | Not verified by repository audits | `docs/final_research_claim_audit.md` |

A completed experiment can support a negative or inconclusive hypothesis result. Completion status must not be confused with statistical significance or practical usefulness.

## 3. H1: Computer Vision Research Findings

The primary H1 experiment used BasketEvent 2025–26 shots matched to NBA records.

The study compared XGBoost with and without nine measurements derived from broadcast-video tracking boxes.

### Evaluation population

- Development: 7,726 shots across 184 games.
- Frozen held-out test: 1,546 shots across 34 games.
- Evaluation emphasized probabilistic prediction quality.

### Frozen held-out Brier scores

| Model | Brier score |
| --- | ---: |
| Constant baseline | 0.2075 |
| Logistic regression | 0.2009 |
| Standard XGBoost | 0.2010 |
| XGBoost + CV box features | 0.2003 |

Primary comparison:

- CV-enhanced XGBoost minus standard XGBoost: approximately -0.00067.
- 95% game-bootstrap interval: [-0.00237, +0.00096].

The added box-derived features produced a small improvement in held-out Brier score, consistent in direction with development findings.

However, the held-out interval includes zero, so the held-out test alone does not establish a statistically significant improvement.

### Measurement limitations

The frozen model used bounding-box-derived measurements and proxies rather than precise physical shooting mechanics.

Independent validation found that box-derived release height agreed only weakly with pose-derived release height within jump shots (Spearman correlation approximately 0.32).

Release timing and player/ball tracking also have documented errors.

An exploratory release-angle measurement was investigated but was not included in the frozen predictive model.

Therefore, H1 results should not be interpreted as proof that precise physical release mechanics were accurately measured or caused the observed improvement.

### H1 reference files

- [Full H1 results](h1_results.md)
- [Frozen H1 protocol](h1_test_protocol.md)
- [H1 measurement checks](h1_measurement_check.md)
- [H1 effect estimates](../reports/basketevent_h1_test/h1_effect_estimates.png)
- [H1 reliability diagram](../reports/basketevent_h1_accuracy/reliability.png)

## 4. H2: Agentic AI Research Findings

The primary frozen H2 experiment used Kaggle 2014–15 NBA shot logs.

The held-out comparison evaluated 100 shots from 50 games with matched input information for the competing systems.

The multi-agent and budget-matched single-agent conditions each used three LLM calls per shot.

### Frozen held-out Brier scores

| Model | Brier score |
| --- | ---: |
| XGBoost | 0.2056 |
| Logistic regression | 0.2198 |
| Constant baseline | 0.2448 |
| Multi-agent LLM | 0.2483 |
| Budget-matched single-agent LLM | 0.2735 |
| One-call single-agent LLM | 0.2931 |

Primary comparison:

- Multi-agent minus budget-matched single-agent: approximately -0.0252.
- 95% game-bootstrap interval: [-0.0441, -0.0073].

The frozen experiment favored the multi-agent LLM over the budget-matched single-agent LLM.

However, XGBoost remained the stronger predictor.

### Exploratory follow-up findings

Later experiments investigated:

- LLM overconfidence and calibration.
- Stronger hosted models.
- Additional shooter-history information.
- Prompt improvements.
- Similar-shot evidence.
- Traditional model predictions supplied as agent tools.

These follow-up studies showed that the multi-agent advantage was configuration-dependent and weakened or disappeared under stronger or better-calibrated LLM settings.

Tool-assisted systems could approach XGBoost performance by using its predictions, but this does not demonstrate independent LLM superiority over XGBoost.

The original H2 decision remains tied to the frozen primary experimental conditions.

### H2 reference files

- [Frozen H2 results](../experiments/h2/02_final_test_100/README.md)
- [Frozen H2 protocol](../experiments/h2/02_final_test_100/protocol.md)
- [Stronger-model follow-up](../experiments/h2/05_stronger_model/README.md)
- [Prompt-tuning follow-up](../experiments/h2/07_prompt_tuning/README.md)
- [H2 model comparison](../experiments/h2/02_final_test_100/report/figures/h2_1_brier_all_models.png)
- [H2 effect intervals](../experiments/h2/02_final_test_100/report/figures/h2_2_effect_intervals.png)

## 5. Original Proposal Versus Final Implementation

The original proposal envisioned precise shooting-mechanics features and a broader comparison involving traditional ML, CV-enhanced ML, single-agent LLMs, and multi-agent LLMs.

The implemented work includes two separate controlled experiments rather than one unified four-model experiment.

Important deviations include:

1. H1 primarily used box-derived features rather than validated physical elbow angles, release angles, or contest heights.
2. The frozen H2 experiment used tabular shot information without the final H1 video-box features.
3. H1 and H2 used different datasets, seasons, features, and evaluation populations.
4. The original H2 frozen test used a local Ollama model; stronger hosted-model comparisons were evaluated later.
5. A comprehensive, comparable cost-versus-prediction-quality analysis across all model families was not established.

These deviations should be explicitly described in the final academic report and presentation.

The repository does not establish whether all four proposed model families would rank the same way on identical shots with equivalent information.

See [Proposal Completion Audit](proposal_completion_audit.md).

## 6. Repository Verification

Previous local audits established:

- Python 3.11 environment compatibility.
- Dependency-lock consistency using `uv lock --check`.
- A full local test-suite result of 384 passed and 2 skipped, recorded before the later integrity-checker tests were added.
- Presence and Git tracking of selected frozen H1/H2 result artifacts.
- Successful execution of the main CLI and selected subcommand help interfaces.

Subsequent repository-integrity work added:

- `tools/check_repository_integrity.py`
- `tests/test_repository_integrity.py`
- `.github/workflows/repository-integrity.yml`

The dedicated integrity test suite passed all 16 tests locally.

The integrity checker confirmed the presence of 15 required artifacts and found no invalid links among 186 inspected Markdown references at the time of validation.

The GitHub Actions integrity workflow also passed on the corrected PR #50.

These results establish the status of the checks at the times they were run. They are not proof of complete experimental reproduction.

### Local verification commands

From the repository root:

```bash
python tools/check_repository_integrity.py
python -m pytest tests/test_repository_integrity.py -q
```

For the broader project test suite in the configured environment:

```bash
uv run --frozen pytest -q
```

The full suite should be rerun before final academic handoff, with its current result recorded.

### Known quality limitations

The earlier reproducibility audit reported repository-wide Ruff lint findings and 15 files requiring formatting.

The dedicated integrity-checker files passed Ruff locally, but that does not resolve the previously documented repository-wide findings.

These limitations should remain disclosed unless later checks demonstrate that they have been corrected.

## 7. Reproduction Requirements

### Traditional baseline

The README provides commands for installing the project and running the tabular baseline.

A complete baseline run may require the Kaggle dataset or an explicitly supplied local input file.

### H1

Full H1 reproduction may require:

- BasketEvent 2025–26 data.
- Hugging Face authentication.
- Access to associated clips and tracking information.
- Computer-vision dependencies and model weights.
- Appropriate computing resources.

### H2

Full H2 reproduction may require:

- Kaggle 2014–15 shot logs.
- Ollama and the specified local LLM.
- The original frozen runner and configuration.
- Locally generated intermediate inputs.

The frozen primary H2 runner is associated with commit `64ee854`; current agent code should not automatically be assumed equivalent.

Raw datasets and some generated artifacts are excluded from Git.

A fresh environment installation and complete regeneration of the original H1/H2 results have not been independently established.

See [Final Reproducibility Audit](final_reproducibility_audit.md).

## 8. Demonstration Handoff

For a dependable demonstration, prioritize existing committed research artifacts rather than initiating resource-intensive inference live.

Recommended sequence:

1. Introduce the two research questions and why their datasets differ.
2. Show the repository's main CLI help interface.
3. Present the H1 held-out effect estimates and reliability diagram.
4. Present the H2 frozen model comparison and paired effect intervals.
5. Explain the primary statistical conclusions.
6. Discuss the measurement, generalization, and reproducibility limitations.

The existing notebook is available at `notebooks/demo.ipynb`, but its execution was not independently verified during the final demo-readiness audit.

See [Final Demo and Execution Readiness](final_demo_readiness.md).

## 9. Final Academic Submission Checklist

The following tasks require confirmation by the project team or course staff.

- [ ] Confirm that two separate controlled H1/H2 evaluations satisfy the accepted final scope.
- [ ] Confirm whether an identical-shot four-model comparison is required.
- [ ] Finalize and review the written report against frozen experimental results.
- [ ] Finalize and review the presentation slides against the research-claim audit.
- [ ] Explain the difference between planned physical shooting measurements and implemented box-derived proxies.
- [ ] Report H1's nonsignificant held-out interval.
- [ ] Report the frozen H2 multi-agent advantage without claiming superiority over XGBoost.
- [ ] Distinguish confirmatory experiments from exploratory follow-ups.
- [ ] Disclose the differences between the H1 and H2 datasets.
- [ ] Describe computational costs using measured information only.
- [ ] Run and record final repository checks.
- [ ] Rehearse the actual demonstration and verify required files are accessible.
- [ ] Confirm submission requirements, deadlines, final artifacts, and team responsibilities.

Unchecked items represent work requiring confirmation, not necessarily work that has not happened outside the repository.

## 10. Handoff Navigation

| Resource | Location |
| --- | --- |
| Main project overview | [README](../README.md) |
| Proposal completion assessment | [Proposal completion audit](proposal_completion_audit.md) |
| Scientific claim boundaries | [Research-claim audit](final_research_claim_audit.md) |
| Reproduction and environment limitations | [Reproducibility audit](final_reproducibility_audit.md) |
| Demonstration instructions | [Demo-readiness guide](final_demo_readiness.md) |
| H1 research evidence | [H1 results](h1_results.md) |
| H2 research evidence | [H2 primary results](../experiments/h2/02_final_test_100/README.md) |
| Automated integrity validation | [Integrity checker](../tools/check_repository_integrity.py) |
| Integrity regression tests | [Test suite](../tests/test_repository_integrity.py) |
| Continuous integration | [GitHub Actions workflow](../.github/workflows/repository-integrity.yml) |

## 11. Final Assessment

The repository contains completed primary H1/H2 research experiments, traditional machine-learning baselines, computer-vision-derived features, a multi-agent prediction system, comparative evaluations, research audits, recorded artifacts, and automated repository-integrity verification.

Its primary conclusions are qualified:

- H1: Video-box-derived measurements yielded a small, directionally consistent improvement in Brier score, but the held-out confidence interval included zero and measurement validity remains limited.
- H2: Multi-agent orchestration improved on a budget-matched single-agent LLM under the frozen small-model configuration, but did not outperform XGBoost and did not consistently retain its advantage in later studies.

The repository should not be presented as having completed a unified four-model experiment, validated precise physical shooting mechanics, or independently reproduced every result from raw data.

The final report, slides, live demonstration, and course-specific submission acceptance require separate confirmation.

This handoff is documentation-only and does not alter any frozen research results, configurations, predictions, or model weights.