# Final Research-Claim Consistency Audit

**Project:** NBA Shot Prediction with Computer Vision and Agentic AI

**Audit date:** November 27, 2026

**Scope:** Original research objectives, frozen H1/H2 experimental evidence, later exploratory findings, and claims requiring qualification in the final presentation and report.

## 1. Purpose

This audit establishes which research claims are supported by the project's recorded experiments and which claims require qualifications.

The purpose is to provide an evidence-based reference for the final presentation and written report, not to introduce new experiments or reinterpret exploratory findings as confirmatory evidence.

No models, datasets, detector thresholds, or frozen experimental results were modified.

## 2. Materials Reviewed

The audit draws on the following repository materials:

- `README.md`
- `docs/proposal_completion_audit.md`
- `docs/final_reproducibility_audit.md`
- `docs/h1_results.md`
- `docs/h1_measurement_check.md`
- `reports/basketevent_h1_test/summary.json`
- `reports/basketevent_h1_accuracy/accuracy.csv`
- `reports/basketevent_h1_accuracy/summary.json`
- `experiments/h2/02_final_test_100/README.md`
- `experiments/h2/05_stronger_model/README.md`
- `experiments/h2/07_prompt_tuning/README.md`

The repository also contains the original proposal PDF.

The repository file search did not identify a tracked final slide deck or final written report. Therefore, this document audits the research claims available for those deliverables rather than asserting that the final submission materials have already been reviewed.

## 3. H1: Computer Vision and Shot Prediction

### Research question

Does adding video-derived measurements improve NBA shot-make probability predictions compared with a traditional tabular machine-learning baseline?

### Experimental design

The frozen H1 experiment evaluated XGBoost models with and without nine measurements derived from tracked broadcast-video bounding boxes.

The evaluation used BasketEvent 2025–26 data:

- Development: 7,726 shots across 184 games
- Held-out test: 1,546 shots across 34 games

The held-out comparison used the same shot cohort and corresponding evaluation protocol.

### Recorded held-out results

| Model | Brier score | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|
| Constant baseline | 0.2075 | 0.6056 | 0.5000 | 70.70% |
| Logistic regression | 0.2009 | 0.5903 | 0.6100 | 70.89% |
| XGBoost | 0.2010 | 0.5904 | 0.6048 | 70.70% |
| XGBoost + CV boxes | 0.2003 | 0.5888 | 0.6089 | 70.76% |

The CV-enhanced XGBoost model achieved the lowest Brier score and log loss of the evaluated models, while logistic regression achieved the highest ROC-AUC and accuracy.

### Primary statistical result

The paired Brier-score difference, calculated as CV-enhanced XGBoost minus baseline XGBoost, was approximately:

- Development cross-validation: -0.00093
- Development 95% interval: [-0.00169, -0.00008]
- Frozen held-out test: -0.00067
- Held-out 95% interval: [-0.00237, +0.00096]

Negative differences favor the CV-enhanced model.

The development interval excluded zero, but the held-out interval included zero.

### Supported interpretation

Adding the video-box features produced a small improvement in probabilistic prediction quality.

The direction of the Brier-score difference was consistent across development and held-out evaluation.

However, the held-out result does not independently establish a statistically significant improvement.

### Accuracy and reliability findings

The later accuracy and reliability analysis used the saved H1 test predictions without refitting the models.

At a fixed probability threshold of 0.5, accuracy differed very little among models.

The constant baseline and standard XGBoost predicted zero makes on the held-out cohort. The CV-enhanced XGBoost model predicted five makes, while logistic regression predicted seven.

Because approximately 29% of shots were made, predicting nearly every shot as a miss already produced about 70.7% accuracy.

Therefore, accuracy is not the most informative primary metric for this study.

Brier score and log loss better reflect the quality of the predicted probabilities.

The reliability diagram provides an additional descriptive calibration check. It does not independently establish a large or statistically significant improvement from the video-derived features.

### Claims requiring qualification

The final materials should not claim that:

- Computer vision produced a large improvement in predictive accuracy.
- The held-out experiment conclusively proved the superiority of CV-enhanced XGBoost.
- The box-derived features represent precise physical shooting measurements.
- The release detector was independently validated as fully accurate.

The appropriate conclusion is that video-derived box features produced a small, directionally consistent improvement, subject to important measurement and statistical uncertainty.

## 4. H1: Measurement and Implementation Limitations

The original proposal discussed physical shooting mechanics, including release angle, elbow angle, defender positioning, and contest-related measurements.

The frozen H1 predictive experiment instead primarily used image-based bounding-box measurements and proxies.

These included release height, jump-related measurements, ball rise, shooter speed, and defender-related spatial features.

An exploratory release-angle measurement was developed separately but was not included in the frozen H1 predictive model.

The implementation should therefore not be described as having comprehensively measured true shooting mechanics.

### Release measurement reliability

The measurement-check documentation reports weak agreement between box-derived release height and independent pose-based measurements within jump shots, with Spearman correlation approximately 0.32.

Later release-candidate audits also identified:

- Provisional human release intervals without overlapping raw detector candidates
- Candidate removals from the `steady()` filtering step
- Missing tracked ball boxes
- Hand-reach geometry failures
- Rising-ball threshold failures

These diagnostic findings help explain detector behavior but do not demonstrate that modifying the detector would improve downstream prediction.

Human reference intervals remain provisional rather than independently adjudicated ground truth.

## 5. H2: Multi-Agent Versus Single-Agent LLMs

### Research question

Does multi-agent LLM orchestration improve shot-make probability estimates compared with a budget-matched single-agent LLM?

### Experimental design

The frozen H2 experiment used Kaggle 2014–15 NBA shot logs.

The primary held-out evaluation included:

- 100 shots
- 50 games
- Identical shot inputs for compared conditions
- Three calls per shot for the multi-agent and budget-matched single-agent systems
- A frozen local LLM configuration

The original frozen test was distinct from later exploratory agent experiments.

### Frozen held-out results

| Model | Brier score | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost | 0.2056 | 0.5990 | 0.7414 | 74.0% |
| Logistic regression | 0.2198 | 0.6299 | 0.6810 | 67.0% |
| Constant baseline | 0.2448 | 0.6827 | 0.5000 | 58.0% |
| Multi-agent LLM | 0.2483 | 0.6821 | 0.6874 | 51.0% |
| Budget-matched single LLM | 0.2735 | 0.7451 | 0.6636 | 53.0% |
| One-call single LLM | 0.2931 | 0.7915 | 0.6537 | 44.0% |

### Primary statistical result

The paired Brier difference, calculated as multi-agent minus budget-matched single-agent, was approximately:

- Difference: -0.0252
- 95% game-bootstrap interval: [-0.0441, -0.0073]

This interval excludes zero and favors the multi-agent system under the frozen comparison.

### Supported interpretation

The multi-agent architecture outperformed the budget-matched single-agent LLM on the frozen H2 test.

However, the multi-agent system did not outperform XGBoost or logistic regression on Brier score.

The constant predictor also achieved a lower Brier score than the multi-agent LLM.

Therefore, the evidence supports an advantage relative to the matched single-agent LLM, not superiority over traditional machine learning.

## 6. H2: Exploratory Findings and Generalization

Subsequent H2 analyses examined calibration, stronger models, additional information, and tool-assisted prompting.

These studies qualified the initial frozen result.

The frozen multi-agent advantage was associated partly with reduced overconfidence. Exploratory calibration analyses showed that correcting probability offsets could substantially narrow the observed difference.

Later comparisons involving stronger models did not consistently preserve the original multi-agent advantage.

Tool-assisted systems could approach traditional-model performance when using predictions from the traditional model as an input, but this does not establish independent LLM predictive superiority.

These exploratory findings should be reported separately from the frozen primary hypothesis test.

### Claims requiring qualification

The final materials should not claim that:

- Multi-agent LLMs consistently outperform single-agent LLMs across model sizes and experimental settings.
- Multi-agent systems outperform XGBoost.
- Agent collaboration necessarily causes better independent predictive reasoning.
- Exploratory calibration or tool-assisted results are additional confirmatory hypothesis tests.

The appropriate conclusion is that the frozen experiment supported a multi-agent advantage over its matched LLM control, but the result was configuration-dependent and did not surpass traditional machine-learning baselines.

## 7. Comparability of H1 and H2

H1 and H2 used different datasets, shot populations, features, and experimental conditions.

H1 used BasketEvent 2025–26 shots, while H2 used Kaggle 2014–15 shot logs.

Consequently, the H1 and H2 Brier scores are not directly comparable as a single unified model leaderboard.

The repository supports two separate primary hypothesis tests.

It does not establish a single four-model comparison in which traditional ML, CV-enhanced ML, single-agent LLMs, and multi-agent LLMs were all evaluated on identical shots with matched input information.

The final presentation should distinguish these experimental comparisons rather than merging their scores into a single ranking.

## 8. Claim Consistency Checklist

| Potential final-presentation claim | Assessment |
|---|---|
| CV box features slightly improved H1 Brier score | Supported |
| H1 held-out improvement was statistically significant | Not established |
| CV substantially increased classification accuracy | Not supported |
| The H1 box measurements precisely represent physical shooting mechanics | Not supported |
| Multi-agent beat budget-matched single-agent on frozen H2 | Supported |
| Multi-agent beat XGBoost on frozen H2 | Not supported |
| H2 advantages generalized consistently to stronger models | Not established |
| H1 and H2 results can be directly ranked by absolute Brier score | Not supported |
| The full project was independently reproduced from raw data | Not verified |
| The final slides and final paper have been audited | Not verified |

## 9. Recommendations for Final Presentation and Report

The final materials should:

1. Introduce H1 and H2 as separate research questions with distinct datasets.
2. Report primary comparisons using Brier score and confidence intervals.
3. Explain why classification accuracy is potentially misleading for H1.
4. State that the frozen H1 held-out interval includes zero.
5. Clearly distinguish the multi-agent advantage over a matched LLM from performance against XGBoost.
6. Discuss later exploratory experiments without presenting them as new confirmatory tests.
7. Explain the difference between proposed physical shooting measurements and implemented image-based proxies.
8. Disclose release-timing uncertainty and measurement reliability limitations.
9. Avoid directly comparing absolute results across the different H1 and H2 cohorts.
10. Distinguish verified repository artifacts from experiments that have not been freshly reproduced.

The final slide deck and paper should be reviewed against this checklist once those actual materials are available.

## 10. Conclusion

The project's primary H1 and H2 experiments provide meaningful but qualified findings.

H1 found a small improvement in probabilistic shot prediction when using video-box-derived features, although the held-out confidence interval included zero.

H2 found that a multi-agent LLM outperformed its budget-matched single-agent control on the frozen test, but traditional XGBoost remained the stronger predictor.

Later exploratory studies and measurement audits further limit the generality and physical interpretation of these findings.

The final research narrative should communicate both the positive findings and their limitations.

This audit does not claim that the final slide deck or written report has been inspected, and it does not modify any frozen experimental result.