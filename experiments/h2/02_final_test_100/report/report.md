# NBA shot probability experiments

Generated: 2026-11-19T05:50:45.058641+00:00

Results below are separated by dataset and partition. Lower Brier score and log loss are better; inspect reliability diagrams separately for calibration.

## kaggle_2014_15 / test

### Shot outcome composition

- Unique shots: 100
- Made shots: 42
- Missed shots: 58
- Make rate: 42.0%
- Both outcome classes present: yes

| Model | Scored / total | Brier | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| constant | 100 / 100 | 0.24480 | 0.68272 | 0.50000 | 0.58000 |
| llm_multi | 100 / 100 | 0.24830 | 0.68214 | 0.68740 | 0.51000 |
| llm_single | 100 / 100 | 0.29307 | 0.79145 | 0.65374 | 0.44000 |
| llm_single_budget | 100 / 100 | 0.27354 | 0.74507 | 0.66359 | 0.53000 |
| logistic_regression | 100 / 100 | 0.21984 | 0.62987 | 0.68103 | 0.67000 |
| xgboost | 100 / 100 | 0.20555 | 0.59898 | 0.74138 | 0.74000 |

- Prediction coverage: llm_single 100/100 (100.0%); llm_multi 100/100 (100.0%).
- Common-success cohort: 100 / 100 attempted shots.
- llm_multi − llm_single: Brier difference -0.04477, 95% game-bootstrap interval [-0.06993, -0.02125].
- Prediction coverage: llm_single_budget 100/100 (100.0%); llm_multi 100/100 (100.0%).
- Common-success cohort: 100 / 100 attempted shots.
- llm_multi − llm_single_budget: Brier difference -0.02524, 95% game-bootstrap interval [-0.04409, -0.00728].

Metrics with incomplete coverage describe successful predictions only. Paired comparisons require matching attempted shot IDs and use only shots where both models succeeded. Coverage is reported separately; common-success results may be biased when failures are not random.

## Interpretation limits

- Validation results are development evidence, not final test results.
- Kaggle and recent-video scores cannot establish a CV improvement across datasets.
- A bootstrap interval spanning zero does not establish an improvement.
- Review clip availability, pose failures, game count and player coverage before generalizing a video-cohort result.
