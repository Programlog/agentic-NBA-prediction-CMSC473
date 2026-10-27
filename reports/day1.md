# Initial tabular baseline results

Saved validation results from October 22, 2026, using the Kaggle 2014–15 shot logs.
The chronological split keeps whole games and dates together: 91,072 training shots
(633 games), 19,069 validation shots (135 games), and 17,928 reserved test shots
(136 games).

| Model | Validation accuracy | Brier score | Log loss | ROC-AUC |
|---|---:|---:|---:|---:|
| Constant | 55.11% | 0.247426 | 0.687991 | 0.500000 |
| Logistic regression | 60.84% | 0.234374 | 0.660993 | 0.627413 |
| XGBoost | 62.08% | 0.228181 | 0.646257 | 0.642430 |

An independent rerun matched prepared data, validation predictions, and metrics.
These are development results; the test set was not evaluated. They do not establish
CV or multi-agent improvements. Real-video and real-Qwen experiments remain pending.

The source run is `artifacts/day1/`; the independent run is
`artifacts/day1_reproduced/`. Generated artifacts are local and ignored by Git.
See the [execution guide](../docs/experiments.md) to reproduce the baseline and
create a new report from saved predictions.
