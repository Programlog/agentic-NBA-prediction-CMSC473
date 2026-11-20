# LLM Architecture Comparison — 30-Shot Validation Experiment

## Research question

Does a multi-agent LLM system improve NBA shot probability prediction compared
with single-agent approaches when evaluated on the same basketball shots?

## Experimental setup

- Dataset: Kaggle 2014–15 NBA shot logs
- Evaluation split: Validation
- Model: qwen3:4b-instruct (local Ollama)
- Prompt version: release-v4-feature-semantics
- Random seed: 42
- Sample: 30 shots from 10 games
- Outcomes: 17 made shots and 13 missed shots
- Conditions: single-agent, budget-matched single-agent, decomposed single-agent,
  single-agent ensemble, and multi-agent
- Sampling: Balanced across games, not selected based on shot outcome

## Validation and execution

### Original execution

- Structural artifact validation: Passed
- Prediction attempts: 150
- Successful predictions: 148
- Failed predictions: 2
- Recorded calls: 329
- Prompt tokens: 331,699
- Completion tokens: 56,739
- Strict acceptance: Failed

Two Ollama HTTP requests timed out after the configured 240-second read
timeout. The failures affected `llm_single_budget` and `llm_single_ensemble`,
one shot each.

### Cache-assisted recovery

The original execution records were backed up before recovery. The runner
also archived the previous execution in `execution_history/`.

The unchanged experiment configuration was rerun using the existing
successful-prediction caches.

- Reused successful predictions: 148
- Fresh predictions recovered: 2
- New inference calls: 6
- New prompt tokens: 6,276
- New completion tokens: 1,067
- Recovered predictions: 150/150 successful
- Structural validation: Passed
- Strict acceptance: Passed
- Failed predictions in recovered execution: 0

The accepted execution combines 148 cached predictions with two newly
generated predictions. It is not an independent, entirely fresh replication.
The original two timeout failures remain part of the execution history.

Across the two executions, 335 model calls were recorded. This counts
recorded calls across attempts, not necessarily all server-side computation.

## Model performance

All five conditions now have complete prediction coverage on the same
30 validation shots. Lower Brier score and log loss are better.

| Condition | Successful | Brier | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Single-agent | 30/30 | 0.22780 | 0.64729 | 0.64480 | 0.60000 |
| Budget-matched single-agent | 30/30 | 0.21801 | 0.62585 | 0.67647 | 0.56667 |
| Decomposed single-agent | 30/30 | 0.23520 | 0.66097 | 0.64932 | 0.56667 |
| Single-agent ensemble | 30/30 | 0.22402 | 0.63903 | 0.65837 | 0.66667 |
| Multi-agent | 30/30 | 0.23205 | 0.65580 | 0.65837 | 0.60000 |

The budget-matched single-agent condition achieved the lowest descriptive
Brier score in the accepted evaluation.

## Paired comparisons

All comparisons use the same 30 successful shots across 10 games.

The reported difference is multi-agent Brier minus control Brier.
Negative differences favor the multi-agent model.

| Control | Common shots | Brier difference | 95% game-bootstrap interval |
|---|---:|---:|---|
| Single-agent | 30 | +0.00425 | [-0.02915, +0.03235] |
| Budget-matched single-agent | 30 | +0.01404 | [-0.00591, +0.03358] |
| Decomposed single-agent | 30 | -0.00315 | [-0.02467, +0.01722] |
| Single-agent ensemble | 30 | +0.00803 | [-0.02466, +0.03692] |

All four intervals include zero. These results do not establish a
reliable predictive advantage or disadvantage for multi-agent coordination.

## Recorded inference usage

The initial execution recorded the following usage:

| Condition | Calls | Tokens | Runtime (seconds) |
|---|---:|---:|---:|
| Multi-agent | 90 | 107,748 | 5,151.4 |
| Single-agent | 30 | 33,417 | 801.4 |
| Budget-matched single-agent | 89 | 113,035 | 3,366.8 |
| Decomposed single-agent | 30 | 34,493 | 779.1 |
| Single-agent ensemble | 90 | 99,745 | 10,243.1 |

Recovery required three additional calls for the budget-matched condition
and three for the ensemble condition.

The accepted execution's near-zero runtimes for cached conditions represent
cache retrieval, not original inference time. Runtime measurements are not
controlled estimates of universal architecture latency.

## Interpretation

After resolving the two timeouts, all five LLM conditions were evaluated
successfully on identical validation shots.

The budget-matched single-agent condition achieved the lowest descriptive
Brier score (0.21801), while multi-agent achieved 0.23205.

However, all four paired game-bootstrap intervals included zero. The
experiment therefore provides insufficient evidence that coordinating
multiple LLM agents improves or worsens prediction accuracy relative to
these single-agent controls.

The recovery improved prediction coverage; it did not increase the number
of evaluated shots or games.

## Limitations

- The sample contains only 30 shots from 10 games.
- These are validation results, not final held-out test findings.
- The original execution experienced two Ollama timeouts.
- The accepted execution reuses cached results from the original run.
- Bootstrap uncertainty is substantial with only 10 sampled games.
- No independent, entirely fresh replication has been performed.
- This experiment does not test whether computer vision improves prediction.

## Reproduction and provenance

Original and recovered execution artifacts:

artifacts/llm_balanced_nov16_30shot/

Independent original-execution backup:

artifacts/llm_balanced_nov16_30shot_original_backup/

Original exploratory reports:

artifacts/llm_balanced_nov16_30shot_exploratory/
artifacts/llm_balanced_nov16_30shot_paired_report/

Accepted recovery reports:

artifacts/llm_balanced_nov16_30shot_accepted_summary/
artifacts/llm_balanced_nov16_30shot_accepted_paired/

These artifact directories are local and may be excluded from version
control. The results document does not imply that the underlying inference
traces or caches are committed to Git.
