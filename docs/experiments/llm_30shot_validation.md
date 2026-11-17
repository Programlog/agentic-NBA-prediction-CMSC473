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

- Structural artifact validation: Passed
- Prediction attempts: 150
- Successful predictions: 148
- Failed predictions: 2
- Overall prediction coverage: 98.7%
- Recorded model calls: 329
- Prompt tokens: 331,699
- Completion tokens: 56,739
- Strict acceptance: Failed due to two unsuccessful predictions

Both failures were Ollama HTTP read timeouts (240 seconds), affecting the
budget-matched single-agent and single-agent ensemble conditions.

This is an exploratory validation experiment, not a strictly accepted run.

## Model performance

Lower Brier score and log loss are better.

| Condition | Successful | Brier | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|
| Single-agent | 30/30 | 0.22780 | 0.64729 | 0.64480 | 0.60000 |
| Budget-matched single-agent | 29/30 | 0.21823 | 0.62619 | 0.66912 | 0.55172 |
| Decomposed single-agent | 30/30 | 0.23520 | 0.66097 | 0.64932 | 0.56667 |
| Single-agent ensemble | 29/30 | 0.20855 | 0.60194 | 0.70833 | 0.68966 |
| Multi-agent | 30/30 | 0.23205 | 0.65580 | 0.65837 | 0.60000 |

Scores use successful predictions only. Because coverage differs, these
descriptive scores are not all calculated on identical shots.

## Paired comparisons

The following comparisons use shots successfully predicted by both models.
The reported difference is multi-agent Brier minus control Brier.
Negative differences favor the multi-agent system.

| Control | Common shots | Brier difference | 95% game-bootstrap interval |
|---|---:|---:|---|
| Single-agent | 30 | +0.00425 | [-0.02915, +0.03235] |
| Budget-matched single-agent | 29 | +0.01760 | [-0.00388, +0.03832] |
| Decomposed single-agent | 30 | -0.00315 | [-0.02467, +0.01722] |
| Single-agent ensemble | 29 | +0.01052 | [-0.02427, +0.04230] |

All four intervals include zero. This experiment does not establish a
reliable predictive advantage or disadvantage for multi-agent coordination.

## Recorded inference usage

| Condition | Calls | Tokens | Runtime (seconds) |
|---|---:|---:|---:|
| Multi-agent | 90 | 107,748 | 5,151.4 |
| Single-agent | 30 | 33,417 | 801.4 |
| Budget-matched single-agent | 89 | 113,035 | 3,366.8 |
| Decomposed single-agent | 30 | 34,493 | 779.1 |
| Single-agent ensemble | 90 | 99,745 | 10,243.1 |

These figures describe recorded local inference usage. Runtime is not a
controlled measurement of universal architecture latency. Local model
execution also incurs compute and electricity costs even without API fees.

## Interpretation

In this validation sample, the multi-agent condition did not demonstrate a
reliable improvement over any of the four controls.

The single-agent ensemble had the lowest descriptive Brier score, but it
also had one timeout, and its comparison with multi-agent was not statistically
conclusive.

The results suggest that using multiple collaborating LLM agents does not
automatically improve basketball shot prediction. They do not prove that
single-agent ensembles are generally superior.

## Limitations

- Only 30 shots across 10 games were evaluated.
- Two request timeouts reduced prediction coverage.
- Common-success comparisons can be affected by nonrandom failures.
- All results are from the validation partition, not the final test set.
- Game-bootstrap intervals are uncertain with only 10 sampled games.
- This experiment evaluates LLM architectures, not the value of CV features.

## Reproduction

Raw experimental artifacts:

artifacts/llm_balanced_nov16_30shot/

Generated reports:

artifacts/llm_balanced_nov16_30shot_exploratory/
artifacts/llm_balanced_nov16_30shot_paired_report/

These artifact paths are local and may be excluded from version control.
The report documents the observed run rather than claiming that the raw
inference outputs are included in the repository.
