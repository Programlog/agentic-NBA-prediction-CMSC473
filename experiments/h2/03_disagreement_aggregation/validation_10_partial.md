# Disagreement-Aware Aggregation — Partial 10-Shot Validation Run

**Status:** preliminary / interrupted run, **28 of 40 planned predictions**
completed. This is **not** the final 10-shot result and does not modify the
previously frozen H2 held-out findings.

## Goal

The original project proposal describes specialized agents that separately
analyze the available shot features from different perspectives and feed
their outputs to a prediction agent that estimates shot-success
probability. This experiment tests whether supplying explicit summaries
of disagreement among the specialist agents helps the prediction agent
combine their outputs.

The new `llm_multi_disagreement` condition provides specialist
probability mean, range, and standard deviation to the aggregator, while
`llm_multi` retains the original aggregation strategy.

## Configuration

- **Dataset:** prepared Kaggle 2014–15 NBA shot logs
  (`artifacts/baseline/prepared/shots.parquet`)
- **Split:** validation, not test
- **Planned sample:** 10 shots across 5 games
- **Completed sample at reporting:** **7 matched shots across 4 games**
- **Seed:** 42
- **Model:** `qwen3:4b-instruct`, running locally through Ollama
- **Conditions:** `llm_single`, `llm_single_budget`, `llm_multi`,
  `llm_multi_disagreement`
- **Experiment output:** `artifacts/adaptive_aggregation_validation_10`
  (local; not included in this commit)
- **Implementation:** `feature/adaptive-aggregation`, commit `83dbab7`

The run was stopped before completing its planned 10-shot evaluation.
The 28 saved predictions comprise **7 successful predictions per
condition, with 0 recorded failures**. The four conditions cover
the same seven `shot_id` values; these seven shots span four games.

### Reproduction command (PowerShell)

```powershell
uv run --frozen --all-extras nba-shots predict-llm `
  --shots artifacts/baseline/prepared/shots.parquet `
  --output artifacts/adaptive_aggregation_validation_10 `
  --model qwen3:4b-instruct `
  --split validation `
  --limit 10 `
  --games 5 `
  --seed 42 `
  --conditions llm_single llm_single_budget llm_multi llm_multi_disagreement
```

## Preliminary results on the seven completed, paired shots

**Brier score:** lower is better. Values below are calculated from the
locally saved `predictions.csv` with `status == "ok"`, using
`mean((p_make - y_true)^2)`.

| Model | Matched shots | Brier score ↓ |
| --- | ---: | ---: |
| `llm_multi_disagreement` | 7 | **0.196086** |
| `llm_single_budget` | 7 | 0.201900 |
| `llm_multi` | 7 | 0.209529 |
| `llm_single` | 7 | 0.238986 |

**Primary exploratory comparison (new vs. original aggregator):**

- Original aggregation: **0.209529** Brier.
- Disagreement-aware aggregation: **0.196086** Brier.
- Paired difference (disagreement-aware minus original): **−0.013443**,
  or a **6.42% relative reduction** in Brier score on these seven shots.
- The lower Brier score in this partial sample is encouraging, but no
  significance test or confidence interval has been calculated.

The matched comparison was computed by pivoting successful predictions
on `shot_id`, retaining shots with both multi-agent predictions, then
using the corresponding `y_true` label for each shot. The sample size,
game count, and model values above were copied from local command output;
the prediction CSV itself is not committed here.

## Interpretation and limitations

This partial validation run shows lower Brier score for
`llm_multi_disagreement` than `llm_multi` on the **seven completed
shots**, and all saved model predictions succeeded. It does **not**
establish that the disagreement-aware aggregator is better in general.

- **Incomplete run:** 12 of the 40 planned condition predictions had
  not been generated when these metrics were calculated; the remaining
  three shots are not included. They are **not prediction failures**.
- **Very small sample:** seven shots from four games cannot support
  reliable significance or broad generalization.
- **Potential reporting bias:** this is an interim, first-completed subset
  from an interrupted run, not a separately pre-registered final sample.
- **Validation only:** the frozen H2 held-out test must not be reused
  as a tuning set for this new strategy.
- **Metric scope:** the output shared for this partial run established
  Brier scores, completion counts, and game count; it did not establish
  partial-run log loss, accuracy, ROC-AUC, or a per-condition cost table.

For context, an earlier [4-shot validation pilot](pilot_4_shots.md)
also found a lower Brier score for disagreement-aware aggregation
(0.2657 versus 0.2767). The samples should **not** be assumed independent
or pooled without checking their saved shot IDs.

## Next steps

Finish the interrupted 10-shot run if practical, recompute the
paired-shot comparison on the complete selection, and preserve the full
prediction and usage artifacts for reproducibility. Any larger,
confirmatory evaluation requires a separately frozen protocol.

See [final frozen H2 results](../02_final_test_100/README.md) for the completed
multi-agent-versus-single-agent experiment, which is distinct from
this exploratory follow-up.
