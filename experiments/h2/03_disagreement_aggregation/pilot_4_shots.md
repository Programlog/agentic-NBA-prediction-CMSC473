# Disagreement-Aware Multi-Agent Aggregation — 4-Shot Validation Pilot

**Status:** completed exploratory smoke test; **not** the frozen H2 held-out evaluation.

## Motivation

The project proposal describes specialized agents that analyze the available
shot information from different perspectives before passing their outputs
to a prediction agent that estimates shot-success probability. This follow-up
experiment investigates whether explicitly summarizing disagreements among
specialists helps the prediction agent combine their outputs.

The additional condition, `llm_multi_disagreement`, supplies specialist
probability summary statistics (mean, range, and standard deviation) to the
final aggregator. The comparison condition, `llm_multi`, uses the original
multi-agent aggregation approach. The new condition was implemented on
`feature/adaptive-aggregation` (commit `83dbab7`).

## Pilot setup

- **Dataset:** Kaggle 2014–15 NBA shot logs, prepared shot table
  `artifacts/baseline/prepared/shots.parquet`.
- **Data split:** validation only; no held-out test evaluation.
- **Selection:** 4 shots across 2 games, seed 42.
- **LLM:** local Ollama, `qwen3:4b-instruct`.
- **Conditions:** `llm_single`, `llm_single_budget`, `llm_multi`,
  and `llm_multi_disagreement`.
- **Output folder:** `artifacts/adaptive_aggregation_pilot_4` (local,
  ignored by Git unless exported separately).
- **Completion:** 2026-11-18 23:28:59 UTC.
- **Execution:** 16/16 attempted predictions scored; 0 failures;
  40 actual LLM calls across all conditions.

Command used (PowerShell):

```powershell
uv run --frozen --all-extras nba-shots predict-llm `
  --shots artifacts/baseline/prepared/shots.parquet `
  --output artifacts/adaptive_aggregation_pilot_4 `
  --model qwen3:4b-instruct `
  --split validation `
  --limit 4 `
  --games 2 `
  --seed 42 `
  --conditions llm_single llm_single_budget llm_multi llm_multi_disagreement
```

## Observed validation metrics

| Condition | Scored | Failed | Brier score ↓ | Log loss ↓ | ROC-AUC ↑ | Accuracy ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_multi` | 4 | 0 | 0.2767 | 0.7469 | 0.0000 | 0.50 |
| `llm_multi_disagreement` | 4 | 0 | **0.2657** | **0.7246** | 0.0000 | 0.25 |
| `llm_single` | 4 | 0 | 0.2289 | 0.6500 | 0.5000 | 0.50 |
| `llm_single_budget` | 4 | 0 | 0.2641 | 0.7219 | 0.8333 | 0.50 |

For the specific aggregation comparison:

- **Brier difference:** `0.2657 - 0.2767 = -0.0110` (lower is better),
  a **3.98% relative reduction** on these four shots.
- **Log-loss difference:** approximately `-0.0223`, also favoring
  disagreement-aware aggregation.
- **Accuracy:** decreased from 0.50 to 0.25; neither aggregation
  condition demonstrated useful ranking on this very small pilot
  (ROC-AUC 0.0 for both).
- **Reliability:** all conditions completed without prediction failures.

## Interpretation and limitations

The disagreement-aware aggregator **improved Brier score and log loss on
these four validation shots**, but the sample is far too small to conclude
it improves performance generally. The new condition had lower accuracy,
and the one-call single-agent baseline had a lower Brier score than either
multi-agent condition on this selection.

This is an **exploratory follow-up**, not an update to the frozen H2 test.
The reported results are aggregate metrics from the completed local run;
the per-shot prediction file and usage report remain in the local
`artifacts/` directory and are not bundled in this document. No confidence
interval, significance test, or claim of generalization is made from
four shots.

## Next step

Compare `llm_multi_disagreement` against `llm_multi` on a larger,
predefined validation selection, inspect paired shot-level differences
and inference cost, and document results separately. Any later held-out
test evaluation should use a new, explicitly frozen protocol rather than
reusing the completed H2 test as development data.

For the original H2 research question and final held-out result, see
[Final H2 results](../02_final_test_100/README.md) and
[Frozen H2 protocol](../02_final_test_100/protocol.md).
