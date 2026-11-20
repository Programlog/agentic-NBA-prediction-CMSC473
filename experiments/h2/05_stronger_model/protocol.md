# Experiment 05 test protocol: stronger model (GPT-6 Luna)

**Status:** frozen on 2026-11-20, after the validation runs and **before any
test-split call** in this experiment. This is a new, exploratory experiment. The
pre-registered H2 answer is still [02](../02_final_test_100/README.md), and
nothing here replaces it.

## Question

Does a stronger hosted model (OpenAI `gpt-6-luna`) predict shot success better
than local `qwen3:4b-instruct`? And once both are anchored to training make rates,
does multi-agent orchestration still beat a budget-matched single analyst?

## What validation showed (used only to choose this design)

On 100 validation shots (44% made), Brier:

| Condition | qwen 4B | Luna `none` | Luna `low` |
|---|---:|---:|---:|
| multi | 0.2734 | 0.2480 | 0.2500 |
| single_budget | 0.2794 | 0.2548 | 0.2555 |
| multi_anchored | 0.2382 | 0.2392 | 0.2379 |
| single_budget_anchored | (running) | 0.2417 | 0.2377 |

Reference on the same shots: XGBoost 0.2318, constant 0.2466.

## Test selection (identical to 02)

`test` split, 100 shots, 50 games, `balanced_games_v1`, seed 42. Data SHA-256 is
`4528a5be…69f1`. The runner was checked to select exactly the shot IDs in
`02_final_test_100/report/predictions.csv`.

## Conditions and model

| Setting | Frozen value |
|---|---|
| Model | `gpt-6-luna` via OpenAI Chat Completions, Structured Outputs (strict JSON schema) |
| Reasoning effort | `none` (temperature 0, seed 42) **and** `low` (provider-default sampling) |
| Conditions | `multi`, `single_budget`, `multi_anchored`, `single_budget_anchored` |
| Calls per shot | 3 for every condition (2 specialists + aggregator, or 3 sequential analyst calls) |
| Anchors | training-split make rates (overall, distance band, defender band, cross when n ≥ 50), as in 04 |
| Code | `openai_client.py` `8cc7f2d6…`, `run_experiments.py` `b6a86184…`, `04/run_experiments.py` `9a069308…`, `src/nba_shots/agents.py` `544486e3…` (unchanged by this experiment) |
| Spend cap | $5 per run (validation cost about $0.18 per effort) |

Secondary: local qwen `multi_anchored` and `single_budget_anchored` on the same
test shots. qwen `multi` and `single_budget` test predictions come from 02.

## Analysis (fixed now)

- **Metric:** Brier score. Log loss, ROC-AUC and mean predicted p are descriptive.
- **Primary comparison:** Luna `none`, `multi_anchored` − `single_budget_anchored`.
- **Key secondary comparisons:**
  1. Luna `none` `multi` − qwen `multi` (does the bigger model help, unanchored?)
  2. Luna `none` `multi` − `single_budget` (the 02 comparison, repeated with Luna)
  3. the best Luna condition − XGBoost
  4. the same comparisons at effort `low`
- **Uncertainty:** 95% percentile interval from 1,000 paired bootstrap resamples
  of games, seed 42 (`paired_bootstrap` in `04_brier_followup/analyze.py`).
  An interval entirely below 0 favours the first term, and one containing 0 means
  no detectable difference.
- **Recalibration:** for each model × effort × condition, a Platt map is fitted
  on **all** of that condition's validation predictions and applied to its test
  predictions. Test labels are never used for fitting. Raw and recalibrated scores
  are both reported. The decision above uses raw scores; recalibrated ones are
  descriptive.
- No multiplicity correction. Everything except the primary comparison is descriptive.

## Execution

```bash
for e in none low; do
  uv run --frozen --all-extras python experiments/h2/05_stronger_model/run_experiments.py \
      --backend openai --reasoning-effort $e --split test --rpm 40 --max-cost 5
done
uv run --frozen --all-extras python experiments/h2/05_stronger_model/run_experiments.py \
    --backend ollama --split test --conditions multi_anchored single_budget_anchored
```

Acceptance: 100/100 `ok` predictions for every condition, with no failed shots
left after resuming.

## Known deviation before the test run

The per-call token and cost fields in the validation caches were mislogged under
parallel workers. Each worker could read another worker's unfinished log, so
`experiment.json` reports about $0.001 when the real figure is about $0.18. This
was fixed with per-thread logs before the test run. Predictions were not affected,
because each call's result is returned directly. Validation costs are taken from
the runner's live totals in the run logs.
