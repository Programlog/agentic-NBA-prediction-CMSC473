# Experiment 07 protocol: tuned prompts, similar shots, XGBoost as a tool

## Part 1: validation rules (fixed 2026-11-21, before any stage-1 validation call)

- **Model:** `gpt-6-luna`, effort `none`, temperature 0, seed 42. Strict JSON output.
  V1+ put the rationale before `p_make`.
- **Validation sample:** 500 shots in 125 games (4 per game), seed 42. It contains the
  100 validation shots used in 04–06.
- **Stage 1:** prompt variants V0–V4 (see `prompts.py`), one call per shot.
- **Choosing W:** the variant among V1–V4 with the lowest validation Brier. The variant
  among V1–V2 (no XGBoost input) with the lowest Brier is also carried forward, as the
  best "no model tool" agent.
- **Stage 2:** `multi:W` and `single_budget:W` on the same 500 shots.
- **Test (Part 2, to be frozen after stage 2):** 1,000 test shots in 125 games (8 per
  game), with 900 that no earlier experiment has used. The test is run once.

## Part 2: test protocol (frozen 2026-11-22, after validation and before any test call)

### What validation showed (500 shots, 45% made)

| Condition | Brier | ROC-AUC | Share within 0.005 of XGBoost |
|---|---:|---:|---:|
| XGBoost (reference) | 0.2284 | 0.633 | 1.00 |
| `single_budget:V4` | 0.2289 | 0.629 | 0.97 |
| V4: rate table + similar shots + XGBoost | **0.2292** | 0.629 | 0.93 |
| `multi:V4` | 0.2306 | 0.626 | 0.77 |
| V3: rate table + XGBoost | 0.2335 | 0.625 | 0.61 |
| V0: original prompt + anchors | 0.2343 | 0.604 | 0.10 |
| kNN-50 (reference) | 0.2356 | 0.610 | 0.04 |
| V2: rate table + similar shots | 0.2358 | 0.605 | 0.09 |
| V1: rewritten prompt + rate table | 0.2363 | 0.598 | 0.09 |
| Logistic regression (reference) | 0.2386 | 0.602 | 0.07 |

By the Part 1 rule, **W = V4**, and the best variant without the XGBoost input is **V2**.
Validation cost $1.36. The first stage-2 pass lost 280 shots to connection errors (a network
drop). They were retried on resume, and every condition has 500/500 `ok` predictions.

### Test setup

| Setting | Frozen value |
|---|---|
| Shots | `test` split, 1,000 shots, 125 games × 8, seed 42; `test_rows.parquet` SHA-256 `a4d0673b…2c4a55` |
| Conditions | V0, V2, V4, `multi:V4`, `single_budget:V4` |
| Model | `gpt-6-luna`, effort `none`, temperature 0, seed 42 |
| Code | `prepare.py` `630a4cd8`, `prompts.py` `6c9f5af3`, `run_experiments.py` `c99a35ae`, `analyze.py` `287975c4`, `src/nba_shots/agents.py` `544486e3` (unchanged) |

### Analysis (fixed now)

- **Primary:** `multi:V4` − XGBoost, Δ Brier on the 1,000 test shots. The interval is a
  95% percentile interval from 1,000 paired game-bootstrap resamples, seed 42.
- **Secondary:**
  - `single_budget:V4` − XGBoost
  - `multi:V4` − `single_budget:V4`
  - V4 − V0
  - V4 − XGBoost
  - V2 − XGBoost
  - V2 − V0
  - V0 − XGBoost
  - kNN-50 − XGBoost
- **Fresh shots:** every comparison is also reported on the 900 shots that no earlier
  experiment used.
- **Descriptive only:** ROC-AUC, log loss, mean p, and the correlation with XGBoost and
  share of copies. There is no multiplicity correction, and no recalibration, because
  V4 is already calibrated.
- **Acceptance:** 1,000/1,000 `ok` predictions per condition after resuming.
