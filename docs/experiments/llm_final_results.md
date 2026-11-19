# Final Held-Out Results — Multi-Agent vs. Single-Agent LLM (RQ2)

**Protocol:** [llm_final_protocol.md](llm_final_protocol.md) (frozen before the run)
**Report:** [`reports/llm_final_test_100/report.md`](../../reports/llm_final_test_100/report.md)
**Run:** 2026-11-19, 03:48–05:50 UTC, commit `aed2c6d`, one attempt, 300/300 predictions, 0 failures

## Answer

**H2 is supported on this test set.** The 95% interval for the primary comparison
lies entirely below zero. Under the frozen decision rule, multi-agent orchestration
**improves** calibration over a single analyst given the same number of calls.

| Comparison | Role | Δ Brier (multi − control) | 95% game-bootstrap interval | Verdict |
|---|---|---:|---|---|
| `llm_multi` − `llm_single_budget` | **Primary (H2)** | **−0.0252** | **[−0.0441, −0.0073]** | Improves |
| `llm_multi` − `llm_single` | Secondary | −0.0448 | [−0.0699, −0.0212] | Improves (descriptive) |

100 shots, 50 games (2 per game), all shots paired; 1,000 game resamples, seed 42.
Negative Δ favors the multi-agent system.

## All models on the same 100 test shots

Make rate on these shots: 42% (42 made, 58 missed).

| Model | Brier | Log loss | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|
| XGBoost (frozen, reference) | **0.2056** | **0.5990** | **0.7414** | **0.740** |
| Logistic regression (reference) | 0.2198 | 0.6299 | 0.6810 | 0.670 |
| Constant (reference) | 0.2448 | 0.6827 | 0.5000 | 0.580 |
| `llm_multi` | 0.2483 | 0.6821 | 0.6874 | 0.510 |
| `llm_single_budget` | 0.2735 | 0.7451 | 0.6636 | 0.530 |
| `llm_single` | 0.2931 | 0.7915 | 0.6537 | 0.440 |

| Condition | Calls | Tokens | Runtime (s) |
|---|---:|---:|---:|
| `llm_multi` | 300 | 360,027 | 2,478.5 |
| `llm_single_budget` | 300 | 388,009 | 2,449.8 |
| `llm_single` | 100 | 112,289 | 846.2 |

The two primary conditions used the same number of calls and similar token counts,
so the difference is not explained by extra compute.

## Interpretation

1. **Orchestration helped relative to the single-analyst controls.** The primary
   effect (−0.025) is about twice the smallest effect the protocol expected this
   sample to detect (~0.013).
2. **None of the LLM conditions beat the simple baselines.** `llm_multi` scored
   slightly worse than predicting the training make rate for every shot (0.2483 vs.
   0.2448), and every LLM condition was well behind XGBoost (0.2056) and logistic
   regression (0.2198). The multi-agent system is the best LLM setup, not a good
   shot predictor.
3. **Most of the gain is less overconfidence.** *Exploratory, not pre-registered.*
   All three conditions predicted makes far more often than they happened (mean
   predicted probability 0.60 multi, 0.645 budget, 0.67 single, vs. a 42% make rate).
   The multi-agent aggregator was the least overconfident. Shifting each condition's
   predictions to the true make rate (using test outcomes, so for diagnosis only)
   narrows the primary gap from 0.025 to about 0.008 Brier. The multi-agent system also
   ranked shots somewhat better (ROC-AUC 0.687 vs. 0.664; correlation with outcome
   0.35 vs. 0.29), so the gain is not purely an offset.
4. **The pilot pointed the other way.** On 30 validation shots, multi-agent was
   *worse* than the budget-matched control (+0.014, interval [−0.006, +0.034]). The
   test result is the pre-registered answer; the reversal between a small pilot and
   the test set is a reason to treat the size of the effect cautiously.

## Deviations and checks

- **Pilot model identity not verified.** The protocol asks that the run's model digest
  match the pilot's. The pilot's `experiment.json` was not committed or available, so
  only the run digest (`0edcdef3…`) is recorded. If it differs from the pilot's, the
  pilot and test used different builds of `qwen3:4b-instruct`.
- **Runtime:** about 2 hours on an Apple M2, not the 8–9 hours estimated from the
  pilot machine. No settings were changed.
- Everything else followed the protocol: frozen `agents.py` hash matched, preflight and
  validation smoke test passed, one attempt accepted, strict artifact validation passed,
  XGBoost scored from the frozen day-1 models without refitting.

## Limitations (fixed in advance, plus one observed)

- One small local model (4B parameters, 4-bit). Results may not hold for larger models.
- One season of public tabular data (Kaggle 2014–15); no computer-vision features.
- 100 shots / 50 games; the interval's upper end (−0.007) is close to zero.
- Decomposed and ensemble controls were run only on validation.
- *Observed:* all LLM conditions were miscalibrated toward predicting makes, which
  drives much of the measured difference (see Interpretation 3).
