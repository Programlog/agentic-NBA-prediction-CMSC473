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
