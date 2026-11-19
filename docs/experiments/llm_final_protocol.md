# Final Held-Out Protocol — Multi-Agent vs. Single-Agent LLM (RQ2)

**Status:** frozen on 2026-11-19, before any test-split LLM inference.
**Runner:** [`scripts/run_llm_final.sh`](../../scripts/run_llm_final.sh)
**Development evidence:** [30-shot validation pilot](llm_30shot_validation.md)

This document fixes every analysis decision before the test split is used. The
test selection below is evaluated once. If any frozen setting must change, the
change and its reason are recorded here and the run is reported as a new
experiment, not as a replacement.

## Research question

When given the same information and features, does a multi-agent LLM pipeline
predict NBA shot success better than a single-LLM baseline?

## Hypotheses

- **Primary (H2).** Multi-agent orchestration (`llm_multi`) changes the Brier
  score relative to a single analyst given the *same number of calls*
  (`llm_single_budget`). The test is two-sided.
- **Secondary.** `llm_multi` versus the one-call baseline (`llm_single`).
- **Reference only.** Frozen XGBoost, Logistic Regression and the constant
  model scored on the same shots. These use the same tabular features but are a
  different model family; they put the LLM numbers in context and do not decide H2.

## Outcome and decision rule

- **Metric:** Brier score (lower is better). Log loss, ROC-AUC and accuracy are
  reported descriptively.
- **Effect:** Δ = Brier(`llm_multi`) − Brier(control), on the common shots.
  Negative favours multi-agent.
- **Uncertainty:** 95% percentile interval from 1,000 paired bootstrap resamples
  of *games* (seed 42), as implemented in `paired_common_success_comparison`.
- **Verdict for H2:**
  - interval entirely below 0 → multi-agent orchestration **improves** calibration;
  - interval entirely above 0 → multi-agent orchestration **hurts** calibration;
  - interval contains 0 → **no detectable difference** at this sample size.
- Only the primary comparison determines the answer to RQ2. Secondary and
  reference comparisons are descriptive and are not corrected for multiplicity.

## Evaluation selection

| Setting | Frozen value |
|---|---|
| Dataset | Kaggle 2014–15 NBA shot logs, prepared by the reproduced day-1 baseline (`artifacts/day1/prepared/shots.parquet`) |
| Split | `test` (17,928 shots / 136 games), run with `--final` |
| Shots | 100 |
| Games | 50 (2 shots per game) |
| Sampling | `balanced_games_v1`: hash-ranked by seed, game ID and shot ID; never by outcome |
| Seed | 42 |

The selected shot IDs are written to `selection.csv` and to `spec.shot_ids` in
`experiment.json`, which also records the SHA-256 of the prepared data.

**Expected precision.** In the pilot, the primary interval had a half-width of
about 0.020 with 10 games. Scaling by √(10/50) gives about **±0.009** with 50
games, so true differences smaller than roughly **0.013 Brier** are unlikely to
be detected. A "no detectable difference" verdict must be reported with this
bound.

## Conditions and model

| Setting | Frozen value |
|---|---|
| Conditions | `llm_single` (1 call), `llm_single_budget` (≤4), `llm_multi` (≤4) |
| Not run on test | `llm_single_decomposed`, `llm_single_ensemble` (validation-only controls) |
| Model | `qwen3:4b-instruct` via local Ollama; the digest is recorded in `experiment.json` |
| Prompt version | `release-v4-feature-semantics` (unchanged from the pilot) |
| Inference | temperature 0, seed 42, `num_predict` 512, `num_ctx` 8192, read timeout 240 s |
| Features | default tabular configuration; player history disabled |
| `src/nba_shots/agents.py` SHA-256 | `aec1e4fe23e723cf89e7de12ffbec30c448a22d99624e93ba5e18f67a9fadac4` |

With player history disabled, two specialists have inputs, so `llm_multi` makes
three calls per shot (two specialists and an aggregator), and `llm_single_budget`
is held to the same three. The pilot shows the same pattern (90 and 89 calls for
30 shots).

The prompts and inference logic are byte-identical to the pilot's code
(`origin/feature/llm-30shot-evaluation`). The only code change is the
`--conditions` option that restricts the run to the three frozen conditions, so
the implementation hash differs from the pilot's.

**Model identity check.** Before the run, confirm that the `qwen3:4b-instruct`
digest from `ollama list` matches the pilot's `spec.model_digest` in
`artifacts/llm_balanced_nov16_30shot/experiment.json`, and record both values
below. A different digest is a different model; report it rather than silently
continuing.

## Execution

Run on a machine that already has Ollama, `qwen3:4b-instruct`, and the
reproduced day-1 baseline under `artifacts/day1/`:

```bash
git checkout <protocol commit>
ollama serve            # in another terminal, if not already running
scripts/run_llm_final.sh
```

The script:

1. refuses to run if `agents.py` differs from the frozen hash or `src/` has
   uncommitted changes;
2. runs the read-only preflight against the registered test selection;
3. smoke-tests 3 **validation** shots to confirm real inference works;
4. runs the test split with `--final`, retrying up to 3 times through the cache
   until acceptance passes (no failed predictions; 100 shots; 50 balanced games);
5. strictly validates artifacts against traces, then scores frozen XGBoost on
   the test split without refitting;
6. writes `reports/llm_final_test_100/` and bundles everything into
   `artifacts/llm_final_test_100_bundle.tar.gz`.

**Expected cost.** 700 model calls. The pilot averaged about 311 s per shot for
these three conditions, so the run takes **about 8–9 hours** on the pilot machine.
Run it on mains power with sleep disabled (for example `caffeinate -i
scripts/run_llm_final.sh` on macOS). If it is interrupted, re-run the same
command: completed predictions are reused from the cache.

A full rehearsal of the script against a stand-in Ollama server and synthetic
data passed preflight, acceptance, strict validation, reporting and bundling,
and a re-run reused all 300 predictions with zero new calls. Those rehearsal
numbers are not results.

## After the run

1. Commit `reports/llm_final_test_100/` and share the bundle.
2. Record the run details below.
3. Write `docs/experiments/llm_final_results.md` applying the decision rule
   above, with the expected-precision bound and the limitations below.

### Run record (fill in after execution)

| Field | Value |
|---|---|
| Machine / OS | |
| Ollama version | |
| Model digest (run) | |
| Model digest (pilot) | |
| Prepared-data SHA-256 | |
| Git commit | |
| Started / finished (UTC) | |
| Attempts needed | |

## Limitations fixed in advance

- One small local model (4B parameters); conclusions do not extend to larger models.
- One season of public tabular data (2014–15); no computer-vision features.
- 100 shots / 50 games bounds the detectable effect (see expected precision).
- The decomposed and ensemble controls are reported only from validation.
