#!/usr/bin/env bash
# Final held-out LLM orchestration experiment (research question 2).
#
# Protocol: docs/experiments/llm_final_protocol.md. Do not edit the frozen
# settings below; a changed setting is a different experiment.
#
# Usage (from the repository root, Ollama running locally):
#   scripts/run_llm_final.sh
#
# Safe to re-run: completed predictions are reused from the run cache, so a
# timeout or interruption only repeats the unfinished calls.
set -euo pipefail

# ---- Frozen protocol settings ----------------------------------------------
MODEL="qwen3:4b-instruct"
SEED=42
LIMIT=100
GAMES=50
CONDITIONS=(llm_single llm_single_budget llm_multi)
FROZEN_AGENTS_SHA256="aec1e4fe23e723cf89e7de12ffbec30c448a22d99624e93ba5e18f67a9fadac4"

# ---- Local paths (override with environment variables if needed) -----------
# Prepared Kaggle data and frozen baseline models from the reproduced day-1 run.
BASELINE_RUN="${BASELINE_RUN:-artifacts/day1}"
SHOTS="${SHOTS:-$BASELINE_RUN/prepared/shots.parquet}"
BASELINE_MODELS="${BASELINE_MODELS:-$BASELINE_RUN/models}"
OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

OUT="artifacts/llm_final_test_${LIMIT}"
SMOKE="artifacts/llm_final_smoke_validation"
XGB="artifacts/xgboost_final_test"
REPORT="reports/llm_final_test_${LIMIT}"
SUMMARY="artifacts/llm_final_test_${LIMIT}_summary"
BUNDLE="artifacts/llm_final_test_${LIMIT}_bundle.tar.gz"

run() { uv run --frozen "$@"; }
log() { printf '\n==> %s\n' "$*"; }

# ---- 0. Environment and freeze checks --------------------------------------
log "Checking the frozen protocol"
actual_sha=$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' src/nba_shots/agents.py)
if [[ "$actual_sha" != "$FROZEN_AGENTS_SHA256" ]]; then
  echo "src/nba_shots/agents.py does not match the frozen protocol ($actual_sha)." >&2
  echo "Check out the protocol commit before running." >&2
  exit 1
fi
if [[ -n "$(git status --porcelain -- src configs pyproject.toml uv.lock)" ]]; then
  echo "Uncommitted changes in src/, configs/ or the lockfile; refusing to run." >&2
  exit 1
fi
for path in "$SHOTS" "$BASELINE_MODELS/run.json"; do
  [[ -f "$path" ]] || { echo "Missing $path (set BASELINE_RUN/SHOTS/BASELINE_MODELS)." >&2; exit 1; }
done
uv sync --frozen --all-extras
ollama list | grep -q "^${MODEL}[[:space:]]" || {
  echo "Ollama model $MODEL is not installed; run: ollama pull $MODEL" >&2
  exit 1
}
mkdir -p artifacts
{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "agents_sha256=$actual_sha"
  echo "shots_sha256=$(python3 -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$SHOTS")"
  echo "host=$(uname -sm)"
  echo "started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ollama --version 2>/dev/null | sed 's/^/ollama_version=/'
} > "artifacts/llm_final_test_${LIMIT}_provenance.txt"

# ---- 1. Read-only preflight on the registered test selection ---------------
if [[ ! -e "$OUT" ]]; then
  log "Preflight (no inference, no files written)"
  run python -m nba_shots.experiment_preflight \
    --shots "$SHOTS" --output "$OUT" --model "$MODEL" --split test \
    --limit "$LIMIT" --games "$GAMES" --seed "$SEED" --ollama-url "$OLLAMA_URL"
fi

# ---- 2. Smoke test on validation shots only --------------------------------
log "Smoke test: 3 validation shots (does not touch the test split)"
run nba-shots predict-llm --shots "$SHOTS" --output "$SMOKE" --model "$MODEL" \
  --split validation --limit 3 --games 3 --seed "$SEED" \
  --conditions "${CONDITIONS[@]}" --ollama-url "$OLLAMA_URL" > /dev/null
run python -m nba_shots.experiment_acceptance "$SMOKE" --shots 3 --games 3

# ---- 3. Final held-out run, retried through the cache ----------------------
accepted=0
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  log "Final test run, attempt $attempt/$MAX_ATTEMPTS ($LIMIT shots, $GAMES games)"
  run nba-shots predict-llm --shots "$SHOTS" --output "$OUT" --model "$MODEL" \
    --split test --final --limit "$LIMIT" --games "$GAMES" --seed "$SEED" \
    --conditions "${CONDITIONS[@]}" --ollama-url "$OLLAMA_URL" > "$OUT.attempt${attempt}.json"
  if run python -m nba_shots.experiment_acceptance "$OUT" --shots "$LIMIT" --games "$GAMES" \
      > "$OUT.acceptance.json"; then
    accepted=1
    break
  fi
  cat "$OUT.acceptance.json" >&2 || true
done
if [[ "$accepted" != 1 ]]; then
  echo "Final run not accepted after $MAX_ATTEMPTS attempts; inspect $OUT/predictions.csv." >&2
  exit 1
fi
cat "$OUT.acceptance.json"
run python -m nba_shots.experiment_validation "$OUT" > "$OUT.validation.json"

# ---- 4. Frozen XGBoost on the same test split (no refitting) ---------------
log "Scoring frozen XGBoost on the test split"
if [[ ! -e "$XGB" ]]; then
  cp -R "$BASELINE_MODELS" "$XGB"
  run nba-shots evaluate --run "$XGB" --shots "$SHOTS" --split test --final > /dev/null
fi

# ---- 5. Matched report and summaries ---------------------------------------
log "Writing the matched-shot report"
rm -rf "$REPORT" "$SUMMARY"
run nba-shots report \
  --predictions "$XGB/test_predictions.csv" "$OUT/predictions.csv" \
  --selection "$OUT/selection.csv" --output "$REPORT" > /dev/null
run python -m nba_shots.experiment_summary --run "$OUT" --output "$SUMMARY" \
  --require-accepted --shots "$LIMIT" --games "$GAMES" > /dev/null

echo "finished_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "artifacts/llm_final_test_${LIMIT}_provenance.txt"

# ---- 6. Bundle everything needed for analysis ------------------------------
log "Bundling results"
tar -czf "$BUNDLE" \
  "$OUT" "$OUT".*.json "artifacts/llm_final_test_${LIMIT}_provenance.txt" \
  "$SMOKE" "$XGB"/test_* "$REPORT" "$SUMMARY"
log "Done. Share $BUNDLE (and commit $REPORT/report.md)."
sed -n '/^## /,$p' "$REPORT/report.md" | head -40
