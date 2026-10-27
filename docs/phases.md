# Project Phases: Execution Plan and Remaining Work

**Course:** CMSC473

**Team:** Praharsh Nanduri, Kiran Manoj, Anuraag Tripathy, Prannav Chettipalayam, Varun Kota

**Status reviewed:** October 27, 2026

This is the task and acceptance checklist for finishing the research project. The
[experiment guide](experiments.md) is the shorter command reference. The
[baseline report](../reports/day1.md) records the completed preliminary results.

**Current position:** tabular baseline results exist and were reproduced. Video,
CV, single-LLM, and CrewAI tooling is implemented, but real-video and real-LLM
experiments have not produced results. The video source is currently blocked for
the sampled shots. Passing unit tests does not complete a research experiment.

## 1. The four experiments and the claims they can support

| Experiment | Implemented | Empirical progress | Required result |
|---|---|---|---|
| E1: XGBoost on traditional statistics | Data cleaning, splitting, training, scoring, saved pipelines | Kaggle validation run reproduced | A final held-out result; also a tabular baseline on the exact video cohort for E2 |
| E2: XGBoost with CV features | Retrieval, annotation UI, release-frame pose extraction, combined-feature training | No usable clips, real annotations, or CV feature table yet | Paired comparison against E1 on the same video-cohort shots and splits |
| E3: Single-LLM baseline | Local inference through CrewAI, schema checks, caches, traces | Integration tested with controlled fake responses; no real inference results | Scored predictions from the frozen local model on a registered evaluation selection |
| E4: Multi-agent CrewAI | Three specialists and one aggregator; call-budget enforcement | Integration tested with controlled fake responses; no real inference results | Scored predictions and paired comparison with E3 using matching information and shots |

The additional `llm_single_budget` condition uses four iterative calls from one
analyst. It is a required control for the orchestration claim, alongside the
one-call single-agent baseline; it is not a fifth independent research question.

Keep these comparisons separate:

- **CV contribution:** `xgboost_cv` versus `xgboost`, trained and evaluated on the
  same recent-video cohort, with CV features as the intended difference.
- **Orchestration contribution:** `llm_multi` versus `llm_single_budget`, both
  using four calls and the same available information. Also report comparison
  with one-call `llm_single`.
- **Predictor comparison:** XGBoost versus the LLM conditions on identical
  evaluation shot IDs, with their feature sets explicitly documented.

Kaggle scores and recent-video scores cannot establish a CV benefit across
datasets. A tabular-only agent pilot can establish operational feasibility and
provide a separate tabular comparison while video collection is blocked.

## 2. Verified snapshot and immediate priorities

### 2.1 Completed evidence

- Kaggle preprocessing, chronological splits, Constant, Logistic Regression,
  and XGBoost training are working.
- Independent runs under `artifacts/day1/` and `artifacts/day1_reproduced/`
  matched prepared data, validation predictions, and metrics. The saved
  reproducibility record is `artifacts/day1/reproducibility.json`.
- Baseline training uses training-only preprocessing and excludes outcome fields
  from model inputs. Final test evaluation requires an explicit `--final` flag.
- The annotation tool is merged. Subsequent fixes use clip metadata instead of a
  30 FPS assumption, show exact decoded still frames, persist discovered clip
  paths, reject invalid release indices, and prevent stale frame boxes.
- Last full verification during the annotation fix: **38 tests passed**, Ruff
  lint and formatting passed. Browser QA covered synthetic 24 and 60 FPS clips;
  regression tests also covered 29.97 FPS. Real pose inference was substituted
  in that QA, so this is not a real-video extraction result.

| Kaggle validation model | Accuracy | Brier score | Log loss | ROC-AUC |
|---|---:|---:|---:|---:|
| Constant | 55.11% | 0.247426 | 0.687991 | 0.500000 |
| Logistic regression | 60.84% | 0.234374 | 0.660993 | 0.627413 |
| XGBoost | 62.08% | 0.228181 | 0.646257 | 0.642430 |

These are development results, not final test scores or minimum thresholds a new
run must beat. The Constant model is a reference baseline, not a performance ceiling.

### 2.2 Existing data and blockers

| Item | Current evidence |
|---|---|
| Kaggle split | Train: 91,072 shots / 633 games; validation: 19,069 / 135; test: 17,928 / 136 |
| Recent NBA snapshot | 102,400 shots / 573 games; dates October 21, 2025 through January 29, 2026; possible row cap; full-season coverage unconfirmed |
| Registered video cohort | 1,050 shots / 30 games; 735 / 21 train, 140 / 4 validation, 175 / 5 test |
| Video availability | Zero MP4s in project data and zero annotated shots at this review |
| Retrieval log | Ten unique requested shots failed with `No video URL returned`; additional live probes from two other registered games also returned empty URLs |
| Alternate video probe | A constructed URL returned a decodable MP4 showing `VIDEO NOT AVAILABLE`; HTTP 200 and successful decoding were insufficient validation |
| CV/agent outputs | No real `cv_features.parquet`, agent `experiment.json`, or test predictions found in the reviewed local data/artifacts |
| Local LLM | Last runtime check found `deepseek-r1:8b` only; the configured Qwen model was absent. Recheck with `ollama list` before inference. |
| Feature importance | Code exists on `origin/feature/xgboost-importance`, not current main. Its three tests passed in isolation; import ordering and formatting needed cleanup. |

Four validation games are a small basis for game-bootstrap uncertainty estimates.
Do not describe the current video sample size as sufficient merely because it has
1,050 registered rows. Availability, usable pose coverage, and independent game
count must all be reported.

### 2.3 Work order

1. **Now:** verify the environment and run a small real tabular-only CrewAI pilot.
2. **Now:** review/clean the feature-importance branch; analyze the existing
   baseline after integration.
3. **Video track:** establish one usable source and a reviewed pilot before
   attempting large-scale annotation or CV training.
4. Freeze the matched cohort, history policy, evaluation selection, and budgets;
   run validation comparisons and failure analysis.
5. Freeze the final protocol, evaluate held-out tests, then finish the report,
   slides, demo, and reproducibility package.

Suggested leads below follow the existing roadmap. The team should confirm task
ownership and record actual completion evidence rather than assuming an owner has
already completed the work.

## Phase 0. Environment and reproducibility readiness

**Status:** core environment verified; real-inference readiness pending.

**Suggested owner:** everyone; each track lead records its execution environment.

### Tasks and commands

- [ ] Use Python 3.11 or 3.12 (`pyproject.toml` requires `>=3.11,<3.13`).
- [ ] Synchronize the locked environment with both optional feature groups.
- [ ] On macOS, confirm the XGBoost OpenMP runtime is installed.

```bash
uv sync --frozen --all-extras
# macOS only, if libomp is missing:
brew install libomp
uv run --frozen --all-extras nba-shots --help
uv run --frozen --all-extras pytest -q
uv run --frozen --all-extras ruff check src tests
uv run --frozen --all-extras ruff format --check src tests
```

Use `--all-extras` for vision and agent commands throughout this plan so a later
`uv run` does not omit their optional dependencies. Core tabular work also works
without those extras. CUDA is optional; the vision CLI defaults to `--device cpu`.
Record the actual device, memory, runtimes, and package versions used.

- [ ] Keep raw data, MP4s, model weights, models, caches, and traces out of Git.
- [ ] Commit source/config/test changes before a registered experiment. Record
  the Git commit and whether the tree was dirty when the run began.
- [ ] Preserve `uv.lock`, source hashes, config, prompts, model digest, weight
  hash, seeds, and evaluation shot IDs with each result.
- [ ] Use a new directory for changed experiments. Do not overwrite existing
  prepared data, frozen splits, extraction outputs, or fitted-model runs.
- [ ] Back up local artifacts in the team's chosen shared storage with a clear
  manifest of files and hashes; a Git checkout alone does not contain ignored data.

**Acceptance:** the complete test/lint/format checks pass in the environment used
for the run. Dependency readiness is recorded separately from real inference.

## Phase 1. Preserve and finish the traditional-statistics baseline (E1)

**Status:** preliminary Kaggle validation complete; final evaluation pending.

**Suggested leads:** Varun and Praharsh.

### 1.1 Existing artifacts to retain

`artifacts/baseline/`, `artifacts/day1/`, and `artifacts/day1_reproduced/` contain
saved baseline runs. Each has `prepared/shots.parquet`, `prepared/splits.json`,
`prepared/config.json`, `prepared/audit.json`, and a `models/` folder with fitted
pipelines, validation predictions, metrics, calibration plots, and `run.json`.
The concise committed result summary is [reports/day1.md](../reports/day1.md).

### 1.2 Reproduce when needed

Do not recreate an existing output directory. The following is a new-run example:

```bash
uv run --frozen --all-extras nba-shots baseline \
  --input data/raw/shot_logs.csv \
  --config configs/default.json \
  --output artifacts/kaggle_validation_run2

uv run --frozen --all-extras nba-shots report \
  --predictions artifacts/kaggle_validation_run2/models/validation_predictions.csv \
  --output reports/kaggle_validation_run2
```

Omit `--input` only when a Kaggle download is needed. Exact reproducibility should
be checked against the saved data/config/software, not by silently substituting a
new upstream CSV or a different feature set.

### 1.3 Remaining checks

- [ ] Preserve input checksums, missing/invalid value audits, and duplicate checks.
- [ ] Verify no game or date spans partitions and that
  `max(train dates) < min(validation dates)` and
  `max(validation dates) < min(test dates)`.
- [ ] Keep `SHOT_RESULT` as the target only. Exclude `FGM`, `PTS`, `W`,
  `FINAL_MARGIN`, identities, and other outcome-bearing inputs from predictors.
- [ ] Document the actual feature list and preprocessing. Missing numeric values
  are imputed from training data, with indicators; logistic features are scaled.
- [ ] Inspect the reliability plot; a good Brier score alone is not a full
  calibration analysis. No fitted post-hoc calibrator currently runs automatically.
- [ ] If tuning XGBoost, record the candidate grid, validation criterion, and chosen
  configuration. Freeze the search before any test evaluation.
- [ ] Keep final Kaggle test results separate from the video cohort's final test.

**Acceptance:** source provenance and splits are reproducible, metrics can be
recomputed from saved probabilities, and the final test remains untouched until
Phase 7. A matched video-cohort baseline must still be trained for E2.

## Phase 2. Freeze data contracts, history policy, and video sourcing

**Status:** source snapshot and registered cohort exist; usable footage blocked.

**Suggested lead:** Prannav, with team review of cohort/protocol decisions.

### 2.1 Use the actual path layout

| Purpose | Existing path |
|---|---|
| Recent raw API response and normalized source | `data/raw/nba_2025_26/` |
| Immutable selection audit and editable annotations | `data/video/manifest.selection.json`, `data/video/manifest.csv` |
| Download attempts and clips | `data/video/clips/retrieval.jsonl`, `data/video/clips/*.mp4` |
| Frozen registered shots and splits | `data/processed/video_cohort/` |
| New CV extraction | `data/video/cv_run1/` |
| New matched model run | `artifacts/video_validation_run1/models/` |
| New multimodal LLM run | `artifacts/video_agents_validation_run1/` |

Paths ending in `run1` below are examples for outputs not yet created. Increment
all related paths for subsequent runs. Reuse the existing registered cohort unless
a documented protocol change requires a separately named cohort.

### 2.2 Audit the source before changing or expanding the cohort

- [ ] Review `source.json` and `coverage_review.json`; retain the possible-cap
  warning and observed date range. Do not claim a full season was collected.
- [ ] Confirm game/event IDs, dates, labels, and source hashes are consistent.
- [ ] Recent shot data lacks `shot_clock`, `defender_distance`, `dribbles`, and
  `touch_time`; keep those missing. Do not fabricate tracking measurements.
- [ ] If broader coverage is needed, implement and validate collection by
  bounded dates/teams or another supported route, deduplicate by shot ID, and
  archive a new source snapshot. The current `collect-recent` command makes one
  request; it does not implement source pagination or prove complete coverage.
- [ ] Evaluate the practical number of sourceable games and an annotation budget
  before committing to a larger experiment. Set sample size and selection rules
  before examining comparative model performance.

### 2.3 Freeze player-history semantics and equal information

`configs/default.json` disables history. `configs/video.json` enables it, and the
existing `data/processed/video_cohort/config.json` confirms **history is enabled**.
Do not overwrite that cohort using the default config and assume it is equivalent.
Training loads a prepared cohort's saved config when `--config` is omitted.

The current history code computes prior attempts and a Beta(1,1)-smoothed make
rate using strictly earlier dates in the available source. This is **observed
source-window history**, not verified career history. It can include outcomes
from earlier held-out dates when predicting a later date.

- [ ] Explicitly choose rolling historical availability (earlier completed games
  are known at prediction time) or a fixed training-cutoff history policy.
- [ ] Document the choice before evaluation. If fixed-cutoff history is required,
  change and test history preparation before generating a new cohort; the CLI
  currently has no flag to switch those two policies.
- [ ] Confirm no same-date or future outcomes enter history.
- [ ] Compare the actual model feature lists and LLM payload fields. In a
  history-disabled experiment the history specialist has no history measurements;
  document or revise this design before claiming a meaningful history-agent test.
- [ ] Keep history policy and available measurements identical across compared
  predictors. Specialists may receive subsets; the aggregator sees the full
  available set. Labels and identifying metadata remain excluded from prompts.

### 2.4 Creation commands for a new source/cohort only

The existing paths are already populated. **Skip these creation commands for the
current cohort.** For a new version, replace every source, manifest, and cohort
path consistently; do not delete an old version to make these commands run.

```bash
uv run --frozen --all-extras nba-shots collect-recent \
  --season 2025-26 --output data/raw/nba_2025_26_v2

uv run --frozen --all-extras nba-shots make-manifest \
  --shots data/raw/nba_2025_26_v2/shots.parquet \
  --output data/video_v2/manifest.csv \
  --games 30 --shots-per-game 35 --seed 42

uv run --frozen --all-extras nba-shots prepare-cohort \
  --shots data/raw/nba_2025_26_v2/shots.parquet \
  --manifest data/video_v2/manifest.csv \
  --config configs/video.json \
  --output data/processed/video_cohort_v2
```

`make-manifest` requires at least three dates for chronological splitting. Register
and split before reviewing model scores or filtering on clip/pose success. The
annotation CSV changes as work proceeds; retain its original selection audit,
backups, final annotation snapshot/hash, and frozen prepared shot IDs.

### 2.5 Video-source decision and retrieval limits

The existing automatic route uses `VideoEventsAsset`. The tested requests returned
valid play metadata but empty media URLs. This is an unresolved source-availability
problem, not proof that dependencies failed. An alternate UUID probe produced a
placeholder video. Repeating the same failed request is not a collection strategy.

A small diagnostic retry is appropriate only when checking changed availability
or a changed retrieval method:

```bash
uv run --frozen --all-extras nba-shots fetch-videos \
  --manifest data/video/manifest.csv \
  --output data/video/clips --limit 10
```

- `--limit 10` means the first ten manifest rows, not ten new rows or ten different
  games. The first ten current rows all belong to one game.
- There is currently no `--offset`, game-stratified retrieval, or skip-known-failure
  option. Implement and test such controls if they are needed for a source probe.
- Downloaded files have a 100 MiB cap and are renamed from `.part` on completion.
  The code sleeps 0.6 seconds between attempts; this is a local throttle, not a
  verified upstream rate-limit guarantee.
- The command can exit successfully with every shot failed. Inspect its JSON
  counts and per-shot log; shell exit code alone does not prove usable footage.
- If improving retrieval, add tests for empty URLs, placeholder rejection,
  invalid downloads, and batch selection. Do not construct guessed UUID URLs and
  assume a decodable response is the requested play.

Choose and record one route:

- [ ] **Existing registered shots:** obtain manually indexed clips from actual
  available game footage, matching exact game/event IDs and source context.
- [ ] **Separately registered accessible-video cohort:** if the current games
  cannot be sourced, preserve the failed cohort and register a new one using
  accessible game recordings. Select games/shots without conditioning on outcome;
  document selection bias and keep it separate from the original experiment.
- [ ] If no usable source is available, label E2 blocked and complete the tabular
  and agent tracks. Do not replace missing CV results with synthetic results.

### 2.6 Acquisition/review log

Create `data/video/clip_review.csv` (or the equivalent new-cohort path) as a manual
review artifact. This is a planned artifact, not an existing CLI-generated file.
Use one row per attempted shot and retain failures. Recommended columns:

```text
shot_id,game_id,game_event_id,source_url,source_recording,clip_path,
source_start_seconds,clip_sha256,reviewer,reviewed_at,identity_status,
footage_status,release_visible,shooter_pose_visible,defender_visible,notes
```

If cropping a longer recording, record its hash and the crop offset as well as the
final clip hash. Never change/crop a clip after its frame indices are annotated
without invalidating and redoing those annotations.

A successful download must pass human review: real footage, correct teams/date,
quarter/clock, shooter/action, and enough pre-release context. Mark placeholder,
wrong-play, obscured-release, and decode failures separately. Clip availability
or score predictions must not silently determine which registered rows survive.

**Acceptance:** there is a frozen source/selection/history contract and a feasible
footage route. Cohort counts are known per split/game. At least one real clip has
verified identity before annotation is scaled.

## Phase 3. Real-video annotation and CV validation (E2 prerequisite)

**Status:** tooling implemented; real-video pilot pending.

**Suggested lead:** Prannav; a second teammate independently reviews annotations.

### 3.1 File discovery and annotation

Store a clip such as `nba:0022500004:163` as:

```text
data/video/clips/nba_0022500004_163.mp4
```

Alternatively, enter `clip_path` in the manifest, relative to the manifest folder
or absolute. The tool resolves discovered files and saves a manifest-relative path.

```bash
uv run --frozen --all-extras nba-shots annotate \
  --manifest data/video/manifest.csv --videos data/video/clips \
  --weights yolo11n-pose.pt --device cpu
```

The server opens at `http://127.0.0.1:8765`. Work with one annotator/writer per
manifest: current saves rewrite the CSV directly, not through an atomic replace or
multi-user conflict-control system. Back it up before a session and after review.

- [ ] Inspect the video and establish the correct play identity first.
- [ ] Agree on a release convention, such as the first frame showing the ball
  separated from the fingers. Record ambiguous release timing for review rather
  than presenting an uncertain frame as exact ground truth.
- [ ] Pause and step using arrows; annotation uses the displayed server-decoded
  frame index, not an assumed 30 FPS browser timestamp.
- [ ] Press **R** to mark that frame and run YOLO. Assign the correct shooter
  box and, when visible, contest defender box. Select left/right shooting hand.
- [ ] Press **S**, reopen the shot, and confirm frame, boxes, hand, and path.
- [ ] Check that changing frames clears old boxes and changing clips does not
  apply a late detector result to the new clip.
- [ ] Check box placement when footage is letterboxed/resized. Do not identify
  the same detection as both shooter and defender.
- [ ] Preserve release-only model inputs. Later trajectory or outcome frames must
  not become CV features or LLM inputs.

### 3.2 Pilot and independent review

1. Verify the complete save/reopen/extract path on one clip.
2. Review 5-10 clips across available games to identify repeated acquisition,
   frame, player-identity, and keypoint problems.
3. Expand to roughly 20 independently reviewed clips across games for a feasibility
   gate. These counts are workflow targets, not a statistical power guarantee.
4. A second reviewer checks release timing, hand, shooter/defender identity, and
   visibility. Record disagreements and their adjudication in the review log.
5. If proceeding to 50 or more annotations, select them according to the registered
   protocol. Fifty clips is a milestone, not evidence that E2 is sufficiently powered.

### 3.3 Extract a new CV table

The first use may download YOLO weights. Record the actual weight file/hash and
use the same model for annotation and extraction. Do not overwrite earlier outputs.

```bash
uv run --frozen --all-extras nba-shots extract-cv \
  --manifest data/video/manifest.csv \
  --output data/video/cv_run1 \
  --weights yolo11n-pose.pt --device cpu
```

| Feature | Measurement and limitation |
|---|---|
| `elbow_angle_deg` | Shooting-side shoulder-elbow-wrist angle in the 2D image; not physical ball release/entry angle |
| `wrist_height_torso` | Vertical image displacement of shooter wrist above shoulder midpoint, divided by torso length |
| `defender_distance_torso` | Shooter/defender hip-midpoint distance in image coordinates, divided by shooter torso length; not court distance |
| `defender_wrist_height_torso` | Highest visible defender wrist relative to shooter shoulder line, in shooter torso units; not physical contest height |

Torso length is shoulder-midpoint to hip-midpoint image distance. Confidence below
0.5 or unavailable keypoints produce missing measurements. Perspective and
foreshortening remain limitations despite normalization. Ball tracking, spin,
homography, and physical 3D mechanics are not implemented.

### 3.4 Validate the extraction

```bash
uv run --frozen --all-extras python - <<'PY'
from pathlib import Path
import pandas as pd
from nba_shots.common import CV_FEATURES

manifest = pd.read_csv('data/video/manifest.csv')
cv = pd.read_parquet('data/video/cv_run1/cv_features.parquet')
assert not cv.shot_id.duplicated().any()
assert set(cv.shot_id) == set(manifest.shot_id)
print('Status counts:', cv.cv_status.value_counts(dropna=False).to_dict())
print('Feature availability:', cv[sorted(CV_FEATURES)].notna().sum().to_dict())
print(cv.loc[cv.cv_status.eq('error'), ['shot_id', 'cv_error']].head(10).to_string(index=False))
print('Saved overlays:', len(list(Path('data/video/cv_run1/audit_frames').glob('*.jpg'))))
PY
```

- `ok` means all four measurements are finite; it does not independently prove
  correct identity or geometry.
- `partial` means extraction ran but one or more measurements are missing; an
  unannotated/occluded defender can yield this status.
- `error` retains the registered shot and error reason with missing CV features.
- During a small pilot, most of 1,050 registered shots may still be errors. Report
  pilot coverage explicitly; do not call the full cohort processed successfully.
- Review `latest_frame_used`, `release_frame`, clip hash, feature plausibility,
  `extraction.json`, and visual overlays against the annotated source frames.
- The extractor saves overlays only for successful/partial rows within the first
  50 manifest positions. This is not 50 successful clips. Arrange additional audit
  images/review evidence for pilot clips outside those positions before scaling.
- Add status and per-feature coverage by split/game/visibility to the manual
  audit. Entirely missing CV features are not evidence of a meaningful CV condition.

**Acceptance:** reviewed real frames yield credible measurements, all registered
rows remain in the CV table, provenance is saved, and failures/coverage are
understood. Resolve poor coverage or revise a separate protocol before E2 training.

## Phase 4. Real local LLM/CrewAI pilot (E3 and E4; can start now)

**Status:** implemented and integration-tested; real inference pending.

**Suggested leads:** Kiran and Anuraag.

### 4.1 Install and verify the selected model

```bash
ollama list
ollama pull qwen3:4b-instruct
ollama list
```

Ensure the local service is running (use `ollama serve` in another terminal only
if it is not already running). The client accepts plain local HTTP endpoints at
`127.0.0.1`, `localhost`, or `::1`; the default is `http://127.0.0.1:11434`.
There is no paid/remote-provider fallback. If the configured tag is unavailable,
resolve and freeze an installed supported model/tag and update all run commands;
do not silently substitute the installed DeepSeek model.

### 4.2 Run a tabular-only smoke experiment first

This uses existing prepared Kaggle data and does not require videos. It is a
separate experiment from the eventual recent-video comparison.

```bash
uv run --frozen --all-extras nba-shots predict-llm \
  --shots artifacts/baseline/prepared/shots.parquet \
  --output artifacts/kaggle_agents_pilot1 \
  --model qwen3:4b-instruct --limit 5 --seed 42 --num-predict 192
```

| Condition | Calls per uncached shot | Information |
|---|---:|---|
| `llm_single` | 1 | All allowlisted measurements |
| `llm_single_budget` | 4 | Same analyst, prior assessments included at each step |
| `llm_multi` | 4 | Context specialist, history specialist, mechanics/defense specialist, then aggregator |

A fully successful uncached five-shot run makes **45 calls** and writes **15
prediction rows**. Four calls match the call budget, not necessarily tokens,
latency, or energy. Report those observed costs instead of claiming equal compute.
The current adapter uses one-agent CrewAI executions for each specialist step and
passes structured outputs to the next step; framework behavior must be checked in
real inference, not inferred from a mock integration test.

### 4.3 Verify the actual run

```bash
uv run --frozen --all-extras python - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path('artifacts/kaggle_agents_pilot1')
pred = pd.read_csv(root / 'predictions.csv')
assert not pred.duplicated(['model', 'shot_id']).any()
print(pred.groupby('model').agg(total=('shot_id', 'size'), successful=('status', lambda s: s.eq('ok').sum())))
good = pred.loc[pred.status.eq('ok')]
assert good.p_make.between(0, 1).all()
expected = {'llm_single': 1, 'llm_single_budget': 4, 'llm_multi': 4}
for name, calls in expected.items():
    fresh = good.loc[good.model.eq(name) & ~good.cached.astype(bool)]
    assert fresh.calls.eq(calls).all(), name
print('Calls this invocation:', json.loads((root / 'summary.json').read_text())['actual_calls_this_run'])
print('Failures:', pred.loc[pred.status.ne('ok'), ['shot_id', 'model', 'error']].to_dict('records'))
PY
```

- [ ] Inspect actual `/api/chat` traces and valid schema-parsed probabilities.
- [ ] Confirm each uncached successful condition used exactly its registered calls.
- [ ] Check traces for target, player/game identity, dates, filenames, or any
  outcome-bearing input. Only allowlisted measurements belong in prompts.
- [ ] Check the actual history payload. The default Kaggle baseline has no history
  columns; this pilot does not validate a populated player-history specialist.
- [ ] Review token counts, latency, model digest, and all failure reasons. The
  default per-call generation cap is 192 tokens; truncated JSON is a failure to
  resolve on validation data before freezing the inference specification.
- [ ] Re-run the exact same command to check cache reuse. If all pilot conditions
  succeeded and inputs/spec are unchanged, the repeated run should make zero new
  calls. Trace logs append; predictions/summary describe the latest invocation.
- [ ] Freeze behavior and explain persistent failures; do not keep changing
  settings in place and mix different model/prompt versions in one result.

No automatic calibration fitting is applied to aggregator output. Assess its
probabilities empirically. Temperature zero and a seed improve repeatability but
do not prove hardware-independent deterministic inference.

**Acceptance:** an actual local model produces valid, auditable predictions for
all pilot conditions, budgets and cache behavior are verified, and failures are
resolved or explicitly reported. Mock-response tests alone do not satisfy this gate.

## Phase 5. Run matched validation experiments

**Status:** pending; tabular-only agent comparisons can proceed before CV.

**Suggested owners:** all track leads jointly review the experiment contract.

### 5.1 Expand the tabular-only agent result

After the pilot passes, use a new output directory and a preregistered evaluation
limit. Selection is hash-ranked using shot IDs and seed, not outcomes or quality.
The CLI accepts limits 1-500; larger studies require a reviewed code/protocol change.

```bash
uv run --frozen --all-extras nba-shots predict-llm \
  --shots artifacts/baseline/prepared/shots.parquet \
  --output artifacts/kaggle_agents_validation_run1 \
  --model qwen3:4b-instruct --limit 100 --seed 42 --num-predict 192

uv run --frozen --all-extras nba-shots report \
  --predictions \
    artifacts/baseline/models/validation_predictions.csv \
    artifacts/kaggle_agents_validation_run1/predictions.csv \
  --selection artifacts/kaggle_agents_validation_run1/selection.csv \
  --output reports/kaggle_agents_validation_run1
```

For 100 fully successful uncached shots, budget 900 calls and 300 LLM prediction
rows. Measure actual runtime from the pilot before committing to the full run.
Changing `--limit` changes the experiment specification; it is a new run, not a
cache-resume of a five-shot pilot.

### 5.2 Train E1 and E2 on the same video cohort

Run only after Phase 3 passes. Use prepared shots, not raw source records. For the
existing history-enabled cohort, omit `--config` so its saved config is used.

```bash
uv run --frozen --all-extras nba-shots train \
  --shots data/processed/video_cohort/shots.parquet \
  --cv data/video/cv_run1/cv_features.parquet \
  --output artifacts/video_validation_run1/models
```

This trains Constant, Logistic Regression, `xgboost`, and `xgboost_cv` on identical
training rows, then scores identical validation rows. The CV table must contain
all registered shots, including failures; feature completeness is not required by
the trainer. Missing features are imputed using training data. Therefore a run can
finish with inadequate CV coverage; the human acceptance gate remains necessary.

- [ ] Confirm saved `run.json` feature lists differ as intended for XGBoost/CV.
- [ ] Check all predictors use the same history policy and source data.
- [ ] Save coverage by split/game alongside metrics. Retain full-cohort results;
  any complete-case or availability analysis must be separately labeled, with its
  selection rule defined in advance rather than chosen for favorable scores.
- [ ] Inspect missingness as a possible availability/context signal. Do not
  attribute gains from source availability alone to shooting mechanics.

### 5.3 Run E3 and E4 with the same registered CV inputs

```bash
uv run --frozen --all-extras nba-shots predict-llm \
  --shots data/processed/video_cohort/shots.parquet \
  --cv data/video/cv_run1/cv_features.parquet \
  --output artifacts/video_agents_validation_run1 \
  --model qwen3:4b-instruct --limit 100 --seed 42 --num-predict 192
```

- [ ] Preserve `experiment.json`, `selection.csv`, `predictions.csv`,
  `summary.json`, `traces.jsonl`, and `cache/`.
- [ ] Verify complete matching shot IDs and labels across conditions.
- [ ] Audit actual prompts and model features for equal information. Record
  intentional differences, including specialist subsets and missing measurements.
- [ ] Keep tabular-only and CV-supplied LLM runs in different directories and
  report them as distinct input conditions; neither establishes the other.

### 5.4 Align the model and agent evaluation selection

The existing video cohort has 140 validation shots. An LLM limit of 100 selects a
subset. Explicit selection is required before combining it with the model run:

```bash
uv run --frozen --all-extras nba-shots report \
  --predictions \
    artifacts/video_validation_run1/models/validation_predictions.csv \
    artifacts/video_agents_validation_run1/predictions.csv \
  --selection artifacts/video_agents_validation_run1/selection.csv \
  --output reports/video_validation_run1
```

Review `selection_audit.json`: excluded baseline rows should be explained by the
registered selection, not by silently dropping inference failures. Preserve both
the full-cohort XGBoost result and the selected-shot comparison.

**Acceptance:** every claimed comparison uses the intended identical shot IDs,
splits, labels, and available measurements. Every condition has metrics, coverage,
provenance, and observed compute costs. Real inference failures remain visible.

## Phase 6. Statistical comparisons, feature importance, and failure analysis

**Status:** comparison/report tooling implemented; research analysis pending.

**Suggested leads:** Praharsh for importance; all members for interpretation.

### 6.1 Required paired comparisons

```bash
uv run --frozen --all-extras nba-shots compare \
  --predictions artifacts/video_validation_run1/models/validation_predictions.csv \
  --baseline xgboost --candidate xgboost_cv \
  --output artifacts/video_validation_run1/compare_cv.json

uv run --frozen --all-extras nba-shots compare \
  --predictions artifacts/video_agents_validation_run1/predictions.csv \
  --baseline llm_single --candidate llm_multi \
  --output artifacts/video_agents_validation_run1/compare_multi_single.json

uv run --frozen --all-extras nba-shots compare \
  --predictions artifacts/video_agents_validation_run1/predictions.csv \
  --baseline llm_single_budget --candidate llm_multi \
  --output artifacts/video_agents_validation_run1/compare_multi_budget.json

uv run --frozen --all-extras nba-shots compare \
  --predictions reports/video_validation_run1/predictions.csv \
  --baseline xgboost_cv --candidate llm_multi \
  --output artifacts/video_agents_validation_run1/compare_multi_xgboost_cv.json

uv run --frozen --all-extras nba-shots compare \
  --predictions reports/video_validation_run1/predictions.csv \
  --baseline xgboost_cv --candidate llm_single \
  --output artifacts/video_agents_validation_run1/compare_single_xgboost_cv.json
```

The last two commands use the already aligned report predictions. The report
currently generates CV and agent-ablation comparisons automatically, but not
XGBoost-versus-agent comparisons; run those explicitly. For a tabular-only result,
use its aligned report predictions and `--baseline xgboost` instead.

The comparison code requires one cohort/split, identical nonempty shot IDs,
consistent labels, complete prediction coverage, and at least two held-out games.
It resamples whole games 1,000 times by default and reports candidate-minus-baseline
Brier difference; negative favors the candidate. A confidence interval spanning
zero does not establish improvement or equivalence. Few games make inference weak,
even when many shots come from those games.

- [ ] Prespecify primary contrasts and primary Brier outcome; treat other metrics,
  subgroups, and many additional comparisons as secondary/exploratory.
- [ ] Report Brier, log loss, ROC-AUC, accuracy (threshold 0.5), reliability plots,
  prediction coverage, held-out games/shots, latency, and token/call counts.
- [ ] Report undefined ROC-AUC for one-class subsets rather than inventing a score.
- [ ] Do not discard errors to force paired comparisons to pass. Resolve them
  under the frozen specification or report that the paired result is unavailable.
  Current successful-only metrics do not automatically penalize failed inference.
- [ ] Use validation for model/prompt development. Label validation uncertainty
  estimates as development evidence when those scores guided decisions.

### 6.2 Integrate and run feature importance

The pending branch contains `src/nba_shots/feature_importance.py`. Before merging:

- [ ] Review exact branch changes and preserve unrelated local work.
- [ ] Fix import order/formatting, run its three tests plus applicable regression
  checks, and verify compatibility with current main.
- [ ] Run the module against a real saved baseline; confirm input hash validation,
  validation-only selection, reproducible rankings, and artifact creation.

**Only after the module is integrated**, this is its invocation (not an existing
`nba-shots` subcommand):

```bash
uv run --frozen --all-extras python -m nba_shots.feature_importance \
  --run artifacts/baseline/models \
  --shots artifacts/baseline/prepared/shots.parquet \
  --output artifacts/kaggle_importance_run1 \
  --model xgboost --repeats 10 --seed 42
```

Repeat for the frozen video run with `--model xgboost_cv`, its matching prepared
shots/model paths, and a new output directory. Expected outputs are
`feature_importance.csv`, `feature_importance.json`, and `feature_importance.png`.
The method shuffles raw feature columns without refitting and measures increased
validation Brier loss. Positive importance means shuffling worsened predictions;
correlated predictors can share importance. This is predictive importance, not a
causal effect or a reason to select features using final test data.

`nba-shots evaluate` only rescored saved models; it does **not** compute feature
importance. Do not substitute that command for this pending analysis.

### 6.3 Failure and disagreement review

- [ ] Join predictions on cohort/split/shot ID; verify identities before ranking.
- [ ] Examine large model disagreements, largest squared errors, and extreme
  confidence predictions. Review at least five representative cases, including
  missing-pose/occlusion cases and both outcomes when available.
- [ ] Reopen the exact release frame and consult adjudicated annotations. Inspect
  the LLM trace, available measurements, and whether the rationale invents facts.
- [ ] Separate wrong source play, bad release timing, wrong player assignment,
  low-confidence pose, missing features, malformed inference, and probabilistic
  uncertainty. A plausible miss at high make probability is not by itself a bug.
- [ ] Record shot IDs, model probabilities, outcome, error type, reviewer,
  evidence references, and interpretation in `reports/failure_cases.md`.
- [ ] Define any subgroup analyses (shot value/distance/visibility) and show their
  sample sizes; label post-hoc patterns exploratory.

**Acceptance:** the two research questions are answered with matched evidence or
explicitly reported as unresolved, including negative/null results and limitations.

## Phase 7. Freeze the protocol and evaluate final held-out tests

**Status:** not started; test sets remain reserved.

**Suggested owner:** team sign-off on a written freeze record.

### 7.1 Freeze checklist

Create `reports/final_protocol.md` before running these commands:

- [ ] Record source/cohort/split/annotation/CV hashes and source coverage.
- [ ] Freeze features, history policy, preprocessing, hyperparameters, YOLO
  weights, prompts, model digest, seeds, generation limits, and execution device.
- [ ] Freeze the deterministic LLM test limit/selection rule. The CLI does not
  support arbitrary externally supplied inference selections; record its hash
  ranking rule and limit, and retain the emitted `selection.csv`.
- [ ] Freeze primary comparisons, metrics, handling of missing CV/inference errors,
  and bootstrap settings. Confirm enough independent held-out games for the claim.
- [ ] Define the runtime budget and how interruptions/cache resumes are handled.
- [ ] Resolve validation issues first. Preserve the original test after any
  accidental exposure and report deviations; a flag alone does not enforce a
  scientifically clean protocol.

### 7.2 Final video-cohort evaluation (only if E2 became feasible)

```bash
uv run --frozen --all-extras nba-shots evaluate \
  --run artifacts/video_validation_run1/models \
  --shots data/processed/video_cohort/shots.parquet \
  --split test --final

uv run --frozen --all-extras nba-shots predict-llm \
  --shots data/processed/video_cohort/shots.parquet \
  --cv data/video/cv_run1/cv_features.parquet \
  --output artifacts/video_agents_test_run1 \
  --model qwen3:4b-instruct --split test --final \
  --limit 100 --seed 42 --num-predict 192

uv run --frozen --all-extras nba-shots report \
  --predictions \
    artifacts/video_validation_run1/models/test_predictions.csv \
    artifacts/video_agents_test_run1/predictions.csv \
  --selection artifacts/video_agents_test_run1/selection.csv \
  --output reports/video_test_run1
```

The `evaluate` command writes test predictions/metrics into the existing fitted
model run without retraining. Back up the run first; repeated evaluation can
replace same-named outputs. Agent test outputs use a new directory. Run the Phase 6
comparison commands on the aligned **test** predictions with new output names;
never combine validation and test rows in a paired comparison.

### 7.3 Final Kaggle/tabular-only evaluation

If the frozen Kaggle models are still the chosen specification:

```bash
uv run --frozen --all-extras nba-shots evaluate \
  --run artifacts/baseline/models \
  --shots artifacts/baseline/prepared/shots.parquet --split test --final
```

For a final tabular-only agent comparison, use the same prepared Kaggle data,
`predict-llm --split test --final`, a new `artifacts/kaggle_agents_test_run1` output,
and the frozen model/budget/limit. Align with its test `selection.csv` when reporting.
If validation selected a different fitted run, use that run and its exact original
prepared input instead of this example.

**Acceptance:** final results come from the frozen protocol, have complete
provenance and matching comparison identities, and were not used to retune models
or prompts. Any deviations or source blockers remain explicit in the paper.

## Phase 8. Research paper, reproducibility, and presentation

**Status:** pending.

**Suggested owner:** all team members, with one person checking claims against artifacts.

### 8.1 Final paper and results checklist

- [ ] Map every claim to E1-E4, a cohort/split, and a saved result artifact.
- [ ] Separate implementation/testing achievements from empirical model results.
- [ ] Describe data selection, missing source coverage, registered failures,
  splits, history timing, features, preprocessing, inference, and cost budgets.
- [ ] Report the full-cohort and explicitly selected comparisons with game/shot
  counts, coverage, probability metrics, reliability plots, and paired intervals.
- [ ] Include the four-call single-agent control when attributing effects to
  orchestration. Do not equate identical call counts with identical token compute.
- [ ] Explain current scope changes from the proposal: local Qwen/Ollama rather
  than its originally named hosted model; 2D pose proxies rather than physical
  release angle/contest height/backspin; combined mechanics/defense specialist.
- [ ] Report source-window history accurately and distinguish player-history
  availability from model identity inference.
- [ ] Include representative failure cases and negative/inconclusive findings.
- [ ] If E2 stays blocked, state that CV effectiveness was not empirically tested;
  do not use synthetic browser tests or an all-missing CV table as evidence.

### 8.2 Deliverables and handoff

- [ ] Validation reports for completed cohorts/conditions, with selections and
  paired comparison files.
- [ ] Separate final test reports for experiments actually completed.
- [ ] `reports/final_protocol.md`, `reports/failure_cases.md`, and source/annotation
  review evidence in the appropriate private/local artifact location.
- [ ] Feature-importance plots/CSV/metadata after branch integration.
- [ ] Final paper, slide deck, and a demo showing a real annotated/extracted clip
  when available; label any synthetic demonstration explicitly.
- [ ] Updated README, experiment guide, roadmap, and this phase checklist with
  consistent paths, truthful status, and supported CLI commands.
- [ ] Fresh-environment verification using locked dependencies and a small
  supplied fixture or documented data-acquisition route. Distinguish unit-test
  reproducibility from availability of external NBA footage/model downloads.
- [ ] Review the final Git diff and run tests/lint/format checks. Commit reviewed
  source/config/docs/tests; share commits through the team's normal review flow.
- [ ] Provide artifact hashes and a minimal reproduction command list. Exclude
  credentials, local environment files, caches, weights, and raw footage from the
  source handoff unless separately and deliberately included.

## Completion rule

A phase is complete only when its acceptance checks have saved evidence. Mark
blocked and unresolved experiments honestly. The project can report a successful
baseline and controlled agent study even if video sourcing prevents the CV study;
that outcome must be stated as a limitation, not presented as proof against CV.
