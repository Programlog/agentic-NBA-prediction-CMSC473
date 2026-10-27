# Project Phases: End-to-End Execution Guide

**Course**: CMSC473  
**Team**: Praharsh Nanduri, Kiran Manoj, Anuraag Tripathy, Prannav Chettipalayam, Varun Kota  

Complete step-by-step execution guide covering every phase from environment setup to final report. Every command, artifact path, function reference, parameter value, and validation check is included.

---

## Phase 0: Environment Setup

**Status**: Complete  
**Owner**: All Team

### 0.1 Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11.x (< 3.13) | Enforced by `.python-version` and `requires-python` in `pyproject.toml` |
| uv | latest | Package manager — replaces pip/venv |
| libomp | any | macOS only: `brew install libomp` for XGBoost OpenMP runtime |
| Ollama | latest | Only needed for Phase 4 LLM experiments |

### 0.2 Install Dependencies

```bash
# Tabular baselines only (numpy, pandas, scikit-learn, xgboost, matplotlib, pyarrow, joblib, requests)
uv sync --frozen

# Full stack including vision extras (ultralytics>=8.3, opencv-python>=4.10, nba_api>=1.10)
# and agent extras (crewai[litellm]>=1.0, pydantic>=2.8)
uv sync --frozen --all-extras
```

The `--frozen` flag ensures `uv.lock` is not modified. Never run without `--frozen` unless intentionally updating dependencies.

### 0.3 Pull the Local LLM

```bash
ollama pull qwen3:4b-instruct
```

`OllamaClient.__init__` (agents.py:84) checks `/api/tags` on startup and raises `ValueError` if the exact model name `qwen3:4b-instruct` is not found in the local registry. The model digest is recorded in `experiment.json` for reproducibility.

### 0.4 Verify the Installation

```bash
# Core tests (no CV or agent dependencies required)
uv run --frozen pytest -q

# Full test suite including CrewAI integration tests
uv run --frozen --all-extras pytest -q

# Linting and import order
uv run --frozen ruff check src tests

# Format check (does not modify files)
uv run --frozen ruff format --check src tests
```

Test files and what they cover:
- `tests/test_data.py` — normalization, splitting, player history, overwrite guard
- `tests/test_models.py` — pipeline fit-on-train-only, end-to-end train/reload, test gate
- `tests/test_evaluation.py` — Brier/log-loss scoring, paired bootstrap
- `tests/test_vision.py` — manifest loading, pose math, torso normalization edge cases
- `tests/test_agents.py` — OllamaClient URL validation, parse_assessment, budget enforcement
- `tests/test_report.py` — report creation, selection alignment
- `tests/test_annotate.py` — validate_box, resolve_clip_path, update_manifest_row, HTTP server endpoints
- `tests/conftest.py` — shared `raw_shots` fixture (12 dates × 2 games × 4 shots = 96 rows)

### Artifacts Produced
None — environment only.

---

## Phase 1: Tabular Data Ingestion & Baselines

**Status**: Complete  
**Owner**: Varun, Praharsh  
**Key files**: `src/nba_shots/data.py`, `src/nba_shots/models.py`, `src/nba_shots/common.py`

This phase downloads the Kaggle 2014–15 shot logs (~128,000 attempts), audits and normalizes the CSV, applies strict chronological whole-date splitting, optionally computes player history, and trains three baseline models evaluated on the validation split only.

### 1.1 One-Command Baseline Run

```bash
uv run --frozen nba-shots baseline --output artifacts/baseline_run1
```

Internally, `cli.py:execute` runs in sequence:
1. `data.download_kaggle("data/raw/shot_logs.csv")` — downloads from `https://www.kaggle.com/api/v1/datasets/download/dansbecker/nba-shot-logs`, extracts `shot_logs.csv` from the ZIP, validates the header contains `SHOT_RESULT`, writes a `.source.json` provenance file
2. `data.prepare_csv(csv, artifacts/baseline_run1/prepared, config)` — normalizes, splits, writes `shots.parquet`
3. `models.train_baselines(shots, artifacts/baseline_run1/models, config)` — fits all models on train, scores validation

To provide an already-downloaded CSV (avoids re-downloading):
```bash
uv run --frozen nba-shots baseline \
  --input path/to/shot_logs.csv \
  --output artifacts/baseline_run1
```

The output directory **must not already exist or must be empty** — `train_baselines` raises `ValueError("Run directory is not empty")` on collision. Use a new directory for every run.

### 1.2 Configuration (`configs/default.json`)

```json
{
  "seed": 42,
  "train_fraction": 0.7,
  "validation_fraction": 0.15,
  "numeric_features": [
    "shot_distance", "defender_distance", "game_clock_seconds", "shot_clock",
    "period", "dribbles", "touch_time", "shot_value"
  ],
  "categorical_features": ["location"],
  "include_player_history": false,
  "xgboost": {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "min_child_weight": 10,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5,
    "n_jobs": 4
  }
}
```

`configs/video.json` sets only `"include_player_history": true` and merges over the defaults. Pass custom configs with `--config configs/video.json`. `load_config` (common.py:72) rejects any key not in `DEFAULT_CONFIG` and rejects any feature not in `TABULAR_FEATURES`.

### 1.3 Normalization (`data.normalize_kaggle`, data.py:84)

Required raw columns: `GAME_ID`, `MATCHUP`, `LOCATION`, `GAME_CLOCK`, `SHOT_RESULT`, `player_id`, `SHOT_NUMBER`, plus all of `NUMERIC_MAPPING`.

Column transformations:

| Raw Column | Output Column | Rule |
|---|---|---|
| `SHOT_RESULT` | `y_true` | `made`→1, `missed`→0; any other value raises `ValueError` |
| `MATCHUP` | `game_date` | Regex extracts `"NOV 14, 2014"` → `"2014-11-14"`; `NaN` raises `ValueError` |
| `GAME_ID` | `game_id` | Zero-padded to 10 digits via `normalize_game_id` |
| `GAME_CLOCK` | `game_clock_seconds` | `"5:30"` → `330.0`; invalid → `NaN`; clamped to 720s (reg) / 300s (OT) |
| `LOCATION` | `location` | `"H"` or `"A"`; anything else → `NaN` |
| `SHOT_DIST` | `shot_distance` | Invalid or negative → `NaN` |
| `CLOSE_DEF_DIST` | `defender_distance` | Invalid or negative → `NaN` |
| `SHOT_CLOCK` | `shot_clock` | Invalid, negative, or > 24 → `NaN` |
| `PERIOD` | `period` | Non-integer, < 1, or non-finite → `NaN` |
| `DRIBBLES` | `dribbles` | Invalid or negative → `NaN` |
| `TOUCH_TIME` | `touch_time` | Invalid or negative → `NaN` |
| `PTS_TYPE` | `shot_value` | Must be 2 or 3; anything else → `NaN` |

Outcome-leaking columns explicitly dropped: `FGM`, `PTS`, `W`, `FINAL_MARGIN`.

`shot_id` is constructed as `"kaggle:<game_id>:<player_id>:<shot_number>"`. Duplicate `shot_id` values raise `ValueError`.

### 1.4 Chronological Splitting (`data.assign_splits`, data.py:182)

Splits are assigned at whole-date granularity — no game or date ever spans partition boundaries.

```
max(train game_dates) < min(validation game_dates)
max(validation game_dates) < min(test game_dates)
```

Default fractions: 70% train / 15% validation / 15% test (by cumulative game count, not shot count). At least 3 distinct dates are required.

`validate_splits` (models.py:87) enforces these invariants before any model is trained and raises `ValueError` if any `game_id` or `game_date` crosses a partition boundary.

### 1.5 Optional: Player History (`data.add_player_history`, data.py:230)

Enable by setting `"include_player_history": true` in the config. Computes two features using only games on strictly earlier dates — never the same calendar day:

| Feature | Formula |
|---|---|
| `player_prior_attempts` | Count of all prior-date shots by this player |
| `player_prior_make_rate` | `(prior_makes + 1) / (prior_attempts + 2)` — Beta(1,1) Laplace prior |

For a player's very first game: `player_prior_attempts = 0`, `player_prior_make_rate = 0.5` (uninformative prior). This prevents learning the prior from future games.

### 1.6 Model Pipelines (`models.make_pipeline`, models.py:45)

All three models share the same `ColumnTransformer` preprocessing:
- **Numeric features**: `SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)` — adds binary missingness indicator columns
- **Categorical features**: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`
- **Logistic Regression only**: additional `StandardScaler` after imputation

| Model name | Estimator | Notes |
|---|---|---|
| `constant` | `DummyClassifier(strategy="prior")` | Predicts training make rate for all shots |
| `logistic_regression` | `LogisticRegression(max_iter=2000)` | With StandardScaler |
| `xgboost` | `XGBClassifier(**config["xgboost"], objective="binary:logistic", eval_metric="logloss")` | Tabular features only |
| `xgboost_cv` | Same XGBClassifier | Tabular + 4 CV features (`elbow_angle_deg`, `wrist_height_torso`, `defender_distance_torso`, `defender_wrist_height_torso`) |

Preprocessor is **fit only on the training split** and applied to validation/test. This is tested in `test_models.py:test_preprocessor_is_fit_only_on_training_data`.

### 1.7 Artifacts Produced

| Path | Description |
|---|---|
| `artifacts/baseline_run1/prepared/shots.parquet` | Normalized, split-tagged shot records (columns: `shot_id`, `game_id`, `game_date`, `player_id`, `cohort`, `y_true`, `split`, + all feature columns) |
| `artifacts/baseline_run1/prepared/audit.json` | `raw_rows`, `prepared_rows`, `make_rate`, `feature_missing_values`, `invalid_numeric_values_replaced_with_missing`, `excluded_outcome_columns` |
| `artifacts/baseline_run1/prepared/splits.json` | Per-split game lists, shot counts, date ranges, `shot_ids_sha256` |
| `artifacts/baseline_run1/prepared/config.json` | Config snapshot used for this run |
| `artifacts/baseline_run1/models/validation_predictions.csv` | Columns: `shot_id`, `game_id`, `cohort`, `split`, `model`, `y_true`, `p_make`, `status` |
| `artifacts/baseline_run1/models/validation_metrics.json` | List of `{cohort, split, model, n_total, n_scored, n_failed, coverage, brier_score, log_loss, roc_auc, accuracy}` |
| `artifacts/baseline_run1/models/validation_metrics.csv` | Same as above in CSV format |
| `artifacts/baseline_run1/models/validation_kaggle_2014_15_validation_calibration.png` | 10-bin uniform reliability diagram, 160 DPI |
| `artifacts/baseline_run1/models/constant.joblib` | `{pipeline, features, name}` dict |
| `artifacts/baseline_run1/models/logistic_regression.joblib` | Same |
| `artifacts/baseline_run1/models/xgboost.joblib` | Same |
| `artifacts/baseline_run1/models/run.json` | `started_at`, `completed_at`, `config`, `environment` (python version, package versions, git commit, git_dirty), `shots_sha256`, `evaluation_split`, per-model `fit_rows`/`validation_rows`/`runtime_seconds` |

### 1.8 Expected Validation Results

| Model | Brier Score | Log Loss | Target |
|---|---|---|---|
| `constant` | ~0.248 | ~0.693 | Ceiling — constant make-rate prior |
| `logistic_regression` | ~0.240 | ~0.670 | Linear tabular features |
| `xgboost` | ~0.237 | ~0.660 | **PRD target: < 0.245** |

---

## Phase 2: Video Collection & Computer Vision Pipeline

**Status**: Complete  
**Owner**: Prannav  
**Key files**: `src/nba_shots/vision.py`, `src/nba_shots/annotate.py`

This phase queries the NBA API for recent shot data, downloads broadcast video clips, runs the interactive annotation tool to mark release frames, and extracts YOLO-Pose keypoints as torso-normalized 2D proxies.

### 2.1 Collect Recent Season Shot Records

```bash
uv run nba-shots collect-recent --season 2025-26 --output data/recent
```

Calls `vision.collect_recent` (vision.py:28) which queries `nba_api.stats.endpoints.ShotChartDetail` with `team_id=0, player_id=0` (all players all teams), `context_measure_simple="FGA"`, `season_type_all_star="Regular Season"`, `timeout=30`.

Normalizes via `normalize_recent` (vision.py:70). `shot_id` is constructed as `"nba:<game_id>:<game_event_id>"`.

**Important**: Four features unavailable from `ShotChartDetail` are set to `NaN` in this cohort:
- `shot_clock`, `defender_distance`, `dribbles`, `touch_time`

These are available in the Kaggle 2014–15 cohort but not the recent API. This means the two cohorts are **not directly comparable** on full-feature models.

Raises `ValueError("NBA returned no shot attempts")` if the response is empty.

### 2.2 Generate Hash-Seeded Annotation Manifest

```bash
uv run nba-shots make-manifest \
  --shots data/recent/shots.parquet \
  --output data/recent/manifest.csv \
  --games 30 \
  --shots-per-game 35 \
  --seed 42
```

`vision.make_manifest` (vision.py:110) selects games and shots by `SHA-256(f"{seed}:{value}")` hash rank — deterministic and outcome-blind. Requires `game_event_id` in the shots file (only present for recent cohort, not Kaggle).

Manifest columns on creation: `shot_id`, `game_id`, `game_event_id`, `clip_path` (empty), `release_frame` (empty), `shooter_box` (empty), `defender_box` (empty), `shooting_side` (empty).

Raises `ValueError` if manifest file already exists — never overwrite annotations.

Also writes `data/recent/manifest.selection.json` with `shots_sha256`, `manifest_sha256`, `seed`, game/shot counts, and selection method description.

### 2.3 Fetch Video Clips

```bash
uv run nba-shots fetch-videos \
  --manifest data/recent/manifest.csv \
  --output data/recent/videos \
  --limit 50
```

`vision.fetch_videos` (vision.py:204) iterates the first `limit` rows of the manifest. For each shot:
1. Calls `VideoEventsAsset(game_id, game_event_id, timeout=20)`
2. Extracts URL from the response dict via `_video_url` (searches keys `lurl`, `murl`, `surl` in order)
3. Downloads the MP4 stream with a 100 MiB hard cap
4. Writes atomically: `.part` file during download, renamed to `<shot_id_colon_replaced_with_underscore>.mp4` on success
5. Sleeps 0.6 s between requests (NBA API rate limit)
6. Appends one JSON line per shot to `data/recent/videos/retrieval.jsonl` with `shot_id`, `url`, `status`, `clip_path`, `sha256`, or `error`

If a clip already exists it is skipped (idempotent). Status is `"downloaded_needs_manual_review"` — every downloaded clip must be inspected since a placeholder HTML page can be a valid MP4.

### 2.4 Interactive Video Annotation Tool

```bash
uv run nba-shots annotate \
  --manifest data/recent/manifest.csv \
  --videos data/recent/videos \
  --weights yolo11n-pose.pt \
  --device cpu \
  --port 8765 \
  --host 127.0.0.1
```

`annotate.start_annotation_server` (annotate.py:812) launches a `ThreadingHTTPServer` at `http://127.0.0.1:8765` and opens the browser automatically after 0.8 s.

**HTTP API surface**:

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | GET | Serves the full single-page annotation UI (HTML_TEMPLATE) |
| `GET /api/shots` | GET | Returns manifest as JSON array; NaN values replaced with `""` |
| `GET /api/video?shot_index=N` | GET | Streams the MP4 clip with HTTP Range request support (enables smooth browser seeking) |
| `POST /api/detect-frame` | POST | Body: `{shot_index, frame_index}` — runs YOLO on the frame, returns `{boxes: [{box: [x1,y1,x2,y2], confidence}], jpeg_data_url, width, height}` |
| `POST /api/save` | POST | Body: `{shot_id, release_frame, shooter_box, defender_box, shooting_side}` — atomically updates `manifest.csv` |

**Per-shot annotation workflow**:
1. Select a shot from the left sidebar (green badge = annotated, gray = pending)
2. Scrub to the frame where the ball **leaves the shooter's fingers** — this is the release frame; no follow-through or ball trajectory frames
3. Press **`R`** to set the release frame — YOLO person detection runs automatically on that frame
4. Click a detected bounding box to assign **Shooter** (blue, `🎯`) or switch mode with the Defender button and click to assign **Defender** (red, `🛡️`)
5. Set the shooting hand (right/left) in the dropdown
6. Press **`S`** to save — writes `release_frame`, `shooter_box`, `defender_box`, `shooting_side` to `manifest.csv`

**Keyboard shortcuts**: `Space` play/pause · `←`/`→` step one frame · `R` set release · `S` save & next

Video clip resolution is via `annotate.resolve_clip_path` (annotate.py:22) which checks: `clip_path` field directly, then relative to manifest dir, then `<shot_id_colons_to_underscores>.mp4` in `--videos`, manifest dir, `manifest_dir/videos`, `data/recent/videos`.

`annotate.update_manifest_row` (annotate.py:129) validates:
- `release_frame >= 0` (raises `ValueError("non-negative")`)
- `shooting_side in {"left", "right"}` (raises `ValueError("left")`)
- `shooter_box` required, `defender_box` optional
- Box format: `[x1, y1, x2, y2]` with `x2 > x1`, `y2 > y1` (raises `ValueError("positive area")`)

**Target**: Annotate ≥ 50 clips before proceeding to Phase 4.

### 2.5 Extract 2D Pose Features

```bash
uv run nba-shots extract-cv \
  --manifest data/recent/manifest.csv \
  --output data/recent/cv \
  --weights yolo11n-pose.pt \
  --device cpu
```

`vision.extract_cv` (vision.py:351) processes each annotated row:
1. Reads **only** the single annotated release frame via `read_release_frame` (vision.py:262) — strictly forbids decoding any follow-through frames
2. Runs `YOLO(weights).predict(image, device=device, verbose=False)` — 17 COCO keypoints per detected person
3. Matches annotated `shooter_box` to a detection via `match_box` (vision.py:280) — IoU threshold 0.2; raises `ValueError` if no match
4. Optionally matches `defender_box` the same way; raises `ValueError` if shooter and defender map to the same detection
5. Computes `pose_features` (vision.py:300) using COCO keypoint indices:
   - Shooting side `"right"`: shoulder=6, elbow=8, wrist=10; `"left"`: 5, 7, 9
   - Torso reference length T = `‖(p5+p6)/2 − (p11+p12)/2‖₂`; features are `NaN` if T < 1e-6
   - Keypoints with confidence < 0.5 produce `NaN` for that feature

**Feature math** (all in torso-normalized 2D image units):

| Feature | Keypoints Used | Formula |
|---|---|---|
| `elbow_angle_deg` | shoulder(s/e), elbow(e), wrist(w) | `arccos(clip((s−e)·(w−e) / (‖s−e‖·‖w−e‖), −1, 1)) × 180/π` |
| `wrist_height_torso` | wrist(w), shoulder midpoint | `(y_shoulders − y_wrist) / T` — positive = wrist above shoulders |
| `defender_distance_torso` | defender hips (p11+p12)/2, shooter hips | `‖hips_defender − hips_shooter‖₂ / T` |
| `defender_wrist_height_torso` | defender wrists (p9, p10), shooter shoulders | `(y_shoulders − min(y_w1, y_w2)) / T` — higher = better contest |

Shots with missing clips or failed pose extraction are **retained** in `cv_features.parquet` with `cv_status = "error"` and `NaN` for all 4 CV features. Never remove failed shots — they are penalized via missingness imputation downstream.

Saves YOLO overlay images to `data/recent/cv/audit_frames/<shot_id>.jpg` for the first 50 shots.

### 2.6 Artifacts Produced

| Path | Description |
|---|---|
| `data/recent/shots.parquet` | Normalized recent shot records with `game_event_id` |
| `data/recent/source_shots.csv` | Raw `ShotChartDetail` API response |
| `data/recent/source.json` | Provenance: season, retrieval timestamp, SHA-256, row/game counts, `possible_row_cap` flag (true if ≥ 100k rows returned — may be truncated) |
| `data/recent/manifest.csv` | Pre-registration manifest, updated in place by annotation tool |
| `data/recent/manifest.selection.json` | Seed, game/shot counts, selection method |
| `data/recent/videos/<shot_id>.mp4` | Downloaded MP4 clips (`:` → `_` in filename) |
| `data/recent/videos/retrieval.jsonl` | Per-shot download log: URL, SHA-256, status, error |
| `data/recent/cv/cv_features.parquet` | Columns: `shot_id`, `elbow_angle_deg`, `wrist_height_torso`, `defender_distance_torso`, `defender_wrist_height_torso`, `cv_status` (`ok`/`partial`/`error`), `cv_error`, `latest_frame_used`, `release_frame`, `clip_sha256` |
| `data/recent/cv/audit_frames/<shot_id>.jpg` | YOLO keypoint overlay debug images (first 50 shots) |
| `data/recent/cv/extraction.json` | `created_at`, `manifest_sha256`, `weights`, `weights_sha256`, `device`, `counts` by status, `environment` |

---

## Phase 3: Multi-Agent LLM System

**Status**: Complete  
**Owner**: Kiran, Anuraag  
**Key file**: `src/nba_shots/agents.py`

Implements three controlled LLM experiment conditions using local Ollama inference via CrewAI. All conditions use `qwen3:4b-instruct`, `temperature=0`, `seed=42`, `num_predict=192`, `num_ctx=8192`.

### 3.1 Feature Allowlist

The function `prompt_features` (agents.py:49) sends **only** these features to the LLM (from `ALLOWED_FEATURES` in common.py:50):

**Tabular**: `shot_distance`, `defender_distance`, `game_clock_seconds`, `shot_clock`, `period`, `dribbles`, `touch_time`, `shot_value`, `location`  
**History**: `player_prior_attempts`, `player_prior_make_rate`  
**CV**: `elbow_angle_deg`, `wrist_height_torso`, `defender_distance_torso`, `defender_wrist_height_torso`

The following are **never sent**: `shot_id`, `game_id`, `game_date`, `player_id`, `cohort`, `split`, `y_true`, `clip_path`, `release_frame`, any box or status columns. `NaN` values are serialized as JSON `null`.

### 3.2 Required Output Schema

Every LLM call is constrained via Ollama's `"format"` parameter to return:

```json
{
  "p_make": 0.42,
  "rationale": "brief evidence summary"
}
```

`parse_assessment` (agents.py:63) validates:
- `p_make` is a finite float, not a bool, in `[0.0, 1.0]`
- `rationale` is a non-null string
- Handles `"Final Answer: {...}"` prefix and ` ```json...``` ` wrapping from CrewAI

### 3.3 Three Ablation Conditions (`predict_condition`, agents.py:224)

All three conditions use exactly the same inference budget to allow fair comparison:

| Condition | Ollama Calls | Architecture |
|---|---|---|
| `llm_single` | 1 | Single `"Shot probability analyst"` agent; all allowlisted features; outputs `p_make` directly |
| `llm_single_budget` | 4 | Same single agent re-prompted 4 times; each call receives the full feature dict plus all prior `{p_make, rationale}` outputs in `previous_assessments`; final call's result is the prediction |
| `llm_multi` | 4 | 3 specialists + 1 aggregator via CrewAI |

**`llm_multi` specialist decomposition** — each specialist receives only their assigned feature subset:

| Agent Role | Features Received |
|---|---|
| `"Game context analyst"` | `game_clock_seconds`, `location`, `period`, `shot_value` (all features minus mechanics and history) |
| `"Player history analyst"` | `player_prior_attempts`, `player_prior_make_rate` |
| `"Shot mechanics and defense analyst"` | `elbow_angle_deg`, `wrist_height_torso`, `defender_distance_torso`, `defender_wrist_height_torso`, `shot_distance`, `defender_distance`, `dribbles`, `touch_time` |
| `"Shot probability aggregator"` | All allowlisted features + `previous_assessments` list of 3 specialist `{role, p_make, rationale}` dicts |

### 3.4 CrewAI Integration (`crewai_assessment`, agents.py:146)

Each agent is constructed with:
- `max_iter=1`, `max_retry_limit=0` — one call maximum, no retries
- `allow_delegation=False`, `tools=[]` — no tool use or inter-agent delegation
- `reasoning=False`, `verbose=False` — no chain-of-thought overhead
- `memory=False`, `cache=False`, `planning=False` on the `Crew` — no hidden state

`OneCallLocalLLM` (agents.py:162) wraps `OllamaClient` and raises `RuntimeError("One-call budget exhausted")` if called more than once per agent instantiation.

Platform telemetry is disabled via environment variables: `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`. CrewAI's platform auth client is patched to raise `AuthError` so no credentials are read or written.

### 3.5 `OllamaClient` Safety (`OllamaClient.__init__`, agents.py:84)

- Accepts only `http://127.0.0.1:11434`, `http://localhost:11434`, or `http://[::1]:11434`
- Rejects any URL with username, password, query string, or fragment
- Rejects `https://` or any non-localhost hostname — raises `ValueError("Only a local Ollama HTTP endpoint is allowed")`
- `session.trust_env = False` — ignores proxy env vars
- `num_predict` must be in `[32, 1024]`

---

## Phase 4: Video Annotation & Inference Run

**Status**: In Progress  
**Owner**: All Team

This is the active execution phase. All infrastructure is built — the remaining work is annotating clips, running extraction, training the CV model, and running LLM inference.

### 4.1 Annotate ≥ 50 Video Clips

Run the annotation tool (see Phase 2.4 for full instructions):

```bash
uv run nba-shots annotate \
  --manifest data/recent/manifest.csv \
  --videos data/recent/videos
```

The sidebar shows "N of M shots annotated (X%)". A shot counts as annotated when both `release_frame` and `shooter_box` are non-empty in `manifest.csv`. Continue until ≥ 50 shots are done.

**Common annotation mistakes to avoid**:
- Do not mark the frame where the ball is already in flight — mark the exact release instant
- Do not annotate the same person as both shooter and defender — `extract-cv` will raise `ValueError("Shooter and defender cannot be the same detection")`
- If YOLO misses a player, try a nearby frame, re-run YOLO, then return to the correct frame

### 4.2 Prepare the Matched Video Cohort

Register the annotated shots as a new cohort with chronological splits:

```bash
uv run nba-shots prepare-cohort \
  --shots data/recent/shots.parquet \
  --manifest data/recent/manifest.csv \
  --output data/recent/cohort \
  --config configs/default.json
```

`vision.prepare_cohort` (vision.py:151):
1. Verifies all `manifest.shot_id` values exist in `shots.parquet`
2. Verifies manifest `game_id` values match source records
3. Optionally adds player history if `include_player_history: true` in config
4. Calls `assign_splits` on the cohort subset — locks in chronological train/val/test assignments
5. Writes `data/recent/cohort/shots.parquet`, `splits.json`, `config.json`, `audit.json`

Raises `ValueError` if cohort directory already exists and is non-empty — use a new path if re-running.

### 4.3 Extract CV Features

```bash
uv run nba-shots extract-cv \
  --manifest data/recent/manifest.csv \
  --output data/recent/cv \
  --weights yolo11n-pose.pt \
  --device cpu
```

See Phase 2.5 for full details. Key constraint: **all pre-registered shots must appear in `cv_features.parquet`** — shots with missing clips or failed YOLO detection are retained with `cv_status="error"` and `NaN` CV features, not dropped.

`train_baselines` (models.py:128) enforces `set(shots.shot_id) == set(cv.shot_id)` — a mismatch raises `ValueError("CV table must cover the entire registered cohort")`.

### 4.4 Train XGBoost + CV Baseline

```bash
uv run nba-shots train \
  --shots data/recent/cohort/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/cv_run1 \
  --config configs/default.json
```

With `--cv` provided, `train_baselines` trains 4 models: `constant`, `logistic_regression`, `xgboost` (tabular only), and `xgboost_cv` (tabular + 4 CV features). The CV features are merged via a left join on `shot_id` using `validate="one_to_one"`.

The `xgboost_cv` feature set is: all 8 numeric tabular features + `location` + `elbow_angle_deg`, `wrist_height_torso`, `defender_distance_torso`, `defender_wrist_height_torso` (NaN values imputed by median).

### 4.5 Run LLM Inference

Ensure Ollama is running (`ollama serve` or the Ollama app), then:

```bash
uv run nba-shots predict-llm \
  --shots data/recent/cohort/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/agent_eval \
  --model qwen3:4b-instruct \
  --split validation \
  --limit 100 \
  --seed 42 \
  --num-predict 192 \
  --ollama-url http://127.0.0.1:11434
```

`run_agents` (agents.py:260):
1. Selects `limit` shots from the validation split by hash-seeded ranking of `shot_id` (no outcome or feature-quality bias) — `select_evaluation_rows` (agents.py:249)
2. Creates `experiment.json` with full spec including `shots_sha256`, `cv_sha256`, `model_digest`, `prompt_version`, `shot_ids` list, `seed`, `num_predict_per_call`, `conditions`
3. If `experiment.json` already exists, checks that the spec is identical — raises `ValueError("Experiment changed; use a new output directory")` on any diff
4. For each shot × condition:
   - Computes cache key: `SHA-256(json({spec, features, condition}))`
   - Hits `artifacts/agent_eval/cache/<key>.json` if it exists (skips Ollama call)
   - Otherwise calls `predict_condition` → `crewai_assessment`
   - Verifies actual Ollama calls == `CONDITIONS[condition]` (1 or 4) — mismatch raises `ValueError`
   - Writes cache and appends to `artifacts/agent_eval/traces.jsonl`
   - Persists `predictions.csv` after every shot/condition (resumable on interruption)

**Resuming an interrupted run**: Re-run the same command with the same `--output`. The spec check ensures the run is identical; cached predictions are replayed without calling Ollama.

### 4.6 Artifacts Produced

| Path | Description |
|---|---|
| `data/recent/cohort/shots.parquet` | Cohort shots with `split` column |
| `data/recent/cohort/splits.json` | Chronological split assignments |
| `data/recent/cohort/audit.json` | `registered_shots`, games, note |
| `artifacts/cv_run1/validation_predictions.csv` | `shot_id`, `game_id`, `cohort`, `split`, `model`, `y_true`, `p_make`, `status` for all 4 models |
| `artifacts/cv_run1/validation_metrics.json` | Brier, log loss, ROC-AUC, accuracy per model |
| `artifacts/cv_run1/xgboost_cv.joblib` | Serialized XGBoost + CV pipeline |
| `artifacts/cv_run1/run.json` | Full provenance metadata |
| `artifacts/agent_eval/experiment.json` | Full experiment spec + environment + started_at |
| `artifacts/agent_eval/selection.csv` | `shot_id`, `game_id`, `cohort`, `split` for evaluated shots |
| `artifacts/agent_eval/predictions.csv` | `shot_id`, `game_id`, `cohort`, `split`, `model` (condition name), `y_true`, `p_make`, `status`, `error`, `cached`, `calls`, `prompt_tokens`, `completion_tokens`, `runtime_seconds` |
| `artifacts/agent_eval/traces.jsonl` | Full per-shot call logs with messages, raw responses, token counts |
| `artifacts/agent_eval/cache/<sha256>.json` | `{p_make, rationale}` per shot/condition |
| `artifacts/agent_eval/validation_metrics.json` | Brier, log loss, ROC-AUC per LLM condition |

---

## Phase 5: Hypothesis Testing & Final Report

**Status**: Upcoming  
**Owner**: All Team  
**Key files**: `src/nba_shots/evaluation.py`, `src/nba_shots/report.py`

### 5.1 Paired Game-Clustered Bootstrap (`evaluation.paired_comparison`, evaluation.py:120)

Compares two models on the same shot cohort using 1,000 bootstrap resamples of whole games. For each game g, computes:

```
Δᵢ = (p_candidate − y)² − (p_baseline − y)²
```

Games are resampled (not shots) to preserve within-game correlation. The 95% CI `[Δ₀.₀₂₅, Δ₀.₉₇₅]` is computed over the weighted mean. **Negative Δ means candidate outperforms baseline.**

Run all three planned comparisons:

```bash
# 1. Does xgboost_cv beat xgboost? (CV enhancement)
uv run nba-shots compare \
  --predictions artifacts/cv_run1/validation_predictions.csv \
  --baseline xgboost \
  --candidate xgboost_cv \
  --output artifacts/compare_xgboost_cv.json \
  --repetitions 1000 \
  --seed 42

# 2. Does llm_multi beat llm_single? (multi-agent vs single-agent)
uv run nba-shots compare \
  --predictions artifacts/agent_eval/predictions.csv \
  --baseline llm_single \
  --candidate llm_multi \
  --output artifacts/compare_llm_multi_vs_single.json

# 3. Does llm_multi beat llm_single_budget? (architecture vs call budget — the critical test)
uv run nba-shots compare \
  --predictions artifacts/agent_eval/predictions.csv \
  --baseline llm_single_budget \
  --candidate llm_multi \
  --output artifacts/compare_llm_multi_vs_budget.json
```

`paired_comparison` requires:
- Both models have identical `shot_id` sets — raises `ValueError` if they differ
- Both models have `status="ok"` for all shots — raises `ValueError` if any failures
- At least 2 held-out games — raises `ValueError` otherwise
- Exactly 1 cohort and 1 split in the input predictions

Output JSON fields: `baseline`, `candidate`, `n_shots`, `n_games`, `brier_delta_candidate_minus_baseline`, `brier_delta_95pct_interval: [lo, hi]`, `method`, `repetitions`, `seed`.

**Interpretation**:
- CI entirely below 0 → candidate is significantly better
- CI spanning 0 → no statistically significant difference — do not claim an improvement
- CI entirely above 0 → candidate is significantly worse

### 5.2 Failure Case Analysis

Find shots where `xgboost` and `llm_multi` predictions diverge most:

1. Load `artifacts/cv_run1/validation_predictions.csv` and `artifacts/agent_eval/predictions.csv`
2. Pivot to wide format on `model` and compute `|p_xgboost − p_llm_multi|`
3. Sort descending — inspect the top divergent shots
4. Re-open the annotation tool to visually review the release frame for those `shot_id`s
5. Check `artifacts/agent_eval/traces.jsonl` for the LLM's rationale on those shots
6. Document systematic failure patterns — e.g.:
   - LLM overconfident on corner 3s (few CV features)
   - CV `NaN`-heavy shots (no defender annotated)
   - XGBoost outliers on unusual shot clock values

Target: document ≥ 5 representative divergent shots with analysis.

### 5.3 Generate Final Summary Report

Combine all prediction files into a Markdown report with reliability diagrams and all paired comparisons:

```bash
uv run nba-shots report \
  --predictions \
    artifacts/cv_run1/validation_predictions.csv \
    artifacts/agent_eval/predictions.csv \
  --output reports/summary_run1
```

`report.create_report` (report.py:11) validates that all models have the same shot IDs within each cohort/split group (raises `ValueError("Models have different evaluation shots")` if not). Use `--selection` to pass a CSV of `cohort`, `split`, `shot_id` rows to restrict to a common evaluation set across all model files.

Automatically runs `paired_comparison` for these pairs if both are present in the predictions: `(xgboost, xgboost_cv)`, `(llm_single, llm_multi)`, `(llm_single_budget, llm_multi)`.

### 5.4 Test Set Evaluation (Final — Run Only Once)

**Do not run until all of the following are frozen: feature set, XGBoost hyperparameters, LLM prompts (`PROMPT_VERSION` in agents.py:29), CV weights, annotation methodology.**

```bash
# Re-evaluate saved XGBoost/CV models on test split
uv run nba-shots evaluate \
  --run artifacts/cv_run1 \
  --shots data/recent/cohort/shots.parquet \
  --split test \
  --final

# Re-run LLM inference on test split
uv run nba-shots predict-llm \
  --shots data/recent/cohort/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/agent_eval_test \
  --split test \
  --final \
  --limit 100
```

The `--final` flag is a hard gate in both `evaluate_run` (models.py:189) and `run_agents` (agents.py:273) — omitting it raises `ValueError("Test evaluation requires --final after feature/prompt freeze")`.

`evaluate_run` also verifies `SHA-256(shots_path) == metadata["shots_sha256"]` from `run.json` — the prepared data must be unchanged since training.

### 5.5 Artifacts Produced

| Path | Description |
|---|---|
| `artifacts/compare_xgboost_cv.json` | Bootstrap CI: `brier_delta_95pct_interval`, `n_shots`, `n_games` |
| `artifacts/compare_llm_multi_vs_single.json` | Same for llm_multi vs llm_single |
| `artifacts/compare_llm_multi_vs_budget.json` | Same for llm_multi vs llm_single_budget |
| `reports/summary_run1/report.md` | Full result table + paired comparison bullets |
| `reports/summary_run1/predictions.csv` | Combined predictions used for report |
| `reports/summary_run1/combined_metrics.json` | All model scores |
| `reports/summary_run1/combined_metrics.csv` | Same in CSV |
| `reports/summary_run1/paired_comparisons.json` | All bootstrap results |
| `reports/summary_run1/*_calibration.png` | Per-cohort/split reliability diagrams (10-bin uniform, 160 DPI) |
| `reports/summary_run1/selection_audit.json` | `input_prediction_rows`, `reported_prediction_rows`, `explicit_selection` |

---

## Phase 6: Final Deliverables

**Status**: Upcoming  
**Owner**: All Team

### 6.1 Deliverables Checklist

- [ ] `reports/summary_run1/report.md` with all models across both cohorts
- [ ] Calibration reliability diagrams for every model/cohort/split combination
- [ ] Bootstrap CI JSON for all 3 paired comparisons — reported regardless of significance
- [ ] `reports/summary_run1/paired_comparisons.json` committed to the repo
- [ ] Failure case writeup — ≥ 5 divergent shots with `shot_id`, predicted probabilities, LLM rationale, and annotation screenshot
- [ ] Final paper: methods, results table, limitations section, reproducibility statement
- [ ] Slide deck for project demo
- [ ] All feature branches merged to `main`; `pytest -q` and `ruff check` passing clean on `main`

### 6.2 Mandatory Interpretation Constraints

These are enforced by `report.py` and stated in the PRD:

- Validation results are **development evidence only** — the final claim is the test set result
- Kaggle 2014–15 and recent 2025–26 cohort scores **cannot be directly compared** — the recent cohort lacks `shot_clock`, `defender_distance`, `dribbles`, `touch_time`
- A bootstrap CI spanning 0 means **no statistically significant improvement** — do not claim one
- Coverage < 100% on any model must be explicitly stated — `n_failed` and `coverage` are in every metrics JSON
- CV features are **2D image proxies**, not physical 3D release angles or heights — state this clearly

### 6.3 Reproducibility Record

Every run produces either `run.json` (tabular) or `experiment.json` (LLM) capturing via `common.environment()`:
- `git_commit` (SHA of HEAD)
- `git_dirty` (true if working tree has uncommitted changes)
- `python` version string
- `packages` dict: all installed versions of `nba-shots`, `numpy`, `pandas`, `scikit-learn`, `xgboost`, `ultralytics`, `crewai`, `litellm`, `matplotlib`
- Input file SHA-256 checksums
- `started_at` and `completed_at` ISO timestamps

---

## Quick Reference: Full Pipeline

```bash
# ── Phase 0: Setup ────────────────────────────────────────────────────────────
uv sync --frozen --all-extras
ollama pull qwen3:4b-instruct
brew install libomp  # macOS only

# ── Phase 1: Tabular baselines ────────────────────────────────────────────────
uv run --frozen nba-shots baseline --output artifacts/baseline_run1
# With existing CSV:
uv run --frozen nba-shots baseline --input data/raw/shot_logs.csv --output artifacts/baseline_run1

# ── Phase 2: Vision pipeline ──────────────────────────────────────────────────
uv run nba-shots collect-recent --season 2025-26 --output data/recent
uv run nba-shots make-manifest --shots data/recent/shots.parquet --output data/recent/manifest.csv --games 30 --shots-per-game 35 --seed 42
uv run nba-shots fetch-videos --manifest data/recent/manifest.csv --output data/recent/videos --limit 50
uv run nba-shots annotate --manifest data/recent/manifest.csv --videos data/recent/videos

# ── Phase 4: CV model + LLM inference ────────────────────────────────────────
uv run nba-shots prepare-cohort --shots data/recent/shots.parquet --manifest data/recent/manifest.csv --output data/recent/cohort --config configs/default.json
uv run nba-shots extract-cv --manifest data/recent/manifest.csv --output data/recent/cv
uv run nba-shots train --shots data/recent/cohort/shots.parquet --cv data/recent/cv/cv_features.parquet --output artifacts/cv_run1
uv run nba-shots predict-llm --shots data/recent/cohort/shots.parquet --cv data/recent/cv/cv_features.parquet --output artifacts/agent_eval --limit 100 --model qwen3:4b-instruct

# ── Phase 5: Hypothesis testing + report ─────────────────────────────────────
uv run nba-shots compare --predictions artifacts/cv_run1/validation_predictions.csv --baseline xgboost --candidate xgboost_cv --output artifacts/compare_xgboost_cv.json
uv run nba-shots compare --predictions artifacts/agent_eval/predictions.csv --baseline llm_single --candidate llm_multi --output artifacts/compare_llm_multi_vs_single.json
uv run nba-shots compare --predictions artifacts/agent_eval/predictions.csv --baseline llm_single_budget --candidate llm_multi --output artifacts/compare_llm_multi_vs_budget.json
uv run nba-shots report --predictions artifacts/cv_run1/validation_predictions.csv artifacts/agent_eval/predictions.csv --output reports/summary_run1

# ── Phase 5: Final test evaluation (ONLY after full freeze) ──────────────────
uv run nba-shots evaluate --run artifacts/cv_run1 --shots data/recent/cohort/shots.parquet --split test --final
uv run nba-shots predict-llm --shots data/recent/cohort/shots.parquet --cv data/recent/cv/cv_features.parquet --output artifacts/agent_eval_test --split test --final --limit 100
```
