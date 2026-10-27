# Project Phases: End-to-End Execution Guide

**Course**: CMSC473  
**Team**: Praharsh Nanduri, Kiran Manoj, Anuraag Tripathy, Prannav Chettipalayam, Varun Kota  

This document walks through every phase of the project in execution order — from environment setup through final report — with every command, artifact, and validation step needed to reproduce the full pipeline.

---

## Phase 0: Environment Setup

**Status**: Complete  
**Owner**: All Team

### 0.1 Prerequisites

- Python 3.11 (enforced by `.python-version` and `pyproject.toml`)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- macOS: `brew install libomp` (XGBoost OpenMP runtime)
- [Ollama](https://ollama.com/) installed locally (for Phase 4)

### 0.2 Install Dependencies

```bash
# Core dependencies only (tabular baselines)
uv sync --frozen

# All dependencies including CV (ultralytics, opencv, nba_api) and agents (crewai, litellm)
uv sync --frozen --all-extras
```

### 0.3 Pull the LLM Model

```bash
ollama pull qwen3:4b-instruct
```

### 0.4 Verify Installation

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv run --frozen --all-extras pytest -q   # includes CrewAI integration tests
```

### Artifacts Produced
None — environment only.

---

## Phase 1: Tabular Data Ingestion & Baselines

**Status**: Complete  
**Owner**: Varun, Praharsh

This phase downloads the Kaggle 2014–15 shot logs (~128,000 attempts), audits and normalizes the data, applies strict chronological splitting, and trains three baseline models.

### 1.1 One-Command Baseline Run

```bash
uv run --frozen nba-shots baseline --output artifacts/baseline_run1
```

This single command:
1. Downloads `shot_logs.csv` from Kaggle to `data/raw/shot_logs.csv`
2. Normalizes and audits the CSV (`normalize_kaggle` in `src/nba_shots/data.py`)
3. Assigns chronological whole-date splits (70% train / 15% validation / 15% test)
4. Fits Constant, Logistic Regression, and XGBoost models on training data
5. Scores the validation split only

To supply a previously downloaded CSV:

```bash
uv run --frozen nba-shots baseline \
  --input path/to/shot_logs.csv \
  --output artifacts/baseline_run1
```

To customize features or XGBoost hyperparameters, edit `configs/default.json` and pass `--config configs/default.json`.

### 1.2 What the Normalization Does

- Drops outcome-leaking columns: `FGM`, `PTS`, `W`, `FINAL_MARGIN`
- Maps `SHOT_RESULT` → binary `y_true` (made=1, missed=0)
- Parses `MATCHUP` → ISO `game_date`
- Clamps invalid numeric values to `NaN` (imputed downstream)
- Constructs `shot_id = "kaggle:<game_id>:<player_id>:<shot_number>"`

### 1.3 Chronological Splitting Invariant

Splits are assigned at the whole-date level; no game spans partition boundaries:

```
max(train dates) < min(validation dates) < min(test dates)
```

### 1.4 Optional: Player History Features

To add Laplace-smoothed career prior make rate (Beta(1,1) prior), set `"include_player_history": true` in `configs/default.json`. History uses only games on strictly earlier dates — never the same day.

### 1.5 Artifacts Produced

| Path | Description |
|---|---|
| `artifacts/baseline_run1/prepared/shots.parquet` | Normalized, split-tagged shot records |
| `artifacts/baseline_run1/prepared/audit.json` | Row counts, missing values, make rate |
| `artifacts/baseline_run1/prepared/splits.json` | Game/date distribution per split |
| `artifacts/baseline_run1/models/validation_predictions.csv` | Model probability outputs |
| `artifacts/baseline_run1/models/validation_metrics.json` | Brier, log loss, ROC-AUC, accuracy |
| `artifacts/baseline_run1/models/validation_kaggle_2014_15_validation_calibration.png` | Reliability diagram |
| `artifacts/baseline_run1/models/*.joblib` | Serialized sklearn pipelines |
| `artifacts/baseline_run1/models/run.json` | Environment, config, timing metadata |

### 1.6 Expected Baseline Results

| Model | Brier Score | Notes |
|---|---|---|
| Constant (make-rate prior) | ~0.248 | Ceiling to beat |
| Logistic Regression | ~0.240 | Linear tabular features |
| XGBoost | ~0.237 | Target: < 0.245 |

---

## Phase 2: Video Collection & Computer Vision Pipeline

**Status**: Complete  
**Owner**: Prannav

This phase queries recent NBA shot data via `nba_api`, downloads broadcast video clips, and extracts YOLO-Pose keypoints from the annotated release frame.

### 2.1 Collect Recent Season Shot Records

```bash
uv run nba-shots collect-recent --season 2025-26 --output data/recent
```

Queries `ShotChartDetail` for all FGA attempts in the 2025-26 regular season. Note: `shot_clock`, `defender_distance`, `dribbles`, and `touch_time` are unavailable in the recent API — those columns are `NaN` for this cohort.

### 2.2 Generate Hash-Seeded Annotation Manifest

```bash
uv run nba-shots make-manifest \
  --shots data/recent/shots.parquet \
  --output data/recent/manifest.csv \
  --games 30 \
  --shots-per-game 35
```

Games and shots are selected by SHA-256 hash of `seed:value` — no outcome or quality bias. The manifest pre-registers `shot_id` before any video is retrieved, locking in the evaluation cohort.

### 2.3 Fetch Video Clips

```bash
uv run nba-shots fetch-videos \
  --manifest data/recent/manifest.csv \
  --output data/recent/videos \
  --limit 50
```

Downloads MP4 clips from `VideoEventsAsset`. Each clip is written atomically (`.part` → `.mp4`). A `retrieval.jsonl` log records URL, SHA-256, and status per shot. Clips are capped at 100 MiB; a 0.6 s throttle respects the NBA API rate limit.

### 2.4 Interactive Video Annotation Tool

```bash
uv run nba-shots annotate \
  --manifest data/recent/manifest.csv \
  --videos data/recent/videos
```

Opens a browser at `http://127.0.0.1:8765`. For each shot:

1. Scrub the video to the exact release instant (ball leaving shooter's fingers)
2. Press `R` to bookmark the release frame — YOLO person detection runs automatically
3. Click a detected bounding box to assign the **Shooter** (blue) or **Defender** (red)
4. Select the shooting hand (right/left)
5. Press `S` to save and advance to the next shot

**Keyboard shortcuts**: `Space` play/pause · `←`/`→` step frames · `R` set release · `S` save

Annotations are written atomically to `manifest.csv` after each save. Target: annotate ≥ 50 clips.

### 2.5 Extract 2D Pose Features

```bash
uv run nba-shots extract-cv \
  --manifest data/recent/manifest.csv \
  --output data/recent/cv
```

Runs YOLO11n-pose on the single annotated release frame per shot. All spatial measurements are normalized by the shooter's torso length (shoulder midpoint → hip midpoint distance) to be scale- and distance-invariant.

| Feature | Definition |
|---|---|
| `elbow_angle_deg` | Angle at shooting elbow (shoulder–elbow–wrist) |
| `wrist_height_torso` | Shooter wrist elevation above shoulders, in torso units |
| `defender_distance_torso` | Hip-to-hip distance between shooter and defender, in torso units |
| `defender_wrist_height_torso` | Defender's highest contesting hand above shooter's shoulder line |

Missing keypoints (confidence < 0.5) or absent defender boxes produce `NaN` — handled by median imputation downstream.

### 2.6 Artifacts Produced

| Path | Description |
|---|---|
| `data/recent/shots.parquet` | Normalized recent shot records |
| `data/recent/source_shots.csv` | Raw NBA API response |
| `data/recent/manifest.csv` | Annotation manifest (updated in place) |
| `data/recent/videos/*.mp4` | Downloaded video clips |
| `data/recent/cv/cv_features.parquet` | Per-shot CV feature table with status |
| `data/recent/cv/audit_frames/*.jpg` | YOLO overlay debug images (first 50 shots) |
| `data/recent/cv/extraction.json` | Counts by status (ok / partial / error) |

---

## Phase 3: Multi-Agent LLM System

**Status**: Complete  
**Owner**: Kiran, Anuraag

This phase implements three controlled LLM experiment conditions using local Ollama inference via CrewAI. All conditions use `qwen3:4b-instruct` at temperature 0 with a fixed seed for determinism.

### 3.1 Architecture

Three conditions isolate the effect of multi-agent orchestration from raw token budget:

| Condition | Total Ollama Calls | Architecture |
|---|---|---|
| `llm_single` | 1 | Single analyst, all features |
| `llm_single_budget` | 4 | Same analyst re-prompted 4× with prior rationales |
| `llm_multi` | 4 | 3 specialist agents + 1 aggregator |

**Multi-agent decomposition** (`llm_multi`):
1. **Game Context Analyst** — period, game clock, location, shot value
2. **Player History Analyst** — prior career attempts, prior make rate
3. **Shot Mechanics & Defense Analyst** — CV pose features, shot distance, defender proximity, dribbles, touch time
4. **Shot Probability Aggregator** — all features + three specialist outputs → final `p_make`

### 3.2 Output Schema

Every LLM call must return:

```json
{ "p_make": 0.42, "rationale": "brief evidence summary" }
```

`p_make` must be a finite float in [0, 1]. Invalid outputs are recorded as errors, not silently dropped.

### 3.3 Safety Controls

- Only `http://127.0.0.1:11434` is accepted — remote endpoints raise `ValueError`
- No player identity, game date, shot ID, or outcome is sent to the LLM
- Each CrewAI agent is budget-capped to exactly 1 call (`max_iter=1`, `max_retry_limit=0`)
- Platform telemetry and external tracing are disabled (`OTEL_SDK_DISABLED=true`)

---

## Phase 4: Video Annotation & Inference Run

**Status**: In Progress  
**Owner**: All Team

### 4.1 Annotate Video Clips

Run the annotation tool (Phase 2.4) until ≥ 50 shots have `release_frame` and `shooter_box` filled in `manifest.csv`. The annotation tool shows progress as "N of M shots annotated (X%)".

### 4.2 Prepare the Matched Cohort

After annotation, register the video cohort with chronological splits:

```bash
uv run nba-shots prepare-cohort \
  --shots data/recent/shots.parquet \
  --manifest data/recent/manifest.csv \
  --output data/recent/cohort \
  --config configs/default.json
```

This locks the cohort split assignments before any model training.

### 4.3 Extract CV Features

```bash
uv run nba-shots extract-cv \
  --manifest data/recent/manifest.csv \
  --output data/recent/cv
```

All pre-registered shots — including those with missing videos or failed pose extraction — are retained in `cv_features.parquet` with `cv_status = "error"` and `NaN` features. Never filter failed shots from the cohort.

### 4.4 Train XGBoost + CV Baseline

```bash
uv run nba-shots train \
  --shots data/recent/cohort/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/cv_run1
```

This trains `constant`, `logistic_regression`, `xgboost`, and `xgboost_cv` (tabular + pose features). CV features augment the XGBoost feature set; the CV model is evaluated only where all shots in the cohort are covered.

### 4.5 Run LLM Inference

Ensure Ollama is running, then:

```bash
uv run nba-shots predict-llm \
  --shots data/recent/cohort/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/agent_eval \
  --limit 100 \
  --model qwen3:4b-instruct
```

Predictions are cached per shot/condition by SHA-256 of the experiment spec and features. If the run is interrupted, re-running resumes from the cache without re-calling Ollama. The call budget per condition is enforced — a mismatch raises an error rather than silently producing extra calls.

### 4.6 Artifacts Produced

| Path | Description |
|---|---|
| `data/recent/cohort/shots.parquet` | Cohort with split assignments |
| `artifacts/cv_run1/validation_predictions.csv` | Tabular + CV model probabilities |
| `artifacts/cv_run1/validation_metrics.json` | Brier, log loss, ROC-AUC per model |
| `artifacts/agent_eval/predictions.csv` | LLM condition probabilities |
| `artifacts/agent_eval/traces.jsonl` | Full per-shot call logs with token counts |
| `artifacts/agent_eval/cache/*.json` | Deterministic per-shot/condition cache |

---

## Phase 5: Hypothesis Testing & Final Report

**Status**: Upcoming  
**Owner**: All Team

### 5.1 Paired Game-Clustered Bootstrap

Compare candidate models against the XGBoost baseline. Each comparison resamples whole games 1,000 times to produce a 95% confidence interval on ΔBrier:

```bash
# CV enhancement: does xgboost_cv beat xgboost?
uv run nba-shots compare \
  --predictions artifacts/cv_run1/validation_predictions.csv \
  --baseline xgboost \
  --candidate xgboost_cv \
  --output artifacts/compare_xgboost_cv.json

# Agent ablation: does llm_multi beat llm_single?
uv run nba-shots compare \
  --predictions artifacts/agent_eval/predictions.csv \
  --baseline llm_single \
  --candidate llm_multi \
  --output artifacts/compare_llm_multi_vs_single.json

# Budget-matched ablation: llm_multi vs llm_single_budget (same 4 calls)
uv run nba-shots compare \
  --predictions artifacts/agent_eval/predictions.csv \
  --baseline llm_single_budget \
  --candidate llm_multi \
  --output artifacts/compare_llm_multi_vs_budget.json
```

**Interpretation**: A 95% CI entirely below zero means the candidate significantly outperforms the baseline. A CI spanning zero means no statistically significant difference.

### 5.2 Permutation Feature Importance

The XGBoost model supports permutation-based Brier feature importance on the validation split. Run this after training to identify which features contribute most:

```bash
uv run nba-shots evaluate \
  --run artifacts/cv_run1 \
  --shots data/recent/cohort/shots.parquet \
  --split validation
```

Review the `run.json` and model internals for feature contribution analysis.

### 5.3 Failure Case Analysis

Inspect shots where models diverge most:

1. Sort predictions by `|p_make_xgboost - p_make_llm_multi|` descending
2. Re-open the annotation tool to review the release frame for outlier shots
3. Check `traces.jsonl` for the LLM rationale on failures
4. Document systematic failure modes (e.g., LLM overconfident on corner 3s, CV NaN-heavy shots)

### 5.4 Generate Final Summary Report

Combine all prediction files into a single Markdown report with reliability diagrams and paired comparisons:

```bash
uv run nba-shots report \
  --predictions \
    artifacts/cv_run1/validation_predictions.csv \
    artifacts/agent_eval/predictions.csv \
  --output reports/summary_run1
```

### 5.5 Test Set Evaluation (Final Only)

**Do not run this until features, prompts, and hyperparameters are fully frozen.**

```bash
uv run nba-shots evaluate \
  --run artifacts/cv_run1 \
  --shots data/recent/cohort/shots.parquet \
  --split test \
  --final
```

The `--final` flag is required to prevent p-hacking. Running this prematurely invalidates the test results.

### 5.6 Artifacts Produced

| Path | Description |
|---|---|
| `artifacts/compare_*.json` | Bootstrap CI results per model pair |
| `reports/summary_run1/report.md` | Full Markdown result table |
| `reports/summary_run1/combined_metrics.json` | All model scores |
| `reports/summary_run1/paired_comparisons.json` | All bootstrap results |
| `reports/summary_run1/*_calibration.png` | Reliability diagrams |

---

## Phase 6: Final Deliverables

**Status**: Upcoming  
**Owner**: All Team

### 6.1 Deliverables Checklist

- [ ] `reports/summary_run1/report.md` — complete result table across all models and cohorts
- [ ] Calibration reliability diagrams for every model/cohort combination
- [ ] Bootstrap CI JSON for all three paired comparisons (CV, single vs multi, budget vs multi)
- [ ] Failure case writeup (≥ 5 representative divergent shots with analysis)
- [ ] Final paper documenting methods, results, and limitations
- [ ] Slide deck for project demo
- [ ] All code merged to `main` and passing `pytest` + `ruff`

### 6.2 Key Interpretation Constraints

Per `report.py` and the PRD:

- Validation results are development evidence — **test results are the final claim**
- Kaggle (2014-15) and recent video cohort scores **cannot be directly compared** — different features are available
- A bootstrap CI spanning zero means **no significant improvement** — do not overclaim
- Coverage < 100% on any model must be explicitly reported and penalized, not silently filtered
- All CV features are 2D image proxies — they are **not** physical release angles or heights

### 6.3 Reproducibility Requirements

Every run produces a `run.json` or `experiment.json` capturing:
- Git commit hash and dirty status
- Python version and all package versions
- SHA-256 checksums of all input data files
- Random seeds and model hyperparameters
- Timestamps for start and completion

---

## Quick Reference: Full Pipeline Commands

```bash
# Phase 1 — Tabular baselines
uv run nba-shots baseline --output artifacts/baseline_run1

# Phase 2 — Vision pipeline
uv run nba-shots collect-recent --season 2025-26 --output data/recent
uv run nba-shots make-manifest --shots data/recent/shots.parquet --output data/recent/manifest.csv
uv run nba-shots fetch-videos --manifest data/recent/manifest.csv --output data/recent/videos --limit 50
uv run nba-shots annotate --manifest data/recent/manifest.csv --videos data/recent/videos

# Phase 4 — Train CV model and run LLM inference
uv run nba-shots prepare-cohort --shots data/recent/shots.parquet --manifest data/recent/manifest.csv --output data/recent/cohort
uv run nba-shots extract-cv --manifest data/recent/manifest.csv --output data/recent/cv
uv run nba-shots train --shots data/recent/cohort/shots.parquet --cv data/recent/cv/cv_features.parquet --output artifacts/cv_run1
uv run nba-shots predict-llm --shots data/recent/cohort/shots.parquet --cv data/recent/cv/cv_features.parquet --output artifacts/agent_eval --limit 100

# Phase 5 — Evaluation and reporting
uv run nba-shots compare --predictions artifacts/cv_run1/validation_predictions.csv --baseline xgboost --candidate xgboost_cv --output artifacts/compare_xgboost_cv.json
uv run nba-shots compare --predictions artifacts/agent_eval/predictions.csv --baseline llm_single --candidate llm_multi --output artifacts/compare_llm_multi_vs_single.json
uv run nba-shots report --predictions artifacts/cv_run1/validation_predictions.csv artifacts/agent_eval/predictions.csv --output reports/summary_run1

# Final test evaluation (only after full freeze)
uv run nba-shots evaluate --run artifacts/cv_run1 --shots data/recent/cohort/shots.parquet --split test --final
```
