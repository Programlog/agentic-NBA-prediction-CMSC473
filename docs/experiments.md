# Experiment Execution Guide

Run commands from the repository root with Python 3.11 or 3.12. Use new output
folders for each experiment; source snapshots, frozen splits, and completed runs
must not be overwritten. Generated data and artifacts are ignored by Git.

## 1. Environment

```bash
uv sync --frozen --all-extras
# macOS: required by XGBoost
brew install libomp
```

Vision dependencies include OpenCV and Ultralytics. The first YOLO invocation may
download `yolo11n-pose.pt`; use the same weights for annotation and extraction.

## 2. Kaggle tabular baseline

```bash
uv run --frozen nba-shots baseline --output artifacts/baseline_run1
# To use a local CSV instead, add: --input path/to/shot_logs.csv
```

This saves prepared data and chronological splits under `prepared/`, and fitted
models, validation predictions, metrics, and calibration plots under `models/`.
Only validation data is evaluated. The default configuration excludes player
history; enable it deliberately and consistently if it is part of the experiment.

Generate a baseline-only report:

```bash
uv run --frozen nba-shots report \
  --predictions artifacts/baseline_run1/models/validation_predictions.csv \
  --output reports/baseline_run1
```

The report is `reports/baseline_run1/report.md`. Kaggle results are a separate
benchmark; they cannot establish the effect of adding CV to a recent-season cohort.

## 3. Register the video cohort before annotation

The paths below match the existing local layout. If a source snapshot, manifest,
or prepared cohort already exists, reuse it and skip its creation command. For a
new experiment, choose new paths consistently throughout this guide.

### Collect source shot records

```bash
uv run --frozen nba-shots collect-recent \
  --season 2025-26 --output data/raw/nba_2025_26
```

Inspect `source.json`, date coverage, game count, and `possible_row_cap` before
selecting a cohort. A source snapshot is not confirmation of full-season coverage.
The recent source lacks shot clock, defender distance, dribbles, and touch time;
these remain missing rather than being invented.

### Select shots without selecting on outcomes

```bash
uv run --frozen nba-shots make-manifest \
  --shots data/raw/nba_2025_26/shots.parquet \
  --output data/video/manifest.csv \
  --games 30 --shots-per-game 35
```

Keep the selection audit and registered shot IDs, including unavailable clips and
failed pose extractions. Do not replace failures with convenient made/missed shots.

### Freeze the cohort and chronological splits

```bash
uv run --frozen nba-shots prepare-cohort \
  --shots data/raw/nba_2025_26/shots.parquet \
  --manifest data/video/manifest.csv \
  --output data/processed/video_cohort
```

Use `data/processed/video_cohort/shots.parquet` for training and LLM inference.
The raw source table has no split assignments and is rejected by those commands.
Review `splits.json`: small held-out game counts limit statistical conclusions.
If enabling player history, supply the chosen config here with `--config`; training
loads the saved cohort config automatically. Freeze this choice before evaluation.

## 4. Retrieve, annotate, and extract

### Attempt retrieval

```bash
uv run --frozen nba-shots fetch-videos \
  --manifest data/video/manifest.csv \
  --output data/video/clips --limit 50
```

`--limit` processes the first N manifest rows; repeating the same limit retries
those rows rather than advancing to another batch. Inspect `retrieval.jsonl` and
manually verify every downloaded clip against its game/event. A valid MP4 may be a
placeholder. Empty video URLs are retrieval failures, not usable footage.

For manually indexed clips, record a verified `clip_path` in the manifest,
relative to the manifest's folder or absolute. Alternatively, place MP4s in the
clips folder named by shot ID with colons replaced by underscores, for example
`nba_0022500004_163.mp4`. Keep the shot IDs and retrieval evidence unchanged.

### Annotate exact release frames

```bash
uv run --frozen nba-shots annotate \
  --manifest data/video/manifest.csv --videos data/video/clips
```

Playback is for finding the release region. Pause and use the frame controls to
inspect the exact server-decoded frame, then press **R** to mark it and run YOLO.
Click the detected shooter box and, when visible, the contest defender box; choose
the shooting hand and press **S** to save. Frame indices are zero-based. The tool
uses clip metadata rather than assuming 30 FPS; the decoded still is the authority
for frame selection, including clips whose browser playback timing differs.

Saving also persists the resolved clip path relative to the manifest, so extraction
can find automatically discovered clips. Reopen a saved annotation and verify its
frame and boxes before starting a batch. Changing frames clears boxes to prevent
assignments from a different frame being saved. Keep annotation to release-time
measurements; do not enter outcome or post-release trajectory information.

### Extract and inspect pose features

```bash
uv run --frozen nba-shots extract-cv \
  --manifest data/video/manifest.csv --output data/video/cv_run1
```

Inspect `extraction.json`, `cv_features.parquet`, and `audit_frames/`. Confirm the
correct frame and player identities, and report complete/partial/failed coverage.
Rows with missing clips, annotations, or keypoints remain in the feature table.
The first 50 manifest rows are eligible for saved overlays when extraction succeeds.
Do not interpret an all-missing CV table as a completed vision experiment. These
features are 2D pose proxies, not physical release angles, heights, or ball spin.

## 5. Train matched tabular and CV models

After the pilot confirms usable features, train both conditions in one run:

```bash
uv run --frozen nba-shots train \
  --shots data/processed/video_cohort/shots.parquet \
  --cv data/video/cv_run1/cv_features.parquet \
  --output artifacts/video_run1/models
```

This fits constant, logistic regression, `xgboost`, and `xgboost_cv` models on the
same training shots and scores the same validation shots. The CV table must cover
every registered shot, including failures. Inspect coverage and calibration plots.

Compare the two XGBoost conditions using this run's predictions:

```bash
uv run --frozen nba-shots compare \
  --predictions artifacts/video_run1/models/validation_predictions.csv \
  --baseline xgboost --candidate xgboost_cv \
  --output artifacts/video_run1/comparison_xgboost_cv.json
```

Negative Brier difference favors CV. An interval spanning zero does not establish
improvement. Report the held-out game count as well as the shot count.

## 6. Run local LLM conditions

Start Ollama locally and install the exact configured model:

```bash
ollama pull qwen3:4b-instruct
ollama list
```

Run one-call single-agent, four-call single-agent, and four-call multi-agent
conditions on a deterministic selection of validation shots:

```bash
uv run --frozen nba-shots predict-llm \
  --shots data/processed/video_cohort/shots.parquet \
  --cv data/video/cv_run1/cv_features.parquet \
  --output artifacts/agent_eval_run1 --limit 100 \
  --model qwen3:4b-instruct
```

Start with a small `--limit` to verify actual inference, probabilities, call counts,
and failure coverage. For a tabular-only pilot, omit `--cv` and use a separate output
directory. Prepared Kaggle data also supports a separate tabular pilot. Agent tests
with deterministic fake responses are not experimental LLM results.

Preserve `experiment.json`, `selection.csv`, caches, traces, predictions, and metrics.
Different models, inputs, prompts, or inference settings require a new directory.

## 7. Report matching evaluation shots

Because the LLM run may select fewer validation shots, explicitly align model
predictions to its saved selection before comparing all conditions:

```bash
uv run --frozen nba-shots report \
  --predictions \
    artifacts/video_run1/models/validation_predictions.csv \
    artifacts/agent_eval_run1/predictions.csv \
  --selection artifacts/agent_eval_run1/selection.csv \
  --output reports/video_agent_run1
```

Read `report.md`, `paired_comparisons.json`, and `selection_audit.json`. Compare the
budget-matched single-agent and multi-agent conditions as well as one-call versus
multi-agent. Failed predictions must be resolved or reported; paired comparisons
reject incomplete coverage. Interpret tabular-versus-agent results only with their
supplied feature sets documented (the default tabular config excludes player history).

## 8. Final evaluation

Keep test sets unused during development. Freeze configurations, features, prompts,
model identity, and evaluation selection before explicitly running test evaluation
with `--split test --final`. Save test results separately from validation findings.
