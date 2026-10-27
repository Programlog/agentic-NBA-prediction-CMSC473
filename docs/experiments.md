# Experiment Execution Guide

This guide provides step-by-step instructions to reproduce all baseline and multimodal experiments for the CMSC473 NBA shot prediction project.

---

## 1. Environment Setup

Prerequisites: Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone and enter the repository
git clone https://github.com/Programlog/agentic-NBA-prediction-CMSC473.git
cd agentic-NBA-prediction-CMSC473

# Synchronize dependencies with optional vision and agent packages
uv sync --frozen --all-extras

# macOS OpenMP runtime for XGBoost
brew install libomp
```

---

## 2. Tabular Baselines (Kaggle 2014–15)

To run the end-to-end tabular baselines (download dataset, audit, chronological split, train models, score validation):

```bash
uv run nba-shots baseline --output artifacts/baseline_run1
```

To provide an existing local CSV instead of downloading:

```bash
uv run nba-shots baseline --input path/to/shot_logs.csv --output artifacts/baseline_run1
```

Generated artifacts:
- `artifacts/baseline_run1/prepared/shots.parquet`: Audited and chronologically partitioned shot records.
- `artifacts/baseline_run1/prepared/audit.json`: Data provenance and missing value audit.
- `artifacts/baseline_run1/prepared/splits.json`: Game and date split distribution.
- `artifacts/baseline_run1/models/validation_metrics.json`: Brier score, log loss, ROC-AUC, and accuracy.
- `artifacts/baseline_run1/models/validation_predictions.csv`: Model probability outputs.
- `artifacts/baseline_run1/models/validation_kaggle_2014_15_validation_calibration.png`: Calibration curve.

---

## 3. Video Collection & Computer Vision Pipeline

### Step 3.1: Cache Recent Season Shot Records
```bash
uv run nba-shots collect-recent --season 2025-26 --output data/recent
```

### Step 3.2: Generate Hash-Seeded Annotation Manifest
Preregisters games and shots without outcome selection bias:
```bash
uv run nba-shots make-manifest \
  --shots data/recent/shots.parquet \
  --output data/recent/manifest.csv \
  --games 30 \
  --shots-per-game 35
```

### Step 3.3: Retrieve Video Clips
Fetches MP4 video clips via the official NBA `VideoEventsAsset` endpoint:
```bash
uv run nba-shots fetch-videos \
  --manifest data/recent/manifest.csv \
  --output data/recent/videos \
  --limit 50
```

### Step 3.4: Interactive Video Annotation Tool
Launch the interactive web UI to scrub clips, set release frames (`R`), run YOLO detection, and click-assign shooter and defender boxes:
```bash
uv run nba-shots annotate \
  --manifest data/recent/manifest.csv \
  --videos data/recent/videos
```

### Step 3.5: Extract 2D Pose Features
Extracts torso-normalized release geometry using YOLO-Pose:
```bash
uv run nba-shots extract-cv \
  --manifest data/recent/manifest.csv \
  --output data/recent/cv
```

---

## 4. Multi-Agent LLM Experiments (Local Ollama)

Ensure [Ollama](https://ollama.com/) is running locally:
```bash
# Pull the target open-source model
ollama pull qwen3:4b-instruct

# Run the controlled single vs multi-agent comparison
uv run nba-shots predict-llm \
  --shots data/recent/shots.parquet \
  --cv data/recent/cv/cv_features.parquet \
  --output artifacts/agent_eval \
  --limit 100 \
  --model qwen3:4b-instruct
```

---

## 5. Statistical Hypothesis Testing & Reporting

Compare candidate models against the XGBoost baseline using paired game-clustered bootstrap (1,000 resamples):

```bash
uv run nba-shots compare \
  --predictions artifacts/baseline_run1/models/validation_predictions.csv \
  --baseline xgboost \
  --candidate xgboost_cv \
  --output artifacts/comparison_xgboost_cv.json
```

Generate a comprehensive Markdown report summarizing all evaluated models:

```bash
uv run nba-shots report \
  --predictions \
    artifacts/baseline_run1/models/validation_predictions.csv \
    artifacts/agent_eval/predictions.csv \
  --output reports/summary_run1
```
