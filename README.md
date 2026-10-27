# NBA Shot Prediction

CMSC473 project comparing tabular shot prediction, release-time video features, and
single versus multi-agent LLMs using free tools.

Tabular baselines are working. CV and agent pipelines are implemented; real-video
annotations and Qwen experiments are pending. See [initial results](reports/day1.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.11.
On macOS, install XGBoost's runtime with `brew install libomp`.

```bash
uv sync --frozen
uv run --frozen nba-shots baseline --output artifacts/baseline
```

Downloads the [Kaggle shot logs](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs),
cleans the data, creates chronological game splits, and trains constant, logistic
regression, and XGBoost models. Evaluates validation data only.

Use a new output directory for each run. To supply a downloaded CSV, add
`--input /path/to/shot_logs.csv`. Runs save prepared data, splits, models, predictions,
metrics, and calibration plots. Data and generated artifacts are ignored by Git.

## Experiments

Configure features and training in `configs/default.json`. Splits keep whole games
and dates together; preprocessing uses training data only and excludes outcome fields.
Reserve test evaluation (`--split test --final`) until configurations and prompts are frozen.

Install optional CV and agent dependencies:

```bash
uv sync --frozen --all-extras
.venv/bin/nba-shots --help
```

### Video Annotation Tool

Use the interactive browser tool to scrub video clips, set release frames (`R`), run YOLO person detection, and click to tag shooter and contest defender:

```bash
uv run nba-shots annotate --manifest data/recent/manifest.csv --videos data/recent/videos
```

See [experiment instructions](docs/experiments.md) for video collection, release
annotations, pose extraction, local LLMs, and comparisons on matching shots.

## Checks

```bash
uv run --frozen pytest -q
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

Use `uv run --frozen --all-extras pytest -q` to include the CrewAI integration tests.

## Resources

- [Baseline results](reports/day1.md)
- [Demo notebook](notebooks/demo.ipynb)
- [Team roadmap](docs/roadmap.md)
