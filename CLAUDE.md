# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CMSC473 research project: does adding release-time vision features (YOLO pose or BasketEvent
player/ball boxes) or LLM agents (single vs CrewAI multi-agent, local Ollama) improve NBA shot-make
probability over tabular baselines? Correctness of the *experimental protocol* (no leakage, fair
paired comparisons, test split untouched) matters more than raw model performance.

## Commands

Uses `uv` with Python 3.11–3.12 (`requires-python = ">=3.11,<3.13"`).

```bash
uv sync --frozen                       # core deps
uv sync --frozen --all-extras          # + vision (ultralytics, opencv, nba_api, huggingface_hub) and agents (crewai, pydantic)
uv run --frozen --all-extras python -m pytest -q            # full suite (needs extras)
uv run --frozen --all-extras python -m pytest tests/test_models.py::test_name -q   # single test
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests               # line length 100, rules E/F/I
uv run --frozen nba-shots --help       # CLI (entry point nba_shots.cli:main)
```

Use `python -m pytest`, not bare `pytest`: several tests import `scripts.<name>`, which only
resolves when the repo root is on `sys.path`. Ruff currently reports pre-existing errors in some
H1 tracking modules and tests that came from `main`; check only the files you touched.

If `ModuleNotFoundError: nba_shots` appears intermittently, prefix commands with `PYTHONPATH=src`
(see `docs/troubleshooting.md`; the `.venv` `.pth` file can get a hidden flag).

## Architecture

`src/nba_shots/` is one package behind a single argparse CLI (`cli.py`). Each subcommand maps to a
pipeline stage; optional-dependency modules (`vision`, `annotate`, `basketevent`, `agents`) are
imported lazily inside `execute()` so core commands work without extras. Stages communicate only
through files on disk (parquet/CSV/JSON in a run directory), not in-memory state.

- `common.py` — shared contracts: `DEFAULT_CONFIG`, the feature-set allowlists
  (`TABULAR_FEATURES`, `HISTORY_FEATURES`, `SHOT_CHART_FEATURES`, `CV_FEATURES`, `BOX_FEATURES`,
  `ALLOWED_FEATURES`), `PREDICTION_COLUMNS`, and `load_config`. Config files in `configs/` are
  *partial overrides* of `DEFAULT_CONFIG` (the `xgboost` dict is merged); unknown keys and
  non-allowlisted features are rejected. Adding a new feature means adding it to the right set here.
- `data.py` — Kaggle 2014-15 ingestion: `normalize_kaggle` drops outcome columns (FGM, PTS, W,
  FINAL_MARGIN), `assign_splits` makes chronological whole-date train/validation/test splits,
  `add_player_history` uses only strictly-earlier dates.
- `models.py` — constant / logistic regression / XGBoost pipelines; `train_baselines` (fit train,
  score validation), `cross_validate` (date-blocked folds over train+validation, `--forward` for
  forward-only), `evaluate_run`. An extra feature table (`--cv`) is merged on `shot_id` and yields an
  additional `xgboost_cv`/box model on the same shots; its kind is inferred from its columns.
- `evaluation.py` — Brier/log loss/AUC/accuracy, calibration plots, and `paired_comparison`: a
  game-clustered paired bootstrap of Brier differences (negative = candidate better).
- `report.py` — combines prediction CSVs into `report.md`, aligned to an explicit `--selection`.
- `vision.py` + `annotate.py` — nba_api 2025-26 shots, label-blind manifests, clip download,
  local web annotation tool (ThreadingHTTPServer on 8765), YOLO11-pose features from the single
  release frame only.
- `basketevent.py` — current main line of work (H1). Uses the BasketEvent HF dataset's per-frame
  boxes (not pixels): detects release from shooter/ball boxes, computes `BOX_FEATURES` in units of
  shooter box height (bh), matches clips to NBA play-by-play, splits and writes a cohort. Tunable
  constants (release margin, glitch filter `STEADY_RATIO`, `DEPTH_STRETCH`) live at the module top.
- `agents.py` — Ollama-only LLM conditions `llm_single` (1 call), `llm_single_budget` (4 calls),
  `llm_multi` (CrewAI: 3 specialists + aggregator, 4 calls). Non-localhost URLs are rejected;
  responses are cached by a hash of spec+features+condition in `<output>/cache` so runs resume.
- `experiment_summary.py`, `experiment_validation.py`, `feature_importance.py`, `sampling.py` —
  post-hoc checks/summaries of run artifacts and reproducible shot sampling.

`scripts/` holds one-off research scripts (BasketEvent download/inspection, release-frame review,
H1 improvement rounds) that import the package and monkeypatch e.g. `basketevent.clip_features`;
they take a cohort directory as `sys.argv[1]`. `reports/` holds committed run outputs;
`data/`, `artifacts/`, and `*.pt` weights are gitignored.

## Experimental rules (enforced in code and expected in changes)

- Never score the test split during development. `--split test` requires `--final`; H1 scripts use
  train+validation only. Don't run test evaluation unless explicitly asked.
- Each run goes in a new, empty output directory; never overwrite completed runs, frozen splits, or
  source snapshots (`baseline` refuses a non-empty directory).
- Keep failures (missing clips, no detected release, failed pose/LLM calls) visible in tables and
  audits; never silently replace them. Missingness can leak the outcome (see H1 log §2).
- Features must be release-time only — no post-release frames, trajectories, or outcomes.
- `ALLOWED_FEATURES` is the LLM feature set: `agents.py` requires a definition for each one and
  embeds all definitions in every prompt. Shot-chart and box features are XGBoost-only inputs
  (accepted in `models.feature_columns`) and must stay out of it, or the H2 prompts change.
- H1 changes are preregistered: the plan and decision rule are written in `docs/` *before* running
  (see `docs/h1_improvement_round.md`), and every result is logged in `docs/h1_experiments.md`,
  kept or not. Follow that pattern for new rounds.
- On 2025-26 data, defender distance, shot clock, dribbles, and touch time are always empty — leave
  them missing rather than inventing values.

## Docs

`docs/experiments.md` is the end-to-end run guide (Kaggle baseline → video cohort → annotate →
extract → train/compare → LLM → report). `docs/SPEC.md` has schemas and feature math;
`docs/h1_experiments.md` is the authoritative H1 log and current result (frozen design
`configs/basketevent_h1_final.json`).
