"""Small shared contracts for data, configuration and experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

DEFAULT_CONFIG = {
    "seed": 42,
    "train_fraction": 0.7,
    "validation_fraction": 0.15,
    "numeric_features": [
        "shot_distance",
        "defender_distance",
        "game_clock_seconds",
        "shot_clock",
        "period",
        "dribbles",
        "touch_time",
        "shot_value",
    ],
    "categorical_features": ["location"],
    "include_player_history": False,
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5,
        "n_jobs": 4,
    },
}
TABULAR_FEATURES = frozenset(DEFAULT_CONFIG["numeric_features"] + ["location"])
HISTORY_FEATURES = frozenset({"player_prior_attempts", "player_prior_make_rate"})
CV_FEATURES = frozenset(
    {
        "elbow_angle_deg",
        "wrist_height_torso",
        "defender_distance_torso",
        "defender_wrist_height_torso",
    }
)
ALLOWED_FEATURES = TABULAR_FEATURES | HISTORY_FEATURES | CV_FEATURES
PREDICTION_COLUMNS = ["shot_id", "game_id", "cohort", "split", "model", "y_true", "p_make"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_config(path: str | Path | None = None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        updates = json.loads(Path(path).read_text())
        unknown = set(updates) - set(cfg)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        xgb = {**cfg["xgboost"], **updates.get("xgboost", {})}
        cfg.update(updates)
        cfg["xgboost"] = xgb
    train, val = cfg["train_fraction"], cfg["validation_fraction"]
    if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
        raise ValueError("Split fractions must leave nonempty train, validation and test portions")
    features = cfg["numeric_features"] + cfg["categorical_features"]
    if len(features) != len(set(features)) or not features:
        raise ValueError("Features must be nonempty and unique")
    if set(features) - TABULAR_FEATURES:
        raise ValueError("Base configuration may contain only approved tabular features")
    return cfg


def environment() -> dict:
    packages = {}
    for name in [
        "nba-shots",
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "ultralytics",
        "crewai",
        "litellm",
        "matplotlib",
    ]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": proc.stdout.strip() or None,
        "git_dirty": bool(status.stdout.strip()),
    }
