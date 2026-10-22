"""Train-only preprocessing and persisted probability baselines."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .common import (
    ALLOWED_FEATURES,
    CV_FEATURES,
    HISTORY_FEATURES,
    environment,
    sha256,
    utc_now,
    write_json,
)
from .data import validate_shots
from .evaluation import write_scores


def feature_columns(config: dict, include_cv=False) -> tuple[list[str], list[str]]:
    numeric = list(config["numeric_features"])
    categorical = list(config["categorical_features"])
    if config["include_player_history"]:
        numeric += sorted(HISTORY_FEATURES)
    if include_cv:
        numeric += sorted(CV_FEATURES)
    features = numeric + categorical
    if not features or len(set(features)) != len(features) or set(features) - ALLOWED_FEATURES:
        raise ValueError("Features must be unique members of the pre-release allowlist")
    return numeric, categorical


def make_pipeline(model: str, config: dict, include_cv=False) -> Pipeline:
    numeric, categorical = feature_columns(config, include_cv)
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if model == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    preprocessing = ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    if model == "constant":
        estimator = DummyClassifier(strategy="prior", random_state=config["seed"])
    elif model == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=config["seed"])
    elif model in {"xgboost", "xgboost_cv"}:
        estimator = XGBClassifier(
            **config["xgboost"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config["seed"],
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def validate_splits(shots: pd.DataFrame) -> None:
    validate_shots(shots)
    if "split" not in shots or not shots["split"].isin(["train", "validation", "test"]).all():
        raise ValueError("Prepared data must have train/validation/test assignments")
    if set(shots["split"]) != {"train", "validation", "test"}:
        raise ValueError("All three partitions must be nonempty")
    for key in ["game_id", "game_date"]:
        if shots.groupby(key)["split"].nunique().gt(1).any():
            raise ValueError(f"A {key} crosses partition boundaries")
    dates = {
        s: shots.loc[shots["split"].eq(s), "game_date"] for s in ["train", "validation", "test"]
    }
    if not (
        dates["train"].max() < dates["validation"].min()
        and dates["validation"].max() < dates["test"].min()
    ):
        raise ValueError("Splits must be strictly chronological")


def prediction_frame(rows: pd.DataFrame, model: str, probabilities) -> pd.DataFrame:
    result = rows[["shot_id", "game_id", "cohort", "split", "y_true"]].copy()
    result["model"] = model
    result["p_make"] = np.asarray(probabilities, dtype=float)
    result["status"] = "ok"
    return result


def train_baselines(
    shots_path: str | Path, output_dir: str | Path, config: dict, cv_path: str | Path | None = None
) -> list[dict]:
    shots_path, output_dir = Path(shots_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Run directory is not empty: {output_dir}; choose a new run")
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = pd.read_parquet(shots_path)
    validate_splits(shots)
    models = ["constant", "logistic_regression", "xgboost"]
    if cv_path:
        cv = pd.read_parquet(cv_path)
        if "shot_id" not in cv or cv["shot_id"].duplicated().any():
            raise ValueError("CV table requires unique shot IDs")
        if set(shots["shot_id"]) != set(cv["shot_id"]):
            raise ValueError("CV table must cover the entire registered cohort, including failures")
        if missing := CV_FEATURES - set(cv):
            raise ValueError(f"Missing CV features: {sorted(missing)}")
        shots = shots.merge(
            cv[["shot_id", *sorted(CV_FEATURES)]], on="shot_id", how="left", validate="one_to_one"
        )
        models.append("xgboost_cv")
    train = shots.loc[shots["split"].eq("train")]
    validation = shots.loc[shots["split"].eq("validation")]
    if train["y_true"].nunique() != 2:
        raise ValueError("Training data must contain both makes and misses")
    started = utc_now()
    metadata = {
        "started_at": started,
        "config": config,
        "environment": environment(),
        "shots_path": str(shots_path.resolve()),
        "shots_sha256": sha256(shots_path),
        "cv_path": str(Path(cv_path).resolve()) if cv_path else None,
        "cv_sha256": sha256(cv_path) if cv_path else None,
        "evaluation_split": "validation",
        "models": {},
    }
    predictions = []
    for name in models:
        tick = time.perf_counter()
        use_cv = name == "xgboost_cv"
        numeric, categorical = feature_columns(config, include_cv=use_cv)
        features = numeric + categorical
        if missing := set(features) - set(shots):
            raise ValueError(f"Missing prepared features: {sorted(missing)}")
        pipeline = make_pipeline(name, config, include_cv=use_cv)
        pipeline.fit(train[features], train["y_true"])
        p = pipeline.predict_proba(validation[features])[:, 1]
        joblib.dump(
            {"pipeline": pipeline, "features": features, "name": name},
            output_dir / f"{name}.joblib",
        )
        predictions.append(prediction_frame(validation, name, p))
        metadata["models"][name] = {
            "features": features,
            "fit_rows": len(train),
            "validation_rows": len(validation),
            "runtime_seconds": time.perf_counter() - tick,
        }
    all_predictions = pd.concat(predictions, ignore_index=True)
    all_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    metrics = write_scores(all_predictions, output_dir)
    metadata["completed_at"] = utc_now()
    write_json(output_dir / "run.json", metadata)
    return metrics


def evaluate_run(
    run_dir: str | Path, shots_path: str | Path, split="validation", final=False
) -> list[dict]:
    import json

    run_dir, shots_path = Path(run_dir), Path(shots_path)
    if split == "test" and not final:
        raise ValueError("Test evaluation requires --final after feature/prompt freeze")
    if split not in {"validation", "test"}:
        raise ValueError("Evaluate validation or test only")
    metadata = json.loads((run_dir / "run.json").read_text())
    if sha256(shots_path) != metadata["shots_sha256"]:
        raise ValueError("Prepared data changed since training; use the original dataset")
    shots = pd.read_parquet(shots_path)
    validate_splits(shots)
    if metadata["cv_path"]:
        cv_path = Path(metadata["cv_path"])
        if sha256(cv_path) != metadata["cv_sha256"]:
            raise ValueError("CV features changed since training")
        cv = pd.read_parquet(cv_path)
        shots = shots.merge(
            cv[["shot_id", *sorted(CV_FEATURES)]], on="shot_id", how="left", validate="one_to_one"
        )
    rows = shots.loc[shots["split"].eq(split)]
    outputs = []
    for name in metadata["models"]:
        artifact = joblib.load(run_dir / f"{name}.joblib")
        p = artifact["pipeline"].predict_proba(rows[artifact["features"]])[:, 1]
        outputs.append(prediction_frame(rows, name, p))
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_csv(run_dir / f"{split}_predictions.csv", index=False)
    metrics = write_scores(predictions, run_dir, prefix=split)
    write_json(
        run_dir / f"{split}_evaluation.json",
        {
            "evaluated_at": utc_now(),
            "split": split,
            "final": final,
            "shots_sha256": sha256(shots_path),
        },
    )
    return metrics
