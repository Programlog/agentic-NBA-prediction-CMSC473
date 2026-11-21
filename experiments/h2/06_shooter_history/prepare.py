"""Build shooter-history features and retrain the tabular baselines on them.

History is "online": every value uses only shots from strictly earlier dates
(the same rule as ``nba_shots.data.add_player_history``), so validation and test
shots can use earlier validation/test games but never their own date.

Outputs (default ``artifacts/h2_history``):
  shots.parquet                 day-1 shots + history columns + player_name
  models/{constant,logistic_regression,xgboost}.joblib   ({"pipeline","features"})
  baseline_predictions.csv      validation + test predictions of the history baselines
  prepare.json                  SHAs, features, sanity checks
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from nba_shots.data import add_player_history

DISTANCE_BANDS = [0, 3, 6, 10, 16, 22, 24, 28, np.inf]  # same as experiment 04
SHRINK_K = 20
HISTORY = [
    "player_prior_attempts",
    "player_prior_make_rate",
    "player_band_attempts",
    "player_band_rate",
]


def band_of(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, DISTANCE_BANDS, right=False).cat.codes


def add_band_history(shots: pd.DataFrame) -> pd.DataFrame:
    """Player's earlier-date make rate in this distance band, shrunk to the league band rate."""
    shots = shots.assign(_band=band_of(shots["shot_distance"]))
    league = shots[shots["split"].eq("train")].groupby("_band")["y_true"].mean()
    daily = (
        shots.groupby(["player_id", "_band", "game_date"], as_index=False)
        .agg(attempts=("y_true", "size"), makes=("y_true", "sum"))
        .sort_values(["player_id", "_band", "game_date"])
    )
    groups = daily.groupby(["player_id", "_band"], sort=False)
    daily["player_band_attempts"] = groups["attempts"].cumsum() - daily["attempts"]
    prior_makes = groups["makes"].cumsum() - daily["makes"]
    league_rate = daily["_band"].map(league)
    daily["player_band_rate"] = (prior_makes + SHRINK_K * league_rate) / (
        daily["player_band_attempts"] + SHRINK_K
    )
    out = shots.merge(
        daily[["player_id", "_band", "game_date", "player_band_attempts", "player_band_rate"]],
        on=["player_id", "_band", "game_date"],
        how="left",
        validate="many_to_one",
    )
    return out.drop(columns="_band")


def pipeline(model: str, numeric: list[str], categorical: list[str], config: dict) -> Pipeline:
    """Same preprocessing and estimators as ``nba_shots.models.make_pipeline``."""
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if model == "logistic_regression":
        numeric_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
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
        est = DummyClassifier(strategy="prior", random_state=config["seed"])
    elif model == "logistic_regression":
        est = LogisticRegression(max_iter=2000, random_state=config["seed"])
    else:
        est = XGBClassifier(
            **config["xgboost"],
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=config["seed"],
        )
    return Pipeline([("preprocess", pre), ("model", est)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="artifacts/day1/prepared/shots.parquet")
    parser.add_argument("--raw", default="data/raw/shot_logs.csv")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--day1-models", default="artifacts/day1/models")
    parser.add_argument("--out", default="artifacts/h2_history")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "models").mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())

    shots = pd.read_parquet(args.shots)
    shots = add_band_history(add_player_history(shots))
    raw = pd.read_csv(args.raw, usecols=["player_id", "player_name"], dtype={"player_id": str})
    names = raw.drop_duplicates()
    assert not names["player_id"].duplicated().any(), "player_id maps to several names"
    shots = shots.merge(names, on="player_id", how="left", validate="many_to_one")
    assert shots["player_name"].notna().all()
    assert shots[HISTORY].notna().all().all()
    shots_path = out / "shots.parquet"
    shots.to_parquet(shots_path, index=False)

    base_numeric = list(config["numeric_features"])
    categorical = list(config["categorical_features"])
    train = shots[shots["split"].eq("train")]
    checks = {}

    # Sanity: without history, the local pipeline must reproduce the frozen day-1 XGBoost.
    frozen = pd.read_csv(Path(args.day1_models) / "validation_predictions.csv")
    frozen = frozen[frozen["model"].eq("xgboost")].set_index("shot_id")["p_make"]
    plain = pipeline("xgboost", base_numeric, categorical, config)
    plain.fit(train[base_numeric + categorical], train["y_true"])
    val = shots[shots["split"].eq("validation")]
    p_plain = pd.Series(plain.predict_proba(val[base_numeric + categorical])[:, 1], val["shot_id"])
    checks["day1_xgboost_max_abs_diff"] = float((p_plain - frozen.loc[p_plain.index]).abs().max())

    numeric = base_numeric + HISTORY
    features = numeric + categorical
    rows = []
    for name in ["constant", "logistic_regression", "xgboost"]:
        pipe = pipeline(name, numeric, categorical, config)
        pipe.fit(train[features], train["y_true"])
        joblib.dump(
            {"pipeline": pipe, "features": features, "name": name},
            out / "models" / f"{name}.joblib",
        )
        for split in ["validation", "test"]:
            part = shots[shots["split"].eq(split)]
            rows.append(
                part[["shot_id", "game_id", "split", "y_true"]].assign(
                    model=f"{name}_hist", p_make=pipe.predict_proba(part[features])[:, 1]
                )
            )
    pd.concat(rows).to_csv(out / "baseline_predictions.csv", index=False)

    meta = {
        "shots_sha256": hashlib.sha256(shots_path.read_bytes()).hexdigest(),
        "source_shots_sha256": hashlib.sha256(Path(args.shots).read_bytes()).hexdigest(),
        "history_features": HISTORY,
        "band_shrink_k": SHRINK_K,
        "distance_bands": [str(b) for b in DISTANCE_BANDS],
        "features": features,
        "checks": checks,
    }
    (out / "prepare.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
