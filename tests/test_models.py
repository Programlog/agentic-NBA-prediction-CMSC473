import json

import joblib
import numpy as np
import pandas as pd
import pytest

from nba_shots.common import load_config
from nba_shots.data import prepare_csv
from nba_shots.models import evaluate_run, make_pipeline, train_baselines


def test_preprocessor_is_fit_only_on_training_data(raw_shots):
    from nba_shots.data import assign_splits, normalize_kaggle

    frame, _ = normalize_kaggle(raw_shots)
    frame, _ = assign_splits(frame)
    cfg = load_config()
    training = frame.loc[frame.split.eq("train")]
    pipeline = make_pipeline("logistic_regression", cfg)
    pipeline.fit(training, training.y_true)
    statistics = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_.copy()
    )
    held_out = frame.loc[frame.split.eq("validation")].copy()
    held_out["shot_distance"] = 100000
    pipeline.predict_proba(held_out)
    np.testing.assert_array_equal(
        statistics,
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute"]
        .statistics_,
    )
    assert statistics[0] == training.shot_distance.median()


def test_end_to_end_models_reload_and_test_gate(raw_shots, tmp_path):
    csv = tmp_path / "source.csv"
    raw_shots.to_csv(csv, index=False)
    cfg = load_config()
    cfg["xgboost"]["n_estimators"] = 5
    shots = prepare_csv(csv, tmp_path / "prepared", cfg)
    output = tmp_path / "run"
    metrics = train_baselines(shots, output, cfg)
    assert len(metrics) == 3
    assert not (output / "test_predictions.csv").exists()
    first = pd.read_csv(output / "validation_predictions.csv")
    data = pd.read_parquet(shots)
    assert set(first.shot_id) == set(data.loc[data.split.eq("validation"), "shot_id"])
    assert (
        first.loc[first.model.eq("constant"), "p_make"]
        .eq(data.loc[data.split.eq("train"), "y_true"].mean())
        .all()
    )
    artifact = joblib.load(output / "xgboost.joblib")
    assert "y_true" not in artifact["features"]
    assert json.loads((output / "run.json").read_text())["shots_sha256"]
    evaluate_run(output, shots)
    second = pd.read_csv(output / "validation_predictions.csv")
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="requires --final"):
        evaluate_run(output, shots, split="test")
    with pytest.raises(ValueError, match="not empty"):
        train_baselines(shots, output, cfg)
