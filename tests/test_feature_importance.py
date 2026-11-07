import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from nba_shots.common import sha256
from nba_shots.feature_importance import analyze_saved_model, permutation_brier_importance


def test_reproducible_importance_identifies_predictive_feature():
    rng = np.random.default_rng(123)
    signal = rng.normal(size=400)
    noise = rng.normal(size=400)
    x = pd.DataFrame({"shot_distance": signal, "noise": noise})
    y = pd.Series((signal > 0).astype(int))
    model = LogisticRegression().fit(x, y)
    first = permutation_brier_importance(model, x, y, repeats=8, seed=7)
    second = permutation_brier_importance(model, x, y, repeats=8, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert first.iloc[0].feature == "shot_distance"
    assert first.iloc[0].importance_mean > 0.1
    assert first.iloc[0].importance_mean > first.iloc[1].importance_mean


def test_rejects_invalid_inputs():
    x = pd.DataFrame({"a": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression().fit(x, y)
    with pytest.raises(ValueError, match="repeats"):
        permutation_brier_importance(model, x, y, repeats=0)
    with pytest.raises(ValueError, match="Labels"):
        permutation_brier_importance(model, x, pd.Series([0, 1]))


def test_saved_model_uses_validation_only_and_checks_checksum(tmp_path):

    rng = np.random.default_rng(99)
    n = 36
    x = pd.DataFrame({"shot_distance": rng.normal(size=n)})
    y = pd.Series((x.shot_distance > 0).astype(int))
    model = LogisticRegression().fit(x, y)
    run = tmp_path / "run"
    run.mkdir()
    joblib.dump({"pipeline": model, "features": ["shot_distance"], "name": "xgboost"},
                run / "xgboost.joblib")
    shots = pd.DataFrame({
        "shot_id": [f"s{i}" for i in range(n)],
        "game_id": [f"g{i // 3}" for i in range(n)],
        "game_date": [f"2026-01-{i // 3 + 1:02d}" for i in range(n)],
        "cohort": "synthetic",
        "y_true": y,
        "shot_distance": x.shot_distance,
        "split": ["train"] * 12 + ["validation"] * 12 + ["test"] * 12,
    })
    path = tmp_path / "shots.parquet"
    shots.to_parquet(path)
    (run / "run.json").write_text(json.dumps({
        "shots_sha256": sha256(path), "models": {"xgboost": {}}, "cv_path": None,
    }))

   
    out = tmp_path / "importance"
    result = analyze_saved_model(run, path, out, repeats=3)
    assert result.n_validation_shots.iloc[0] == 12
    assert (out / "feature_importance.csv").exists()
    assert (out / "feature_importance.png").exists()
    assert json.loads((out / "feature_importance.json").read_text())["split"] == "validation"
    with pytest.raises(ValueError, match="not empty"):
        analyze_saved_model(run, path, out)
    shots.loc[0, "shot_distance"] = 999
    shots.to_parquet(path)
    with pytest.raises(ValueError, match="differs"):
        analyze_saved_model(run, path, tmp_path / "other")
