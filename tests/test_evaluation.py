import numpy as np
import pandas as pd
import pytest

from nba_shots.evaluation import (
    paired_common_success_comparison,
    paired_comparison,
    score_predictions,
    validate_predictions,
)


def predictions():
    rows = []
    for model in ["base", "perfect"]:
        for i in range(20):
            rows.append(
                {
                    "shot_id": str(i),
                    "game_id": str(i // 2),
                    "cohort": "test",
                    "split": "validation",
                    "model": model,
                    "y_true": i % 2,
                    "p_make": 0.5 if model == "base" else float(i % 2),
                    "status": "ok",
                }
            )
    return pd.DataFrame(rows)


def test_known_probabilities_and_game_bootstrap():
    frame = predictions()
    metrics = {r["model"]: r for r in score_predictions(frame)}
    assert metrics["base"]["brier_score"] == 0.25
    assert metrics["base"]["roc_auc"] == 0.5
    assert metrics["perfect"]["brier_score"] == 0
    result = paired_comparison(frame, "base", "perfect")
    assert result["brier_delta_candidate_minus_baseline"] == -0.25
    assert result["brier_delta_95pct_interval"] == [-0.25, -0.25]
    assert result == paired_comparison(frame, "base", "perfect")


def test_mismatched_ids_and_failures_are_not_silently_intersected():
    frame = predictions()
    with pytest.raises(ValueError, match="same nonempty shot IDs"):
        paired_comparison(frame.iloc[:-1], "base", "perfect")
    frame.loc[0, ["status", "p_make"]] = ["error", np.nan]
    metrics = {r["model"]: r for r in score_predictions(frame)}
    assert metrics["base"]["n_failed"] == 1
    assert metrics["base"]["coverage"] == 0.95
    with pytest.raises(ValueError, match="Resolve prediction failures"):
        paired_comparison(frame, "base", "perfect")


def test_invalid_probabilities_and_duplicate_predictions_rejected():
    frame = predictions()
    frame.loc[0, "p_make"] = 1.01
    with pytest.raises(ValueError, match="finite p_make"):
        validate_predictions(frame)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_predictions(pd.concat([predictions(), predictions().iloc[:1]]))


def test_single_class_auc_is_undefined():
    frame = predictions().loc[lambda p: p.y_true.eq(1)]
    assert all(r["roc_auc"] is None for r in score_predictions(frame))


def test_common_success_comparison_reports_failures():
    frame = predictions()

    # Each model fails on one different shot.
    frame.loc[
        frame.model.eq("base") & frame.shot_id.eq("0"),
        ["status", "p_make"],
    ] = ["error", np.nan]

    frame.loc[
        frame.model.eq("perfect") & frame.shot_id.eq("1"),
        ["status", "p_make"],
    ] = ["error", np.nan]

    result = paired_common_success_comparison(frame, "base", "perfect")

    assert result["n_attempted"] == 20
    assert result["baseline_successes"] == 19
    assert result["candidate_successes"] == 19
    assert result["baseline_coverage"] == 0.95
    assert result["candidate_coverage"] == 0.95
    assert result["n_common_success"] == 18
    assert result["n_excluded_from_pairing"] == 2
    assert result["n_shots"] == 18
    assert result["brier_delta_candidate_minus_baseline"] == -0.25


def test_common_success_requires_identical_attempted_shots():
    frame = predictions()
    frame = frame.loc[~(frame.model.eq("perfect") & frame.shot_id.eq("0"))]

    with pytest.raises(ValueError, match="same attempted shot IDs"):
        paired_common_success_comparison(frame, "base", "perfect")


def test_common_success_rejects_empty_overlap():
    frame = predictions()

    frame.loc[frame.model.eq("perfect"), ["status", "p_make"]] = ["error", np.nan]

    with pytest.raises(ValueError, match="No commonly successful shots"):
        paired_common_success_comparison(frame, "base", "perfect")


def test_common_success_matches_strict_comparison_without_failures():
    frame = predictions()

    strict = paired_comparison(frame, "base", "perfect")
    common = paired_common_success_comparison(frame, "base", "perfect")

    assert common["n_attempted"] == 20
    assert common["n_common_success"] == 20
    assert common["n_excluded_from_pairing"] == 0
    assert (
        common["brier_delta_candidate_minus_baseline"]
        == (strict["brier_delta_candidate_minus_baseline"])
    )
    assert common["brier_delta_95pct_interval"] == (strict["brier_delta_95pct_interval"])
