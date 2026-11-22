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


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("shot_id", ""),
        ("game_id", "   "),
        ("cohort", ""),
        ("split", "\t"),
        ("model", " "),
    ],
)
def test_rejects_blank_prediction_identity(column, invalid_value):
    frame = predictions()
    frame.loc[0, column] = invalid_value

    with pytest.raises(ValueError, match=f"Prediction {column} must contain nonblank strings"):
        validate_predictions(frame)


def test_accepts_nonblank_prediction_identities():
    frame = predictions()

    validate_predictions(frame)

    assert len(score_predictions(frame)) == 2


def test_game_bootstrap_weights_unequal_game_sizes():
    rows = []

    for model in ("base", "candidate"):
        for i in range(10):
            rows.append(
                {
                    "shot_id": f"shot-{i}",
                    "game_id": "game-large" if i < 9 else "game-small",
                    "cohort": "test",
                    "split": "validation",
                    "model": model,
                    "y_true": 0,
                    "p_make": (0.5 if model == "base" else (0.0 if i < 9 else 1.0)),
                    "status": "ok",
                }
            )

    frame = pd.DataFrame(rows)

    result = paired_comparison(
        frame,
        "base",
        "candidate",
        repetitions=1000,
        seed=42,
    )

    assert result["n_shots"] == 10
    assert result["n_games"] == 2

    # Nine improvements of -0.25 and one regression of +0.75.
    # The correct shot-weighted mean is -0.15, not +0.25.
    assert result["brier_delta_candidate_minus_baseline"] == pytest.approx(-0.15)

    # Whole-game resampling must preserve the game clusters.
    # Sampling only the large game gives -0.25.
    # Sampling only the small game gives +0.75.
    assert result["brier_delta_95pct_interval"] == pytest.approx([-0.25, 0.75])

    # Repeated evaluation with the same seed must be reproducible.
    repeated = paired_comparison(
        frame,
        "base",
        "candidate",
        repetitions=1000,
        seed=42,
    )

    assert result == repeated


def test_paired_bootstrap_rejects_single_game():
    frame = predictions()

    frame = frame.loc[frame["game_id"].eq("0")].copy()

    with pytest.raises(
        ValueError,
        match="At least two held-out games",
    ):
        paired_comparison(frame, "base", "perfect")
