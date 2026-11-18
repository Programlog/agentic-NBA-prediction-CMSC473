
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

MODULE = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "h2_matched_analysis.py"
)

spec = importlib.util.spec_from_file_location(
    "h2_matched_analysis",
    MODULE,
)

h2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h2)


def frames():
    base = []

    for i in range(6):
        base.append({
            "cohort": "kaggle_2014_15",
            "split": "validation",
            "shot_id": str(i),
            "game_id": str(i // 2),
            "y_true": i % 2,
            "model": "xgboost",
            "p_make": 0.2 if i % 2 == 0 else 0.8,
            "status": "ok",
        })

    xgb = pd.DataFrame(base)

    llm = xgb.copy()
    llm["model"] = "llm_multi"

    llm["p_make"] = [
        0.3 if i % 2 == 0 else 0.7
        for i in range(6)
    ]

    return xgb, llm, xgb[h2.KEYS].copy()


def test_matched_and_bootstrap(tmp_path):
    xgb, llm, selection = frames()

    for name, frame in [
        ("x.csv", xgb),
        ("l.csv", llm),
        ("s.csv", selection),
    ]:
        frame.to_csv(tmp_path / name, index=False)

    summary, comparisons = h2.evaluate(
        tmp_path / "x.csv",
        tmp_path / "l.csv",
        tmp_path / "s.csv",
        tmp_path / "out",
        repeats=100,
    )

    assert len(summary) == 2

    assert comparisons[0]["n_common_success"] == 6
    assert comparisons[0]["n_games"] == 3

    assert (
        comparisons[0]["brier_delta_llm_minus_xgb"] > 0
    )

    assert (
        tmp_path / "out" / "largest_disagreements.csv"
    ).exists()


def test_rejects_missing_llm_shot():
    xgb, llm, selection = frames()

    with pytest.raises(
        ValueError,
        match="exactly match",
    ):
        h2.matched_predictions(
            xgb,
            llm.iloc[:-1],
            selection,
        )


def test_rejects_wrong_labels():
    xgb, llm, selection = frames()

    llm.loc[0, "y_true"] = 1

    with pytest.raises(
        ValueError,
        match="outcomes",
    ):
        h2.matched_predictions(
            xgb,
            llm,
            selection,
        )


def test_reports_failed_prediction_coverage(tmp_path):
    xgb, llm, selection = frames()

    llm.loc[
        0,
        ["status", "p_make"],
    ] = ["error", float("nan")]

    for name, frame in [
        ("x.csv", xgb),
        ("l.csv", llm),
        ("s.csv", selection),
    ]:
        frame.to_csv(tmp_path / name, index=False)

    summary, comparisons = h2.evaluate(
        tmp_path / "x.csv",
        tmp_path / "l.csv",
        tmp_path / "s.csv",
        tmp_path / "out",
        repeats=25,
    )

    llm_metrics = next(
        x
        for x in summary
        if x["model"] == "llm_multi"
    )

    assert llm_metrics["coverage"] == 5 / 6
    assert comparisons[0]["n_common_success"] == 5
