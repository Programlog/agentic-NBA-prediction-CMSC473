import json

import pandas as pd
import pytest

from nba_shots.evaluation import write_scores
from nba_shots.report import create_report


def test_report_requires_explicit_common_selection(tmp_path):
    a = pd.DataFrame(
        {
            "shot_id": ["a", "b", "c"],
            "game_id": ["1", "2", "3"],
            "cohort": "recent",
            "split": "validation",
            "y_true": [0, 1, 0],
            "model": "xgboost",
            "p_make": [0.4, 0.6, 0.5],
        }
    )
    b = a.iloc[:2].copy()
    b["model"] = "llm_single"
    combined = pd.concat([a, b])
    with pytest.raises(ValueError, match="different evaluation shots"):
        create_report(combined, tmp_path, write_scores)
    selection = b[["cohort", "split", "shot_id"]]
    result = create_report(combined, tmp_path, write_scores, selection)
    assert all(m["n_total"] == 2 for m in result["metrics"])
    assert (tmp_path / "report.md").is_file()


def report_predictions():
    """Four games and eight shots shared by two LLM conditions."""
    rows = []

    for model in ("llm_single", "llm_multi"):
        for i in range(8):
            y_true = i % 2

            rows.append(
                {
                    "shot_id": f"shot-{i}",
                    "game_id": f"game-{i // 2}",
                    "cohort": "kaggle_test",
                    "split": "validation",
                    "model": model,
                    "y_true": y_true,
                    "p_make": (0.3 if y_true == 0 else 0.7)
                    if model == "llm_multi"
                    else (0.4 if y_true == 0 else 0.6),
                    "status": "ok",
                }
            )

    return pd.DataFrame(rows)


def test_report_writes_complete_paired_comparison(tmp_path):
    frame = report_predictions()

    create_report(frame, tmp_path, write_scores)

    report = (tmp_path / "report.md").read_text()
    comparisons = json.loads((tmp_path / "paired_comparisons.json").read_text())

    assert "llm_single 8/8 (100.0%)" in report
    assert "llm_multi 8/8 (100.0%)" in report
    assert "Common-success cohort: 8 / 8" in report

    assert len(comparisons) == 1
    comparison = comparisons[0]

    assert comparison["n_attempted"] == 8
    assert comparison["n_common_success"] == 8
    assert comparison["n_games"] == 4
    assert comparison["baseline_coverage"] == 1.0
    assert comparison["candidate_coverage"] == 1.0


def test_report_preserves_coverage_with_different_failures(tmp_path):
    frame = report_predictions()

    frame.loc[
        frame.model.eq("llm_single") & frame.shot_id.eq("shot-0"),
        ["status", "p_make"],
    ] = ["error", float("nan")]

    frame.loc[
        frame.model.eq("llm_multi") & frame.shot_id.eq("shot-1"),
        ["status", "p_make"],
    ] = ["error", float("nan")]

    create_report(frame, tmp_path, write_scores)

    report = (tmp_path / "report.md").read_text()
    comparisons = json.loads((tmp_path / "paired_comparisons.json").read_text())

    assert "llm_single 7/8 (87.5%)" in report
    assert "llm_multi 7/8 (87.5%)" in report
    assert "Common-success cohort: 6 / 8" in report

    assert len(comparisons) == 1
    comparison = comparisons[0]

    assert comparison["n_common_success"] == 6
    assert comparison["n_excluded_from_pairing"] == 2
    assert comparison["baseline_coverage"] == 0.875
    assert comparison["candidate_coverage"] == 0.875


def test_report_shows_coverage_when_no_common_success(tmp_path):
    frame = report_predictions()

    frame.loc[
        frame.model.eq("llm_multi"),
        ["status", "p_make"],
    ] = ["error", float("nan")]

    create_report(frame, tmp_path, write_scores)

    report = (tmp_path / "report.md").read_text()
    comparisons = json.loads((tmp_path / "paired_comparisons.json").read_text())

    assert "llm_single 8/8 (100.0%)" in report
    assert "llm_multi 0/8 (0.0%)" in report
    assert "No commonly successful shots" in report
    assert comparisons == []


def test_report_shows_coverage_with_insufficient_games(tmp_path):
    frame = report_predictions()

    frame = frame.loc[frame["game_id"].eq("game-0")].copy()

    create_report(frame, tmp_path, write_scores)

    report = (tmp_path / "report.md").read_text()
    comparisons = json.loads((tmp_path / "paired_comparisons.json").read_text())

    assert "llm_single 2/2 (100.0%)" in report
    assert "llm_multi 2/2 (100.0%)" in report
    assert "At least two held-out games" in report
    assert comparisons == []
