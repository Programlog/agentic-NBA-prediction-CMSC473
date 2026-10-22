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
