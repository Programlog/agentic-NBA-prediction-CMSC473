import importlib.util
from pathlib import Path
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/annotate_basketevent_assisted.py"
spec = importlib.util.spec_from_file_location("h1_annot", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_xywh_conversion():
    assert m.xywh_to_xyxy([100, 50, 30, 40], 1280, 720) == [100, 50, 130, 90]
    assert m.xywh_to_xyxy([-5, 2, 10, 10], 1280, 720) == [0, 2, 5, 12]
    assert m.xywh_to_xyxy([10, 10, 0, 10], 1280, 720) is None
    assert m.xywh_to_xyxy([10, 10, 10, 10], 15, 15) == [10, 10, 15, 15]


def test_role_at_frame():
    d = {"player_8": {"trajectory": [[3, 4, 10, 20], None]}}
    assert m.role_at_frame(d, "player_8", 0, 100, 100) == [3, 4, 13, 24]
    assert m.role_at_frame(d, "player_8", 1, 100, 100) is None
    assert m.role_at_frame(d, "player_8", 2, 100, 100) is None


def test_role_matching_selected_only():
    report = pd.DataFrame([
        {"shot_id": "nba:0022500004:58", "json_path": "test/a.json", "source_role": "player_8", "selection_status": "selected", "match_status": "matched"},
        {"shot_id": "nba:0022500004:58", "json_path": "test/a.json", "source_role": "player_2", "selection_status": "not_selected_quota", "match_status": "matched"},
    ])
    assert m.source_role_for_shot(report, "nba:0022500004:58", "test/a.json") == "player_8"
    with pytest.raises(ValueError):
        m.source_role_for_shot(report, "missing", "test/a.json")
