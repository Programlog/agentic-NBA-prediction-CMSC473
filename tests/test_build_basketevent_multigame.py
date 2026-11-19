"""Offline tests for the multi-game JSON selection and NBA matching logic."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_basketevent_multigame.py"
spec = importlib.util.spec_from_file_location("basket_multigame", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_index_only_paired_annotation_video_and_valid_split():
    files = ["test/0022500003/a.json", "test/0022500003/a.mp4",
             "train/0022500004/b.json", "valid/0022500005/c.json",
             "valid/0022500005/c.mp4", "test/garbage/d.json", "test/garbage/d.mp4"]
    assert module.indexed_paths(files) == {
        "0022500003": ["test/0022500003/a.json"],
        "0022500005": ["valid/0022500005/c.json"],
    }


def test_clock_parsing_and_invalid():
    assert module.parse_clock("PT09M58.00S") == 598
    assert module.parse_clock("PT00M04.5S") == 4.5
    with pytest.raises(ValueError):
        module.parse_clock("9:58")


def test_extract_shots_ignores_non_shot_events_and_deduplicates():
    make = {"clock": "PT08M00.00S", "period": 2, "actionType": "Made Shot",
            "playerName": "X", "description": "X Jump Shot"}
    source = {"player_0": {"event": make}, "player_1": {"event": dict(make)},
              "player_2": {"event": {"actionType": "Rebound"}},
              "player_3": {"event": {**make, "actionType": "Missed Shot"}}}
    rows = module.extract_events(source, "test/0022500003/clip.json")
    assert len(rows) == 2
    assert {r["event_type"] for r in rows} == {"made shot", "missed shot"}
    assert all(r["video_path"].endswith("clip.mp4") for r in rows)


def sample_shots():
    return pd.DataFrame([
        {"shot_id": "nba:0022500003:1", "game_id": "0022500003", "game_event_id": "1",
         "period": 1, "game_clock_seconds": 420, "y_true": 0},
        {"shot_id": "nba:0022500003:2", "game_id": "0022500003", "game_event_id": "2",
         "period": 2, "game_clock_seconds": 420, "y_true": 1},
    ])


def test_unique_matching_requires_game_period_clock_and_checks_outcome():
    event = {"event_clock": "PT07M00.00S", "event_period": 2, "event_type": "made shot"}
    status, result = module.nba_matches(event, "0022500003", sample_shots())
    assert status == "matched"
    assert result["shot_id"] == "nba:0022500003:2"
    status, result = module.nba_matches({**event, "event_type": "missed shot"},
                                        "0022500003", sample_shots())
    assert status == "outcome_mismatch" and result is None


def test_ambiguous_matching_not_resolved_by_outcome():
    source = pd.concat([sample_shots(), sample_shots().iloc[[1]].assign(shot_id="another")],
                       ignore_index=True)
    event = {"event_clock": "PT07M00.00S", "event_period": 2, "event_type": "made shot"}
    status, result = module.nba_matches(event, "0022500003", source)
    assert status == "ambiguous_nba_match" and result is None


def test_invalid_event_time_does_not_crash():
    status, row = module.nba_matches(
        {"event_clock": "bad", "event_period": "?", "event_type": "made shot"},
        "0022500003", sample_shots())
    assert status.startswith("invalid_event_time") and row is None


def test_game_sampling_reproducible_and_independent_of_files_order():
    grouped = {str(i): [f"file{i}"] for i in range(10)}
    assert module.choose_games(grouped, 5, 42) == module.choose_games(
        dict(reversed(list(grouped.items()))), 5, 42)
    assert len(module.choose_games(grouped, 5, 42)) == 5
