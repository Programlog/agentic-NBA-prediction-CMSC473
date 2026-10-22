import numpy as np
import pandas as pd
import pytest

from nba_shots.common import DEFAULT_CONFIG, TABULAR_FEATURES, load_config
from nba_shots.data import add_player_history, assign_splits, normalize_kaggle, prepare_csv


def test_normalization_drops_outcomes_and_preserves_stable_ids(raw_shots):
    shots, audit = normalize_kaggle(raw_shots)
    shuffled, _ = normalize_kaggle(raw_shots.sample(frac=1, random_state=3).reset_index(drop=True))
    pd.testing.assert_frame_equal(shots, shuffled)
    assert shots.shot_id.is_unique
    assert shots.game_id.str.len().eq(10).all()
    assert not {"FGM", "PTS", "W", "FINAL_MARGIN", "SHOT_RESULT"} & set(shots)
    assert TABULAR_FEATURES <= set(shots)
    assert audit["make_rate"] == 0.5


def test_invalid_values_become_missing_without_dropping_shots(raw_shots):
    raw_shots.loc[0, ["TOUCH_TIME", "SHOT_CLOCK", "GAME_CLOCK"]] = [-2, 30, "bad"]
    shots, audit = normalize_kaggle(raw_shots)
    row = shots.loc[shots.shot_id.eq("kaggle:0021400002:100:1")].iloc[0]
    assert np.isnan(row.touch_time) and np.isnan(row.shot_clock)
    assert np.isnan(row.game_clock_seconds)
    assert len(shots) == len(raw_shots)
    assert audit["invalid_numeric_values_replaced_with_missing"]["touch_time"] == 1


def test_duplicate_keys_and_bad_labels_are_rejected(raw_shots):
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_kaggle(pd.concat([raw_shots, raw_shots.iloc[:1]], ignore_index=True))
    raw_shots.loc[0, "SHOT_RESULT"] = "unknown"
    with pytest.raises(ValueError, match="made/missed"):
        normalize_kaggle(raw_shots)


def test_splits_keep_dates_and_games_together(raw_shots):
    frame, _ = normalize_kaggle(raw_shots)
    shots, manifest = assign_splits(frame)
    assert shots.groupby("game_date").split.nunique().eq(1).all()
    assert shots.groupby("game_id").split.nunique().eq(1).all()
    a, b, c = [set(manifest["games"][s]) for s in ["train", "validation", "test"]]
    assert not a & b and not a & c and not b & c
    train = shots.loc[shots.split.eq("train"), "game_date"]
    val = shots.loc[shots.split.eq("validation"), "game_date"]
    test = shots.loc[shots.split.eq("test"), "game_date"]
    assert train.max() < val.min() and val.max() < test.min()


def test_player_history_excludes_entire_current_date_and_future(raw_shots):
    frame, _ = normalize_kaggle(raw_shots)
    history = add_player_history(frame)
    first = history.loc[history.game_date.eq("2014-11-01")]
    second = history.loc[history.game_date.eq("2014-11-02")]
    assert first.player_prior_attempts.eq(0).all()
    assert first.player_prior_make_rate.eq(0.5).all()
    assert second.player_prior_attempts.eq(4).all()
    changed = frame.copy()
    changed.loc[changed.game_date.ge("2014-11-02"), "y_true"] = 1
    changed_history = add_player_history(changed)
    columns = ["shot_id", "player_prior_attempts", "player_prior_make_rate"]
    pd.testing.assert_frame_equal(
        history.loc[history.game_date.le("2014-11-02"), columns],
        changed_history.loc[changed_history.game_date.le("2014-11-02"), columns],
    )


def test_prepare_saves_source_audit_and_refuses_overwrite(raw_shots, tmp_path):
    source = tmp_path / "shots.csv"
    raw_shots.to_csv(source, index=False)
    result = prepare_csv(source, tmp_path / "prepared", load_config())
    assert result.exists()
    assert (result.parent / "splits.json").exists()
    assert (result.parent / "audit.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        prepare_csv(source, result.parent, load_config())


def test_configuration_cannot_enable_label_columns(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"numeric_features": ["y_true"]}')
    with pytest.raises(ValueError, match="approved tabular"):
        load_config(path)
    assert load_config("configs/default.json") == DEFAULT_CONFIG
