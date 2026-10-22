from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nba_shots.common import load_config
from nba_shots.data import assign_splits, normalize_kaggle
from nba_shots.vision import (
    make_manifest,
    match_box,
    normalize_recent,
    pose_features,
    prepare_cohort,
    read_release_frame,
)


def test_pose_geometry_and_low_confidence():
    pose = np.zeros((17, 3))
    pose[:, 2] = 0.9
    pose[5, :2], pose[6, :2] = [-1, 0], [1, 0]
    pose[11, :2], pose[12, :2] = [-1, 2], [1, 2]
    pose[8, :2], pose[10, :2] = [2, 0], [2, -1]
    defender = pose.copy()
    defender[:, 0] += 4
    defender[9, 1] = -2
    values = pose_features(pose, defender)
    assert values["elbow_angle_deg"] == pytest.approx(90)
    assert values["wrist_height_torso"] == pytest.approx(0.5)
    assert values["defender_distance_torso"] == pytest.approx(2)
    assert values["defender_wrist_height_torso"] == pytest.approx(1)
    pose[10, 2] = 0.1
    assert np.isnan(pose_features(pose)["elbow_angle_deg"])
    pose[5, 2] = 0.1
    assert all(np.isnan(v) for v in pose_features(pose).values())


def test_release_reader_reads_exactly_one_annotated_frame(tmp_path):
    class Capture:
        def __init__(self):
            self.reads, self.index, self.released = 0, None, False

        def isOpened(self):
            return True

        def set(self, prop, index):
            self.index = index

        def read(self):
            self.reads += 1
            return True, "release-image"

        def get(self, prop):
            return self.index + 1

        def release(self):
            self.released = True

    capture = Capture()
    fake_cv2 = SimpleNamespace(VideoCapture=lambda p: capture, CAP_PROP_POS_FRAMES=1)
    assert read_release_frame(tmp_path / "video.mp4", 7, fake_cv2) == "release-image"
    assert capture.index == 7 and capture.reads == 1 and capture.released
    capture.get = lambda prop: 9  # A seek to a future frame is rejected.
    with pytest.raises(ValueError, match="decoded frame 8"):
        read_release_frame(tmp_path / "video.mp4", 7, fake_cv2)


def test_manual_box_identity_requires_overlap():
    boxes = np.array([[0, 0, 10, 20], [30, 0, 40, 20]])
    assert match_box(boxes, "[29,0,41,21]") == 1
    with pytest.raises(ValueError, match="cannot be matched"):
        match_box(boxes, "[100,100,110,110]")


def test_cohort_registration_keeps_missing_clips_and_uses_no_labels(raw_shots, tmp_path):
    shots, _ = normalize_kaggle(raw_shots)
    shots["game_event_id"] = shots.groupby("game_id").cumcount().astype(str)
    source = tmp_path / "shots.parquet"
    shots.to_parquet(source)
    manifest = tmp_path / "manifest.csv"
    first = make_manifest(source, manifest, games=6, shots_per_game=2)
    selected = pd.read_csv(manifest)
    assert len(selected) == 12 and first["games"] == 6
    assert "y_true" not in selected and selected.clip_path.isna().all()
    prepared = prepare_cohort(source, manifest, tmp_path / "prepared", load_config())
    cohort = pd.read_parquet(prepared["shots"])
    assert set(cohort.shot_id) == set(selected.shot_id)
    assert set(cohort.split) == {"train", "validation", "test"}
    changed = shots.copy()
    changed["y_true"] = 1 - changed.y_true
    changed.to_parquet(source)
    second = tmp_path / "manifest2.csv"
    make_manifest(source, second, games=6, shots_per_game=2)
    pd.testing.assert_frame_equal(pd.read_csv(manifest), pd.read_csv(second))


def test_recent_source_does_not_invent_unavailable_tracking_fields():
    raw = pd.DataFrame(
        [
            {
                "GAME_ID": "0022500001",
                "GAME_EVENT_ID": 9,
                "GAME_DATE": "20251022",
                "PLAYER_ID": 123,
                "TEAM_ID": 1,
                "HTM": "ATL",
                "SHOT_MADE_FLAG": 1,
                "SHOT_DISTANCE": 22,
                "PERIOD": 1,
                "MINUTES_REMAINING": 5,
                "SECONDS_REMAINING": 30,
                "SHOT_TYPE": "3PT Field Goal",
            }
        ]
    )
    shots = normalize_recent(raw, "2025-26", {1: "ATL"})
    assert shots.iloc[0].shot_id == "nba:0022500001:9"
    assert shots.iloc[0].location == "H" and shots.iloc[0].shot_value == 3
    assert shots.defender_distance.isna().all() and shots.shot_clock.isna().all()
    assert "SHOT_MADE_FLAG" not in shots


def test_cv_comparison_preserves_every_registered_shot(raw_shots, tmp_path):
    from nba_shots.common import CV_FEATURES
    from nba_shots.models import train_baselines

    frame, _ = normalize_kaggle(raw_shots)
    frame, _ = assign_splits(frame)
    shots_path = tmp_path / "shots.parquet"
    frame.to_parquet(shots_path)
    cv = pd.DataFrame({"shot_id": frame.shot_id, **{name: np.nan for name in CV_FEATURES}})
    cv_path = tmp_path / "cv.parquet"
    cv.to_parquet(cv_path)
    cfg = load_config()
    cfg["xgboost"]["n_estimators"] = 2
    train_baselines(shots_path, tmp_path / "run", cfg, cv_path)
    predictions = pd.read_csv(tmp_path / "run" / "validation_predictions.csv")
    sets = predictions.groupby("model").shot_id.apply(set)
    assert sets["xgboost"] == sets["xgboost_cv"]
    cv.iloc[:-1].to_parquet(cv_path)
    with pytest.raises(ValueError, match="entire registered cohort"):
        train_baselines(shots_path, tmp_path / "bad-run", cfg, cv_path)
