"""Tests for the raw release-candidate failure audit."""

import csv
import json

import numpy as np
import pytest

from tools import audit_release_raw_candidate_failures as audit


def make_tracks(ball_x=115, ball_y=115, future_ball_y=70):
    """Construct simple synthetic shooter and ball trajectories."""
    shooter = np.tile([100.0, 100.0, 40.0, 100.0], (12, 1))
    ball = np.tile([ball_x, ball_y, 10.0, 10.0], (12, 1))

    # Make subsequent ball positions higher in the image.
    ball[3:10, 1] = future_ball_y

    return shooter, ball


def test_missing_shooter_box():
    shooter, ball = make_tracks()
    shooter[2] = np.nan

    assert audit.inspect_frame(shooter, ball, 2) == ("missing_shooter_box")


def test_missing_ball_box():
    shooter, ball = make_tracks()
    ball[2] = np.nan

    assert audit.inspect_frame(shooter, ball, 2) == ("missing_ball_box")


def test_outside_hand_reach():
    shooter, ball = make_tracks(ball_x=400)

    assert audit.inspect_frame(shooter, ball, 2) == ("outside_hand_reach")


def test_rise_threshold_not_met():
    shooter, ball = make_tracks(future_ball_y=110)

    assert audit.inspect_frame(shooter, ball, 2) == ("rise_threshold_not_met")


def test_qualifying_raw_candidate():
    shooter, ball = make_tracks()

    assert audit.inspect_frame(shooter, ball, 2) == ("qualifying_raw_candidate")


def test_insufficient_future_ball_tracking():
    shooter, ball = make_tracks()

    ball[3:] = np.nan

    assert audit.inspect_frame(shooter, ball, 2) == ("insufficient_future_ball_tracking")


@pytest.mark.parametrize("frame", [-1, 12, 100])
def test_out_of_range(frame):
    shooter, ball = make_tracks()

    assert audit.inspect_frame(shooter, ball, frame) == ("out_of_range")


def test_reject_negative_radius(tmp_path):
    with pytest.raises(ValueError, match="Radius"):
        audit.run(
            tmp_path / "audit.csv",
            tmp_path,
            tmp_path / "out.csv",
            radius=-1,
        )


def test_end_to_end_csv_audit(tmp_path, monkeypatch):
    """Verify JSON matching and written failure classifications."""
    audit_csv = tmp_path / "input.csv"
    json_root = tmp_path / "dataset"
    output = tmp_path / "output" / "per_clip.csv"

    video = "test/game/sample_1280x720.mp4"
    json_path = json_root / "test/game/sample_1280x720.json"
    json_path.parent.mkdir(parents=True)

    shooter = [[100.0, 100.0, 40.0, 100.0] for _ in range(12)]
    ball = [[400.0, 115.0, 10.0, 10.0] for _ in range(12)]

    json_path.write_text(
        json.dumps(
            {
                "shooter": {"trajectory": shooter},
                "ball": {"trajectory": ball},
            }
        ),
        encoding="utf-8",
    )

    with audit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_path",
                "human_status",
                "human_interval",
                "raw_inside_interval",
                "raw_candidate_count",
                "raw_nearest_distance",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_path": video,
                "human_status": "provisional",
                "human_interval": "2-3",
                "raw_inside_interval": "False",
                "raw_candidate_count": "0",
                "raw_nearest_distance": "",
            }
        )

    def fake_diagnose_clip(_clip):
        return {
            "shooter_key": "shooter",
            "status": "release_not_found",
        }

    monkeypatch.setattr(
        audit.detector,
        "diagnose_clip",
        fake_diagnose_clip,
    )

    audit.run(audit_csv, json_root, output, radius=3)

    assert output.exists()

    with output.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1

    row = rows[0]

    assert row["human_interval"] == "2-3"
    assert row["exact_outside_hand_reach"] == "2"
    assert row["exact_missing_ball"] == "0"
    assert row["exact_missing_shooter"] == "0"
    assert row["nearby_qualifying_candidates"] == "0"
    assert json.loads(row["exact_frame_reasons"]) == [
        "outside_hand_reach",
        "outside_hand_reach",
    ]
