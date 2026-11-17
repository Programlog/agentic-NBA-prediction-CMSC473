import gzip
import json

import numpy as np
import pandas as pd
import pytest

from nba_shots.basketevent import (
    FEATURE_COLUMNS,
    clip_rows,
    game_rows,
    match_events,
    prepare_basketevent,
)
from nba_shots.common import BOX_FEATURES, load_config
from nba_shots.data import assign_splits, normalize_kaggle

FRAMES, PAN = 90, 2.0  # the camera pans 2 px right per frame; every box drifts with it
EVENT = {"clock": "PT07M40.00S", "period": 4, "description": "Kolek 18' Jump Shot (7 PTS)"}


def track(feet_at, w=50, h=150):
    """Boxes [x, y, w, h] whose bottom centre (feet) follows feet_at(frame) plus the pan."""
    return [[feet_at(k)[0] + PAN * k - w / 2, feet_at(k)[1] - h, w, h] for k in range(FRAMES)]


def ball_y(k):
    """Held at chest height until frame 60, then shot upward 40 px per frame."""
    return 380.0 if k <= 60 else 380.0 - 40 * (k - 60)


def ball_x(k):
    """Drifts right toward the basket once released."""
    return 600.0 + max(0, k - 61) * 5


def synthetic_clip(action="Made Shot", ball=True):
    clip = {
        "player_0": {
            "jersey_color": "white",
            "event": {**EVENT, "actionType": action},
            "trajectory": track(lambda k: (600, 500)),
        },
        # The nearest defender closes from 1.5 to 1.0 box heights in the 5 frames before release.
        "player_1": {
            "jersey_color": "blue",
            "trajectory": track(lambda k: (600 + max(150, 150 + 15 * (61 - k)), 500)),
        },
        "player_2": {"jersey_color": "blue", "trajectory": track(lambda k: (1100, 400))},
        "player_3": {"jersey_color": "white", "trajectory": track(lambda k: (300, 500))},
        "player_4": {"jersey_color": "white", "trajectory": track(lambda k: (200, 450))},
    }
    if ball:
        clip["ball"] = {
            "trajectory": [[ball_x(k) + PAN * k - 5, ball_y(k) - 5, 10, 10] for k in range(FRAMES)]
        }
    return clip


def test_release_and_features_from_boxes():
    (row,) = clip_rows("test/0022500003/a.json", synthetic_clip())
    assert row["box_status"] == "ok"
    # Frame 61 is the last with the ball inside the widened shooter box before it climbs.
    assert row["release_frame"] == 61
    assert row["release_height_bh"] == pytest.approx((500 - 340) / 150)
    assert row["jump_bh"] == pytest.approx(0)
    assert row["ball_rise_bhps"] == pytest.approx((340 - 180) / 150 * 25 / 4)
    assert row["defender_distance_bh"] == pytest.approx(1.0)
    assert row["defenders_within_1bh"] == 1
    assert row["defender_closing_bhps"] == pytest.approx(0.5 * 25 / 5)
    # The ball flies right, toward the defender standing on the shooter's right.
    assert row["defender_in_front"] == 1
    assert row["defender_hand_up"] == pytest.approx(1.0)
    # The pan moves every box equally, so the stationary shooter has no relative speed.
    assert row["shooter_speed_bhps"] == pytest.approx(0, abs=1e-9)
    assert row["shooter_box"] == "[697.0, 350.0, 747.0, 500.0]"


def test_features_ignore_outcome_and_keep_failures():
    made, missed = (
        clip_rows("a", synthetic_clip("Made Shot")),
        clip_rows("a", synthetic_clip("Missed Shot")),
    )
    assert {k: v for k, v in made[0].items() if k != "action"} == pytest.approx(
        {k: v for k, v in missed[0].items() if k != "action"}, nan_ok=True
    )
    assert clip_rows("a", synthetic_clip(ball=False))[0]["box_status"] == "no_ball"
    still = synthetic_clip()
    still["ball"]["trajectory"] = [[600 + PAN * k - 5, 375, 10, 10] for k in range(FRAMES)]
    assert clip_rows("a", still)[0]["box_status"] == "release_not_found"
    twice = synthetic_clip()
    twice["player_3"]["event"] = {**EVENT, "actionType": "Made Shot"}
    assert clip_rows("a", twice)[0]["box_status"] == "ambiguous_shooter"
    twice["player_3"]["event"]["clock"] = "PT07M10.00S"
    rows = clip_rows("a", twice)
    assert [r["box_status"] for r in rows] == ["multiple_shot_events"] * 2


def plays(*rows):
    frame = pd.DataFrame(rows, columns=["actionNumber", "isFieldGoal", *EVENT])
    return frame.assign(description=frame.description + " ")  # NBA text can carry spaces


def test_events_match_only_one_field_goal():
    events = pd.DataFrame([EVENT, {**EVENT, "clock": "PT01M00.00S"}, {**EVENT, "period": 1}])
    nba = plays(
        (412, 1, *EVENT.values()),
        (500, 1, "PT01M00.00S", 4, EVENT["description"]),
        (501, 1, "PT01M00.00S", 4, EVENT["description"]),
        (600, 0, "PT07M40.00S", 1, EVENT["description"]),  # not a field goal
    )
    assert match_events(events, nba).tolist()[0] == 412
    assert match_events(events, nba).iloc[1:].isna().all()


def test_shot_in_two_clips_keeps_the_clip_with_a_release():
    broken = synthetic_clip()
    broken["ball"]["trajectory"] = [[0, 0, 10, 10]] * FRAMES
    rows = game_rows(
        {"z.json": synthetic_clip(), "a.json": broken}, plays((412, 1, *EVENT.values()))
    )
    assert rows[["clip", "box_status", "game_event_id", "duplicate_clips"]].values.tolist() == [
        ["z.json", "ok", 412, 1]
    ]


def test_box_table_trains_a_separate_xgboost(raw_shots, tmp_path):
    from nba_shots.models import train_baselines

    frame, _ = normalize_kaggle(raw_shots)
    frame, _ = assign_splits(frame)
    shots_path = tmp_path / "shots.parquet"
    frame.to_parquet(shots_path)
    rng = np.random.default_rng(0)
    table = pd.DataFrame(
        {"shot_id": frame.shot_id, **{n: rng.normal(size=len(frame)) for n in BOX_FEATURES}}
    )
    table_path = tmp_path / "boxes.parquet"
    table.to_parquet(table_path)
    cfg = load_config()
    cfg["xgboost"]["n_estimators"] = 2
    train_baselines(shots_path, tmp_path / "run", cfg, table_path)
    predictions = pd.read_csv(tmp_path / "run" / "validation_predictions.csv")
    sets = predictions.groupby("model").shot_id.apply(set)
    assert sets["xgboost"] == sets["xgboost_boxes"] and "xgboost_cv" not in sets
    assert sorted(BOX_FEATURES) == FEATURE_COLUMNS
    table[["shot_id"]].to_parquet(table_path)
    with pytest.raises(ValueError, match="exactly one feature set"):
        train_baselines(shots_path, tmp_path / "bad-run", cfg, table_path)


def test_depth_gaps_count_more_than_sideways_gaps():
    clip = synthetic_clip()
    # Same on-screen gap as the sideways defender, but toward the camera: it counts double.
    clip["player_1"]["trajectory"] = track(lambda k: (600, 500 + 150))
    (row,) = clip_rows("a", clip)
    assert row["defender_distance_bh"] == pytest.approx(2.0)
    assert row["defenders_within_1bh"] == 0


def test_prepare_keeps_only_shots_with_a_release(tmp_path):
    games = tmp_path / "labels" / "games"
    games.mkdir(parents=True)
    two_shots = synthetic_clip("Missed Shot")
    two_shots["player_3"]["event"] = {**EVENT, "clock": "PT07M10.00S", "actionType": "Made Shot"}
    for day in range(1, 7):
        game = f"00225000{day:02d}"
        clips = {f"test/{game}/a.json": synthetic_clip()}
        plays = [(412, 1, *EVENT.values())]
        if day == 1:
            clips[f"test/{game}/b.json"] = two_shots
            plays += [(500, 1, "PT07M10.00S", 4, EVENT["description"])]
        with gzip.open(games / f"{game}_labels.json.gz", "wt") as stream:
            json.dump(clips, stream)
        nba = pd.DataFrame(plays, columns=["actionNumber", "isFieldGoal", *EVENT])
        nba.to_csv(games / f"{game}_plays.csv", index=False)
        shots = [
            {
                "GAME_ID": game,
                "GAME_EVENT_ID": number,
                "GAME_DATE": f"202510{20 + day}",
                "PLAYER_ID": 1,
                "TEAM_ID": 1610612737,
                "HTM": "ATL",
                "SHOT_MADE_FLAG": int(number == 412),
                "SHOT_DISTANCE": 18,
                "PERIOD": 4,
                "MINUTES_REMAINING": 7,
                "SECONDS_REMAINING": 40,
                "SHOT_TYPE": "2PT Field Goal",
            }
            for number, *_ in plays
        ]
        pd.DataFrame(shots).to_csv(games / f"{game}_shots.csv", index=False)
        (games / f"{game}.json").write_text(json.dumps({"split": "test"}))
    (tmp_path / "labels" / "source.json").write_text(json.dumps({"videos": []}))
    prepare_basketevent(tmp_path / "labels", tmp_path / "cohort", load_config())
    cohort = pd.read_parquet(tmp_path / "cohort" / "shots.parquet")
    features = pd.read_parquet(tmp_path / "cohort" / "box_features.parquet")
    audit = json.loads((tmp_path / "cohort" / "audit.json").read_text())
    # The two-shot clip (a miss and its putback) is reported but not modeled.
    assert len(cohort) == 6 and set(cohort.shot_id) == set(features.shot_id)
    assert features[FEATURE_COLUMNS].notna().all().all()
    assert audit["status_counts"] == {"ok": 6, "multiple_shot_events": 1}
