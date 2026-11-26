import json

import numpy as np
import pytest

import nba_shots.release_diagnostics as diagnostics
from nba_shots.basketevent import release_frame
from nba_shots.release_diagnostics import (
    ReleaseDetectorMismatchError,
    candidate_episodes,
    diagnose_clip,
    release_candidates,
    scan_directory,
    write_report,
)


def tracks(second_release=False):
    n = 45
    shooter = np.tile([0.0, 0.0, 100.0, 100.0], (n, 1))
    ball = np.tile([45.0, 45.0, 10.0, 10.0], (n, 1))

    # The ball rises after frame 10.
    for k in range(11, 20):
        ball[k, 1] = 45 - 12 * (k - 10)

    if second_release:
        # A separate later event also satisfies the geometry.
        for k in range(26, 35):
            ball[k, 1] = 45 - 12 * (k - 25)

    return shooter, ball


def make_clip(shooter, ball):
    return {
        "player_0": {
            "event": {"actionType": "Made Shot"},
            "trajectory": shooter.tolist(),
        },
        "ball": {"trajectory": ball.tolist()},
    }


def test_candidates_reproduce_frozen_detector():
    shooter, ball = tracks()
    candidates = release_candidates(shooter, ball)

    assert candidates
    assert candidates[-1] == release_frame(shooter, ball)


def test_separate_release_episodes_are_reported():
    shooter, ball = tracks(second_release=True)
    result = diagnose_clip(make_clip(shooter, ball))

    assert result["status"] == "multiple_candidate_episodes"
    assert result["episode_count"] >= 2
    assert result["selected_release_frame"] == release_frame(shooter, ball)


def test_no_upward_motion_does_not_create_release():
    shooter, ball = tracks()
    ball[:] = [45.0, 45.0, 10.0, 10.0]

    result = diagnose_clip(make_clip(shooter, ball))

    assert result["status"] == "release_not_found"
    assert result["candidate_count"] == 0


def test_ambiguous_shooter_is_not_assumed():
    shooter, ball = tracks()
    clip = make_clip(shooter, ball)
    clip["player_1"] = clip["player_0"].copy()

    result = diagnose_clip(clip)

    assert result["status"] == "ambiguous_shot_event"
    assert result["selected_release_frame"] == ""


def test_candidate_episodes_group_consecutive_frames():
    assert candidate_episodes([2, 3, 4, 12, 13]) == [
        [2, 3, 4],
        [12, 13],
    ]


def test_directory_scan_and_no_overwrite(tmp_path):
    shooter, ball = tracks()
    source = tmp_path / "clips"
    source.mkdir()

    path = source / "example.json"
    path.write_text(
        json.dumps(make_clip(shooter, ball)),
        encoding="utf-8",
    )

    results = scan_directory(source)
    assert len(results) == 1
    assert results[0]["clip_path"] == "example.json"

    output = tmp_path / "report.csv"
    write_report(results, output)
    assert output.exists()

    with pytest.raises(FileExistsError):
        write_report(results, output)


def test_gap_tolerance_merges_small_interruptions():
    frames = [120, 121, 122, 127, 129, 131, 132]

    raw = candidate_episodes(frames)
    tolerant = candidate_episodes(frames, max_missing_frames=2)

    assert len(raw) == 4
    assert len(tolerant) == 2


def test_large_separation_remains_distinct():
    frames = [88, 113, 114, 115, 116]

    tolerant = candidate_episodes(frames, max_missing_frames=2)

    assert tolerant == [[88], [113, 114, 115, 116]]


def test_invalid_candidate_sequence_rejected():
    with pytest.raises(ValueError, match="sorted and unique"):
        candidate_episodes([10, 8, 12])

    with pytest.raises(ValueError, match="nonnegative"):
        candidate_episodes([10, 12], max_missing_frames=-1)


def test_diagnostic_reports_candidate_gap_measurements():
    shooter, ball = tracks(second_release=True)
    result = diagnose_clip(make_clip(shooter, ball))

    candidates = json.loads(result["candidate_frames"])

    assert result["candidate_count"] == len(candidates)
    assert result["episode_count"] == len(candidate_episodes(candidates))
    assert result["gap_tolerant_episode_count"] == len(
        candidate_episodes(candidates, max_missing_frames=2)
    )

    expected_gap = max(b - a for a, b in zip(candidates, candidates[1:]))

    assert result["largest_candidate_gap_frames"] == expected_gap
    assert result["widely_separated_candidates"] == (expected_gap >= 25)


def test_no_release_has_no_invented_gap():
    shooter, ball = tracks()
    ball[:] = [45.0, 45.0, 10.0, 10.0]

    result = diagnose_clip(make_clip(shooter, ball))

    assert result["status"] == "release_not_found"
    assert result["gap_tolerant_episode_count"] == 0
    assert result["largest_candidate_gap_frames"] == ""
    assert result["widely_separated_candidates"] is False


def test_detector_mismatch_raises_explicit_error(monkeypatch):
    shooter, ball = tracks()
    clip = make_clip(shooter, ball)

    # Simulate the frozen detector disagreeing with candidate enumeration.
    monkeypatch.setattr(diagnostics, "release_frame", lambda *_: None)

    with pytest.raises(
        ReleaseDetectorMismatchError,
        match="Candidate scan disagrees",
    ):
        diagnose_clip(clip)


def test_directory_scan_does_not_hide_detector_mismatch(tmp_path, monkeypatch):
    shooter, ball = tracks()

    source = tmp_path / "clips"
    source.mkdir()
    (source / "shot.json").write_text(
        json.dumps(make_clip(shooter, ball)),
        encoding="utf-8",
    )

    monkeypatch.setattr(diagnostics, "release_frame", lambda *_: None)

    with pytest.raises(
        ReleaseDetectorMismatchError,
        match="Candidate scan disagrees",
    ):
        scan_directory(source)


def test_directory_scan_still_records_invalid_json(tmp_path):
    source = tmp_path / "clips"
    source.mkdir()
    (source / "broken.json").write_text(
        "{ invalid json",
        encoding="utf-8",
    )

    results = scan_directory(source)

    assert len(results) == 1
    assert results[0]["status"] == "invalid_json_or_trajectory"
    assert "error" in results[0]
