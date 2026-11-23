"""Unit tests that do not load CoTracker or require any GPU/video files."""
import numpy as np
import pytest

from nba_shots.h1_motion_tracker import parse_box, seed_queries, summarize_tracks


def test_parse_box_accepts_missing_and_valid():
    assert parse_box("") is None
    assert parse_box(float("nan")) is None
    np.testing.assert_array_equal(parse_box("[1,2,8,9]"), [1, 2, 8, 9])


@pytest.mark.parametrize("value", ["bad", "[1,2,1,9]", "[1,2,3]", "[0,1,2,null]"])
def test_parse_box_rejects_bad(value):
    with pytest.raises(ValueError):
        parse_box(value)


def test_release_query_grid_is_inside_boxes():
    pts, labels = seed_queries({"shooter": np.array([10, 20, 30, 40]),
                                "defender": np.array([50, 60, 70, 80])}, 100, 100)
    assert pts.shape == (18, 2)
    assert labels.count("shooter") == 9
    assert labels.count("defender") == 9
    assert np.all((pts[:9, 0] > 10) & (pts[:9, 0] < 30))
    assert np.all((pts[9:, 1] > 60) & (pts[9:, 1] < 80))


def test_tracks_reject_mismatched_dimensions():
    with pytest.raises(ValueError, match="Visibility"):
        summarize_tracks(np.zeros((3, 2, 2)), np.ones((3, 3)), ["shooter"] * 2, 20, 20)


def test_motion_reports_visible_endpoint_displacement_and_missing_defender():
    tracks = np.array([[[0., 0.], [5., 5.]], [[3., 4.], [6., 6.]]])
    visibility = np.array([[True, True], [True, False]])
    result = summarize_tracks(tracks, visibility, ["shooter", "shooter"], 30, 40)
    assert result["cotracker_shooter_points"] == 2
    assert result["cotracker_shooter_endpoint_visible_points"] == 1
    assert result["cotracker_shooter_median_displacement_norm"] == pytest.approx(0.1)
    assert result["cotracker_shooter_median_step_norm"] == pytest.approx(0.1)
    assert result["cotracker_defender_points"] == 0
    assert np.isnan(result["cotracker_defender_median_displacement_norm"])


def test_no_visible_endpoints_remains_nan():
    tracks = np.zeros((2, 1, 2))
    result = summarize_tracks(tracks, np.array([[True], [False]]), ["shooter"], 100, 100)
    assert result["cotracker_shooter_endpoint_visible_points"] == 0
    assert np.isnan(result["cotracker_shooter_median_displacement_norm"])
