"""Fast synthetic tests; never download CoTracker weights."""
import cv2
import numpy as np
import pytest
from nba_shots.h1_motion_tracker import seed_queries
from nba_shots.h1_track_visualizer import prepare_queries, render_frame, save_video


def test_queries_are_identical_to_motion_extractor():
    box = {"shooter": np.array([10, 20, 30, 40]), "defender": np.array([50, 10, 70, 35])}
    queries, labels, scaled = prepare_queries(box, (100, 100), (200, 200), 13)
    expected, expected_labels = seed_queries(scaled, 100, 100)
    np.testing.assert_array_equal(queries[:, 1:], expected)
    assert labels == expected_labels
    assert queries.shape == (18, 3)
    assert np.all(queries[:, 0] == 12)


def test_visualizer_uses_release_box_only():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    tracks = np.array([[[20, 20]], [[22, 22]]], dtype=float)
    visibility = np.array([[True], [True]])
    box = {"shooter": np.array([10, 10, 40, 50])}
    earlier = render_frame(img, tracks, visibility, ["shooter"], 0, 100, box)
    last = render_frame(img, tracks, visibility, ["shooter"], 1, 100, box)
    # At (10,40), only the release-frame annotation box draws green.
    assert not np.any(earlier[40, 10])
    assert last[40, 10, 1] > 0


def test_invisible_points_not_drawn():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    xy = np.array([[[40, 40]], [[60, 60]]], dtype=float)
    shown = render_frame(img, xy, np.array([[True], [False]]), ["shooter"], 1, 5, {}, trail=3)
    assert not shown[60, 60].any()


def test_video_writer_and_frame_count(tmp_path):
    video = np.zeros((3, 64, 96, 3), dtype=np.uint8)
    xy = np.array([[[30, 30]], [[31, 30]], [[32, 30]]], dtype=float)
    vis = np.ones((3, 1), dtype=bool)
    path = tmp_path / "audit.mp4"
    save_video(path, video, xy, vis, ["shooter"], 12, {}, trail=2)
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    capture.release()


def test_bad_track_shapes_rejected():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Mismatched"):
        render_frame(img, np.zeros((2, 1, 2)), np.zeros((3, 1)), ["shooter"], 0, 0, {})
