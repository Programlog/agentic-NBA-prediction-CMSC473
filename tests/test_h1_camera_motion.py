"""Synthetic smoke and regression tests; no CoTracker weights needed."""
import cv2
import numpy as np
import pytest

from nba_shots.h1_camera_motion import (
    background_mask,
    corrected_metrics,
    estimate_camera_translation,
)


def test_background_mask_excludes_players():
    boxes = {'shooter': np.array([50, 40, 100, 115])}
    mask = background_mask((160, 240), boxes, padding=12)
    assert mask[65, 65] == 0
    assert mask[5, 5] == 0
    assert mask[75, 180] == 255


def test_known_background_translation():
    rng = np.random.default_rng(21)
    base = np.zeros((180, 260), dtype=np.uint8)
    for x, y in zip(rng.integers(25, 235, 130), rng.integers(25, 155, 130)):
        cv2.circle(base, (int(x), int(y)), 2, 255, -1)
    frames = []
    for i in range(5):
        matrix = np.float32([[1, 0, 2*i], [0, 1, i]])
        image = cv2.warpAffine(base, matrix, (260, 180))
        frames.append(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    cumulative, meta = estimate_camera_translation(
        np.stack(frames), {'shooter': np.array([110., 50., 140., 130.])}, min_points=8
    )
    assert meta['camera_valid'], meta
    np.testing.assert_allclose(cumulative[0], (0, 0), atol=0.4)
    np.testing.assert_allclose(cumulative[-1], (8, 4), atol=1.5)


def test_blank_background_refuses_to_correct():
    video = np.zeros((4, 120, 180, 3), dtype=np.uint8)
    movement, meta = estimate_camera_translation(video, {})
    assert not meta['camera_valid']
    assert np.isnan(movement).all()


def test_corrected_and_raw_motion_are_separate():
    tracks = np.array([[[5., 5.]], [[8., 5.]], [[11., 5.]]])
    correction = np.array([[0., 0.], [3., 0.], [6., 0.]])
    v = np.ones((3, 1), dtype=bool)
    result = corrected_metrics(tracks, v, ['shooter'], 100, 100, correction, True)
    assert result['cotracker_shooter_median_displacement_norm'] > 0
    assert result['corrected_cotracker_shooter_median_displacement_norm'] == pytest.approx(0)
    assert result['defender_tracking_flag'] == 'no_annotation'


def test_bad_camera_estimate_is_missing_not_zero():
    tracks = np.array([[[5., 5.]], [[8., 5.]]])
    result = corrected_metrics(tracks, np.ones((2, 1), dtype=bool), ['shooter'],
                               100, 100, np.full((2, 2), np.nan), False)
    assert np.isnan(result['corrected_cotracker_shooter_median_displacement_norm'])
    assert result['cotracker_shooter_median_displacement_norm'] > 0
    assert result['shooter_tracking_flag'] == 'few_endpoint_points'


def test_rejects_invalid_video_dimensions():
    with pytest.raises(ValueError, match='Expected RGB'):
        estimate_camera_translation(np.zeros((3, 5, 5), dtype=np.uint8), {})
