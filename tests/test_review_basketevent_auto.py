import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_basketevent_auto import (parse_candidates, ready_to_save, review_indices,
                                     resolve_clip, REVIEWED, REJECTED, VERIFIED)


def test_candidate_parsing():
    assert parse_candidates('[80, 72, 80, -1, 72.2, "14"]') == [80, 72, 14]
    assert parse_candidates(None) == []
    assert parse_candidates(float('nan')) == []
    assert parse_candidates('bad') == []


def test_review_sampling_skips_manual_verified_and_reviewed():
    shots = pd.DataFrame({
        'shot_id': ['nba:0022500004:1', 'nba:0022500004:2',
                    'nba:0022500004:3', 'nba:0022500004:4',
                    'nba:0022500032:5', 'nba:0022500032:6',
                    'nba:0022500032:7'],
        'annotation_status': [VERIFIED, 'auto_proposed_review_required',
                              'review_required_no_release_candidate', 'auto_proposed_review_required',
                              REVIEWED, 'auto_proposed_review_required', REJECTED],
    })
    assert review_indices(shots, per_game=2) == [1, 2, 5]
    assert review_indices(shots, per_game=None) == [1, 2, 3, 5]
    assert review_indices(shots, per_game=1, start_index=2) == [2, 5]


def test_save_requires_explicit_release_hand_box():
    assert ready_to_save(76, 'right', [1004, 206, 1073, 446])
    assert not ready_to_save(None, 'right', [1, 2, 3, 4])
    assert not ready_to_save(76, None, [1, 2, 3, 4])
    assert not ready_to_save(76, 'right', None)


def test_original_manifest_relative_paths(tmp_path):
    original_manifest = tmp_path / 'artifacts' / 'h1_multigame_video_v1' / 'annotated_smoke.csv'
    clip = resolve_clip('../../data/basketevent/test/0022500004/example.mp4', original_manifest)
    assert clip == tmp_path / 'data' / 'basketevent' / 'test' / '0022500004' / 'example.mp4'
