"""Tests for the human-versus-candidate audit."""

import csv
import json

import pytest

from tools import audit_release_candidates_against_human as audit
from tools.audit_release_candidates_against_human import (
    interval_distance,
    nearest_distance,
    parse_interval,
)


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (30, 2),
        (31, 1),
        (32, 0),
        (33, 0),
        (34, 0),
        (35, 1),
        (38, 4),
    ],
)
def test_distance_to_inclusive_interval(frame, expected):
    assert interval_distance(frame, 32, 34) == expected


def test_parse_interval():
    assert parse_interval("32-34") == (32, 34)
    assert parse_interval("32–34") == (32, 34)
    assert parse_interval("82") == (82, 82)
    assert parse_interval("") is None


@pytest.mark.parametrize("value", ["34-32", "1-2-3", "abc"])
def test_invalid_interval(value):
    with pytest.raises(ValueError):
        parse_interval(value)


def test_nearest_candidate_overlaps_human_interval():
    assert nearest_distance([12, 33, 180], 32, 34) == 0


def test_all_candidates_removed():
    raw = [12, 32, 33, 70]
    retained = []
    removed = raw

    assert nearest_distance(raw, 32, 34) == 0
    assert nearest_distance(retained, 32, 34) == ""
    assert nearest_distance(removed, 32, 34) == 0


def test_removed_candidates_far_from_release():
    retained = [81, 82, 100]
    removed = [230, 231]

    assert nearest_distance(retained, 82, 83) == 0
    assert nearest_distance(removed, 82, 83) == 147


def test_audit_end_to_end_with_synthetic_data(tmp_path, monkeypatch):
    reviews = tmp_path / "reviews.csv"
    json_root = tmp_path / "dataset"
    output = tmp_path / "results" / "per_clip.csv"

    rows = []
    for i in range(20):
        video = f"test/game/clip_{i:02d}_1280x720.mp4"
        json_path = json_root / video.replace(".mp4", ".json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"case": i}), encoding="utf-8")

        rows.append(
            {
                "video_path": video,
                "manual_release_frame": "" if i == 19 else "32-34",
                "review_status": "uncertain" if i == 19 else "provisional",
            }
        )

    with reviews.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_path",
                "manual_release_frame",
                "review_status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    def fake_diagnose_clip(clip):
        case = clip["case"]

        if case == 0:
            # All candidates removed, including one in human interval.
            return {
                "status": "release_not_found",
                "candidate_frames": "[]",
                "removed_candidate_frames": "[12, 33, 70]",
            }

        if case == 1:
            # Retained candidate overlaps; removal is far away.
            return {
                "status": "single_candidate_episode",
                "candidate_frames": "[32, 33]",
                "removed_candidate_frames": "[180]",
            }

        return {
            "status": "single_candidate_episode",
            "candidate_frames": "[40]",
            "removed_candidate_frames": "[]",
        }

    monkeypatch.setattr(audit, "diagnose_clip", fake_diagnose_clip)

    audit.run(reviews, json_root, output)

    assert output.is_file()

    with output.open(encoding="utf-8", newline="") as handle:
        results = list(csv.DictReader(handle))

    assert len(results) == 20

    removed_all = results[0]
    assert removed_all["detector_status"] == "release_not_found"
    assert removed_all["raw_candidate_count"] == "3"
    assert removed_all["retained_candidate_count"] == "0"
    assert removed_all["removed_candidate_count"] == "3"
    assert removed_all["raw_nearest_distance"] == "0"
    assert removed_all["retained_nearest_distance"] == ""
    assert removed_all["filter_removed_nearer_candidate"] == "True"

    retained_overlap = results[1]
    assert retained_overlap["retained_inside_interval"] == "True"
    assert retained_overlap["removed_inside_interval"] == "False"
    assert retained_overlap["filter_removed_nearer_candidate"] == "False"

    uncertain = results[19]
    assert uncertain["human_status"] == "uncertain"
    assert uncertain["human_interval"] == ""
    assert uncertain["raw_nearest_distance"] == ""
    assert uncertain["filter_removed_nearer_candidate"] == ""
