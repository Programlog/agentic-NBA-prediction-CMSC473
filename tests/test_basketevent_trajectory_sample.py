import csv
import hashlib
import json

import pytest

from tools import audit_basketevent_trajectory_sample as audit


def test_sampling_is_deterministic():
    paths = [f"train/game/clip_{i:03d}.json" for i in range(20)] + [
        f"test/game/clip_{i:03d}.json" for i in range(20)
    ]

    first = audit.select_sample(paths, ["train", "test"], 5, 473)
    second = audit.select_sample(list(reversed(paths)), ["train", "test"], 5, 473)

    assert first == second
    assert len(first) == 10
    assert len(set(first)) == 10


def test_sampling_respects_splits():
    paths = [
        "train/game/a.json",
        "train/game/b.json",
        "test/game/c.json",
        "test/game/d.json",
        "valid/game/e.json",
        "train/game/video.mp4",
    ]

    selected = audit.select_sample(paths, ["train", "test"], 2, 473)

    assert set(selected) == {
        "train/game/a.json",
        "train/game/b.json",
        "test/game/c.json",
        "test/game/d.json",
    }


def test_sampling_rejects_insufficient_files():
    with pytest.raises(ValueError, match="requested 2 files"):
        audit.select_sample(
            ["train/game/a.json"],
            ["train"],
            2,
            473,
        )


def test_summary_uses_eligible_shots_as_denominator():
    records = [
        {
            "split": "train",
            "status": "multiple_candidate_episodes",
            "eligible": True,
            "raw_geometry_candidate_count": 5,
            "candidate_count": 3,
            "candidates_removed_by_steady": 2,
        },
        {
            "split": "train",
            "status": "release_not_found",
            "eligible": True,
            "raw_geometry_candidate_count": 4,
            "candidate_count": 0,
            "candidates_removed_by_steady": 4,
        },
        {
            "split": "train",
            "status": "no_shot_event",
            "eligible": False,
            "raw_geometry_candidate_count": 0,
            "candidate_count": 0,
            "candidates_removed_by_steady": 0,
        },
    ]

    result = audit.summarize(records, ["train"])[0]

    assert result["sampled_files"] == 3
    assert result["eligible_shot_clips"] == 2
    assert result["affected_clips"] == 2
    assert result["candidate_frames_removed"] == 6
    assert result["complete_candidate_loss_clips"] == 1
    assert result["no_shot_event"] == 1


def test_missing_file_fails_without_downloading(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing"):
        audit.audit_sample(
            ["train/game/missing.json"],
            tmp_path,
            audit.REVISION,
            download=False,
        )


def test_local_audit_uses_existing_detector(tmp_path, monkeypatch):
    path = tmp_path / "train" / "game" / "clip.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"example": "trajectory"}),
        encoding="utf-8",
    )

    observed = []

    def fake_diagnose(clip):
        observed.append(clip)
        return {
            "status": "single_candidate_episode",
            "raw_geometry_candidate_count": 3,
            "candidate_count": 2,
            "candidates_removed_by_steady": 1,
        }

    monkeypatch.setattr(audit, "diagnose_clip", fake_diagnose)

    records = audit.audit_sample(
        ["train/game/clip.json"],
        tmp_path,
        audit.REVISION,
        download=False,
    )

    assert observed == [{"example": "trajectory"}]
    assert len(records) == 1
    assert records[0]["eligible"] is True
    assert records[0]["candidates_removed_by_steady"] == 1


def test_csv_refuses_to_overwrite(tmp_path):
    path = tmp_path / "report.csv"
    fields = ["clip_path"]
    rows = [{"clip_path": "train/game/a.json"}]

    audit.write_csv(path, fields, rows)

    with pytest.raises(FileExistsError):
        audit.write_csv(path, fields, rows)

    with path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == rows


def test_audit_records_sha256_of_local_json(tmp_path, monkeypatch):
    path = tmp_path / "train" / "game" / "clip.json"
    path.parent.mkdir(parents=True)

    content = '{"example": "trajectory"}'
    path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(
        audit,
        "diagnose_clip",
        lambda clip: {"status": "no_shot_event"},
    )

    records = audit.audit_sample(
        ["train/game/clip.json"],
        tmp_path,
        audit.REVISION,
        download=False,
    )

    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert records[0]["sha256"] == expected
