import json

import pytest

from tools import compare_release_intervals as audit
from tools.annotate_release_review import normalize_row
from tools.review_release_event_identity import extract_event_identity


def sample_reviews():
    annotations = {
        "test/game/a.mp4": normalize_row(
            {
                "video_path": "test/game/a.mp4",
                "manual_release_frame": "119-120",
                "reviewer_notes": "Observed ball leaving hands.",
            }
        ),
        "test/game/b.mp4": normalize_row(
            {
                "video_path": "test/game/b.mp4",
            }
        ),
    }

    identities = {}
    for path in annotations:
        identity = extract_event_identity(
            {
                "player_1": {
                    "event": {
                        "actionType": "Missed Shot",
                        "playerName": "Example",
                    }
                }
            }
        )
        identity["video_path"] = path
        identities[path] = identity

    return annotations, identities


def confirm_identity(row):
    row["identity_decision"] = "confirmed"
    row["observed_shooter"] = "Example"
    row["observed_action"] = "Shot attempt"
    row["identity_notes"] = "Matched the labeled shot in video."


def test_interval_error_inside():
    assert audit.interval_error("119-120", 119) == 0
    assert audit.interval_error("119-120", 120) == 0


def test_interval_error_before_and_after():
    assert audit.interval_error("119-120", 110) == 9
    assert audit.interval_error("119-120", 125) == 5


def test_single_frame_interval():
    assert audit.interval_error("32", 32) == 0
    assert audit.interval_error("32", 35) == 3


def test_readiness_never_exposes_detector_predictions():
    annotations, identities = sample_reviews()
    rows = audit.make_readiness(annotations, identities)

    assert len(rows) == 2
    assert rows[0]["reason"] == "identity_review_pending"
    assert rows[1]["reason"] == "manual_review_pending"
    assert "detector_release_frame" not in rows[0]


def test_comparison_blocked_when_review_incomplete(tmp_path):
    annotations, identities = sample_reviews()

    with pytest.raises(ValueError, match="Blind review incomplete"):
        audit.compare(annotations, identities, tmp_path)


def test_review_completion_still_requires_identity_decisions(tmp_path):
    annotations, identities = sample_reviews()
    annotations["test/game/b.mp4"] = normalize_row(
        {
            "video_path": "test/game/b.mp4",
            "review_status": "uncertain",
            "reviewer_notes": "Shooter obscured.",
        }
    )

    with pytest.raises(ValueError, match="2 pending identity"):
        audit.compare(annotations, identities, tmp_path)


def test_pending_identity_can_become_comparison_eligible():
    annotations, identities = sample_reviews()
    confirm_identity(identities["test/game/a.mp4"])

    rows = audit.make_readiness(annotations, identities)

    assert rows[0]["ready_for_comparison"] == "true"
    assert rows[1]["ready_for_comparison"] == "false"


def test_confirmed_identity_uses_frozen_detector(tmp_path, monkeypatch):
    annotations, identities = sample_reviews()
    annotations.pop("test/game/b.mp4")
    identities.pop("test/game/b.mp4")

    confirm_identity(identities["test/game/a.mp4"])

    path = tmp_path / "test" / "game" / "a.json"
    path.parent.mkdir(parents=True)

    clip = {
        "player_1": {
            "event": {
                "actionType": "Missed Shot",
                "playerName": "Example",
            }
        }
    }
    path.write_text(json.dumps(clip), encoding="utf-8")

    monkeypatch.setattr(
        audit,
        "diagnose_clip",
        lambda _: {"selected_release_frame": 123},
    )

    rows = audit.compare(annotations, identities, tmp_path)

    assert rows[0]["interval_error_frames"] == 3
    assert rows[0]["comparison_status"] == "outside_provisional_interval"


def test_missing_detector_release_is_distinct(tmp_path, monkeypatch):
    annotations, identities = sample_reviews()
    annotations.pop("test/game/b.mp4")
    identities.pop("test/game/b.mp4")

    confirm_identity(identities["test/game/a.mp4"])

    path = tmp_path / "test" / "game" / "a.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "player_1": {
                    "event": {
                        "actionType": "Missed Shot",
                        "playerName": "Example",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        audit,
        "diagnose_clip",
        lambda _: {"selected_release_frame": ""},
    )

    rows = audit.compare(annotations, identities, tmp_path)

    assert rows[0]["comparison_status"] == "detector_no_release"
    assert rows[0]["interval_error_frames"] == ""


def test_identity_mismatch_is_excluded(tmp_path):
    annotations, identities = sample_reviews()
    annotations.pop("test/game/b.mp4")
    identities.pop("test/game/b.mp4")

    identity = identities["test/game/a.mp4"]
    identity["identity_decision"] = "mismatch"
    identity["identity_notes"] = "Video shows another shot."

    rows = audit.compare(annotations, identities, tmp_path)

    assert rows[0]["comparison_status"] == "excluded_unverifiable"
    assert rows[0]["detector_release_frame"] == ""


def test_event_metadata_changes_are_rejected(tmp_path):
    annotations, identities = sample_reviews()
    annotations.pop("test/game/b.mp4")
    identities.pop("test/game/b.mp4")
    confirm_identity(identities["test/game/a.mp4"])

    path = tmp_path / "test" / "game" / "a.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "player_1": {
                    "event": {
                        "actionType": "Made Shot",
                        "playerName": "Example",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Event metadata changed"):
        audit.compare(annotations, identities, tmp_path)


def test_duplicate_rows_rejected():
    rows = [
        {"video_path": "test/a.mp4"},
        {"video_path": r"test\a.mp4"},
    ]

    with pytest.raises(ValueError, match="Duplicate annotation"):
        audit.index_rows(rows, "annotation")


def test_report_refuses_overwrite(tmp_path):
    path = tmp_path / "readiness.csv"
    rows = [{"video_path": "test/a.mp4"}]

    audit.write_new_csv(path, ["video_path"], rows)

    with pytest.raises(FileExistsError):
        audit.write_new_csv(path, ["video_path"], rows)
