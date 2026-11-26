import csv
import json

import pytest

from tools.review_release_event_identity import (
    FIELDS,
    build_worksheet,
    extract_event_identity,
    validate_identity_rows,
    write_worksheet,
)


def example_clip():
    return {
        "player_2": {
            "jersey_color": "white",
            "event": {
                "playerName": "Bridges",
                "teamTricode": "NYK",
                "actionType": "Missed Shot",
                "subType": "Running Reverse Layup Shot",
                "description": "MISS Bridges 1' Running Reverse Layup",
                "period": 2,
                "clock": "PT06M40.00S",
            },
        },
        "player_3": {
            "event": {
                "playerName": "Wade",
                "actionType": "Rebound",
                "clock": "PT06M38.00S",
            },
        },
    }


def test_extracts_unique_shot_and_shooter():
    row = extract_event_identity(example_clip())

    assert row["event_status"] == "one_shot_event"
    assert row["shooter_key"] == "player_2"
    assert row["player_name"] == "Bridges"
    assert row["team"] == "NYK"
    assert row["action_type"] == "Missed Shot"
    assert row["identity_decision"] == "pending"


def test_retains_other_events_as_context():
    row = extract_event_identity(example_clip())
    other = json.loads(row["other_events"])

    assert len(other) == 1
    assert other[0]["player"] == "Wade"
    assert other[0]["action"] == "Rebound"


def test_no_shot_event_is_not_confirmed():
    row = extract_event_identity({"player_0": {"event": {"actionType": "steal"}}})

    assert row["event_status"] == "no_shot_event"
    assert row["shooter_key"] == ""


def test_multiple_shots_are_flagged():
    clip = example_clip()
    clip["player_4"] = {
        "event": {
            "actionType": "Made Shot",
            "playerName": "Another Player",
        }
    }

    row = extract_event_identity(clip)

    assert row["event_status"] == "multiple_shot_events"
    assert row["shooter_key"] == ""


def test_build_worksheet_uses_matching_json(tmp_path):
    path = tmp_path / "test" / "game" / "clip.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(example_clip()), encoding="utf-8")

    rows = build_worksheet(
        [{"video_path": "test/game/clip.mp4"}],
        tmp_path,
    )

    assert len(rows) == 1
    assert rows[0]["player_name"] == "Bridges"
    assert rows[0]["video_path"] == "test/game/clip.mp4"


def test_missing_json_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_worksheet(
            [{"video_path": "test/game/missing.mp4"}],
            tmp_path,
        )


def test_duplicate_manifest_entries_fail(tmp_path):
    path = tmp_path / "test" / "game" / "clip.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(example_clip()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate video"):
        build_worksheet(
            [
                {"video_path": "test/game/clip.mp4"},
                {"video_path": r"test\game\clip.mp4"},
            ],
            tmp_path,
        )


def test_confirmation_requires_human_evidence():
    row = extract_event_identity(example_clip())
    row["video_path"] = "test/game/clip.mp4"
    row["identity_decision"] = "confirmed"

    with pytest.raises(ValueError, match="needs notes"):
        validate_identity_rows([row])

    row["identity_notes"] = "Observed matching layup."

    with pytest.raises(ValueError, match="observed shooter"):
        validate_identity_rows([row])

    row["observed_shooter"] = "Bridges"

    with pytest.raises(ValueError, match="observed action"):
        validate_identity_rows([row])

    row["observed_action"] = "Reverse layup"
    validate_identity_rows([row])


def test_ambiguous_event_cannot_be_confirmed():
    row = extract_event_identity({})
    row["video_path"] = "test/game/clip.mp4"
    row["identity_decision"] = "confirmed"
    row["identity_notes"] = "Looks like a shot."
    row["observed_shooter"] = "Unknown"
    row["observed_action"] = "Jump shot"

    with pytest.raises(ValueError, match="Cannot confirm"):
        validate_identity_rows([row])


def test_uncertain_review_can_be_recorded():
    row = extract_event_identity(example_clip())
    row["video_path"] = "test/game/clip.mp4"
    row["identity_decision"] = "uncertain"
    row["identity_notes"] = "Shooter partly obscured."

    validate_identity_rows([row])


def test_pending_review_cannot_contain_observations():
    row = extract_event_identity(example_clip())
    row["video_path"] = "test/game/clip.mp4"
    row["observed_shooter"] = "Bridges"

    with pytest.raises(ValueError, match="Pending"):
        validate_identity_rows([row])


def test_worksheet_refuses_overwrite(tmp_path):
    row = extract_event_identity(example_clip())
    row["video_path"] = "test/game/clip.mp4"
    path = tmp_path / "identity.csv"

    write_worksheet(path, [row])

    with pytest.raises(FileExistsError):
        write_worksheet(path, [row])

    with path.open(newline="", encoding="utf-8") as stream:
        loaded = list(csv.DictReader(stream))

    assert loaded[0]["player_name"] == "Bridges"
    assert list(loaded[0]) == FIELDS


def test_event_identity_module_help():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.review_release_event_identity",
            "--help",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--json-root" in result.stdout
    assert "--validate" in result.stdout
