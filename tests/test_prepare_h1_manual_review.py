"""Fast offline tests of the independent H1 manual-review worksheet."""

import csv

import pytest

from scripts.prepare_h1_manual_review import (
    FIELDS,
    SOURCE_FIELDS,
    inspect,
    prepare,
    validate,
)


def source_row(shot_id="nba:0022500032:283"):
    row = {key: "" for key in SOURCE_FIELDS}
    row.update(
        {
            "shot_id": shot_id,
            "game_id": "0022500032",
            "game_event_id": "283",
            "video_path": "test/0022500032/clip.mp4",
            "download_status": "success",
            "visual_review_status": "provisional_release_interval",
            "event_identity_verified": "false",
            "release_frame_start": "100",
            "release_frame_end": "102",
            "exact_release_frame": "",
            "cv_training_eligible": "false",
            "notes": "Not verified",
        }
    )
    return row


def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def worksheet():
    row = dict(source_row())
    row.update({key: "" for key in FIELDS if key not in SOURCE_FIELDS})
    row["identity_decision"] = "not_reviewed"
    row["timing_decision"] = "not_reviewed"
    return row


def test_default_unreviewed_is_not_approved():
    result = inspect(worksheet())
    assert result["review_status"] == "not_reviewed"
    assert result["training_eligible"] is False
    assert result["independently_verified"] is False


def test_provisional_source_frames_are_not_human_observations():
    row = worksheet()
    assert row["release_frame_start"] == "100"
    assert row["observed_release_start"] == ""
    assert inspect(row)["review_status"] == "not_reviewed"


def test_positive_claim_requires_structured_evidence():
    row = worksheet()
    row["identity_decision"] = "match"
    assert inspect(row)["review_status"] == "invalid_review"
    assert any("missing" in e for e in inspect(row)["issues"])


def test_complete_claim_still_not_approved():
    row = worksheet()
    row.update(
        {
            "identity_decision": "match",
            "timing_decision": "exact",
            "reviewer_id": "reviewer-one",
            "observed_shooter": "player #1",
            "observed_team": "SAS",
            "observed_period": "2",
            "observed_clock": "3:54",
            "observed_shot_type": "three point",
            "identity_evidence": "Manual player/jersey comparison",
            "observed_release_start": "101",
            "observed_release_end": "101",
            "timing_evidence": "Manual ball hand inspection",
        }
    )
    result = inspect(row)
    assert result["review_status"] == "human_claim_pending_independent_adjudication"
    assert result["training_eligible"] is False


@pytest.mark.parametrize("start,end", [("101", ""), ("102", "101"), ("-1", "2")])
def test_invalid_interval(start, end):
    row = worksheet()
    row["observed_release_start"] = start
    row["observed_release_end"] = end
    assert inspect(row)["review_status"] == "invalid_review"


def test_exact_must_have_equal_endpoints():
    row = worksheet()
    row.update(
        {
            "timing_decision": "exact",
            "observed_release_start": "100",
            "observed_release_end": "102",
            "reviewer_id": "a",
            "timing_evidence": "b",
        }
    )
    assert inspect(row)["review_status"] == "invalid_review"


def test_uncertain_identity_requires_note():
    row = worksheet()
    row["identity_decision"] = "uncertain"
    assert inspect(row)["review_status"] == "invalid_review"


def test_non_specific_timing_cannot_assert_frame():
    row = worksheet()
    row.update(
        {
            "timing_decision": "uncertain",
            "observed_release_start": "100",
            "observed_release_end": "101",
        }
    )
    assert inspect(row)["review_status"] == "invalid_review"


def test_prepare_retains_source_and_clears_user_decisions(tmp_path):
    path = tmp_path / "source.csv"
    write_csv(path, SOURCE_FIELDS, [source_row()])
    [row] = prepare(path)
    assert row["release_frame_start"] == "100"
    assert row["identity_decision"] == "not_reviewed"
    assert row["observed_release_start"] == ""


def test_validate_worksheet_unchanged(tmp_path):
    source = tmp_path / "source.csv"
    sheet = tmp_path / "sheet.csv"
    write_csv(source, SOURCE_FIELDS, [source_row()])
    write_csv(sheet, FIELDS, [worksheet()])
    result = validate(source, sheet)
    assert result["status_counts"] == {"not_reviewed": 1}
    assert result["approved_training_clips"] == 0


def test_validate_rejects_changed_source_columns(tmp_path):
    source = tmp_path / "source.csv"
    sheet = tmp_path / "sheet.csv"
    write_csv(source, SOURCE_FIELDS, [source_row()])
    row = worksheet()
    row["cv_training_eligible"] = "true"
    write_csv(sheet, FIELDS, [row])
    assert validate(source, sheet)["invalid_reviews"] == 1


def test_validate_rejects_missing_shot(tmp_path):
    source = tmp_path / "source.csv"
    sheet = tmp_path / "sheet.csv"
    write_csv(source, SOURCE_FIELDS, [source_row(), source_row("nba:0022500032:363")])
    write_csv(sheet, FIELDS, [worksheet()])
    assert validate(source, sheet)["invalid_reviews"] == 1


def test_validate_rejects_duplicate_shots(tmp_path):
    source = tmp_path / "source.csv"
    sheet = tmp_path / "sheet.csv"
    write_csv(source, SOURCE_FIELDS, [source_row()])
    write_csv(sheet, FIELDS, [worksheet(), worksheet()])
    with pytest.raises(ValueError, match="duplicate shot_id"):
        validate(source, sheet)
