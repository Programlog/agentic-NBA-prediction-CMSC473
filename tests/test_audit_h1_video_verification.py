"""Offline tests for the H1 video-verification quality gate."""

import csv
from copy import deepcopy

import pytest

from scripts.audit_h1_video_verification import (
    DOWNLOAD_COLUMNS,
    REVIEW_COLUMNS,
    audit,
    check_record,
    index_rows,
    optional_frame,
    parse_boolean,
    read_records,
)

SHOT_ID = "nba:0022500032:283"

VIDEO_PATH = "test/0022500032/bb91a081-0ed1-8c46-a8e9-8c95102fc685_1280x720.mp4"


def sample_records():
    download = {
        "shot_id": SHOT_ID,
        "game_id": "0022500032",
        "game_event_id": "283",
        "video_path": VIDEO_PATH,
        "download_status": "success",
        "file_size_bytes": "5572443",
        "error": "",
    }

    review = {
        "shot_id": SHOT_ID,
        "game_id": "0022500032",
        "game_event_id": "283",
        "video_path": VIDEO_PATH,
        "download_status": "success",
        "visual_review_status": "provisional_release_interval",
        "event_identity_verified": "false",
        "release_frame_start": "100",
        "release_frame_end": "102",
        "exact_release_frame": "",
        "shooter_box": "",
        "defender_box": "",
        "cv_training_eligible": "false",
        "notes": "Candidate release interval, not confirmed.",
    }

    return download, review


def test_valid_provisional_interval_remains_unverified():
    download, review = sample_records()

    result = check_record(download, review)

    assert result["issues"] == []
    assert result["status"] == "provisional_interval_unverified"
    assert result["training_eligible"] is False


def test_preliminary_record_remains_unverified():
    download, review = sample_records()
    review["visual_review_status"] = "preliminary"
    review["release_frame_start"] = ""
    review["release_frame_end"] = ""

    result = check_record(download, review)

    assert result["status"] == "preliminary_unverified"
    assert result["training_eligible"] is False


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("game_id", "wrong", "game_id mismatch"),
        ("game_event_id", "999", "game_event_id mismatch"),
        ("video_path", "wrong.mp4", "video_path mismatch"),
        ("download_status", "failed", "download_status mismatch"),
    ],
)
def test_rejects_cross_report_mismatch(field, value, expected):
    download, review = sample_records()
    review[field] = value

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert any(expected in issue for issue in result["issues"])


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("100", "", "only one endpoint"),
        ("", "102", "only one endpoint"),
        ("103", "100", "start exceeds end"),
        ("-1", "102", "negative release_frame_start"),
        ("abc", "102", "invalid release_frame_start"),
    ],
)
def test_rejects_bad_release_intervals(start, end, expected):
    download, review = sample_records()
    review["release_frame_start"] = start
    review["release_frame_end"] = end

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert any(expected in issue for issue in result["issues"])


def test_rejects_exact_release_outside_interval():
    download, review = sample_records()
    review["exact_release_frame"] = "110"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert any("outside candidate interval" in x for x in result["issues"])


def test_rejects_invalid_boolean():
    download, review = sample_records()
    review["cv_training_eligible"] = "maybe"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert result["training_eligible"] is False


def test_rejects_duplicate_shot_ids():
    download, _ = sample_records()

    with pytest.raises(ValueError, match="duplicate shot_id"):
        index_rows([download, deepcopy(download)], "downloads")


def test_rejects_false_download_success():
    download, review = sample_records()
    download["file_size_bytes"] = "0"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert any("nonpositive file size" in x for x in result["issues"])


def test_rejects_success_with_download_error():
    download, review = sample_records()
    download["error"] = "download failed"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"


def test_positive_identity_claim_without_evidence_is_invalid():
    download, review = sample_records()
    review["event_identity_verified"] = "true"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert result["training_eligible"] is False


def test_positive_eligibility_never_automatically_approved():
    download, review = sample_records()

    review.update(
        {
            "visual_review_status": "verified",
            "event_identity_verified": "true",
            "cv_training_eligible": "true",
            "exact_release_frame": "101",
            "shooter_box": "[389, 377, 496, 556]",
            "defender_box": "[485, 340, 658, 507]",
            "event_identity_evidence": "Reviewer assertion",
            "release_frame_evidence": "Reviewer assertion",
            "feature_quality_evidence": "Reviewer assertion",
        }
    )

    result = check_record(download, review)

    assert result["status"] == ("eligibility_claim_requires_independent_verification")
    assert result["training_eligible"] is False


def test_invalid_frame_and_boolean_helpers():
    assert optional_frame("", "frame", SHOT_ID) is None
    assert optional_frame("0", "frame", SHOT_ID) == 0
    assert parse_boolean("TRUE", "flag", SHOT_ID) is True

    with pytest.raises(ValueError):
        optional_frame("-1", "frame", SHOT_ID)

    with pytest.raises(ValueError):
        parse_boolean("", "flag", SHOT_ID)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted(columns))
        writer.writeheader()
        writer.writerows(rows)


def test_end_to_end_audit(tmp_path):
    download, review = sample_records()
    downloads_path = tmp_path / "downloads.csv"
    reviews_path = tmp_path / "reviews.csv"

    write_csv(downloads_path, [download], DOWNLOAD_COLUMNS)
    write_csv(reviews_path, [review], REVIEW_COLUMNS)

    summary = audit(downloads_path, reviews_path)

    assert summary["shots_audited"] == 1
    assert summary["invalid_records"] == 0
    assert summary["training_eligibility_claims"] == 0
    assert summary["claims_not_independently_approved"] == 0
    assert summary["independently_approved_training_clips"] == 0
    assert summary["independent_visual_verification_performed"] is False


def test_missing_review_is_invalid(tmp_path):
    download, review = sample_records()
    other_download = deepcopy(download)
    other_download["shot_id"] = "nba:0022500032:999"

    downloads_path = tmp_path / "downloads.csv"
    reviews_path = tmp_path / "reviews.csv"

    write_csv(
        downloads_path,
        [download, other_download],
        DOWNLOAD_COLUMNS,
    )
    write_csv(reviews_path, [review], REVIEW_COLUMNS)

    summary = audit(downloads_path, reviews_path)

    assert summary["shots_audited"] == 2
    assert summary["invalid_records"] == 1


def test_missing_required_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("shot_id\nexample\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        read_records(path, DOWNLOAD_COLUMNS)


def test_rejects_unknown_download_status():
    download, review = sample_records()
    download["download_status"] = "mystery"
    review["download_status"] = "mystery"

    result = check_record(download, review)

    assert result["status"] == "invalid_record"
    assert any("unknown download_status" in issue for issue in result["issues"])
    assert result["training_eligible"] is False


def test_failed_download_is_not_preliminary_video():
    download, review = sample_records()
    download["download_status"] = "failed"
    review["download_status"] = "failed"

    result = check_record(download, review)

    assert result["status"] == "download_unavailable"
    assert result["training_eligible"] is False


def test_positive_claim_counted_but_never_approved(tmp_path):
    download, review = sample_records()

    review.update(
        {
            "visual_review_status": "verified",
            "event_identity_verified": "true",
            "cv_training_eligible": "true",
            "exact_release_frame": "101",
            "shooter_box": "[389, 377, 496, 556]",
            "defender_box": "[485, 340, 658, 507]",
            "event_identity_evidence": "Reviewer assertion",
            "release_frame_evidence": "Reviewer assertion",
            "feature_quality_evidence": "Reviewer assertion",
        }
    )

    downloads_path = tmp_path / "downloads.csv"
    reviews_path = tmp_path / "reviews.csv"

    write_csv(downloads_path, [download], DOWNLOAD_COLUMNS)
    write_csv(
        reviews_path,
        [review],
        REVIEW_COLUMNS
        | {
            "event_identity_evidence",
            "release_frame_evidence",
            "feature_quality_evidence",
        },
    )

    summary = audit(downloads_path, reviews_path)

    assert summary["training_eligibility_claims"] == 1
    assert summary["claims_not_independently_approved"] == 1
    assert summary["independently_approved_training_clips"] == 0
    assert summary["invalid_records"] == 0
