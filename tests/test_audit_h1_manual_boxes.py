"""Tests for H1 manual bounding-box validation and measurements."""

import csv
import math

import pytest

from scripts.audit_h1_manual_boxes import (
    calculate_distances,
    validate_annotations,
)

SHOT_ID = "nba:0022500032:283"

ROLES = (
    "shooter_wembanyama_1",
    "candidate_defender_houston_12",
    "candidate_defender_headband",
)

COLUMNS = ("shot_id", "frame", "role", "x1", "y1", "x2", "y2")


def make_row(role, frame=90, **overrides):
    row = {
        "shot_id": SHOT_ID,
        "frame": frame,
        "role": role,
        "x1": 100,
        "y1": 100,
        "x2": 200,
        "y2": 200,
    }
    row.update(overrides)
    return row


def write_csv(tmp_path, rows):
    path = tmp_path / "boxes.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return path


def audit(tmp_path, rows):
    path = write_csv(tmp_path, rows)
    return validate_annotations([path], SHOT_ID, 1280, 720)


def test_valid_three_player_annotations(tmp_path):
    rows = [make_row(role) for role in ROLES]

    records, errors = audit(tmp_path, rows)

    assert errors == []
    assert len(records) == 3
    assert {record["role"] for record in records} == set(ROLES)


def test_rejects_invalid_horizontal_coordinates(tmp_path):
    rows = [make_row(ROLES[0], x1=300, x2=200)]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("invalid x coordinates" in error for error in errors)


def test_rejects_out_of_bounds_coordinates(tmp_path):
    rows = [make_row(ROLES[0], y2=721)]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("invalid y coordinates" in error for error in errors)


def test_rejects_duplicate_player_annotation(tmp_path):
    row = make_row(ROLES[0])

    records, errors = audit(tmp_path, [row, row.copy()])

    assert len(records) == 1
    assert any("duplicate annotation" in error for error in errors)


def test_rejects_incorrect_shot_id(tmp_path):
    rows = [make_row(ROLES[0], shot_id="nba:wrong:123")]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("unexpected shot ID" in error for error in errors)


def test_rejects_unknown_player_role(tmp_path):
    rows = [make_row("unidentified_player")]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("unknown role" in error for error in errors)


def test_rejects_negative_frame(tmp_path):
    rows = [make_row(ROLES[0], frame=-1)]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("negative frame index" in error for error in errors)


def test_rejects_invalid_numeric_value(tmp_path):
    rows = [make_row(ROLES[0], x1="not-a-number")]

    records, errors = audit(tmp_path, rows)

    assert records == []
    assert any("invalid numeric value" in error for error in errors)


def test_missing_defender_prevents_distance_calculation(tmp_path):
    rows = [make_row(ROLES[0]), make_row(ROLES[1])]

    records, validation_errors = audit(tmp_path, rows)
    distances, distance_errors = calculate_distances(records)

    assert validation_errors == []
    assert distances == []
    assert any("missing roles" in error for error in distance_errors)


def test_center_distance_uses_euclidean_pixels(tmp_path):
    rows = [
        make_row(ROLES[0], x1=0, y1=0, x2=20, y2=20),
        make_row(ROLES[1], x1=30, y1=40, x2=50, y2=60),
        make_row(ROLES[2], x1=10, y1=20, x2=30, y2=40),
    ]

    records, validation_errors = audit(tmp_path, rows)
    distances, distance_errors = calculate_distances(records)

    assert validation_errors == []
    assert distance_errors == []
    assert len(distances) == 2

    by_role = {result["defender_role"]: result["center_distance_pixels"] for result in distances}

    assert by_role[ROLES[1]] == pytest.approx(50.0)
    assert by_role[ROLES[2]] == pytest.approx(round(math.sqrt(500), 4))


def test_missing_required_columns(tmp_path):
    path = tmp_path / "missing_columns.csv"
    path.write_text("shot_id,frame,role\n", encoding="utf-8")

    records, errors = validate_annotations([path], SHOT_ID, 1280, 720)

    assert records == []
    assert any("missing required columns" in error for error in errors)


def test_rejects_frame_number_mismatch(tmp_path):
    """A frame-85 annotation file must not contain frame-90 records."""
    path = write_csv(
        tmp_path,
        [make_row(role, frame=90) for role in ROLES],
    )

    records, errors = validate_annotations(
        [path],
        SHOT_ID,
        1280,
        720,
        expected_frames={path: 85},
    )

    assert records == []
    assert len(errors) == 3
    assert all("frame mismatch" in error for error in errors)


def test_rejects_unexpected_extra_frame(tmp_path):
    """Valid frame-85 rows cannot hide extra frame-95 annotations."""
    rows = [make_row(role, frame=85) for role in ROLES]
    rows.append(make_row(ROLES[0], frame=95))

    path = write_csv(tmp_path, rows)

    records, errors = validate_annotations(
        [path],
        SHOT_ID,
        1280,
        720,
        expected_frames={path: 85},
    )

    assert len(records) == 3
    assert len(errors) == 1
    assert "expected 85, found 95" in errors[0]
