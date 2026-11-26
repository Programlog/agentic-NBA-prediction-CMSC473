import csv

from tools.annotate_release_review import (
    FIELDS,
    read_csv,
    save_csv,
    valid_frame_range,
)


def test_valid_release_frame_ranges():
    assert valid_frame_range("128")
    assert valid_frame_range("128-129")
    assert valid_frame_range("32-34")
    assert valid_frame_range("0")


def test_invalid_release_frame_ranges():
    assert not valid_frame_range("")
    assert not valid_frame_range("129-128")
    assert not valid_frame_range("-1")
    assert not valid_frame_range("abc")
    assert not valid_frame_range("12-15-18")
    assert not valid_frame_range("12.5")


def test_annotation_csv_round_trip(tmp_path):
    path = tmp_path / "annotations.csv"

    rows = [
        {
            "video_path": "test/clip_a.mp4",
            "manual_release_frame": "119-120",
            "manual_release_valid": "",
            "manual_error_type": "",
            "reviewer_notes": "Release timing uncertain.",
        },
        {
            "video_path": "test/clip_b.mp4",
            "manual_release_frame": "",
            "manual_release_valid": "",
            "manual_error_type": "",
            "reviewer_notes": "",
        },
    ]

    save_csv(path, rows)

    assert path.exists()
    assert read_csv(path) == rows


def test_existing_annotations_survive_resaving(tmp_path):
    path = tmp_path / "annotations.csv"

    original = [
        {
            "video_path": "test/clip_a.mp4",
            "manual_release_frame": "32-34",
            "manual_release_valid": "",
            "manual_error_type": "",
            "reviewer_notes": "Possible reverse layup.",
        }
    ]

    save_csv(path, original)
    loaded = read_csv(path)

    save_csv(path, loaded)

    assert read_csv(path) == original


def test_annotation_csv_has_expected_columns(tmp_path):
    path = tmp_path / "annotations.csv"

    save_csv(path, [])

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS
