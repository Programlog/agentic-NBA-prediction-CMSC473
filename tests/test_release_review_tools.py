import csv

import pytest

from tools.annotate_release_review import (
    FIELDS,
    normalize_path,
    normalize_row,
    prepare_rows,
    read_csv,
    save_csv,
    valid_frame_range,
)


def test_valid_release_frame_ranges():
    for value in ("0", "128", "128-129", "32-34"):
        assert valid_frame_range(value)


def test_invalid_release_frame_ranges():
    for value in ("", "-1", "129-128", "abc", "1-2-3", "12.5"):
        assert not valid_frame_range(value)


def test_legacy_observation_becomes_provisional():
    row = normalize_row(
        {
            "video_path": r"test\clip.mp4",
            "manual_release_frame": "119-120",
            "reviewer_notes": "Possible release.",
        }
    )

    assert row["video_path"] == "test/clip.mp4"
    assert row["review_status"] == "provisional"
    assert row["manual_release_frame"] == "119-120"
    assert row["reviewer_notes"] == "Possible release."


def test_legacy_unreviewed_remains_unreviewed():
    row = normalize_row({"video_path": "test/clip.mp4"})
    assert row["review_status"] == "unreviewed"


def test_legacy_notes_without_frame_become_uncertain():
    row = normalize_row(
        {
            "video_path": "test/clip.mp4",
            "reviewer_notes": "Release not visible.",
        }
    )
    assert row["review_status"] == "uncertain"


def test_uncertain_review_requires_notes():
    with pytest.raises(ValueError, match="need notes"):
        normalize_row(
            {
                "video_path": "test/clip.mp4",
                "review_status": "uncertain",
            }
        )


def test_uncertain_review_cannot_have_frame():
    with pytest.raises(ValueError, match="need notes"):
        normalize_row(
            {
                "video_path": "test/clip.mp4",
                "review_status": "uncertain",
                "manual_release_frame": "32",
                "reviewer_notes": "Not sure.",
            }
        )


def test_unreviewed_cannot_contain_observations():
    with pytest.raises(ValueError, match="Unreviewed"):
        normalize_row(
            {
                "video_path": "test/clip.mp4",
                "review_status": "unreviewed",
                "manual_release_frame": "32",
            }
        )


def test_provisional_requires_frame():
    with pytest.raises(ValueError, match="require"):
        normalize_row(
            {
                "video_path": "test/clip.mp4",
                "review_status": "provisional",
            }
        )


def test_invalid_video_path_rejected():
    for path in (
        "../clip.mp4",
        "/tmp/clip.mp4",
        "C:/videos/clip.mp4",
        "test/clip.json",
    ):
        with pytest.raises(ValueError):
            normalize_path(path)


def test_prepare_rows_preserves_existing_observation():
    manifest = [
        {"video_path": "test/a.mp4"},
        {"video_path": "test/b.mp4"},
    ]
    existing = [
        {
            "video_path": r"test\a.mp4",
            "manual_release_frame": "128-129",
            "reviewer_notes": "Possible jump shot.",
        }
    ]

    rows = prepare_rows(manifest, existing)

    assert len(rows) == 2
    assert rows[0]["review_status"] == "provisional"
    assert rows[0]["manual_release_frame"] == "128-129"
    assert rows[1]["review_status"] == "unreviewed"


def test_duplicate_manifest_rejected():
    with pytest.raises(ValueError, match="Duplicate manifest"):
        prepare_rows(
            [
                {"video_path": "test/a.mp4"},
                {"video_path": r"test\a.mp4"},
            ],
            [],
        )


def test_duplicate_existing_observation_rejected():
    with pytest.raises(ValueError, match="Duplicate existing"):
        prepare_rows(
            [{"video_path": "test/a.mp4"}],
            [
                {"video_path": "test/a.mp4"},
                {"video_path": r"test\a.mp4"},
            ],
        )


def test_unknown_existing_clip_rejected():
    with pytest.raises(ValueError, match="missing from manifest"):
        prepare_rows(
            [{"video_path": "test/a.mp4"}],
            [{"video_path": "test/b.mp4"}],
        )


def test_annotation_csv_round_trip(tmp_path):
    path = tmp_path / "annotations.csv"
    rows = [
        normalize_row(
            {
                "video_path": "test/a.mp4",
                "manual_release_frame": "119-120",
                "reviewer_notes": "Approximate release.",
            }
        ),
        normalize_row(
            {
                "video_path": "test/b.mp4",
            }
        ),
    ]

    save_csv(path, rows)

    assert read_csv(path) == rows


def test_existing_annotations_survive_resaving(tmp_path):
    path = tmp_path / "annotations.csv"
    original = normalize_row(
        {
            "video_path": "test/a.mp4",
            "manual_release_frame": "32-34",
            "reviewer_notes": "Possible reverse layup.",
        }
    )

    save_csv(path, [original])
    save_csv(path, read_csv(path))

    assert read_csv(path) == [original]


def test_failed_validation_does_not_overwrite_csv(tmp_path):
    path = tmp_path / "annotations.csv"
    original = normalize_row({"video_path": "test/a.mp4"})
    save_csv(path, [original])

    before = path.read_bytes()

    with pytest.raises(ValueError):
        save_csv(
            path,
            [
                {
                    "video_path": "test/a.mp4",
                    "review_status": "provisional",
                }
            ],
        )

    assert path.read_bytes() == before


def test_duplicate_rows_cannot_be_saved(tmp_path):
    path = tmp_path / "annotations.csv"

    with pytest.raises(ValueError, match="Duplicate annotation"):
        save_csv(
            path,
            [
                {"video_path": "test/a.mp4"},
                {"video_path": r"test\a.mp4"},
            ],
        )

    assert not path.exists()


def test_annotation_csv_has_expected_columns(tmp_path):
    path = tmp_path / "annotations.csv"
    save_csv(path, [])

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == FIELDS
