"""Independent fixtures for the blind-safe human-review report generator."""

import csv
from pathlib import Path

import pytest

from tools.report_release_human_review import build_report

RELEASE_HEADER = [
    "video_path",
    "manual_release_frame",
    "manual_release_valid",
    "manual_error_type",
    "reviewer_notes",
    "review_status",
]
IDENTITY_HEADER = [
    "video_path",
    "event_status",
    "player_name",
    "team",
    "shot_subtype",
    "description",
    "identity_decision",
    "observed_shooter",
    "observed_action",
    "identity_notes",
]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def fixtures(tmp_path):
    releases = tmp_path / "blind.csv"
    identities = tmp_path / "identity.csv"
    write_csv(
        releases,
        RELEASE_HEADER,
        [
            ["a.mp4", "79-80", "", "", "Ball left hand | slight occlusion", "provisional"],
            ["b.mp4", "", "", "", "", "unreviewed"],
        ],
    )
    write_csv(
        identities,
        IDENTITY_HEADER,
        [
            [
                "a.mp4",
                "one_shot_event",
                "Player A",
                "TEAM",
                "Jump Shot",
                "Missed jumper",
                "confirmed",
                "Player A",
                "Jumper",
                "Checked video",
            ],
            [
                "b.mp4",
                "one_shot_event",
                "Player B",
                "TEAM",
                "Layup",
                "Missed layup",
                "pending",
                "",
                "",
                "",
            ],
        ],
    )
    return releases, identities


def test_checkpoint_counts_and_provenance(tmp_path):
    releases, identities = fixtures(tmp_path)
    report = build_report(releases, identities)
    assert "Provisional release intervals: **1**" in report
    assert "Unreviewed release clips: **1**" in report
    assert "User-confirmed event identities: **1**" in report
    assert "Pending event identities: **1**" in report
    assert "Provisional intervals with a user-confirmed event identity: **1**" in report
    assert "79-80" in report
    assert "Ball left hand \\| slight occlusion" in report
    assert "AI assistant" in report
    assert "SHA-256" in report
    assert "detector predictions" in report


def test_rejects_nonmatching_paths(tmp_path):
    releases, identities = fixtures(tmp_path)
    write_csv(
        identities,
        IDENTITY_HEADER,
        [
            [
                "wrong.mp4",
                "one_shot_event",
                "Other",
                "TEAM",
                "Jump Shot",
                "Missed jumper",
                "pending",
                "",
                "",
                "",
            ],
        ],
    )
    with pytest.raises(ValueError, match="video_path sets differ"):
        build_report(releases, identities)


def test_rejects_duplicate_paths(tmp_path):
    releases, identities = fixtures(tmp_path)
    write_csv(
        releases,
        RELEASE_HEADER,
        [
            ["a.mp4", "79-80", "", "", "Seen", "provisional"],
            ["a.mp4", "81-82", "", "", "Seen", "provisional"],
        ],
    )
    with pytest.raises(ValueError, match="duplicate video_path"):
        build_report(releases, identities)


@pytest.mark.parametrize("frame", ["80-79", "maybe", "79-80-81", "-80"])
def test_rejects_invalid_interval(tmp_path, frame):
    releases, identities = fixtures(tmp_path)
    write_csv(
        releases,
        RELEASE_HEADER,
        [
            ["a.mp4", frame, "", "", "Seen", "provisional"],
            ["b.mp4", "", "", "", "", "unreviewed"],
        ],
    )
    with pytest.raises(ValueError, match="manual_release_frame"):
        build_report(releases, identities)


def test_rejects_confirmed_without_observed_action(tmp_path):
    releases, identities = fixtures(tmp_path)
    write_csv(
        identities,
        IDENTITY_HEADER,
        [
            [
                "a.mp4",
                "one_shot_event",
                "Player A",
                "TEAM",
                "Jump Shot",
                "Missed jumper",
                "confirmed",
                "Player A",
                "",
                "Checked video",
            ],
            [
                "b.mp4",
                "one_shot_event",
                "Player B",
                "TEAM",
                "Layup",
                "Missed layup",
                "pending",
                "",
                "",
                "",
            ],
        ],
    )
    with pytest.raises(ValueError, match="observed_shooter and observed_action"):
        build_report(releases, identities)


def test_completed_first_pass_preserves_uncertain_release(tmp_path):
    releases, identities = fixtures(tmp_path)

    write_csv(
        releases,
        RELEASE_HEADER,
        [
            ["a.mp4", "79-80", "", "", "Observed separation", "provisional"],
            ["b.mp4", "", "", "", "Two possible shooting actions", "uncertain"],
        ],
    )
    write_csv(
        identities,
        IDENTITY_HEADER,
        [
            [
                "a.mp4",
                "one_shot_event",
                "Player A",
                "TEAM",
                "Jump Shot",
                "Missed jumper",
                "confirmed",
                "Player A",
                "Jump Shot",
                "Reviewer confirmed event",
            ],
            [
                "b.mp4",
                "one_shot_event",
                "Player B",
                "TEAM",
                "Jump Shot",
                "Made jumper",
                "confirmed",
                "Player B",
                "Jump Shot",
                "Event confirmed; release ambiguous",
            ],
        ],
    )

    report = build_report(releases, identities)

    assert "first-pass manual-review checkpoint" in report
    assert "Sample clips: **2**" in report
    assert "Provisional release intervals: **1**" in report
    assert "Uncertain release clips: **1**" in report
    assert "Unreviewed release clips: **0**" in report
    assert "User-confirmed event identities: **2**" in report
    assert "## Uncertain release observations" in report
    assert "Two possible shooting actions" in report
    assert "b.mp4" in report
