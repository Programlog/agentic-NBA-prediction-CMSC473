"""Create and validate a fail-closed H1 video-event manual-review worksheet.

This tool never verifies a video, assigns release timing, or approves training.
It separates machine-suggested intervals from independent human observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

SOURCE_FIELDS = (
    "shot_id",
    "game_id",
    "game_event_id",
    "video_path",
    "download_status",
    "visual_review_status",
    "event_identity_verified",
    "release_frame_start",
    "release_frame_end",
    "exact_release_frame",
    "cv_training_eligible",
    "notes",
)
REVIEW_FIELDS = (
    "reviewer_id",
    "identity_decision",
    "observed_shooter",
    "observed_team",
    "observed_period",
    "observed_clock",
    "observed_shot_type",
    "identity_evidence",
    "timing_decision",
    "observed_release_start",
    "observed_release_end",
    "timing_evidence",
    "unresolved_issues",
)
FIELDS = (*SOURCE_FIELDS, *REVIEW_FIELDS)
IDENTITY_CHOICES = {"not_reviewed", "match", "mismatch", "uncertain"}
TIMING_CHOICES = {"not_reviewed", "exact", "interval", "not_visible", "uncertain"}


def read_csv(path: Path, required: tuple[str, ...]) -> list[dict[str, str]]:
    """Read a CSV and reject malformed or duplicate identifiers."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing or duplicate headers")
        absent = set(required) - set(reader.fieldnames)
        if absent:
            raise ValueError(f"{path}: missing columns {sorted(absent)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no rows")
    seen = set()
    for row in rows:
        shot_id = (row.get("shot_id") or "").strip()
        if not shot_id or shot_id in seen:
            raise ValueError(f"{path}: missing or duplicate shot_id {shot_id!r}")
        seen.add(shot_id)
    return rows


def prepare(source: Path) -> list[dict[str, str]]:
    """Build an unreviewed worksheet; preserve suggestions only as source context."""
    rows = read_csv(source, SOURCE_FIELDS)
    result = []
    for row in rows:
        item = {field: row[field] for field in SOURCE_FIELDS}
        item.update({field: "" for field in REVIEW_FIELDS})
        item["identity_decision"] = "not_reviewed"
        item["timing_decision"] = "not_reviewed"
        result.append(item)
    return result


def _frame(value: str) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError("frame must be an integer") from exc
    if result < 0:
        raise ValueError("frame must be nonnegative")
    return result


def inspect(row: dict[str, str]) -> dict:
    """Check review completeness; never infer ground truth from user-entered text."""
    shot_id = row["shot_id"]
    errors: list[str] = []
    identity = (row["identity_decision"] or "").strip()
    timing = (row["timing_decision"] or "").strip()
    if identity not in IDENTITY_CHOICES:
        errors.append("invalid identity_decision")
    if timing not in TIMING_CHOICES:
        errors.append("invalid timing_decision")
    try:
        start = _frame(row["observed_release_start"])
        end = _frame(row["observed_release_end"])
    except ValueError as exc:
        errors.append(str(exc))
        start = end = None
    if (start is None) != (end is None):
        errors.append("release interval requires both endpoints")
    if start is not None and end is not None and start > end:
        errors.append("release interval start exceeds end")
    if timing == "exact" and (start is None or end is None or start != end):
        errors.append("exact release requires equal start and end frames")
    if timing == "interval" and (start is None or end is None):
        errors.append("interval decision requires both endpoints")
    if timing in {"not_reviewed", "not_visible", "uncertain"} and start is not None:
        errors.append("non-specific timing decision must not assert release frames")
    if identity == "match":
        needed = (
            "reviewer_id",
            "observed_shooter",
            "observed_team",
            "observed_period",
            "observed_clock",
            "observed_shot_type",
            "identity_evidence",
        )
        for field in needed:
            if not (row[field] or "").strip():
                errors.append(f"identity match missing {field}")
    if timing in {"exact", "interval"}:
        if not (row["reviewer_id"] or "").strip():
            errors.append("timing decision missing reviewer_id")
        if not (row["timing_evidence"] or "").strip():
            errors.append("timing decision missing timing_evidence")
    if identity in {"mismatch", "uncertain"} and not (row["unresolved_issues"] or "").strip():
        errors.append("mismatch/uncertain identity requires unresolved_issues")
    if errors:
        status = "invalid_review"
    elif identity == "match" and timing in {"exact", "interval"}:
        status = "human_claim_pending_independent_adjudication"
    elif identity == "mismatch":
        status = "human_reported_mismatch"
    elif identity == "not_reviewed" and timing == "not_reviewed":
        status = "not_reviewed"
    else:
        status = "unresolved_review"
    return {
        "shot_id": shot_id,
        "review_status": status,
        "issues": errors,
        "training_eligible": False,
        "independently_verified": False,
    }


def validate(source: Path, worksheet: Path) -> dict:
    """Cross-check worksheet against immutable source and summarize outcomes."""
    original = {row["shot_id"]: row for row in read_csv(source, SOURCE_FIELDS)}
    entries = read_csv(worksheet, FIELDS)
    output = []
    for row in entries:
        shot_id = row["shot_id"]
        if shot_id not in original:
            output.append(
                {
                    "shot_id": shot_id,
                    "review_status": "invalid_review",
                    "issues": ["unknown shot_id"],
                    "training_eligible": False,
                    "independently_verified": False,
                }
            )
            continue
        changed = [field for field in SOURCE_FIELDS if row[field] != original[shot_id][field]]
        if changed:
            output.append(
                {
                    "shot_id": shot_id,
                    "review_status": "invalid_review",
                    "issues": [f"source columns modified: {changed}"],
                    "training_eligible": False,
                    "independently_verified": False,
                }
            )
            continue
        output.append(inspect(row))
    for shot_id in sorted(set(original) - {r["shot_id"] for r in entries}):
        output.append(
            {
                "shot_id": shot_id,
                "review_status": "invalid_review",
                "issues": ["missing worksheet row"],
                "training_eligible": False,
                "independently_verified": False,
            }
        )
    counts = Counter(r["review_status"] for r in output)
    return {
        "source_shots": len(original),
        "worksheet_rows": len(entries),
        "status_counts": dict(sorted(counts.items())),
        "invalid_reviews": counts["invalid_review"],
        "independently_verified": 0,
        "approved_training_clips": 0,
        "independent_video_verification_performed": False,
        "records": output,
    }


def _write_worksheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    check_parser = sub.add_parser("validate")
    check_parser.add_argument("--source", type=Path, required=True)
    check_parser.add_argument("--worksheet", type=Path, required=True)
    check_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite: {args.output}")
    if args.command == "prepare":
        rows = prepare(args.source)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _write_worksheet(args.output, rows)
        print(f"Prepared {len(rows)} unreviewed shots: {args.output}")
    else:
        report = validate(args.source, args.worksheet)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
        if report["invalid_reviews"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
