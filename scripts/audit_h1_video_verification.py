"""Fail-closed consistency audit for H1 video download and review records."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

DOWNLOAD_COLUMNS = {
    "shot_id",
    "game_id",
    "game_event_id",
    "video_path",
    "download_status",
    "file_size_bytes",
    "error",
}

REVIEW_COLUMNS = {
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
    "shooter_box",
    "defender_box",
    "cv_training_eligible",
    "notes",
}

# Additional evidence is required before accepting positive verification claims.
# Existing v2 records do not contain these columns and remain unverified.
EVIDENCE_COLUMNS = (
    "event_identity_evidence",
    "release_frame_evidence",
    "feature_quality_evidence",
)

ALLOWED_REVIEW_STATUSES = {
    "preliminary",
    "provisional_release_interval",
    "verified",
}


def read_records(path: Path, required: set[str]) -> list[dict[str, str]]:
    """Read a CSV, rejecting missing columns and duplicate headers."""
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")

        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: duplicate column names")

        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no records")

    return rows


def parse_boolean(value: str, field: str, shot_id: str) -> bool:
    """Accept only explicit true or false."""
    normalized = str(value).strip().lower()

    if normalized not in {"true", "false"}:
        raise ValueError(f"{shot_id}: invalid {field}: {value!r}")

    return normalized == "true"


def optional_frame(value: str, field: str, shot_id: str) -> int | None:
    """Parse a blank or nonnegative integer frame number."""
    text = str(value or "").strip()

    if not text:
        return None

    try:
        frame = int(text)
    except ValueError as exc:
        raise ValueError(f"{shot_id}: invalid {field}: {text!r}") from exc

    if frame < 0:
        raise ValueError(f"{shot_id}: negative {field}")

    return frame


def index_rows(rows: list[dict[str, str]], label: str) -> dict[str, dict]:
    """Reject missing or duplicate shot identifiers."""
    indexed = {}

    for row in rows:
        shot_id = str(row.get("shot_id") or "").strip()

        if not shot_id:
            raise ValueError(f"{label}: empty shot_id")

        if shot_id in indexed:
            raise ValueError(f"{label}: duplicate shot_id {shot_id}")

        indexed[shot_id] = row

    return indexed


def check_record(download: dict, review: dict) -> dict:
    """Evaluate one shot; never infer verification from metadata agreement."""
    shot_id = review["shot_id"]
    issues = []

    for field in ("game_id", "game_event_id", "video_path", "download_status"):
        if download[field] != review[field]:
            issues.append(f"{field} mismatch between reports")

    parts = shot_id.split(":")
    if (
        len(parts) != 3
        or parts[0] != "nba"
        or parts[1] != review["game_id"]
        or parts[2] != review["game_event_id"]
    ):
        issues.append("shot_id does not match game_id/game_event_id")

    download_status = download["download_status"]

    if download_status not in {"success", "failed", "error"}:
        issues.append(f"unknown download_status: {download_status!r}")

    downloaded = download_status == "success"

    if downloaded:
        try:
            size = int(download["file_size_bytes"])
            if size <= 0:
                issues.append("successful download has nonpositive file size")
        except (ValueError, TypeError):
            issues.append("successful download has invalid file size")

        if str(download["error"]).strip():
            issues.append("successful download also has an error message")
    elif review["download_status"] == "success":
        issues.append("review claims success for unsuccessful download")

    try:
        identity_verified = parse_boolean(
            review["event_identity_verified"], "event_identity_verified", shot_id
        )
        training_eligible = parse_boolean(
            review["cv_training_eligible"], "cv_training_eligible", shot_id
        )

        start = optional_frame(review["release_frame_start"], "release_frame_start", shot_id)
        end = optional_frame(review["release_frame_end"], "release_frame_end", shot_id)
        exact = optional_frame(review["exact_release_frame"], "exact_release_frame", shot_id)
    except ValueError as exc:
        issues.append(str(exc))
        return {
            "shot_id": shot_id,
            "status": "invalid_record",
            "issues": issues,
            "eligibility_claimed": False,
            "training_eligible": False,
        }

    status = review["visual_review_status"]

    if status not in ALLOWED_REVIEW_STATUSES:
        issues.append(f"unknown visual_review_status: {status!r}")

    if (start is None) != (end is None):
        issues.append("release interval has only one endpoint")

    if start is not None and end is not None and start > end:
        issues.append("release interval start exceeds end")

    if exact is not None and start is not None and end is not None:
        if not start <= exact <= end:
            issues.append("exact release lies outside candidate interval")

    if status == "provisional_release_interval" and (start is None or end is None):
        issues.append("provisional release status requires an interval")

    evidence = {field: str(review.get(field) or "").strip() for field in EVIDENCE_COLUMNS}

    if identity_verified and not evidence["event_identity_evidence"]:
        issues.append("verified identity lacks explicit supporting evidence")

    if exact is not None and not evidence["release_frame_evidence"]:
        issues.append("exact release lacks explicit supporting evidence")

    if status == "verified" and not identity_verified:
        issues.append("verified review status contradicts identity flag")

    if training_eligible:
        if not downloaded:
            issues.append("training eligible without successful download")
        if not identity_verified:
            issues.append("training eligible without verified event identity")
        if exact is None:
            issues.append("training eligible without exact release frame")
        if not str(review["shooter_box"]).strip():
            issues.append("training eligible without shooter box")
        if not str(review["defender_box"]).strip():
            issues.append("training eligible without defender box")
        if not all(evidence.values()):
            issues.append("training eligible without complete evidence fields")

    if issues:
        result_status = "invalid_record"
    elif not downloaded:
        result_status = "download_unavailable"
    elif training_eligible:
        result_status = "eligibility_claim_requires_independent_verification"
    elif identity_verified:
        result_status = "identity_claim_requires_independent_verification"
    elif start is not None and end is not None:
        result_status = "provisional_interval_unverified"
    else:
        result_status = "preliminary_unverified"

    return {
        "shot_id": shot_id,
        "status": result_status,
        "issues": issues,
        "eligibility_claimed": training_eligible,
        "training_eligible": False,
    }


def audit(download_path: Path, review_path: Path) -> dict:
    """Join reports by exact shot ID and summarize all discrepancies."""
    downloads = index_rows(read_records(download_path, DOWNLOAD_COLUMNS), "downloads")
    reviews = index_rows(read_records(review_path, REVIEW_COLUMNS), "reviews")

    all_ids = sorted(set(downloads) | set(reviews))
    results = []

    for shot_id in all_ids:
        if shot_id not in downloads or shot_id not in reviews:
            missing = "download" if shot_id not in downloads else "review"
            results.append(
                {
                    "shot_id": shot_id,
                    "status": "invalid_record",
                    "issues": [f"missing {missing} record"],
                    "training_eligible": False,
                }
            )
            continue

        results.append(check_record(downloads[shot_id], reviews[shot_id]))

    counts = Counter(row["status"] for row in results)

    claims = sum(row.get("eligibility_claimed", False) for row in results)

    return {
        "download_records": len(downloads),
        "review_records": len(reviews),
        "shots_audited": len(results),
        "status_counts": dict(sorted(counts.items())),
        "invalid_records": counts["invalid_record"],
        "training_eligibility_claims": claims,
        "claims_not_independently_approved": claims,
        "independently_approved_training_clips": 0,
        "independently_verified_video_events": None,
        "independent_visual_verification_performed": False,
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    summary = audit(args.downloads, args.reviews)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "records"},
            indent=2,
        )
    )

    if summary["invalid_records"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
