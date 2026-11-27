"""Blind-safe readiness audit and exploratory release-frame comparison.

The readiness audit never reads detector predictions.
Comparison is blocked until all manual and identity reviews are complete.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath

from nba_shots.release_diagnostics import diagnose_clip
from tools.annotate_release_review import normalize_row, read_csv
from tools.review_release_event_identity import (
    extract_event_identity,
    validate_identity_rows,
)

READINESS_FIELDS = [
    "video_path",
    "review_status",
    "identity_decision",
    "ready_for_comparison",
    "reason",
]

COMPARISON_FIELDS = [
    "video_path",
    "manual_release_frame",
    "detector_release_frame",
    "interval_error_frames",
    "comparison_status",
]

IDENTITY_FIELDS = (
    "event_status",
    "shooter_key",
    "player_name",
    "team",
    "action_type",
    "shot_subtype",
    "description",
    "period",
    "clock",
)


def index_rows(rows, kind):
    indexed = {}
    for row in rows:
        key = row["video_path"].replace("\\", "/")
        if key in indexed:
            raise ValueError(f"Duplicate {kind} row: {key}")
        indexed[key] = row
    return indexed


def load_reviews(annotation_path, identity_path):
    annotations = [normalize_row(row) for row in read_csv(annotation_path)]
    identities = read_csv(identity_path)
    validate_identity_rows(identities)

    annotation_map = index_rows(annotations, "annotation")
    identity_map = index_rows(identities, "identity")

    if set(annotation_map) != set(identity_map):
        raise ValueError("Annotation and identity worksheets have different video sets")

    return annotation_map, identity_map


def interval_error(manual_range, detected_frame):
    """Distance to inclusive human interval; zero means within it."""
    parts = [int(value) for value in manual_range.split("-")]
    start = parts[0]
    end = parts[-1]
    detected = int(detected_frame)
    return max(start - detected, 0, detected - end)


def make_readiness(annotations, identities):
    rows = []

    for path, annotation in annotations.items():
        identity = identities[path]
        review = annotation["review_status"]
        decision = identity["identity_decision"]

        if review == "unreviewed":
            reason = "manual_review_pending"
        elif decision == "pending":
            reason = "identity_review_pending"
        elif review == "uncertain":
            reason = "manual_release_uncertain"
        elif decision == "uncertain":
            reason = "identity_uncertain"
        elif decision == "mismatch":
            reason = "identity_mismatch"
        elif identity["event_status"] != "one_shot_event":
            reason = "ambiguous_shot_event"
        else:
            reason = "provisional_comparison_eligible"

        rows.append(
            {
                "video_path": path,
                "review_status": review,
                "identity_decision": decision,
                "ready_for_comparison": str(reason == "provisional_comparison_eligible").lower(),
                "reason": reason,
            }
        )

    return rows


def ensure_blind_review_complete(annotations, identities):
    unreviewed = [key for key, row in annotations.items() if row["review_status"] == "unreviewed"]
    pending = [key for key, row in identities.items() if row["identity_decision"] == "pending"]

    if unreviewed or pending:
        raise ValueError(
            "Blind review incomplete: "
            f"{len(unreviewed)} unreviewed clips; "
            f"{len(pending)} pending identity decisions. "
            "Detector comparison is blocked."
        )


def compare(annotations, identities, json_root):
    """Produce exploratory comparisons, not verified ground truth."""
    ensure_blind_review_complete(annotations, identities)
    results = []

    for path, annotation in annotations.items():
        identity = identities[path]

        if (
            annotation["review_status"] != "provisional"
            or identity["identity_decision"] != "confirmed"
        ):
            results.append(
                {
                    "video_path": path,
                    "manual_release_frame": "",
                    "detector_release_frame": "",
                    "interval_error_frames": "",
                    "comparison_status": "excluded_unverifiable",
                }
            )
            continue

        relative = PurePosixPath(path).with_suffix(".json")
        json_path = json_root.joinpath(*relative.parts)

        with json_path.open(encoding="utf-8") as stream:
            clip = json.load(stream)

        current = extract_event_identity(clip)
        for field in IDENTITY_FIELDS:
            if current[field] != identity[field]:
                raise ValueError(f"Event metadata changed for {path}: {field}")

        diagnostics = diagnose_clip(clip)
        detected = diagnostics["selected_release_frame"]

        if detected == "" or detected is None:
            error = ""
            status = "detector_no_release"
        else:
            error = interval_error(
                annotation["manual_release_frame"],
                detected,
            )
            status = "within_provisional_interval" if error == 0 else "outside_provisional_interval"

        results.append(
            {
                "video_path": path,
                "manual_release_frame": annotation["manual_release_frame"],
                "detector_release_frame": detected,
                "interval_error_frames": error,
                "comparison_status": status,
            }
        )

    return results


def write_new_csv(path, fields, rows):
    """Never overwrite existing research reports."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("readiness", "compare"),
        required=True,
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--json-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations, identities = load_reviews(
        args.annotations,
        args.identities,
    )

    if args.mode == "readiness":
        results = make_readiness(annotations, identities)
        fields = READINESS_FIELDS
    else:
        if args.json_root is None:
            parser.error("--json-root is required for comparison")
        results = compare(annotations, identities, args.json_root)
        fields = COMPARISON_FIELDS

    write_new_csv(args.output, fields, results)

    print(f"Rows: {len(results)}")
    for status, count in sorted(
        Counter(row.get("reason", row.get("comparison_status")) for row in results).items()
    ):
        print(f"{status}: {count}")

    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
