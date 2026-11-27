"""Compare provisional human releases with frozen detector candidates."""

import argparse
import csv
import json
from pathlib import Path

from src.nba_shots.release_diagnostics import diagnose_clip


def interval_distance(frame, start, end):
    """Distance from a frame to an inclusive interval."""
    return max(start - frame, 0, frame - end)


def nearest_distance(candidates, start, end):
    if not candidates:
        return ""
    return min(interval_distance(frame, start, end) for frame in candidates)


def parse_interval(value):
    value = value.strip().replace("–", "-")
    if not value:
        return None
    parts = value.split("-")
    if len(parts) == 1:
        frame = int(parts[0])
        return frame, frame
    if len(parts) == 2:
        start, end = map(int, parts)
        if start < 0 or end < start:
            raise ValueError(f"Invalid interval: {value}")
        return start, end
    raise ValueError(f"Invalid interval: {value}")


def run(review_path, json_root, output):
    with review_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "manual_release_frame", "review_status"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing review columns: {sorted(missing)}")
        reviews = list(reader)

    if len(reviews) != 20:
        raise ValueError(f"Expected 20 reviewed clips, found {len(reviews)}")

    results = []

    for row in reviews:
        video_path = Path(row["video_path"])
        json_path = json_root / video_path.with_suffix(".json")

        if not json_path.is_file():
            raise FileNotFoundError(json_path)

        with json_path.open(encoding="utf-8") as handle:
            clip = json.load(handle)

        diag = diagnose_clip(clip)

        retained = json.loads(diag.get("candidate_frames") or "[]")
        removed = json.loads(diag.get("removed_candidate_frames") or "[]")
        raw = sorted(set(retained) | set(removed))

        status = row["review_status"]
        interval = parse_interval(row["manual_release_frame"])

        if status == "provisional" and interval is None:
            raise ValueError(f"Missing provisional interval: {video_path}")
        if status == "uncertain" and interval is not None:
            raise ValueError(f"Uncertain clip has interval: {video_path}")

        result = {
            "video_path": row["video_path"],
            "human_status": status,
            "human_interval": row["manual_release_frame"],
            "detector_status": diag["status"],
            "raw_candidate_count": len(raw),
            "retained_candidate_count": len(retained),
            "removed_candidate_count": len(removed),
            "raw_candidate_frames": json.dumps(raw),
            "retained_candidate_frames": json.dumps(retained),
            "removed_candidate_frames": json.dumps(removed),
            "raw_nearest_distance": "",
            "retained_nearest_distance": "",
            "removed_nearest_distance": "",
            "raw_inside_interval": "",
            "retained_inside_interval": "",
            "removed_inside_interval": "",
            "filter_removed_nearer_candidate": "",
        }

        if interval:
            start, end = interval
            raw_distance = nearest_distance(raw, start, end)
            retained_distance = nearest_distance(retained, start, end)
            removed_distance = nearest_distance(removed, start, end)

            result.update(
                {
                    "raw_nearest_distance": raw_distance,
                    "retained_nearest_distance": retained_distance,
                    "removed_nearest_distance": removed_distance,
                    "raw_inside_interval": any(start <= frame <= end for frame in raw),
                    "retained_inside_interval": any(start <= frame <= end for frame in retained),
                    "removed_inside_interval": any(start <= frame <= end for frame in removed),
                    "filter_removed_nearer_candidate": (
                        removed_distance != ""
                        and (retained_distance == "" or removed_distance < retained_distance)
                    ),
                }
            )

        results.append(result)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Audited clips: {len(results)}")
    print(
        "Provisional intervals:",
        sum(r["human_status"] == "provisional" for r in results),
    )
    print(
        "Uncertain intervals:",
        sum(r["human_status"] == "uncertain" for r in results),
    )
    print(
        "Clips with removed candidates:",
        sum(r["removed_candidate_count"] > 0 for r in results),
    )
    print(
        "Clips with removed candidates nearer to human interval:",
        sum(r["filter_removed_nearer_candidate"] is True for r in results),
    )
    print(f"Output: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reviews",
        type=Path,
        default=Path("artifacts/release_manual_review/blind_review.csv"),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("data/basketevent"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/release_candidate_filter_audit/per_clip.csv"),
    )
    args = parser.parse_args()
    run(args.reviews, args.json_root, args.output)


if __name__ == "__main__":
    main()
