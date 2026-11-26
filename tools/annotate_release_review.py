"""Blind release-frame annotation for a BasketEvent review sample."""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

FIELDS = [
    "video_path",
    "manual_release_frame",
    "manual_release_valid",
    "manual_error_type",
    "reviewer_notes",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    """Save annotations without leaving a partially written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    os.replace(temporary, path)


def valid_frame_range(value):
    if not re.fullmatch(r"\d+(?:-\d+)?", value):
        return False

    numbers = [int(part) for part in value.split("-")]
    return len(numbers) == 1 or numbers[0] <= numbers[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0:
        parser.error("--limit must be nonnegative")

    manifest = read_csv(args.manifest)
    existing = read_csv(args.annotations) if args.annotations.exists() else []

    previous = {}
    for row in existing:
        key = row["video_path"].replace("\\", "/")
        if key in previous:
            raise ValueError(f"Duplicate existing annotation: {key}")
        previous[key] = row

    rows = []
    seen = set()

    for item in manifest:
        name = item["video_path"].replace("\\", "/")
        relative = PurePosixPath(name)

        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".mp4":
            raise ValueError(f"Unsafe video path: {name}")

        if name in seen:
            raise ValueError(f"Duplicate manifest video: {name}")

        seen.add(name)

        old = previous.get(name, {})
        rows.append(
            {
                "video_path": name,
                "manual_release_frame": old.get("manual_release_frame", ""),
                "manual_release_valid": old.get("manual_release_valid", ""),
                "manual_error_type": old.get("manual_error_type", ""),
                "reviewer_notes": old.get("reviewer_notes", ""),
            }
        )

    unknown = set(previous) - seen
    if unknown:
        raise ValueError(f"Existing annotations not in manifest: {sorted(unknown)}")

    viewer = Path(__file__).with_name("review_release_frames.py")

    pending = [row for row in rows if not row["manual_release_frame"] and not row["reviewer_notes"]]

    print(f"Total sample: {len(rows)}")
    print(f"Previously annotated: {len(rows) - len(pending)}")
    print(f"Remaining: {len(pending)}")
    print("Diagnostic results are intentionally hidden.")

    reviewed = 0
    opened = 0

    for row in pending:
        if args.limit and opened >= args.limit:
            break

        opened += 1

        video = args.video_root.joinpath(*PurePosixPath(row["video_path"]).parts)

        if not video.is_file():
            raise FileNotFoundError(video)

        print(f"\nVideo: {row['video_path']}")
        print("Inspect the video, then press Q to close the viewer.")

        subprocess.run(
            [sys.executable, str(viewer), str(video)],
            check=True,
        )

        while True:
            value = input(
                "Release frame/range (e.g. 119-120), 'uncertain', or Enter to skip: "
            ).strip()

            if not value or value.lower() == "uncertain" or valid_frame_range(value):
                break

            print("Use a frame number, range, or 'uncertain'.")

        if not value:
            print("Skipped; no annotation recorded.")
            continue

        notes = input("What did you observe? ").strip()

        row["manual_release_frame"] = "" if value.lower() == "uncertain" else value
        row["reviewer_notes"] = notes or "Release could not be identified confidently."

        save_csv(args.annotations, rows)
        reviewed += 1

        print("Annotation saved.")

    print(f"\nNew annotations saved: {reviewed}")
    print(f"Worksheet: {args.annotations}")


if __name__ == "__main__":
    main()
