"""Blind release-frame annotations with explicit review status."""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

FIELDS = [
    "video_path",
    "manual_release_frame",
    "manual_release_valid",
    "manual_error_type",
    "reviewer_notes",
    "review_status",
]

STATUSES = {"unreviewed", "provisional", "uncertain"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def valid_frame_range(value):
    if not re.fullmatch(r"\d+(?:-\d+)?", value):
        return False
    values = [int(part) for part in value.split("-")]
    return len(values) == 1 or values[0] <= values[1]


def normalize_path(value):
    name = value.replace("\\", "/")
    path = PurePosixPath(name)

    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or ":" in name
        or path.suffix.lower() != ".mp4"
    ):
        raise ValueError(f"Invalid video path: {value}")
    return path.as_posix()


def normalize_row(row):
    """Upgrade legacy rows without claiming they are verified."""
    result = {field: row.get(field, "") or "" for field in FIELDS}
    result["video_path"] = normalize_path(result["video_path"])

    frame = result["manual_release_frame"]
    notes = result["reviewer_notes"]
    status = result["review_status"]

    if not status:
        if frame:
            status = "provisional"
        elif notes:
            status = "uncertain"
        else:
            status = "unreviewed"

    if status not in STATUSES:
        raise ValueError(f"Unknown review status: {status}")

    if frame and not valid_frame_range(frame):
        raise ValueError(f"Invalid release frame: {frame}")

    if status == "unreviewed" and (frame or notes):
        raise ValueError("Unreviewed rows cannot contain observations")

    if status == "provisional" and not frame:
        raise ValueError("Provisional reviews require a frame estimate")

    if status == "uncertain" and (frame or not notes.strip()):
        raise ValueError("Uncertain reviews need notes and no frame estimate")

    result["review_status"] = status
    return result


def prepare_rows(manifest, existing):
    """Preserve existing observations and reject conflicting identities."""
    previous = {}

    for row in existing:
        normalized = normalize_row(row)
        key = normalized["video_path"]
        if key in previous:
            raise ValueError(f"Duplicate existing annotation: {key}")
        previous[key] = normalized

    rows = []
    seen = set()

    for item in manifest:
        key = normalize_path(item["video_path"])
        if key in seen:
            raise ValueError(f"Duplicate manifest entry: {key}")

        seen.add(key)
        rows.append(
            previous.get(
                key,
                normalize_row({"video_path": key}),
            )
        )

    unknown = set(previous) - seen
    if unknown:
        raise ValueError(f"Existing annotations missing from manifest: {sorted(unknown)}")

    return rows


def save_csv(path, rows):
    """Validate before writing and atomically replace the CSV."""
    normalized = [normalize_row(row) for row in rows]
    keys = [row["video_path"] for row in normalized]

    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate annotation video paths")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=".release_review_",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(normalized)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    rows = prepare_rows(manifest, existing)

    pending = [row for row in rows if row["review_status"] == "unreviewed"]

    print(f"Sample size: {len(rows)}")
    print(f"Previously recorded: {len(rows) - len(pending)}")
    print(f"Unreviewed: {len(pending)}")
    print("Detector predictions remain hidden.")

    viewer = Path(__file__).with_name("review_release_frames.py")
    saved = 0

    for opened, row in enumerate(pending):
        if args.limit and opened >= args.limit:
            break

        relative = PurePosixPath(row["video_path"])
        video = args.video_root.joinpath(*relative.parts)

        if not video.is_file():
            raise FileNotFoundError(video)

        print(f"\nVideo: {row['video_path']}")

        subprocess.run(
            [sys.executable, str(viewer), str(video)],
            check=True,
        )

        while True:
            value = input("Release frame/range, 'uncertain', or Enter to skip: ").strip()

            if not value or value.lower() == "uncertain" or valid_frame_range(value):
                break

            print("Enter a nonnegative frame or ascending range.")

        if not value:
            print("Skipped: remains unreviewed.")
            continue

        notes = input("What did you observe? ").strip()

        if value.lower() == "uncertain":
            if not notes:
                print("An uncertain review needs an explanation; nothing saved.")
                continue
            row["manual_release_frame"] = ""
            row["review_status"] = "uncertain"
        else:
            row["manual_release_frame"] = value
            row["review_status"] = "provisional"

        row["reviewer_notes"] = notes
        save_csv(args.annotations, rows)
        saved += 1
        print(f"Saved: {row['review_status']}")

    print(f"\nNew observations saved: {saved}")
    print(f"Worksheet: {args.annotations}")


if __name__ == "__main__":
    main()
