"""Validate H1 manual boxes and compute image-space center distances."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

EXPECTED_ROLES = (
    "shooter_wembanyama_1",
    "candidate_defender_houston_12",
    "candidate_defender_headband",
)

REQUIRED_COLUMNS = (
    "shot_id",
    "frame",
    "role",
    "x1",
    "y1",
    "x2",
    "y2",
)


def validate_annotations(
    paths: list[Path],
    shot_id: str,
    width: int,
    height: int,
    expected_frames: dict[Path, int] | None = None,
) -> tuple[list[dict], list[str]]:
    """Validate box records without silently dropping invalid entries."""
    records = []
    errors = []
    seen = set()

    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)

            if not reader.fieldnames or not set(REQUIRED_COLUMNS).issubset(reader.fieldnames):
                errors.append(f"{path}: missing required columns")
                continue

            for line_number, row in enumerate(reader, start=2):
                prefix = f"{path}:{line_number}"

                try:
                    frame = int(row["frame"])
                    x1, y1, x2, y2 = (int(row[key]) for key in ("x1", "y1", "x2", "y2"))
                except (ValueError, TypeError):
                    errors.append(f"{prefix}: invalid numeric value")
                    continue

                if expected_frames is not None:
                    expected_frame = expected_frames.get(path)

                    if expected_frame is None:
                        errors.append(f"{prefix}: unexpected annotation file")
                        continue

                    if frame != expected_frame:
                        errors.append(
                            f"{prefix}: frame mismatch: expected {expected_frame}, found {frame}"
                        )
                        continue

                role = row["role"]
                key = (row["shot_id"], frame, role)

                if row["shot_id"] != shot_id:
                    errors.append(f"{prefix}: unexpected shot ID")
                    continue

                if role not in EXPECTED_ROLES:
                    errors.append(f"{prefix}: unknown role {role}")
                    continue

                if key in seen:
                    errors.append(f"{prefix}: duplicate annotation {key}")
                    continue

                seen.add(key)

                if frame < 0:
                    errors.append(f"{prefix}: negative frame index")
                    continue

                if not (0 <= x1 < x2 <= width):
                    errors.append(f"{prefix}: invalid x coordinates")
                    continue

                if not (0 <= y1 < y2 <= height):
                    errors.append(f"{prefix}: invalid y coordinates")
                    continue

                records.append(
                    {
                        "shot_id": shot_id,
                        "frame": frame,
                        "role": role,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "center_x": (x1 + x2) / 2,
                        "center_y": (y1 + y2) / 2,
                    }
                )

    return records, errors


def calculate_distances(records: list[dict]) -> tuple[list[dict], list[str]]:
    """Compute image-space center distances for complete frame annotations."""
    grouped = {}
    for row in records:
        grouped.setdefault(row["frame"], {})[row["role"]] = row

    results = []
    errors = []

    for frame, players in sorted(grouped.items()):
        missing = set(EXPECTED_ROLES) - set(players)

        if missing:
            errors.append(f"Frame {frame}: missing roles {sorted(missing)}")
            continue

        shooter = players[EXPECTED_ROLES[0]]

        for role in EXPECTED_ROLES[1:]:
            defender = players[role]

            distance = math.hypot(
                shooter["center_x"] - defender["center_x"],
                shooter["center_y"] - defender["center_y"],
            )

            results.append(
                {
                    "shot_id": shooter["shot_id"],
                    "frame": frame,
                    "defender_role": role,
                    "center_distance_pixels": round(distance, 4),
                }
            )

    return results, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot-id", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, nargs="+", required=True)

    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    if args.width <= 0 or args.height <= 0:
        parser.error("Image dimensions must be positive")

    if len(set(args.frames)) != len(args.frames) or any(frame < 0 for frame in args.frames):
        parser.error("Frame indices must be unique and nonnegative")

    paths = [args.annotations / f"shot283_frame{frame}_boxes.csv" for frame in args.frames]

    for path in paths:
        if not path.is_file():
            parser.error(f"Missing annotation file: {path}")

    expected_frames = dict(zip(paths, args.frames))

    records, errors = validate_annotations(
        paths,
        args.shot_id,
        args.width,
        args.height,
        expected_frames=expected_frames,
    )

    counts = Counter(row["frame"] for row in records)

    for frame in args.frames:
        if counts[frame] != len(EXPECTED_ROLES):
            errors.append(
                f"Frame {frame}: expected {len(EXPECTED_ROLES)} valid boxes; found {counts[frame]}"
            )

    distances, distance_errors = calculate_distances(records)
    errors.extend(distance_errors)

    summary = {
        "shot_id": args.shot_id,
        "image_width": args.width,
        "image_height": args.height,
        "frames_requested": args.frames,
        "valid_boxes": len(records),
        "expected_boxes": len(args.frames) * len(EXPECTED_ROLES),
        "distance_measurements": len(distances),
        "measurement_unit": "image_pixels",
        "physical_defender_distance_available": False,
        "primary_contesting_defender_identified": False,
        "errors": errors,
        "validation_passed": not errors,
    }

    args.output.mkdir(parents=True)

    with (args.output / "center_distances.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "shot_id",
                "frame",
                "defender_role",
                "center_distance_pixels",
            ],
        )
        writer.writeheader()
        writer.writerows(distances)

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
