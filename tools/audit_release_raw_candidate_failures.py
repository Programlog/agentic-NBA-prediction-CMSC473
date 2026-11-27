"""Audit raw release-candidate conditions near provisional human intervals."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.nba_shots import release_diagnostics as detector
from tools.audit_release_candidates_against_human import parse_interval


def inspect_frame(shooter, ball, frame):
    """Inspect frozen candidate conditions without changing detector behavior."""
    n = min(len(shooter), len(ball))
    if frame < 0 or frame >= n:
        return "out_of_range"

    s = shooter[frame]
    b = ball[frame]

    if not np.isfinite(s).all():
        return "missing_shooter_box"
    if not np.isfinite(b).all():
        return "missing_ball_box"

    x, y, w, h = s
    bx = b[0] + b[2] / 2
    by = b[1] + b[3] / 2

    held = (
        x - detector.HAND_REACH * w < bx < x + (1 + detector.HAND_REACH) * w
        and y - detector.HAND_REACH * h < by < y + h
    )

    if not held:
        return "outside_hand_reach"

    ahead = ball[frame + 1 : frame + 1 + detector.RISE_FRAMES]
    ahead_y = ahead[:, 1] + ahead[:, 3] / 2
    ahead_y = ahead_y[np.isfinite(ahead_y)]

    if len(ahead_y) < 3:
        return "insufficient_future_ball_tracking"

    if not ahead_y[-1] < by - detector.RISE_BH * h:
        return "rise_threshold_not_met"

    return "qualifying_raw_candidate"


def run(audit_path, json_root, output, radius=3):
    if radius < 0:
        raise ValueError("Radius must be nonnegative")

    with audit_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    targets = [
        row
        for row in rows
        if row["human_status"] == "provisional" and row["raw_inside_interval"] == "False"
    ]

    if not targets:
        raise ValueError("No provisional intervals without raw overlap found")

    results = []

    for row in targets:
        video_path = Path(row["video_path"])
        trajectory_path = json_root / video_path.with_suffix(".json")

        with trajectory_path.open(encoding="utf-8") as handle:
            clip = json.load(handle)

        diagnosis = detector.diagnose_clip(clip)
        shooter_key = diagnosis["shooter_key"]

        if not shooter_key:
            raise ValueError(f"No unique shot-event shooter: {video_path}")

        shooter = detector.boxes(clip[shooter_key])
        ball = detector.boxes(clip.get("ball") or {})

        interval = parse_interval(row["human_interval"])
        if interval is None:
            raise ValueError(f"Missing interval: {video_path}")

        start, end = interval

        # First inspect the exact provisional interval.
        exact = [inspect_frame(shooter, ball, frame) for frame in range(start, end + 1)]

        # Also inspect a nearby diagnostic window. This is not
        # an expanded ground-truth interval.
        nearby = [
            inspect_frame(shooter, ball, frame)
            for frame in range(max(0, start - radius), end + radius + 1)
        ]

        # Verify that our condition reconstruction agrees with
        # the frozen detector for the inspected nearby frames.
        raw_candidates = set(detector.release_candidates(shooter, ball))
        for frame in range(max(0, start - radius), end + radius + 1):
            qualifies = inspect_frame(shooter, ball, frame) == "qualifying_raw_candidate"
            if qualifies != (frame in raw_candidates):
                raise ValueError(f"Candidate condition mismatch: {video_path}, frame {frame}")

        result = {
            "video_path": row["video_path"],
            "human_interval": row["human_interval"],
            "shooter_key": shooter_key,
            "detector_status": diagnosis["status"],
            "raw_candidate_count": row["raw_candidate_count"],
            "nearest_raw_distance": row["raw_nearest_distance"],
            "exact_frame_reasons": json.dumps(exact),
            "nearby_radius": radius,
            "nearby_frame_reasons": json.dumps(nearby),
            "exact_missing_shooter": exact.count("missing_shooter_box"),
            "exact_missing_ball": exact.count("missing_ball_box"),
            "exact_outside_hand_reach": exact.count("outside_hand_reach"),
            "exact_insufficient_future_ball": exact.count("insufficient_future_ball_tracking"),
            "exact_rise_threshold_failed": exact.count("rise_threshold_not_met"),
            "nearby_qualifying_candidates": nearby.count("qualifying_raw_candidate"),
        }

        results.append(result)

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Clips audited: {len(results)}")
    print(f"Output: {output}")
    print()

    for result in results:
        print(Path(result["video_path"]).name[:8])
        print("  Human interval:", result["human_interval"])
        print("  Exact-frame reasons:", result["exact_frame_reasons"])
        print("  Nearby qualifying candidates:", result["nearby_qualifying_candidates"])
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/release_candidate_filter_audit/per_clip.csv"),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("data/basketevent"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/release_raw_candidate_failure_audit/per_clip.csv"),
    )
    parser.add_argument("--radius", type=int, default=3)

    args = parser.parse_args()
    run(args.audit, args.json_root, args.output, args.radius)


if __name__ == "__main__":
    main()
