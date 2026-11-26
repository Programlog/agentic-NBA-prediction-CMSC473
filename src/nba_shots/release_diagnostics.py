"""Read-only diagnostics for BasketEvent release-frame candidates.

This module does not alter the frozen H1 release detector.
Candidate frames use its existing geometric rules.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .basketevent import (
    HAND_REACH,
    RISE_BH,
    RISE_FRAMES,
    SHOT_ACTIONS,
    boxes,
    release_frame,
    steady,
)


class ReleaseDetectorMismatchError(RuntimeError):
    """Candidate enumeration disagrees with the frozen release detector."""


def release_candidates(shooter: np.ndarray, ball: np.ndarray) -> list[int]:
    """Return every frame satisfying the original release criteria."""
    n = min(len(shooter), len(ball))
    if n == 0:
        return []

    x, y, w, h = shooter[:n].T
    bx = ball[:n, 0] + ball[:n, 2] / 2
    by = ball[:n, 1] + ball[:n, 3] / 2

    with np.errstate(invalid="ignore"):
        held = (
            (bx > x - HAND_REACH * w)
            & (bx < x + (1 + HAND_REACH) * w)
            & (by > y - HAND_REACH * h)
            & (by < y + h)
        )

    candidates = []
    for k in np.flatnonzero(held):
        ahead = by[k + 1 : k + 1 + RISE_FRAMES]
        ahead = ahead[np.isfinite(ahead)]
        if len(ahead) >= 3 and ahead[-1] < by[k] - RISE_BH * h[k]:
            candidates.append(int(k))

    return candidates


def candidate_episodes(
    candidates: list[int],
    max_missing_frames: int = 0,
) -> list[list[int]]:
    """Group candidates, optionally tolerating brief missing-frame gaps."""
    if max_missing_frames < 0:
        raise ValueError("max_missing_frames must be nonnegative")

    if candidates != sorted(set(candidates)):
        raise ValueError("Candidates must be sorted and unique")

    episodes = []

    for frame in candidates:
        if not episodes or frame - episodes[-1][-1] > max_missing_frames + 1:
            episodes.append([frame])
        else:
            episodes[-1].append(frame)

    return episodes


def diagnose_clip(clip: dict) -> dict:
    """Describe candidate ambiguity without claiming a verified release."""
    shooters = []

    for key, entity in clip.items():
        if not isinstance(entity, dict) or key == "ball":
            continue
        event = entity.get("event")
        if isinstance(event, dict) and event.get("actionType") in SHOT_ACTIONS:
            shooters.append(key)

    result = {
        "shooter_key": "",
        "status": "",
        "selected_release_frame": "",
        "candidate_count": 0,
        "episode_count": 0,
        "earliest_candidate": "",
        "latest_candidate": "",
        "candidate_span_frames": "",
        "candidate_frames": "",
        "gap_tolerant_episode_count": 0,
        "largest_candidate_gap_frames": "",
        "widely_separated_candidates": False,
    }

    if not shooters:
        result["status"] = "no_shot_event"
        return result

    if len(shooters) != 1:
        result["status"] = "ambiguous_shot_event"
        return result

    key = shooters[0]
    result["shooter_key"] = key

    shooter = steady(boxes(clip[key]))
    ball = boxes(clip.get("ball") or {})

    if not np.isfinite(ball).any():
        result["status"] = "no_ball"
        return result

    candidates = release_candidates(shooter, ball)
    selected = release_frame(shooter, ball)

    # Fail rather than silently disagreeing with the frozen algorithm.
    expected = candidates[-1] if candidates else None
    if selected != expected:
        raise ReleaseDetectorMismatchError("Candidate scan disagrees with frozen release detector")

    result["candidate_count"] = len(candidates)
    result["episode_count"] = len(candidate_episodes(candidates))

    # NEW: Group nearby candidates while tolerating up to 2 missing frames.
    result["gap_tolerant_episode_count"] = len(candidate_episodes(candidates, max_missing_frames=2))

    # NEW: Measure the largest separation between consecutive candidates.
    if len(candidates) >= 2:
        gaps = np.diff(candidates)
        largest_gap = int(gaps.max())

        result["largest_candidate_gap_frames"] = largest_gap
        result["widely_separated_candidates"] = largest_gap >= 25

    result["selected_release_frame"] = selected if selected is not None else ""

    if not candidates:
        result["status"] = "release_not_found"
        return result

    result["earliest_candidate"] = candidates[0]
    result["latest_candidate"] = candidates[-1]
    result["candidate_span_frames"] = candidates[-1] - candidates[0]
    result["candidate_frames"] = json.dumps(candidates)
    result["status"] = (
        "multiple_candidate_episodes" if result["episode_count"] > 1 else "single_candidate_episode"
    )
    return result


def scan_directory(source: Path) -> list[dict]:
    """Scan local JSON files without editing them or reading shot outcomes."""
    if not source.is_dir():
        raise ValueError(f"Input directory does not exist: {source}")

    files = sorted(source.rglob("*.json"))
    if not files:
        raise ValueError(f"No JSON trajectory files found: {source}")

    results = []
    for path in files:
        try:
            with path.open(encoding="utf-8") as stream:
                clip = json.load(stream)
            if not isinstance(clip, dict):
                raise ValueError("Expected a JSON object")
            result = diagnose_clip(clip)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            result = {
                "status": "invalid_json_or_trajectory",
                "error": str(exc),
            }

        results.append(
            {
                "clip_path": str(path.relative_to(source)),
                **result,
            }
        )

    return results


def write_report(results: list[dict], output: Path) -> None:
    """Write a new CSV, refusing to overwrite an existing report."""
    fields = [
        "clip_path",
        "shooter_key",
        "status",
        "selected_release_frame",
        "candidate_count",
        "episode_count",
        "earliest_candidate",
        "latest_candidate",
        "candidate_span_frames",
        "candidate_frames",
        "error",
        "gap_tolerant_episode_count",
        "largest_candidate_gap_frames",
        "widely_separated_candidates",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit release candidates in BasketEvent JSONs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = scan_directory(args.input)
    write_report(results, args.output)

    print(f"Clips scanned: {len(results)}")
    for status, count in sorted(Counter(r["status"] for r in results).items()):
        print(f"{status}: {count}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
