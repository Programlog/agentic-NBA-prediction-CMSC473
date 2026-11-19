"""Pre-release CoTracker3 motion pilot for annotated NBA shot clips.

Run: python -m nba_shots.h1_motion_tracker --manifest data/basketevent/batch_01_manifest.csv --output artifacts/cotracker3_pilot --limit 2

The release frame is the final input frame. No post-release video is decoded or
passed to the tracker. CoTracker point tracks are *not* automatically pose joints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_box(value: object) -> np.ndarray | None:
    """Return a valid xyxy box or None for an unannotated/missing box."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid box JSON") from exc
    box = np.asarray(value, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        raise ValueError("Box must contain finite [x1,y1,x2,y2] with positive area")
    return box


def seed_queries(boxes: dict[str, np.ndarray], height: int, width: int) -> tuple[np.ndarray, list[str]]:
    """Create 3x3 interior point seeds per annotated person at the release frame."""
    if width < 2 or height < 2:
        raise ValueError("Frame must be at least 2x2")
    positions, labels = [], []
    for name, box in boxes.items():
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f"{name} box extends outside frame")
        for fy in (0.25, 0.5, 0.75):
            for fx in (0.25, 0.5, 0.75):
                positions.append((x1 + fx * (x2 - x1), y1 + fy * (y2 - y1)))
                labels.append(name)
    if not positions:
        raise ValueError("At least one annotated person is required")
    return np.asarray(positions, dtype=np.float32), labels


def summarize_tracks(tracks: np.ndarray, visible: np.ndarray, labels: list[str], width: int, height: int) -> dict:
    """Compute descriptive point-motion proxies using visible tracks only."""
    tracks = np.asarray(tracks, dtype=float)
    visible = np.asarray(visible).astype(bool)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must be [time, points, xy]")
    t, n, _ = tracks.shape
    if t < 2 or visible.shape != (t, n) or len(labels) != n:
        raise ValueError("Visibility, labels and track dimensions do not match")
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be positive")
    result = {}
    for name in ("shooter", "defender"):
        ids = [i for i, label in enumerate(labels) if label == name]
        prefix = f"cotracker_{name}"
        result[f"{prefix}_points"] = len(ids)
        result[f"{prefix}_visibility_fraction"] = (
            float(np.mean(visible[:, ids])) if ids else np.nan
        )
        result[f"{prefix}_endpoint_visible_points"] = 0
        result[f"{prefix}_median_displacement_norm"] = np.nan
        result[f"{prefix}_median_step_norm"] = np.nan
        if not ids:
            continue
        xy = tracks[:, ids, :]
        good = visible[:, ids] & np.isfinite(xy).all(axis=2)
        endpoint = good[0] & good[-1]
        result[f"{prefix}_endpoint_visible_points"] = int(endpoint.sum())
        diagonal = float(np.hypot(width, height))
        if endpoint.any():
            displacement = np.linalg.norm(xy[-1, endpoint] - xy[0, endpoint], axis=1)
            result[f"{prefix}_median_displacement_norm"] = float(np.median(displacement) / diagonal)
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=2)
        valid_steps = good[:-1] & good[1:]
        if valid_steps.any():
            result[f"{prefix}_median_step_norm"] = float(np.median(steps[valid_steps]) / diagonal)
    return result


def read_prerelease_window(path: Path, release_frame: int, lookback: int, max_side: int):
    """Decode only frames [release-lookback, release], optionally resize."""
    import cv2

    if release_frame < 1 or lookback < 1 or max_side < 64:
        raise ValueError("release_frame >= 1, lookback >= 1, max_side >= 64 required")
    start = max(0, release_frame - lookback)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for expected in range(start, release_frame + 1):
            ok, frame = cap.read()
            actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
            if not ok or actual != expected:
                raise ValueError(f"Failed to decode expected frame {expected}; got {actual}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    original_h, original_w = frames[0].shape[:2]
    scale = min(1.0, max_side / max(original_h, original_w))
    w, h = max(2, round(original_w * scale)), max(2, round(original_h * scale))
    if (w, h) != (original_w, original_h):
        frames = [cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA) for frame in frames]
    return np.stack(frames), (original_w, original_h), start


def run(manifest: Path, output: Path, *, limit: int = 2, lookback: int = 12,
        max_side: int = 384, device: str = "cpu") -> dict:
    """Run offline CoTracker3 once per annotated shot; preserve failed shots."""
    if limit < 1 or lookback < 1 or device not in {"cpu", "cuda"}:
        raise ValueError("limit/lookback must be positive and device must be cpu/cuda")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new, empty output directory")
    frame = pd.read_csv(manifest, dtype={"shot_id": str, "game_id": str})
    required = {"shot_id", "clip_path", "release_frame", "shooter_box", "defender_box"}
    if not required <= set(frame):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(frame))}")
    if frame.shot_id.isna().any() or frame.shot_id.duplicated().any():
        raise ValueError("Manifest needs unique nonempty shot IDs")

    # Imports/weight loading are deferred until the command is actually executed.
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    rows = []
    for shot in frame.head(limit).to_dict("records"):
        rec = {"shot_id": shot["shot_id"], "status": "error", "error": None,
               "window_start_frame": None, "latest_frame_used": None}
        try:
            release = float(shot["release_frame"])
            if not np.isfinite(release) or not release.is_integer():
                raise ValueError("Missing or non-integer release_frame")
            release = int(release)
            shooter_box = parse_box(shot["shooter_box"])
            defender_box = parse_box(shot["defender_box"])
            if shooter_box is None:
                raise ValueError("Missing shooter_box annotation")
            boxes = {"shooter": shooter_box}
            if defender_box is not None:
                boxes["defender"] = defender_box
            clip = Path(str(shot["clip_path"]))
            if not clip.is_absolute():
                clip = manifest.parent / clip
            if not clip.is_file():
                raise ValueError(f"Missing video: {clip}")
            video, (original_w, original_h), start = read_prerelease_window(
                clip, release, lookback, max_side
            )
            h, w = video.shape[1:3]
            scaled_boxes = {
                name: np.array(box) * np.array([w / original_w, h / original_h] * 2)
                for name, box in boxes.items()
            }
            points, labels = seed_queries(scaled_boxes, h, w)
            # Queries [time,x,y] are anchored at the last (release) frame.
            # Tracking backward uses ONLY frames before and at release.
            query = np.column_stack([np.full(len(points), len(video) - 1), points])
            tensor = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)[None].float().to(device)
            with torch.inference_mode():
                tracks, visibility = model(
                    tensor,
                    queries=torch.from_numpy(query.astype(np.float32))[None].to(device),
                    backward_tracking=True,
                )
            xy = tracks[0].detach().cpu().numpy()
            vis = visibility[0].detach().cpu().numpy()
            if xy.shape != (len(video), len(points), 2):
                raise ValueError(f"Unexpected CoTracker track shape: {xy.shape}")
            if vis.shape == (len(video), len(points), 1):
                vis = vis[..., 0]
            rec.update(summarize_tracks(xy, vis, labels, w, h))
            rec.update({"status": "ok", "window_start_frame": start,
                        "latest_frame_used": release, "n_frames": len(video),
                        "width": w, "height": h})
        except Exception as exc:
            rec["error"] = str(exc)
        rows.append(rec)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "cotracker_features.csv", index=False)
    summary = {
        "shots_attempted": len(table), "shots_succeeded": int(table.status.eq("ok").sum()),
        "shots_failed": int(table.status.eq("error").sum()),
        "pre_release_only": True, "lookback_frames": lookback,
        "max_side": max_side, "device": device,
        "note": "2D image-point motion proxies only. Camera movement may dominate; no outcome or post-release frames used.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
