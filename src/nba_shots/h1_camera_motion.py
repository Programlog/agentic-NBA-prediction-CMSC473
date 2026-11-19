"""Exploratory H1 camera translation correction for pre-release CoTracker3 tracks.

Run: python -m nba_shots.h1_camera_motion --manifest ... --output ... --limit 2

Uses optical flow on background corners to estimate frame-to-frame image
translation. This is NOT a calibrated physical velocity or camera homography.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .h1_motion_tracker import parse_box, read_prerelease_window, seed_queries, summarize_tracks


def background_mask(shape: tuple[int, int], boxes: dict[str, np.ndarray],
                    padding: int = 20) -> np.ndarray:
    """Mask out release-frame players and a narrow boundary/scoreboard band."""
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    margin = max(8, int(min(height, width) * 0.06))
    mask[:margin] = 0
    mask[-margin:] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0
    for box in boxes.values():
        x1, y1, x2, y2 = np.asarray(box, dtype=float)
        a, b = max(0, int(x1) - padding), max(0, int(y1) - padding)
        c, d = min(width, int(np.ceil(x2)) + padding), min(height, int(np.ceil(y2)) + padding)
        mask[b:d, a:c] = 0
    return mask


def estimate_camera_translation(video: np.ndarray, boxes: dict[str, np.ndarray],
                                *, max_corners: int = 180,
                                min_points: int = 8,
                                max_mad_px: float = 4.0) -> tuple[np.ndarray, dict]:
    """Track background corners backward and return cumulative release-relative shifts.

    Translation[t] is the image displacement of static scene points from the
    first input frame to frame t. It is zero at t=0. For too few/stable
    background tracks, the correction is invalid and NaNs are returned.
    """
    if video.ndim != 4 or video.shape[-1] != 3 or len(video) < 2:
        raise ValueError("Expected RGB frames [T,H,W,3], T >= 2")
    if min_points < 3 or max_mad_px <= 0:
        raise ValueError("min_points must be >= 3 and max_mad_px positive")
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video]
    height, width = gray[0].shape
    corners = cv2.goodFeaturesToTrack(
        gray[-1], maxCorners=max_corners, qualityLevel=0.01,
        minDistance=6, mask=background_mask((height, width), boxes),
        blockSize=5,
    )
    counts = []
    mad_values = []
    shifts = np.zeros((len(video)-1, 2), dtype=float)
    if corners is None or len(corners) < min_points:
        return np.full((len(video), 2), np.nan), {
            "camera_valid": False, "camera_reason": "insufficient_background_corners",
            "camera_seed_points": 0 if corners is None else int(len(corners)),
            "camera_min_flow_points": 0, "camera_max_flow_mad_px": None,
        }
    # Track each adjacent pair backward, retaining only surviving corners.
    current = corners.astype(np.float32)
    for t in range(len(gray)-1, 0, -1):
        previous, status, _ = cv2.calcOpticalFlowPyrLK(
            gray[t], gray[t-1], current, None, winSize=(21, 21),
            maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if previous is None or status is None:
            break
        good = status.reshape(-1).astype(bool)
        good &= np.isfinite(previous.reshape(-1, 2)).all(axis=1)
        good &= np.isfinite(current.reshape(-1, 2)).all(axis=1)
        before = previous[good].reshape(-1, 2)
        after = current[good].reshape(-1, 2)
        if len(before) < min_points:
            counts.append(len(before))
            break
        delta = after - before  # positive: image shifted from earlier to later
        median = np.median(delta, axis=0)
        residual = np.linalg.norm(delta - median, axis=1)
        mad = float(np.median(residual))
        counts.append(len(before))
        mad_values.append(mad)
        if mad > max_mad_px:
            break
        shifts[t-1] = median
        current = before.reshape(-1, 1, 2).astype(np.float32)
    valid = len(counts) == len(shifts) and min(counts) >= min_points and (
        not mad_values or max(mad_values) <= max_mad_px
    )
    meta = {
        "camera_valid": bool(valid),
        "camera_reason": "ok" if valid else "weak_or_inconsistent_background_flow",
        "camera_seed_points": int(len(corners)),
        "camera_min_flow_points": int(min(counts)) if counts else 0,
        "camera_max_flow_mad_px": float(max(mad_values)) if mad_values else None,
    }
    if not valid:
        return np.full((len(video), 2), np.nan), meta
    correction = np.concatenate([np.zeros((1, 2)), np.cumsum(shifts, axis=0)], axis=0)
    return correction, meta


def corrected_metrics(xy: np.ndarray, visible: np.ndarray, labels: list[str],
                      width: int, height: int, correction: np.ndarray,
                      correction_valid: bool) -> dict:
    """Preserve raw motion and separately calculate translation-corrected motion."""
    raw = summarize_tracks(xy, visible, labels, width, height)
    if correction_valid:
        fixed = summarize_tracks(xy - correction[:, None, :], visible, labels, width, height)
    else:
        fixed = {}
    result = dict(raw)
    for name in ("shooter", "defender"):
        for metric in ("median_displacement_norm", "median_step_norm"):
            key = f"cotracker_{name}_{metric}"
            result[f"corrected_{key}"] = fixed.get(key, np.nan)
        fraction = raw[f"cotracker_{name}_visibility_fraction"]
        endpoints = raw[f"cotracker_{name}_endpoint_visible_points"]
        # Exploratory audit flags, not established accuracy thresholds.
        result[f"{name}_tracking_flag"] = (
            "no_annotation" if raw[f"cotracker_{name}_points"] == 0 else
            "low_visibility" if not np.isfinite(fraction) or fraction < 0.60 else
            "few_endpoint_points" if endpoints < 5 else "review_required"
        )
    return result


def run(manifest: Path, output: Path, *, limit: int = 10, lookback: int = 12,
        max_side: int = 384, device: str = "cpu") -> dict:
    if limit < 1 or lookback < 1 or max_side < 64 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid run settings")
    manifest, output = Path(manifest), Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Choose a new, empty output directory")
    df = pd.read_csv(manifest, dtype={"shot_id": str})
    required = {"shot_id", "clip_path", "release_frame", "shooter_box", "defender_box"}
    if df.empty or not required <= set(df) or df.shot_id.isna().any() or df.shot_id.duplicated().any():
        raise ValueError("Manifest missing columns, has no shots, or has invalid shot IDs")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    records = []
    selected = df.head(limit).to_dict("records")
    for i, row in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Processing {row['shot_id']}...", flush=True)
        rec = {"shot_id": row["shot_id"], "status": "error", "error": "",
               "camera_valid": False, "pre_release_only": True}
        try:
            release_value = float(row["release_frame"])
            if not np.isfinite(release_value) or not release_value.is_integer():
                raise ValueError("Missing/invalid release_frame")
            release = int(release_value)
            boxes = {"shooter": parse_box(row["shooter_box"])}
            defender = parse_box(row["defender_box"])
            if boxes["shooter"] is None:
                raise ValueError("Missing shooter_box")
            if defender is not None:
                boxes["defender"] = defender
            clip = Path(str(row["clip_path"]))
            if not clip.is_absolute():
                clip = manifest.parent / clip
            if not clip.is_file():
                raise ValueError(f"Video not found: {clip}")
            video, (original_w, original_h), start = read_prerelease_window(
                clip, release, lookback, max_side
            )
            h, w = video.shape[1:3]
            scaled = {
                name: box * np.array([w/original_w, h/original_h,
                                      w/original_w, h/original_h])
                for name, box in boxes.items()
            }
            pts, labels = seed_queries(scaled, h, w)
            queries = np.column_stack([
                np.full(len(pts), len(video)-1), pts
            ]).astype(np.float32)
            tensor = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)[None].float().to(device)
            with torch.inference_mode():
                tracks, visibility = model(
                    tensor, queries=torch.from_numpy(queries)[None].to(device),
                    backward_tracking=True,
                )
            xy = tracks[0].detach().cpu().numpy()
            vis = visibility[0].detach().cpu().numpy()
            if vis.shape == (len(video), len(labels), 1):
                vis = vis[..., 0]
            if xy.shape != (len(video), len(labels), 2) or vis.shape != (len(video), len(labels)):
                raise ValueError("Unexpected tracker dimensions")
            vis = np.asarray(vis) > 0.5
            correction, camera = estimate_camera_translation(video, scaled)
            rec.update(camera)
            rec.update(corrected_metrics(xy, vis, labels, w, h, correction, camera["camera_valid"]))
            if camera["camera_valid"]:
                rec["camera_total_displacement_norm"] = (
                    float(np.linalg.norm(correction[-1]) / np.hypot(w, h))
                )
            else:
                rec["camera_total_displacement_norm"] = np.nan
            rec.update({
                "status": "ok", "window_start_frame": start,
                "latest_frame_used": release, "n_frames": len(video),
                "width": w, "height": h,
            })
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
        print(f"[{i}/{len(selected)}] {rec['status']} - camera correction "
              f"{'valid' if rec.get('camera_valid') else 'unavailable'}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output / "camera_motion_audit.csv", index=False)
    summary = {
        "shots_attempted": len(records),
        "shots_succeeded": sum(r["status"] == "ok" for r in records),
        "shots_failed": sum(r["status"] == "error" for r in records),
        "camera_correction_valid_shots": sum(
            r["status"] == "ok" and r["camera_valid"] for r in records
        ),
        "pre_release_only": True,
        "method": "background Lucas-Kanade optical flow; per-step median image translation",
        "lookback_frames": lookback,
        "max_side": max_side,
        "device": device,
        "warning": "Translation-only exploratory estimate; camera zoom/perspective, "
                   "moving background players and occlusions can bias correction. "
                   "Do not interpret screen motion as physical speed.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
