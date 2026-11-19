"""Frame-by-frame CoTracker3 audit videos using the H1 extractor's exact seed grid.

Run: python -m nba_shots.h1_track_visualizer --manifest data/basketevent/batch_01_manifest.csv --output artifacts/cotracker3_audit_v2 --limit 10
Only pre-release frames (including release) are decoded or passed to the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .h1_motion_tracker import parse_box, read_prerelease_window, seed_queries, summarize_tracks

# OpenCV BGR: green shooter, red defender.
COLORS = {"shooter": (0, 255, 0), "defender": (0, 0, 255)}


def prepare_queries(boxes: dict[str, np.ndarray], size: tuple[int, int], original: tuple[int, int],
                    n_frames: int) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    """Use the *same* 25/50/75% point locations as h1_motion_tracker."""
    h, w = size
    ow, oh = original
    scaled = {name: box * np.array([w / ow, h / oh, w / ow, h / oh])
              for name, box in boxes.items()}
    points, labels = seed_queries(scaled, h, w)
    queries = np.column_stack((np.full(len(points), n_frames - 1), points)).astype(np.float32)
    return queries, labels, scaled


def render_frame(rgb: np.ndarray, tracks: np.ndarray, visible: np.ndarray,
                 labels: list[str], time_index: int, start_frame: int,
                 scaled_boxes: dict[str, np.ndarray], trail: int = 5) -> np.ndarray:
    """Draw current visible points and short, consecutive-visible past tracks only."""
    if trail < 0:
        raise ValueError("trail must be nonnegative")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected RGB frame")
    t, n, dim = tracks.shape
    if dim != 2 or visible.shape != (t, n) or len(labels) != n or not 0 <= time_index < t:
        raise ValueError("Mismatched tracks, visibility, labels, or frame index")
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for j, name in enumerate(labels):
        if not visible[time_index, j] or not np.isfinite(tracks[time_index, j]).all():
            continue
        color = COLORS[name]
        end = time_index
        begin = end
        while begin > max(0, end - trail):
            if not visible[begin-1, j] or not np.isfinite(tracks[begin-1, j]).all():
                break
            begin -= 1
        if begin < end:
            segment = np.rint(tracks[begin:end+1, j]).astype(np.int32)
            cv2.polylines(canvas, [segment], False, color, 1, cv2.LINE_AA)
        x, y = np.rint(tracks[time_index, j]).astype(int)
        cv2.circle(canvas, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)

    # The annotated boxes belong to the RELEASE frame only. Do not pretend they
    # are verified detection boxes on earlier frames.
    if time_index == t - 1:
        for name, box in scaled_boxes.items():
            x1, y1, x2, y2 = np.rint(box).astype(int)
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)), COLORS[name], 1)
            cv2.putText(canvas, name, (int(x1), max(14, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS[name], 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(canvas, f"Frame {start_frame + time_index} | release {start_frame + t - 1} | green: shooter, red: defender",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def save_video(path: Path, video: np.ndarray, tracks: np.ndarray, visible: np.ndarray,
               labels: list[str], start_frame: int, boxes: dict[str, np.ndarray],
               *, trail: int = 5, fps: float = 8.0) -> None:
    if len(video) != len(tracks) or fps <= 0:
        raise ValueError("Video frames and tracks must align; fps must be positive")
    h, w = video.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create MP4 video: {path}")
    try:
        for i, frame in enumerate(video):
            writer.write(render_frame(frame, tracks, visible, labels, i, start_frame, boxes, trail))
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Video writer produced no file: {path}")


def run(manifest: Path, output: Path, *, limit: int = 10, lookback: int = 12,
        max_side: int = 384, device: str = "cpu", trail: int = 5) -> dict:
    manifest, output = Path(manifest), Path(output)
    if limit < 1 or lookback < 1 or max_side < 64 or trail < 0 or device not in ("cpu", "cuda"):
        raise ValueError("Invalid run arguments")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new, empty output directory to preserve previous audits")
    df = pd.read_csv(manifest, dtype={"shot_id": str})
    required = {"shot_id", "clip_path", "release_frame", "shooter_box", "defender_box"}
    if not required <= set(df):
        raise ValueError(f"Missing manifest columns: {sorted(required - set(df))}")
    if df.shot_id.isna().any() or df.shot_id.duplicated().any():
        raise ValueError("Manifest shot_id values must be nonempty and unique")
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()
    videos = output / "videos"
    videos.mkdir(parents=True)
    rows = []
    selected = df.head(limit).to_dict("records")
    for idx, shot in enumerate(selected, 1):
        shot_id = str(shot["shot_id"])
        print(f"[{idx}/{len(selected)}] Processing {shot_id}...", flush=True)
        rec = {"shot_id": shot_id, "status": "error", "error": "", "video_path": ""}
        path = None
        try:
            release = float(shot["release_frame"])
            if not np.isfinite(release) or not release.is_integer():
                raise ValueError("release_frame must be an integer")
            release = int(release)
            shooter = parse_box(shot["shooter_box"])
            defender = parse_box(shot["defender_box"])
            if shooter is None:
                raise ValueError("Missing shooter annotation")
            boxes = {"shooter": shooter}
            if defender is not None:
                boxes["defender"] = defender
            clip = Path(str(shot["clip_path"]))
            if not clip.is_absolute():
                clip = manifest.parent / clip
            if not clip.is_file():
                raise FileNotFoundError(f"Missing clip: {clip}")
            video, original, start = read_prerelease_window(clip, release, lookback, max_side)
            h, w = video.shape[1:3]
            queries, labels, scaled_boxes = prepare_queries(boxes, (h, w), original, len(video))
            tensor = torch.from_numpy(video.copy()).permute(0, 3, 1, 2)[None].float().to(device)
            with torch.inference_mode():
                tracks, visibility = model(tensor, queries=torch.from_numpy(queries)[None].to(device),
                                           backward_tracking=True)
            xy = tracks[0].detach().cpu().numpy()
            vis = visibility[0].detach().cpu().numpy()
            if vis.shape == (len(video), len(labels), 1):
                vis = vis[..., 0]
            if xy.shape != (len(video), len(labels), 2) or vis.shape != (len(video), len(labels)):
                raise ValueError("Unexpected CoTracker output shape")
            vis = np.asarray(vis) > 0.5
            name = shot_id.replace(":", "_") + ".mp4"
            path = videos / name
            save_video(path, video, xy, vis, labels, start, scaled_boxes, trail=trail)
            rec.update(summarize_tracks(xy, vis, labels, w, h))
            rec.update({"status": "ok", "video_path": str(path),
                        "window_start_frame": start, "latest_frame_used": release,
                        "n_frames": len(video), "n_points": len(labels)})
            print(f"[{idx}/{len(selected)}] Saved {name}", flush=True)
        except Exception as exc:
            if path is not None:
                path.unlink(missing_ok=True)
            rec["error"] = str(exc)
            print(f"[{idx}/{len(selected)}] Error: {exc}", flush=True)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(output / "tracking_audit.csv", index=False)
    summary = {"shots_attempted": len(rows), "shots_succeeded": sum(r["status"] == "ok" for r in rows),
               "shots_failed": sum(r["status"] != "ok" for r in rows),
               "pre_release_only": True, "seed_grid": "shared h1_motion_tracker.seed_queries (25/50/75 percent)",
               "lookback_frames": lookback, "max_side": max_side, "device": device,
               "note": "Manual inspection required; visibility and screen motion are not validated player identity or physical movement."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=384)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--trail", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
