"""H1 measurement check, as written in docs/h1_measurement_check.md. Reads no shot outcomes.

    python scripts/h1_measurement_check.py measure COHORT COLLECTION OUT --stage pose|track
    python scripts/h1_measurement_check.py summarize COHORT OUT

`measure` downloads one video at a time, measures it, deletes it, and appends to
OUT/<stage>.jsonl, so it can be stopped and resumed. The pose pass (YOLO, ~1 s per shot) is the
release-height check; the track pass (CoTracker3, ~30 s per shot on CPU) checks jump and ball
rise. [--limit N] caps shots per run. `--clips DIR` reads local videos laid
out like the dataset (DIR/<split>/<game>/<clip>.mp4) instead of downloading.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots import basketevent as be

STANCE = (20, 10)  # box stance frames r-20 .. r-10, as in basketevent._stance
RISE = be.FLIGHT_FRAMES  # 4
MAX_SIDE = 384
MIN_CONF = 0.5
MIN_IOU = 0.5
SHOULDERS, HIPS, WRISTS, ANKLES = (5, 6), (11, 12), (9, 10), (15, 16)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def pose_measures(keypoints: np.ndarray, ball_centre: np.ndarray) -> dict:
    """Pose release height and ball-to-wrist distance, in torso lengths, from one person."""

    def mean_of(idx):
        pts = [keypoints[i, :2] for i in idx if keypoints[i, 2] >= MIN_CONF]
        return np.mean(pts, axis=0) if pts else None

    shoulders, hips, ankles = mean_of(SHOULDERS), mean_of(HIPS), mean_of(ANKLES)
    wrists = [keypoints[i, :2] for i in WRISTS if keypoints[i, 2] >= MIN_CONF]
    if shoulders is None or hips is None:
        return {"pose_status": "no_torso"}
    torso = float(np.linalg.norm(shoulders - hips))
    if torso < 1:
        return {"pose_status": "no_torso"}
    if not wrists:
        return {"pose_status": "no_wrist", "torso_px": torso}
    wrist = min(wrists, key=lambda p: p[1])  # higher on screen = smaller y
    out = {"torso_px": torso, "pose_status": "ok"}
    out["ball_to_wrist_torso"] = float(np.linalg.norm(ball_centre - wrist) / torso)
    if ankles is None:
        out["pose_status"] = "no_ankle"
        return out
    out["pose_release_height_torso"] = float((ankles[1] - wrist[1]) / torso)
    return out


def box_measures(shooter: np.ndarray, ball: np.ndarray, r: int) -> dict:
    """Box jump and ball rise in full-resolution pixels (the bh features before scaling)."""
    out = {}
    stance = shooter[max(0, r - STANCE[0]) : max(0, r - STANCE[1])]
    feet_r = be._feet(shooter[r])[1]
    if np.isfinite(stance).any() and np.isfinite(feet_r):
        out["box_jump_px"] = float(np.nanmedian(be._feet(stance)[:, 1]) - feet_r)
    later = be._at(ball, r + RISE)
    if np.isfinite(ball[r]).all() and np.isfinite(later).all():
        out["box_rise_px"] = float(be._centre(ball[r])[1] - be._centre(later)[1])
    return out


def query_points(shooter_box, ball_box, all_boxes, width, height):
    """Shooter lower-body points, ball centre, and background grid points outside player boxes."""
    x, y, w, h = shooter_box
    pts = [(x + f * w, y + 0.85 * h) for f in (0.3, 0.4, 0.5, 0.6, 0.7)]
    labels = ["shooter"] * 5
    pts.append(tuple(be._centre(ball_box)))
    labels.append("ball")
    for gx in np.linspace(0.08, 0.92, 6) * width:
        for gy in np.linspace(0.1, 0.9, 4) * height:
            inside = any(
                bx - 0.05 * bw <= gx <= bx + 1.05 * bw and by - 0.05 * bh <= gy <= by + 1.05 * bh
                for bx, by, bw, bh in all_boxes
            )
            if not inside:
                pts.append((gx, gy))
                labels.append("background")
    return np.asarray(pts, dtype=np.float32), np.asarray(labels)


def displacement(xy, vis, idx, early: slice, at: int) -> float:
    """Median over points of (median y over visible early frames - y at `at`), visible at both."""
    values = []
    for p in idx:
        early_vis = vis[early, p]
        if vis[at, p] and early_vis.any():
            values.append(np.median(xy[early, p, 1][early_vis]) - xy[at, p, 1])
    return float(np.median(values)) if values else np.nan


def _track(model, frames: np.ndarray, points: np.ndarray, scale: float):
    """Forward-only CoTracker3 from frame 0 of `frames`; returns full-res xy and visibility."""
    import torch

    query = np.column_stack([np.zeros(len(points)), points * scale]).astype(np.float32)
    video = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None].float()
    with torch.inference_mode():
        tracks, visible = model(video, queries=torch.from_numpy(query)[None])
    vis = visible[0].numpy()
    return tracks[0].numpy() / scale, (vis[..., 0] if vis.ndim == 3 else vis) > 0.5


def track_measures(model, frames: np.ndarray, r_idx: int, points, labels, scale: float) -> dict:
    """Two forward runs from the release frame: r back to r-20 (time reversed), and r to r+4.

    Equivalent to one run queried at r with backward tracking, at half the CPU cost.
    """
    shooter = np.flatnonzero(labels == "shooter")
    background = np.flatnonzero(labels == "background")
    ball = int(np.flatnonzero(labels == "ball")[0])
    out = {"background_points": len(background)}
    if r_idx >= STANCE[1]:
        back = frames[: r_idx + 1][::-1]  # index k = frame r - k
        sel = np.concatenate([shooter, background])
        xy, vis = _track(model, back, points[sel], scale)
        xy, vis = xy[::-1], vis[::-1]  # back to forward time: index r_idx = release
        n = len(shooter)
        early = slice(max(0, r_idx - STANCE[0]), r_idx - STANCE[1])
        out["track_jump_px"] = displacement(xy, vis, range(n), early, r_idx)
        camera = displacement(xy, vis, range(n, len(sel)), early, r_idx)
        out["camera_jump_px"] = camera
        out["track_jump_corrected_px"] = out["track_jump_px"] - camera
    if r_idx + RISE < len(frames):
        xy, vis = _track(model, frames[r_idx : r_idx + RISE + 1], points[[ball]], scale)
        if vis[0, 0] and vis[RISE, 0]:
            out["track_rise_px"] = float(xy[0, 0, 1] - xy[RISE, 0, 1])
    return out


def read_frames(path: Path, start: int, stop: int):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stop = min(stop, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for expected in range(start, stop + 1):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"failed to decode frame {expected}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), total


def measure_shot(row, clip: dict, video: Path, pose_model, track_model) -> dict:
    shooter_key = next(
        k
        for k, v in clip.items()
        if k != "ball"
        and isinstance(v, dict)
        and isinstance(v.get("event"), dict)
        and v["event"].get("actionType") in be.SHOT_ACTIONS
    )
    shooter = be.steady(be.boxes(clip[shooter_key]))
    ball = be.boxes(clip.get("ball") or {})
    r = int(row.release_frame)
    out = {"release_frame": r, **box_measures(shooter, ball, r)}
    start = max(0, r - STANCE[0])
    frames, total = read_frames(video, start, r + RISE)
    out["frames_match"] = total == len(shooter)
    if not out["frames_match"]:
        return {**out, "status": "frame_count_mismatch"}
    r_idx = r - start
    height, width = frames.shape[1:3]
    if pose_model is not None:
        out.update(_pose(pose_model, frames[r_idx], shooter[r], ball[r]))
    if track_model is None:
        return {**out, "status": "ok"}
    return {**out, **_cotracker(track_model, clip, frames, r, r_idx, shooter, ball), "status": "ok"}


def _pose(pose_model, frame, shooter_box, ball_box) -> dict:
    """YOLO pose on the full-resolution release frame, matched to the box shooter by IoU."""
    out = {}
    result = pose_model.predict(frame[..., ::-1], verbose=False, device="cpu")[0]  # RGB->BGR
    sx1, sy1, sw, sh = shooter_box
    target = (sx1, sy1, sx1 + sw, sy1 + sh)
    best, best_iou = None, 0.0
    if result.keypoints is not None and len(result.boxes):
        for i, box in enumerate(result.boxes.xyxy.numpy()):
            overlap = iou(box, target)
            if overlap > best_iou:
                best, best_iou = i, overlap
    out["pose_iou"] = best_iou
    if best is None or best_iou < MIN_IOU:
        out["pose_status"] = "no_matching_person"
    else:
        kp = result.keypoints.data.numpy()[best]
        out.update(pose_measures(kp, be._centre(ball_box)))
    return out


def _cotracker(track_model, clip, frames, r, r_idx, shooter, ball) -> dict:
    """CoTracker3 on the r-20 .. r+4 window, downscaled."""
    import cv2

    height, width = frames.shape[1:3]
    others = [
        be.boxes(v)[r]
        for k, v in clip.items()
        if k != "ball" and isinstance(v, dict) and len(be.boxes(v)) > r
    ]
    others = [b for b in others if np.isfinite(b).all()]
    points, labels = query_points(shooter[r], ball[r], others, width, height)
    scale = min(1.0, MAX_SIDE / max(width, height))
    small = np.stack(
        [
            cv2.resize(f, (round(width * scale), round(height * scale)), cv2.INTER_AREA)
            for f in frames
        ]
    )
    return track_measures(track_model, small, r_idx, points, labels, scale)


def measure(
    cohort: Path,
    collection: Path,
    out: Path,
    stage: str,
    limit: int | None,
    clips: Path | None,
    sample: Path | None = None,
):
    import torch
    from huggingface_hub import hf_hub_download

    torch.set_num_threads(int(os.environ.get("H1_CHECK_THREADS", "4")))
    out.mkdir(parents=True, exist_ok=True)
    log = out / f"{stage}.jsonl"
    done = set()
    if log.exists():
        # Errors (e.g. a failed download) are retried; measured shots are not redone.
        latest = {}
        for line in log.read_text().splitlines():
            if line:
                record = json.loads(line)
                latest[record["shot_id"]] = record["status"]
        done = {shot for shot, status in latest.items() if status != "error"}
    revision = json.loads((collection / "source.json").read_text())["revision"]
    features = pd.read_parquet(cohort / "box_features.parquet")
    shots = set(pd.read_parquet(cohort / "shots.parquet", columns=["shot_id"]).shot_id)
    rows = features.loc[
        features.has_video & features.box_status.eq("ok") & features.shot_id.isin(shots)
    ].sort_values("shot_id")
    if clips is not None:  # pilot on local videos only
        rows = rows.loc[rows["clip"].map(lambda c: (clips / c.replace(".json", ".mp4")).exists())]
    if sample is not None:  # a pre-drawn list of shot IDs, one per line
        wanted = set(sample.read_text().split())
        if not wanted <= set(rows.shot_id):
            raise ValueError("Sample contains shots that are not cohort video shots")
        rows = rows.loc[rows.shot_id.isin(wanted)]
    rows = rows.loc[~rows.shot_id.isin(done)].head(limit)
    pose_model = track_model = None
    if stage == "pose":
        from ultralytics import YOLO

        pose_model = YOLO("yolo11x-pose.pt")
    else:
        track_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").eval()
    labels_cache: dict[str, dict] = {}
    for row in rows.itertuples():
        record = {"shot_id": row.shot_id, "clip": row.clip, "revision": revision}
        tmp = Path(tempfile.mkdtemp(prefix="h1check_"))
        try:
            game = row.clip.split("/")[1]
            if game not in labels_cache:
                labels_cache.clear()  # one game in memory at a time
                path = collection / "games" / f"{game}_labels.json.gz"
                with gzip.open(path, "rt") as stream:
                    labels_cache[game] = json.load(stream)
            mp4 = row.clip.replace(".json", ".mp4")
            if clips is not None:
                video = clips / mp4
            else:
                video = Path(
                    hf_hub_download(
                        be.REPO, mp4, repo_type="dataset", revision=revision, local_dir=tmp
                    )
                )
            record.update(
                measure_shot(row, labels_cache[game][row.clip], video, pose_model, track_model)
            )
        except Exception as error:  # kept and counted, never dropped
            record.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        with log.open("a") as stream:
            stream.write(json.dumps(record, default=float) + "\n")
        print(row.shot_id, record.get("status"), record.get("pose_status"), flush=True)


def spearman_ci(a: np.ndarray, b: np.ndarray, reps=1000, seed=42):
    from scipy.stats import spearmanr

    rho = float(spearmanr(a, b).statistic)
    rng = np.random.default_rng(seed)
    boot = [
        spearmanr(a[i], b[i]).statistic
        for i in (rng.integers(0, len(a), len(a)) for _ in range(reps))
    ]
    return rho, float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))


def agreement(table: pd.DataFrame, box: str, other: str) -> dict:
    if box not in table or other not in table:
        return {"measured": 0, "verdict": "not run"}
    pair = table[[box, other]].dropna()
    result = {"measured": len(pair)}
    if len(pair) < 100:
        return {**result, "verdict": "inconclusive (<100 shots)"}
    rho, lo, hi = spearman_ci(pair[box].to_numpy(), pair[other].to_numpy())
    passed = rho >= 0.5 and lo > 0.3
    return {**result, "rho": rho, "ci": [lo, hi], "verdict": "pass" if passed else "fail"}


def summarize(cohort: Path, out: Path) -> dict:
    def load(stage):
        path = out / f"{stage}.jsonl"
        if not path.exists():
            return pd.DataFrame(columns=["shot_id"])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        return pd.DataFrame(rows).drop_duplicates("shot_id", keep="last")

    pose, track = load("pose"), load("track")
    # Box values and frame checks are recomputed identically in both passes; status is per pass.
    track = track.rename(columns={"status": "track_status", "error": "track_error"})
    shared = [c for c in track if c in pose and c != "shot_id"]
    table = pose.merge(track.drop(columns=shared), on="shot_id", how="outer")
    feats = pd.read_parquet(cohort / "box_features.parquet")
    shots = pd.read_parquet(cohort / "shots.parquet", columns=["shot_id", "shot_family", "blocked"])
    table = table.merge(feats[["shot_id", "release_height_bh"]], on="shot_id", how="left")
    table = table.merge(shots, on="shot_id", how="left")
    result = {}
    for name, part in {
        "all_video_shots": table,
        "d0_unblocked_jump": table.loc[table.shot_family.eq("jump") & ~table.blocked.astype(bool)],
    }.items():
        near = part.get("ball_to_wrist_torso", pd.Series(dtype=float)).dropna()
        share = float((near <= 1).mean()) if len(near) else None
        result[name] = {
            "shots": len(part),
            "status": part.status.value_counts().to_dict(),
            "pose_status": part.get("pose_status", pd.Series(dtype=str)).value_counts().to_dict(),
            "release_height": agreement(part, "release_height_bh", "pose_release_height_torso"),
            "ball_at_hands": {
                "measured": len(near),
                "share_within_1_torso": share,
                "verdict": "inconclusive (<100 shots)"
                if len(near) < 100
                else ("pass" if share >= 0.7 else "fail"),
            },
            "jump": agreement(part, "box_jump_px", "track_jump_px"),
            "jump_camera_corrected_descriptive": agreement(
                part, "box_jump_px", "track_jump_corrected_px"
            ),
            "camera_jump_px_median_abs": float(part.get("camera_jump_px").abs().median())
            if "camera_jump_px" in part
            else None,
            "ball_rise": agreement(part, "box_rise_px", "track_rise_px"),
        }
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("cohort", type=Path)
    m.add_argument("collection", type=Path)
    m.add_argument("out", type=Path)
    m.add_argument("--stage", choices=["pose", "track"], required=True)
    m.add_argument("--limit", type=int)
    m.add_argument("--clips", type=Path)
    m.add_argument("--sample", type=Path, help="File of shot IDs to measure, one per line")
    s = sub.add_parser("summarize")
    s.add_argument("cohort", type=Path)
    s.add_argument("out", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        measure(
            args.cohort, args.collection, args.out, args.stage, args.limit, args.clips, args.sample
        )
    else:
        print(json.dumps(summarize(args.cohort, args.out), indent=2, default=float))


if __name__ == "__main__":
    main()
