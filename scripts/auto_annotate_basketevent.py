"""Generate auditable UNVERIFIED annotation proposals from BasketEvent trajectories.

No outcome or event timestamps are used to guess release frames. Candidate frame
is inferred solely from shooter/ball tracks. A human must confirm release,
shooter/defender and shooting side before training/evaluating H1 models.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHOT_ACTIONS = {"made shot", "missed shot"}
VERIFIED_STATUS = "manual_release_verified_basketevent_boxes"


def xywh_to_xyxy(value, width: int, height: int):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, w, h = map(float, value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0:
        return None
    box = [max(0, x), max(0, y), min(width, x + w), min(height, y + h)]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return [int(round(v)) for v in box]


def trajectory_boxes(record: dict, width: int, height: int, n: int):
    trajectory = record.get("trajectory", [])
    if not isinstance(trajectory, list) or len(trajectory) != n:
        return [None] * n
    return [xywh_to_xyxy(v, width, height) for v in trajectory]


def center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def ball_candidates(shooter: list, ball: list, n: int) -> dict:
    """Conservative geometry-only proposals. Not a verified ball-release detector.

    Require the ball to be near the shooter's upper torso in preceding frames,
    then separate while moving upward. Return ranked candidates, not certainty.
    """
    scores = []
    for i in range(3, n - 3):
        if any(shooter[j] is None or ball[j] is None for j in range(i - 3, i + 4)):
            continue
        s = shooter[i]
        bw = max(s[2] - s[0], 1)
        bh = max(s[3] - s[1], 1)
        scale = max(bw, bh)
        # Upper-mid torso approximate hand vicinity; avoid exact anatomical claims.
        anchor = np.array([(s[0] + s[2]) / 2, s[1] + 0.35 * bh])
        distances = []
        for j in range(i - 3, i + 4):
            distances.append(float(np.linalg.norm(center(ball[j]) - anchor) / scale))
        before = float(np.median(distances[:3]))
        after = float(np.median(distances[4:]))
        upward = (center(ball[i - 2])[1] - center(ball[i + 2])[1]) / scale
        # Requiring separation + upward trajectory reduces casual ball proximity.
        if before <= 0.9 and after - before >= 0.15 and upward >= 0.12:
            score = (after - before) + 0.4 * upward - 0.2 * before
            scores.append((float(score), i))
    scores.sort(key=lambda item: (-item[0], item[1]))
    # Collapse adjacent peaks so confidence is not inflated by correlated frames.
    distinct = []
    for score, frame in scores:
        if all(abs(frame - j) > 5 for _, j in distinct):
            distinct.append((score, frame))
        if len(distinct) == 5:
            break
    return {"candidates": [j for _, j in distinct], "scores": [round(s, 5) for s, _ in distinct]}


def defender_proposal(data: dict, shooter_role: str, frame: int, width: int, height: int, n: int):
    shooter_rec = data[shooter_role]
    shooter_boxes = trajectory_boxes(shooter_rec, width, height, n)
    sb = shooter_boxes[frame]
    if sb is None:
        return None, None, "shooter_box_missing"
    shooter_color = str(shooter_rec.get("jersey_color") or "").strip().lower()
    opponents = []
    for role, rec in data.items():
        if role in (shooter_role, "ball") or not isinstance(rec, dict):
            continue
        color = str(rec.get("jersey_color") or "").strip().lower()
        # Only consider known, different jersey colors as opponent proxies.
        if not color or not shooter_color or color == shooter_color:
            continue
        boxes = trajectory_boxes(rec, width, height, n)
        box = boxes[frame]
        if box is None:
            continue
        dist = float(np.linalg.norm(center(sb) - center(box)) / np.hypot(width, height))
        opponents.append((dist, role, box))
    if not opponents:
        return None, None, "no_opponent_color_candidate"
    opponents.sort()
    dist, role, box = opponents[0]
    # Guard against an obviously unrelated far-away defender.
    if dist > 0.30:
        return None, None, "nearest_opponent_too_far"
    return role, box, "nearest_different_jersey_color_unverified"


def unique_shooter_role(report: pd.DataFrame, shot_id: str, source_json: str) -> str:
    matches = report.loc[
        report.shot_id.eq(shot_id)
        & report.json_path.eq(source_json)
        & report.selection_status.eq("selected")
        & report.match_status.eq("matched")
    ]
    roles = matches.source_role.dropna().unique().tolist()
    if len(roles) != 1:
        raise ValueError(f"No unique matching shooter role: {roles}")
    return str(roles[0])


def propose(row: dict, source_manifest: Path, report: pd.DataFrame, root: Path):
    import cv2

    shot_id = str(row["shot_id"])
    source_json = str(row["source_json"])
    role = unique_shooter_role(report, shot_id, source_json)
    json_file = (root / source_json).resolve()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    shooter = data.get(role)
    if not isinstance(shooter, dict):
        raise ValueError("Shooter missing from JSON")
    event = shooter.get("event")
    if not isinstance(event, dict) or str(event.get("actionType", "")).strip().lower() not in SHOT_ACTIONS:
        raise ValueError("Shooter's event is not a shot")
    clip = Path(str(row["clip_path"]))
    if not clip.is_absolute():
        clip = source_manifest.parent / clip
    clip = clip.resolve()
    if not clip.is_file():
        raise FileNotFoundError(f"Missing clip: {clip}")
    cap = cv2.VideoCapture(str(clip))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open MP4")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if n < 10 or width < 2 or height < 2:
        raise ValueError("Invalid video dimensions/frame count")
    shooter_tracks = shooter.get("trajectory")
    if not isinstance(shooter_tracks, list) or len(shooter_tracks) != n:
        raise ValueError(f"Shooter/video frame count mismatch: {len(shooter_tracks) if isinstance(shooter_tracks,list) else -1} != {n}")
    ball = data.get("ball")
    if not isinstance(ball, dict) or not isinstance(ball.get("trajectory"), list) or len(ball["trajectory"]) != n:
        raise ValueError("Ball track missing or frame count mismatch")
    shooter_boxes = trajectory_boxes(shooter, width, height, n)
    ball_boxes = trajectory_boxes(ball, width, height, n)
    result = ball_candidates(shooter_boxes, ball_boxes, n)
    candidates = result["candidates"]
    if not candidates:
        return {"status": "review_required_no_release_candidate", "source_role": role,
                "candidate_frames": [], "candidate_scores": [], "reason": "no_geometry_transition"}
    frame = candidates[0]
    sb = shooter_boxes[frame]
    if sb is None:
        raise ValueError("Shooter box missing at proposed frame")
    defender_role, defender_box, defender_reason = defender_proposal(data, role, frame, width, height, n)
    reason = "multiple_plausible_release_frames" if len(candidates) > 1 else "single_geometry_candidate_unverified"
    return {"status": "auto_proposed_review_required", "source_role": role,
            "candidate_frames": candidates, "candidate_scores": result["scores"],
            "release_frame": frame, "shooter_box": sb, "defender_role": defender_role,
            "defender_box": defender_box, "defender_reason": defender_reason,
            "shooting_side": None, "reason": reason}


def run(manifest: Path, output: Path, root: Path):
    manifest, output, root = Path(manifest), Path(output), Path(root)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    frame = pd.read_csv(manifest, dtype={"shot_id": str, "game_id": str, "game_event_id": str})
    required = {"shot_id", "clip_path", "source_json", "release_frame", "shooter_box", "defender_box", "shooting_side"}
    if missing := required - set(frame):
        raise ValueError(f"Manifest missing {sorted(missing)}")
    if frame.shot_id.isna().any() or frame.shot_id.duplicated().any():
        raise ValueError("Shot IDs must be unique and nonempty")
    report = pd.read_csv(manifest.parent / "event_matching_report.csv",
                         dtype={"shot_id": str, "json_path": str})
    output.mkdir(parents=True)
    records = []
    n_manual = n_proposed = 0
    for i, row in frame.iterrows():
        item = row.to_dict()
        audit = {"shot_id": item["shot_id"], "status": "error", "error": ""}
        try:
            if pd.notna(row["release_frame"]):
                # Never modify a manually annotated row; preserve entire row.
                audit["status"] = "preserved_existing_annotation"
                n_manual += 1
            else:
                proposal = propose(item, manifest, report, root)
                audit.update(proposal)
                audit["status"] = proposal["status"]
                if audit["status"] == "auto_proposed_review_required":
                    for col in ["release_frame", "shooter_box", "defender_box"]:
                        v = proposal.get(col)
                        if v is not None:
                            frame.at[i, col] = json.dumps(v) if isinstance(v, list) else v
                    frame.at[i, "annotation_status"] = audit["status"]
                    n_proposed += 1
                else:
                    frame.at[i, "annotation_status"] = audit["status"]
        except Exception as exc:
            audit["error"] = str(exc)
            frame.at[i, "annotation_status"] = "auto_annotation_error"
        records.append(audit)
        print(f"[{i+1}/{len(frame)}] {item['shot_id']}: {audit['status']}", flush=True)
    frame.to_csv(output / "auto_annotated_manifest.csv", index=False)
    pd.DataFrame(records).to_csv(output / "annotation_audit.csv", index=False)
    summary = {
        "source_manifest": str(manifest), "shots": len(frame),
        "existing_annotations_preserved": n_manual,
        "auto_proposals_unverified": n_proposed,
        "no_release_candidate": sum(r["status"] == "review_required_no_release_candidate" for r in records),
        "errors": sum(r["status"] == "error" for r in records),
        "critical_warning": "Automatic release frames, shooter/defender boxes are UNVERIFIED proposals. Shooting side intentionally missing. Do not train H1 from unreviewed proposals.",
        "pre_release_only_model_features": "Annotation may inspect full clip; predictive features must be computed at/before manually verified release, never after.",
    }
    (output / "annotation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data/basketevent"))
    args = parser.parse_args()
    run(args.manifest, args.output, args.source_root)


if __name__ == "__main__":
    main()
