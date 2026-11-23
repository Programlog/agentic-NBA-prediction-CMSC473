"""Human-verified BasketEvent-assisted release annotation for H1.

The script proposes boxes from BasketEvent trajectories, but NEVER infers
release frame or defender automatically. It writes a NEW manifest; source
manifest is never modified. Only the manually selected release frame and its
boxes are saved. Event outcomes are not used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

SHOT_ACTIONS = {"made shot", "missed shot"}


def xywh_to_xyxy(box, frame_width: int, frame_height: int):
    """Convert a BasketEvent [x,y,w,h] entry into a valid clipped xyxy box."""
    if box is None or len(box) != 4:
        return None
    try:
        import math
        x, y, w, h = map(float, box)
        if not all(math.isfinite(z) for z in (x, y, w, h)) or w <= 0 or h <= 0:
            return None
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(float(frame_width), x + w), min(float(frame_height), y + h)
        if x2 <= x1 or y2 <= y1:
            return None
        return [int(round(v)) for v in (x1, y1, x2, y2)]
    except (ValueError, TypeError):
        return None


def role_at_frame(data: dict, role: str, frame: int, width: int, height: int):
    if role not in data or not isinstance(data[role], dict):
        return None
    trajectory = data[role].get("trajectory")
    if not isinstance(trajectory, list) or frame < 0 or frame >= len(trajectory):
        return None
    return xywh_to_xyxy(trajectory[frame], width, height)


def source_role_for_shot(report: pd.DataFrame, shot_id: str, source_json: str) -> str:
    """Use previously audited shot-to-event matching, never guessing the shooter."""
    matches = report.loc[
        report.shot_id.eq(shot_id)
        & report.json_path.eq(source_json)
        & report.selection_status.eq("selected")
        & report.match_status.eq("matched")
    ]
    roles = matches.source_role.dropna().unique().tolist()
    if len(roles) != 1:
        raise ValueError(f"Expected unique matched event role for {shot_id}; got {roles}")
    return str(roles[0])


def resolve_clip(manifest_path: Path, path: str) -> Path:
    clip = Path(path)
    return (clip if clip.is_absolute() else manifest_path.parent / clip).resolve()


def draw(frame, data: dict, shooter_role: str, defender_role: str | None,
         current: int, release: int | None, side: str | None, shot_id: str):
    import cv2
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    # The drawn boxes are proposals, not verified annotations until saved.
    for role, payload in data.items():
        if role == "ball" or not isinstance(payload, dict):
            continue
        box = role_at_frame(data, role, current, w, h)
        if box is None:
            continue
        selected_shooter = role == shooter_role
        selected_defender = role == defender_role
        color = (70, 220, 80) if selected_shooter else ((40, 160, 255) if selected_defender else (140, 140, 140))
        thickness = 3 if (selected_shooter or selected_defender) else 1
        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), color, thickness)
        name = str(payload.get("player_name") or role)
        cv2.putText(canvas, name, (box[0], max(16, box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    lines = [
        f"Shot: {shot_id} | frame {current} | release: {release if release is not None else 'NOT SET'}",
        f"Shooter GREEN | defender ORANGE: {defender_role or 'not chosen'} | hand: {side or 'not set'}",
        "A/D: prev/next 1 | J/L: prev/next 10 | R: mark release",
        "Click player: defender | X: clear defender | 1/2: left/right hand",
        "S: save (must mark release + hand) | K: skip | Q: quit",
    ]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, min(h, 124)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (12, 20 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def annotate_one(row: dict, manifest_path: Path, report: pd.DataFrame,
                 root: Path):
    import cv2

    shot_id = str(row["shot_id"])
    json_path = str(row["source_json"])
    role = source_role_for_shot(report, shot_id, json_path)
    data = json.loads((root / json_path).read_text(encoding="utf-8"))
    event = data.get(role, {}).get("event")
    if not isinstance(event, dict) or str(event.get("actionType", "")).lower() not in SHOT_ACTIONS:
        raise ValueError(f"{shot_id}: shooter role does not refer to a shot event")
    clip = resolve_clip(manifest_path, str(row["clip_path"]))
    if not clip.is_file():
        raise FileNotFoundError(f"Missing clip: {clip}")
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise ValueError(f"Could not open clip: {clip}")
    try:
        length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        trajectory_length = len(data[role].get("trajectory", []))
        # Do not silently misalign trajectories if video differs from annotation.
        if length < 2 or length != trajectory_length:
            raise ValueError(f"Frame alignment mismatch {shot_id}: mp4={length}, json={trajectory_length}")
        current = length // 2
        release = None
        defender = None
        side = None
        cached = {"frame": None, "img": None}
        window = "H1 BasketEvent assisted annotation"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def frame_image(index: int):
            if cached["frame"] != index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, img = capture.read()
                actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                if not ok or actual != index:
                    raise ValueError(f"Failed exact frame seek: requested {index}, got {actual}")
                cached.update({"frame": index, "img": img})
            return cached["img"]

        def mouse(event_type, x, y, flags, params):
            nonlocal defender
            if event_type != cv2.EVENT_LBUTTONDOWN:
                return
            img = frame_image(current)
            h, w = img.shape[:2]
            possibilities = []
            for name in data:
                if name in ("ball", role):
                    continue
                box = role_at_frame(data, name, current, w, h)
                if box and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    possibilities.append((area, name))
            if possibilities:
                defender = min(possibilities)[1]
                print(f"Defender proposal: {defender} (verify visually)")

        cv2.setMouseCallback(window, mouse)
        while True:
            img = frame_image(current)
            h, w = img.shape[:2]
            proposal = role_at_frame(data, role, current, w, h)
            cv2.imshow(window, draw(img, data, role, defender, current, release, side, shot_id))
            key = cv2.waitKeyEx(0)
            if key in (ord("a"), ord("A")):
                current = max(0, current - 1)
            elif key in (ord("d"), ord("D")):
                current = min(length - 1, current + 1)
            elif key in (ord("j"), ord("J")):
                current = max(0, current - 10)
            elif key in (ord("l"), ord("L")):
                current = min(length - 1, current + 10)
            elif key in (ord("r"), ord("R")):
                if proposal is None:
                    print("Cannot select release: missing shooter box at this frame")
                else:
                    release = current
                    print(f"Release marked at frame {release}; verify visually")
            elif key == ord("1"):
                side = "left"
            elif key == ord("2"):
                side = "right"
            elif key in (ord("x"), ord("X")):
                defender = None
            elif key in (ord("k"), ord("K")):
                return "skipped", {}
            elif key in (ord("q"), ord("Q"), 27):
                return "quit", {}
            elif key in (ord("s"), ord("S")):
                if release is None or side is None:
                    print("Mark release with R and choose shooting hand with 1 or 2 first")
                    continue
                shooter_box = role_at_frame(data, role, release, w, h)
                defender_box = role_at_frame(data, defender, release, w, h) if defender else None
                if shooter_box is None:
                    print("Shooter box missing at release; cannot save")
                    continue
                # Defense is optional, but a selected defender needs a valid release box.
                if defender and defender_box is None:
                    print("Selected defender has no box at release; clear with X or choose another")
                    continue
                record = {
                    "release_frame": release,
                    "shooter_box": json.dumps(shooter_box),
                    "defender_box": json.dumps(defender_box) if defender_box else "",
                    "shooting_side": side,
                    "annotation_status": "manual_release_verified_basketevent_boxes",
                }
                return "saved", record
    finally:
        capture.release()
        cv2.destroyAllWindows()


def run(manifest: Path, output: Path, root: Path, limit: int | None, start_index: int) -> None:
    if output.resolve() == manifest.resolve():
        raise ValueError("Output must differ from input manifest")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    frame = pd.read_csv(manifest, dtype={"shot_id": str, "game_id": str, "game_event_id": str})
    needed = {"shot_id", "clip_path", "source_json", "release_frame", "shooter_box", "defender_box", "shooting_side"}
    if missing := needed - set(frame):
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame.shot_id.isna().any() or frame.shot_id.duplicated().any():
        raise ValueError("Shot IDs must be unique and nonempty")
    report_path = manifest.parent / "event_matching_report.csv"
    report = pd.read_csv(report_path, dtype={"shot_id": str, "json_path": str})
    output.parent.mkdir(parents=True, exist_ok=True)
    # Create a copy early and checkpoint after each shot. Existing outputs remain immutable.
    frame.to_csv(output, index=False)
    stop = len(frame) if limit is None else min(len(frame), start_index + limit)
    for i in range(start_index, stop):
        if pd.notna(frame.loc[i, "release_frame"]):
            continue
        print(f"Annotating {i + 1}/{len(frame)}: {frame.loc[i, 'shot_id']}", flush=True)
        try:
            status, updates = annotate_one(frame.loc[i].to_dict(), manifest, report, root)
            if status == "quit":
                print("Stopped by user; saved progress in", output)
                return
            if status == "saved":
                for key, value in updates.items():
                    frame.at[i, key] = value
                frame.to_csv(output, index=False)
                print("Saved:", frame.loc[i, "shot_id"])
            else:
                print("Skipped:", frame.loc[i, "shot_id"])
        except Exception as exc:
            print("FAILED (retained unannotated):", frame.loc[i, "shot_id"], str(exc))
    print("Finished. Output:", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("data/basketevent"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1 or args.start_index < 0:
        parser.error("--limit must be >=1, --start-index must be >=0")
    run(args.manifest, args.output, args.source_root, args.limit, args.start_index)


if __name__ == "__main__":
    main()
