"""Interactive visual review of UNVERIFIED BasketEvent automatic H1 annotations.

Inputs: auto annotation directory from auto_annotate_basketevent.py.
The MP4 path is resolved relative to the ORIGINAL manifest, not the auto
manifest (which lives in another directory). Original CSVs are never changed.
Reviewed records are checkpointed to a new CSV after every accepted/rejected shot.

Keys: A/D previous/next frame, J/L +/- 10, C/V cycle auto release candidates,
R mark current frame as release, click a player to choose defender, X clear,
1/2 shooting hand left/right, S accept with manual review, K skip, N reject,
Q quit. F is toggle display of all non-shooter player boxes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from auto_annotate_basketevent import (ball_candidates, trajectory_boxes,
                                      unique_shooter_role, xywh_to_xyxy)

VERIFIED = 'manual_release_verified_basketevent_boxes'
REVIEWED = 'manual_review_verified_basketevent_boxes'
REJECTED = 'manual_review_rejected'


def resolve_clip(clip_value: str, original_manifest: Path) -> Path:
    p = Path(str(clip_value))
    return (p if p.is_absolute() else original_manifest.parent / p).resolve()


def parse_candidates(value) -> list[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        try:
            raw = json.loads(str(value))
        except (ValueError, TypeError):
            return []
    if not isinstance(raw, list):
        return []
    result = []
    for item in raw:
        if isinstance(item, bool):
            continue
        try:
            x = int(item)
            if float(item) == x and x >= 0 and x not in result:
                result.append(x)
        except (ValueError, TypeError):
            pass
    return result


def review_indices(manifest: pd.DataFrame, *, per_game: int | None, start_index: int = 0,
                   include_existing: bool = False) -> list[int]:
    """Deterministic spread across games; never select by made/missed outcome."""
    indices, counts = [], {}
    for index in range(start_index, len(manifest)):
        row = manifest.iloc[index]
        status = str(row.get('annotation_status', ''))
        if status in {REVIEWED, REJECTED}:
            continue
        if not include_existing and status == VERIFIED:
            continue
        game = str(row['shot_id']).split(':')[1]
        if per_game is not None and counts.get(game, 0) >= per_game:
            continue
        counts[game] = counts.get(game, 0) + 1
        indices.append(index)
    return indices


def ready_to_save(release, side, shooter_box) -> bool:
    return (isinstance(release, int) and release >= 0
            and side in ('left', 'right') and shooter_box is not None)


def read_exact(cap, index: int):
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    actual = int(round(cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
    if not ok or actual != index:
        raise ValueError(f'Frame seek mismatch expected={index}, actual={actual}')
    return img


def review_one(row: dict, original_manifest: Path, report: pd.DataFrame,
               root: Path, audit_row: dict) -> tuple[str, dict]:
    import cv2

    shot_id = str(row['shot_id'])
    source = str(row['source_json'])
    role = unique_shooter_role(report, shot_id, source)
    data = json.loads((root / source).read_text(encoding='utf-8'))
    clip = resolve_clip(row['clip_path'], original_manifest)
    if not clip.is_file():
        raise FileNotFoundError(clip)
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video {clip}')
    window = 'H1 Auto Annotation Review'
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if n < 2 or w < 2 or h < 2:
            raise ValueError('Video has invalid dimensions')
        if len(data[role].get('trajectory', [])) != n:
            raise ValueError('Shooter trajectory / MP4 frame-count mismatch')
        if any(isinstance(v, dict) and isinstance(v.get('trajectory'), list) and
               len(v['trajectory']) != n for v in data.values()):
            raise ValueError('JSON / MP4 frame-count mismatch')

        proposed = parse_candidates(audit_row.get('candidate_frames'))
        proposed = [p for p in proposed if p < n]
        # Status may have no candidate; open in middle, not at a fabricated release.
        current = proposed[0] if proposed else n // 2
        release = None
        side = None
        defender = None
        show_all = True
        original_defender = audit_row.get('defender_role')
        if isinstance(original_defender, str) and original_defender in data and original_defender != role:
            defender = original_defender  # proposal only; must be visually checked
        candidate_position = 0
        cached = {'index': None, 'image': None}
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        def image_at(index):
            if cached['index'] != index:
                cached['image'] = read_exact(cap, index)
                cached['index'] = index
            return cached['image']

        def box_at(person, index):
            obj = data.get(person)
            if not isinstance(obj, dict):
                return None
            arr = obj.get('trajectory')
            if not isinstance(arr, list) or index >= len(arr):
                return None
            return xywh_to_xyxy(arr[index], w, h)

        def on_mouse(event, x, y, flags, userdata):
            nonlocal defender
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            picks = []
            for p in data:
                if p in (role, 'ball'):
                    continue
                box = box_at(p, current)
                if box is not None and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                    picks.append(((box[2]-box[0])*(box[3]-box[1]), p))
            if picks:
                defender = min(picks)[1]
                print('Selected defender:', defender, '(verify visually)')

        cv2.setMouseCallback(window, on_mouse)
        while True:
            base = image_at(current)
            canvas = base.copy()
            if show_all:
                for p, obj in data.items():
                    if p == 'ball' or not isinstance(obj, dict):
                        continue
                    b = box_at(p, current)
                    if b is None:
                        continue
                    color = ((70, 225, 70) if p == role else
                             ((30, 170, 255) if p == defender else (150, 150, 150)))
                    cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color,
                                  3 if p in (role, defender) else 1)
                    cv2.putText(canvas, str(obj.get('player_name') or p),
                                (b[0], max(16, b[1]-6)), cv2.FONT_HERSHEY_SIMPLEX,
                                .46, color, 1, cv2.LINE_AA)
            else:
                for p, color in [(role, (70, 225, 70)), (defender, (30, 170, 255))]:
                    if not p:
                        continue
                    b = box_at(p, current)
                    if b:
                        cv2.rectangle(canvas, (b[0], b[1]), (b[2], b[3]), color, 3)

            ball = box_at('ball', current)
            if ball:
                cv2.rectangle(canvas, (ball[0], ball[1]), (ball[2], ball[3]),
                              (20, 255, 255), 2)
            if current in proposed:
                cv2.putText(canvas, 'AUTO CANDIDATE', (max(5, w-230), min(h-10, 160)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
            lines = [
                f'{shot_id} | frame {current}/{n-1} | proposed: {proposed or "NONE"}',
                f'Manual release: {release if release is not None else "UNSET"} | hand: {side or "UNSET"} | defender: {defender or "NONE"}',
                'A/D +/-1 | J/L +/-10 | C/V cycle candidates | R confirm current as release',
                'Click defender | X clear | 1 left / 2 right hand | F toggle boxes',
                'S SAVE reviewed | K skip | N reject (explicit) | Q quit',
            ]
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (w, min(h, 122)), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
            for k, line in enumerate(lines):
                cv2.putText(canvas, line, (10, 19+k*23), cv2.FONT_HERSHEY_SIMPLEX,
                            .42, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(0)
            if key in (ord('a'), ord('A')):
                current = max(0, current-1)
            elif key in (ord('d'), ord('D')):
                current = min(n-1, current+1)
            elif key in (ord('j'), ord('J')):
                current = max(0, current-10)
            elif key in (ord('l'), ord('L')):
                current = min(n-1, current+10)
            elif key in (ord('c'), ord('C'), ord('v'), ord('V')):
                if proposed:
                    delta = 1 if key in (ord('c'), ord('C')) else -1
                    candidate_position = (candidate_position + delta) % len(proposed)
                    current = proposed[candidate_position]
            elif key in (ord('r'), ord('R')):
                if box_at(role, current) is None:
                    print('Shooter box missing: cannot mark this release frame')
                else:
                    release = current
                    print('Manually marked release:', release)
            elif key == ord('1'):
                side = 'left'
            elif key == ord('2'):
                side = 'right'
            elif key in (ord('x'), ord('X')):
                defender = None
            elif key in (ord('f'), ord('F')):
                show_all = not show_all
            elif key in (ord('s'), ord('S')):
                shooter_box = box_at(role, release) if release is not None else None
                if not ready_to_save(release, side, shooter_box):
                    print('Use R to manually mark release and 1/2 to choose hand before saving')
                    continue
                defender_box = box_at(defender, release) if defender else None
                if defender and defender_box is None:
                    print('Defender box invalid at release; choose another or X to clear')
                    continue
                return 'saved', {
                    'release_frame': release,
                    'shooter_box': json.dumps(shooter_box),
                    'defender_box': json.dumps(defender_box) if defender_box else '',
                    'shooting_side': side,
                    'annotation_status': REVIEWED,
                    'review_frame_delta': release - proposed[0] if proposed else pd.NA,
                    'review_candidate_frames': json.dumps(proposed),
                    'review_shooter_role': role,
                    'review_defender_role': defender or '',
                }
            elif key in (ord('k'), ord('K')):
                return 'skipped', {}
            elif key in (ord('n'), ord('N')):
                return 'rejected', {'annotation_status': REJECTED,
                                    'release_frame': pd.NA, 'shooter_box': '',
                                    'defender_box': '', 'shooting_side': '',
                                    'review_candidate_frames': json.dumps(proposed),
                                    'review_shooter_role': role}
            elif key in (ord('q'), ord('Q'), 27):
                return 'quit', {}
    finally:
        cap.release()
        cv2.destroyAllWindows()


def run(auto_dir: Path, output: Path, original_manifest: Path | None,
        root: Path, per_game: int | None, start_index: int, include_existing: bool,
        input_reviewed: Path | None = None) -> dict:
    auto_dir, output, root = Path(auto_dir), Path(output), Path(root)
    if output.exists():
        raise FileExistsError(f'Choose a new output CSV, refusing overwrite: {output}')
    auto_manifest = auto_dir / 'auto_annotated_manifest.csv'
    audit_path = auto_dir / 'annotation_audit.csv'
    summary = json.loads((auto_dir / 'annotation_summary.json').read_text(encoding='utf-8'))
    if original_manifest is None:
        original_manifest = Path(summary['source_manifest'])
    original_manifest = Path(original_manifest)
    if not original_manifest.is_file():
        raise FileNotFoundError(f'Original manifest missing: {original_manifest}')
    if output.resolve() in {auto_manifest.resolve(), original_manifest.resolve(), audit_path.resolve()}:
        raise ValueError('Never write over input or audit')
    manifest_to_read = Path(input_reviewed) if input_reviewed is not None else auto_manifest
    if not manifest_to_read.is_file():
        raise FileNotFoundError(f'Review manifest missing: {manifest_to_read}')
    if output.resolve() == manifest_to_read.resolve():
        raise ValueError('Output must differ from input')
    frame = pd.read_csv(manifest_to_read, dtype={'shot_id': str, 'game_id': str, 'game_event_id': str})
    if frame.shot_id.isna().any() or frame.shot_id.duplicated().any():
        raise ValueError('Need unique non-empty shot IDs')
    audit = pd.read_csv(audit_path, dtype={'shot_id': str})
    if audit.shot_id.duplicated().any() or set(frame.shot_id) != set(audit.shot_id):
        raise ValueError('Audit/manifest shot IDs must align')
    report = pd.read_csv(original_manifest.parent / 'event_matching_report.csv',
                         dtype={'shot_id': str, 'json_path': str})
    audit_by_id = audit.set_index('shot_id')
    # Cast object fields upfront so pandas can persist JSON strings without warnings.
    for col in ['shooter_box', 'defender_box', 'shooting_side', 'annotation_status',
                'review_candidate_frames', 'review_shooter_role', 'review_defender_role']:
        if col not in frame:
            frame[col] = pd.Series([None] * len(frame), dtype='object')
        else:
            frame[col] = frame[col].astype('object')
    for col in ['review_frame_delta']:
        if col not in frame:
            frame[col] = pd.Series([float('nan')] * len(frame), dtype='float64')
    indices = review_indices(frame, per_game=per_game, start_index=start_index,
                             include_existing=include_existing)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)  # checkpoint even when no rows reviewed
    print('Scheduled:', len(indices), 'shots across', len(set(frame.iloc[i].shot_id.split(':')[1] for i in indices)), 'games')
    counts = {'saved': 0, 'rejected': 0, 'skipped': 0, 'errors': 0}
    for k, i in enumerate(indices, 1):
        row = frame.iloc[i].to_dict()
        shot_id = row['shot_id']
        print(f'[{k}/{len(indices)}] Reviewing {shot_id}', flush=True)
        try:
            status, updates = review_one(row, original_manifest, report, root,
                                         audit_by_id.loc[shot_id].to_dict())
            if status == 'quit':
                print('Stopped. Reviewed rows checkpointed at:', output)
                break
            if status in ('saved', 'rejected'):
                for col, value in updates.items():
                    frame.at[i, col] = value
                frame.to_csv(output, index=False)
                print('Checkpoint:', status, shot_id)
            counts[status] += 1
        except Exception as exc:
            counts['errors'] += 1
            print('Review error (row unchanged):', shot_id, str(exc))
    print('Review progress:', json.dumps(counts, indent=2))
    print('Output:', output)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--auto-dir', type=Path, default=Path('artifacts/h1_auto_annotation_v1'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--input-reviewed', type=Path, default=None,
                    help='Prior review CSV to resume; always choose a NEW output path')
    ap.add_argument('--original-manifest', type=Path, default=None)
    ap.add_argument('--source-root', type=Path, default=Path('data/basketevent'))
    ap.add_argument('--sample-per-game', type=int, default=2,
                    help='Number of unverified shots per game (default: 2). Use 0 for all.')
    ap.add_argument('--start-index', type=int, default=0)
    ap.add_argument('--include-existing', action='store_true',
                    help='Review even manually verified shots; do not enable for normal run')
    args = ap.parse_args()
    if args.sample_per_game < 0 or args.start_index < 0:
        ap.error('Sample per game and start index must be nonnegative')
    run(args.auto_dir, args.output, args.original_manifest, args.source_root,
        args.sample_per_game or None, args.start_index, args.include_existing,
        args.input_reviewed)


if __name__ == '__main__':
    main()
