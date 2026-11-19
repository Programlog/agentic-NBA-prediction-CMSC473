"""BasketEvent release-frame proposals v2: geometry transitions + sustained motion.

Outcome and event clock are never used in candidate scoring. All proposed frames
require visual verification before H1 model training. Keeps v1 outputs unchanged.
Run from repo root: uv run --no-sync python scripts/auto_annotate_basketevent_v2.py --manifest ... --output ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # Direct script invocation: python scripts/auto_annotate_basketevent_v2.py
    from auto_annotate_basketevent import (
        SHOT_ACTIONS, ball_candidates, defender_proposal, trajectory_boxes,
        unique_shooter_role,
    )
except ModuleNotFoundError:
    # Repo-root test invocation: pytest -q tests/test_auto_annotate_basketevent_v2.py
    from scripts.auto_annotate_basketevent import (
        SHOT_ACTIONS, ball_candidates, defender_proposal, trajectory_boxes,
        unique_shooter_role,
    )


def release_candidates_v2(shooter: list, ball: list, n: int, max_candidates: int = 5) -> dict:
    """Rank candidate release onsets, not the maximum later ball displacement.

    Geometry-only ranking uses normalised ball-to-upper-shooter distance, 3-frame
    rise in that distance, and short-term upward ball motion. Mask missing boxes,
    forbid discontinuous tracks, and cluster nearby candidates. This is an
    experiment, *not* validated release detection or a calibrated confidence.
    """
    if n != len(shooter) or n != len(ball) or n < 9:
        return {"candidates": [], "scores": [], "reason": "insufficient_or_misaligned_tracks"}
    dist = np.full(n, np.nan)
    ball_y = np.full(n, np.nan)
    for i, (sb, bb) in enumerate(zip(shooter, ball)):
        if sb is None or bb is None:
            continue
        s = np.asarray(sb, dtype=float)
        b = np.asarray(bb, dtype=float)
        if not (np.isfinite(s).all() and np.isfinite(b).all()):
            continue
        width, height = s[2] - s[0], s[3] - s[1]
        if width <= 0 or height <= 0:
            continue
        scale = max(width, height)
        anchor = np.array([(s[0] + s[2]) / 2, s[1] + 0.35 * height])
        bc = (b[:2] + b[2:]) / 2
        dist[i] = np.linalg.norm(bc - anchor) / scale
        ball_y[i] = bc[1] / scale
    peaks = []
    for i in range(3, n - 5):
        d = dist[i - 3:i + 5]
        y = ball_y[i - 3:i + 5]
        if not (np.isfinite(d).all() and np.isfinite(y).all()):
            continue
        # Require a nearby ball before the event and sustained separation after.
        near = np.median(d[:3])
        rise = np.median(d[4:]) - near
        upward = y[1] - y[5]  # positive = moving up in image coordinates
        local_start = d[4] - d[2]
        persistence = np.min(d[4:]) - near
        if near > 1.05 or rise < 0.12 or upward < 0.07 or persistence < 0.04:
            continue
        # Favor the first supported separation onset rather than later peak.
        score = 1.2 * local_start + 0.6 * rise + 0.25 * upward - 0.2 * near
        peaks.append((float(score), i))
    peaks.sort(key=lambda x: (-x[0], x[1]))
    chosen = []
    for score, frame in peaks:
        if all(abs(frame - prev) > 5 for _, prev in chosen):
            chosen.append((score, frame))
        if len(chosen) >= max_candidates:
            break
    # If stringent v2 gate does not produce a candidate, use v1's candidates
    # explicitly labeled as a fallback, so downstream review still has context.
    if not chosen:
        fallback = ball_candidates(shooter, ball, n)
        return {**fallback, "reason": "v1_fallback_unverified" if fallback['candidates'] else "no_geometry_candidate"}
    return {"candidates": [frame for _, frame in chosen],
            "scores": [round(score, 5) for score, _ in chosen],
            "reason": "v2_geometry_onset_unverified"}


def propose_v2(row: dict, manifest: Path, report: pd.DataFrame, root: Path) -> dict:
    import cv2
    shot_id = str(row['shot_id'])
    path = str(row['source_json'])
    role = unique_shooter_role(report, shot_id, path)
    data = json.loads((root / path).read_text(encoding='utf-8'))
    person = data.get(role)
    if not isinstance(person, dict) or not isinstance(person.get('event'), dict):
        raise ValueError('Missing shooter event')
    if str(person['event'].get('actionType', '')).lower().strip() not in SHOT_ACTIONS:
        raise ValueError('Not a shooting event')
    clip = Path(str(row['clip_path']))
    if not clip.is_absolute():
        clip = manifest.parent / clip
    cap = cv2.VideoCapture(str(clip.resolve()))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open clip {clip}')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if n < 10 or width < 2 or height < 2:
        raise ValueError('Invalid video metadata')
    if not isinstance(data.get('ball'), dict):
        raise ValueError('Missing ball record')
    if len(person.get('trajectory', [])) != n or len(data['ball'].get('trajectory', [])) != n:
        raise ValueError('Trajectory/video frame count mismatch')
    shooter = trajectory_boxes(person, width, height, n)
    ball = trajectory_boxes(data['ball'], width, height, n)
    candidates = release_candidates_v2(shooter, ball, n)
    if not candidates['candidates']:
        return {'status': 'review_required_no_release_candidate', 'source_role': role,
                'candidate_frames': [], 'candidate_scores': [], 'reason': candidates['reason']}
    f = candidates['candidates'][0]
    box = shooter[f]
    if box is None:
        raise ValueError('Missing shooter box at release proposal')
    defender, db, dreason = defender_proposal(data, role, f, width, height, n)
    return {'status': 'auto_proposed_review_required', 'source_role': role,
            'candidate_frames': candidates['candidates'], 'candidate_scores': candidates['scores'],
            'release_frame': f, 'shooter_box': box, 'defender_role': defender,
            'defender_box': db, 'defender_reason': dreason, 'shooting_side': None,
            'reason': candidates['reason']}


def compare_with_review(audit: pd.DataFrame, reviewed_path: Path) -> dict:
    """Diagnostic against DEVELOPMENT reviews only; not independent validation."""
    ref = pd.read_csv(reviewed_path, dtype={'shot_id': str})
    ref = ref[ref.get('annotation_status', pd.Series(dtype=str)).eq(
        'manual_review_verified_basketevent_boxes')]
    merged = audit[['shot_id', 'release_frame']].merge(
        ref[['shot_id', 'release_frame', 'review_frame_delta']], on='shot_id',
        how='inner', suffixes=('_v2', '_manual'), validate='one_to_one')
    if merged.empty:
        return {'compared': 0}
    merged['v2_abs_error'] = (merged.release_frame_v2 - merged.release_frame_manual).abs()
    # Reviewer delta = manual - v1, so absolute value is V1 error.
    merged['v1_abs_error'] = merged.review_frame_delta.abs()
    good = merged.dropna(subset=['v2_abs_error', 'v1_abs_error'])
    def report(series):
        return {'mae': float(series.mean()), 'median_ae': float(series.median()),
                'within_3': int(series.le(3).sum()), 'exact': int(series.eq(0).sum())}
    return {'compared': int(len(good)), 'v1': report(good.v1_abs_error),
            'v2': report(good.v2_abs_error),
            'note': 'Development-set diagnostic ONLY; cannot claim generalization from these reviewed shots.',
            'per_shot': good[['shot_id', 'v1_abs_error', 'v2_abs_error']].to_dict('records')}


def run(manifest: Path, output: Path, root: Path, reviewed: Path | None = None) -> dict:
    manifest, output, root = Path(manifest), Path(output), Path(root)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing output: {output}')
    source = pd.read_csv(manifest, dtype={'shot_id': str, 'game_id': str, 'game_event_id': str})
    required = {'shot_id', 'clip_path', 'source_json', 'release_frame', 'shooter_box',
                'defender_box', 'shooting_side', 'annotation_status'}
    if missing := required - set(source):
        raise ValueError(f'Manifest missing: {sorted(missing)}')
    if source.shot_id.isna().any() or source.shot_id.duplicated().any():
        raise ValueError('Shot IDs must be unique and nonmissing')
    report = pd.read_csv(manifest.parent / 'event_matching_report.csv',
                         dtype={'shot_id': str, 'json_path': str})
    # Only write output after validating inputs. Leave manual rows unmodified.
    for col in ['shooter_box', 'defender_box', 'shooting_side', 'annotation_status']:
        source[col] = source[col].astype('object')
    rows = []
    for i, record in source.iterrows():
        audit = {'shot_id': record.shot_id, 'status': 'error', 'error': None}
        try:
            if pd.notna(record.release_frame):
                audit['status'] = 'preserved_existing_annotation'
            else:
                proposal = propose_v2(record.to_dict(), manifest, report, root)
                audit.update(proposal)
                if proposal['status'] == 'auto_proposed_review_required':
                    source.at[i, 'release_frame'] = proposal['release_frame']
                    for key in ['shooter_box', 'defender_box']:
                        if proposal.get(key) is not None:
                            source.at[i, key] = json.dumps(proposal[key])
                source.at[i, 'annotation_status'] = proposal['status']
        except Exception as exc:
            audit['status'] = 'error'
            audit['error'] = str(exc)
            source.at[i, 'annotation_status'] = 'auto_annotation_error'
        rows.append(audit)
        print(f'[{i+1}/{len(source)}] {record.shot_id}: {audit["status"]}', flush=True)
    audit_df = pd.DataFrame(rows)
    output.mkdir(parents=True)
    source.to_csv(output / 'auto_annotated_manifest.csv', index=False)
    audit_df.to_csv(output / 'annotation_audit.csv', index=False)
    summary = {
        'source_manifest': str(manifest), 'version': 'v2_geometry_onset',
        'shots': len(source),
        'existing_annotations_preserved': int(audit_df.status.eq('preserved_existing_annotation').sum()),
        'auto_proposals_unverified': int(audit_df.status.eq('auto_proposed_review_required').sum()),
        'no_release_candidate': int(audit_df.status.eq('review_required_no_release_candidate').sum()),
        'errors': int(audit_df.status.eq('error').sum()),
        'critical_warning': 'NO auto release, shooter, or defender proposal is verified. Never train on unreviewed proposals.',
        'pre_release_only_model_features': 'Use frames up to the verified release; exclude all subsequent motion and shot outcome.'
    }
    (output / 'annotation_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if reviewed is not None:
        comparison = compare_with_review(audit_df, Path(reviewed))
        (output / 'development_comparison.json').write_text(json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
        print('Development comparison:', json.dumps(comparison, indent=2))
    print('Summary:', json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source-root', type=Path, default=Path('data/basketevent'))
    p.add_argument('--development-reviews', type=Path, default=None,
                   help='Optional 8-shot review CSV; development-only V1/V2 diagnostic')
    args = p.parse_args()
    run(args.manifest, args.output, args.source_root, args.development_reviews)


if __name__ == '__main__':
    main()
