"""Conservative V1/V2 release-frame reconciliation for BasketEvent H1.

Reads existing annotation manifests/audits; does NOT infer new release frames,
train models, use shot outcomes, or auto-verify any prediction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MANUAL_STATUSES = {
    'manual_release_verified_basketevent_boxes',
    'manual_review_verified_basketevent_boxes',
}
REJECTED_STATUSES = {'manual_review_rejected', 'manual_rejected_unusable'}
FIELDS = ('release_frame', 'shooter_box', 'defender_box', 'shooting_side')


def frame_number(value):
    if pd.isna(value):
        return None
    try:
        f = float(value)
    except (ValueError, TypeError):
        return None
    if not (f >= 0 and f.is_integer() and f < 1000000):
        return None
    return int(f)


def decision(v1, v2, threshold=5):
    """Return an UNVERIFIED decision; a small delta is not proof of accuracy."""
    if threshold < 0:
        raise ValueError('threshold must be nonnegative')
    a, b = frame_number(v1), frame_number(v2)
    if a is None and b is None:
        return 'review_required_no_candidate', None, None
    if a is None or b is None:
        return 'review_required_single_estimator', None, None
    gap = abs(a - b)
    if gap <= threshold:
        # Midpoint offers no claim that either method is correct; use V1's
        # actual frame (and its associated boxes) as an unverified proposal.
        return 'agreement_candidate_unverified', gap, a
    return 'review_required_disagreement', gap, None


def checked_csv(path, expected_ids=None):
    frame = pd.read_csv(path, dtype={'shot_id': str, 'game_id': str, 'game_event_id': str})
    if 'shot_id' not in frame or frame.shot_id.isna().any() or frame.shot_id.duplicated().any():
        raise ValueError(f'Invalid or duplicate shot IDs in {path}')
    if expected_ids is not None and set(frame.shot_id) != expected_ids:
        raise ValueError(f'Shot IDs do not match across inputs: {path}')
    return frame.set_index('shot_id', drop=False)


def reconcile(v1_manifest, v1_audit, v2_manifest, v2_audit, *,
              reviews=None, threshold=5):
    """Reconcile exact shot IDs, preserve verified overrides, and never auto-promote."""
    source = checked_csv(v1_manifest)
    ids = set(source.index)
    a1 = checked_csv(v1_audit, ids)
    m2 = checked_csv(v2_manifest, ids)
    a2 = checked_csv(v2_audit, ids)
    reviewed = checked_csv(reviews, ids) if reviews is not None else None
    if 'release_frame' not in a1 or 'release_frame' not in a2:
        raise ValueError('Both annotation audits must have release_frame')
    if 'annotation_status' not in source or 'annotation_status' not in m2:
        raise ValueError('Both manifests require annotation_status')
    for col in FIELDS:
        if col not in source or col not in m2:
            raise ValueError(f'Missing manifest field {col}')

    result = source.copy()
    for col in (*FIELDS, 'annotation_status', 'v3_status', 'v3_review_reason'):
        result[col] = result[col].astype(object) if col in result else pd.Series(index=result.index, dtype=object)

    records = []
    for shot_id in source.index:
        base = source.loc[shot_id]
        other = m2.loc[shot_id]
        a = a1.loc[shot_id]
        b = a2.loc[shot_id]
        v1 = frame_number(a['release_frame'])
        v2 = frame_number(b['release_frame'])
        status, gap, chosen = decision(v1, v2, threshold)
        origin = 'none'
        override = None
        if reviewed is not None:
            r = reviewed.loc[shot_id]
            rs = str(r.get('annotation_status', ''))
            if rs in MANUAL_STATUSES or rs in REJECTED_STATUSES:
                override = r
                origin = 'manual_review'
        if override is None:
            for candidate, name in ((base, 'v1_manifest'), (other, 'v2_manifest')):
                cs = str(candidate.get('annotation_status', ''))
                if cs in MANUAL_STATUSES or cs in REJECTED_STATUSES:
                    override, origin = candidate, name
                    break
        if override is not None:
            saved_status = str(override['annotation_status'])
            if saved_status in MANUAL_STATUSES:
                if frame_number(override['release_frame']) is None or pd.isna(override['shooter_box']) or str(override['shooting_side']) not in {'left','right'}:
                    raise ValueError(f'Malformed verified annotation: {shot_id}')
                for col in FIELDS:
                    result.at[shot_id, col] = override[col]
                result.at[shot_id, 'annotation_status'] = saved_status
                status = 'preserved_manual_verified'
            else:
                for col in FIELDS:
                    result.at[shot_id, col] = pd.NA
                result.at[shot_id, 'annotation_status'] = saved_status
                status = 'preserved_manual_rejected'
        else:
            # Don't let disagreement / missing candidates masquerade as valid
            # annotation rows. All unverified rows have *blank* release fields,
            # including agreement candidates, for training safety.
            for col in FIELDS:
                result.at[shot_id, col] = pd.NA
            result.at[shot_id, 'annotation_status'] = status
        result.at[shot_id, 'v3_status'] = status
        result.at[shot_id, 'v3_review_reason'] = (
            'V1 and V2 agree; manual confirmation still required' if status == 'agreement_candidate_unverified'
            else status
        )
        records.append({
            'shot_id': shot_id,
            'v1_frame': v1,
            'v2_frame': v2,
            'frame_gap': gap,
            'threshold_frames': threshold,
            'status': status,
            'proposed_frame_unverified': chosen if status == 'agreement_candidate_unverified' else None,
            'v1_candidate_frames': a.get('candidate_frames'),
            'v2_candidate_frames': b.get('candidate_frames'),
            'manual_override_source': origin,
            'requires_manual_review': status not in {'preserved_manual_verified', 'preserved_manual_rejected'},
            'note': 'Agreement does not establish correctness; no outcome or post-release data is a model feature',
        })
    return result.reset_index(drop=True), pd.DataFrame(records)


def run(v1_dir: Path, v2_dir: Path, output: Path, reviews: Path | None = None,
        threshold: int = 5):
    v1_dir, v2_dir, output = Path(v1_dir), Path(v2_dir), Path(output)
    if output.exists():
        raise FileExistsError(f'Output exists: {output}; choose a new directory')
    manifest, audit = reconcile(
        v1_dir / 'auto_annotated_manifest.csv', v1_dir / 'annotation_audit.csv',
        v2_dir / 'auto_annotated_manifest.csv', v2_dir / 'annotation_audit.csv',
        reviews=reviews, threshold=threshold,
    )
    # Preserve provenance in immutable run directory; avoid any partial outputs.
    output.mkdir(parents=True)
    manifest.to_csv(output / 'v3_review_manifest.csv', index=False)
    audit.to_csv(output / 'v3_reconciliation_audit.csv', index=False)
    counts = {str(k): int(v) for k, v in audit.status.value_counts().items()}
    summary = {
        'method': 'V3 V1/V2 agreement-based review triage',
        'threshold_frames': threshold,
        'shots': len(manifest),
        'status_counts': counts,
        'v1_dir': str(v1_dir), 'v2_dir': str(v2_dir),
        'manual_reviews': str(reviews) if reviews else None,
        'warning': 'ALL agreement candidates remain UNVERIFIED. Only manually verified rows have usable release/box values in V3 manifest.',
        'evaluation': 'Agreement counts measure consensus, not release-frame accuracy; evaluate on unseen manual references.',
    }
    (output / 'v3_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v1-dir', type=Path, required=True)
    p.add_argument('--v2-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reviews', type=Path, default=None,
                   help='Optional prior review checkpoint CSV (40 shot IDs); development only')
    p.add_argument('--threshold', type=int, default=5)
    args = p.parse_args()
    run(args.v1_dir, args.v2_dir, args.output, args.reviews, args.threshold)


if __name__ == '__main__':
    main()
