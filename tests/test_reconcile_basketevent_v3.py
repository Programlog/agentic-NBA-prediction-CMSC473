import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.reconcile_basketevent_v3 import decision, reconcile, run


def fixture_files(tmp_path):
    ids = ['nba:0022500003:317', 'nba:0022500004:270',
           'nba:0022500003:228', 'nba:0022500025:223', 'nba:0022500004:58']
    v1 = tmp_path / 'v1'
    v2 = tmp_path / 'v2'
    v1.mkdir()
    v2.mkdir()
    a1 = [120, 80, 4, None, None]
    a2 = [259, 81, 33, None, None]
    for folder, frames in [(v1,a1),(v2,a2)]:
        pd.DataFrame([{
            'shot_id': s, 'release_frame': f,
            'shooter_box': '[0, 0, 20, 40]' if f is not None else None,
            'defender_box': None, 'shooting_side': None,
            'clip_path':'../data/example.mp4', 'game_id':s.split(':')[1],
            'annotation_status': 'auto_proposed_review_required' if f is not None else 'review_required_no_release_candidate',
        } for s, f in zip(ids, frames)]).to_csv(folder / 'auto_annotated_manifest.csv', index=False)
        pd.DataFrame([{'shot_id':s, 'release_frame':f,
                       'candidate_frames':json.dumps([f]) if f is not None else '[]'}
                      for s,f in zip(ids,frames)]).to_csv(folder / 'annotation_audit.csv', index=False)
    # Realistic preserved hand-verified row from the user's first review.
    for folder in (v1,v2):
        f = pd.read_csv(folder / 'auto_annotated_manifest.csv')
        for col in ('defender_box', 'shooting_side'):
            f[col] = f[col].astype(object)
        f.loc[4,'release_frame']=76
        f.loc[4,'shooter_box']='[1004, 206, 1073, 446]'
        f.loc[4,'defender_box']='[768, 275, 934, 454]'
        f.loc[4,'shooting_side']='right'
        f.loc[4,'annotation_status']='manual_release_verified_basketevent_boxes'
        f.to_csv(folder / 'auto_annotated_manifest.csv', index=False)
    reviews = tmp_path / 'reviews.csv'
    rows = pd.read_csv(v1 / 'auto_annotated_manifest.csv')
    rows.loc[0,'annotation_status']='manual_review_verified_basketevent_boxes'
    rows.loc[0,'release_frame']=121
    rows.loc[0,'shooter_box']='[428, 302, 504, 497]'
    rows.loc[0,'shooting_side']='left'
    rows.to_csv(reviews,index=False)
    return v1, v2, reviews


def test_decisions():
    assert decision(120, 259) == ('review_required_disagreement', 139, None)
    assert decision(80, 81) == ('agreement_candidate_unverified', 1, 80)
    assert decision(80, 85) == ('agreement_candidate_unverified', 5, 80)
    assert decision(80, 86) == ('review_required_disagreement', 6, None)
    assert decision(None, 90) == ('review_required_single_estimator', None, None)
    assert decision(None, None) == ('review_required_no_candidate', None, None)


def test_preserve_verified_and_block_unverified(tmp_path):
    v1,v2,reviews = fixture_files(tmp_path)
    out,audit = reconcile(v1/'auto_annotated_manifest.csv',v1/'annotation_audit.csv',
                          v2/'auto_annotated_manifest.csv',v2/'annotation_audit.csv',reviews=reviews)
    records=out.set_index('shot_id')
    a=audit.set_index('shot_id')
    assert records.loc['nba:0022500003:317','release_frame']==121
    assert records.loc['nba:0022500003:317','shooting_side']=='left'
    assert a.loc['nba:0022500003:317','frame_gap']==139
    assert a.loc['nba:0022500003:317','status']=='preserved_manual_verified'
    assert a.loc['nba:0022500004:270','status']=='agreement_candidate_unverified'
    assert pd.isna(records.loc['nba:0022500004:270','release_frame'])
    assert pd.isna(records.loc['nba:0022500004:270','shooter_box'])
    assert a.loc['nba:0022500003:228','status']=='review_required_disagreement'
    assert a.loc['nba:0022500025:223','status']=='review_required_no_candidate'
    assert records.loc['nba:0022500004:58','release_frame']==76
    assert records.loc['nba:0022500004:58','shooting_side']=='right'


def test_no_review_still_flags_catastrophic_disagreement(tmp_path):
    v1,v2,_=fixture_files(tmp_path)
    _,audit = reconcile(v1/'auto_annotated_manifest.csv',v1/'annotation_audit.csv',
                        v2/'auto_annotated_manifest.csv',v2/'annotation_audit.csv')
    r=audit.set_index('shot_id').loc['nba:0022500003:317']
    assert r.status=='review_required_disagreement' and r.frame_gap==139


def test_rejects_duplicate_or_mismatched_shots(tmp_path):
    v1,v2,_=fixture_files(tmp_path)
    f=v2/'annotation_audit.csv'
    df=pd.read_csv(f)
    df.loc[0,'shot_id']='other'
    df.to_csv(f,index=False)
    with pytest.raises(ValueError,match='Shot IDs'):
        reconcile(v1/'auto_annotated_manifest.csv',v1/'annotation_audit.csv',
                  v2/'auto_annotated_manifest.csv',f)


def test_output_immutable(tmp_path):
    v1,v2,reviews=fixture_files(tmp_path)
    out=tmp_path/'results'
    s=run(v1,v2,out,reviews=reviews)
    assert s['shots']==5
    assert (out/'v3_reconciliation_audit.csv').exists()
    with pytest.raises(FileExistsError):
        run(v1,v2,out,reviews=reviews)


def test_bad_threshold():
    with pytest.raises(ValueError):
        decision(1,2,-1)
