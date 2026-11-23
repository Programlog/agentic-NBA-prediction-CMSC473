import numpy as np
import pandas as pd
from scripts.auto_annotate_basketevent_v2 import release_candidates_v2, compare_with_review


def tracks(n=45, release=20):
    shooter = [[50, 50, 90, 150] for _ in range(n)]
    ball = []
    for i in range(n):
        # Ball near upper body until release; accelerates upward afterward.
        t = max(0, i - release)
        ball.append([66, 82 - 8*t, 74, 90 - 8*t])
    return shooter, ball


def test_catches_simple_separation():
    s, b = tracks()
    r = release_candidates_v2(s, b, len(s))
    assert r['candidates']
    assert min(abs(i-20) for i in r['candidates']) <= 5


def test_missing_or_bad_tracks_are_not_guessed():
    s, b = tracks()
    assert release_candidates_v2(s[:-1], b, len(b))['candidates'] == []
    assert release_candidates_v2([None]*len(b), b, len(b))['candidates'] == []


def test_static_ball_does_not_create_release():
    s = [[50, 50, 90, 150] for _ in range(40)]
    b = [[66, 82, 74, 90] for _ in range(40)]
    assert release_candidates_v2(s, b, len(s))['candidates'] == []


def test_review_comparison_calculates_both_errors(tmp_path):
    audit = pd.DataFrame({'shot_id':['s1','s2'], 'release_frame':[10, 20]})
    ref = pd.DataFrame({'shot_id':['s1','s2'], 'release_frame':[12, 22],
                        'review_frame_delta':[5, -4],
                        'annotation_status':['manual_review_verified_basketevent_boxes']*2})
    path = tmp_path/'ref.csv'; ref.to_csv(path,index=False)
    result = compare_with_review(audit, path)
    assert result['compared']==2
    assert result['v2']['mae']==2
    assert result['v1']['mae']==4.5


def test_does_not_claim_improvement_when_worse(tmp_path):
    audit = pd.DataFrame({'shot_id':['s'], 'release_frame':[0]})
    ref = pd.DataFrame({'shot_id':['s'], 'release_frame':[30], 'review_frame_delta':[1],
                        'annotation_status':['manual_review_verified_basketevent_boxes']})
    path = tmp_path/'r.csv'; ref.to_csv(path,index=False)
    result = compare_with_review(audit,path)
    assert result['v2']['mae'] > result['v1']['mae']
