"""E1: release-frame rules scored against the hand-marked review clips (item 20 left out)."""
import gzip, json
import numpy as np, pandas as pd
from nba_shots import basketevent as be

R = "data/release_review/"
review = pd.read_csv(R + "review.csv", dtype={"item": str}).merge(pd.read_csv(R + "answers.csv", dtype={"item": str}), on="item")
review = review[review.true_release_frame.notna() & review.item.ne("20")]

def release(shooter, ball, top):
    """The current rule with the margin above the shooter's box as a parameter (current: 0.25)."""
    n = min(len(shooter), len(ball)); x, y, w, h = shooter[:n].T; bx, by = be._centre(ball[:n]).T
    with np.errstate(invalid="ignore"):
        held = (bx > x - be.HAND_REACH * w) & (bx < x + (1 + be.HAND_REACH) * w) & (by > y - top * h) & (by < y + h)
    for k in np.flatnonzero(held)[::-1]:
        ahead = by[k + 1:k + 1 + be.RISE_FRAMES]; ahead = ahead[np.isfinite(ahead)]
        if len(ahead) >= 3 and ahead[-1] < by[k] - be.RISE_BH * h[k]:
            return int(k)
    return None

rows = []
for r in review.itertuples():
    game = r.clip.split("/")[1]
    with gzip.open(f"data/basketevent_labels_v2/games/{game}_labels.json.gz", "rt") as s:
        clip = json.load(s)[r.clip]
    key = next(k for k, v in clip.items() if isinstance(v, dict) and isinstance(v.get("event"), dict) and v["event"].get("actionType") in be.SHOT_ACTIONS)
    shooter, ball = be.steady(be.boxes(clip[key])), be.boxes(clip.get("ball") or {})
    assert release(shooter, ball, be.HAND_REACH) == be.release_frame(shooter, ball)
    rows.append({"item": r.item, "human": r.true_release_frame, **{t: release(shooter, ball, t) for t in (0.25, 0.15, 0.1, 0.05, 0.0)}})
d = pd.DataFrame(rows)
print(f"{len(d)} clips")
for t in (0.25, 0.15, 0.1, 0.05, 0.0):
    e = (d[t] - d.human).dropna()
    print(f"top margin {t:4}: found {len(e)}/{len(d)}  within1 {(e.abs()<=1).sum():2d} ({(e.abs()<=1).mean():.0%})  within3 {(e.abs()<=3).mean():.0%}  median|err| {e.abs().median()}  mean err {e.mean():+.2f}")
