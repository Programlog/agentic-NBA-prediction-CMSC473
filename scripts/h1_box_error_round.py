"""H1 box-error round, as written in docs/h1_box_error_round.md. Train+validation only; never the test split."""
import gzip, json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from nba_shots import basketevent as be, models
from nba_shots.common import BOX_FEATURES, load_config
from nba_shots.evaluation import paired_comparison

S = Path(sys.argv[1]); OUT = S / "h1_box_error_round"; OUT.mkdir(exist_ok=True)
CONFIG = load_config("configs/basketevent_h1_final.json")
IOU, MIN_PX, SCALE = 0.5, 50, 0.25
original_release, original_features = be.release_frame, be.clip_features


def release_g1(shooter, ball):
    """The current rule, plus: the ball must rise above the top of the shooter's box at k."""
    n = min(len(shooter), len(ball)); x, y, w, h = shooter[:n].T; bx, by = be._centre(ball[:n]).T
    with np.errstate(invalid="ignore"):
        held = (bx > x - be.HAND_REACH * w) & (bx < x + (1 + be.HAND_REACH) * w) & (by > y - be.HAND_REACH * h) & (by < y + h)
    for k in np.flatnonzero(held)[::-1]:
        ahead = by[k + 1:k + 1 + be.RISE_FRAMES]; ahead = ahead[np.isfinite(ahead)]
        if len(ahead) >= 3 and ahead[-1] < by[k] - be.RISE_BH * h[k] and ahead.min() < y[k]:
            return int(k)
    return None


def iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    inter = max(0, min(ax2, bx2) - max(a[0], b[0])) * max(0, min(ay2, by2) - max(a[1], b[1]))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def duplicates(clip, shooter_key):
    """Opponent keys whose box at release overlaps the shooter's with IoU >= 0.5."""
    shooter, ball = be.steady(be.boxes(clip[shooter_key])), be.boxes(clip.get("ball") or {})
    if not np.isfinite(ball).any():
        return []
    r = be.release_frame(shooter, ball)
    team = clip[shooter_key].get("jersey_color")
    if r is None or team is None or not np.isfinite(shooter[r]).all():
        return []
    found = []
    for k, v in clip.items():
        if k in (shooter_key, "ball") or not isinstance(v, dict) or v.get("jersey_color") in (None, team):
            continue
        box = be._at(be.steady(be.boxes(v)), r)
        if np.isfinite(box).all() and iou(box, shooter[r]) >= IOU:
            found.append(k)
    return found


def bad_scale(clip, shooter_key, r):
    shooter = be.steady(be.boxes(clip[shooter_key])); stance = be._stance(shooter, r)
    h = np.nanmedian(stance[:, 3]) if np.isfinite(stance).any() else np.nanmedian(shooter[:r, 3])
    usual = np.nanmedian(shooter[:r, 3])
    return bool(h < MIN_PX or abs(h / usual - 1) > SCALE)


def make_features(g2, g3):
    def features(clip, shooter_key):
        if g2:
            dup = duplicates(clip, shooter_key)
            # A null jersey colour makes the box a non-opponent; it still counts for camera motion.
            clip = {k: ({**v, "jersey_color": None} if k in dup else v) for k, v in clip.items()}
        values = original_features(clip, shooter_key)
        if g3 and values["box_status"] == "ok" and bad_scale(clip, shooter_key, values["release_frame"]):
            values.update({f: np.nan for f in be.FEATURE_COLUMNS})
        return values
    return features


def cohort(g1, g2, g3):
    path = S / f"cohort_g{int(g1)}{int(g2)}{int(g3)}"
    if not (path / "shots.parquet").exists():
        shutil.rmtree(path, ignore_errors=True)
        be.release_frame = release_g1 if g1 else original_release
        be.clip_features = make_features(g2, g3)
        be.prepare_basketevent("data/basketevent_labels_v2", path, load_config("configs/basketevent_strong.json"))
        be.release_frame, be.clip_features = original_release, original_features
    return path


def load_clip(name):
    with gzip.open(f"data/basketevent_labels_v2/games/{name.split('/')[1]}_labels.json.gz", "rt") as s:
        clip = json.load(s)[name]
    key = next(k for k, v in clip.items() if isinstance(v, dict) and isinstance(v.get("event"), dict) and v["event"].get("actionType") in be.SHOT_ACTIONS)
    return clip, key


# Measurement check for G1 on the hand-marked clips.
R = "data/release_review/"
review = pd.read_csv(R + "review.csv", dtype={"item": str}).merge(pd.read_csv(R + "answers.csv", dtype={"item": str}), on="item")
review = review[review.true_release_frame.notna() & review.item.ne("20")]
marked = []
for r in review.itertuples():
    clip, key = load_clip(r.clip)
    shooter, ball = be.steady(be.boxes(clip[key])), be.boxes(clip.get("ball") or {})
    marked.append({"item": r.item, "human": r.true_release_frame, "current": original_release(shooter, ball), "g1": release_g1(shooter, ball)})
marked = pd.DataFrame(marked)
within = {}
for rule in ("current", "g1"):
    e = (marked[rule] - marked.human).dropna()
    within[rule] = int((e.abs() <= 1).sum())
    print(f"{rule:8} found {len(e)}/{len(marked)} within1 {within[rule]} within3 {(e.abs() <= 3).sum()} median|err| {e.abs().median()}")
marked.to_csv(OUT / "g1_hand_marked.csv", index=False)
g1_allowed = within["g1"] >= within["current"]
print("G1 allowed by the measurement check:", g1_allowed, flush=True)

base = cohort(False, False, False)
table = pd.read_parquet(base / "box_features.parquet")
v5 = pd.read_parquet("data/processed/basketevent_cohort_v5/box_features.parquet")
F = sorted(BOX_FEATURES)
same = set(table.shot_id) == set(v5.shot_id) and table.set_index("shot_id")[F].equals(v5.set_index("shot_id").loc[table.shot_id, F])
print("rebuilt D0 cohort matches v5:", same, flush=True)
assert same

failures = pd.read_csv("reports/basketevent_h1_failure_cases/review.csv")
failures["item"] = failures.image.str.extract(r"(\d+)_").astype(int)
WRITTEN_FOR = {"G1": [1, 6], "G2": [2], "G3": [4, 12]}


def changes(step, path, ref):
    """Share of D0 cohort shots whose box features change, and what happens to the review items."""
    new = pd.read_parquet(path / "box_features.parquet").set_index("shot_id")
    old = pd.read_parquet(ref / "box_features.parquet").set_index("shot_id")
    shots = set(pd.read_parquet(ref / "shots.parquet").shot_id)
    old = old.loc[old.index.isin(shots)]
    gone = old.index.difference(new.index[new.box_status.eq("ok")])
    both = old.index.intersection(new.index[new.box_status.eq("ok")])
    a, b = old.loc[both, F], new.loc[both, F]
    changed = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
    items = []
    for item in WRITTEN_FOR[step]:
        sid = failures.loc[failures["item"].eq(item), "shot_id"].item()
        if sid in gone:
            items.append(f"item {item}: no release, leaves the cohort")
        elif sid in both and changed.get(sid, False):
            diff = {f: (round(a.at[sid, f], 2), round(b.at[sid, f], 2)) for f in F if not (a.at[sid, f] == b.at[sid, f] or (pd.isna(a.at[sid, f]) and pd.isna(b.at[sid, f])))}
            items.append(f"item {item}: changed {diff}")
        else:
            items.append(f"item {item}: unchanged" + ("" if sid in old.index else " (not in cohort)"))
    out = {"step": step, "cohort_shots": len(old), "leave_cohort": len(gone), "features_changed": int(changed.sum()), "share_affected": (len(gone) + int(changed.sum())) / len(old), "review_items": "; ".join(items)}
    print(out, flush=True)
    return out


def run(name, path):
    out = OUT / name
    if not (out / "cv_predictions.csv").exists():
        shutil.rmtree(out, ignore_errors=True)
        models.cross_validate(path / "shots.parquet", out, CONFIG, path / "box_features.parquet")
    p = pd.read_csv(out / "cv_predictions.csv", dtype={"game_id": str})
    return p, paired_comparison(p, "xgboost", "xgboost_boxes")


def better(new, old):
    a, b = (p.loc[p.model.eq("xgboost_boxes")] for p in (old, new))
    common = set(a.shot_id) & set(b.shot_id)
    c = paired_comparison(pd.concat([a[a.shot_id.isin(common)].assign(model="old"), b[b.shot_id.isin(common)].assign(model="new")]), "old", "new")
    lo, hi = c["brier_delta_95pct_interval"]; se = (hi - lo) / 3.92
    return c, se, -c["brier_delta_candidate_minus_baseline"] > se


rows = []
def record(step, h1, vs=None, se=None, kept=None, extra=None):
    lo, hi = h1["brier_delta_95pct_interval"]
    row = dict(step=step, shots=h1["n_shots"], h1_delta=h1["brier_delta_candidate_minus_baseline"], h1_lo=lo, h1_hi=hi)
    if vs:
        row.update(gain=-vs["brier_delta_candidate_minus_baseline"], gain_lo=-vs["brier_delta_95pct_interval"][1], gain_hi=-vs["brier_delta_95pct_interval"][0], one_se=se, kept=kept)
    row.update(extra or {})
    rows.append(row); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


kept, kept_p = (False, False, False), None
kept_p, h1 = run("D0", base); record("D0 (current final, rebuilt)", h1)
for i, step in enumerate(("G1", "G2", "G3")):
    if step == "G1" and not g1_allowed:
        record("G1 not run: measurement check failed", h1, extra={"step_run": False}); continue
    flags = tuple(True if j == i else f for j, f in enumerate(kept))
    path = cohort(*flags)
    info = changes(step, path, cohort(*kept))
    p, h1_step = run("".join(s for s, f in zip(("G1", "G2", "G3"), flags) if f), path)
    vs, se, ok = better(p, kept_p)
    record(f"{step} = last kept + {step}", h1_step, vs, se, ok, {k: v for k, v in info.items() if k != "step"})
    if ok:
        kept, kept_p = flags, p
pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
print("final design:", [s for s, f in zip(("G1", "G2", "G3"), kept) if f] or "unchanged (D0)")
