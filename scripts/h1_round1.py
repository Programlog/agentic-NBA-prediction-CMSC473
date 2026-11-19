"""H1 improvement round, as written in docs/h1_improvement_round.md. Train+validation only."""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from nba_shots import basketevent as be, models
from nba_shots.common import EXTRA_FEATURES, load_config
from nba_shots.evaluation import paired_comparison

S = Path(sys.argv[1]); OUT = S / "h1_round"; OUT.mkdir(exist_ok=True)
CONFIG = load_config("configs/basketevent_h1_final.json")
TOP = 0.15  # E1 rule chosen on the hand-marked clips
NEW = "release_height_vs_player"
original_release, original_features = be.release_frame, be.clip_features

def release_e1(shooter, ball):
    n = min(len(shooter), len(ball)); x, y, w, h = shooter[:n].T; bx, by = be._centre(ball[:n]).T
    with np.errstate(invalid="ignore"):
        held = (bx > x - be.HAND_REACH * w) & (bx < x + (1 + be.HAND_REACH) * w) & (by > y - TOP * h) & (by < y + h)
    for k in np.flatnonzero(held)[::-1]:
        ahead = by[k + 1:k + 1 + be.RISE_FRAMES]; ahead = ahead[np.isfinite(ahead)]
        if len(ahead) >= 3 and ahead[-1] < by[k] - be.RISE_BH * h[k]:
            return int(k)
    return None

def features_e2(clip, shooter_key):
    """E2: release height as the median over the release frame and its neighbours."""
    values = original_features(clip, shooter_key)
    if values["box_status"] != "ok":
        return values
    shooter, ball = be.steady(be.boxes(clip[shooter_key])), be.boxes(clip.get("ball") or {})
    r = values["release_frame"]; stance = be._stance(shooter, r)
    h = np.nanmedian(stance[:, 3]) if np.isfinite(stance).any() else np.nanmedian(shooter[:r, 3])
    heights = [(be._feet(be._at(shooter, k))[1] - be._centre(be._at(ball, k))[1]) / h for k in (r - 1, r, r + 1)]
    values["release_height_bh"] = float(np.nanmedian(heights)) if np.isfinite(heights).any() else np.nan
    return values

def prepare(name, e1, e2):
    out = S / f"cohort_{name}"
    if not (out / "shots.parquet").exists():
        shutil.rmtree(out, ignore_errors=True)
        be.release_frame = release_e1 if e1 else original_release
        be.clip_features = features_e2 if e2 else original_features
        be.prepare_basketevent("data/basketevent_labels_v2", out, load_config("configs/basketevent_strong.json"))
        be.release_frame, be.clip_features = original_release, original_features
    return out

def add_player_relative(cohort):
    """E3: release height minus the shooter's mean over earlier-date shots (>=5), else blank."""
    path = cohort / "box_features_e3.parquet"
    if not path.exists():
        x = pd.read_parquet(cohort / "shots.parquet")[["shot_id", "player_id", "game_date"]].merge(pd.read_parquet(cohort / "box_features.parquet"), on="shot_id")
        daily = x.groupby(["player_id", "game_date"]).release_height_bh.agg(["sum", "count"]).reset_index().sort_values(["player_id", "game_date"])
        g = daily.groupby("player_id")
        daily["prior_n"] = g["count"].cumsum() - daily["count"]; daily["prior_sum"] = g["sum"].cumsum() - daily["sum"]
        x = x.merge(daily[["player_id", "game_date", "prior_n", "prior_sum"]], on=["player_id", "game_date"])
        x[NEW] = np.where(x.prior_n >= 5, x.release_height_bh - x.prior_sum / x.prior_n.where(x.prior_n > 0), np.nan)
        x.drop(columns=["player_id", "game_date", "prior_n", "prior_sum"]).to_parquet(path)
    return path

def run(name, cohort, table, e3=False):
    out = OUT / name
    if e3:  # the new column joins the box set for this run only
        EXTRA_FEATURES["boxes"] = EXTRA_FEATURES["boxes"] | {NEW}; models.ALLOWED_FEATURES = models.ALLOWED_FEATURES | {NEW}
    if not (out / "cv_predictions.csv").exists():
        shutil.rmtree(out, ignore_errors=True)
        models.cross_validate(cohort / "shots.parquet", out, CONFIG, table)
    if e3:
        EXTRA_FEATURES["boxes"] = EXTRA_FEATURES["boxes"] - {NEW}
    p = pd.read_csv(out / "cv_predictions.csv", dtype={"game_id": str})
    h1 = paired_comparison(p, "xgboost", "xgboost_boxes")
    return p, h1

def better(new, old):
    """Paired game bootstrap of box-model Brier, new minus old, on the shots both score."""
    a, b = (p.loc[p.model.eq("xgboost_boxes")] for p in (old, new))
    common = set(a.shot_id) & set(b.shot_id)
    pair = pd.concat([a[a.shot_id.isin(common)].assign(model="old"), b[b.shot_id.isin(common)].assign(model="new")])
    c = paired_comparison(pair, "old", "new")
    se = (c["brier_delta_95pct_interval"][1] - c["brier_delta_95pct_interval"][0]) / 3.92
    return c, se, -c["brier_delta_candidate_minus_baseline"] > se

summary = []
def record(step, h1, vs=None, se=None, kept=None):
    lo, hi = h1["brier_delta_95pct_interval"]
    row = dict(step=step, shots=h1["n_shots"], h1_delta=h1["brier_delta_candidate_minus_baseline"], h1_lo=lo, h1_hi=hi)
    if vs:
        row.update(vs_kept_delta=vs["brier_delta_candidate_minus_baseline"], vs_kept_lo=vs["brier_delta_95pct_interval"][0],
                   vs_kept_hi=vs["brier_delta_95pct_interval"][1], one_se=se, vs_shots=vs["n_shots"], kept=kept)
    summary.append(row); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

v5 = Path("data/processed/basketevent_cohort_v5")
kept_p, h1 = run("D0", v5, v5 / "box_features.parquet"); record("D0 (current final)", h1)
e1_kept = False
c1 = prepare("e1", e1=True, e2=False)
p, h1 = run("D1_e1", c1, c1 / "box_features.parquet"); vs, se, ok = better(p, kept_p); record("D1 = D0 + E1 release rule", h1, vs, se, ok)
if ok: kept_p, e1_kept = p, True
c2 = prepare("e1e2" if e1_kept else "e2", e1=e1_kept, e2=True)
p, h1 = run("D2_e2", c2, c2 / "box_features.parquet"); vs, se, ok = better(p, kept_p); record("D2 = last kept + E2 smoothing", h1, vs, se, ok)
if ok: kept_p, e2_kept, base = p, True, c2
else:
    e2_kept = False; base = c1 if e1_kept else v5
p, h1 = run("D3_e3", base, add_player_relative(base), e3=True); vs, se, ok = better(p, kept_p); record("D3 = last kept + E3 player-relative", h1, vs, se, ok)
pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
print("kept:", {"E1": e1_kept, "E2": e2_kept, "E3": ok})
