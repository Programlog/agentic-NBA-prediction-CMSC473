"""H1 round 2, as written in docs/h1_improvement_round.md. Train+validation only; never the test split."""
import shutil, sys
from pathlib import Path
import numpy as np, pandas as pd
from nba_shots import basketevent as be, models
from nba_shots.common import BOX_FEATURES, EXTRA_FEATURES, load_config
from nba_shots.evaluation import paired_comparison

S = Path(sys.argv[1]); OUT = S / "h1_round2"; OUT.mkdir(exist_ok=True)
CONFIG = load_config("configs/basketevent_h1_final.json")
F1, F2, F3 = "shooter_extension", "release_after_peak_s", "lean_toward_basket_bh"
NEW = [F1, F2, F3]
original = be.clip_features

def features_r2(clip, shooter_key):
    values = original(clip, shooter_key)
    if values["box_status"] != "ok":
        return values
    shooter, ball = be.steady(be.boxes(clip[shooter_key])), be.boxes(clip.get("ball") or {})
    r = values["release_frame"]; stance = be._stance(shooter, r)
    h = np.nanmedian(stance[:, 3]) if np.isfinite(stance).any() else np.nanmedian(shooter[:r, 3])
    values[F1] = float(shooter[r, 3] / h)
    window = range(max(0, r - 8), min(len(shooter), r + 9))
    feet_y = np.array([be._feet(shooter[k])[1] for k in window])
    if np.isfinite(feet_y).sum() >= 5:
        peak = window[int(np.nanargmin(feet_y))]  # smallest y = highest on screen
        values[F2] = (r - peak) / be.FPS
    ball_x = be._centre(ball[r])[0]; flight_x = be._centre(be._at(ball, r + be.FLIGHT_FRAMES))[0]
    toward = np.sign(flight_x - ball_x)
    if toward:
        values[F3] = float((ball_x - be._feet(shooter[r])[0]) * toward / h)
    return values

cohort = S / "cohort_r2"
if not (cohort / "shots.parquet").exists():
    shutil.rmtree(cohort, ignore_errors=True)
    be.FEATURE_COLUMNS = sorted(BOX_FEATURES | set(NEW)); be.clip_features = features_r2
    be.prepare_basketevent("data/basketevent_labels_v2", cohort, load_config("configs/basketevent_strong.json"))
    be.FEATURE_COLUMNS = sorted(BOX_FEATURES); be.clip_features = original
EXTRA_FEATURES["boxes"] = BOX_FEATURES | set(NEW); models.ALLOWED_FEATURES = models.ALLOWED_FEATURES | set(NEW)
table = pd.read_parquet(cohort / "box_features.parquet")
v5 = pd.read_parquet("data/processed/basketevent_cohort_v5/box_features.parquet")
same = table.set_index("shot_id")[sorted(BOX_FEATURES)].equals(v5.set_index("shot_id").loc[table.shot_id, sorted(BOX_FEATURES)])
print("same shots and original features as v5:", same and set(table.shot_id) == set(v5.shot_id))
print(table[NEW].describe(percentiles=[.01, .5, .99]).round(3).to_string())

def run(name, extra):
    out = OUT / name
    if not (out / "cv_predictions.csv").exists():
        shutil.rmtree(out, ignore_errors=True)
        models.cross_validate(cohort / "shots.parquet", out, {**CONFIG, "extra_features": extra}, cohort / "box_features.parquet")
    p = pd.read_csv(out / "cv_predictions.csv", dtype={"game_id": str})
    return p, paired_comparison(p, "xgboost", "xgboost_boxes")

def better(new, old):
    a, b = (p.loc[p.model.eq("xgboost_boxes")] for p in (old, new))
    common = set(a.shot_id) & set(b.shot_id)
    c = paired_comparison(pd.concat([a[a.shot_id.isin(common)].assign(model="old"), b[b.shot_id.isin(common)].assign(model="new")]), "old", "new")
    lo, hi = c["brier_delta_95pct_interval"]; se = (hi - lo) / 3.92
    return c, se, -c["brier_delta_candidate_minus_baseline"] > se

rows = []
def record(step, h1, vs=None, se=None, kept=None):
    lo, hi = h1["brier_delta_95pct_interval"]
    row = dict(step=step, shots=h1["n_shots"], h1_delta=h1["brier_delta_candidate_minus_baseline"], h1_lo=lo, h1_hi=hi)
    if vs:
        row.update(gain=-vs["brier_delta_candidate_minus_baseline"], gain_lo=-vs["brier_delta_95pct_interval"][1], gain_hi=-vs["brier_delta_95pct_interval"][0], one_se=se, kept=kept)
    rows.append(row); print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)

kept = sorted(BOX_FEATURES)
kept_p, h1 = run("D0", kept); record("D0 (current final, rebuilt)", h1)
for name, f in [("R1_extension", F1), ("R2_release_after_peak", F2), ("R3_lean", F3)]:
    p, h1 = run(name, kept + [f]); vs, se, ok = better(p, kept_p); record(f"{name} = last kept + {f}", h1, vs, se, ok)
    if ok: kept, kept_p = kept + [f], p
pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
print("final box features:", [f for f in kept if f in NEW] or "unchanged (D0)")
