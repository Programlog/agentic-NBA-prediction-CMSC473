"""Seed check for G3, as written in docs/h1_box_error_round.md. Train+validation only; never the test split.

    python scripts/h1_box_error_seeds.py SCRATCH

Uses the cohorts built by scripts/h1_box_error_round.py in SCRATCH (cohort_g000 = D0, cohort_g001 = D0 + G3).
"""
import shutil, sys
from pathlib import Path
import pandas as pd
from nba_shots import models
from nba_shots.common import load_config
from nba_shots.evaluation import paired_comparison

S = Path(sys.argv[1]); OUT = S / "h1_box_error_round"
CONFIG = load_config("configs/basketevent_h1_final.json")
SEEDS = [42, 1, 2, 3, 4, 5]
DESIGNS = {"D0": "cohort_g000", "G3": "cohort_g001"}


def run(design, seed):
    # Seed 42 is the round's own run.
    out = OUT / (design if seed == 42 else f"{design}_seed{seed}")
    if not (out / "cv_predictions.csv").exists():
        shutil.rmtree(out, ignore_errors=True)
        cohort = S / DESIGNS[design]
        models.cross_validate(cohort / "shots.parquet", out, {**CONFIG, "seed": seed}, cohort / "box_features.parquet")
    return pd.read_csv(out / "cv_predictions.csv", dtype={"game_id": str})


def gain(new, old):
    a, b = (p.loc[p.model.eq("xgboost_boxes")] for p in (old, new))
    c = paired_comparison(pd.concat([a.assign(model="old"), b.assign(model="new")]), "old", "new")
    lo, hi = c["brier_delta_95pct_interval"]
    return -c["brier_delta_candidate_minus_baseline"], -hi, -lo, (hi - lo) / 3.92


def h1(p):
    c = paired_comparison(p, "xgboost", "xgboost_boxes")
    return c["brier_delta_candidate_minus_baseline"], *c["brier_delta_95pct_interval"]


rows, runs = [], {d: [] for d in DESIGNS}
for seed in SEEDS:
    d0, g3 = run("D0", seed), run("G3", seed)
    runs["D0"].append(d0); runs["G3"].append(g3)
    g, glo, ghi, se = gain(g3, d0)
    rows.append({"seed": seed, "gain": g, "gain_lo": glo, "gain_hi": ghi, "one_se": se,
                 **dict(zip(["d0_h1", "d0_lo", "d0_hi"], h1(d0))), **dict(zip(["g3_h1", "g3_lo", "g3_hi"], h1(g3)))})
    print({k: round(v, 5) if isinstance(v, float) else v for k, v in rows[-1].items()}, flush=True)


def average(parts):
    keys = ["shot_id", "game_id", "cohort", "split", "model", "y_true", "status"]
    return pd.concat(parts).groupby(keys, as_index=False).p_make.mean()


d0, g3 = average(runs["D0"]), average(runs["G3"])
g, glo, ghi, se = gain(g3, d0)
rows.append({"seed": "mean of 6", "gain": g, "gain_lo": glo, "gain_hi": ghi, "one_se": se,
             **dict(zip(["d0_h1", "d0_lo", "d0_hi"], h1(d0))), **dict(zip(["g3_h1", "g3_lo", "g3_hi"], h1(g3)))})
print({k: round(v, 5) if isinstance(v, float) else v for k, v in rows[-1].items()})
table = pd.DataFrame(rows)
table.to_csv(OUT / "seeds.csv", index=False)
positive = int((table.gain.iloc[:-1] > 0).sum())
robust = g > se and positive >= 5
print(f"gain positive for {positive}/6 seeds; seed-averaged gain {g:.5f} vs 1 SE {se:.5f}; robust: {robust}")
