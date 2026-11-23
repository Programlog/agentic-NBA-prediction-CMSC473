"""Offline evidence for H2 experiment 07 (no model calls).

For the validation (500 shots, 125 games x 4) and test (1,000 shots, 125 games x 8)
selections, computes from the TRAINING split only:
  p_xgb, p_lr, p_constant   frozen day-1 baselines
  knn_*                     the 50 nearest training shots (standardized features)
  effect_text               training make rates in this shot's distance band,
                            broken down by each other factor's band
Writes artifacts/h2_prompts/{validation,test}_rows.parquet and prepare.json, plus
reference metrics (incl. a kNN-50 model) on the selections and on the full splits.

Usage: uv run --frozen --all-extras python experiments/h2/07_prompt_tuning/prepare.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from nba_shots.agents import select_evaluation_rows

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "analyze04", HERE.parent / "04_brier_followup" / "analyze.py"
)
a04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(a04)

SHOTS = Path("artifacts/day1/prepared/shots.parquet")
MODELS = Path("artifacts/day1/models")
OUT = Path("artifacts/h2_prompts")
SEED = 42
SAMPLES = {"validation": (500, 125), "test": (1000, 125)}
K = 50
N_EXAMPLES = 5
KNN_FEATURES = [
    "shot_distance",
    "defender_distance",
    "shot_clock",
    "dribbles",
    "touch_time",
    "shot_value",
]
DISTANCE_BANDS = [0, 3, 6, 10, 16, 22, 24, 28, np.inf]  # same as 04
FACTOR_BANDS = {
    "defender_distance": ([0, 2, 4, 6, 10, np.inf], "closest defender {} ft away"),
    "shot_clock": ([0, 4, 8, 14, 20, np.inf], "shot clock {} s"),
    "dribbles": ([0, 1, 2, 5, 10, np.inf], "{} dribbles"),
    "touch_time": ([-np.inf, 1, 2, 4, 8, np.inf], "touch time {} s"),
}
MIN_CELL = 30


def select_rows(shots: pd.DataFrame, split: str, limit: int, seed: int, games: int):
    """``select_evaluation_rows``' balanced-games branch without its 500-shot cap."""
    rows = shots.loc[shots["split"].eq(split)].copy()

    def rank(kind, value):
        return hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest()

    base_count, remainder = divmod(limit, games)
    per_game = base_count + (remainder > 0)
    sizes = rows.groupby("game_id").size()
    eligible = [g for g, n in sizes.items() if n >= per_game]
    if len(eligible) < games:
        raise ValueError(f"only {len(eligible)} games have {per_game}+ shots")
    chosen = sorted(eligible, key=lambda g: (rank("game", str(g)), str(g)))[:games]
    parts = []
    for position, game in enumerate(chosen):
        take = base_count + (position < remainder)
        g = rows.loc[rows["game_id"].eq(game)].copy()
        g["_r"] = g["shot_id"].map(lambda s: rank("shot", str(s)))
        parts.append(g.sort_values(["_r", "shot_id"]).head(take))
    out = pd.concat(parts, ignore_index=True).drop(columns="_r")
    assert len(out) == limit and out.game_id.nunique() == games
    return out.sort_values("shot_id").reset_index(drop=True)


def band_label(interval: pd.Interval, integer: bool = False) -> str:
    lo, hi = interval.left, interval.right
    if integer and not np.isinf(hi):
        return f"{lo:g}" if hi - lo == 1 else f"{lo:g}-{hi - 1:g}"
    if np.isinf(hi):
        return f"{lo:g}+"
    if np.isinf(lo):
        return f"under {hi:g}"
    return f"{lo:g}-{hi:g}"


class Effects:
    """Training make rates within the shot's distance band, by each other factor."""

    def __init__(self, train: pd.DataFrame):
        self.overall = float(train.y_true.mean())
        t = train.assign(_d=pd.cut(train.shot_distance, DISTANCE_BANDS, right=False))
        self.distance = t.groupby("_d", observed=True).y_true.agg(["mean", "size"])
        self.cells = {}
        for col, (bands, _) in FACTOR_BANDS.items():
            cut = pd.cut(t[col], bands, right=False)
            self.cells[col] = t.groupby(["_d", cut], observed=True).y_true.agg(["mean", "size"])

    def text(self, row: dict) -> str:
        d = pd.cut([row["shot_distance"]], DISTANCE_BANDS, right=False)[0]
        rate, n = self.distance.loc[d]
        parts = [
            f"Training make rates (91k earlier shots; overall {self.overall:.1%}). "
            f"Shots from {band_label(d)} ft: {rate:.1%} (n={int(n)}). Within that distance band:"
        ]
        for col, (bands, label) in FACTOR_BANDS.items():
            value = row[col]
            if value is None or pd.isna(value):
                continue
            band = pd.cut([value], bands, right=False)[0]
            cell = self.cells[col]
            if (d, band) in cell.index and cell.loc[(d, band), "size"] >= MIN_CELL:
                r, m = cell.loc[(d, band)]
                parts.append(
                    f"{label.format(band_label(band, col == 'dribbles'))}: {r:.1%} (n={int(m)});"
                )
        return " ".join(parts).rstrip(";") + "."


def knn_matrix(frame: pd.DataFrame, medians: pd.Series, mean, std) -> np.ndarray:
    x = frame[KNN_FEATURES].astype(float).fillna(medians)
    return ((x - mean) / std).to_numpy()


def describe(r) -> str:
    clock = "?" if pd.isna(r.shot_clock) else f"{r.shot_clock:.0f}s"
    touch = "?" if pd.isna(r.touch_time) else f"{r.touch_time:.1f}s"
    return (
        f"{r.shot_distance:.1f}ft, defender {r.defender_distance:.1f}ft, clock {clock}, "
        f"{int(r.dribbles)} dribbles, touch {touch}, {int(r.shot_value)}pt: "
        f"{'MADE' if r.y_true else 'missed'}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shots = pd.read_parquet(SHOTS)
    train = shots[shots.split.eq("train")].reset_index(drop=True)

    # the uncapped selector must reproduce the frozen one, and old samples must nest
    for split in ["validation", "test"]:
        for limit, games in [(100, 50), (500, 125)]:
            ours = select_rows(shots, split, limit, SEED, games)
            frozen = select_evaluation_rows(shots, split, limit, SEED, games)
            assert ours.shot_id.tolist() == frozen.shot_id.tolist(), (split, limit)

    models = {n: joblib.load(MODELS / f"{n}.joblib") for n in ["xgboost", "logistic_regression"]}
    medians = train[KNN_FEATURES].median()
    xtrain = train[KNN_FEATURES].astype(float).fillna(medians)
    mean, std = xtrain.mean(), xtrain.std()
    nn = NearestNeighbors(n_neighbors=K).fit(((xtrain - mean) / std).to_numpy())
    train_y = train.y_true.to_numpy()
    effects = Effects(train)

    def knn_rate(frame):
        _, idx = nn.kneighbors(knn_matrix(frame, medians, mean, std))
        return idx, train_y[idx].mean(axis=1)

    meta = {"shots_sha256": hashlib.sha256(SHOTS.read_bytes()).hexdigest(), "k": K, "splits": {}}
    frozen = pd.concat(
        pd.read_csv(f, dtype={"game_id": str})
        for f in [
            "artifacts/day1/report/predictions.csv",  # full validation split
            HERE.parent / "02_final_test_100" / "report" / "predictions.csv",  # 100 test shots
        ]
    )
    for split, (limit, games) in SAMPLES.items():
        rows = select_rows(shots, split, limit, SEED, games)
        old = select_evaluation_rows(shots, split, 100, SEED, 50)
        rows["in_old_100"] = rows.shot_id.isin(old.shot_id)
        assert rows.in_old_100.sum() == 100
        for name, m in models.items():
            rows[f"p_{name}"] = m["pipeline"].predict_proba(rows[m["features"]])[:, 1]
        rows["p_constant"] = float(train.y_true.mean())
        idx, rate = knn_rate(rows)
        assert not set(train.shot_id.iloc[idx.ravel()]) & set(rows.shot_id)  # train only
        rows["knn_made"] = train_y[idx].sum(axis=1)
        rows["knn_k"] = K
        rows["p_knn"] = rate
        rows["knn_examples"] = [
            "; ".join(describe(train.iloc[j]) for j in neighbours[:N_EXAMPLES])
            for neighbours in idx
        ]
        rows["effect_text"] = [effects.text(r) for r in rows.to_dict("records")]

        # the recomputed XGBoost must match the frozen day-1 predictions
        ref = frozen[(frozen.split == split) & (frozen.model == "xgboost")].set_index("shot_id")
        common = rows[rows.shot_id.isin(ref.index)]
        diff = float(np.abs(common.p_xgboost.to_numpy() - ref.loc[common.shot_id, "p_make"]).max())
        assert len(common) == (limit if split == "validation" else 100) and diff < 1e-9, diff

        rows.to_parquet(OUT / f"{split}_rows.parquet", index=False)
        full = shots[shots.split.eq(split)].copy()
        for name, m in models.items():
            full[f"p_{name}"] = m["pipeline"].predict_proba(full[m["features"]])[:, 1]
        full["p_knn"] = knn_rate(full)[1]
        refs = ["p_xgboost", "p_logistic_regression", "p_knn", "p_constant"]
        full["p_constant"] = float(train.y_true.mean())
        meta["splits"][split] = {
            "n": len(rows),
            "games": games,
            "make_rate": round(float(rows.y_true.mean()), 4),
            "max_abs_diff_vs_frozen_xgb": diff,
            "sha256": hashlib.sha256((OUT / f"{split}_rows.parquet").read_bytes()).hexdigest(),
            "selection": {c: a04.metrics(rows.y_true, rows[c]) for c in refs},
            "full_split": {c: a04.metrics(full.y_true, full[c]) for c in refs},
        }
        print(split, json.dumps(meta["splits"][split], indent=1))
    (OUT / "prepare.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
