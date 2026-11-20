"""Score scripts/h2_brier_experiments.py outputs and post-hoc calibration variants.

Validation: raw conditions, specialist logit-mean pooling, and Platt
recalibration evaluated with leave-one-game-out cross-validation (so no shot
is calibrated by a map fitted on itself or its own game).

Test: Platt maps fitted on ALL validation predictions of a condition are
applied to that condition's frozen test predictions. No test labels are used
for fitting; this re-scores already recorded outputs and makes no new calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS = 1e-4
TEST_NAME = {"multi": "llm_multi", "single_budget": "llm_single_budget"}


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def metrics(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    from sklearn.metrics import roc_auc_score

    return {
        "n": len(y),
        "mean_p": round(p.mean(), 3),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "auc": round(float(roc_auc_score(y, p)), 4),
    }


def platt(p_fit, y_fit):
    model = LogisticRegression(C=1.0)  # mild L2 keeps slope sane on ~100 shots
    model.fit(logit(p_fit).reshape(-1, 1), y_fit)
    return lambda p: model.predict_proba(logit(p).reshape(-1, 1))[:, 1]


def logo_platt(frame):
    out = np.empty(len(frame))
    for game in frame["game_id"].unique():
        held = frame["game_id"].eq(game).to_numpy()
        out[held] = platt(frame.loc[~held, "p_make"], frame.loc[~held, "y_true"])(
            frame.loc[held, "p_make"]
        )
    return out


def paired_bootstrap(frame, a, b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    games = frame["game_id"].unique()
    per_game = frame.assign(d=(frame[a] - frame.y_true) ** 2 - (frame[b] - frame.y_true) ** 2)
    sums = per_game.groupby("game_id")["d"].agg(["sum", "size"])
    stats = []
    for _ in range(n):
        pick = sums.loc[rng.choice(games, len(games))]
        stats.append(pick["sum"].sum() / pick["size"].sum())
    point = per_game["d"].mean()
    return round(float(point), 4), [round(float(np.quantile(stats, q)), 4) for q in (0.025, 0.975)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="artifacts/h2_brier_experiments")
    parser.add_argument(
        "--val-baselines", default="artifacts/day1/models/validation_predictions.csv"
    )
    parser.add_argument("--test", default="reports/llm_final_test_100/predictions.csv")
    parser.add_argument("--out", help="where to write CSVs and summary.json (default: --run)")
    args = parser.parse_args()
    run = Path(args.run)

    records = [json.loads(x) for x in (run / "predictions.jsonl").read_text().splitlines()]
    preds = pd.DataFrame(records)
    preds = preds[preds.status.eq("ok")]
    selection = pd.read_csv(run / "selection.csv")
    base = pd.read_csv(args.val_baselines)
    base = base[base.shot_id.isin(selection.shot_id)]

    wide = selection[["shot_id", "game_id", "y_true"]].copy()
    for model, group in base.groupby("model"):
        wide = wide.merge(
            group[["shot_id", "p_make"]].rename(columns={"p_make": model}), on="shot_id"
        )

    for condition, group in preds.groupby("condition"):
        g = group[["shot_id", "game_id", "y_true", "p_make", "opinions"]].reset_index(drop=True)
        wide = wide.merge(
            g[["shot_id", "p_make"]].rename(columns={"p_make": condition}), on="shot_id", how="left"
        )
        if g["opinions"].map(len).gt(0).all():
            pooled = g["opinions"].map(
                lambda ops: 1 / (1 + np.exp(-np.mean(logit([o["p_make"] for o in ops]))))
            )
            wide = wide.merge(
                g[["shot_id"]].assign(**{f"{condition}+pool": pooled}), on="shot_id", how="left"
            )
        if len(g) >= 20:
            wide = wide.merge(
                g[["shot_id"]].assign(**{f"{condition}+platt_cv": logo_platt(g)}),
                on="shot_id",
                how="left",
            )

    llm_cols = [c for c in wide.columns if c not in {"shot_id", "game_id", "y_true"}]
    complete = wide.dropna(subset=llm_cols)
    rate = complete.y_true.mean()
    print(f"Validation: {len(complete)} shots with every condition, make rate {rate:.2f}\n")
    table = pd.DataFrame({c: metrics(complete.y_true, complete[c]) for c in llm_cols}).T
    print(table.sort_values("brier").to_string(), "\n")
    summary = {"validation": table.to_dict("index"), "validation_vs_multi": {}}
    if "multi" in complete:
        for c in llm_cols:
            if c != "multi":
                diff = paired_bootstrap(complete, c, "multi")
                summary["validation_vs_multi"][c] = diff
                print(f"  {c:28s} - multi: Brier diff, 95% CI = {diff}")

    test = pd.read_csv(args.test)
    test_wide = test.pivot(index="shot_id", columns="model", values="p_make").reset_index()
    test_wide = test_wide.merge(
        test[["shot_id", "game_id", "y_true"]].drop_duplicates(), on="shot_id"
    )
    print("\nTest (frozen predictions, calibrators fitted on validation only):")
    rows = {}
    for condition, name in TEST_NAME.items():
        g = preds[preds.condition.eq(condition)]
        if len(g) < 20:
            continue
        test_wide[f"{name}+platt"] = platt(g.p_make, g.y_true)(test_wide[name])
    for c in [c for c in test_wide.columns if c not in {"shot_id", "game_id", "y_true"}]:
        rows[c] = metrics(test_wide.y_true, test_wide[c])
    print(pd.DataFrame(rows).T.sort_values("brier").to_string())
    summary["test"] = rows
    summary["test_paired"] = {}
    for a, b in [
        ("llm_multi", "llm_single_budget"),
        ("llm_multi+platt", "llm_single_budget+platt"),
        ("llm_multi+platt", "llm_multi"),
        ("llm_multi+platt", "logistic_regression"),
        ("llm_multi+platt", "xgboost"),
    ]:
        if a in test_wide and b in test_wide:
            diff = paired_bootstrap(test_wide, a, b)
            summary["test_paired"][f"{a} - {b}"] = diff
            print(f"  {a} - {b}: {diff}")
    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out / "validation_wide.csv", index=False)
    test_wide.to_csv(out / "test_recalibrated.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
