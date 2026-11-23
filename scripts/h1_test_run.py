"""The one H1 test-split evaluation, as fixed in docs/h1_test_protocol.md.

    python scripts/h1_test_run.py COHORT OUT

Fits D0 on training games for six seeds, scores each once on the test games, averages each
shot's probability over seeds, and compares the box model with the baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from nba_shots.common import load_config, write_json
from nba_shots.evaluation import paired_comparison
from nba_shots.models import evaluate_run, train_baselines

SEEDS = [42, 1, 2, 3, 4, 5]
MODELS = ["constant", "logistic_regression", "xgboost", "xgboost_boxes"]


def scores(predictions: pd.DataFrame) -> dict:
    from sklearn.metrics import log_loss, roc_auc_score

    out = {}
    for model, part in predictions.groupby("model"):
        y, p = part["y_true"], part["p_make"]
        out[model] = {
            "brier": float(((p - y) ** 2).mean()),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "roc_auc": float(roc_auc_score(y, p)),
            "shots": len(part),
            "games": int(part["game_id"].nunique()),
        }
    return out


def averaged(frames: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat(frames, ignore_index=True)
    keys = ["shot_id", "game_id", "cohort", "split", "model", "y_true"]
    return stacked.groupby(keys, as_index=False)["p_make"].mean()


def main(cohort: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; the test split is scored once")
    shots, boxes = cohort / "shots.parquet", cohort / "box_features.parquet"
    base = load_config("configs/basketevent_h1_final.json")
    per_seed, frames = {}, {"validation": [], "test": []}
    for seed in SEEDS:
        run = out / f"seed_{seed}"
        train_baselines(shots, run, {**base, "seed": seed}, boxes)
        evaluate_run(run, shots, split="test", final=True)
        for split in frames:
            preds = pd.read_csv(run / f"{split}_predictions.csv", dtype={"game_id": str})
            frames[split].append(preds)
        test = frames["test"][-1]
        per_seed[seed] = paired_comparison(test, "xgboost", "xgboost_boxes")
        print(f"seed {seed}: {per_seed[seed]['brier_delta_candidate_minus_baseline']:+.5f}")

    result = {"protocol": "docs/h1_test_protocol.md", "seeds": SEEDS}
    for split, parts in frames.items():
        mean = averaged(parts)
        mean.to_csv(out / f"{split}_predictions_seed_mean.csv", index=False)
        result[split] = {
            "h1_seed_mean": paired_comparison(mean, "xgboost", "xgboost_boxes"),
            "scores_seed_mean": scores(mean),
        }
    result["test"]["h1_per_seed"] = {str(s): per_seed[s] for s in SEEDS}
    write_json(out / "summary.json", result)
    print(json.dumps(result["test"]["h1_seed_mean"], indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
