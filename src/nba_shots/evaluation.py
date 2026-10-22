"""Common scoring and paired, game-clustered uncertainty estimates."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from .common import PREDICTION_COLUMNS, write_json


def validate_predictions(predictions: pd.DataFrame) -> None:
    if missing := set(PREDICTION_COLUMNS) - set(predictions):
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("No predictions to evaluate")
    if predictions[["shot_id", "game_id", "cohort", "split", "model", "y_true"]].isna().any().any():
        raise ValueError("Prediction identity fields and labels cannot be missing")
    if predictions.duplicated(["cohort", "model", "shot_id"]).any():
        raise ValueError("Duplicate predictions for a model/shot")
    if not predictions["y_true"].isin([0, 1]).all():
        raise ValueError("Prediction labels must be binary")
    if predictions.groupby(["cohort", "shot_id"])["y_true"].nunique().gt(1).any():
        raise ValueError("Models disagree on target labels")
    if predictions.groupby(["cohort", "shot_id"])["game_id"].nunique().gt(1).any():
        raise ValueError("Models disagree on game identity")
    if predictions.groupby(["cohort", "shot_id"])["split"].nunique().gt(1).any():
        raise ValueError("Models disagree on split identity")
    if "status" in predictions:
        if not predictions["status"].isin(["ok", "error"]).all():
            raise ValueError("Prediction status must be ok/error")
        successful = predictions["status"].eq("ok")
    else:
        successful = pd.Series(True, index=predictions.index)
    probabilities = pd.to_numeric(predictions["p_make"], errors="coerce")
    valid = np.isfinite(probabilities) & probabilities.between(0, 1)
    if (successful & ~valid).any():
        raise ValueError("Successful predictions must have finite p_make in [0,1]")
    if (~successful & probabilities.notna()).any():
        raise ValueError("Failed predictions must not invent a probability")


def score_predictions(predictions: pd.DataFrame) -> list[dict]:
    validate_predictions(predictions)
    results = []
    for (cohort, split, model), rows in predictions.groupby(["cohort", "split", "model"]):
        good = rows if "status" not in rows else rows.loc[rows["status"].eq("ok")]
        entry = {
            "cohort": cohort,
            "split": split,
            "model": model,
            "n_total": len(rows),
            "n_scored": len(good),
            "n_failed": len(rows) - len(good),
            "coverage": len(good) / len(rows),
            "brier_score": None,
            "log_loss": None,
            "roc_auc": None,
            "accuracy": None,
        }
        if len(good):
            y, p = good["y_true"].to_numpy(), good["p_make"].to_numpy(dtype=float)
            entry.update(
                {
                    "brier_score": float(brier_score_loss(y, p)),
                    "log_loss": float(log_loss(y, p, labels=[0, 1])),
                    "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                    "accuracy": float(accuracy_score(y, p >= 0.5)),
                }
            )
        results.append(entry)
    return results


def write_scores(predictions: pd.DataFrame, output_dir: str | Path, prefix="validation"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = score_predictions(predictions)
    write_json(output_dir / f"{prefix}_metrics.json", metrics)
    pd.DataFrame(metrics).to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    # Use a headless backend; never require a desktop or mutate notebook state.
    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for (cohort, split), group in predictions.groupby(["cohort", "split"]):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
        for model, rows in group.groupby("model"):
            good = rows if "status" not in rows else rows.loc[rows["status"].eq("ok")]
            if good.empty:
                continue
            actual, predicted = calibration_curve(
                good["y_true"],
                good["p_make"],
                n_bins=10,
                strategy="uniform",
            )
            ax.plot(predicted, actual, marker="o", label=model)
        ax.set(
            xlabel="Mean predicted make probability",
            ylabel="Observed make fraction",
            xlim=(0, 1),
            ylim=(0, 1),
            title=f"{cohort} / {split}",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{cohort}_{split}_calibration.png", dpi=160)
        plt.close(fig)
    return metrics


def paired_comparison(
    predictions: pd.DataFrame, baseline: str, candidate: str, repetitions=1000, seed=42
) -> dict:
    """Require identical cohorts, labels and shot IDs; never silently intersect rows."""
    validate_predictions(predictions)
    if repetitions < 1 or baseline == candidate:
        raise ValueError("Use different models and at least one bootstrap repetition")
    if predictions["cohort"].nunique() != 1 or predictions["split"].nunique() != 1:
        raise ValueError("Compare one cohort and one split at a time")
    a = predictions.loc[predictions["model"].eq(baseline)].set_index("shot_id")
    b = predictions.loc[predictions["model"].eq(candidate)].set_index("shot_id")
    if a.empty or set(a.index) != set(b.index):
        raise ValueError("Paired comparison requires exactly the same nonempty shot IDs")
    for part in [a, b]:
        if "status" in part and not part["status"].eq("ok").all():
            raise ValueError("Resolve prediction failures before paired comparison; see coverage")
    b = b.loc[a.index]
    if not np.array_equal(a["y_true"].to_numpy(), b["y_true"].to_numpy()):
        raise ValueError("Paired predictions have inconsistent labels")
    y = a["y_true"].to_numpy()
    delta = (b["p_make"].to_numpy() - y) ** 2 - (a["p_make"].to_numpy() - y) ** 2
    groups = pd.DataFrame({"game": a["game_id"].to_numpy(), "delta": delta}).groupby("game")
    sums = groups["delta"].sum().to_numpy()
    sizes = groups.size().to_numpy()
    if len(sums) < 2:
        raise ValueError("At least two held-out games are required for game bootstrap intervals")
    indices = np.random.default_rng(seed).integers(0, len(sums), size=(repetitions, len(sums)))
    samples = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
    lo, hi = np.quantile(samples, [0.025, 0.975])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "n_shots": len(a),
        "n_games": len(sums),
        "brier_delta_candidate_minus_baseline": float(delta.mean()),
        "brier_delta_95pct_interval": [float(lo), float(hi)],
        "method": "paired bootstrap of whole games; negative delta favors candidate",
        "repetitions": repetitions,
        "seed": seed,
    }
