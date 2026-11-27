"""Accuracy and a reliability diagram for the saved H1 test predictions.

    python scripts/h1_accuracy_reliability.py OUT

Reads the six-seed averaged test predictions written by the one H1 test run. Nothing is refit and
no prediction is changed. Accuracy uses a fixed 0.5 threshold (the proposal's metric, not tuned).
The reliability diagram groups shots into ten equal-count bins of predicted probability.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots.evaluation import score_predictions

TEST = Path("reports/basketevent_h1_test/test_predictions_seed_mean.csv")
SUMMARY = Path("reports/basketevent_h1_test/summary.json")
MODELS = ["constant", "logistic_regression", "xgboost", "xgboost_boxes"]
PLOTTED = {
    "xgboost": ("XGBoost baseline", "#2a78d6"),
    "xgboost_boxes": ("XGBoost + boxes", "#eb6834"),
}
BINS = 10
THRESHOLD = 0.5


def reliability(rows: pd.DataFrame, bins: int = BINS) -> list[dict]:
    """Equal-count bins of predicted probability; Wilson 95% interval on the observed rate."""
    order = rows.sort_values("p_make", kind="mergesort").reset_index(drop=True)
    out = []
    edges = np.linspace(0, len(order), bins + 1).round().astype(int)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]), 1):
        part = order.iloc[lo:hi]
        n, made = len(part), float(part.y_true.mean())
        z = 1.96
        centre = (made + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * np.sqrt(made * (1 - made) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
        out.append(
            {
                "bin": k,
                "shots": n,
                "predicted": float(part.p_make.mean()),
                "made": made,
                "made_low": float(centre - half),
                "made_high": float(centre + half),
            }
        )
    return out


def main(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(TEST, dtype={"game_id": str, "shot_id": str})
    scores = {s["model"]: s for s in score_predictions(preds)}
    saved = json.loads(SUMMARY.read_text())["test"]["scores_seed_mean"]
    for model in MODELS:  # the recomputed scores must match the ones already reported
        for ours, theirs in [
            ("brier_score", "brier"),
            ("log_loss", "log_loss"),
            ("roc_auc", "roc_auc"),
        ]:
            assert abs(scores[model][ours] - saved[model][theirs]) < 1e-9, (model, ours)
    table = []
    for model in MODELS:
        rows = preds[preds.model.eq(model)]
        table.append(
            {
                "model": model,
                "shots": len(rows),
                "brier": scores[model]["brier_score"],
                "log_loss": scores[model]["log_loss"],
                "roc_auc": scores[model]["roc_auc"],
                "accuracy": scores[model]["accuracy"],
                "predicted_makes": int((rows.p_make >= THRESHOLD).sum()),
                "max_p_make": float(rows.p_make.max()),
            }
        )
    curves = {m: reliability(preds[preds.model.eq(m)]) for m in PLOTTED}
    made_rate = float(preds[preds.model.eq("xgboost")].y_true.mean())
    summary = {
        "source": str(TEST).replace("\\", "/"),
        "threshold": THRESHOLD,
        "test_make_rate": made_rate,
        "accuracy_if_every_shot_called_a_miss": 1 - made_rate,
        "scores": table,
        "reliability_bins": curves,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame(table).to_csv(out / "accuracy.csv", index=False)

    os.environ.setdefault("MPLCONFIGDIR", str((out / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot([0.1, 0.6], [0.1, 0.6], "--", color="#8a8984", lw=1.2, label="Perfect calibration")
    for i, (model, (label, colour)) in enumerate(PLOTTED.items()):
        c = pd.DataFrame(curves[model])
        shift = (i - 0.5) * 0.003  # keep the two sets of error bars from overlapping exactly
        ax.errorbar(
            c.predicted + shift,
            c.made,
            yerr=[c.made - c.made_low, c.made_high - c.made],
            color=colour,
            lw=2,
            marker="o",
            ms=7,
            mec="white",
            mew=1.5,
            capsize=0,
            elinewidth=1,
            label=label,
        )
    ax.set(
        xlabel="Predicted make probability (bin average)",
        ylabel="Observed make rate",
        xlim=(0.1, 0.6),
        ylim=(0.1, 0.6),
        title="H1 test split: reliability (1,546 shots, 10 equal-count bins)",
    )
    ax.title.set_fontsize(10)
    ax.grid(color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.99,
        0.01,
        "Bars: 95% interval on the observed rate",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#52514e",
    )
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
