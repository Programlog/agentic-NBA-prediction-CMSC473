"""Figures for H2 experiment 06 (shooter history).

Usage: uv run python experiments/h2/06_shooter_history/make_figures.py [REPORT_DIR]
Reads REPORT_DIR/summary.json (default results/ next to this script) and, for the
contamination plot, artifacts/h2_history/contamination.csv if present.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
OUT = REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((REPORT / "summary.json").read_text())
CONTAM = Path("artifacts/h2_history/contamination.csv")

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0"
ARMS = [
    ("nohist", "No history (05)", "#b9b8b2"),
    ("hist", "+ history rates", "#2a78d6"),
    ("named", "+ history + name", "#1baf7a"),
]
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "DejaVu Sans"],
        "font.size": 13,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": SURFACE,
    }
)


def bars():
    splits = [s for s in ["validation", "test"] if s in S]
    fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 5.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        m = S[split]["metrics"]
        x = np.arange(2)
        width = 0.26
        for i, (arm, label, colour) in enumerate(ARMS):
            vals = [m[f"{arm}/{c}"]["brier"] for c in ["multi", "single"]]
            bs = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=colour, label=label)
            for b, v in zip(bs, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.0005,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color=INK2,
                )
        for name, label, style in [
            ("xgboost", "XGBoost", "-"),
            ("xgboost_hist", "XGBoost + history", "--"),
        ]:
            y = m[name]["brier"]
            other = m["xgboost_hist" if name == "xgboost" else "xgboost"]["brier"]
            ax.hlines(y, -0.5, 1.5, color=INK, lw=1.2, ls=style, zorder=0)
            ax.text(
                1.55,
                y,
                f" {label} {y:.3f}",
                va="top" if y <= other else "bottom",
                fontsize=10,
                color=INK,
            )
        vals = [v["brier"] for k, v in m.items() if "/" in k] + [m["xgboost"]["brier"]]
        ax.set_ylim(min(vals) - 0.008, max(vals) + 0.008)
        ax.set_xlim(-0.55, 2.3)
        ax.set_xticks(x, ["Multi-agent\n(anchored)", "Single LLM\n(anchored)"])
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.set_title(f"{split.title()} ({S[split]['n']} shots)", loc="left", fontsize=14)
    axes[0].set_ylabel("Brier score (lower is better)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=10.5)
    fig.suptitle(
        "GPT-6 Luna with and without shooter history", x=0.01, ha="left", fontsize=15, color=INK
    )
    fig.tight_layout()
    fig.savefig(OUT / "h2d_1_brier.png", dpi=160)
    plt.close(fig)


LABEL = {
    "hist/multi": "multi + history",
    "nohist/multi": "multi",
    "named/multi": "multi + name",
    "hist/single": "single + history",
    "nohist/single": "single",
    "named/single": "single + name",
    "xgboost_hist": "XGBoost + history",
    "xgboost": "XGBoost",
    "logistic_regression_hist": "LR + history",
    "logistic_regression": "LR",
}


def effects(split="test"):
    if split not in S:
        return
    rows = S[split]["comparisons"]
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        lo, hi = r["ci95"]
        c = "#2a78d6" if hi < 0 else ("#eb6834" if lo > 0 else MUTED)
        ax.plot([lo, hi], [yi, yi], color=c, lw=3, solid_capstyle="round")
        ax.plot(r["delta_brier"], yi, "o", color=c, ms=8)
        ax.text(
            max(hi, 0) + 0.002,
            yi,
            f"{r['delta_brier']:+.4f} [{lo:+.4f}, {hi:+.4f}]",
            va="center",
            fontsize=10,
            color=INK2,
        )
    names = [
        f"{'★ ' if r['role'] == 'primary' else ''}{LABEL[r['a']]} − {LABEL[r['b']]}" for r in rows
    ]
    ax.set_yticks(y, names)
    ax.axvline(0, color=INK, lw=1)
    lo = min(r["ci95"][0] for r in rows)
    hi = max(r["ci95"][1] for r in rows)
    ax.set_xlim(lo - 0.005, max(hi, 0) + 0.035)
    ax.set_xlabel("Δ Brier, first minus second (negative = first is better)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(
        f"{split.title()}: paired differences, 95% game-bootstrap intervals (★ = primary)",
        loc="left",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "h2d_2_effects.png", dpi=160)
    plt.close(fig)


def contamination():
    if not CONTAM.exists():
        return
    c = pd.read_csv(CONTAM)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.scatter(
        c["data_fg"],
        c["llm_fg"],
        s=28,
        color="#1baf7a",
        alpha=0.8,
        label="GPT-6 Luna's recalled 2014–15 FG%",
    )
    ax.scatter(
        c["data_fg"],
        c["player_prior_make_rate"],
        s=18,
        color="#2a78d6",
        alpha=0.6,
        marker="x",
        label="History rate given to the agents",
    )
    lims = [0.3, 0.75]
    ax.plot(lims, lims, color=INK, lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("FG% in this dataset (2014–15)")
    ax.set_ylabel("Estimate")
    ax.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=10.5)
    ax.set_title(f"The model remembers season FG% ({len(c)} players)", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "h2d_3_contamination.png", dpi=160)
    plt.close(fig)


bars()
effects()
contamination()
print("wrote", sorted(p.name for p in OUT.glob("h2d_*.png")))
