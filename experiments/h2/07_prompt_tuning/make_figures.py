"""Figures for H2 experiment 07.

Usage: uv run --frozen python experiments/h2/07_prompt_tuning/make_figures.py [REPORT_DIR]
Reads REPORT_DIR/summary.json and REPORT_DIR/test_wide.csv (default results/ here).
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
W = S["winner"]

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0"
BLUE, GREEN, ORANGE, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#b9b8b2"
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
LABEL = {
    "V0": "V0  original prompt + anchors",
    "V1": "V1  rewritten prompt + rate table",
    "V2": "V2  V1 + similar shots",
    "V3": "V3  V1 + XGBoost tool",
    "V4": "V4  V1 + similar shots + XGBoost",
    f"multi:{W}": f"Multi-agent ({W} evidence)",
    f"single_budget:{W}": f"Single LLM, same calls ({W})",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic regression",
    "knn50": "kNN-50 (similar shots alone)",
    "constant": "Constant",
}


def colour(name):
    if name in {"xgboost", "logistic_regression", "knn50", "constant"}:
        return GREY
    if name.startswith("multi"):
        return GREEN
    if name.startswith("single"):
        return ORANGE
    return BLUE


def bars(split, fname, title):
    if split not in S:
        return
    m = S[split]["metrics"]
    names = sorted((c for c in m if c in LABEL and c != "constant"), key=lambda c: m[c]["brier"])
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(names) + 1.6))
    y = np.arange(len(names))[::-1]
    vals = [m[c]["brier"] for c in names]
    ax.barh(y, vals, color=[colour(c) for c in names], height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + 0.0002, yi, f"{v:.4f}", va="center", fontsize=10.5, color=INK2)
    ax.axvline(m["xgboost"]["brier"], color=INK, lw=1, ls="--")
    ax.set_yticks(y, [LABEL[c] for c in names])
    ax.set_xlim(min(vals) - 0.004, max(vals) + 0.003)
    ax.set_xlabel("Brier score (lower is better); dashed line = XGBoost")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(f"{title} ({S[split]['n']} shots)", loc="left", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def effects(split="test"):
    if split not in S:
        return
    rows = S[split]["comparisons"]
    fig, ax = plt.subplots(figsize=(11.5, 0.62 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        lo, hi = r["ci95"]
        c = BLUE if hi < 0 else (ORANGE if lo > 0 else MUTED)
        ax.plot([lo, hi], [yi, yi], color=c, lw=3, solid_capstyle="round")
        ax.plot(r["delta_brier"], yi, "o", color=c, ms=8)
        ax.text(
            max(hi, 0) + 0.0008,
            yi,
            f"{r['delta_brier']:+.4f} [{lo:+.4f}, {hi:+.4f}]",
            va="center",
            fontsize=10,
            color=INK2,
        )
    short = {k: v.split("  ")[0] if k.startswith("V") else v for k, v in LABEL.items()}
    ax.set_yticks(
        y,
        [
            f"{'★ ' if r['role'] == 'primary' else ''}{short[r['a']]} − {short[r['b']]}"
            for r in rows
        ],
    )
    ax.axvline(0, color=INK, lw=1)
    lo = min(r["ci95"][0] for r in rows)
    hi = max(r["ci95"][1] for r in rows)
    ax.set_xlim(lo - 0.002, max(hi, 0) + 0.012)
    ax.set_xlabel("Δ Brier, first minus second (negative = first is better)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Test (n={S[split]['n']}): paired differences, 95% game-bootstrap CIs (★ = primary)",
        loc="left",
        fontsize=14,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "h2e_3_test_effects.png", dpi=160)
    plt.close(fig)


def scatter():
    path = REPORT / "test_wide.csv"
    if not path.exists():
        return
    t = pd.read_csv(path)
    panels = [c for c in ["V0", S["nomodel"], f"multi:{W}"] if c in t]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), panels):
        ax.scatter(t.xgboost, t[c], s=9, alpha=0.45, color=colour(c))
        ax.plot([0, 1], [0, 1], color=INK, lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        r = np.corrcoef(t.xgboost, t[c])[0, 1]
        ax.set_title(f"{LABEL[c]}\nr = {r:.2f} with XGBoost", loc="left", fontsize=12)
        ax.set_xlabel("XGBoost p_make")
        ax.grid(True, color=GRID)
        ax.set_axisbelow(True)
    np.atleast_1d(axes)[0].set_ylabel("Agent p_make")
    fig.tight_layout()
    fig.savefig(OUT / "h2e_4_test_vs_xgboost.png", dpi=160)
    plt.close(fig)


bars("validation", "h2e_1_validation_brier.png", "Validation: prompt variants and orchestration")
bars("test", "h2e_2_test_brier.png", "Test")
effects()
scatter()
print("wrote", sorted(p.name for p in OUT.glob("h2e_*.png")))
