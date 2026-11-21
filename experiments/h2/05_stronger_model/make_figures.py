"""Figures for H2 experiment 05 (GPT-6 Luna vs. qwen).

Usage: uv run python experiments/h2/05_stronger_model/make_figures.py [REPORT_DIR]
Default REPORT_DIR: results/ next to this script (reads summary.json, writes
REPORT_DIR/figures/*.png). Styling matches experiments 02 and 04.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
OUT = REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((REPORT / "summary.json").read_text())

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0"
QWEN, LUNA, LUNA_LOW = "#8f8e88", "#2a78d6", "#7fb0ea"
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
MODELS = [
    ("qwen", "qwen3 4B (local)", QWEN),
    ("luna_none", "GPT-6 Luna, effort none", LUNA),
    ("luna_low", "GPT-6 Luna, effort low", LUNA_LOW),
]
CONDS = [
    ("multi", "Multi-agent"),
    ("single_budget", "Single LLM\n(same 3 calls)"),
    ("multi_anchored", "Multi-agent,\nanchored"),
    ("single_budget_anchored", "Single LLM,\nanchored"),
]


def grouped(split, key, fname, title, suffix=""):
    table = S[split][key]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = np.arange(len(CONDS))
    width = 0.26
    for i, (m, label, colour) in enumerate(MODELS):
        vals = [table.get(f"{m}/{c}{suffix}", {}).get("brier", np.nan) for c, _ in CONDS]
        bars = ax.bar(x + (i - 1) * width, vals, width * 0.92, color=colour, label=label)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    v + 0.0008,
                    f"{v:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9.5,
                    color=INK2,
                )
    raw = S[split]["raw"]
    for name, label, style in [
        ("xgboost", "XGBoost", "-"),
        ("constant", "Always guess average", "--"),
    ]:
        y = raw[name]["brier"]
        ax.hlines(y, -0.6, len(CONDS) - 0.5, color=INK, lw=1.2, ls=style, zorder=0)
        ax.text(len(CONDS) - 0.45, y, f"{label}\n{y:.3f}", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(x, [c for _, c in CONDS])
    ax.set_ylabel("Brier score (lower is better)")
    lo = min(raw["xgboost"]["brier"], *[v["brier"] for v in table.values()]) - 0.01
    ax.set_ylim(lo, max(v["brier"] for v in table.values()) + 0.014)
    ax.set_xlim(-0.6, len(CONDS) + 0.4)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", fontsize=11, ncol=3)
    ax.set_title(title, loc="left", fontsize=15, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


def overconfidence():
    raw = S["test"]["raw"]
    rate = S["test"]["make_rate"]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    x = np.arange(len(CONDS))
    width = 0.26
    for i, (m, label, colour) in enumerate(MODELS):
        vals = [raw.get(f"{m}/{c}", {}).get("mean_p", np.nan) for c, _ in CONDS]
        ax.bar(x + (i - 1) * width, vals, width * 0.92, color=colour, label=label)
    ax.hlines(rate, -0.6, len(CONDS) - 0.5, color=INK, lw=1.4)
    ax.text(len(CONDS) - 0.45, rate, f"actual\nmake rate {rate:.0%}", va="center", fontsize=10.5)
    ax.set_xticks(x, [c for _, c in CONDS])
    ax.set_ylabel("Average predicted make probability")
    ax.set_ylim(0, 0.72)
    ax.set_xlim(-0.6, len(CONDS) + 0.4)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper center", fontsize=11, ncol=3)
    ax.set_title("Test: how often each setup predicts a make", loc="left", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "h2c_3_test_overconfidence.png", dpi=160)
    plt.close(fig)


def effects():
    label = {
        "luna_none/multi_anchored": "Luna: anchored multi",
        "luna_none/single_budget_anchored": "anchored single",
        "luna_none/multi": "Luna multi",
        "luna_none/single_budget": "Luna single",
        "qwen/multi": "qwen multi",
        "luna_low/single_budget_anchored": "Best LLM (Luna low, anchored single)",
    }
    keep = [
        ("luna_none/multi_anchored", "luna_none/single_budget_anchored"),
        ("luna_none/multi", "luna_none/single_budget"),
        ("luna_none/multi", "qwen/multi"),
    ]
    rows = [r for r in S["test"]["comparisons_raw"] if (r["a"], r["b"]) in keep]
    rows += [r for r in S["test"]["comparisons_raw"] if r["b"] == "xgboost"]
    names = []
    for r in rows:
        a = label.get(r["a"], r["a"].replace("/", " "))
        b = "XGBoost" if r["b"] == "xgboost" else label.get(r["b"], r["b"])
        names.append(f"{a} − {b}")
    fig, ax = plt.subplots(figsize=(11, 0.9 * len(rows) + 1.6))
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        lo, hi = r["ci95"]
        c = LUNA if hi < 0 else ("#eb6834" if lo > 0 else MUTED)
        ax.plot([lo, hi], [yi, yi], color=c, lw=3, solid_capstyle="round")
        ax.plot(r["delta_brier"], yi, "o", color=c, ms=9)
        ax.text(
            max(hi, 0) + 0.002,
            yi,
            f"{r['delta_brier']:+.3f} [{lo:+.3f}, {hi:+.3f}]",
            va="center",
            fontsize=10.5,
            color=INK2,
        )
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(y, names)
    ax.set_xlabel("Δ Brier, first minus second (negative = first is better)")
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    xmin = min(r["ci95"][0] for r in rows) - 0.01
    xmax = max(r["ci95"][1] for r in rows) + 0.05
    ax.set_xlim(xmin, xmax)
    ax.set_title(
        "Test: paired differences with 95% game-bootstrap intervals",
        loc="left",
        fontsize=15,
        pad=12,
    )
    fig.tight_layout()
    fig.savefig(OUT / "h2c_4_test_effects.png", dpi=160)
    plt.close(fig)


grouped("validation", "raw", "h2c_1_validation_brier.png", "Validation: Brier by model and setup")
grouped("test", "raw", "h2c_2_test_brier.png", "Test: Brier by model and setup")
overconfidence()
effects()
print("wrote", sorted(p.name for p in OUT.glob("h2c_*.png")))
