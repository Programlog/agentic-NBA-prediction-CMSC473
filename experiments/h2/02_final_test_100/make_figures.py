"""Slide figures for the final H2 result (light theme, 16:9 PNG).

Usage: uv run python experiments/h2/02_final_test_100/make_figures.py [REPORT_DIR] [OUTPUT_DIR]
Defaults: experiments/h2/02_final_test_100/report and REPORT_DIR/figures.
"""

import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "report"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8f8e88"
GRID = "#e6e5e0"
LLM = "#2a78d6"  # categorical slot 1
TAB = "#eb6834"  # categorical slot 2

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "DejaVu Sans"],
        "font.size": 13,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "savefig.facecolor": SURFACE,
    }
)

NAMES = {
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic regression",
    "constant": "Always guess average",
    "llm_multi": "Multi-agent LLM (3 calls)",
    "llm_single_budget": "Single LLM (3 calls)",
    "llm_single": "Single LLM (1 call)",
}
IS_LLM = {k: k.startswith("llm") for k in NAMES}

preds = pd.read_csv(REPORT / "predictions.csv")
metrics = pd.read_csv(REPORT / "combined_metrics.csv")
paired = json.loads((REPORT / "paired_comparisons.json").read_text())


def title(fig, main, sub):
    fig.text(0.04, 0.95, main, fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=13, color=INK2, va="top")


def legend_handles():
    from matplotlib.patches import Patch

    return [Patch(color=LLM, label="LLM (Qwen3 4B)"), Patch(color=TAB, label="Tabular baseline")]


# ---- 1. Brier score, all models -------------------------------------------
m = metrics.set_index("model").loc[list(NAMES)].reset_index()
m = m.sort_values("brier_score", ascending=False)
fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=0.25, right=0.93, top=0.76, bottom=0.12)
y = range(len(m))
ax.barh(
    y,
    m["brier_score"],
    height=0.62,
    color=[LLM if IS_LLM[k] else TAB for k in m["model"]],
    edgecolor=SURFACE,
    linewidth=2,
)
ax.set_yticks(list(y), [NAMES[k] for k in m["model"]])
for yi, (k, b) in enumerate(zip(m["model"], m["brier_score"])):
    ax.text(
        b + 0.004,
        yi,
        f"{b:.3f}",
        va="center",
        color=INK,
        fontsize=13,
        fontweight="bold" if k == "llm_multi" else "normal",
    )
ax.set_xlim(0, 0.34)
ax.set_xlabel("Brier score (lower is better)")
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.legend(handles=legend_handles(), loc="upper right", frameon=False, fontsize=12)
title(
    fig,
    "Multi-agent was the best LLM setup, but no LLM beat the baselines",
    "Brier score on the same 100 held-out test shots (50 games, 2014–15 Kaggle data)",
)
fig.savefig(OUT / "h2_1_brier_all_models.png", dpi=150)
plt.close(fig)

# ---- 2. H2 effect with confidence intervals --------------------------------
rows = [
    (
        "Final test: vs. single LLM, 3 calls\n(primary H2 test, 100 shots)",
        next(p for p in paired if p["baseline"] == "llm_single_budget"),
    ),
    (
        "Final test: vs. single LLM, 1 call\n(secondary, 100 shots)",
        next(p for p in paired if p["baseline"] == "llm_single"),
    ),
]
effects = [
    (lbl, p["brier_delta_candidate_minus_baseline"], *p["brier_delta_95pct_interval"], LLM)
    for lbl, p in rows
]
# Validation pilot (30 shots / 10 games), from experiments/h2/01_validation_pilot_30/README.md
effects.append(
    ("Pilot: vs. single LLM, 3 calls\n(validation, 30 shots)", 0.01404, -0.00591, 0.03358, MUTED)
)

fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=0.30, right=0.95, top=0.74, bottom=0.17)
for i, (lbl, d, lo, hi, c) in enumerate(reversed(effects)):
    ax.plot([lo, hi], [i, i], color=c, linewidth=3, solid_capstyle="round")
    ax.plot(d, i, "o", color=c, markersize=12, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(
        d,
        i + 0.24,
        f"{d:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
        ha="center",
        color=INK,
        fontsize=12,
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2),
        zorder=5,
    )
ax.set_yticks(range(len(effects)), [e[0] for e in reversed(effects)])
ax.axvline(0, color=INK2, linewidth=1.2)
ax.set_xlim(-0.085, 0.05)
ax.set_ylim(-0.6, len(effects) - 0.4)
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_xlabel(
    "Brier difference, multi-agent minus single agent (dot = estimate, line = 95% interval)"
)
ax.text(-0.004, -0.55, "← multi-agent better", ha="right", va="bottom", color=INK2, fontsize=12)
ax.text(0.004, -0.55, "multi-agent worse →", ha="left", va="bottom", color=INK2, fontsize=12)
title(
    fig,
    "H2 supported: multi-agent beat the matched single agent on the test set",
    "Whole-game bootstrap, 1,000 resamples. Interval entirely below 0 = significant improvement.",
)
fig.savefig(OUT / "h2_2_effect_intervals.png", dpi=150)
plt.close(fig)

# ---- 3. Overconfidence: mean predicted vs actual make rate ------------------
mean_p = preds.groupby("model")["p_make"].mean().loc[list(NAMES)].sort_values()
actual = preds.groupby("shot_id")["y_true"].first().mean()
fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=0.25, right=0.93, top=0.76, bottom=0.12)
for i, (k, v) in enumerate(mean_p.items()):
    c = LLM if IS_LLM[k] else TAB
    ax.plot([actual, v], [i, i], color=c, linewidth=2, alpha=0.5)
    ax.plot(v, i, "o", color=c, markersize=12, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.text(v + 0.008, i, f"{v:.0%}", va="center", color=INK, fontsize=13)
ax.axvline(actual, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(
    actual,
    len(mean_p) - 0.35,
    f"Actual make rate {actual:.0%}",
    ha="center",
    color=INK,
    fontsize=12,
    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2),
)
ax.set_yticks(range(len(mean_p)), [NAMES[k] for k in mean_p.index])
ax.set_xlim(0.35, 0.72)
ax.set_ylim(-0.6, len(mean_p) - 0.1)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.xaxis.grid(True, color=GRID, linewidth=1)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Average predicted chance the shot goes in")
ax.legend(handles=legend_handles(), loc="lower right", frameon=False, fontsize=12)
title(
    fig,
    "Every LLM over-predicted makes; multi-agent did so the least",
    "Average predicted make probability vs. how often the 100 test shots actually went in",
)
fig.savefig(OUT / "h2_3_overconfidence.png", dpi=150)
plt.close(fig)

# ---- 4. Experiment design diagram -------------------------------------------
fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=0.02, right=0.98, top=0.80, bottom=0.04)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")


def box(x, y, w, h, text, edge=LLM, fill="#eaf2fc", size=12, bold=False):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.5,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight="bold" if bold else "normal",
    )


def arrow(x1, y1, x2, y2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=INK2, linewidth=1.3, mutation_scale=14),
    )


box(
    1,
    22,
    17,
    16,
    "Shot stats\n(distance, defender,\nshot clock, game\nsituation)",
    edge=MUTED,
    fill="#f0efec",
)
rows_y = [47, 27, 5]
labels = ["Single LLM, 1 call", "Single LLM, 3 calls\n(matched control)", "Multi-agent, 3 calls"]
for ty in [51, 31, 13.1, 4.9]:
    arrow(18, 30, 25, ty)
box(25, 47, 40, 8, "One prompt → probability")
box(25, 27, 40, 8, "Same analyst reasons in 3 steps → probability")
box(25, 9.8, 19, 6.7, "Game context analyst", size=11)
box(25, 1.5, 19, 6.7, "Shot & defense analyst", size=11)
box(50, 1.5, 15, 15, "Aggregator\ncombines both\n→ probability", size=11)
arrow(44, 13.1, 50, 10.5)
arrow(44, 4.9, 50, 7.5)
for yy, lbl in zip(rows_y, labels):
    ax.text(25, yy + (9.8 if yy != 5 else 13.8), lbl, fontsize=12, color=INK2, fontweight="bold")
for yy in [51, 31, 9]:
    arrow(65, yy, 72, 30)
box(
    72,
    22,
    26,
    16,
    "Score each on the same\n100 test shots\n(Brier score + whole-game\nbootstrap 95% interval)",
    edge=TAB,
    fill="#fdeee7",
)
title(
    fig,
    "H2 design: same model, same inputs, same call budget",
    "Only how the calls are organized changes. All three use Qwen3 4B locally via Ollama.",
)
fig.savefig(OUT / "h2_0_design.png", dpi=150)
plt.close(fig)

print("wrote", sorted(p.name for p in OUT.glob("*.png")))
