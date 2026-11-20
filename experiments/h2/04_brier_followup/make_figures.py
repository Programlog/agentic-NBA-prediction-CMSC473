"""Figures for the H2 Brier follow-up (validation anchoring + test recalibration).

Usage: uv run python experiments/h2/04_brier_followup/make_figures.py [REPORT_DIR]
Default REPORT_DIR: results/ next to this script (reads summary.json,
writes REPORT_DIR/figures/*.png). Styling matches ../02_final_test_100/make_figures.py.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
OUT = REPORT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((REPORT / "summary.json").read_text())

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8f8e88", "#e6e5e0"
MULTI, SINGLE, STAT = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
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
    "multi": "Multi-agent (frozen prompts)",
    "multi_anchored": "Multi-agent, anchored",
    "multi_prior": "Multi-agent, anchored + LR prior",
    "multi+platt_cv": "Multi-agent + recalibration",
    "single_budget": "Single LLM (frozen prompts)",
    "single_budget+platt_cv": "Single LLM + recalibration",
    "llm_multi": "Multi-agent LLM",
    "llm_single_budget": "Single LLM (budget-matched)",
}


def color(key):
    return SINGLE if "single" in key else MULTI if "multi" in key else STAT


def legend(ax, loc):
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (MULTI, SINGLE, STAT)]
    ax.legend(
        handles,
        ["Multi-agent LLM", "Single LLM", "Statistical model"],
        frameon=False,
        loc=loc,
        fontsize=11,
    )


def hbar(keys, values, title, xlabel, xlim, fname, ref=None, fmt="{:.3f}", loc="upper right"):
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(keys) + 1.6))
    y = range(len(keys))[::-1]
    ax.barh(
        list(y),
        [values[k] - xlim[0] for k in keys],
        left=xlim[0],
        height=0.55,
        color=[color(k) for k in keys],
    )
    for yi, k in zip(y, keys):
        ax.text(
            values[k] + (xlim[1] - xlim[0]) * 0.008,
            yi,
            fmt.format(values[k]),
            va="center",
            fontsize=11,
            color=INK2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
        )
    if ref is not None:
        ax.axvline(ref[0], color=MUTED, linestyle="--", linewidth=1)
        ax.text(ref[0], len(keys) - 0.4, ref[1], color=INK2, fontsize=11, ha="center")
    ax.set_yticks(list(y), [NAMES[k] for k in keys])
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=15, color=INK, pad=14)
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    legend(ax, loc)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=160)
    plt.close(fig)


val = S["validation"]
keys = [
    "xgboost",
    "multi_anchored",
    "multi_prior",
    "logistic_regression",
    "constant",
    "single_budget+platt_cv",
    "multi+platt_cv",
    "multi",
    "single_budget",
]
hbar(
    keys,
    {k: val[k]["brier"] for k in keys},
    "Validation Brier score (100 shots, lower is better)",
    "Brier score",
    (0.20, 0.29),
    "h2b_1_validation_brier.png",
    ref=(val["constant"]["brier"], "constant"),
)

keys = ["single_budget", "multi", "multi_prior", "xgboost", "logistic_regression", "multi_anchored"]
hbar(
    keys,
    {k: val[k]["mean_p"] for k in keys},
    "Average predicted make probability vs. actual 44% (validation)",
    "Average predicted p_make",
    (0.35, 0.65),
    "h2b_2_overconfidence.png",
    ref=(0.44, "actual 44%"),
    fmt="{:.1%}",
    loc="lower right",
)

# Test: before/after validation-fitted recalibration.
test = S["test"]
fig, ax = plt.subplots(figsize=(11, 3.6))
for i, (raw, c) in enumerate([("llm_multi", MULTI), ("llm_single_budget", SINGLE)]):
    a, b = test[raw]["brier"], test[raw + "+platt"]["brier"]
    y = 1 - i
    ax.plot([a, b], [y, y], color=c, alpha=0.5, linewidth=2)
    ax.scatter([a], [y], s=70, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=3)
    ax.scatter([b], [y], s=90, color=c, zorder=3)
    ax.text(a + 0.0012, y + 0.12, f"{a:.3f} published", fontsize=11, color=INK2)
    ax.text(b, y - 0.3, f"{b:.3f} recalibrated", fontsize=11, color=INK2, ha="left")
for k, label in [("xgboost", "XGBoost"), ("logistic_regression", "LR"), ("constant", "constant")]:
    ax.axvline(test[k]["brier"], color=STAT, linestyle="--", linewidth=1.2)
    ax.text(
        test[k]["brier"],
        1.55,
        f" {label} {test[k]['brier']:.3f}",
        fontsize=10.5,
        color=INK2,
        ha="left",
    )
ax.set_yticks([1, 0], [NAMES["llm_multi"], NAMES["llm_single_budget"]])
ax.set_ylim(-0.6, 1.8)
ax.set_xlim(0.20, 0.28)
ax.set_xlabel("Test Brier score (frozen predictions; calibrator fitted on validation only)")
ax.set_title(
    "Recalibrating both arms on test: multi-agent and single LLM tie",
    loc="left",
    fontsize=15,
    color=INK,
    pad=14,
)
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)
fig.tight_layout()
fig.savefig(OUT / "h2b_3_test_recalibration.png", dpi=160)
plt.close(fig)
print(f"Wrote figures to {OUT}")
