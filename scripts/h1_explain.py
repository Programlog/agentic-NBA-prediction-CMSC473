"""What the frozen H1 box model learned (exploratory; presentation only, no new test).

    python scripts/h1_explain.py COHORT TEST_RUN OUT

Uses the six D0 box models saved by scripts/h1_test_run.py (fit on training games) and the
development shots only (train + validation games); the test split is not read. Writes:

- feature_importance.png: mean |SHAP| per feature (exact tree SHAP from XGBoost, log-odds units),
  averaged over the six seeds, one-hot and missing-indicator columns summed into their feature.
- release_height.png: observed make rate by release-height fifth (2s and 3s, game-bootstrap
  intervals), and the model's SHAP contribution of release height against its value.
- summary.json: the numbers behind both figures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_shots.common import BOX_FEATURES, load_config
from nba_shots.models import merge_extra, select_shots

SEEDS = [42, 1, 2, 3, 4, 5]
NAMES = {
    "shot_distance": "Shot distance",
    "game_clock_seconds": "Game clock",
    "period": "Period",
    "shot_value": "2 vs 3",
    "loc_x": "Court x",
    "loc_y": "Court y",
    "player_prior_attempts": "Shooter's prior attempts",
    "player_prior_make_rate": "Shooter's prior make rate",
    "location": "Home / away",
    "action_type": "Shot type",
    "shot_zone": "Zone",
    "shot_area": "Area",
    "release_height_bh": "Release height",
    "jump_bh": "Jump",
    "ball_rise_bhps": "Ball rise speed",
    "shooter_speed_bhps": "Shooter speed",
    "defender_distance_bh": "Defender distance",
    "defenders_within_1bh": "Defenders within 1 body height",
    "defender_closing_bhps": "Defender closing speed",
    "defender_in_front": "Defender on basket side",
    "defender_hand_up": "Defender arms up",
}
# Reference palette (dataviz skill): box features carry the accent, tabular features stay neutral.
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#a3a29c", "#0b0b0b", "#52514e"


def owner(column: str, features: list[str]) -> str:
    """Original feature behind a transformed column name."""
    name = column.split("__", 1)[1].removeprefix("missingindicator_")
    if name in features:
        return name
    return max((f for f in features if name.startswith(f + "_")), key=len)


def shap_by_feature(artifact: dict, rows: pd.DataFrame) -> pd.DataFrame:
    from xgboost import DMatrix

    pipeline, features = artifact["pipeline"], artifact["features"]
    X = pipeline["preprocess"].transform(rows[features])
    contribs = pipeline["model"].get_booster().predict(DMatrix(X), pred_contribs=True)[:, :-1]
    columns = pipeline["preprocess"].get_feature_names_out()
    table = pd.DataFrame(contribs, columns=[owner(c, features) for c in columns])
    return table.T.groupby(level=0).sum().T  # sum one-hot and indicator columns per feature


def game_bootstrap(part: pd.DataFrame, reps=1000, seed=42) -> list[float]:
    games = part.groupby("game_id")["y_true"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(games), (reps, len(games)))
    rates = games["sum"].to_numpy()[idx].sum(1) / games["count"].to_numpy()[idx].sum(1)
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GREY)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x" if ax.get_ylabel() == "" else "y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def main(cohort: Path, runs: Path, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/basketevent_h1_final.json")
    shots, _ = merge_extra(
        pd.read_parquet(cohort / "shots.parquet"), cohort / "box_features.parquet"
    )
    dev = select_shots(shots, config)
    dev = dev.loc[dev["split"].isin(["train", "validation"])].reset_index(drop=True)

    tables = [
        shap_by_feature(joblib.load(runs / f"seed_{s}" / "xgboost_boxes.joblib"), dev)
        for s in SEEDS
    ]
    shap = sum(tables) / len(tables)
    importance = shap.abs().mean().sort_values()
    box_share = float(importance[list(BOX_FEATURES)].sum() / importance.sum())

    # Figure 1: mean |SHAP| per feature.
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=200)
    colours = [BLUE if f in BOX_FEATURES else GREY for f in importance.index]
    ax.barh([NAMES[f] for f in importance.index], importance.values, color=colours, height=0.62)
    for y, v in enumerate(importance.values):
        ax.text(v + importance.max() * 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlabel(
        "Mean |SHAP|: average push on a shot's log-odds of a make", color=MUTED, fontsize=9
    )
    ax.set_ylabel("")
    style(ax)
    ax.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color=BLUE, label="Box feature"),
            Patch(color=GREY, label="Tabular feature"),
        ],
        frameon=False,
        fontsize=9,
        loc="lower right",
    )
    ax.set_title("What the H1 box model relies on", loc="left", fontsize=12, color=INK)
    fig.text(
        0.01,
        0.005,
        f"XGBoost D0 + boxes, 6 seeds averaged; {len(dev):,} development shots "
        f"({dev.game_id.nunique()} games). Box features: {box_share:.0%} of the total.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / "feature_importance.png")
    plt.close(fig)

    # Figure 2: release height, observed and learned.
    dev["fifth"] = pd.qcut(dev["release_height_bh"], 5, labels=False) + 1
    edges = dev.groupby("fifth")["release_height_bh"].agg(["min", "max"])
    rates = []
    for value, label in [(2, "2-pointers"), (3, "3-pointers")]:
        for fifth, part in dev.loc[dev.shot_value.eq(value)].groupby("fifth"):
            rates.append(
                {
                    "shots": label,
                    "fifth": int(fifth),
                    "n": len(part),
                    "make_rate": float(part.y_true.mean()),
                    "ci": game_bootstrap(part),
                }
            )
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=200)
    for (label, colour), offset in zip(
        [("2-pointers", BLUE), ("3-pointers", ORANGE)], (-0.06, 0.06)
    ):
        part = [r for r in rates if r["shots"] == label]
        x = np.array([r["fifth"] for r in part]) + offset
        y = np.array([r["make_rate"] for r in part])
        lo, hi = np.array([r["ci"] for r in part]).T
        left.errorbar(
            x,
            y,
            yerr=[y - lo, hi - y],
            color=colour,
            marker="o",
            markersize=6,
            linewidth=2,
            capsize=0,
            elinewidth=1.2,
            label=label,
        )
        left.annotate(
            label,
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )
    left.set_xticks(range(1, 6), ["Lowest", "2nd", "3rd", "4th", "Highest"])
    left.set_xlim(0.6, 6.0)
    left.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    left.set_ylabel("Made", color=MUTED, fontsize=9)
    left.set_xlabel("Release height, fifths of development shots", color=MUTED, fontsize=9)
    left.set_title("Low releases go in less", loc="left", fontsize=11, color=INK)
    style(left)

    x = dev["release_height_bh"].to_numpy()
    s = shap["release_height_bh"].to_numpy()
    keep = (x >= np.nanpercentile(x, 1)) & (x <= np.nanpercentile(x, 99))
    right.scatter(x[keep], s[keep], s=4, color=BLUE, alpha=0.18, linewidths=0)
    # Fixed-width bins, so the model's steps are not smoothed over where shots are sparse.
    frame = pd.DataFrame({"x": x[keep], "s": s[keep]})
    frame["b"] = (frame.x / 0.05).round()
    binned = frame.groupby("b").agg(x=("x", "mean"), s=("s", "mean"), n=("s", "size"))
    binned = binned.loc[binned.n >= 20]
    right.plot(binned.x, binned.s, color=INK, linewidth=2, label="Average")
    right.axhline(0, color=MUTED, linewidth=1, linestyle="--")
    right.set_xlabel("Release height (shooter body heights)", color=MUTED, fontsize=9)
    right.set_ylabel("Effect on log-odds of a make", color=MUTED, fontsize=9)
    right.set_title("What the model learned from it", loc="left", fontsize=11, color=INK)
    right.annotate(
        "each dot is a shot; line = average per 0.05",
        (0.98, 0.04),
        xycoords="axes fraction",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    style(right)
    fig.text(
        0.01,
        0.005,
        "Development shots only (train + validation games); bars are 95% "
        "game-bootstrap intervals. SHAP from 6 seeds averaged; middle 98% of values shown.",
        fontsize=7.5,
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "release_height.png")
    plt.close(fig)

    below = dev.release_height_bh < 0.9
    low = {
        "share": float(below.mean()),
        "make_rate": float(dev.y_true[below].mean()),
        "make_rate_rest": float(dev.y_true[~below].mean()),
        "three_share": float(dev.shot_value[below].eq(3).mean()),
        "three_share_rest": float(dev.shot_value[~below].eq(3).mean()),
    }
    summary = {
        "note": "Exploratory; development shots only; no test-split data read.",
        "models": str(runs),
        "seeds": SEEDS,
        "shots": len(dev),
        "games": int(dev.game_id.nunique()),
        "mean_abs_shap": {f: float(v) for f, v in importance.sort_values(ascending=False).items()},
        "box_share_of_mean_abs_shap": box_share,
        "release_height_fifths": {int(k): [float(a), float(b)] for k, (a, b) in edges.iterrows()},
        "make_rate_by_fifth": rates,
        "release_height_shap_binned": binned[["x", "s", "n"]].round(4).values.tolist(),
        "below_0_9_bh": low,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("shots", "games", "box_share_of_mean_abs_shap")}))
    for r in rates:
        print(
            r["shots"], r["fifth"], r["n"], f"{r['make_rate']:.3f}", [round(c, 3) for c in r["ci"]]
        )
    print("below 0.9 bh", low)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
