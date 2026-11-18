
"""Strictly matched H2 evaluation and shot-level failure analysis.

Usage:
    uv run python scripts/h2_matched_analysis.py \
        --xgb PATH --llm PATH --selection PATH --output PATH

Requires matching validation shots across XGBoost and LLM predictions.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

KEYS = ["cohort", "split", "shot_id"]
IDENTITY = KEYS + ["game_id", "y_true"]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype={
            "shot_id": str,
            "game_id": str,
            "cohort": str,
            "split": str,
        },
    )

    required = set(IDENTITY + ["model", "p_make"])
    if missing := required - set(df):
        raise ValueError(f"{path}: missing columns: {sorted(missing)}")

    if df.empty or df.duplicated(KEYS + ["model"]).any():
        raise ValueError(f"{path}: empty input or duplicate model/shot rows")

    if "status" not in df:
        df["status"] = "ok"

    if not df.status.isin(["ok", "error"]).all():
        raise ValueError(f"{path}: invalid status")

    df["p_make"] = pd.to_numeric(df["p_make"], errors="coerce")

    ok = df.status.eq("ok")

    if (ok & (~np.isfinite(df.p_make) | ~df.p_make.between(0, 1))).any():
        raise ValueError(
            f"{path}: successful predictions need probabilities in [0,1]"
        )

    if ((~ok) & df.p_make.notna()).any():
        raise ValueError(
            f"{path}: failed predictions must not contain probabilities"
        )

    if (
        df[IDENTITY + ["model"]].isna().any().any()
        or not df.y_true.isin([0, 1]).all()
    ):
        raise ValueError(f"{path}: invalid identities or labels")

    return df


def matched_predictions(
    xgb: pd.DataFrame,
    llm: pd.DataFrame,
    selection: pd.DataFrame,
) -> pd.DataFrame:
    if missing := set(KEYS) - set(selection):
        raise ValueError(f"Selection missing {sorted(missing)}")

    selection = selection[KEYS].copy()

    if selection.empty or selection.duplicated(KEYS).any():
        raise ValueError("Selection must have unique shots")

    if selection.split.nunique() != 1 or selection.cohort.nunique() != 1:
        raise ValueError("Evaluate one cohort and split at a time")

    if selection.split.iloc[0] != "validation":
        raise ValueError(
            "Use validation only until the final test protocol is frozen"
        )

    if (xgb.model == "xgboost").sum() == 0:
        raise ValueError("XGBoost predictions not found")

    xgb = xgb[xgb.model.eq("xgboost")].copy()

    selected_ids = set(map(tuple, selection.to_numpy()))
    llm_ids = set(map(tuple, llm[KEYS].drop_duplicates().to_numpy()))

    if llm_ids != selected_ids:
        raise ValueError(
            "LLM shot IDs do not exactly match registered selection"
        )

    for model, rows in llm.groupby("model"):
        if set(map(tuple, rows[KEYS].to_numpy())) != selected_ids:
            raise ValueError(f"{model} has missing or extra shots")

    xgb = xgb.merge(
        selection,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    if set(map(tuple, xgb[KEYS].to_numpy())) != selected_ids:
        raise ValueError(
            "XGBoost predictions do not cover the registered selection"
        )

    merged = pd.concat([xgb, llm], ignore_index=True)

    if merged.groupby(KEYS)["game_id"].nunique().gt(1).any():
        raise ValueError("Models disagree about game IDs")

    if merged.groupby(KEYS)["y_true"].nunique().gt(1).any():
        raise ValueError("Models disagree about shot outcomes")

    return merged


def metrics(rows: pd.DataFrame) -> dict:
    ok = rows.loc[rows.status.eq("ok")]

    out = {
        "n_attempted": len(rows),
        "n_successful": len(ok),
        "coverage": len(ok) / len(rows),
    }

    if len(ok):
        y = ok.y_true.to_numpy(dtype=int)
        p = ok.p_make.to_numpy(dtype=float)

        out.update(
            brier=float(brier_score_loss(y, p)),
            log_loss=float(log_loss(y, p, labels=[0, 1])),
            accuracy=float(accuracy_score(y, p >= 0.5)),
            roc_auc=(
                float(roc_auc_score(y, p))
                if len(np.unique(y)) == 2
                else None
            ),
        )
    else:
        out.update(
            brier=None,
            log_loss=None,
            accuracy=None,
            roc_auc=None,
        )

    return out


def pairwise(
    xgb: pd.DataFrame,
    candidate: pd.DataFrame,
    rng: np.random.Generator,
    repeats: int,
):
    cols = IDENTITY + ["p_make", "status"]

    a = xgb[cols].rename(
        columns={
            "p_make": "p_xgb",
            "status": "s_xgb",
        }
    )

    b = candidate[KEYS + ["p_make", "status"]].rename(
        columns={
            "p_make": "p_llm",
            "status": "s_llm",
        }
    )

    merged = a.merge(
        b,
        on=KEYS,
        how="inner",
        validate="one_to_one",
    )

    good = merged.loc[
        merged.s_xgb.eq("ok") & merged.s_llm.eq("ok")
    ].copy()

    result = {
        "n_attempted": len(merged),
        "n_common_success": len(good),
        "n_games": int(good.game_id.nunique()),
        "brier_delta_llm_minus_xgb": None,
        "ci95_game_bootstrap": None,
    }

    if good.empty:
        return result, good

    good["xgb_sq_error"] = (good.p_xgb - good.y_true) ** 2
    good["llm_sq_error"] = (good.p_llm - good.y_true) ** 2
    good["delta_sq_error"] = (
        good.llm_sq_error - good.xgb_sq_error
    )

    result["brier_delta_llm_minus_xgb"] = float(
        good.delta_sq_error.mean()
    )

    games = good.game_id.unique()

    if len(games) >= 2:
        chunks = {
            g: good.loc[
                good.game_id.eq(g), "delta_sq_error"
            ].to_numpy()
            for g in games
        }

        samples = []

        for _ in range(repeats):
            sampled = rng.choice(
                games,
                size=len(games),
                replace=True,
            )

            values = np.concatenate(
                [chunks[g] for g in sampled]
            )

            samples.append(values.mean())

        result["ci95_game_bootstrap"] = [
            float(v)
            for v in np.quantile(samples, [0.025, 0.975])
        ]

    return result, good


def evaluate(
    xgb_path: Path,
    llm_path: Path,
    selection_path: Path,
    output: Path,
    repeats=1000,
    seed=42,
):
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output directory exists and is not empty; use a new name"
        )

    if repeats < 1:
        raise ValueError("Bootstrap repetitions must be positive")

    xgb = load_predictions(xgb_path)
    llm = load_predictions(llm_path)

    selection = pd.read_csv(
        selection_path,
        dtype={k: str for k in KEYS},
    )

    all_rows = matched_predictions(xgb, llm, selection)

    summary = []

    for name, rows in all_rows.groupby("model", sort=True):
        summary.append({
            "model": name,
            **metrics(rows),
        })

    rng = np.random.default_rng(seed)

    baseline = all_rows.loc[
        all_rows.model.eq("xgboost")
    ]

    comparisons = []
    failures = []

    for name, rows in all_rows.groupby("model", sort=True):
        if name == "xgboost":
            continue

        result, paired = pairwise(
            baseline,
            rows,
            rng,
            repeats,
        )

        comparisons.append({
            "model": name,
            **result,
        })

        if not paired.empty:
            paired["model"] = name

            paired["absolute_disagreement"] = (
                paired.p_llm - paired.p_xgb
            ).abs()

            paired["llm_worse"] = (
                paired.delta_sq_error > 0
            )

            failures.append(paired)

    if output.exists():
        output.mkdir(parents=True, exist_ok=True)
    else:
        output.mkdir(parents=True)

    pd.DataFrame(summary).to_csv(
        output / "model_metrics.csv",
        index=False,
    )

    (output / "paired_comparisons.json").write_text(
        json.dumps(comparisons, indent=2),
        encoding="utf-8",
    )

    if failures:
        cases = pd.concat(
            failures,
            ignore_index=True,
        )

        cases.sort_values(
            "absolute_disagreement",
            ascending=False,
        ).head(30).to_csv(
            output / "largest_disagreements.csv",
            index=False,
        )

        cases.sort_values(
            "delta_sq_error",
            ascending=False,
        ).head(30).to_csv(
            output / "llm_worst_relative_errors.csv",
            index=False,
        )

    lines = [
        "# H2: Matched XGBoost vs LLM Evaluation",
        "",
        "Validation-only; exact registered shot IDs required.",
        "",
        "Brier delta = LLM minus XGBoost; negative favors LLM.",
        "",
        "## Model metrics",
        "",
        pd.DataFrame(summary).to_string(index=False),
        "",
        "## Paired Brier comparisons",
        "",
        pd.DataFrame(comparisons).to_string(index=False),
        "",
        "Bootstrap resamples entire games.",
        "Intervals including zero are inconclusive.",
        "",
        "Feature parity must be audited separately before "
        "claiming architectural superiority.",
        "",
        "Prediction failures are reported through coverage "
        "and omitted from paired scores.",
    ]

    (output / "report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary, comparisons


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    for flag in ("xgb", "llm", "selection", "output"):
        parser.add_argument(
            f"--{flag}",
            type=Path,
            required=True,
        )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    summary, comparisons = evaluate(
        args.xgb,
        args.llm,
        args.selection,
        args.output,
        args.repeats,
        args.seed,
    )

    print(
        f"Wrote {len(summary)} model rows and "
        f"{len(comparisons)} paired comparisons "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
