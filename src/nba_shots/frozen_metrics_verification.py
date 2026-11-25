"""Read-only verification of frozen H1 and H2 prediction metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nba_shots.evaluation import score_predictions


class VerificationError(ValueError):
    """Saved predictions or published metrics are inconsistent."""


def require_close(actual, expected, description, tolerance=1e-9):
    """Compare numeric metrics without silently accepting mismatches."""
    if actual is None or expected is None:
        if actual is not None or expected is not None:
            raise VerificationError(f"{description}: {actual} != {expected}")
        return

    if not np.isclose(float(actual), float(expected), atol=tolerance, rtol=0):
        raise VerificationError(f"{description}: calculated {actual}, published {expected}")


def load_scores(path):
    """Load predictions and check consistent shot membership."""
    frame = pd.read_csv(path, dtype={"shot_id": str, "game_id": str})

    try:
        scores = score_predictions(frame)
    except ValueError as exc:
        raise VerificationError(f"{path}: {exc}") from exc

    for (cohort, split), group in frame.groupby(["cohort", "split"]):
        models = group["model"].unique()
        reference = set(group.loc[group.model.eq(models[0]), "shot_id"])

        for model in models[1:]:
            current = set(group.loc[group.model.eq(model), "shot_id"])
            if current != reference:
                raise VerificationError(f"{cohort}/{split}: {model} has different shot IDs")

    return frame, {(row["split"], row["model"]): row for row in scores}


def verify_h1(root):
    """Verify H1's validation and frozen test results."""
    directory = root / "reports" / "basketevent_h1_test"
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8-sig"))

    checked = 0

    for split, filename in (
        ("validation", "validation_predictions_seed_mean.csv"),
        ("test", "test_predictions_seed_mean.csv"),
    ):
        frame, scores = load_scores(directory / filename)
        expected = summary[split]["scores_seed_mean"]

        if set(expected) != {model for score_split, model in scores if score_split == split}:
            raise VerificationError(f"H1 {split}: model set mismatch")

        for model, published in expected.items():
            actual = scores[(split, model)]

            for metric, key in (
                ("brier", "brier_score"),
                ("log_loss", "log_loss"),
                ("roc_auc", "roc_auc"),
            ):
                require_close(actual[key], published[metric], f"H1 {split}/{model}/{metric}")

            if actual["n_total"] != published["shots"]:
                raise VerificationError(f"H1 {split}/{model}: shot count mismatch")

            game_count = frame.loc[frame.model.eq(model), "game_id"].nunique()

            if game_count != published["games"]:
                raise VerificationError(f"H1 {split}/{model}: game count mismatch")

            checked += 1

        comparison = summary[split]["h1_seed_mean"]
        baseline = scores[(split, comparison["baseline"])]
        candidate = scores[(split, comparison["candidate"])]

        require_close(
            candidate["brier_score"] - baseline["brier_score"],
            comparison["brier_delta_candidate_minus_baseline"],
            f"H1 {split}: paired Brier difference",
        )

        if baseline["n_total"] != comparison["n_shots"]:
            raise VerificationError(f"H1 {split}: paired comparison shot count mismatch")

    return checked


def verify_h2(root):
    """Verify H2's frozen model metrics."""
    directory = root / "experiments" / "h2" / "02_final_test_100" / "report"

    _, scores = load_scores(directory / "predictions.csv")
    published = pd.read_csv(directory / "combined_metrics.csv")

    expected_keys = set(zip(published["split"], published["model"]))

    if len(expected_keys) != len(published):
        raise VerificationError("H2: duplicate published model rows")

    if set(scores) != expected_keys:
        raise VerificationError("H2: published model set mismatch")

    for row in published.to_dict("records"):
        actual = scores[(row["split"], row["model"])]

        for metric in (
            "n_total",
            "n_scored",
            "n_failed",
            "coverage",
            "brier_score",
            "log_loss",
            "roc_auc",
            "accuracy",
        ):
            require_close(actual[metric], row[metric], f"H2 {row['model']}/{metric}")

        if actual["cohort"] != row["cohort"]:
            raise VerificationError(f"H2 {row['model']}: cohort mismatch")

    return len(published)


def verify_all(root):
    return {
        "H1_model_evaluations": verify_h1(root),
        "H2_model_evaluations": verify_h2(root),
    }


def main():
    parser = argparse.ArgumentParser(description="Verify published H1/H2 frozen metrics")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    try:
        results = verify_all(args.root)
    except (VerificationError, OSError, KeyError) as exc:
        parser.exit(1, f"Verification FAILED: {exc}\n")

    print("Frozen metrics verification PASSED")
    for name, count in results.items():
        print(f"  {name}: {count} model evaluations")


if __name__ == "__main__":
    main()
