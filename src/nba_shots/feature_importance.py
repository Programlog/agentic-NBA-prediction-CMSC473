from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from .common import CV_FEATURES, sha256, write_json
from .models import validate_splits


def permutation_brier_importance(
    pipeline, features: pd.DataFrame, labels: pd.Series,
    repeats: int = 10, seed: int = 42,
) -> pd.DataFrame:
    """Permute raw input columns, retaining the fitted preprocessing pipeline.

    Uses higher-is-worse Brier loss. No retraining or fitting occurs.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if features.empty or features.columns.duplicated().any():
        raise ValueError("Features must have rows and unique column names")
    if len(features) != len(labels) or not labels.isin([0, 1]).all():
        raise ValueError("Labels must match feature rows and be binary")
    y = labels.to_numpy(dtype=int)
    baseline = brier_score_loss(y, pipeline.predict_proba(features)[:, 1])
    rng = np.random.default_rng(seed)
    rows = []

    for column in features.columns:
        scores = []
        original = features[column].to_numpy(copy=True)
        for _ in range(repeats):
            shuffled = features.copy()
            # Permute an entire *original* feature before any encoding or imputation.
            shuffled[column] = original[rng.permutation(len(original))]
            predicted = pipeline.predict_proba(shuffled)[:, 1]
            scores.append(brier_score_loss(y, predicted) - baseline)
        rows.append({
            "feature": column,
            "importance_mean": float(np.mean(scores)),
            "importance_std": float(np.std(scores, ddof=0)),
            "baseline_brier": float(baseline),
            "repeats": repeats,
            "n_validation_shots": len(features),
        })
    return pd.DataFrame(rows).sort_values(
        "importance_mean", ascending=False, kind="stable"
    ).reset_index(drop=True)


def analyze_saved_model(
    run_dir: str | Path, shots_path: str | Path, output_dir: str | Path,
    model: str = "xgboost", repeats: int = 10, seed: int = 42,
) -> pd.DataFrame:
    """Read saved baseline run artifacts and analyze *validation* shots only."""
    if model not in {"xgboost", "xgboost_cv"}:
        raise ValueError("Only xgboost and xgboost_cv are supported")
    run_dir, shots_path, output_dir = map(Path, (run_dir, shots_path, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new location")

    metadata = json.loads((run_dir / "run.json").read_text())
    if sha256(shots_path) != metadata["shots_sha256"]:
        raise ValueError("Prepared data differs from the original model training data")
    if model not in metadata["models"]:
        raise ValueError(f"Model {model} was not trained in this run")
    shots = pd.read_parquet(shots_path)
    validate_splits(shots)

    if model == "xgboost_cv":
        if not metadata.get("cv_path"):
            raise ValueError("No CV feature table")
        cv_path = Path(metadata["cv_path"])
        if sha256(cv_path) != metadata.get("cv_sha256"):
            raise ValueError("CV data is different from the original model training data")
        cv = pd.read_parquet(cv_path)
        if ("shot_id" not in cv or cv.shot_id.duplicated().any()
                or set(cv.shot_id) != set(shots.shot_id)):
            raise ValueError("CV rows have to match the entire registered shot cohort")
        if missing := CV_FEATURES - set(cv.columns):
            raise ValueError(f"Missing CV features: {sorted(missing)}")
        shots = shots.merge(
            cv[["shot_id", *sorted(CV_FEATURES)]], on="shot_id",
            how="left", validate="one_to_one",
        )

    artifact = joblib.load(run_dir / f"{model}.joblib")
    if artifact.get("name") != model:
        raise ValueError("Loaded artifact does not match the requested model")
    feature_names = artifact["features"]
    if not feature_names or len(feature_names) != len(set(feature_names)):
        raise ValueError("Saved feature list is empty or duplicated")
    if set(feature_names) - set(shots.columns):
        raise ValueError("Prepared data lacks features required by saved model")
    validation = shots.loc[shots.split.eq("validation")]
    if validation.empty:
        raise ValueError("No validation shots found")
    result = permutation_brier_importance(
        artifact["pipeline"], validation[feature_names], validation["y_true"],
        repeats=repeats, seed=seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_dir / "feature_importance.csv", index=False)
    write_json(output_dir / "feature_importance.json", {
        "model": model,
        "split": "validation",
        "method": "raw-column permutation; positive delta means worse Brier score",
        "repeats": repeats,
        "seed": seed,
        "n_shots": int(len(validation)),
        "shots_sha256": sha256(shots_path),
        "baseline_brier": float(result.baseline_brier.iloc[0]),
    })
    ordered = result.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(ordered) + 1)))
    ax.barh(ordered.feature, ordered.importance_mean, xerr=ordered.importance_std)
    ax.axvline(0, linewidth=0.8, color="gray")
    ax.set_xlabel("Increase in Brier score after feature permutation")
    ax.set_title(f"{model}: validation permutation importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved baseline models directory")
    parser.add_argument("--shots", type=Path, required=True, help="Prepared shots.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["xgboost", "xgboost_cv"], default="xgboost")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = analyze_saved_model(
        args.run, args.shots, args.output, args.model, args.repeats, args.seed
    )
    print(result[["feature", "importance_mean", "importance_std"]].to_string(index=False))


if __name__ == "__main__":
    main()