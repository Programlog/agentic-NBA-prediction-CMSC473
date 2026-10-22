"""Produce an honest, cohort-separated result summary from saved predictions."""

from pathlib import Path

import pandas as pd

from .common import utc_now, write_json
from .evaluation import paired_comparison


def create_report(predictions: pd.DataFrame, output, scorer, selection=None) -> dict:
    original_rows = len(predictions)
    if selection is not None:
        keys = ["cohort", "split", "shot_id"]
        if not set(keys) <= set(selection) or selection.duplicated(keys).any():
            raise ValueError("Selection requires unique cohort/split/shot_id rows")
        predictions = predictions.merge(
            selection[keys], on=keys, how="inner", validate="many_to_one"
        )
    for _, group in predictions.groupby(["cohort", "split"]):
        expected = set(group.shot_id)
        for _, model_rows in group.groupby("model"):
            if set(model_rows.shot_id) != expected:
                raise ValueError("Models have different evaluation shots; use --selection to align")
    if selection is not None:
        actual = set(predictions[["cohort", "split", "shot_id"]].itertuples(index=False, name=None))
        wanted = set(selection[keys].itertuples(index=False, name=None))
        if actual != wanted:
            raise ValueError("Some selected evaluation shots have no predictions")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = scorer(predictions, output, prefix="combined")
    predictions.to_csv(output / "predictions.csv", index=False)
    lines = [
        "# NBA shot probability experiments",
        "",
        f"Generated: {utc_now()}",
        "",
        "Results below are separated by dataset and partition. Lower Brier score and log "
        "loss are better; inspect reliability diagrams separately for calibration.",
        "",
    ]
    comparisons = []
    for (cohort, split), rows in predictions.groupby(["cohort", "split"]):
        lines += [
            f"## {cohort} / {split}",
            "",
            "| Model | Scored / total | Brier | Log loss | ROC-AUC | Accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for m in metrics:
            if (m["cohort"], m["split"]) != (cohort, split):
                continue
            values = [
                "undefined" if m[k] is None else f"{m[k]:.5f}"
                for k in ["brier_score", "log_loss", "roc_auc", "accuracy"]
            ]
            lines.append(
                f"| {m['model']} | {m['n_scored']} / {m['n_total']} | " + " | ".join(values) + " |"
            )
        lines.append("")
        for base, candidate in [
            ("xgboost", "xgboost_cv"),
            ("llm_single", "llm_multi"),
            ("llm_single_budget", "llm_multi"),
        ]:
            if {base, candidate} <= set(rows.model):
                try:
                    result = paired_comparison(rows, base, candidate)
                    comparisons.append({"cohort": cohort, "split": split, **result})
                    lo, hi = result["brier_delta_95pct_interval"]
                    lines.append(
                        f"- {candidate} − {base}: Brier difference "
                        f"{result['brier_delta_candidate_minus_baseline']:.5f}, "
                        f"95% game-bootstrap interval [{lo:.5f}, {hi:.5f}]."
                    )
                except ValueError as error:
                    lines.append(f"- {candidate} versus {base}: comparison unavailable: {error}.")
        lines += [
            "",
            "Metrics with incomplete coverage describe successful predictions only. "
            "Paired comparisons require complete, matching shot IDs.",
            "",
        ]
    lines += [
        "## Interpretation limits",
        "",
        "- Validation results are development evidence, not final test results.",
        "- Kaggle and recent-video scores cannot establish a CV improvement across datasets.",
        "- A bootstrap interval spanning zero does not establish an improvement.",
        "- Review clip availability, pose failures, game count and player coverage before "
        "generalizing a video-cohort result.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines))
    write_json(output / "paired_comparisons.json", comparisons)
    write_json(
        output / "selection_audit.json",
        {
            "input_prediction_rows": original_rows,
            "reported_prediction_rows": len(predictions),
            "explicit_selection": selection is not None,
        },
    )
    return {"report": str(output / "report.md"), "metrics": metrics}
