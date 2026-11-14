"""Validate and summarize saved local LLM experiment results.

No model inference is performed. Costs describe the recorded execution,
not hypothetical uncached inference or monetary API charges.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .common import sha256, write_json
from .evaluation import score_predictions, validate_predictions

KEYS = ["shot_id", "model"]

USAGE_COLUMNS = [
    "calls",
    "prompt_tokens",
    "completion_tokens",
    "runtime_seconds",
]


def read_traces(path: Path) -> dict:
    """Read exactly one inference trace per shot and condition."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing traces: {path}")

    traces = {}

    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            try:
                trace = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid trace JSON on line {number}") from error

            if not isinstance(trace, dict):
                raise ValueError("Each trace must be an object")

            key = (trace.get("shot_id"), trace.get("condition"))

            if not all(isinstance(x, str) and x for x in key):
                raise ValueError("Trace needs shot_id and condition")

            if key in traces:
                raise ValueError(f"Duplicate inference trace for {key}")

            if not isinstance(trace.get("calls"), list):
                raise ValueError("Trace calls must be a list")

            if not isinstance(trace.get("result"), dict):
                raise ValueError("Trace result must be an object")

            traces[key] = trace

    if not traces:
        raise ValueError("No inference traces found")

    return traces


def numeric_value(value, label: str) -> float:
    """Reject absent, nonnumeric, negative, or nonfinite usage."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error

    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {label}")

    return result


def exact_count(value, label: str) -> int:
    number = numeric_value(value, label)

    if not number.is_integer():
        raise ValueError(f"{label} must be an integer")

    return int(number)


def summarize_experiment(
    run_dir: str | Path,
    output: str | Path,
) -> dict:
    """Build an audited summary from predictions and inference traces."""
    run_dir = Path(run_dir)
    output = Path(output)

    experiment_path = run_dir / "experiment.json"
    predictions_path = run_dir / "predictions.csv"
    selection_path = run_dir / "selection.csv"
    traces_path = run_dir / "traces.jsonl"

    for path in [
        experiment_path,
        predictions_path,
        selection_path,
        traces_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment file: {path}")

    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is not empty; choose a new directory")

    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    spec = experiment["spec"]

    conditions = spec["conditions"]
    shot_ids = spec["shot_ids"]
    split = spec["split"]

    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Experiment has no registered conditions")

    if not isinstance(shot_ids, list) or not shot_ids:
        raise ValueError("Experiment has no registered shots")

    if len(shot_ids) != len(set(shot_ids)):
        raise ValueError("Duplicate registered shot IDs")

    predictions = pd.read_csv(
        predictions_path,
        dtype={
            "shot_id": str,
            "game_id": str,
            "cohort": str,
            "split": str,
            "model": str,
        },
    )

    validate_predictions(predictions)

    selection = pd.read_csv(
        selection_path,
        dtype={
            "shot_id": str,
            "game_id": str,
            "cohort": str,
            "split": str,
        },
    )

    required_selection = {
        "shot_id",
        "game_id",
        "cohort",
        "split",
    }

    if not required_selection <= set(selection.columns):
        raise ValueError("Selection is missing required columns")

    if selection[sorted(required_selection)].isna().any().any():
        raise ValueError("Selection metadata cannot be missing")

    if selection["shot_id"].duplicated().any():
        raise ValueError("Duplicate selected shot IDs")

    if set(selection["shot_id"]) != set(shot_ids):
        raise ValueError("Selection shot IDs differ from experiment registration")

    if not selection["split"].eq(split).all():
        raise ValueError("Selection split differs from experiment")

    selected = selection.set_index("shot_id")
    expected_pairs = {(shot_id, condition) for shot_id in shot_ids for condition in conditions}

    actual_pairs = set(predictions[KEYS].itertuples(index=False, name=None))

    if actual_pairs != expected_pairs:
        raise ValueError("Prediction rows do not match registered shots and conditions")

    for row in predictions.itertuples(index=False):
        selected_row = selected.loc[row.shot_id]

        if (
            row.game_id != selected_row["game_id"]
            or row.cohort != selected_row["cohort"]
            or row.split != selected_row["split"]
        ):
            raise ValueError("Prediction metadata differs from selection")

    traces = read_traces(traces_path)

    if set(traces) != expected_pairs:
        raise ValueError("Inference traces do not match registered shots and conditions")

    for row in predictions.itertuples(index=False):
        trace = traces[(row.shot_id, row.model)]
        trace_result = trace["result"]
        calls = trace["calls"]

        if (
            trace_result.get("shot_id") != row.shot_id
            or trace_result.get("model") != row.model
            or trace_result.get("status") != row.status
        ):
            raise ValueError("Inference trace result disagrees with prediction")

        if exact_count(row.calls, "prediction calls") != len(calls):
            raise ValueError("Prediction call count differs from trace")

        for column in ["prompt_tokens", "completion_tokens"]:
            trace_total = sum(exact_count(call.get(column, 0), column) for call in calls)

            if exact_count(getattr(row, column), column) != trace_total:
                raise ValueError(f"Prediction {column} differs from trace")

        trace_runtime = numeric_value(
            trace_result.get("runtime_seconds"),
            "trace runtime",
        )
        prediction_runtime = numeric_value(
            row.runtime_seconds,
            "prediction runtime",
        )

        if not math.isclose(
            trace_runtime,
            prediction_runtime,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("Prediction runtime differs from trace")

        if bool(row.cached) and calls:
            raise ValueError("Cached result unexpectedly made inference calls")

    metrics = score_predictions(predictions)
    summaries = []

    for metric in metrics:
        condition = metric["model"]

        rows = predictions.loc[predictions["model"].eq(condition)]

        calls = sum(exact_count(value, "calls") for value in rows["calls"])

        prompt_tokens = sum(exact_count(value, "prompt_tokens") for value in rows["prompt_tokens"])

        completion_tokens = sum(
            exact_count(value, "completion_tokens") for value in rows["completion_tokens"]
        )

        runtime = sum(numeric_value(value, "runtime_seconds") for value in rows["runtime_seconds"])

        cache_hits = int(rows["cached"].eq(True).sum())

        summaries.append(
            {
                "cohort": metric["cohort"],
                "split": metric["split"],
                "condition": condition,
                "shots_attempted": metric["n_total"],
                "shots_succeeded": metric["n_scored"],
                "shots_failed": metric["n_failed"],
                "coverage": metric["coverage"],
                "brier_score": metric["brier_score"],
                "log_loss": metric["log_loss"],
                "roc_auc": metric["roc_auc"],
                "accuracy": metric["accuracy"],
                "cache_hits": cache_hits,
                "calls_this_run": calls,
                "prompt_tokens_this_run": prompt_tokens,
                "completion_tokens_this_run": completion_tokens,
                "total_tokens_this_run": (prompt_tokens + completion_tokens),
                "runtime_seconds_this_run": runtime,
            }
        )

    summaries.sort(key=lambda row: row["condition"])

    audit = {
        "model": spec["model"],
        "model_digest": spec["model_digest"],
        "prompt_version": spec["prompt_version"],
        "split": split,
        "registered_shots": len(shot_ids),
        "registered_conditions": sorted(conditions),
        "games": int(selection["game_id"].nunique()),
        "prediction_rows": len(predictions),
        "trace_rows": len(traces),
        "source_files_sha256": {
            name: sha256(path)
            for name, path in [
                ("experiment.json", experiment_path),
                ("predictions.csv", predictions_path),
                ("selection.csv", selection_path),
                ("traces.jsonl", traces_path),
            ]
        },
        "cost_interpretation": (
            "Calls, tokens and runtime describe this recorded "
            "execution. Cached predictions have zero new calls "
            "but do not represent zero original inference cost."
        ),
        "comparison_limit": (
            "Descriptive results only. One-shot or few-game "
            "experiments cannot establish model superiority."
        ),
    }

    lines = [
        "# LLM experiment cost and performance",
        "",
        f"Model: `{audit['model']}`",
        "",
        f"Prompt version: `{audit['prompt_version']}`",
        "",
        f"Split: `{split}`",
        "",
        f"Registered shots: {len(shot_ids)}",
        "",
        f"Games: {audit['games']}",
        "",
        "| Condition | Scored / attempted | Brier | Log loss | Calls | Tokens | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def display(value):
        return "N/A" if value is None else f"{value:.4f}"

    for row in summaries:
        lines.append(
            f"| {row['condition']} | "
            f"{row['shots_succeeded']} / {row['shots_attempted']} | "
            f"{display(row['brier_score'])} | "
            f"{display(row['log_loss'])} | "
            f"{row['calls_this_run']} | "
            f"{row['total_tokens_this_run']} | "
            f"{row['runtime_seconds_this_run']:.1f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Lower Brier score and log loss are better.",
        "- Performance scores use successful predictions only; coverage must also be considered.",
        "- Calls, tokens and runtime reflect this recorded run. "
        "Cache hits are not free original inference.",
        "- No causal or statistical superiority claim is made from this descriptive table.",
        "- Game-level uncertainty requires sufficient "
        "matched games and a separate paired analysis.",
        "",
    ]

    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(summaries).to_csv(
        output / "experiment_summary.csv",
        index=False,
    )

    write_json(
        output / "experiment_summary.json",
        {
            "audit": audit,
            "conditions": summaries,
        },
    )

    (output / "experiment_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "summary_csv": str(output / "experiment_summary.csv"),
        "summary_json": str(output / "experiment_summary.json"),
        "summary_markdown": str(output / "experiment_summary.md"),
        "conditions": len(summaries),
        "shots": len(shot_ids),
        "games": audit["games"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a saved local LLM experiment")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    print(
        json.dumps(
            summarize_experiment(args.run, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
