"""Produce trace-validated cost and prediction summaries from saved LLM runs.

Run:
    python -m nba_shots.experiment_summary --run DIR --output NEW_DIR

No model inference is performed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .common import sha256, write_json
from .evaluation import score_predictions, validate_predictions
from .experiment_validation import validate_artifacts

KEYS = ["shot_id", "model"]
IDENTITY = ["shot_id", "game_id", "cohort", "split"]


def read_traces(path: Path) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing traces: {path}")

    traces = {}

    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue

            try:
                trace = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid trace JSON on line {number}") from error

            if not isinstance(trace, dict):
                raise ValueError("Each inference trace must be an object")

            key = (
                trace.get("shot_id"),
                trace.get("condition"),
            )

            if not all(isinstance(value, str) and value for value in key):
                raise ValueError("Trace needs nonempty shot_id and condition")

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


def numeric_value(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")

    try:
        value = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error

    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {label}")

    return value


def exact_count(value, label):
    value = numeric_value(value, label)

    if not value.is_integer():
        raise ValueError(f"{label} must be an integer")

    return int(value)


def cached_flag(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"

    raise ValueError(f"Invalid prediction cached flag: {value!r}")


def summarize_experiment(
    run_dir: str | Path,
    output: str | Path,
) -> dict:
    run_dir = Path(run_dir)
    output = Path(output)

    input_files = {
        name: run_dir / name
        for name in (
            "experiment.json",
            "predictions.csv",
            "selection.csv",
            "traces.jsonl",
            "usage_by_condition.csv",
        )
    }

    for path in input_files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing experiment file: {path}")

    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory is not empty; choose a new directory")

    strict_validation = validate_artifacts(run_dir)

    experiment = json.loads(input_files["experiment.json"].read_text(encoding="utf-8"))

    spec = experiment["spec"]
    conditions = spec["conditions"]
    shot_ids = spec["shot_ids"]
    split = spec["split"]

    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("Experiment has no registered conditions")

    if not isinstance(shot_ids, list) or not shot_ids or len(shot_ids) != len(set(shot_ids)):
        raise ValueError("Invalid registered shot IDs")

    dtype = {field: str for field in [*IDENTITY, "model"]}

    predictions = pd.read_csv(
        input_files["predictions.csv"],
        dtype=dtype,
    )

    validate_predictions(predictions)

    selection = pd.read_csv(
        input_files["selection.csv"],
        dtype=dtype,
    )

    if not set(IDENTITY) <= set(selection):
        raise ValueError("Selection is missing required columns")

    if selection[IDENTITY].isna().any().any():
        raise ValueError("Selection metadata cannot be missing")

    if selection["shot_id"].duplicated().any():
        raise ValueError("Duplicate selected shot IDs")

    if set(selection["shot_id"]) != set(shot_ids):
        raise ValueError("Selection shot IDs differ from experiment registration")

    if not selection["split"].eq(split).all():
        raise ValueError("Selection split differs from experiment")

    selected = selection.set_index("shot_id")

    expected_pairs = {(shot, condition) for shot in shot_ids for condition in conditions}

    actual_pairs = set(
        predictions[KEYS].itertuples(
            index=False,
            name=None,
        )
    )

    if actual_pairs != expected_pairs:
        raise ValueError("Prediction rows do not match registered shots and conditions")

    traces = read_traces(input_files["traces.jsonl"])

    if set(traces) != expected_pairs:
        raise ValueError("Inference traces do not match registered shots and conditions")

    provenance_rows = []

    for row in predictions.itertuples(index=False):
        selected_row = selected.loc[row.shot_id]

        if any(getattr(row, field) != selected_row[field] for field in IDENTITY[1:]):
            raise ValueError("Prediction metadata differs from selection")

        trace = traces[(row.shot_id, row.model)]
        result = trace["result"]
        calls = trace["calls"]
        cached = cached_flag(row.cached)

        if (
            result.get("shot_id", row.shot_id) != row.shot_id
            or result.get("model", row.model) != row.model
            or result.get("status") != row.status
        ):
            raise ValueError("Inference trace result disagrees with prediction")

        if "cached" in result and result["cached"] != cached:
            raise ValueError("Inference trace cached flag disagrees with prediction")

        if exact_count(
            row.calls,
            "prediction calls",
        ) != len(calls):
            raise ValueError("Prediction call count differs from trace")

        for column in (
            "prompt_tokens",
            "completion_tokens",
        ):
            trace_total = sum(
                exact_count(
                    call.get(column, 0),
                    column,
                )
                for call in calls
            )

            if (
                exact_count(
                    getattr(row, column),
                    column,
                )
                != trace_total
            ):
                raise ValueError(f"Prediction {column} differs from trace")

        trace_runtime = numeric_value(
            result.get("runtime_seconds"),
            "trace runtime",
        )

        runtime = numeric_value(
            row.runtime_seconds,
            "prediction runtime",
        )

        if not math.isclose(
            trace_runtime,
            runtime,
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            raise ValueError("Prediction runtime differs from trace")

        if cached and calls:
            raise ValueError("Cached result unexpectedly made inference calls")

        provenance_flag = getattr(
            row,
            "original_cost_provenance",
            None,
        )

        fresh = not cached

        if provenance_flag is None:
            provenance_flag = "cached_original_unknown" if cached else "fresh_trace"

        elif provenance_flag == "unknown":
            provenance_flag = "cached_original_unknown" if cached else "fresh_unknown"

        verified = provenance_flag in {
            "fresh_trace",
            "cache_v2",
        }

        original_calls = getattr(
            row,
            "original_calls",
            None,
        )

        original_prompt = getattr(
            row,
            "original_prompt_tokens",
            None,
        )

        original_completion = getattr(
            row,
            "original_completion_tokens",
            None,
        )

        original_runtime = getattr(
            row,
            "original_runtime_seconds",
            None,
        )

        current_calls = exact_count(
            row.calls,
            "calls",
        )

        current_prompt = exact_count(
            row.prompt_tokens,
            "prompt_tokens",
        )

        current_completion = exact_count(
            row.completion_tokens,
            "completion_tokens",
        )

        verified_calls = (
            exact_count(original_calls, "original_calls")
            if verified and original_calls is not None
            else (current_calls if fresh and provenance_flag == "fresh_trace" else None)
        )

        verified_prompt = (
            exact_count(
                original_prompt,
                "original_prompt_tokens",
            )
            if verified and original_prompt is not None
            else (current_prompt if fresh and provenance_flag == "fresh_trace" else None)
        )

        verified_completion = (
            exact_count(
                original_completion,
                "original_completion_tokens",
            )
            if verified and original_completion is not None
            else (current_completion if fresh and provenance_flag == "fresh_trace" else None)
        )

        verified_runtime = (
            numeric_value(
                original_runtime,
                "original_runtime_seconds",
            )
            if verified and original_runtime is not None
            else (runtime if fresh and provenance_flag == "fresh_trace" else None)
        )

        provenance_rows.append(
            {
                "shot_id": row.shot_id,
                "game_id": row.game_id,
                "cohort": row.cohort,
                "split": row.split,
                "condition": row.model,
                "status": row.status,
                "cached": cached,
                "cost_provenance": provenance_flag,
                "calls_this_run": current_calls,
                "prompt_tokens_this_run": current_prompt,
                "completion_tokens_this_run": current_completion,
                "runtime_seconds_this_run": runtime,
                "original_calls_verified": verified_calls,
                "original_prompt_tokens_verified": verified_prompt,
                "original_completion_tokens_verified": verified_completion,
                "original_runtime_seconds_verified": verified_runtime,
            }
        )

    provenance = pd.DataFrame(provenance_rows)
    metrics = score_predictions(predictions)
    summaries = []

    for metric in metrics:
        condition = metric["model"]

        rows = predictions.loc[
            predictions["model"].eq(condition)
            & predictions["cohort"].eq(metric["cohort"])
            & predictions["split"].eq(metric["split"])
        ]

        condition_provenance = provenance.loc[
            provenance["condition"].eq(condition)
            & provenance["cohort"].eq(metric["cohort"])
            & provenance["split"].eq(metric["split"])
        ]

        calls = sum(exact_count(v, "calls") for v in rows["calls"])

        prompt_tokens = sum(exact_count(v, "prompt_tokens") for v in rows["prompt_tokens"])

        completion_tokens = sum(
            exact_count(v, "completion_tokens") for v in rows["completion_tokens"]
        )

        runtime = sum(numeric_value(v, "runtime_seconds") for v in rows["runtime_seconds"])

        cache_hits = int(condition_provenance["cached"].sum())

        original_complete = bool(
            condition_provenance["original_prompt_tokens_verified"].notna().all()
        )

        verified_original_tokens = None

        if original_complete:
            verified_original_tokens = int(
                condition_provenance["original_prompt_tokens_verified"].sum()
                + condition_provenance["original_completion_tokens_verified"].sum()
            )

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
                "fresh_predictions": len(rows) - cache_hits,
                "unknown_original_costs": int(
                    condition_provenance["original_prompt_tokens_verified"].isna().sum()
                ),
                "original_cost_complete": original_complete,
                "verified_original_tokens": verified_original_tokens,
                "calls_this_run": calls,
                "prompt_tokens_this_run": prompt_tokens,
                "completion_tokens_this_run": completion_tokens,
                "total_tokens_this_run": (prompt_tokens + completion_tokens),
                "runtime_seconds_this_run": runtime,
            }
        )

    summaries.sort(
        key=lambda item: (
            item["cohort"],
            item["split"],
            item["condition"],
        )
    )

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
        "strict_artifact_validation": strict_validation,
        "fresh_predictions": int((~provenance["cached"]).sum()),
        "cached_predictions_original_cost_unknown": int(
            (provenance["cached"] & provenance["original_calls_verified"].isna()).sum()
        ),
        "original_cost_accounting_complete": bool(
            provenance["original_calls_verified"].notna().all()
        ),
        "source_files_sha256": {name: sha256(path) for name, path in input_files.items()},
        "cost_interpretation": (
            "Current-run costs are validated against traces and "
            "usage reports. Version-2 cache entries preserve original "
            "costs. Legacy cached assessments may lack original costs."
        ),
        "comparison_limit": (
            "Descriptive results only. Few-shot experiments cannot establish model superiority."
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

    def show_score(value):
        return "N/A" if value is None else f"{value:.4f}"

    for item in summaries:
        lines.append(
            f"| {item['condition']} | "
            f"{item['shots_succeeded']} / {item['shots_attempted']} | "
            f"{show_score(item['brier_score'])} | "
            f"{show_score(item['log_loss'])} | "
            f"{item['calls_this_run']} | "
            f"{item['total_tokens_this_run']} | "
            f"{item['runtime_seconds_this_run']:.1f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower Brier score and log loss are better.",
            "- Scores use successful predictions only; inspect coverage.",
            "- Strict artifact validation passed.",
            "- Recorded runtime is not universal architecture latency.",
            "- Cache hits are not free original inference.",
            "- Legacy cached predictions can have unknown original costs.",
            "- Matched-game analysis is required for meaningful comparisons.",
            "",
        ]
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(summaries).to_csv(
        output / "experiment_summary.csv",
        index=False,
    )

    provenance.to_csv(
        output / "cost_provenance.csv",
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
        "cost_provenance_csv": str(output / "cost_provenance.csv"),
        "strict_validation_passed": strict_validation["valid"],
        "conditions": len(summaries),
        "shots": len(shot_ids),
        "games": audit["games"],
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize a saved local LLM experiment")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(
        json.dumps(
            summarize_experiment(
                args.run,
                args.output,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
