"""Produce trace-validated cost and prediction summaries from saved LLM runs.

Run with ``python -m nba_shots.experiment_summary --run DIR --output NEW_DIR``.
This does not invoke Ollama or change the original experiment.
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
    """Load single-run trace records, rejecting duplicate shot/arm pairs."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing traces: {path}")
    traces: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                trace = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid trace JSON on line {line_number}") from error
            if not isinstance(trace, dict):
                raise ValueError("Each inference trace must be an object")
            key = (trace.get("shot_id"), trace.get("condition"))
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


def numeric_value(value: object, label: str) -> float:
    """Require finite nonnegative measurements; booleans are not numbers."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid {label}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid {label}")
    return result


def exact_count(value: object, label: str) -> int:
    value = numeric_value(value, label)
    if not value.is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(value)


def cached_flag(value: object) -> bool:
    """Parse either the original CSV boolean string or a Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"Invalid prediction cached flag: {value!r}")


def summarize_experiment(run_dir: str | Path, output: str | Path) -> dict:
    """Validate a complete single run, then write descriptive cost summaries.

    A cached assessment has unknown *original* usage because old caches contain
    only the assessment, even though its current-run usage is trace-validated.
    """
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

    # The existing independent validator reconciles individual call records,
    # prediction rows, and the run's aggregate usage_by_condition.csv.
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
    predictions = pd.read_csv(input_files["predictions.csv"], dtype=dtype)
    validate_predictions(predictions)
    selection = pd.read_csv(input_files["selection.csv"], dtype=dtype)
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
    expected_pairs = {(shot, arm) for shot in shot_ids for arm in conditions}
    actual_pairs = set(predictions[KEYS].itertuples(index=False, name=None))
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
        if exact_count(row.calls, "prediction calls") != len(calls):
            raise ValueError("Prediction call count differs from trace")
        for column in ("prompt_tokens", "completion_tokens"):
            trace_total = sum(exact_count(call.get(column, 0), column) for call in calls)
            if exact_count(getattr(row, column), column) != trace_total:
                raise ValueError(f"Prediction {column} differs from trace")
        trace_runtime = numeric_value(result.get("runtime_seconds"), "trace runtime")
        runtime = numeric_value(row.runtime_seconds, "prediction runtime")
        if not math.isclose(trace_runtime, runtime, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("Prediction runtime differs from trace")
        if cached and calls:
            raise ValueError("Cached result unexpectedly made inference calls")

        fresh = not cached
        current_calls = exact_count(row.calls, "calls")
        current_prompt = exact_count(row.prompt_tokens, "prompt_tokens")
        current_completion = exact_count(row.completion_tokens, "completion_tokens")
        provenance_rows.append(
            {
                "shot_id": row.shot_id,
                "game_id": row.game_id,
                "cohort": row.cohort,
                "split": row.split,
                "condition": row.model,
                "status": row.status,
                "cached": cached,
                "cost_provenance": "fresh_trace" if fresh else "cached_original_unknown",
                "calls_this_run": current_calls,
                "prompt_tokens_this_run": current_prompt,
                "completion_tokens_this_run": current_completion,
                "runtime_seconds_this_run": runtime,
                "original_calls_verified": current_calls if fresh else None,
                "original_prompt_tokens_verified": current_prompt if fresh else None,
                "original_completion_tokens_verified": current_completion if fresh else None,
                "original_runtime_seconds_verified": runtime if fresh else None,
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
        calls = sum(exact_count(value, "calls") for value in rows["calls"])
        prompt_tokens = sum(exact_count(value, "prompt_tokens") for value in rows["prompt_tokens"])
        completion_tokens = sum(
            exact_count(value, "completion_tokens") for value in rows["completion_tokens"]
        )
        runtime = sum(numeric_value(value, "runtime_seconds") for value in rows["runtime_seconds"])
        cache_hits = int(condition_provenance["cached"].sum())
        original_complete = cache_hits == 0
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
                "unknown_original_costs": cache_hits,
                "original_cost_complete": original_complete,
                "verified_original_tokens": (
                    prompt_tokens + completion_tokens if original_complete else None
                ),
                "calls_this_run": calls,
                "prompt_tokens_this_run": prompt_tokens,
                "completion_tokens_this_run": completion_tokens,
                "total_tokens_this_run": prompt_tokens + completion_tokens,
                "runtime_seconds_this_run": runtime,
            }
        )
    summaries.sort(key=lambda value: (value["cohort"], value["split"], value["condition"]))

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
        "cached_predictions_original_cost_unknown": int(provenance["cached"].sum()),
        "original_cost_accounting_complete": not bool(provenance["cached"].any()),
        "source_files_sha256": {name: sha256(path) for name, path in input_files.items()},
        "cost_interpretation": (
            "Calls, tokens and runtime describe recorded execution, reconciled against "
            "per-call traces and the aggregate usage file. Cached entries have zero new "
            "calls, but original inference usage is unknown because legacy caches only "
            "store the assessment. Recorded runtime is not universal architecture latency."
        ),
        "comparison_limit": (
            "Descriptive results only. A one-shot or few-game experiment cannot "
            "establish model superiority; compare matched games separately."
        ),
    }

    def show_score(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.4f}"

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
    for row in summaries:
        lines.append(
            f"| {row['condition']} | "
            f"{row['shots_succeeded']} / {row['shots_attempted']} | "
            f"{show_score(row['brier_score'])} | {show_score(row['log_loss'])} | "
            f"{row['calls_this_run']} | {row['total_tokens_this_run']} | "
            f"{row['runtime_seconds_this_run']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Lower Brier score and log loss are better.",
            "- Scores use successful predictions only; inspect coverage for failures.",
            "- Strict artifact validation passed, including usage-summary reconciliation.",
            "- Recorded execution costs are not universal costs of the model architecture.",
            "- Cache hits are not free original inference: historical costs are unknown.",
            (
                "- For each condition, verified original totals are blank "
                "if any cache hit lacks provenance."
            ),
            "- No superiority claim follows from this descriptive table; use matched games.",
            "",
        ]
    )

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(output / "experiment_summary.csv", index=False)
    provenance.to_csv(output / "cost_provenance.csv", index=False)
    write_json(output / "experiment_summary.json", {"audit": audit, "conditions": summaries})
    (output / "experiment_summary.md").write_text("\n".join(lines), encoding="utf-8")
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a saved local LLM experiment")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_experiment(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
