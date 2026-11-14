"""Validate saved NBA LLM inference artifacts against their individual call traces.

Usage:
    python -m nba_shots.experiment_validation artifacts/llm_pilot_nov11_1shot

This validates completed, single-run artifact directories. A resumed run appends
traces, so it must be validated with a separately captured run trace or a
future run-aware validator; duplicate trace keys are rejected explicitly.
No files are modified, and no model or network connection is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


class ValidationError(ValueError):
    """A saved experiment violates its declared accounting invariants."""


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _number(value: object, context: str, *, integer: bool = False) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError) as exc:
        raise ValidationError(f"{context}: not numeric ({value!r})") from exc
    if not math.isfinite(number) or number < 0 or (integer and not number.is_integer()):
        raise ValidationError(f"{context}: invalid nonnegative measurement ({value!r})")
    return number


def _equal(actual: object, expected: object, label: str, *, tolerance: float = 1e-6) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-8, abs_tol=tolerance):
        raise ValidationError(f"{label}: recorded {actual}, traced {expected}")


def validate_artifacts(folder: str | Path) -> dict[str, object]:
    """Check shot/condition parity, traces, token totals, and usage summaries.

    Intended for one complete run with exactly one trace record per shot/arm.
    Raises ValidationError on an inconsistent or incomplete artifact set.
    """
    folder = Path(folder)
    spec = json.loads((folder / "experiment.json").read_text(encoding="utf-8"))["spec"]
    expected_shots = spec["shot_ids"]
    expected_arms = spec["conditions"]
    if not expected_shots or len(set(expected_shots)) != len(expected_shots):
        raise ValidationError("experiment.json: invalid or repeated shot IDs")
    if not expected_arms:
        raise ValidationError("experiment.json: no experimental conditions")

    predictions = _read_csv(folder / "predictions.csv")
    usage = _read_csv(folder / "usage_by_condition.csv")
    traces = []
    with (folder / "traces.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"traces.jsonl line {line_no}: malformed JSON") from exc
    expected_keys = {(shot, arm) for shot in expected_shots for arm in expected_arms}
    pred_by_key = {}
    for row in predictions:
        key = (row["shot_id"], row["model"])
        if key in pred_by_key:
            raise ValidationError(f"predictions.csv: duplicate shot/condition {key}")
        pred_by_key[key] = row
    if set(pred_by_key) != expected_keys:
        raise ValidationError(
            "predictions.csv: shot/condition coverage differs from experiment.json"
        )

    traces_by_key = {}
    for trace in traces:
        key = (trace["shot_id"], trace["condition"])
        if key in traces_by_key:
            raise ValidationError(
                f"traces.jsonl: duplicate shot/condition {key}; validate a single run"
            )
        traces_by_key[key] = trace
    if set(traces_by_key) != expected_keys:
        raise ValidationError("traces.jsonl: missing or unexpected shot/condition records")

    totals = defaultdict(
        lambda: {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "cache_hits": 0,
            "calls": 0,
            "prompt": 0,
            "completion": 0,
            "runtime": 0.0,
        }
    )
    for (shot, arm), row in pred_by_key.items():
        trace = traces_by_key[(shot, arm)]
        status = row["status"]
        if status not in {"ok", "error"}:
            raise ValidationError(f"{shot}/{arm}: unknown prediction status {status!r}")
        cached_text = row["cached"].strip().lower()
        if cached_text not in {"true", "false"}:
            raise ValidationError(f"{shot}/{arm}: invalid cached flag")
        cached = cached_text == "true"
        calls = trace["calls"]
        if not isinstance(calls, list):
            raise ValidationError(f"{shot}/{arm}: calls must be a list")
        if cached and calls:
            raise ValidationError(f"{shot}/{arm}: cached prediction contains new calls")
        if trace["result"]["status"] != status:
            raise ValidationError(f"{shot}/{arm}: trace and prediction statuses disagree")
        result = trace["result"]
        if result.get("shot_id") not in (None, shot):
            raise ValidationError(f"{shot}/{arm}: trace shot identity differs")
        if result.get("model") not in (None, arm):
            raise ValidationError(f"{shot}/{arm}: trace model identity differs")
        if "cached" in result:
            if not isinstance(result["cached"], bool):
                raise ValidationError(f"{shot}/{arm}: invalid trace cached flag")
            if result["cached"] != cached:
                raise ValidationError(f"{shot}/{arm}: trace cached flag disagrees")
        if cached and status != "ok":
            raise ValidationError(f"{shot}/{arm}: cached prediction is not successful")
        if status == "ok":
            probability = _number(row["p_make"], f"{shot}/{arm}: p_make")
            if probability > 1:
                raise ValidationError(f"{shot}/{arm}: probability above one")
            _equal(probability, trace["result"]["p_make"], f"{shot}/{arm}: traced probability")
        call_prompt = sum(
            int(
                _number(
                    c.get("prompt_tokens", 0),
                    f"{shot}/{arm}: call prompt",
                    integer=True,
                )
            )
            for c in calls
        )
        call_completion = sum(
            int(
                _number(
                    c.get("completion_tokens", 0),
                    f"{shot}/{arm}: call completion",
                    integer=True,
                )
            )
            for c in calls
        )
        for col, actual in (
            ("calls", len(calls)),
            ("prompt_tokens", call_prompt),
            ("completion_tokens", call_completion),
        ):
            _equal(
                _number(row[col], f"{shot}/{arm}: {col}", integer=True),
                actual,
                f"{shot}/{arm}: {col}",
            )
            _equal(
                _number(trace["result"][col], f"{shot}/{arm}: trace {col}", integer=True),
                actual,
                f"{shot}/{arm}: trace {col}",
            )
        runtime = _number(row["runtime_seconds"], f"{shot}/{arm}: runtime")
        _equal(runtime, trace["result"]["runtime_seconds"], f"{shot}/{arm}: trace runtime")
        call_runtime = sum(
            _number(
                c.get("runtime_seconds", 0),
                f"{shot}/{arm}: call runtime",
            )
            for c in calls
        )
        if call_runtime > runtime + 0.05:
            raise ValidationError(f"{shot}/{arm}: call runtime exceeds overall prediction runtime")
        x = totals[arm]
        x["attempted"] += 1
        x["succeeded"] += status == "ok"
        x["failed"] += status == "error"
        x["cache_hits"] += cached
        x["calls"] += len(calls)
        x["prompt"] += call_prompt
        x["completion"] += call_completion
        x["runtime"] += runtime

    usage_by_arm = {}
    for row in usage:
        arm = row["condition"]
        if arm in usage_by_arm:
            raise ValidationError(f"usage_by_condition.csv: duplicate condition {arm}")
        usage_by_arm[arm] = row
    if set(usage_by_arm) != set(expected_arms):
        raise ValidationError("usage_by_condition.csv: missing or unexpected conditions")
    fields = {
        "shots_attempted": "attempted",
        "shots_succeeded": "succeeded",
        "shots_failed": "failed",
        "cache_hits": "cache_hits",
        "calls_this_run": "calls",
        "prompt_tokens_this_run": "prompt",
        "completion_tokens_this_run": "completion",
        "runtime_seconds_this_run": "runtime",
    }
    for arm, x in totals.items():
        row = usage_by_arm[arm]
        for column, key in fields.items():
            _equal(_number(row[column], f"{arm}: {column}"), x[key], f"{arm}: {column}")
        _equal(
            _number(row["coverage"], f"{arm}: coverage"),
            x["succeeded"] / x["attempted"],
            f"{arm}: coverage",
        )
    return {
        "valid": True,
        "shots": len(expected_shots),
        "conditions": len(expected_arms),
        "predictions": len(predictions),
        "calls": sum(x["calls"] for x in totals.values()),
        "prompt_tokens": sum(x["prompt"] for x in totals.values()),
        "completion_tokens": sum(x["completion"] for x in totals.values()),
    }


def main() -> int:
    """Run validation from a terminal, returning a nonzero exit code on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_artifacts(args.artifact_dir), indent=2))
    except (ValidationError, FileNotFoundError, KeyError, TypeError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Artifact validation failed: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
