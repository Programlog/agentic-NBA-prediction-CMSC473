"""Tests for saved experiment artifact and provenance consistency."""

import csv
import json
from pathlib import Path

import pytest

from nba_shots.experiment_validation import (
    ValidationError,
    validate_artifacts,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path):
    (tmp_path / "experiment.json").write_text(
        json.dumps(
            {
                "spec": {
                    "shot_ids": ["s1"],
                    "conditions": {
                        "llm_single": 1,
                    },
                }
            }
        )
    )

    pred = {
        "shot_id": "s1",
        "model": "llm_single",
        "status": "ok",
        "cached": "False",
        "p_make": "0.4",
        "calls": "1",
        "prompt_tokens": "100",
        "completion_tokens": "20",
        "runtime_seconds": "4.0",
    }

    _write_csv(
        tmp_path / "predictions.csv",
        [pred],
    )

    _write_csv(
        tmp_path / "usage_by_condition.csv",
        [
            {
                "condition": "llm_single",
                "shots_attempted": 1,
                "shots_succeeded": 1,
                "shots_failed": 0,
                "coverage": 1,
                "cache_hits": 0,
                "calls_this_run": 1,
                "prompt_tokens_this_run": 100,
                "completion_tokens_this_run": 20,
                "runtime_seconds_this_run": 4.0,
            }
        ],
    )

    trace = {
        "shot_id": "s1",
        "condition": "llm_single",
        "calls": [
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "runtime_seconds": 3.5,
            }
        ],
        "result": {
            "status": "ok",
            "p_make": 0.4,
            "calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "runtime_seconds": 4.0,
        },
    }

    (tmp_path / "traces.jsonl").write_text(json.dumps(trace) + "\n")

    return pred, trace


def test_valid_run_counts_tokens_and_calls(tmp_path):
    make_run(tmp_path)

    result = validate_artifacts(tmp_path)

    assert result["valid"] is True
    assert result["shots"] == 1
    assert result["conditions"] == 1
    assert result["calls"] == 1
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 20


def test_rejects_missing_condition_prediction(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "experiment.json"
    data = json.loads(path.read_text())

    data["spec"]["conditions"]["llm_multi"] = 4
    path.write_text(json.dumps(data))

    with pytest.raises(
        ValidationError,
        match="coverage",
    ):
        validate_artifacts(tmp_path)


def test_rejects_incorrect_prediction_token_count(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "predictions.csv"

    with path.open() as stream:
        pred = next(csv.DictReader(stream))

    pred["prompt_tokens"] = "99"

    _write_csv(path, [pred])

    with pytest.raises(
        ValidationError,
        match="prompt_tokens",
    ):
        validate_artifacts(tmp_path)


def test_rejects_incorrect_usage_total(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "usage_by_condition.csv"

    with path.open() as stream:
        row = next(csv.DictReader(stream))

    row["calls_this_run"] = "2"

    _write_csv(path, [row])

    with pytest.raises(
        ValidationError,
        match="calls_this_run",
    ):
        validate_artifacts(tmp_path)


def test_rejects_duplicate_trace_entries_from_repeated_runs(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "traces.jsonl"
    trace = path.read_text()
    path.write_text(trace + trace)

    with pytest.raises(
        ValidationError,
        match="duplicate",
    ):
        validate_artifacts(tmp_path)


def test_rejects_invalid_probability(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "predictions.csv"

    with path.open() as stream:
        pred = next(csv.DictReader(stream))

    pred["p_make"] = "1.5"
    _write_csv(path, [pred])

    with pytest.raises(
        ValidationError,
        match="probability",
    ):
        validate_artifacts(tmp_path)


def test_allows_framework_overhead_in_runtime(tmp_path):
    make_run(tmp_path)

    assert validate_artifacts(tmp_path)["valid"] is True


def test_rejects_impossible_runtime(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "traces.jsonl"
    trace = json.loads(path.read_text())

    trace["calls"][0]["runtime_seconds"] = 5.0

    path.write_text(json.dumps(trace) + "\n")

    with pytest.raises(
        ValidationError,
        match="runtime exceeds",
    ):
        validate_artifacts(tmp_path)


def test_rejects_disagreement_about_cache_status(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "traces.jsonl"
    trace = json.loads(path.read_text(encoding="utf-8"))

    trace["result"]["cached"] = True

    path.write_text(
        json.dumps(trace) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="cached flag disagrees",
    ):
        validate_artifacts(tmp_path)


def test_rejects_trace_result_identity_mismatch(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "traces.jsonl"
    trace = json.loads(path.read_text(encoding="utf-8"))

    trace["result"]["model"] = "another_arm"

    path.write_text(
        json.dumps(trace) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError,
        match="trace model identity differs",
    ):
        validate_artifacts(tmp_path)


def test_rejects_false_fresh_original_usage(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "predictions.csv"

    with path.open(encoding="utf-8") as stream:
        pred = next(csv.DictReader(stream))

    pred.update(
        {
            "original_cost_provenance": "fresh_trace",
            "original_calls": "1",
            "original_prompt_tokens": "999",
            "original_completion_tokens": "20",
            "original_runtime_seconds": "4.0",
        }
    )

    _write_csv(path, [pred])

    with pytest.raises(
        ValidationError,
        match="original cost original_prompt_tokens",
    ):
        validate_artifacts(tmp_path)


def test_rejects_unknown_original_cost_with_fabricated_tokens(tmp_path):
    make_run(tmp_path)

    path = tmp_path / "predictions.csv"

    with path.open(encoding="utf-8") as stream:
        pred = next(csv.DictReader(stream))

    pred.update(
        {
            "original_cost_provenance": "unknown",
            "original_calls": "",
            "original_prompt_tokens": "100",
            "original_completion_tokens": "",
            "original_runtime_seconds": "",
        }
    )

    _write_csv(path, [pred])

    with pytest.raises(
        ValidationError,
        match="unknown original costs must be blank",
    ):
        validate_artifacts(tmp_path)
