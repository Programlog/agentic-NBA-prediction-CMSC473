"""Fine-grained tests for saved LLM artifact consistency checks.

Run: pytest -q tests/test_experiment_validation.py
"""

import csv
import json
from pathlib import Path

import pytest

from nba_shots.experiment_validation import ValidationError, validate_artifacts


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path):
    """Create one complete one-shot experiment without any network dependencies."""
    (tmp_path / "experiment.json").write_text(json.dumps({"spec": {
        "shot_ids": ["s1"], "conditions": {"llm_single": 1}}}))
    pred = {"shot_id": "s1", "model": "llm_single", "status": "ok",
            "cached": "False", "p_make": "0.4", "calls": "1",
            "prompt_tokens": "100", "completion_tokens": "20", "runtime_seconds": "4.0"}
    _write_csv(tmp_path / "predictions.csv", [pred])
    _write_csv(tmp_path / "usage_by_condition.csv", [{
        "condition": "llm_single", "shots_attempted": 1,
        "shots_succeeded": 1, "shots_failed": 0, "coverage": 1,
        "cache_hits": 0, "calls_this_run": 1, "prompt_tokens_this_run": 100,
        "completion_tokens_this_run": 20, "runtime_seconds_this_run": 4.0}])
    trace = {"shot_id": "s1", "condition": "llm_single",
             "calls": [{"prompt_tokens": 100, "completion_tokens": 20,
                        "runtime_seconds": 3.5}],
             "result": {"status": "ok", "p_make": 0.4, "calls": 1,
                        "prompt_tokens": 100, "completion_tokens": 20,
                        "runtime_seconds": 4.0}}
    (tmp_path / "traces.jsonl").write_text(json.dumps(trace) + "\n")
    return pred, trace


def test_valid_run_counts_tokens_and_calls(tmp_path):
    make_run(tmp_path)
    result = validate_artifacts(tmp_path)
    assert result["valid"] is True
    assert (result["shots"], result["conditions"], result["calls"]) == (1, 1, 1)
    assert (result["prompt_tokens"], result["completion_tokens"]) == (100, 20)


def test_rejects_missing_condition_prediction(tmp_path):
    make_run(tmp_path)
    exp = tmp_path / "experiment.json"
    data = json.loads(exp.read_text())
    data["spec"]["conditions"]["llm_multi"] = 4
    exp.write_text(json.dumps(data))
    with pytest.raises(ValidationError, match="coverage"):
        validate_artifacts(tmp_path)


def test_rejects_incorrect_prediction_token_count(tmp_path):
    make_run(tmp_path)
    pred = next(csv.DictReader((tmp_path / "predictions.csv").open()))
    pred["prompt_tokens"] = "99"
    _write_csv(tmp_path / "predictions.csv", [pred])
    with pytest.raises(ValidationError, match="prompt_tokens"):
        validate_artifacts(tmp_path)


def test_rejects_incorrect_usage_total(tmp_path):
    make_run(tmp_path)
    row = next(csv.DictReader((tmp_path / "usage_by_condition.csv").open()))
    row["calls_this_run"] = "2"
    _write_csv(tmp_path / "usage_by_condition.csv", [row])
    with pytest.raises(ValidationError, match="calls_this_run"):
        validate_artifacts(tmp_path)


def test_rejects_duplicate_trace_entries_from_repeated_runs(tmp_path):
    make_run(tmp_path)
    trace = (tmp_path / "traces.jsonl").read_text()
    (tmp_path / "traces.jsonl").write_text(trace + trace)
    with pytest.raises(ValidationError, match="duplicate"):
        validate_artifacts(tmp_path)


def test_rejects_invalid_probability(tmp_path):
    make_run(tmp_path)
    pred = next(csv.DictReader((tmp_path / "predictions.csv").open()))
    pred["p_make"] = "1.5"
    _write_csv(tmp_path / "predictions.csv", [pred])
    with pytest.raises(ValidationError, match="probability"):
        validate_artifacts(tmp_path)


def test_allows_framework_overhead_in_runtime(tmp_path):
    make_run(tmp_path)
    assert validate_artifacts(tmp_path)["valid"] is True


def test_rejects_impossible_runtime(tmp_path):
    make_run(tmp_path)
    trace_file = tmp_path / "traces.jsonl"
    trace = json.loads(trace_file.read_text())
    trace["calls"][0]["runtime_seconds"] = 5.0
    trace_file.write_text(json.dumps(trace) + "\n")
    with pytest.raises(ValidationError, match="runtime exceeds"):
        validate_artifacts(tmp_path)
