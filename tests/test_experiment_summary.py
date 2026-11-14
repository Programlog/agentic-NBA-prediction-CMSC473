"""Regression tests for saved LLM experiment summaries."""

import json

import pandas as pd
import pytest

from nba_shots.experiment_summary import summarize_experiment


def make_experiment(tmp_path, failed=False, cached=False):
    """Build a synthetic two-game, two-condition experiment."""
    run = tmp_path / "run"
    run.mkdir()

    shots = ["shot-a", "shot-b"]
    conditions = ["llm_single", "llm_multi"]

    spec = {
        "model": "local-test-model",
        "model_digest": "test-digest",
        "prompt_version": "test-v1",
        "split": "validation",
        "shot_ids": shots,
        "conditions": {
            "llm_single": 1,
            "llm_multi": 1,
        },
    }

    (run / "experiment.json").write_text(
        json.dumps({"spec": spec}),
        encoding="utf-8",
    )

    selection = pd.DataFrame(
        [
            {
                "shot_id": "shot-a",
                "game_id": "game-1",
                "cohort": "test_cohort",
                "split": "validation",
            },
            {
                "shot_id": "shot-b",
                "game_id": "game-2",
                "cohort": "test_cohort",
                "split": "validation",
            },
        ]
    )

    selection.to_csv(run / "selection.csv", index=False)

    predictions = []
    traces = []

    for shot_number, shot in enumerate(shots):
        for condition in conditions:
            is_failed = failed and shot == "shot-b" and condition == "llm_multi"

            is_cached = cached and shot == "shot-a" and condition == "llm_single"

            call_count = 0 if is_cached else 1
            prompt_tokens = 0 if is_cached else 100
            completion_tokens = 0 if is_cached else 20

            record = {
                "shot_id": shot,
                "game_id": f"game-{shot_number + 1}",
                "cohort": "test_cohort",
                "split": "validation",
                "y_true": shot_number,
                "model": condition,
                "p_make": (None if is_failed else 0.3 + shot_number * 0.4),
                "status": "error" if is_failed else "ok",
                "error": "Synthetic failure" if is_failed else None,
                "cached": is_cached,
                "calls": call_count,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "runtime_seconds": 0.01 if is_cached else 2.5,
            }

            predictions.append(record)

            calls = []

            if call_count:
                calls.append(
                    {
                        "status": ("error" if is_failed else "ok"),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "runtime_seconds": 2.0,
                    }
                )

            traces.append(
                {
                    "shot_id": shot,
                    "condition": condition,
                    "timestamp": "2026-11-13T00:00:00Z",
                    "calls": calls,
                    "result": record,
                }
            )

    pd.DataFrame(predictions).to_csv(
        run / "predictions.csv",
        index=False,
    )

    with (run / "traces.jsonl").open(
        "w",
        encoding="utf-8",
    ) as stream:
        for trace in traces:
            stream.write(json.dumps(trace) + "\n")

    return run


def test_creates_all_three_summary_outputs(tmp_path):
    run = make_experiment(tmp_path)
    output = tmp_path / "summary"

    result = summarize_experiment(run, output)

    assert result["conditions"] == 2
    assert result["shots"] == 2
    assert result["games"] == 2

    assert (output / "experiment_summary.csv").exists()
    assert (output / "experiment_summary.json").exists()
    assert (output / "experiment_summary.md").exists()

    table = pd.read_csv(output / "experiment_summary.csv")

    assert set(table["condition"]) == {
        "llm_single",
        "llm_multi",
    }

    assert (table["shots_attempted"] == 2).all()
    assert (table["shots_succeeded"] == 2).all()
    assert (table["coverage"] == 1.0).all()
    assert (table["calls_this_run"] == 2).all()
    assert (table["total_tokens_this_run"] == 240).all()


def test_failures_are_included_in_coverage(tmp_path):
    run = make_experiment(tmp_path, failed=True)

    summarize_experiment(run, tmp_path / "summary")

    table = pd.read_csv(tmp_path / "summary" / "experiment_summary.csv")

    multi = table.loc[table["condition"].eq("llm_multi")].iloc[0]

    assert multi["shots_attempted"] == 2
    assert multi["shots_succeeded"] == 1
    assert multi["shots_failed"] == 1
    assert multi["coverage"] == 0.5


def test_cached_predictions_do_not_claim_new_calls(tmp_path):
    run = make_experiment(tmp_path, cached=True)

    summarize_experiment(run, tmp_path / "summary")

    table = pd.read_csv(tmp_path / "summary" / "experiment_summary.csv")

    single = table.loc[table["condition"].eq("llm_single")].iloc[0]

    assert single["cache_hits"] == 1
    assert single["calls_this_run"] == 1
    assert single["prompt_tokens_this_run"] == 100
    assert single["completion_tokens_this_run"] == 20

    markdown = (tmp_path / "summary" / "experiment_summary.md").read_text(encoding="utf-8")

    assert "Cache hits are not free original inference" in markdown


def test_rejects_trace_token_mismatch(tmp_path):
    run = make_experiment(tmp_path)

    path = run / "traces.jsonl"
    traces = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    traces[0]["calls"][0]["prompt_tokens"] = 999

    path.write_text(
        "\n".join(json.dumps(t) for t in traces) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="prompt_tokens differs from trace",
    ):
        summarize_experiment(run, tmp_path / "summary")


def test_rejects_missing_registered_prediction(tmp_path):
    run = make_experiment(tmp_path)

    path = run / "predictions.csv"
    frame = pd.read_csv(path)
    frame.iloc[:-1].to_csv(path, index=False)

    with pytest.raises(
        ValueError,
        match="do not match registered shots",
    ):
        summarize_experiment(run, tmp_path / "summary")


def test_rejects_duplicate_traces(tmp_path):
    run = make_experiment(tmp_path)

    path = run / "traces.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()

    with path.open("a", encoding="utf-8") as stream:
        stream.write(lines[0] + "\n")

    with pytest.raises(
        ValueError,
        match="Duplicate inference trace",
    ):
        summarize_experiment(run, tmp_path / "summary")


def test_refuses_to_overwrite_existing_summary(tmp_path):
    run = make_experiment(tmp_path)
    output = tmp_path / "summary"

    summarize_experiment(run, output)

    with pytest.raises(
        ValueError,
        match="not empty",
    ):
        summarize_experiment(run, output)
