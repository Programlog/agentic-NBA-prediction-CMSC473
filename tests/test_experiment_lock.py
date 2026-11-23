"""Tests for exclusive NBA LLM experiment execution locking."""

from pathlib import Path

import pytest
from filelock import FileLock

from nba_shots.agents import exclusive_experiment_output


def test_execution_lock_allows_one_run(tmp_path):
    output = tmp_path / "experiment"

    @exclusive_experiment_output
    def fake_run(shots_path, output):
        Path(output).mkdir()
        (Path(output) / "result.txt").write_text("complete")
        return "success"

    result = fake_run("unused.parquet", output)

    assert result == "success"
    assert (output / "result.txt").read_text() == "complete"


def test_concurrent_execution_is_rejected_without_modifying_artifacts(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()

    artifact = output / "predictions.csv"
    artifact.write_text("original predictions")

    lock_path = output.with_name(output.name + ".run.lock")

    @exclusive_experiment_output
    def fake_run(shots_path, output):
        artifact.write_text("overwritten")
        return "unexpected"

    with FileLock(str(lock_path), timeout=0):
        with pytest.raises(ValueError, match="already locked"):
            fake_run("unused.parquet", output)

        assert artifact.read_text() == "original predictions"


def test_lock_is_released_after_exception(tmp_path):
    output = tmp_path / "experiment"

    @exclusive_experiment_output
    def failing_run(shots_path, output):
        raise RuntimeError("simulated inference failure")

    @exclusive_experiment_output
    def successful_run(shots_path, output):
        return "recovered"

    with pytest.raises(RuntimeError, match="simulated inference failure"):
        failing_run("unused.parquet", output)

    assert successful_run("unused.parquet", output) == "recovered"


def test_different_output_directories_do_not_share_a_lock(tmp_path):
    first = tmp_path / "experiment_a"
    second = tmp_path / "experiment_b"

    first_lock_path = first.with_name(first.name + ".run.lock")

    @exclusive_experiment_output
    def fake_run(shots_path, output):
        return Path(output).name

    with FileLock(str(first_lock_path), timeout=0):
        assert fake_run("unused.parquet", second) == "experiment_b"
