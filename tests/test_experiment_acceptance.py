"""Tests for post-run NBA LLM experiment acceptance."""

import csv
import json

import pytest

from nba_shots.agents import CONDITIONS
from nba_shots.experiment_acceptance import accept_experiment
from nba_shots.experiment_validation import ValidationError


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    run_dir = tmp_path / "experiment"
    run_dir.mkdir()

    shots = ["shot-1", "shot-2"]

    spec = {
        "shot_ids": shots,
        "conditions": CONDITIONS,
        "sampling_method": "balanced_games_v1",
        "sampling_games": 2,
    }

    (run_dir / "experiment.json").write_text(
        json.dumps({"spec": spec}),
        encoding="utf-8",
    )

    selection = [
        {"shot_id": "shot-1", "game_id": "game-1"},
        {"shot_id": "shot-2", "game_id": "game-2"},
    ]

    predictions = [
        {"shot_id": shot, "model": model, "status": "ok"} for shot in shots for model in CONDITIONS
    ]

    def write_rows(filename, rows):
        with (run_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    write_rows("selection.csv", selection)
    write_rows("predictions.csv", predictions)

    def fake_validation(folder):
        assert folder == run_dir
        return {
            "valid": True,
            "shots": 2,
            "conditions": 5,
            "predictions": 10,
            "calls": 22,
            "prompt_tokens": 1000,
            "completion_tokens": 200,
        }

    monkeypatch.setattr(
        "nba_shots.experiment_acceptance.validate_artifacts",
        fake_validation,
    )

    return run_dir, spec, selection, predictions, write_rows


def test_accepts_complete_balanced_experiment(experiment):
    run_dir, _, _, _, _ = experiment

    result = accept_experiment(run_dir, expected_shots=2, expected_games=2)

    assert result["accepted"] is True
    assert result["games"] == 2
    assert result["conditions"] == 5
    assert result["predictions"] == 10
    assert result["failed_predictions"] == 0
    assert result["calls"] == 22


def test_rejects_wrong_shot_count(experiment):
    run_dir, _, _, _, _ = experiment

    with pytest.raises(ValidationError, match="Expected 3 shots"):
        accept_experiment(run_dir, expected_shots=3)


def test_rejects_wrong_game_count(experiment):
    run_dir, _, _, _, _ = experiment

    with pytest.raises(ValidationError, match="Expected 3 games"):
        accept_experiment(run_dir, expected_games=3)


def test_rejects_failed_predictions(experiment):
    run_dir, _, _, predictions, write_rows = experiment

    predictions[0]["status"] = "error"
    write_rows("predictions.csv", predictions)

    with pytest.raises(ValidationError, match="unsuccessful predictions"):
        accept_experiment(run_dir)


def test_rejects_incomplete_condition_set(experiment):
    run_dir, spec, _, _, _ = experiment

    spec["conditions"] = {"llm_single": 1}

    (run_dir / "experiment.json").write_text(
        json.dumps({"spec": spec}),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="five required LLM conditions"):
        accept_experiment(run_dir)


def test_rejects_duplicate_selection(experiment):
    run_dir, _, selection, _, write_rows = experiment

    selection[1]["shot_id"] = "shot-1"
    write_rows("selection.csv", selection)

    with pytest.raises(ValidationError, match="duplicate shot IDs"):
        accept_experiment(run_dir)


def test_rejects_unbalanced_game_allocation(experiment):
    run_dir, spec, selection, _, write_rows = experiment

    # Create four distinct shots across two games:
    # game-1 receives 3 shots; game-2 receives 1.
    selection.extend(
        [
            {"shot_id": "shot-3", "game_id": "game-1"},
            {"shot_id": "shot-4", "game_id": "game-1"},
        ]
    )

    spec["shot_ids"].extend(["shot-3", "shot-4"])

    (run_dir / "experiment.json").write_text(
        json.dumps({"spec": spec}),
        encoding="utf-8",
    )
    write_rows("selection.csv", selection)

    with pytest.raises(ValidationError, match="Shots are not balanced"):
        accept_experiment(run_dir, expected_shots=4, expected_games=2)


def test_strict_validator_failure_is_propagated(experiment, monkeypatch):
    run_dir, _, _, _, _ = experiment

    def fail_validation(folder):
        raise ValidationError("duplicate trace records")

    monkeypatch.setattr(
        "nba_shots.experiment_acceptance.validate_artifacts",
        fail_validation,
    )

    with pytest.raises(ValidationError, match="duplicate trace records"):
        accept_experiment(run_dir)
