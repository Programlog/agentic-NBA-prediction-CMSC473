"""Tests for read-only experiment preflight validation."""

from pathlib import Path

import pandas as pd
import pytest

from nba_shots.experiment_preflight import preflight


@pytest.fixture
def shot_file(tmp_path: Path) -> Path:
    rows = []

    for game in range(4):
        for shot in range(5):
            rows.append(
                {
                    "shot_id": f"shot-{game}-{shot}",
                    "game_id": f"game-{game}",
                    "cohort": "test_cohort",
                    "split": "validation",
                    "period": 1,
                    "shot_distance": 12,
                    "y_true": shot % 2,
                }
            )

    path = tmp_path / "shots.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


@pytest.fixture
def available_model(monkeypatch):
    def fake_check(model, ollama_url):
        return {"name": model, "digest": "test-digest"}

    monkeypatch.setattr(
        "nba_shots.experiment_preflight.check_ollama_model",
        fake_check,
    )


def test_balanced_preflight_is_read_only(shot_file, available_model, tmp_path):
    output = tmp_path / "new_experiment"

    result = preflight(
        shots_path=shot_file,
        output=output,
        limit=6,
        games=2,
        seed=42,
    )

    assert result["ready"] is True
    assert result["selected_shots"] == 6
    assert result["selected_games"] == 2
    assert result["inference_calls_made"] == 0
    assert not output.exists()


def test_preflight_rejects_existing_output(shot_file, available_model, tmp_path):
    output = tmp_path / "previous_experiment"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        preflight(
            shots_path=shot_file,
            output=output,
            limit=6,
            games=2,
        )


def test_preflight_rejects_impossible_game_request(shot_file, available_model, tmp_path):
    with pytest.raises(ValueError):
        preflight(
            shots_path=shot_file,
            output=tmp_path / "new_experiment",
            limit=6,
            games=5,
        )


def test_preflight_rejects_missing_dataset(available_model, tmp_path):
    with pytest.raises(ValueError, match="not found"):
        preflight(
            shots_path=tmp_path / "missing.parquet",
            output=tmp_path / "new_experiment",
        )


def test_preflight_rejects_invalid_game_count(shot_file, available_model, tmp_path):
    with pytest.raises(ValueError, match="Balanced games"):
        preflight(
            shots_path=shot_file,
            output=tmp_path / "new_experiment",
            limit=6,
            games=1,
        )


def test_preflight_rejects_unavailable_model(shot_file, monkeypatch, tmp_path):
    def unavailable(model, ollama_url):
        raise ValueError("Ollama model unavailable")

    monkeypatch.setattr(
        "nba_shots.experiment_preflight.check_ollama_model",
        unavailable,
    )

    with pytest.raises(ValueError, match="unavailable"):
        preflight(
            shots_path=shot_file,
            output=tmp_path / "new_experiment",
            limit=6,
            games=2,
        )
