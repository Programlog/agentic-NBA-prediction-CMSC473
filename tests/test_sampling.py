"""Tests for reproducible and game-balanced NBA shot sampling."""

import json

import pandas as pd
import pytest

from nba_shots.agents import select_evaluation_rows
from nba_shots.sampling import create_selection_manifest


def sample_shots():
    """Six games, five shots each, with consistent metadata."""
    rows = []

    for game in range(6):
        for shot in range(5):
            rows.append(
                {
                    "shot_id": f"shot-{game}-{shot}",
                    "game_id": f"game-{game}",
                    "cohort": "kaggle_2014_15",
                    "split": "validation",
                    "y_true": shot % 2,
                }
            )

    return pd.DataFrame(rows)


def test_balanced_sampling_has_three_shots_per_game():
    frame = sample_shots()

    selected = select_evaluation_rows(
        frame,
        split="validation",
        limit=15,
        seed=42,
        games=5,
    )

    counts = selected.groupby("game_id").size()

    assert len(selected) == 15
    assert selected["game_id"].nunique() == 5
    assert (counts == 3).all()
    assert selected["shot_id"].is_unique


def test_balanced_sampling_handles_remainder():
    frame = sample_shots()

    selected = select_evaluation_rows(
        frame,
        split="validation",
        limit=17,
        seed=42,
        games=5,
    )

    counts = selected.groupby("game_id").size()

    assert len(selected) == 17
    assert len(counts) == 5
    assert counts.min() == 3
    assert counts.max() == 4
    assert counts.sum() == 17


def test_balanced_sampling_ignores_outcomes_and_order():
    frame = sample_shots()

    original = select_evaluation_rows(
        frame,
        "validation",
        15,
        seed=42,
        games=5,
    )

    changed = frame.sample(frac=1, random_state=123).copy()
    changed["y_true"] = 1 - changed["y_true"]

    repeated = select_evaluation_rows(
        changed,
        "validation",
        15,
        seed=42,
        games=5,
    )

    assert original["shot_id"].tolist() == repeated["shot_id"].tolist()


def test_balanced_sampling_rejects_insufficient_games():
    frame = sample_shots()

    with pytest.raises(ValueError, match="only 6 qualify"):
        select_evaluation_rows(
            frame,
            "validation",
            30,
            seed=42,
            games=10,
        )


def test_legacy_sampling_remains_unchanged():
    frame = sample_shots()

    old = select_evaluation_rows(
        frame,
        "validation",
        10,
        seed=42,
    )

    explicit = select_evaluation_rows(
        frame,
        "validation",
        10,
        seed=42,
        games=None,
    )

    assert old["shot_id"].tolist() == explicit["shot_id"].tolist()


def test_manifest_matches_llm_sampler(tmp_path):
    frame = sample_shots()

    source = tmp_path / "shots.parquet"
    output = tmp_path / "selection.csv"

    frame.to_parquet(source, index=False)

    audit = create_selection_manifest(
        source,
        output,
        limit=15,
        games=5,
        seed=42,
    )

    manifest = pd.read_csv(output)

    expected = select_evaluation_rows(
        frame,
        "validation",
        15,
        seed=42,
        games=5,
    )

    assert set(manifest["shot_id"]) == set(expected["shot_id"])
    assert audit["selected_shots"] == 15
    assert audit["selected_games"] == 5
    assert audit["sampling_method"] == "balanced_games_v1"
    assert output.with_suffix(".selection.json").exists()

    counts = manifest.groupby("game_id").size()
    assert (counts == 3).all()


def test_manifest_ignores_outcomes(tmp_path):
    frame = sample_shots()

    source_a = tmp_path / "shots_a.parquet"
    source_b = tmp_path / "shots_b.parquet"

    frame.to_parquet(source_a, index=False)

    changed = frame.copy()
    changed["y_true"] = 1 - changed["y_true"]
    changed.to_parquet(source_b, index=False)

    output_a = tmp_path / "selection_a.csv"
    output_b = tmp_path / "selection_b.csv"

    create_selection_manifest(source_a, output_a, limit=15, games=5)

    create_selection_manifest(source_b, output_b, limit=15, games=5)

    assert output_a.read_bytes() == output_b.read_bytes()


def test_manifest_ignores_input_order(tmp_path):
    frame = sample_shots()

    source_a = tmp_path / "ordered.parquet"
    source_b = tmp_path / "shuffled.parquet"

    frame.to_parquet(source_a, index=False)

    frame.sample(frac=1, random_state=123).to_parquet(
        source_b,
        index=False,
    )

    output_a = tmp_path / "ordered.csv"
    output_b = tmp_path / "shuffled.csv"

    create_selection_manifest(source_a, output_a, limit=15, games=5)

    create_selection_manifest(source_b, output_b, limit=15, games=5)

    assert output_a.read_bytes() == output_b.read_bytes()


def test_manifest_audit_contains_game_counts(tmp_path):
    frame = sample_shots()

    source = tmp_path / "shots.parquet"
    output = tmp_path / "selection.csv"

    frame.to_parquet(source, index=False)

    create_selection_manifest(source, output, limit=15, games=5)

    audit = json.loads(output.with_suffix(".selection.json").read_text())

    assert audit["requested_games"] == 5
    assert audit["selected_games"] == 5
    assert audit["requested_shots"] == 15
    assert audit["selected_shots"] == 15

    assert len(audit["shots_per_game"]) == 5
    assert set(audit["shots_per_game"].values()) == {3}


def test_manifest_rejects_duplicate_shots(tmp_path):
    frame = sample_shots()
    frame.loc[1, "shot_id"] = frame.loc[0, "shot_id"]

    source = tmp_path / "shots.parquet"
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="unique"):
        create_selection_manifest(
            source,
            tmp_path / "selection.csv",
            limit=15,
            games=5,
        )


def test_manifest_rejects_overwriting(tmp_path):
    frame = sample_shots()

    source = tmp_path / "shots.parquet"
    output = tmp_path / "selection.csv"

    frame.to_parquet(source, index=False)

    create_selection_manifest(source, output, limit=15, games=5)

    with pytest.raises(ValueError, match="already exists"):
        create_selection_manifest(source, output, limit=15, games=5)


def test_manifest_rejects_insufficient_shots(tmp_path):
    frame = sample_shots()

    source = tmp_path / "shots.parquet"
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="only 30"):
        create_selection_manifest(
            source,
            tmp_path / "selection.csv",
            limit=40,
            games=5,
        )


def test_manifest_rejects_training_split(tmp_path):
    frame = sample_shots()

    source = tmp_path / "shots.parquet"
    frame.to_parquet(source, index=False)

    with pytest.raises(ValueError, match="validation/test"):
        create_selection_manifest(
            source,
            tmp_path / "selection.csv",
            split="train",
            limit=15,
            games=5,
        )
