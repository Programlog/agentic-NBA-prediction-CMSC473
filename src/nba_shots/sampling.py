"""Create reproducible, auditable NBA shot selection manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agents import select_evaluation_rows
from .common import sha256, write_json


def create_selection_manifest(
    shots_path: str | Path,
    output: str | Path,
    split: str = "validation",
    limit: int = 30,
    seed: int = 42,
    games: int = 10,
) -> dict:
    """Select balanced shots using the same algorithm as the LLM runner."""
    shots_path = Path(shots_path)
    output = Path(output)
    audit_path = output.with_suffix(".selection.json")

    if split not in {"validation", "test"}:
        raise ValueError("Only validation/test shots can be selected")

    if output.exists() or audit_path.exists():
        raise ValueError("Selection output already exists; do not overwrite")

    shots = pd.read_parquet(shots_path)

    required = {"shot_id", "game_id", "cohort", "split"}

    if missing := required - set(shots.columns):
        raise ValueError(f"Missing sampling columns: {sorted(missing)}")

    if shots.empty or shots[list(required)].isna().any().any():
        raise ValueError("Sampling metadata must be present and nonempty")

    if shots["shot_id"].duplicated().any():
        raise ValueError("Shot IDs must be unique")

    available = shots.loc[shots["split"].eq(split)]

    if available.empty:
        raise ValueError("No shots in requested split")

    if available["cohort"].nunique() != 1:
        raise ValueError("Cannot mix dataset cohorts")

    if len(available) < limit:
        raise ValueError(f"Requested {limit} shots but only {len(available)} are available")

    selected = select_evaluation_rows(
        shots,
        split,
        limit,
        seed,
        games=games,
    )

    game_count = int(selected["game_id"].nunique())

    if len(selected) != limit or game_count != games:
        raise ValueError("Selection does not match requested shots and games")

    # Selection files include only identifiers, never outcomes.
    selection = (
        selected[["cohort", "split", "shot_id", "game_id"]]
        .sort_values("shot_id")
        .reset_index(drop=True)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(output, index=False)

    counts = selection["game_id"].value_counts()

    audit = {
        "source_sha256": sha256(shots_path),
        "selection_sha256": sha256(output),
        "sampling_method": "balanced_games_v1",
        "split": split,
        "cohort": str(available["cohort"].iloc[0]),
        "seed": seed,
        "requested_shots": limit,
        "requested_games": games,
        "selected_shots": len(selection),
        "selected_games": game_count,
        "available_shots": len(available),
        "available_games": int(available["game_id"].nunique()),
        "shots_per_game": {str(game): int(count) for game, count in sorted(counts.items())},
        "selection_policy": (
            "Hash-seeded game selection followed by "
            "hash-seeded shot selection within each game. "
            "Selected games must contain enough shots to "
            "satisfy the largest per-game allocation. "
            "No outcome-based selection."
        ),
    }

    write_json(audit_path, audit)

    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced NBA shot selection manifest")

    parser.add_argument(
        "--shots",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--games",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    audit = create_selection_manifest(
        shots_path=args.shots,
        output=args.output,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        games=args.games,
    )

    print(f"Selected {audit['selected_shots']} shots from {audit['selected_games']} games.")


if __name__ == "__main__":
    main()
