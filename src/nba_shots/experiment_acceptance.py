"""Read-only acceptance checks for completed NBA LLM experiments."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from .agents import CONDITIONS
from .experiment_validation import ValidationError, validate_artifacts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def accept_experiment(
    run_dir: str | Path,
    expected_shots: int | None = None,
    expected_games: int | None = None,
) -> dict:
    """Check whether a completed experiment meets its research requirements."""
    run_dir = Path(run_dir)

    # Do not duplicate the existing trace, cache, or cost checks.
    validation = validate_artifacts(run_dir)

    with (run_dir / "experiment.json").open(encoding="utf-8") as stream:
        spec = json.load(stream)["spec"]

    selection = read_csv(run_dir / "selection.csv")
    predictions = read_csv(run_dir / "predictions.csv")

    registered_shots = spec["shot_ids"]
    registered_conditions = set(spec["conditions"])

    if registered_conditions != set(CONDITIONS):
        raise ValidationError("Experiment does not contain the five required LLM conditions")

    if expected_shots is None:
        expected_shots = len(registered_shots)

    if expected_shots < 1:
        raise ValidationError("Expected shot count must be positive")

    selected_ids = [row["shot_id"] for row in selection]

    if len(selected_ids) != len(set(selected_ids)):
        raise ValidationError("Selection contains duplicate shot IDs")

    if set(selected_ids) != set(registered_shots):
        raise ValidationError("Selection does not match registered shot IDs")

    if len(selection) != expected_shots:
        raise ValidationError(f"Expected {expected_shots} shots, found {len(selection)}")

    games = Counter(row["game_id"] for row in selection)

    if not games or any(not game_id for game_id in games):
        raise ValidationError("Selection contains missing game IDs")

    registered_games = spec.get("sampling_games")

    if expected_games is None:
        expected_games = registered_games

    if expected_games is not None:
        if expected_games < 1:
            raise ValidationError("Expected game count must be positive")

        if len(games) != expected_games:
            raise ValidationError(f"Expected {expected_games} games, found {len(games)}")

    if spec.get("sampling_method") == "balanced_games_v1":
        if registered_games is None or len(games) != registered_games:
            raise ValidationError("Balanced sampling does not match registered games")

        if max(games.values()) - min(games.values()) > 1:
            raise ValidationError("Shots are not balanced across games")

    failures = [row for row in predictions if row["status"] != "ok"]

    if failures:
        raise ValidationError(f"Experiment has {len(failures)} unsuccessful predictions")

    return {
        "accepted": True,
        "strict_validation_passed": validation["valid"],
        "shots": len(selection),
        "games": len(games),
        "conditions": len(registered_conditions),
        "predictions": len(predictions),
        "failed_predictions": 0,
        "sampling_method": spec.get("sampling_method"),
        "calls": validation["calls"],
        "prompt_tokens": validation["prompt_tokens"],
        "completion_tokens": validation["completion_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check acceptance of a completed NBA LLM experiment"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--shots", type=int)
    parser.add_argument("--games", type=int)

    args = parser.parse_args()

    try:
        result = accept_experiment(
            args.run_dir,
            expected_shots=args.shots,
            expected_games=args.games,
        )
    except (ValidationError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Experiment acceptance failed: {exc}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
