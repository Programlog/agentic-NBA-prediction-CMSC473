"""Read-only checks before running an NBA LLM experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

from .agents import select_evaluation_rows


def check_ollama_model(model: str, ollama_url: str) -> dict:
    """Confirm a locally running Ollama server lists the requested model."""
    url = ollama_url.rstrip("/") + "/api/tags"

    try:
        with urlopen(url, timeout=5) as response:
            payload = json.load(response)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot query Ollama at {url}: {exc}") from exc

    models = payload.get("models", [])
    matches = [item for item in models if item.get("name") == model]

    if not matches:
        available = sorted(item.get("name", "") for item in models if item.get("name"))
        raise ValueError(f"Ollama model {model!r} is unavailable. Installed: {available}")

    return {
        "name": matches[0]["name"],
        "digest": matches[0].get("digest"),
    }


def preflight(
    shots_path: str | Path,
    output: str | Path,
    model: str = "qwen3:4b-instruct",
    split: str = "validation",
    limit: int = 30,
    seed: int = 42,
    games: int | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict:
    """Validate an experiment without writing files or running inference."""
    shots_path = Path(shots_path)
    output = Path(output)

    if not shots_path.is_file():
        raise ValueError(f"Shot dataset not found: {shots_path}")

    if output.exists():
        raise ValueError(f"Output path already exists: {output}. Choose a fresh directory.")

    if split not in {"validation", "test"}:
        raise ValueError("Split must be validation or test")

    if not 1 <= limit <= 500:
        raise ValueError("Shot limit must be between 1 and 500")

    if games is not None and not 2 <= games <= limit:
        raise ValueError("Balanced games must be between 2 and the shot limit")

    try:
        shots = pd.read_parquet(shots_path)
    except (OSError, ValueError, ImportError) as exc:
        raise ValueError(f"Cannot read shot dataset: {exc}") from exc

    required = {"shot_id", "game_id", "cohort", "split"}
    missing = required - set(shots.columns)

    if missing:
        raise ValueError(f"Missing sampling columns: {sorted(missing)}")

    # Use the EXACT selection algorithm employed by run_agents().
    selected = select_evaluation_rows(
        shots,
        split,
        limit,
        seed,
        games=games,
    )

    model_info = check_ollama_model(model, ollama_url)

    result = {
        "ready": True,
        "shots_path": str(shots_path),
        "output": str(output),
        "split": split,
        "model": model_info,
        "seed": seed,
        "requested_shots": limit,
        "requested_games": games,
        "selected_shots": len(selected),
        "selected_games": int(selected["game_id"].nunique()),
        "cohort": str(selected["cohort"].iloc[0]),
        "inference_calls_made": 0,
    }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check NBA LLM experiment readiness without running inference"
    )

    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")

    args = parser.parse_args()

    try:
        result = preflight(
            shots_path=args.shots,
            output=args.output,
            model=args.model,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            games=args.games,
            ollama_url=args.ollama_url,
        )
    except ValueError as exc:
        parser.exit(1, f"Preflight failed: {exc}\n")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
