"""Explicit CLI stages; test evaluation and heavy dependencies are opt-in."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

from .common import load_config, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="nba-shots")
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download", help="Download the original Kaggle CSV")
    download.add_argument("--output", type=Path, default=Path("data/raw/shot_logs.csv"))
    prepare = sub.add_parser("prepare", help="Audit CSV and save features and game splits")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--output", type=Path, default=Path("data/processed/kaggle"))
    prepare.add_argument("--config", type=Path)
    train = sub.add_parser("train", help="Fit baselines on training games; score validation only")
    train.add_argument("--shots", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--config", type=Path)
    train.add_argument("--cv", type=Path)
    baseline = sub.add_parser("baseline", help="Download, prepare and train in one command")
    baseline.add_argument("--input", type=Path)
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baseline"))
    baseline.add_argument("--config", type=Path)
    evaluate = sub.add_parser("evaluate", help="Evaluate saved models without refitting")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--shots", type=Path, required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluate.add_argument("--final", action="store_true")
    compare = sub.add_parser("compare", help="Paired game bootstrap for two prediction series")
    compare.add_argument("--predictions", type=Path, nargs="+", required=True)
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--repetitions", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=42)
    # These stages import optional dependencies only when invoked.
    recent = sub.add_parser("collect-recent", help="Cache recent NBA shot records")
    recent.add_argument("--season", default="2025-26")
    recent.add_argument("--output", type=Path, required=True)
    manifest = sub.add_parser(
        "make-manifest", help="Preregister games/shots without outcome selection"
    )
    manifest.add_argument("--shots", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--games", type=int, default=30)
    manifest.add_argument("--shots-per-game", type=int, default=35)
    manifest.add_argument("--seed", type=int, default=42)
    cohort = sub.add_parser("prepare-cohort", help="Register and split a matched video cohort")
    cohort.add_argument("--shots", type=Path, required=True)
    cohort.add_argument("--manifest", type=Path, required=True)
    cohort.add_argument("--output", type=Path, required=True)
    cohort.add_argument("--config", type=Path)
    videos = sub.add_parser("fetch-videos", help="Retrieve clips for a preregistered manifest")
    videos.add_argument("--manifest", type=Path, required=True)
    videos.add_argument("--output", type=Path, required=True)
    videos.add_argument("--limit", type=int, default=20)
    cv = sub.add_parser("extract-cv", help="Extract release-frame pose proxies")
    cv.add_argument("--manifest", type=Path, required=True)
    cv.add_argument("--output", type=Path, required=True)
    cv.add_argument("--weights", default="yolo11n-pose.pt")
    cv.add_argument("--device", default="cpu")
    annotate = sub.add_parser(
        "annotate", help="Interactive tool to annotate release frames and player boxes"
    )
    annotate.add_argument("--manifest", type=Path, required=True)
    annotate.add_argument("--videos", type=Path, help="Directory containing downloaded video clips")
    annotate.add_argument("--weights", default="yolo11n-pose.pt")
    annotate.add_argument("--device", default="cpu")
    annotate.add_argument("--port", type=int, default=8765)
    annotate.add_argument("--host", default="127.0.0.1")
    annotate.add_argument(
        "--no-browser", action="store_true", help="Do not automatically launch browser"
    )
    agents = sub.add_parser("predict-llm", help="Run a free local Ollama/CrewAI comparison")
    agents.add_argument("--shots", type=Path, required=True)
    agents.add_argument("--cv", type=Path)
    agents.add_argument("--output", type=Path, required=True)
    agents.add_argument("--model", default="qwen3:4b-instruct")
    agents.add_argument("--split", choices=["validation", "test"], default="validation")
    agents.add_argument("--final", action="store_true")
    agents.add_argument("--limit", type=int, default=100)
    agents.add_argument("--seed", type=int, default=42)
    agents.add_argument("--num-predict", type=int, default=192)
    agents.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    report = sub.add_parser("report", help="Combine scores and predictions without retraining")
    report.add_argument("--predictions", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--selection", type=Path, help="Explicit common evaluation selection CSV")
    args = parser.parse_args(argv)
    try:
        result = execute(args)
    except (
        ValueError,
        FileNotFoundError,
        ImportError,
        ConnectionError,
        requests.RequestException,
    ) as error:
        parser.exit(1, f"Error: {error}\n")
    if result is not None:
        print(json.dumps(result, indent=2, allow_nan=False, default=str))


def execute(args):
    from .data import download_kaggle, prepare_csv

    if args.command == "download":
        return {"csv": str(download_kaggle(args.output))}
    if args.command == "prepare":
        return {"shots": str(prepare_csv(args.input, args.output, load_config(args.config)))}
    if args.command in {"train", "baseline", "evaluate"}:
        from .models import evaluate_run, train_baselines

        if args.command == "evaluate":
            return evaluate_run(args.run, args.shots, args.split, args.final)
        if args.command == "baseline":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError(f"Run directory is not empty: {args.output}")
            config = load_config(args.config)
            csv = args.input or download_kaggle("data/raw/shot_logs.csv")
            prepared = prepare_csv(csv, args.output / "prepared", config)
            return train_baselines(prepared, args.output / "models", config)
        prepared_config = args.shots.parent / "config.json"
        config_path = args.config or (prepared_config if prepared_config.exists() else None)
        return train_baselines(args.shots, args.output, load_config(config_path), args.cv)
    if args.command in {"compare", "report"}:
        import pandas as pd

        from .evaluation import paired_comparison, write_scores

        predictions = pd.concat(
            [pd.read_csv(p, dtype={"game_id": str}) for p in args.predictions], ignore_index=True
        )
        if args.command == "compare":
            result = paired_comparison(
                predictions, args.baseline, args.candidate, args.repetitions, args.seed
            )
            write_json(args.output, result)
            return result
        from .report import create_report

        selection = pd.read_csv(args.selection) if args.selection else None
        return create_report(predictions, args.output, write_scores, selection)
    if args.command in {
        "collect-recent",
        "make-manifest",
        "prepare-cohort",
        "fetch-videos",
        "extract-cv",
    }:
        from . import vision

        if args.command == "collect-recent":
            return vision.collect_recent(args.season, args.output)
        if args.command == "make-manifest":
            return vision.make_manifest(
                args.shots, args.output, args.games, args.shots_per_game, args.seed
            )
        if args.command == "prepare-cohort":
            return vision.prepare_cohort(
                args.shots, args.manifest, args.output, load_config(args.config)
            )
        if args.command == "fetch-videos":
            return vision.fetch_videos(args.manifest, args.output, args.limit)
        return vision.extract_cv(args.manifest, args.output, args.weights, args.device)
    if args.command == "annotate":
        from .annotate import start_annotation_server

        start_annotation_server(
            manifest_path=args.manifest,
            video_dir=args.videos,
            host=args.host,
            port=args.port,
            weights=args.weights,
            device=args.device,
            open_browser=not args.no_browser,
        )
        return {"status": "completed"}
    if args.command == "predict-llm":
        from .agents import run_agents

        return run_agents(
            args.shots,
            args.output,
            args.cv,
            args.model,
            args.split,
            args.final,
            args.limit,
            args.seed,
            args.num_predict,
            args.ollama_url,
        )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
