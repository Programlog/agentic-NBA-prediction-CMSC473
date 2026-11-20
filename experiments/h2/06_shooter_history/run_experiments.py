"""H2 experiment 06: give the agents shooter history (arm A) and the shooter's name (arm B).

Conditions (GPT-6 Luna by default; all anchored as in 04/05):
  multi_hist     anchored multi-agent; the existing Player history analyst is
                 switched on and the history lines are added to its role and the aggregator's
  single_hist    anchored single analyst with the same history lines, same call budget
  multi_named    multi_hist + the shooter's name (the LLM may use its own knowledge)
  single_named   single_hist + the shooter's name

Usage (from the repository root, after prepare.py)
  uv run --frozen --all-extras python experiments/h2/06_shooter_history/run_experiments.py \
      --split validation
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nba_shots.agents import (
    crewai_assessment,
    prompt_features,
    select_evaluation_rows,
    specialist_inputs,
)

HERE = Path(__file__).resolve().parent
H2 = HERE.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(H2 / "05_stronger_model"))
run05 = _load("run05", H2 / "05_stronger_model" / "run_experiments.py")
brier04 = run05.brier04
from openai_client import OpenAIClient  # noqa: E402

CONDITIONS = ["multi_hist", "single_hist", "multi_named", "single_named"]
HISTORY_ROLE = "Player history analyst"


def band_label(distance: float) -> str:
    bands = brier04.DISTANCE_BANDS
    for lo, hi in zip(bands[:-1], bands[1:]):
        if lo <= distance < hi:
            return f"{lo:g}+ ft" if hi == float("inf") else f"{lo:g}-{hi:g} ft"
    raise ValueError(distance)


def history_text(row: dict, named: bool) -> str:
    text = (
        f"Shooter history from earlier games only: this player made "
        f"{row['player_band_rate']:.1%} of shots from {band_label(row['shot_distance'])} "
        f"(n={int(row['player_band_attempts'])} earlier attempts there; small samples are "
        f"shrunk toward the league rate for that distance) and "
        f"{row['player_prior_make_rate']:.1%} of all {int(row['player_prior_attempts'])} "
        "earlier attempts."
    )
    if named:
        text += (
            f" The shooter is {row['player_name'].title()}. You may use general knowledge of this "
            "player's shooting ability, but nothing about this specific game or its outcome."
        )
    return text


def run_condition(name: str, row: dict, client, anchors) -> tuple[float, list]:
    features = prompt_features(row)
    named = name.endswith("_named")
    hist = history_text(row, named)
    if not named:
        assert row["player_name"] not in hist

    def assess(role, f, previous):
        return crewai_assessment(client, role, f, previous)

    if name.startswith("multi"):
        opinions = []
        for role, subset in specialist_inputs(features):
            text = f"{role}. {anchors.text(subset)}"
            if role == HISTORY_ROLE:
                text += " " + hist
            opinions.append({"role": role, **assess(text, subset, [])})
        final = assess(
            f"Shot probability aggregator. {anchors.text(features)} {hist}", features, opinions
        )
        return final["p_make"], opinions

    role = f"Shot probability analyst. {anchors.text(features)} {hist}"
    previous = []
    for _ in range(len(specialist_inputs(features)) + 1):  # same budget as multi
        previous.append(assess(role, features, previous))
    return previous[-1]["p_make"], previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--shots", default="artifacts/h2_history/shots.parquet")
    parser.add_argument("--day1-shots", default="artifacts/day1/prepared/shots.parquet")
    parser.add_argument("--output-root", default="artifacts/h2_history/runs")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost", type=float, default=5.0)
    parser.add_argument("--rpm", type=float, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    raw_client = OpenAIClient(
        args.model, args.reasoning_effort, seed=args.seed, max_cost_usd=args.max_cost, rpm=args.rpm
    )
    tag = f"openai_{args.reasoning_effort}_{args.split}"
    out = Path(args.output_root) / tag
    out.mkdir(parents=True, exist_ok=True)

    shots = pd.read_parquet(args.shots)
    # anchors come from the unchanged day-1 training split, exactly as in 04/05
    day1 = pd.read_parquet(args.day1_shots)
    anchors = brier04.Anchors(day1[day1["split"].eq("train")])
    rows = select_evaluation_rows(shots, args.split, args.limit, args.seed, args.games)
    rows[["shot_id", "game_id", "y_true"]].to_csv(out / "selection.csv", index=False)

    client = run05.CachedClient(raw_client, out / "call_cache.jsonl")
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["condition"], r["shot_id"]))
    lock = threading.Lock()

    def one(condition, row):
        if raw_client.fatal:
            return None
        try:
            p, opinions = run_condition(condition, row, client, anchors)
            record = {"status": "ok", "p_make": p, "opinions": opinions}
        except Exception as error:  # noqa: BLE001
            if raw_client.fatal:
                return None
            record = {"status": "error", "error": str(error), "p_make": None}
        record.update(
            condition=condition,
            shot_id=row["shot_id"],
            game_id=row["game_id"],
            y_true=int(row["y_true"]),
        )
        with lock, pred_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    started = time.time()
    for condition in args.conditions:
        todo = [r for r in rows.to_dict("records") if (condition, r["shot_id"]) not in done]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(one, condition, r) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                if record is None:
                    continue
                print(
                    f"[{tag}:{condition}] {i}/{len(todo)} p={record['p_make']} "
                    f"y={record['y_true']} fresh={client.fresh_calls} "
                    f"cost=${raw_client.cost_usd:.3f} t={time.time() - started:.0f}s",
                    flush=True,
                )
    if raw_client.fatal:
        raise SystemExit(f"Stopped early: {raw_client.fatal}. Re-run to resume.")

    cache = out / "call_cache.jsonl"
    usage = [json.loads(x).get("usage", {}) for x in cache.read_text().splitlines()]
    meta = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sampling": {"temperature": 0, "seed": args.seed} if raw_client.sampling_params else None,
        "split": args.split,
        "n_shots": len(rows),
        "shots_sha256": hashlib.sha256(Path(args.shots).read_bytes()).hexdigest(),
        "git_commit": run05.git_commit(),
        "cached_calls_total": len(usage),
        "prompt_tokens_total": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
        "completion_tokens_total": sum(u.get("completion_tokens", 0) or 0 for u in usage),
        "cost_usd_total": round(sum(u.get("cost_usd", 0) or 0 for u in usage), 4),
    }
    (out / "experiment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
