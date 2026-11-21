"""H2 experiment 05: a stronger hosted model (OpenAI gpt-6-luna) vs. local qwen.

Reuses experiment 04's anchors and conditions and adds ``single_budget_anchored``
(the budget-matched single analyst given the same training make-rate anchors),
so multi- vs. single-agent can be compared fairly once both are anchored.

Every model call is cached on disk by its exact messages, so re-running resumes,
and failed shots are retried. For the Ollama backend the cache key matches
experiment 04's, so 04's cache can be reused without new calls.

Usage (from the repository root)
  uv run --frozen --all-extras python experiments/h2/05_stronger_model/run_experiments.py \
      --backend openai --reasoning-effort none --split validation
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import joblib
import pandas as pd

from nba_shots.agents import (
    OllamaClient,
    crewai_assessment,
    prompt_features,
    select_evaluation_rows,
)

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "brier04", HERE.parent / "04_brier_followup" / "run_experiments.py"
)
brier04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(brier04)

CONDITIONS = ["multi", "single_budget", "multi_anchored", "single_budget_anchored"]


def last_call_log(client) -> dict:
    """This thread's most recent call log (OpenAIClient), else the last one (Ollama)."""
    local = getattr(client, "local", None)
    if local is not None:
        return getattr(local, "last", {})
    return client.calls[-1] if client.calls else {}


class CachedClient:
    """Thread-safe disk cache around a client's ``call``.

    The key is sha256([cache_id, seed, messages]); for Ollama ``cache_id`` is the
    plain model name, which reproduces experiment 04's keys exactly.
    """

    def __init__(self, client, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.model = client.model
        self.cache_id = getattr(client, "cache_id", client.model)
        self.cache = {}
        self.lock = threading.Lock()
        self.fresh_calls = 0
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["result"]

    def call(self, messages):
        key = hashlib.sha256(
            json.dumps([self.cache_id, self.client.seed, messages], sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        result = self.client.call(messages)
        usage = {
            k: v
            for k, v in last_call_log(self.client).items()
            if k.endswith("_tokens") or k in {"cost_usd", "runtime_seconds"}
        }
        with self.lock:
            self.fresh_calls += 1
            self.cache[key] = result
            with self.cache_path.open("a") as handle:
                handle.write(json.dumps({"key": key, "result": result, "usage": usage}) + "\n")
        return result


def run_condition(name, features, client, anchors, lr_prob):
    if name == "single_budget_anchored":
        role = f"Shot probability analyst. {anchors.text(features)}"
        n_calls = len(brier04.specialist_inputs(features)) + 1  # same budget as multi
        previous = []
        for _ in range(n_calls):
            result = crewai_assessment(client, role, features, previous)
            previous.append(result)
        return result["p_make"], previous
    return brier04.run_condition(name, features, client, anchors, lr_prob)


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", help="default: gpt-6-luna / qwen3:4b-instruct")
    parser.add_argument(
        "--reasoning-effort", default="none", choices=["none", "low", "medium", "high"]
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument("--shots", default="artifacts/day1/prepared/shots.parquet")
    parser.add_argument("--models", default="artifacts/day1/models")
    parser.add_argument("--output-root", default="artifacts/h2_stronger_model")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost", type=float, default=10.0, help="USD cap for this run")
    parser.add_argument("--rpm", type=float, default=45, help="client-side requests/minute cap")
    parser.add_argument("--workers", type=int, default=4, help="parallel shots (openai only)")
    args = parser.parse_args()

    if args.backend == "openai":
        from openai_client import OpenAIClient

        model = args.model or "gpt-6-luna"
        raw_client = OpenAIClient(
            model, args.reasoning_effort, seed=args.seed, max_cost_usd=args.max_cost, rpm=args.rpm
        )
        tag = f"openai_{args.reasoning_effort}_{args.split}"
        workers = args.workers
    else:
        model = args.model or "qwen3:4b-instruct"
        raw_client = OllamaClient(model, seed=args.seed)
        tag = f"ollama_{args.split}"
        workers = 1

    out = Path(args.output_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    shots_path = Path(args.shots)
    shots = pd.read_parquet(shots_path)
    anchors = brier04.Anchors(shots[shots["split"].eq("train")])
    rows = select_evaluation_rows(shots, args.split, args.limit, args.seed, args.games)
    lr = joblib.load(Path(args.models) / "logistic_regression.joblib")
    rows = rows.assign(lr_p=lr["pipeline"].predict_proba(rows[lr["features"]])[:, 1])
    rows[["shot_id", "game_id", "y_true", "lr_p"]].to_csv(out / "selection.csv", index=False)

    client = CachedClient(raw_client, out / "call_cache.jsonl")
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":  # failed shots are retried on resume
                done.add((r["condition"], r["shot_id"]))
    write_lock = threading.Lock()

    def one(condition, row):
        if getattr(raw_client, "fatal", None):
            return None
        try:
            p, opinions = run_condition(
                condition, prompt_features(row), client, anchors, row["lr_p"]
            )
            record = {"status": "ok", "p_make": p, "opinions": opinions}
        except Exception as error:  # noqa: BLE001 - record and continue
            if getattr(raw_client, "fatal", None):
                return None  # quota/budget stop: leave the shot for a later resume
            record = {"status": "error", "error": str(error), "p_make": None}
        record.update(
            condition=condition,
            shot_id=row["shot_id"],
            game_id=row["game_id"],
            y_true=int(row["y_true"]),
        )
        with write_lock, pred_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record

    started = time.time()
    for condition in args.conditions:
        todo = [r for r in rows.to_dict("records") if (condition, r["shot_id"]) not in done]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(one, condition, r) for r in todo]
            for i, future in enumerate(as_completed(futures), 1):
                record = future.result()
                if record is None:
                    continue
                print(
                    f"[{tag}:{condition}] {i}/{len(todo)} p={record['p_make']} "
                    f"y={record['y_true']} fresh={client.fresh_calls} "
                    f"cost=${getattr(raw_client, 'cost_usd', 0.0):.3f} "
                    f"t={time.time() - started:.0f}s",
                    flush=True,
                )

    if getattr(raw_client, "fatal", None):
        raise SystemExit(f"Stopped early: {raw_client.fatal}. Re-run the same command to resume.")
    cache_file = out / "call_cache.jsonl"
    usage = [
        json.loads(x).get("usage", {})
        for x in (cache_file.read_text().splitlines() if cache_file.exists() else [])
    ]
    meta = {
        "backend": args.backend,
        "model": model,
        "model_digest": getattr(raw_client, "digest", None),
        "reasoning_effort": args.reasoning_effort if args.backend == "openai" else None,
        "sampling": (
            {"temperature": 0, "seed": args.seed}
            if args.backend == "ollama" or getattr(raw_client, "sampling_params", False)
            else "provider default (temperature/seed not sent with reasoning)"
        ),
        "split": args.split,
        "n_shots": len(rows),
        "data_sha256": hashlib.sha256(shots_path.read_bytes()).hexdigest(),
        "git_commit": git_commit(),
        "cached_calls_total": len(usage),
        "prompt_tokens_total": sum(u.get("prompt_tokens", 0) or 0 for u in usage),
        "completion_tokens_total": sum(u.get("completion_tokens", 0) or 0 for u in usage),
        "reasoning_tokens_total": sum(u.get("reasoning_tokens", 0) or 0 for u in usage),
        "cost_usd_total": round(sum(u.get("cost_usd", 0) or 0 for u in usage), 4),
    }
    (out / "experiment.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    main()
