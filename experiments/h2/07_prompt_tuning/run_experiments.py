"""H2 experiment 07: tuned prompts, similar-shot retrieval and XGBoost as a tool.

Conditions (GPT-6 Luna, effort none by default)
  V0 ... V4          one CrewAI analyst call per shot with prompt variant V* (see prompts.py)
  multi:V*           one specialist per evidence type in V*, then an aggregator
  single_budget:V*   one analyst with all of V*'s evidence, same number of sequential calls

Every call is cached by its exact messages, so re-running resumes and failed shots
are retried. V0 uses 05's client and cache key, so 05's calls are reused.

Usage (from the repository root, after prepare.py)
  uv run --frozen --all-extras python experiments/h2/07_prompt_tuning/run_experiments.py \
      --split validation --conditions V0 V1 V2 V3 V4
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from nba_shots.agents import crewai_assessment, prompt_features

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(H2 / "05_stronger_model"))
_spec = importlib.util.spec_from_file_location(
    "run05", H2 / "05_stronger_model" / "run_experiments.py"
)
run05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run05)
import openai_client  # noqa: E402
import prompts  # noqa: E402

REASON_FIRST_SCHEMA = {
    "type": "object",
    "properties": {"rationale": {"type": "string"}, "p_make": {"type": "number"}},
    "required": ["rationale", "p_make"],
    "additionalProperties": False,
}


class ReasonFirstClient(openai_client.OpenAIClient):
    """Same client, but the strict schema asks for the rationale before p_make."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_id += ":schema=rationale_first"

    def _request(self, messages):
        self._throttle()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "assessment",
                    "strict": True,
                    "schema": REASON_FIRST_SCHEMA,
                },
            },
        }
        if self.sampling_params:
            kwargs.update(temperature=0, seed=self.seed)
        return self.api.chat.completions.create(**kwargs)


class Guard:
    """Per-shot wrapper: refuses prompts that leak identifiers and records messages."""

    def __init__(self, client, row: dict, record: list | None):
        self.client = client
        self.model = client.model
        self.seed = client.client.seed
        self.banned = [str(row["shot_id"]), str(row["game_id"]), str(row["player_id"]), "y_true"]
        self.record = record

    def call(self, messages):
        text = json.dumps(messages)
        leaked = [b for b in self.banned if b in text]
        if leaked:
            raise RuntimeError(f"prompt leaks {leaked}")
        if self.record is not None:
            self.record.append(messages)
        return self.client.call(messages)


def parse_condition(name: str) -> tuple[str, str]:
    kind, _, variant = name.rpartition(":")
    if variant not in prompts.VARIANTS or kind not in {"", "multi", "single_budget"}:
        raise ValueError(name)
    if kind and variant == "V0":
        raise ValueError("orchestration is only defined for V1-V4")
    return kind or "one", variant


def run_condition(name: str, row: dict, client, anchors) -> tuple[float, list]:
    kind, variant = parse_condition(name)
    features = prompt_features(row)
    if kind == "one":
        role = prompts.role(variant, row, anchors, features)
        result = crewai_assessment(client, role, features, [])
        return result["p_make"], [result]
    evidence = prompts.EVIDENCE[variant]
    if kind == "multi":
        opinions = []
        for k in evidence:
            opinion = crewai_assessment(client, prompts.specialist_role(k, row), features, [])
            opinions.append({"role": prompts.ROLE_NAMES[k], **opinion})
        role = prompts.role(variant, row, title="Shot probability aggregator")
        final = crewai_assessment(client, role, features, opinions)
        return final["p_make"], opinions + [final]
    role = prompts.role(variant, row)
    previous = []
    for _ in range(len(evidence) + 1):  # same budget as multi
        previous.append(crewai_assessment(client, role, features, previous))
    return previous[-1]["p_make"], previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--conditions", nargs="+", default=prompts.VARIANTS)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--rows-dir", default="artifacts/h2_prompts")
    parser.add_argument("--day1-shots", default="artifacts/day1/prepared/shots.parquet")
    parser.add_argument("--output-root", default="artifacts/h2_prompts/runs")
    parser.add_argument("--cache05", default="artifacts/h2_stronger_model")
    parser.add_argument("--limit", type=int, help="first N selected shots (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--rpm", type=float, default=40)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    for c in args.conditions:
        parse_condition(c)

    tag = f"openai_{args.reasoning_effort}_{args.split}"
    out = Path(args.output_root) / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(exist_ok=True)

    common = dict(seed=args.seed, max_cost_usd=args.max_cost, rpm=args.rpm)
    raw_v0 = openai_client.OpenAIClient(args.model, args.reasoning_effort, **common)
    raw_new = ReasonFirstClient(args.model, args.reasoning_effort, **common)
    v0_cache = out / "call_cache_v0.jsonl"
    seed_cache = Path(args.cache05) / tag / "call_cache.jsonl"
    seed_lines = out / "call_cache_v0.seeded_lines"
    if not v0_cache.exists() and seed_cache.exists():
        shutil.copy(seed_cache, v0_cache)  # 05's anchored single analyst == V0's first call
        seed_lines.write_text(str(len(seed_cache.read_text().splitlines())))
    clients = {
        "v0": run05.CachedClient(raw_v0, v0_cache),
        "new": run05.CachedClient(raw_new, out / "call_cache.jsonl"),
    }

    rows = pd.read_parquet(Path(args.rows_dir) / f"{args.split}_rows.parquet")
    if args.limit:
        rows = rows.head(args.limit)
    day1 = pd.read_parquet(args.day1_shots)
    anchors = run05.brier04.Anchors(day1[day1["split"].eq("train")])
    rows[["shot_id", "game_id", "y_true", "in_old_100"]].to_csv(out / "selection.csv", index=False)

    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":
                done.add((r["condition"], r["shot_id"]))
    lock = threading.Lock()
    fatal = lambda: raw_v0.fatal or raw_new.fatal  # noqa: E731

    def one(condition, row, record):
        if fatal():
            return None
        client = clients["v0" if parse_condition(condition)[1] == "V0" else "new"]
        try:
            p, opinions = run_condition(condition, row, Guard(client, row, record), anchors)
            rec = {"status": "ok", "p_make": p, "opinions": opinions}
        except Exception as error:  # noqa: BLE001
            if fatal():
                return None
            rec = {"status": "error", "error": str(error), "p_make": None}
        rec.update(
            condition=condition,
            shot_id=row["shot_id"],
            game_id=row["game_id"],
            y_true=int(row["y_true"]),
        )
        with lock, pred_path.open("a") as handle:
            handle.write(json.dumps(rec) + "\n")
        return rec

    started = time.time()
    records = rows.to_dict("records")
    for condition in args.conditions:
        _, variant = parse_condition(condition)
        todo = [r for r in records if (condition, r["shot_id"]) not in done]
        first = []  # rendered messages for the first selected shot, for review and the README
        with prompts.instructions(variant), ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(one, condition, r, first if r is records[0] else None) for r in todo
            ]
            for i, future in enumerate(as_completed(futures), 1):
                rec = future.result()
                if rec is None:
                    continue
                cost = raw_v0.cost_usd + raw_new.cost_usd
                fresh = sum(c.fresh_calls for c in clients.values())
                print(
                    f"[{tag}:{condition}] {i}/{len(todo)} p={rec['p_make']} y={rec['y_true']} "
                    f"fresh={fresh} cost=${cost:.3f} t={time.time() - started:.0f}s",
                    flush=True,
                )
        if first:
            safe = condition.replace(":", "_")
            (out / "prompts" / f"{safe}.json").write_text(json.dumps(first, indent=2) + "\n")
    if fatal():
        raise SystemExit(f"Stopped early: {fatal()}. Re-run to resume.")

    usage = []  # calls made by this experiment (05's seeded cache lines excluded)
    skip = int(seed_lines.read_text()) if seed_lines.exists() else 0
    for path, start in [(v0_cache, skip), (out / "call_cache.jsonl", 0)]:
        if path.exists():
            lines = path.read_text().splitlines()[start:]
            usage += [json.loads(x).get("usage", {}) for x in lines]
    meta = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "sampling": {"temperature": 0, "seed": args.seed} if raw_new.sampling_params else None,
        "split": args.split,
        "n_shots": len(rows),
        "rows_sha256": hashlib.sha256(
            (Path(args.rows_dir) / f"{args.split}_rows.parquet").read_bytes()
        ).hexdigest(),
        "prompts_sha256": hashlib.sha256((HERE / "prompts.py").read_bytes()).hexdigest(),
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
