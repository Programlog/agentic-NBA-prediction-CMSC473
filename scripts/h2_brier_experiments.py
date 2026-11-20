"""Exploratory H2 follow-up: can the multi-agent LLM's Brier score be improved?

Runs on the VALIDATION split only (the frozen test protocol is untouched).
Every Ollama call is cached by its exact messages, so re-running resumes and
conditions that share specialist calls reuse them.

Conditions
  multi            The frozen llm_multi condition (reference).
  multi_anchored   Same agents, but each one is told training-split make rates
                   relevant to the features it sees (overall, by distance band,
                   by defender band) and asked to adjust from that anchor.
  multi_prior      Anchored specialists, plus the aggregator is shown the
                   train-fitted logistic-regression probability as a prior.
  single_budget    The frozen llm_single_budget control (needed to calibrate
                   controls fairly).

Usage
  uv run --frozen --all-extras python scripts/h2_brier_experiments.py \
      --conditions multi multi_anchored multi_prior single_budget
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from nba_shots import agents
from nba_shots.agents import (
    OllamaClient,
    crewai_assessment,
    predict_condition,
    prompt_features,
    select_evaluation_rows,
    specialist_inputs,
)

DISTANCE_BANDS = [0, 3, 6, 10, 16, 22, 24, 28, np.inf]
DEFENDER_BANDS = [0, 2, 4, 6, 10, np.inf]


class CachedClient:
    """Wrap OllamaClient.call with an on-disk cache keyed by exact messages."""

    def __init__(self, client: OllamaClient, cache_path: Path):
        self.client = client
        self.cache_path = cache_path
        self.cache = {}
        if cache_path.exists():
            for line in cache_path.read_text().splitlines():
                entry = json.loads(line)
                self.cache[entry["key"]] = entry["result"]
        self.model = client.model
        self.fresh_calls = 0

    def call(self, messages):
        key = hashlib.sha256(
            json.dumps([self.client.model, self.client.seed, messages], sort_keys=True).encode()
        ).hexdigest()
        if key in self.cache:
            return self.cache[key]
        result = self.client.call(messages)
        self.fresh_calls += 1
        self.cache[key] = result
        with self.cache_path.open("a") as handle:
            handle.write(json.dumps({"key": key, "result": result}) + "\n")
        return result


def band_rates(train: pd.DataFrame, column: str, bands) -> pd.DataFrame:
    cut = pd.cut(train[column], bands, right=False)
    rates = train.groupby(cut, observed=True)["y_true"].agg(["mean", "size"])
    return rates


def lookup(rates: pd.DataFrame, value) -> tuple[str, float, int]:
    for interval, row in rates.iterrows():
        if value is not None and interval.left <= value < interval.right:
            hi = "+" if np.isinf(interval.right) else f"{interval.right:g}"
            label = f"{interval.left:g}-{hi}" if hi != "+" else f"{interval.left:g}+"
            return label, float(row["mean"]), int(row["size"])
    raise ValueError(f"No band for {value}")


class Anchors:
    def __init__(self, train: pd.DataFrame):
        self.overall = float(train["y_true"].mean())
        self.distance = band_rates(train, "shot_distance", DISTANCE_BANDS)
        self.defender = band_rates(train, "defender_distance", DEFENDER_BANDS)
        self.cross = (
            train.assign(
                _d=pd.cut(train["shot_distance"], DISTANCE_BANDS, right=False),
                _f=pd.cut(train["defender_distance"], DEFENDER_BANDS, right=False),
            )
            .groupby(["_d", "_f"], observed=True)["y_true"]
            .agg(["mean", "size"])
        )

    def text(self, features: dict) -> str:
        parts = [
            f"Historical reference from training shots: overall make rate "
            f"{self.overall:.1%}. Most NBA shots are missed."
        ]
        d = features.get("shot_distance")
        f = features.get("defender_distance")
        if d is not None:
            label, rate, n = lookup(self.distance, d)
            parts.append(f"Shots from {label} ft were made {rate:.1%} of the time (n={n}).")
        if f is not None:
            label, rate, n = lookup(self.defender, f)
            parts.append(
                f"Shots with the closest defender {label} ft away were made {rate:.1%} (n={n})."
            )
        if d is not None and f is not None:
            dk = pd.cut([d], DISTANCE_BANDS, right=False)[0]
            fk = pd.cut([f], DEFENDER_BANDS, right=False)[0]
            if (dk, fk) in self.cross.index and self.cross.loc[(dk, fk), "size"] >= 50:
                row = self.cross.loc[(dk, fk)]
                parts.append(
                    f"With both that distance and that defender spacing: "
                    f"{row['mean']:.1%} (n={int(row['size'])})."
                )
        parts.append(
            "Start from the most specific reference rate and adjust only modestly "
            "for the other measurements; p_make should rarely be far from it."
        )
        return " ".join(parts)


def run_condition(name, features, client, anchors, lr_prob):
    """Return (p_make, opinions) for one shot."""

    def assess(role, f, prior):
        return crewai_assessment(client, role, f, prior)

    if name == "multi":
        opinions = [
            {"role": role, **assess(role, subset, [])}
            for role, subset in specialist_inputs(features)
        ]
        final = assess("Shot probability aggregator", features, opinions)
        return final["p_make"], opinions

    if name == "single_budget":
        return predict_condition(features, "llm_single_budget", assess)["p_make"], []

    if name in {"multi_anchored", "multi_prior"}:
        opinions = []
        for role, subset in specialist_inputs(features):
            anchored_role = f"{role}. {anchors.text(subset)}"
            opinions.append({"role": role, **assess(anchored_role, subset, [])})
        agg_role = f"Shot probability aggregator. {anchors.text(features)}"
        if name == "multi_prior":
            agg_role += (
                f" A statistical model fit on training shots with these same features "
                f"estimates p_make = {lr_prob:.3f}; treat it as a well-calibrated prior "
                "and move away from it only when the specialists give clear reasons."
            )
        final = assess(agg_role, features, opinions)
        return final["p_make"], opinions

    raise ValueError(name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", default="artifacts/day1/prepared/shots.parquet")
    parser.add_argument("--models", default="artifacts/day1/models")
    parser.add_argument("--output", default="artifacts/h2_brier_experiments")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", nargs="+", default=["multi", "multi_anchored"])
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    shots = pd.read_parquet(args.shots)
    train = shots[shots["split"].eq("train")]
    anchors = Anchors(train)
    rows = select_evaluation_rows(shots, "validation", args.limit, args.seed, args.games)
    lr = joblib.load(Path(args.models) / "logistic_regression.joblib")
    lr_p = lr["pipeline"].predict_proba(rows[lr["features"]])[:, 1]
    rows = rows.assign(lr_p=lr_p)
    rows[["shot_id", "game_id", "y_true", "lr_p"]].to_csv(out / "selection.csv", index=False)

    client = CachedClient(OllamaClient(args.model, seed=args.seed), out / "call_cache.jsonl")
    pred_path = out / "predictions.jsonl"
    done = set()
    if pred_path.exists():
        for line in pred_path.read_text().splitlines():
            r = json.loads(line)
            if r["status"] == "ok":  # failed shots are retried on resume
                done.add((r["condition"], r["shot_id"]))

    for condition in args.conditions:
        started = time.time()
        for i, row in enumerate(rows.to_dict("records")):
            if (condition, row["shot_id"]) in done:
                continue
            features = prompt_features(row)
            try:
                p, opinions = run_condition(condition, features, client, anchors, row["lr_p"])
                record = {"status": "ok", "p_make": p, "opinions": opinions}
            except Exception as error:  # noqa: BLE001 - record and continue
                record = {"status": "error", "error": str(error), "p_make": None}
            record.update(
                condition=condition,
                shot_id=row["shot_id"],
                game_id=row["game_id"],
                y_true=int(row["y_true"]),
            )
            with pred_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(
                f"[{condition}] {i + 1}/{len(rows)} p={record['p_make']} "
                f"y={row['y_true']} fresh_calls={client.fresh_calls} "
                f"elapsed={time.time() - started:.0f}s",
                flush=True,
            )


if __name__ == "__main__":
    agents  # keep module import explicit for provenance
    main()
