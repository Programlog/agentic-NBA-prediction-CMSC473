"""A local-only, cached one-call/four-call agent experiment with strict budgets."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

from .common import (
    ALLOWED_FEATURES,
    CV_FEATURES,
    HISTORY_FEATURES,
    environment,
    sha256,
    utc_now,
    write_json,
)
from .evaluation import write_scores
from .models import validate_splits

PROMPT_VERSION = "release-v3-controls"
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "p_make": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["p_make", "rationale"],
    "additionalProperties": False,
}
# Maximum calls per shot; call_budget drops a specialist (and a budget call) without inputs.
CONDITIONS = {
    "llm_single": 1,
    "llm_single_budget": 4,  # Existing iterative control retained for comparison.
    "llm_single_decomposed": 1,
    "llm_single_ensemble": 4,  # Actual call count adapts to present specialists.
    "llm_multi": 4,
}
DECOMPOSITION_ROLE = (
    "Single analyst: in ONE assessment, separately consider game context, "
    "player history, and shot mechanics/defense; ignore missing groups, "
    "then combine the available evidence into one probability. "
    "Use only supplied features; do not invent outside information"
)
INSTRUCTIONS = (
    "Estimate the probability an NBA field goal is made at the instant of release. "
    "Only use the supplied pre-release measurements; null means unknown. "
    "The supplied measurements are enough to estimate: judge how each one makes this shot "
    "easier or harder, and let p_make reflect that difficulty. "
    "Do not infer player identities, retrieve facts, or assume an outcome. "
    "Any CV measurements are 2D image proxies, not physical release/entry angles or heights. "
    "Return JSON with p_make (a number between 0 and 1) and a short rationale."
)
# A sample probability here is copied into answers, so this stays free of numbers.
EXPECTED_OUTPUT = "JSON with your numeric p_make estimate and a brief evidence rationale"


def prompt_features(row: dict) -> dict:
    """Never send target, game/date/player identity, filename or shot ID to the LLM."""
    return {
        key: (
            None
            if pd.isna(row[key])
            else row[key].item()
            if isinstance(row[key], np.generic)
            else row[key]
        )
        for key in sorted(ALLOWED_FEATURES & set(row))
    }


def parse_assessment(text: str) -> dict:
    text = text.strip()
    if text.startswith("Final Answer:"):
        text = text.removeprefix("Final Answer:").strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    value = json.loads(text)
    p = value.get("p_make")
    if (
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not np.isfinite(p)
        or not 0 <= p <= 1
    ):
        raise ValueError("LLM returned no valid p_make in [0,1]")
    if not isinstance(value.get("rationale"), str):
        raise ValueError("LLM returned no rationale")
    return {"p_make": float(p), "rationale": value["rationale"]}


class OllamaClient:
    def __init__(self, model, base_url="http://127.0.0.1:11434", seed=42, num_predict=512):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local Ollama HTTP endpoint is allowed; no paid fallback")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama URL must be a plain local endpoint")
        if num_predict < 32 or num_predict > 1024:
            raise ValueError("num_predict must be between 32 and 1024")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.seed, self.num_predict = seed, num_predict
        self.temperature = 0.0
        self.session = requests.Session()
        self.session.trust_env = False
        self.calls = []
        response = self.session.get(self.base_url + "/api/tags", timeout=10)
        response.raise_for_status()
        matches = [m for m in response.json().get("models", []) if m["name"] == model]
        if len(matches) != 1:
            raise ValueError(f"Local model {model!r} is not installed. Run: ollama pull {model}")
        self.digest = matches[0]["digest"]

    def call(self, messages):
        started = time.perf_counter()
        log = {
            "messages": messages,
            "status": "error",
            "p_make": None,
            "sampling_seed": self.seed,
            "sampling_temperature": self.temperature,
        }
        self.calls.append(log)
        try:
            response = self.session.post(
                self.base_url + "/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": OUTPUT_SCHEMA,
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": self.temperature,
                        "seed": self.seed,
                        "num_ctx": 8192,
                        "num_predict": self.num_predict,
                    },
                },
                timeout=(10, 240),
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload["message"]["content"]
            log.update(
                {
                    "raw_response": raw,
                    "prompt_tokens": payload.get("prompt_eval_count", 0),
                    "completion_tokens": payload.get("eval_count", 0),
                }
            )
            assessment = parse_assessment(raw)
            log.update({"status": "ok", **assessment})
            return assessment
        except Exception as error:
            log["error"] = str(error)
            raise
        finally:
            log["runtime_seconds"] = time.perf_counter() - started


def crewai_assessment(client: OllamaClient, role: str, features: dict, previous: list) -> dict:
    # No remote tracing, memory embeddings, tools, provider defaults or hidden retries.
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["CREWAI_TRACING_ENABLED"] = "false"
    os.environ.setdefault("CREWAI_STORAGE_DIR", str(Path("artifacts/crewai_state").resolve()))
    # CrewAI 1.15 initializes its optional platform trace client on import even when
    # tracing is disabled. Treat this experiment as logged out without reading or
    # creating per-user platform credentials. This does not affect Ollama access.
    from crewai_core.auth import token as platform_auth

    def no_platform_account():
        raise platform_auth.AuthError("Platform access disabled for this local-only experiment")

    with patch.object(platform_auth, "get_auth_token", no_platform_account):
        from crewai import Agent, BaseLLM, Crew, Task

    class OneCallLocalLLM(BaseLLM):
        def __init__(self):
            super().__init__(model=client.model, temperature=0)
            self.used = False

        def call(self, messages, tools=None, callbacks=None, available_functions=None, **kwargs):
            if self.used:
                raise RuntimeError("One-call budget exhausted; framework retries are disabled")
            self.used = True
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            result = client.call(messages)
            return "Final Answer: " + json.dumps(result)

        def supports_function_calling(self):
            return False

        def supports_stop_words(self):
            return False

        def get_context_window_size(self):
            return 8192

    adapter = OneCallLocalLLM()
    agent = Agent(
        role=role,
        goal=INSTRUCTIONS,
        backstory="You assess supplied measurements without outside information.",
        llm=adapter,
        function_calling_llm=adapter,
        tools=[],
        allow_delegation=False,
        max_iter=1,
        max_retry_limit=0,
        verbose=False,
        reasoning=False,
    )
    task = Task(
        description=INSTRUCTIONS
        + "\n"
        + json.dumps(
            {
                "features": features,
                "previous_assessments": previous,
            },
            allow_nan=False,
        ),
        expected_output=EXPECTED_OUTPUT,
        agent=agent,
    )
    result = Crew(
        agents=[agent],
        tasks=[task],
        memory=False,
        cache=False,
        planning=False,
        verbose=False,
        tracing=False,
    ).kickoff()
    return parse_assessment(result.raw)


def specialist_inputs(features: dict) -> list[tuple[str, dict]]:
    """Specialists with no known inputs are skipped; an empty prompt invites a guessed prior."""
    mechanics = CV_FEATURES | {"shot_distance", "defender_distance", "dribbles", "touch_time"}
    specialists = []
    for role, keys in [
        ("Game context analyst", ALLOWED_FEATURES - mechanics - HISTORY_FEATURES),
        ("Player history analyst", HISTORY_FEATURES),
        ("Shot mechanics and defense analyst", mechanics),
    ]:
        subset = {k: v for k, v in features.items() if k in keys}
        if any(v is not None for v in subset.values()):
            specialists.append((role, subset))
    return specialists


def call_budget(features: dict, condition: str) -> int:
    """llm_single_budget matches the calls llm_multi actually makes for the same shot."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown agent condition: {condition}")
    return (
        1
        if condition in {"llm_single", "llm_single_decomposed"}
        else len(specialist_inputs(features)) + 1
    )


def predict_condition(features: dict, condition: str, assessor) -> dict:
    if set(features) - ALLOWED_FEATURES:
        raise ValueError("Agent inputs must contain only allowlisted pre-release features")
    if condition == "llm_single":
        return assessor("Shot probability analyst", features, [])
    if condition == "llm_single_decomposed":
        # Same complete feature payload as the multi-agent aggregator; no extra call.
        return assessor(DECOMPOSITION_ROLE, features, [])
    if condition == "llm_single_ensemble":
        # Independent stochastic assessments: never expose earlier predictions.
        # The run harness varies seeds/temperature for this arm only.
        results = [
            assessor("Independent shot probability analyst", features, [])
            for _ in range(call_budget(features, condition))
        ]
        return {
            "p_make": float(np.mean([r["p_make"] for r in results])),
            "rationale": "Mean of independent probability estimates (no shared assessments)",
        }
    if condition == "llm_single_budget":
        previous = []
        for _ in range(call_budget(features, condition)):
            result = assessor("Shot probability analyst", features, previous)
            previous.append(result)
        return result
    if condition == "llm_multi":
        opinions = [
            {"role": role, **assessor(role, subset, [])}
            for role, subset in specialist_inputs(features)
        ]
        return assessor("Shot probability aggregator", features, opinions)
    raise ValueError(f"Unknown agent condition: {condition}")


def select_evaluation_rows(shots: pd.DataFrame, split: str, limit: int, seed=42):
    if limit < 1 or limit > 500:
        raise ValueError("Use between 1 and 500 preregistered LLM evaluation shots")
    rows = shots.loc[shots.split.eq(split)].sort_values("shot_id")
    if rows.empty:
        raise ValueError("No shots in the requested split")
    # Selection depends on IDs/seed, never labels or extracted feature quality.
    ranking = rows.shot_id.map(lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    return rows.loc[ranking.sort_values().head(limit).index].sort_values("shot_id")



def summarize_agent_usage(predictions: pd.DataFrame) -> pd.DataFrame:
    """Per-arm execution coverage and *this-run* compute, including failed calls.

    Cache hits count toward coverage but consume zero new calls/tokens; those
    historical costs remain in their original run's trace, not this report.
    """
    required = {
        "model", "status", "cached", "calls", "prompt_tokens",
        "completion_tokens", "runtime_seconds",
    }
    if missing := required - set(predictions):
        raise ValueError(f"Usage report missing columns: {sorted(missing)}")

    # Reject incomplete or impossible measurements before aggregation.
    # Pandas may otherwise skip missing values when computing totals.
    numeric_columns = (
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "runtime_seconds",
    )

    for column in numeric_columns:
        values = predictions[column]

        if (
            not pd.api.types.is_numeric_dtype(values)
            or not np.isfinite(values.to_numpy(dtype=float)).all()
            or (values < 0).any()
        ):
            raise ValueError(
                f"Invalid usage measurement in column: {column}"
            )
    records = []
    for name, rows in predictions.groupby("model", sort=True):
        ok = rows["status"].eq("ok")
        records.append({
            "condition": name,
            "shots_attempted": len(rows),
            "shots_succeeded": int(ok.sum()),
            "shots_failed": int((~ok).sum()),
            "coverage": float(ok.mean()),
            "cache_hits": int(rows["cached"].sum()),
            "calls_this_run": int(rows["calls"].sum()),
            "prompt_tokens_this_run": int(rows["prompt_tokens"].sum()),
            "completion_tokens_this_run": int(rows["completion_tokens"].sum()),
            "runtime_seconds_this_run": float(rows["runtime_seconds"].sum()),
        })
    return pd.DataFrame.from_records(records)


def run_agents(
    shots_path,
    output,
    cv_path=None,
    model="qwen3:4b-instruct",
    split="validation",
    final=False,
    limit=100,
    seed=42,
    num_predict=512,
    base_url="http://127.0.0.1:11434",
) -> dict:
    if split == "test" and not final:
        raise ValueError("LLM test evaluation requires --final after prompt freeze")
    if split not in {"validation", "test"}:
        raise ValueError("LLM evaluation is limited to validation/test")
    shots = pd.read_parquet(shots_path)
    validate_splits(shots)
    if cv_path:
        cv = pd.read_parquet(cv_path)
        if set(shots.shot_id) != set(cv.shot_id) or cv.shot_id.duplicated().any():
            raise ValueError("CV features must cover the same complete registered cohort")
        shots = shots.merge(
            cv[["shot_id", *sorted(CV_FEATURES)]], on="shot_id", validate="one_to_one"
        )
    selected = select_evaluation_rows(shots, split, limit, seed)
    client = OllamaClient(model, base_url, seed, num_predict)
    spec = {
        "shots_sha256": sha256(shots_path),
        "cv_sha256": sha256(cv_path) if cv_path else None,
        "model": model,
        "model_digest": client.digest,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "shot_ids": selected.shot_id.tolist(),
        "seed": seed,
        "num_predict_per_call": num_predict,
        "conditions": CONDITIONS,
        "ensemble_temperature": 0.7,
        "ensemble_seed_rule": "base seed plus zero-based call index within shot/arm",
        "implementation_sha256": sha256(__file__),
    }
    output = Path(output)
    if (output / "experiment.json").exists():
        existing = json.loads((output / "experiment.json").read_text())
        if existing["spec"] != spec:
            raise ValueError("Experiment changed; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / "cache"
    cache_dir.mkdir(exist_ok=True)
    write_json(
        output / "experiment.json",
        {"spec": spec, "environment": environment(), "started_at": utc_now()},
    )
    selected[["shot_id", "game_id", "cohort", "split"]].to_csv(
        output / "selection.csv",
        index=False,
    )
    predictions = []
    for row in selected.to_dict("records"):
        features = prompt_features(row)
        for condition in CONDITIONS:
            expected_calls = call_budget(features, condition)
            tick = time.perf_counter()
            cache_input = {"spec": spec, "features": features, "condition": condition}
            key = hashlib.sha256(json.dumps(cache_input, sort_keys=True).encode()).hexdigest()
            cache = cache_dir / f"{key}.json"
            record = {k: row[k] for k in ["shot_id", "game_id", "cohort", "split", "y_true"]}
            record.update(
                {
                    "model": condition,
                    "p_make": None,
                    "status": "error",
                    "error": None,
                    "cached": False,
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            )
            start_calls = len(client.calls)
            try:
                if cache.exists():
                    assessment = parse_assessment(cache.read_text())
                    record["cached"] = True
                else:
                    def assess(role, f, prior):
                        # Do not alter the other deterministic arms. Within the
                        # ensemble, change the seed for each independent draw.
                        # Sampling temperature is a declared experimental difference,
                        # NOT proof that token/call budgets have been matched.
                        try:
                            if condition == "llm_single_ensemble":
                                client.seed = seed + len(client.calls) - start_calls
                                client.temperature = 0.7
                            return crewai_assessment(client, role, f, prior)
                        finally:
                            client.seed = seed
                            client.temperature = 0.0

                    assessment = predict_condition(features, condition, assess)
                    if len(client.calls) - start_calls != expected_calls:
                        raise ValueError(
                            "Actual calls did not match the registered inference budget"
                        )
                    write_json(cache, assessment)
                record.update({"p_make": assessment["p_make"], "status": "ok"})
            except Exception as error:
                record["error"] = str(error)
            calls = client.calls[start_calls:]
            record.update(
                {
                    "calls": len(calls),
                    "prompt_tokens": sum(c.get("prompt_tokens", 0) for c in calls),
                    "completion_tokens": sum(c.get("completion_tokens", 0) for c in calls),
                    "runtime_seconds": time.perf_counter() - tick,
                }
            )
            with (output / "traces.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "shot_id": row["shot_id"],
                            "condition": condition,
                            "timestamp": utc_now(),
                            "calls": calls,
                            "result": record,
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            predictions.append(record)
            # Persist after each shot/condition; interruptions can resume from the cache.
            pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    frame = pd.DataFrame(predictions)
    metrics = write_scores(frame, output, prefix=split)
    usage = summarize_agent_usage(frame)
    usage.to_csv(output / "usage_by_condition.csv", index=False)
    result = {
        "predictions": str(output / "predictions.csv"),
        "metrics": metrics,
        "usage_report": str(output / "usage_by_condition.csv"),
        "actual_calls_this_run": len(client.calls),
        "completed_at": utc_now(),
    }
    write_json(output / "summary.json", result)
    return result
