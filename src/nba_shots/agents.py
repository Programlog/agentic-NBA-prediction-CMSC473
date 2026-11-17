"""A local-only, cached five-condition agent experiment with strict budgets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import uuid
from functools import wraps
from inspect import signature
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from filelock import FileLock, Timeout

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

PROMPT_VERSION = "release-v4-feature-semantics"

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "p_make": {"type": "number", "minimum": 0, "maximum": 1},
        "rationale": {"type": "string"},
    },
    "required": ["p_make", "rationale"],
    "additionalProperties": False,
}

# Maximum calls per shot. Empty specialists are skipped.
CONDITIONS = {
    "llm_single": 1,
    "llm_single_budget": 4,
    "llm_single_decomposed": 1,
    "llm_single_ensemble": 4,
    "llm_multi": 4,
}

DECOMPOSITION_ROLE = (
    "Single analyst: in ONE assessment, separately consider game context, "
    "player history, and shot mechanics/defense; ignore missing groups, "
    "then combine the available evidence into one probability. "
    "Use only supplied features; do not invent outside information"
)

FEATURE_DEFINITIONS = {
    "shot_distance": "Shot distance in feet.",
    "defender_distance": "Distance to the closest defender in feet.",
    "game_clock_seconds": "Seconds remaining in the current game period.",
    "shot_clock": (
        "Seconds remaining on the possession shot clock; a higher value means more time remains."
    ),
    "period": "Game period number, not seconds remaining.",
    "dribbles": "Number of dribbles before the shot attempt.",
    "touch_time": "Seconds the player held the ball before the attempt.",
    "shot_value": (
        "Points awarded if the shot is made (2 or 3); not the importance of the attempt."
    ),
    "location": (
        "The shooter's team venue: H means home and A means away; "
        "not the shooter's position on the court."
    ),
    "player_prior_attempts": ("Number of recorded shot attempts by this player before this shot."),
    "player_prior_make_rate": (
        "Fraction of the player's earlier recorded shots that were made; "
        "not the current shot's outcome."
    ),
    "elbow_angle_deg": ("Estimated elbow joint angle in degrees from 2D image landmarks."),
    "wrist_height_torso": (
        "2D wrist-height proxy normalized by torso size; not physical release height."
    ),
    "defender_distance_torso": (
        "2D defender-distance proxy normalized by torso size; not physical distance in feet."
    ),
    "defender_wrist_height_torso": (
        "2D defender wrist-height proxy normalized by torso size; not physical height."
    ),
}

if set(FEATURE_DEFINITIONS) != ALLOWED_FEATURES:
    raise ValueError("LLM feature definitions must match the allowed features")

FEATURE_GUIDANCE = (
    "Interpret supplied measurements using these definitions: "
    + "; ".join(
        f"{name}: {description}" for name, description in sorted(FEATURE_DEFINITIONS.items())
    )
    + " Missing or null measurements are unknown, not zero."
)

INSTRUCTIONS = (
    "Estimate the probability an NBA field goal is made at the instant of release. "
    "Only use the supplied pre-release measurements; null means unknown. "
    "The supplied measurements are enough to estimate: judge how each one makes this shot "
    "easier or harder, and let p_make reflect that difficulty. "
    "Do not infer player identities, retrieve facts, or assume an outcome. "
    "Any CV measurements are 2D image proxies, not physical release/entry angles or heights. "
    "Return JSON with p_make (a number between 0 and 1) and a short rationale. " + FEATURE_GUIDANCE
)

EXPECTED_OUTPUT = "JSON with your numeric p_make estimate and a brief evidence rationale"


def prompt_features(row: dict) -> dict:
    """Exclude outcomes, identities, filenames, and post-shot information."""
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

    return {
        "p_make": float(p),
        "rationale": value["rationale"],
    }


class OllamaClient:
    def __init__(
        self,
        model,
        base_url="http://127.0.0.1:11434",
        seed=42,
        num_predict=512,
    ):
        parsed = urlparse(base_url)

        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Only a local Ollama HTTP endpoint is allowed; no paid fallback")

        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Ollama URL must be a plain local endpoint")

        if num_predict < 32 or num_predict > 1024:
            raise ValueError("num_predict must be between 32 and 1024")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.num_predict = num_predict
        self.temperature = 0.0

        self.session = requests.Session()
        self.session.trust_env = False
        self.calls = []

        response = self.session.get(
            self.base_url + "/api/tags",
            timeout=10,
        )
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

            log.update(
                {
                    "status": "ok",
                    **assessment,
                }
            )

            return assessment

        except Exception as error:
            log["error"] = str(error)
            raise

        finally:
            log["runtime_seconds"] = time.perf_counter() - started


def crewai_assessment(
    client: OllamaClient,
    role: str,
    features: dict,
    previous: list,
) -> dict:
    """Use CrewAI with a strict one-call local Ollama adapter."""
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["CREWAI_TRACING_ENABLED"] = "false"

    os.environ.setdefault(
        "CREWAI_STORAGE_DIR",
        str(Path("artifacts/crewai_state").resolve()),
    )

    from crewai_core.auth import token as platform_auth

    def no_platform_account():
        raise platform_auth.AuthError("Platform access disabled for this local-only experiment")

    with patch.object(
        platform_auth,
        "get_auth_token",
        no_platform_account,
    ):
        from crewai import Agent, BaseLLM, Crew, Task

    class OneCallLocalLLM(BaseLLM):
        def __init__(self):
            super().__init__(
                model=client.model,
                temperature=0,
            )
            self.used = False

        def call(
            self,
            messages,
            tools=None,
            callbacks=None,
            available_functions=None,
            **kwargs,
        ):
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
        backstory=("You assess supplied measurements without outside information."),
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
        description=(
            INSTRUCTIONS
            + "\n"
            + json.dumps(
                {
                    "features": features,
                    "previous_assessments": previous,
                },
                allow_nan=False,
            )
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


def specialist_inputs(
    features: dict,
) -> list[tuple[str, dict]]:
    """Skip specialists with no known inputs."""
    mechanics = CV_FEATURES | {
        "shot_distance",
        "defender_distance",
        "dribbles",
        "touch_time",
    }

    specialists = []

    for role, keys in [
        (
            "Game context analyst",
            ALLOWED_FEATURES - mechanics - HISTORY_FEATURES,
        ),
        ("Player history analyst", HISTORY_FEATURES),
        ("Shot mechanics and defense analyst", mechanics),
    ]:
        subset = {k: v for k, v in features.items() if k in keys}

        if any(v is not None for v in subset.values()):
            specialists.append((role, subset))

    return specialists


def call_budget(
    features: dict,
    condition: str,
) -> int:
    """Return the expected calls for this condition and shot."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown agent condition: {condition}")

    if condition in {
        "llm_single",
        "llm_single_decomposed",
    }:
        return 1

    return len(specialist_inputs(features)) + 1


def predict_condition(
    features: dict,
    condition: str,
    assessor,
) -> dict:
    """Run one experimental condition with only allowed features."""
    if set(features) - ALLOWED_FEATURES:
        raise ValueError("Agent inputs must contain only allowlisted pre-release features")

    if condition == "llm_single":
        return assessor(
            "Shot probability analyst",
            features,
            [],
        )

    if condition == "llm_single_decomposed":
        return assessor(
            DECOMPOSITION_ROLE,
            features,
            [],
        )

    if condition == "llm_single_ensemble":
        results = [
            assessor(
                "Independent shot probability analyst",
                features,
                [],
            )
            for _ in range(call_budget(features, condition))
        ]

        return {
            "p_make": float(np.mean([result["p_make"] for result in results])),
            "rationale": ("Mean of independent probability estimates (no shared assessments)"),
        }

    if condition == "llm_single_budget":
        previous = []

        for _ in range(call_budget(features, condition)):
            result = assessor(
                "Shot probability analyst",
                features,
                previous,
            )
            previous.append(result)

        return result

    if condition == "llm_multi":
        opinions = [
            {
                "role": role,
                **assessor(role, subset, []),
            }
            for role, subset in specialist_inputs(features)
        ]

        return assessor(
            "Shot probability aggregator",
            features,
            opinions,
        )

    raise ValueError(f"Unknown agent condition: {condition}")


def select_evaluation_rows(
    shots: pd.DataFrame,
    split: str,
    limit: int,
    seed: int = 42,
    games: int | None = None,
) -> pd.DataFrame:
    """Select shots reproducibly, optionally balancing across games."""
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("Shot limit must be an integer")

    if not 1 <= limit <= 500:
        raise ValueError("Use between 1 and 500 LLM evaluation shots")

    if split not in {"validation", "test"}:
        raise ValueError("Only validation/test shots can be selected")

    required = {"shot_id", "game_id", "cohort", "split"}

    if missing := required - set(shots.columns):
        raise ValueError(f"Missing sampling columns: {sorted(missing)}")

    rows = shots.loc[shots["split"].eq(split)].copy()

    if rows.empty:
        raise ValueError("No shots in the requested split")

    if rows[list(required)].isna().any().any():
        raise ValueError("Sampling metadata cannot be missing")

    if rows["shot_id"].duplicated().any():
        raise ValueError("Shot IDs must be unique")

    if rows["cohort"].nunique() != 1:
        raise ValueError("Cannot mix dataset cohorts")

    if len(rows) < limit:
        raise ValueError(f"Requested {limit} shots but only {len(rows)} are available")

    if games is None:
        rows = rows.sort_values("shot_id")

        ranking = rows["shot_id"].map(
            lambda shot: hashlib.sha256(f"{seed}:{shot}".encode()).hexdigest()
        )

        return rows.loc[ranking.sort_values().head(limit).index].sort_values("shot_id")

    if isinstance(games, bool) or not isinstance(games, int):
        raise ValueError("Number of games must be an integer")

    if not 2 <= games <= limit:
        raise ValueError("Number of games must be between 2 and the shot limit")

    def rank(kind: str, value: str) -> str:
        return hashlib.sha256(f"{seed}:{kind}:{value}".encode()).hexdigest()

    base_count, remainder = divmod(limit, games)
    max_per_game = base_count + (remainder > 0)

    game_sizes = rows.groupby("game_id").size()

    eligible_games = [game for game, count in game_sizes.items() if count >= max_per_game]

    if len(eligible_games) < games:
        raise ValueError(
            f"Need {games} games with at least "
            f"{max_per_game} shots each; "
            f"only {len(eligible_games)} qualify"
        )

    selected_games = sorted(
        eligible_games,
        key=lambda game: (
            rank("game", str(game)),
            str(game),
        ),
    )[:games]

    selected_parts = []

    for position, game in enumerate(selected_games):
        allocation = base_count + (position < remainder)
        game_rows = rows.loc[rows["game_id"].eq(game)].copy()

        game_rows["_sampling_rank"] = game_rows["shot_id"].map(lambda shot: rank("shot", str(shot)))

        selected_parts.append(game_rows.sort_values(["_sampling_rank", "shot_id"]).head(allocation))

    selected = pd.concat(selected_parts, ignore_index=True)
    counts = selected.groupby("game_id").size()

    if len(selected) != limit or len(counts) != games or counts.max() - counts.min() > 1:
        raise ValueError("Balanced sampling violated shot/game allocation")

    return selected.drop(columns="_sampling_rank").sort_values("shot_id").reset_index(drop=True)


def _usage_number(
    value,
    label,
    *,
    integer=False,
):
    """Reject missing, negative, or invalid usage measurements."""
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{label}: missing or invalid usage measurement")

    number = float(value)

    if not np.isfinite(number) or number < 0 or (integer and not number.is_integer()):
        raise ValueError(f"{label}: invalid nonnegative usage measurement")

    return int(number) if integer else number


def _original_usage(
    calls,
    runtime,
):
    """Capture usage from a fresh successful inference."""
    return {
        "calls": len(calls),
        "prompt_tokens": sum(
            _usage_number(
                call["prompt_tokens"],
                "prompt_tokens",
                integer=True,
            )
            for call in calls
        ),
        "completion_tokens": sum(
            _usage_number(
                call["completion_tokens"],
                "completion_tokens",
                integer=True,
            )
            for call in calls
        ),
        "runtime_seconds": _usage_number(
            runtime,
            "runtime_seconds",
        ),
    }


def _verify_cache_usage(
    usage,
    expected_calls,
):
    """Validate stored original usage against the call budget."""
    if not isinstance(usage, dict):
        raise ValueError("Cache original usage must be an object")

    values = {
        "calls": _usage_number(
            usage.get("calls"),
            "original calls",
            integer=True,
        ),
        "prompt_tokens": _usage_number(
            usage.get("prompt_tokens"),
            "original prompt tokens",
            integer=True,
        ),
        "completion_tokens": _usage_number(
            usage.get("completion_tokens"),
            "original completion tokens",
            integer=True,
        ),
        "runtime_seconds": _usage_number(
            usage.get("runtime_seconds"),
            "original runtime",
        ),
    }

    if values["calls"] != expected_calls:
        raise ValueError("Cached original call count differs from registered budget")

    return values


def _load_assessment_cache(
    path,
    key,
    identity,
    expected_calls,
):
    """Read version-2 cache or legacy assessment-only cache."""
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError("Cache entry must be an object")

    if "schema_version" not in value:
        if set(value) != {
            "p_make",
            "rationale",
        }:
            raise ValueError("Invalid legacy cache structure")

        return (
            parse_assessment(json.dumps(value)),
            None,
        )

    if value.get("schema_version") != 2:
        raise ValueError("Unknown cache schema version")

    if value.get("cache_key_sha256") != key or value.get("identity") != identity:
        raise ValueError("Cache identity does not match registered experiment")

    assessment = parse_assessment(json.dumps(value.get("assessment")))

    return assessment, _verify_cache_usage(
        value.get("original_usage"),
        expected_calls,
    )


def _archive_previous_execution(output):
    """Preserve previous run records before creating new execution files."""
    filenames = [
        "experiment.json",
        "selection.csv",
        "predictions.csv",
        "traces.jsonl",
        "summary.json",
        "usage_by_condition.csv",
    ]

    available = [name for name in filenames if (output / name).is_file()]

    if not available:
        return

    archive = output / "execution_history" / uuid.uuid4().hex
    archive.mkdir(parents=True)

    for name in available:
        shutil.copy2(
            output / name,
            archive / name,
        )

    for name in (
        "traces.jsonl",
        "predictions.csv",
        "usage_by_condition.csv",
        "summary.json",
    ):
        (output / name).unlink(missing_ok=True)


def summarize_agent_usage(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize current-run inference, including failures."""
    required = {
        "model",
        "status",
        "cached",
        "calls",
        "prompt_tokens",
        "completion_tokens",
        "runtime_seconds",
    }

    if missing := required - set(predictions):
        raise ValueError(f"Usage report missing columns: {sorted(missing)}")

    records = []

    for name, rows in predictions.groupby(
        "model",
        sort=True,
    ):
        ok = rows["status"].eq("ok")

        records.append(
            {
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
            }
        )

    return pd.DataFrame.from_records(records)


def exclusive_experiment_output(func):
    """Allow only one execution per experiment output directory."""

    @wraps(func)
    def guarded(*args, **kwargs):
        bound = signature(func).bind(*args, **kwargs)
        output = Path(bound.arguments["output"]).resolve()

        # Keep the lock outside the experiment artifact directory.
        output.parent.mkdir(parents=True, exist_ok=True)
        lock_path = output.with_name(output.name + ".run.lock")

        lock = FileLock(str(lock_path), timeout=0)

        try:
            with lock:
                return func(*args, **kwargs)
        except Timeout as exc:
            raise ValueError(
                f"Experiment output is already locked by another execution: {output}"
            ) from exc

    return guarded


@exclusive_experiment_output
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
    games=None,
) -> dict:
    if split == "test" and not final:
        raise ValueError("LLM test evaluation requires --final after prompt freeze")

    if split not in {
        "validation",
        "test",
    }:
        raise ValueError("LLM evaluation is limited to validation/test")

    shots = pd.read_parquet(shots_path)
    validate_splits(shots)

    if cv_path:
        cv = pd.read_parquet(cv_path)

        if set(shots.shot_id) != set(cv.shot_id) or cv.shot_id.duplicated().any():
            raise ValueError("CV features must cover the same complete registered cohort")

        shots = shots.merge(
            cv[
                [
                    "shot_id",
                    *sorted(CV_FEATURES),
                ]
            ],
            on="shot_id",
            validate="one_to_one",
        )

    selected = select_evaluation_rows(
        shots,
        split,
        limit,
        seed,
        games=games,
    )

    client = OllamaClient(
        model,
        base_url,
        seed,
        num_predict,
    )

    spec = {
        "shots_sha256": sha256(shots_path),
        "cv_sha256": (sha256(cv_path) if cv_path else None),
        "model": model,
        "model_digest": client.digest,
        "prompt_version": PROMPT_VERSION,
        "split": split,
        "shot_ids": selected.shot_id.tolist(),
        "seed": seed,
        "sampling_method": ("legacy_hash_v1" if games is None else "balanced_games_v1"),
        "sampling_games": games,
        "num_predict_per_call": num_predict,
        "conditions": CONDITIONS,
        "ensemble_temperature": 0.7,
        "ensemble_seed_rule": ("base seed plus zero-based call index within shot/arm"),
        "implementation_sha256": sha256(__file__),
    }

    output = Path(output)

    if (output / "experiment.json").exists():
        existing = json.loads((output / "experiment.json").read_text())

        if existing["spec"] != spec:
            raise ValueError("Experiment changed; use a new output directory")

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    _archive_previous_execution(output)

    cache_dir = output / "cache"
    cache_dir.mkdir(exist_ok=True)

    if not (output / "experiment.json").exists():
        write_json(
            output / "experiment.json",
            {
                "spec": spec,
                "environment": environment(),
                "started_at": utc_now(),
            },
        )

    selected[
        [
            "shot_id",
            "game_id",
            "cohort",
            "split",
        ]
    ].to_csv(
        output / "selection.csv",
        index=False,
    )

    (output / "traces.jsonl").write_text(
        "",
        encoding="utf-8",
    )

    predictions = []

    for row in selected.to_dict("records"):
        features = prompt_features(row)

        for condition in CONDITIONS:
            expected_calls = call_budget(
                features,
                condition,
            )

            tick = time.perf_counter()

            cache_input = {
                "spec": spec,
                "shot_id": row["shot_id"],
                "features": features,
                "condition": condition,
            }

            key = hashlib.sha256(
                json.dumps(
                    cache_input,
                    sort_keys=True,
                ).encode()
            ).hexdigest()

            cache = cache_dir / f"{key}.json"

            identity = {
                "shot_id": row["shot_id"],
                "condition": condition,
                "model": model,
                "model_digest": client.digest,
                "prompt_version": PROMPT_VERSION,
                "implementation_sha256": spec["implementation_sha256"],
            }

            record = {
                k: row[k]
                for k in [
                    "shot_id",
                    "game_id",
                    "cohort",
                    "split",
                    "y_true",
                ]
            }

            record.update(
                {
                    "model": condition,
                    "p_make": None,
                    "status": "error",
                    "error": None,
                    "cached": False,
                    "cache_key_sha256": key,
                    "original_cost_provenance": "unknown",
                    "original_calls": None,
                    "original_prompt_tokens": None,
                    "original_completion_tokens": None,
                    "original_runtime_seconds": None,
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                }
            )

            start_calls = len(client.calls)
            original_usage = None

            try:
                if cache.exists():
                    assessment, original_usage = _load_assessment_cache(
                        cache,
                        key,
                        identity,
                        expected_calls,
                    )
                    record["cached"] = True

                else:

                    def assess(role, f, prior):
                        try:
                            if condition == "llm_single_ensemble":
                                client.seed = seed + len(client.calls) - start_calls
                                client.temperature = 0.7

                            return crewai_assessment(
                                client,
                                role,
                                f,
                                prior,
                            )

                        finally:
                            client.seed = seed
                            client.temperature = 0.0

                    assessment = predict_condition(
                        features,
                        condition,
                        assess,
                    )

                    if len(client.calls) - start_calls != expected_calls:
                        raise ValueError(
                            "Actual calls did not match the registered inference budget"
                        )

                record.update(
                    {
                        "p_make": assessment["p_make"],
                        "status": "ok",
                    }
                )

            except Exception as error:
                record["error"] = str(error)

            calls = client.calls[start_calls:]
            runtime = time.perf_counter() - tick

            record.update(
                {
                    "calls": len(calls),
                    "prompt_tokens": sum(call.get("prompt_tokens", 0) for call in calls),
                    "completion_tokens": sum(call.get("completion_tokens", 0) for call in calls),
                    "runtime_seconds": runtime,
                }
            )

            if record["status"] == "ok":
                if not record["cached"]:
                    try:
                        original_usage = _original_usage(
                            calls,
                            runtime,
                        )

                        _verify_cache_usage(
                            original_usage,
                            expected_calls,
                        )

                        write_json(
                            cache,
                            {
                                "schema_version": 2,
                                "cache_key_sha256": key,
                                "identity": identity,
                                "assessment": assessment,
                                "original_usage": original_usage,
                            },
                        )

                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as error:
                        record.update(
                            {
                                "status": "error",
                                "p_make": None,
                                "error": str(error),
                            }
                        )
                        original_usage = None

                if original_usage is not None and record["status"] == "ok":
                    provenance = "cache_v2" if record["cached"] else "fresh_trace"

                    record.update(
                        {
                            "original_cost_provenance": provenance,
                            "original_calls": original_usage["calls"],
                            "original_prompt_tokens": original_usage["prompt_tokens"],
                            "original_completion_tokens": original_usage["completion_tokens"],
                            "original_runtime_seconds": original_usage["runtime_seconds"],
                        }
                    )

            with (output / "traces.jsonl").open(
                "a",
                encoding="utf-8",
            ) as stream:
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

            pd.DataFrame(predictions).to_csv(
                output / "predictions.csv",
                index=False,
            )

    frame = pd.DataFrame(predictions)

    metrics = write_scores(
        frame,
        output,
        prefix=split,
    )

    usage = summarize_agent_usage(frame)
    usage.to_csv(
        output / "usage_by_condition.csv",
        index=False,
    )

    result = {
        "predictions": str(output / "predictions.csv"),
        "metrics": metrics,
        "usage_report": str(output / "usage_by_condition.csv"),
        "actual_calls_this_run": len(client.calls),
        "completed_at": utc_now(),
    }

    write_json(
        output / "summary.json",
        result,
    )

    return result
