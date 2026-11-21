"""OpenAI Chat Completions client with the same interface as ``OllamaClient``.

``crewai_assessment`` and the cached runner only need ``.model``, ``.seed``,
``.calls`` and ``.call(messages) -> {"p_make", "rationale"}``. Kept outside
``src/`` so the frozen ``agents.py`` hash is unchanged.

The API key is read from ``OPENAI_API_KEY`` (loaded from the gitignored
``.env``) and never logged.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from nba_shots.agents import parse_assessment

# USD per 1M tokens (developers.openai.com/api/docs/models/gpt-6-luna, 2026-11-20)
PRICES = {"gpt-6-luna": {"input": 0.10, "cached": 0.01, "output": 0.50}}

STRICT_SCHEMA = {
    "type": "object",
    "properties": {
        "p_make": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["p_make", "rationale"],
    "additionalProperties": False,
}


def load_dotenv(path: str | Path = ".env") -> None:
    """Minimal KEY=VALUE loader; existing environment variables win."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class BudgetExceeded(RuntimeError):
    pass


class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-6-luna",
        reasoning_effort: str = "none",
        seed: int = 42,
        max_cost_usd: float = 10.0,
        sampling_params: bool = True,
        rpm: float = 45,
    ):
        from openai import OpenAI

        load_dotenv()
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set (add it to .env)")
        if model not in PRICES:
            raise ValueError(f"No price table for {model!r}")
        self.api = OpenAI(max_retries=0)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.max_cost_usd = max_cost_usd
        # temperature/seed are only sent with effort "none" (reasoning modes reject them)
        self.sampling_params = sampling_params and reasoning_effort == "none"
        self.cache_id = f"openai:{model}:effort={reasoning_effort}"
        self.digest = None
        self.calls = []
        self.cost_usd = 0.0
        self.tokens = {"prompt": 0, "cached": 0, "completion": 0, "reasoning": 0}
        self.min_interval = 60.0 / rpm  # client-side throttle under the account RPM limit
        self._next_slot = 0.0
        self._slot_lock = threading.Lock()
        self.local = threading.local()
        self.fatal = None  # set on quota/daily-limit errors; the runner stops instead of retrying

    def _throttle(self):
        with self._slot_lock:
            now = time.monotonic()
            wait = self._next_slot - now
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            time.sleep(wait)

    def _request(self, messages):
        self._throttle()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "assessment", "strict": True, "schema": STRICT_SCHEMA},
            },
        }
        if self.sampling_params:
            kwargs.update(temperature=0, seed=self.seed)
        return self.api.chat.completions.create(**kwargs)

    def _charge(self, usage) -> dict:
        prompt = usage.prompt_tokens or 0
        completion = usage.completion_tokens or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
        cdetails = getattr(usage, "completion_tokens_details", None)
        reasoning = (getattr(cdetails, "reasoning_tokens", 0) or 0) if cdetails else 0
        price = PRICES[self.model]
        cost = (
            (prompt - cached) * price["input"]
            + cached * price["cached"]
            + completion * price["output"]
        ) / 1e6
        self.cost_usd += cost
        for key, value in [
            ("prompt", prompt),
            ("cached", cached),
            ("completion", completion),
            ("reasoning", reasoning),
        ]:
            self.tokens[key] += value
        return {
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "completion_tokens": completion,
            "reasoning_tokens": reasoning,
            "cost_usd": cost,
        }

    def call(self, messages):
        import openai

        if self.fatal:
            raise BudgetExceeded(self.fatal)
        if self.cost_usd >= self.max_cost_usd:
            self.fatal = f"Spent ${self.cost_usd:.2f} >= cap ${self.max_cost_usd:.2f}"
            raise BudgetExceeded(f"Spent ${self.cost_usd:.2f} >= cap ${self.max_cost_usd:.2f}")
        started = time.perf_counter()
        log = {"status": "error", "p_make": None, "reasoning_effort": self.reasoning_effort}
        self.calls.append(log)
        self.local.last = log  # per-thread, so concurrent callers read their own usage
        try:
            for attempt in range(4):
                try:
                    response = self._request(messages)
                    break
                except (
                    openai.RateLimitError,
                    openai.APIConnectionError,
                    openai.InternalServerError,
                ) as err:
                    text = str(err)
                    if "per day" in text or "insufficient_quota" in text:
                        self.fatal = f"OpenAI quota/daily limit reached: {text[:200]}"
                        raise
                    if attempt == 3:
                        raise
                    time.sleep(2 ** (attempt + 1))
            log.update(self._charge(response.usage))
            raw = response.choices[0].message.content
            log["raw_response"] = raw
            assessment = parse_assessment(raw)
            log.update(status="ok", **assessment)
            return assessment
        except Exception as error:
            log["error"] = str(error)
            raise
        finally:
            log["runtime_seconds"] = time.perf_counter() - started
