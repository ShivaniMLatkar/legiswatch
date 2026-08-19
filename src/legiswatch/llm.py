"""Provider-agnostic structured-output LLM client.

Design goals, in priority order:

1. **Reproducibility.** The `replay` provider serves recorded responses from
   disk, so a run is deterministic and requires no API key or network access.
   This is what makes the test suite hermetic and CI runs free.
2. **Schema enforcement.** Callers ask for a pydantic model, not a string. On
   validation failure we retry with the error appended (error-driven refinement)
   rather than pushing malformed data downstream.
3. **Swappable inference.** anthropic / openai / ollama behind one interface, so
   the same pipeline can run against a hosted API or fully on-premises. For
   institutional data that cannot leave the environment, Ollama is the escape
   hatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .config import settings
from .logging_setup import get_logger
from .paths import REPLAY_DIR

T = TypeVar("T", bound=BaseModel)
log = get_logger(__name__)


# Rough public list prices, USD per 1M tokens. Used only for the cost estimate
# shown on the dashboard -- directional, not billing-grade.
_PRICING: dict[str, tuple] = {
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "_default": (1.00, 5.00),
}


class LLMError(RuntimeError):
    pass


def _cache_key(system: str, user: str, schema_name: str, model: str) -> str:
    payload = json.dumps(
        {"system": system, "user": user, "schema": schema_name, "model": model},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


class LLMClient:
    """One call surface: `structured(system, user, ResponseModel)`."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
        record: bool = False,
    ) -> None:
        self.provider = provider or settings.provider
        self.temperature = settings.temperature if temperature is None else temperature
        self.max_retries = settings.max_retries if max_retries is None else max_retries
        provider = self.provider
        self.record = record

        self.model = model or {
            "anthropic": "claude-sonnet-4-5",
            "openai": "gpt-4o",
            "ollama": "llama3.1:8b",
            "replay": "replay-cache",
        }.get(provider, "replay-cache")

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.call_count = 0
        self.cache_hits = 0

        self._client: Any = None
        if provider != "replay":
            self._client = self._init_client(provider)

        REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    # -- provider wiring ---------------------------------------------------

    def _init_client(self, provider: str) -> Any:
        if provider == "anthropic":
            try:
                import anthropic
            except ImportError as e:
                raise LLMError("pip install anthropic") from e
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set")
            return anthropic.Anthropic()

        if provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise LLMError("pip install openai") from e
            if not os.getenv("OPENAI_API_KEY"):
                raise LLMError("OPENAI_API_KEY is not set")
            return OpenAI()

        if provider == "ollama":
            try:
                import ollama
            except ImportError as e:
                raise LLMError("pip install ollama") from e
            return ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))

        raise LLMError(f"unknown provider: {provider}")

    # -- public API --------------------------------------------------------

    def structured(
        self,
        system: str,
        user: str,
        response_model: type[T],
        *,
        cache_tag: str | None = None,
    ) -> T:
        """Return a validated instance of `response_model`.

        Retries on validation failure with the error text appended so the model
        can correct itself, up to `max_retries`.
        """
        key = cache_tag or _cache_key(system, user, response_model.__name__, self.model)
        cache_path = REPLAY_DIR / f"{key}.json"

        if self.provider == "replay":
            if not cache_path.exists():
                raise LLMError(
                    f"replay mode: no cached response for key {key} "
                    f"({response_model.__name__}). Run once with a live provider "
                    f"and --record to populate the cache."
                )
            self.cache_hits += 1
            self.call_count += 1
            data = json.loads(cache_path.read_text())
            return response_model.model_validate(data["response"])

        schema = response_model.model_json_schema()
        attempt_user = user
        last_error: str | None = None

        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            raw = self._raw_call(system, attempt_user, schema, response_model.__name__)
            elapsed = (time.perf_counter() - t0) * 1000
            self.call_count += 1

            try:
                parsed = response_model.model_validate_json(raw)
            except (ValidationError, ValueError) as e:
                last_error = str(e)[:1500]
                log.warning(
                    "structured_output_validation_failed",
                    extra={
                        "schema": response_model.__name__,
                        "attempt": attempt,
                        "error": last_error[:300],
                    },
                )
                attempt_user = (
                    f"{user}\n\n---\nYour previous response failed schema "
                    f"validation with this error:\n{last_error}\n\n"
                    f"Return corrected JSON that satisfies the schema exactly."
                )
                continue

            log.info(
                "llm_call_ok",
                extra={
                    "schema": response_model.__name__,
                    "latency_ms": round(elapsed, 1),
                    "attempt": attempt,
                },
            )
            if self.record:
                cache_path.write_text(
                    json.dumps(
                        {
                            "key": key,
                            "schema": response_model.__name__,
                            "model": self.model,
                            "response": parsed.model_dump(mode="json"),
                        },
                        indent=2,
                    )
                )
            return parsed

        raise LLMError(
            f"{response_model.__name__} failed validation after "
            f"{self.max_retries + 1} attempts. Last error: {last_error}"
        )

    # -- raw provider calls ------------------------------------------------

    def _raw_call(self, system: str, user: str, schema: dict, schema_name: str) -> str:
        instruction = (
            f"{user}\n\nRespond with a single JSON object conforming to this "
            f"JSON Schema. Output JSON only -- no prose, no markdown fences.\n"
            f"{json.dumps(schema)}"
        )

        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": instruction}],
            )
            self.prompt_tokens += resp.usage.input_tokens
            self.completion_tokens += resp.usage.output_tokens
            return _strip_fences(resp.content[0].text)

        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ],
            )
            if resp.usage:
                self.prompt_tokens += resp.usage.prompt_tokens
                self.completion_tokens += resp.usage.completion_tokens
            return _strip_fences(resp.choices[0].message.content or "")

        if self.provider == "ollama":
            resp = self._client.chat(
                model=self.model,
                format="json",
                options={"temperature": self.temperature},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": instruction},
                ],
            )
            return _strip_fences(resp["message"]["content"])

        raise LLMError(f"unsupported provider: {self.provider}")

    # -- accounting --------------------------------------------------------

    def estimated_cost_usd(self) -> float:
        in_rate, out_rate = _PRICING.get(self.model, _PRICING["_default"])
        return (
            self.prompt_tokens / 1_000_000 * in_rate + self.completion_tokens / 1_000_000 * out_rate
        )


def _strip_fences(text: str) -> str:
    """Models sometimes wrap JSON in markdown fences despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t[:-3]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()
