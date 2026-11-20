"""OpenAI-compatible chat client over an endpoint registry.

Registry ``r*.json`` files declare ready endpoints, models, credential variables and
optional request profiles. The client resolves the configured records for each call.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import requests

from .inference_key_state import inference_quota_depleted

# Output budget below which shrinking is worse than failing: the 8192 default exists because
# 2048 truncated the observation JSON mid-list on ~10% of clips, so a fitted budget under this
# floor buys a parse error instead of a 400.
_MIN_FITTED_TOKENS = 1024


def _validate_token_pricing(pricing: dict) -> None:
    """Require explicit positive finite operator rates, expressed in USD per million tokens."""
    if not isinstance(pricing, dict) or any(
        type(pricing.get(name)) not in (int, float)
        or not math.isfinite(pricing[name])
        or pricing[name] <= 0
        for name in ("input_usd_per_million_tokens", "output_usd_per_million_tokens")
    ):
        raise ValueError("token pricing requires positive finite input and output rates")


def estimate_usage_cost(usage: dict | None, pricing: dict) -> float | None:
    """Estimate from provider token counts; cached input receives no assumed discount.

    Completion usage already includes reasoning tokens. Missing or malformed usage remains
    unknown; configured rates are estimates, not provider-confirmed charges or bounds.
    """
    _validate_token_pricing(pricing)
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if (
        type(prompt) is not int
        or not 0 < prompt <= 2**63 - 1
        or type(completion) is not int
        or not 0 <= completion <= 2**63 - 1
    ):
        return None
    if "total_tokens" in usage and (
        type(usage["total_tokens"]) is not int or usage["total_tokens"] != prompt + completion
    ):
        return None
    amount = (
        prompt * pricing["input_usd_per_million_tokens"]
        + completion * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000
    return amount if math.isfinite(amount) and amount > 0 else None


def ready_endpoints(registry_dir: str | Path) -> list[dict]:
    """Load ready records and validate explicit public profiles before run allocation.

    No provider request is made. Unprofiled records retain their existing validation
    at dispatch; public profiles require configured rates and a nonblank credential.
    """
    records = [json.loads(entry.read_text()) for entry in Path(registry_dir).glob("r*.json")]
    ready = [record for record in records if record.get("status") == "ready"]
    ready.sort(key=lambda record: record.get("updated", ""), reverse=True)
    if not ready:
        raise RuntimeError(f"no ready endpoint in {registry_dir}")
    for record in ready:
        profile = record.get("request_profile")
        if profile is None:
            continue
        if profile != "openai_reasoning":
            raise ValueError(f"unknown request profile: {profile}")
        effort = record.get("reasoning_effort")
        if not isinstance(effort, str) or effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("openai_reasoning requires a supported reasoning effort")
        key_env = record.get("api_key_env")
        if not isinstance(key_env, str) or not key_env.strip():
            raise ValueError("openai_reasoning requires an explicit credential variable")
        _validate_token_pricing(record.get("pricing"))
        if not os.environ.get(key_env, "").strip():
            raise ValueError(f"registry record needs ${key_env} but it is not set")
    return ready


def _fitted_max_tokens(body: str, requested: int) -> int | None:
    """The output budget that fits, read off a context-length 400, or None.

    vLLM reports the arithmetic it rejected -- context limit and prompt size -- so the
    endpoint itself says how much room is left. Observation prompts vary in size per round
    (temperature 0.7 makes the context non-deterministic), so a clip can sit at the limit and
    fail one attempt while fitting the next; shrinking the ASK to the reported headroom fixes
    it deterministically instead of leaving it to sampling luck.
    """
    if "maximum context length" not in body:
        return None
    limit = re.search(r"maximum context length is (\d+) tokens", body)
    prompt = re.search(r"prompt contains at least (\d+) input tokens", body)
    if not (limit and prompt):
        return None
    # the server says "at least" N input tokens, so leave a little room for its own accounting
    fitted = int(limit.group(1)) - int(prompt.group(1)) - 64
    if fitted < _MIN_FITTED_TOKENS or fitted >= requested:
        return None
    return fitted


class ChatClient:
    """Minimal chat-completions client bound to one endpoint registry."""

    def __init__(
        self,
        registry_dir: str | Path,
        *,
        timeout_s: float = 300.0,
        max_tokens: int = 8192,
        temperature: float = 0.7,
        on_attempt: Callable[[dict], None] | None = None,
        before_attempt: Callable[[], None] | None = None,
        transient_retries: int = 0,
    ) -> None:
        """Bind the registry directory; the endpoint is resolved per call."""
        self.registry_dir = Path(registry_dir)
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.on_attempt = on_attempt
        self.before_attempt = before_attempt
        if not 0 <= transient_retries <= 3:
            raise ValueError("transient retries must be bounded in [0, 3]")
        self.transient_retries = transient_retries

    def _post(self, record: dict, payload: dict, headers: dict) -> requests.Response:
        """Dispatch once, preserving usage even for retries and failed HTTP responses.

        Hooks are optional and do not change the completion contract. The event never
        contains messages, endpoint URLs, headers, response text, or credentials. Costs
        absent from the provider remain unknown unless the registry explicitly supplies
        operator token rates. Estimates never populate the provider-confirmed cost field.
        """
        bearer = headers.get("Authorization", "")
        fingerprint = None
        if bearer.startswith("Bearer "):
            from .inference_key_state import (
                mark_inference_key_exhausted,
                require_non_exhausted_inference_key,
                resolve_inference_state_dir,
            )

            fingerprint = require_non_exhausted_inference_key(bearer.removeprefix("Bearer "))
        if self.before_attempt is not None:
            self.before_attempt()
        started = time.monotonic()
        event = {
            "attempt_id": uuid.uuid4().hex,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "model": record["model"],
            "key_fingerprint": fingerprint,
            "status_code": None,
            "usage": None,
            "cost_usd": None,
            "error_type": None,
            "max_tokens_requested": payload.get("max_tokens", payload.get("max_completion_tokens")),
            "temperature": payload.get("temperature"),
        }
        url = f"{record['url']}/v1/chat/completions"
        if record.get("request_profile") == "openai_reasoning":
            base = record["url"].rstrip("/")
            url = base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
            event["status_code"] = response.status_code
            if self.on_attempt is not None:
                try:
                    body = response.json()
                except ValueError:
                    body = {}
                if isinstance(body, dict):
                    event["usage"] = body.get("usage")
                else:
                    body = {}
                billed = response.headers.get("x-litellm-response-cost")
                if billed is not None:
                    try:
                        amount = float(billed)
                    except ValueError:
                        amount = float("nan")
                    if math.isfinite(amount) and amount >= 0:
                        event["cost_usd"] = amount
                if (
                    record.get("request_profile") == "openai_reasoning"
                    and event["cost_usd"] is None
                    and response.status_code == 200
                ):
                    event["token_pricing"] = dict(record["pricing"])
                    event["cost_estimate_policy"] = "configured_token_rates_v1"
                    event["service_tier"] = body.get("service_tier")
                    event["estimated_cost_usd"] = (
                        estimate_usage_cost(event["usage"], record["pricing"])
                        if body.get("service_tier") in (None, "default")
                        else None
                    )
            if (
                response.status_code == 429
                and fingerprint is not None
                and inference_quota_depleted(response.status_code, response.content)
            ):
                if os.environ.get("ALPAMAYO_INFERENCE_STATE_DIR", "").strip():
                    mark_inference_key_exhausted(resolve_inference_state_dir(), fingerprint)
                raise RuntimeError(f"exhausted inference key {fingerprint} is blocked")
            return response
        except requests.RequestException as error:
            event["error_type"] = type(error).__name__
            raise
        finally:
            if self.on_attempt is not None:
                event["elapsed_s"] = time.monotonic() - started
                self.on_attempt(event)

    def _candidates(self) -> list[dict]:
        """Ready endpoint records, round-robin rotated per call.

        Every ready replica shares the load (one model, many servers); the rotation
        offset advances per call so consecutive requests spread across the fleet, and
        the remaining candidates stay in order as failovers. Registry entries can go
        stale when a serving job is preempted and re-registers elsewhere, so callers
        try candidates in order. A record may carry ``api_key_env`` (hosted gateway:
        bearer auth from that env var, vLLM-only kwargs omitted).
        """
        ready = ready_endpoints(self.registry_dir)
        self._rotation = getattr(self, "_rotation", -1) + 1
        offset = self._rotation % len(ready)
        return ready[offset:] + ready[:offset]

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        response_schema: dict | None = None,
    ) -> str:
        """One chat completion; returns the assistant text.

        A response schema opts into schema-locked chat with max_completion_tokens and
        omitted temperature unless explicitly supplied. Ordinary calls are unchanged.
        An explicit OpenAI reasoning profile uses its supported parameters for either mode.
        Thinking is disabled via ``chat_template_kwargs`` (the only switch Qwen honors); if
        the serving stack still routes text into ``reasoning_content``, that is returned as a
        fallback rather than ``None``. Stale endpoints (preempted jobs whose registry entry
        still says ready) are skipped by falling through to the next candidate.
        """
        last_error: Exception | None = None
        for record in self._candidates():
            key_env = record.get("api_key_env")
            headers = {}
            payload = {
                "model": record["model"],
                "messages": messages,
                # a record-pinned temperature is an endpoint constraint (e.g. the hosted
                # gpt-5.5 gateway 400s on anything but 1.0) and overrides the caller's
                "temperature": record.get(
                    "temperature", self.temperature if temperature is None else temperature
                ),
                # whole-clip observation rounds list many components; 2048
                # truncated the JSON mid-list on ~10% of clips (default 8192)
                "max_tokens": self.max_tokens,
            }
            if response_schema is not None:
                payload["max_completion_tokens"] = payload.pop("max_tokens")
                payload.pop("temperature")
                if temperature is not None:
                    payload["temperature"] = temperature
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "reply", "schema": response_schema, "strict": True},
                }
            if record.get("request_profile") == "openai_reasoning":
                if "max_tokens" in payload:
                    payload["max_completion_tokens"] = payload.pop("max_tokens")
                payload.pop("temperature", None)
                payload["reasoning_effort"] = record["reasoning_effort"]
                payload["service_tier"] = "default"
            if key_env:
                key = os.environ.get(key_env)
                if not key:
                    raise RuntimeError(f"registry record needs ${key_env} but it is not set")
                headers["Authorization"] = f"Bearer {key}"
            else:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                for retry in range(self.transient_retries + 1):
                    try:
                        response = self._post(record, payload, headers)
                    except requests.ReadTimeout:
                        if retry == self.transient_retries:
                            raise
                        time.sleep(2.0 ** (retry + 1))
                        continue
                    if (
                        response.status_code not in {429, 500, 502, 503, 504}
                        or retry == self.transient_retries
                    ):
                        break
                    # Each retry traverses the same admission and accounting hooks.
                    delay = response.headers.get("Retry-After", "")
                    time.sleep(min(30.0, float(delay)) if delay.isdigit() else 2.0 ** (retry + 1))
                if response.status_code >= 400:
                    # a context-length 400 is recoverable: the body reports the headroom, so
                    # re-ask the SAME endpoint for a budget that fits rather than losing the clip
                    token_key = (
                        "max_completion_tokens"
                        if "max_completion_tokens" in payload
                        else "max_tokens"
                    )
                    fitted = _fitted_max_tokens(response.text, payload[token_key])
                    if fitted is not None:
                        response = self._post(record, {**payload, token_key: fitted}, headers)
                if response.status_code >= 400:
                    # the body names the cause (context overflow, bad schema);
                    # raise_for_status alone hides it
                    raise RuntimeError(
                        f"{response.status_code} from {record['url']}: {response.text[:500]}"
                    )
            except requests.ConnectionError as error:
                last_error = error
                continue
            message = response.json()["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content")
            if not content:
                raise RuntimeError(f"empty completion from {record['url']}")
            return content
        raise RuntimeError(f"every ready endpoint unreachable: {last_error}")
