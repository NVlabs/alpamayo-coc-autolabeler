# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.modules.setdefault("cv2", ModuleType("cv2"))
sys.modules.setdefault("torch", ModuleType("torch"))

from coc_labeling.model_clients.vlm_wrappers import cloud  # noqa: E402


class _FakeResponse:
    """Minimal chat-completions response used by cloud wrapper tests."""

    choices = [
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="ok"),
        )
    ]
    usage = SimpleNamespace(
        prompt_tokens=3,
        total_tokens=5,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=1),
    )
    system_fingerprint = "fp-test"


class _RecordingCompletions:
    """Record request kwargs and return a small successful response."""

    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.kwargs = kwargs
        return _FakeResponse()


class _StatusError(Exception):
    """Exception carrying an OpenAI-style HTTP status code."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FailingCompletions:
    """Raise configured errors before returning a successful response."""

    def __init__(self, errors: list[Exception]) -> None:
        self.errors = errors
        self.calls = 0

    def create(self, **kwargs: Any) -> _FakeResponse:
        del kwargs
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return _FakeResponse()


def test_openai_wrapper_logs_selected_backend_and_model(
    caplog: Any,
    monkeypatch: Any,
) -> None:
    """Tell users which OpenAI-compatible backend and model are active."""
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=object(), backend="openai"),
    )

    with caplog.at_level(logging.INFO, logger=cloud.__name__):
        cloud.OpenAIWrapper("gpt5.5")

    assert ("Using OpenAI-compatible backend 'openai' with model 'gpt-5.5'.") in caplog.text


def test_cloud_wrapper_retry_cap_is_bounded_for_release() -> None:
    """Avoid long hangs from repeated transient cloud failures."""
    assert cloud.MAX_CLOUD_REQUEST_ATTEMPTS == 5


def test_cloud_wrapper_returns_usage_tokens(monkeypatch: Any) -> None:
    """Return API token usage for persisted run metadata."""
    completions = _RecordingCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=fake_client, backend="openai"),
    )

    wrapper = cloud.OpenAIWrapper("custom-model")
    response = wrapper.infer(messages=[{"role": "user", "content": "hello"}])

    assert response["prompt_tokens"] == 3
    assert response["response_tokens"] == 2
    assert response["reasoning_tokens"] == 1


def test_public_gpt55_uses_supported_reasoning_request_shape(
    monkeypatch: Any,
) -> None:
    """Use modern token limits and omit sampling controls for public GPT-5.5."""
    completions = _RecordingCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=fake_client, backend="openai"),
    )

    wrapper = cloud.OpenAIWrapper("gpt5.5")
    wrapper.infer(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=123,
        seed=7,
        temperature=0.0,
    )

    assert wrapper.model_name == "gpt-5.5"
    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "gpt-5.5"
    assert completions.kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in completions.kwargs
    assert "temperature" not in completions.kwargs
    assert "seed" not in completions.kwargs


def test_non_gpt5_model_keeps_legacy_request_shape(monkeypatch: Any) -> None:
    """Keep non-GPT-5 requests deterministic with max_tokens, seed, and temperature."""
    completions = _RecordingCompletions()
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=fake_client, backend="openai"),
    )

    wrapper = cloud.OpenAIWrapper("custom-model")
    wrapper.infer(
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=123,
        seed=7,
        temperature=0.0,
    )

    assert completions.kwargs is not None
    assert completions.kwargs["model"] == "custom-model"
    assert completions.kwargs["max_tokens"] == 123
    assert completions.kwargs["temperature"] == 0.0
    assert completions.kwargs["seed"] == 7
    assert "max_completion_tokens" not in completions.kwargs


def test_cloud_wrapper_does_not_retry_non_retryable_400(
    monkeypatch: Any,
) -> None:
    """Fail fast on deterministic client-side request errors."""
    completions = _FailingCompletions(
        [_StatusError("Unsupported value: temperature", status_code=400)]
    )
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=fake_client, backend="openai"),
    )
    monkeypatch.setattr(
        cloud.time,
        "sleep",
        lambda seconds: pytest.fail(f"unexpected retry sleep: {seconds}"),
    )

    wrapper = cloud.OpenAIWrapper("custom-model")
    with pytest.raises(_StatusError, match="Unsupported value"):
        wrapper.infer(messages=[{"role": "user", "content": "hello"}])

    assert completions.calls == 1


def test_cloud_wrapper_still_retries_transient_500(monkeypatch: Any) -> None:
    """Keep retry behavior for transient server-side errors."""
    completions = _FailingCompletions([_StatusError("server unavailable", status_code=500)])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    sleep_calls = []
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=fake_client, backend="openai"),
    )
    monkeypatch.setattr(cloud.time, "sleep", sleep_calls.append)

    wrapper = cloud.OpenAIWrapper("custom-model")
    response = wrapper.infer(messages=[{"role": "user", "content": "hello"}])

    assert response["content"] == "ok"
    assert completions.calls == 2
    assert sleep_calls == [60]


@pytest.mark.parametrize("alias,model", [("gpt5", "gpt-5"), ("gpt5.5", "gpt-5.5")])
def test_cloud_aliases_resolve_to_public_model_ids(monkeypatch, alias, model):
    """The public client must not insert provider-specific routing prefixes."""
    monkeypatch.setattr(
        cloud,
        "create_cloud_openai_client",
        lambda: SimpleNamespace(client=object(), backend="openai"),
    )
    assert cloud.OpenAIWrapper(alias).model_name == model
