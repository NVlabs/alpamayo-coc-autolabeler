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

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coc_labeling.model_clients import chat  # noqa: E402


def _completion_response(content: str = "local response") -> Any:
    """Build a minimal OpenAI-compatible response object for routing tests."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_explicit_local_backend_routes_unknown_qwen_model_to_local_server(
    monkeypatch: Any,
) -> None:
    """Route explicit local Qwen calls to the vLLM server instead of cloud."""
    local_calls: list[dict[str, Any]] = []

    def fake_local_completion(**kwargs: Any) -> Any:
        local_calls.append(kwargs)
        return _completion_response()

    def fake_cloud_completion(**kwargs: Any) -> Any:
        raise AssertionError("Cloud backend should not be called")

    monkeypatch.setattr(chat, "completion_with_backoff_local_server", fake_local_completion)
    monkeypatch.setattr(chat, "completion_with_backoff_cloud", fake_cloud_completion)

    messages = [{"role": "user", "content": "hello"}]
    _, response_message = chat.run_one_round_conversation(
        full_messages=messages,
        model_name="Qwen/Qwen3.5-35B-A3B",
        json_schema={"type": "object"},
        video_kwargs={"fps": [2]},
        ip_addr="10.0.0.5",
        timeout=123,
        chat_template_kwargs={"enable_thinking": False},
        backend="local",
    )

    assert response_message == {"content": "local response", "role": "assistant"}
    assert len(local_calls) == 1
    request = local_calls[0]
    assert request["model"] == "Qwen/Qwen3.5-35B-A3B"
    assert request["ip_addr"] == "10.0.0.5"
    assert request["timeout"] == 123
    assert request["extra_body"] == {
        "guided_json": {"type": "object"},
        "guided_decoding_backend": "outlines",
        "mm_processor_kwargs": {"fps": [2]},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_backend_must_be_explicit() -> None:
    """Prevent unknown model names from silently falling through to cloud."""
    with pytest.raises(ValueError, match="backend must be explicitly set"):
        chat.run_one_round_conversation(
            full_messages=[{"role": "user", "content": "hello"}],
            model_name="Qwen/Qwen3.5-35B-A3B",
        )
