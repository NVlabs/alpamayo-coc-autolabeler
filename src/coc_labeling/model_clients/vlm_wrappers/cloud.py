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

"""Cloud-hosted OpenAI wrapper implementation for VLM inference."""

import logging
import time
from typing import Any, Dict, List, Optional

from coc_labeling.model_clients.openai_client import create_cloud_openai_client
from coc_labeling.model_clients.timeout import TIMEOUT_MAX
from coc_labeling.model_clients.vlm_wrappers.common import BaseWrapper, encode_image

logger = logging.getLogger(__name__)

MAX_CLOUD_REQUEST_ATTEMPTS = 5
RETRYABLE_CLOUD_STATUS_CODES = {408, 409, 429}


def _is_gpt5_family_model(model_name: str) -> bool:
    """Return whether a direct or provider-qualified model name is GPT-5-family."""
    model_leaf = str(model_name).lower().replace("_", "-").rsplit("/", 1)[-1]
    return model_leaf.startswith("gpt-5") or model_leaf.startswith("gpt5")


def _cloud_error_status_code(exc: Exception) -> Optional[int]:
    """Extract an HTTP status code from OpenAI-compatible errors."""
    for source in (exc, getattr(exc, "response", None)):
        if source is None:
            continue
        for attr_name in ("status_code", "status"):
            status_code = getattr(source, attr_name, None)
            if isinstance(status_code, int):
                return status_code
            if isinstance(status_code, str) and status_code.isdigit():
                return int(status_code)
    return None


def _is_non_retryable_cloud_error(exc: Exception) -> bool:
    """Return whether a cloud error is a deterministic client-side failure."""
    status_code = _cloud_error_status_code(exc)
    if status_code is None:
        return False
    return 400 <= status_code < 500 and status_code not in RETRYABLE_CLOUD_STATUS_CODES


def _usage_field(usage: Any, field_name: str) -> Any:
    """Read an OpenAI usage field from object or dict payloads."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage.get(field_name)
    return getattr(usage, field_name, None)


def _reasoning_tokens(usage: Any) -> Any:
    """Return hidden reasoning token count when the API includes it."""
    direct_value = _usage_field(usage, "reasoning_tokens")
    if direct_value is not None:
        return direct_value
    token_details = _usage_field(usage, "completion_tokens_details") or _usage_field(
        usage, "output_tokens_details"
    )
    return _usage_field(token_details, "reasoning_tokens")


class OpenAIWrapper(BaseWrapper):
    """Wrapper using OpenAI-compatible cloud VLM endpoints."""

    def __init__(
        self,
        model_name: str,
        init_model: bool = True,
        ip_addr: Optional[str] = None,
        timeout_sec: int = TIMEOUT_MAX,
    ) -> None:
        del init_model, ip_addr
        model_name_map = {
            "gpt5": "gpt-5",
            "gpt5.5": "gpt-5.5",
            "gpt-5": "gpt-5",
            "gpt-5.5": "gpt-5.5",
        }
        self.model_name = model_name_map.get(model_name, model_name)
        self.timeout_sec = timeout_sec
        cloud_client = create_cloud_openai_client()
        self.client = cloud_client.client
        self.client_backend = cloud_client.backend
        logger.info(
            "Using OpenAI-compatible backend '%s' with model '%s'.",
            self.client_backend,
            self.model_name,
        )

    def infer(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        seed: int = 42,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        json_schema: Optional[Any] = None,
    ) -> Dict[str, Any]:
        del top_p, repetition_penalty
        tried_cnt = 0
        max_cnt = MAX_CLOUD_REQUEST_ATTEMPTS
        response = None
        is_gpt5_family = _is_gpt5_family_model(self.model_name)
        # GPT-5-family models require `max_completion_tokens` instead of `max_tokens`.
        token_param_name = "max_completion_tokens" if is_gpt5_family else "max_tokens"
        token_kwargs = {token_param_name: max_tokens}
        extra_kwargs = {}
        # Some GPT-5 deployments only support default sampling behavior.
        if not is_gpt5_family:
            extra_kwargs["temperature"] = temperature
        # Keep seed for non-GPT-5 models to avoid unsupported-parameter errors.
        if not is_gpt5_family:
            extra_kwargs["seed"] = seed
        while response is None and tried_cnt < max_cnt:
            try:
                if json_schema is not None:
                    response = self.client.beta.chat.completions.parse(
                        model=self.model_name,
                        messages=messages,
                        **token_kwargs,
                        **extra_kwargs,
                        response_format=json_schema,
                        timeout=self.timeout_sec,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **token_kwargs,
                        **extra_kwargs,
                        timeout=self.timeout_sec,
                    )
            except Exception as exc:
                if "ResponsibleAIPolicyViolation" in str(exc):
                    logger.warning("Responsible AI Policy Violation.")
                    return {"content": "Responsible AI Policy Violation."}
                elif "token rate limit" in str(exc):
                    logger.warning(
                        "Token rate limit exceeded on attempt %d/%d. Retrying after 60 seconds.",
                        tried_cnt + 1,
                        max_cnt,
                    )
                    time.sleep(60)
                elif "not JSON serializable" in str(exc):
                    raise TypeError("Model messages are not JSON serializable.") from exc
                elif _is_non_retryable_cloud_error(exc):
                    logger.error(
                        "Cloud model request failed with non-retryable HTTP status %d: %s",
                        _cloud_error_status_code(exc),
                        exc,
                    )
                    raise
                else:
                    logger.warning(
                        "Cloud model request failed on attempt %d/%d: %s. "
                        "Retrying after 60 seconds.",
                        tried_cnt + 1,
                        max_cnt,
                        exc,
                    )
                    time.sleep(60)
                tried_cnt += 1

        if response is None:
            raise RuntimeError(f"Failed to get response from the model after {max_cnt} attempts.")

        if json_schema is None:
            output_content = response.choices[0].message.content
        else:
            output_content = response.choices[0].message.parsed
            output_content = output_content.dict()

        return {
            "finish_reason": response.choices[0].finish_reason,
            "content": output_content,
            "prompt_tokens": response.usage.prompt_tokens,
            "response_tokens": response.usage.total_tokens - response.usage.prompt_tokens,
            "reasoning_tokens": _reasoning_tokens(response.usage),
            "system_fingerprint": response.system_fingerprint,
        }

    def add_message(self, role: str, m_type: str, content: Any) -> Dict[str, Any]:
        if role not in ["user", "system", "assistant"]:
            raise ValueError(f"Invalid message role: {role}")
        if m_type not in ["text", "image", "video"]:
            raise ValueError(f"Invalid message type: {m_type}")

        if m_type == "text":
            return {"role": role, "content": [{"type": "text", "text": content}]}
        if m_type == "image":
            return {
                "role": role,
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(content)}",
                            "detail": "high",
                        },
                    },
                ],
            }
        if m_type == "video":
            if not isinstance(content, list):
                raise TypeError("Video content must be a list of frames.")
            message = []
            for f_i in content:
                message.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(f_i)}",
                            "detail": "high",
                        },
                    }
                )
            return {"role": role, "content": message}
        raise ValueError(f"Invalid message type: {m_type}")
