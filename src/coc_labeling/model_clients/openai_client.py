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

"""User-configured standard OpenAI-compatible clients."""

import os
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI


@dataclass(frozen=True)
class CloudOpenAIClient:
    """Selected client and public backend metadata."""

    client: Any
    backend: Literal["openai"]


def create_standard_openai_client() -> OpenAI:
    """Create a standard OpenAI API client from public OpenAI environment variables."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        raise ValueError("Please set OPENAI_API_KEY in your environment variables.")

    client_kwargs: dict[str, str] = {"api_key": api_key}
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        client_kwargs["base_url"] = base_url
    organization = os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
    if organization:
        client_kwargs["organization"] = organization
    project = os.environ.get("OPENAI_PROJECT")
    if project:
        client_kwargs["project"] = project
    return OpenAI(**client_kwargs)


def create_cloud_openai_client() -> CloudOpenAIClient:
    """Use OPENAI_API_KEY and optional caller-owned endpoint/account settings."""
    return CloudOpenAIClient(client=create_standard_openai_client(), backend="openai")
