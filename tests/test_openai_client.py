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

"""Public cloud credentials must select only the caller's configured service."""

import pytest

from coc_labeling.model_clients import openai_client


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {
            "OPENAI_BASE_URL": "https://example.test/v1",
            "OPENAI_ORG_ID": "test-org",
            "OPENAI_PROJECT": "test-project",
        },
        {"OPENAI_ORGANIZATION": "alternate-org"},
    ],
)
def test_public_key_and_account_settings_reach_only_the_configured_client(monkeypatch, settings):
    """Preserve the public SDK credential/base/account contract without any network calls."""
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_ORG_ID",
        "OPENAI_ORGANIZATION",
        "OPENAI_PROJECT",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    calls = []
    monkeypatch.setattr(openai_client, "OpenAI", lambda **kwargs: calls.append(kwargs) or object())
    client = openai_client.create_cloud_openai_client()
    expected = {"api_key": "synthetic-openai-key"}
    for name, keyword in [("OPENAI_BASE_URL", "base_url"), ("OPENAI_PROJECT", "project")]:
        if name in settings:
            expected[keyword] = settings[name]
    organization = settings.get("OPENAI_ORG_ID", settings.get("OPENAI_ORGANIZATION"))
    if organization:
        expected["organization"] = organization
    assert calls == [expected]
    assert client.backend == "openai"


@pytest.mark.parametrize("key", [None, "", "   "])
def test_absent_or_blank_public_key_fails_before_client_initialization(monkeypatch, key):
    """Unrelated provider credentials cannot silently authorize a public request."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UNRELATED_PROVIDER_KEY", "synthetic-other-key")
    if key is not None:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    calls = []
    monkeypatch.setattr(openai_client, "OpenAI", lambda **kwargs: calls.append(kwargs))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        openai_client.create_cloud_openai_client()
    assert not calls
