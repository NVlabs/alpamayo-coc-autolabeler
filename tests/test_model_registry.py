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
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.modules.setdefault("cv2", ModuleType("cv2"))
sys.modules.setdefault("torch", ModuleType("torch"))

from coc_labeling.model_clients.vlm_wrappers.factory import (  # noqa: E402
    MODEL_WRAPPER_REGISTRY,
    create_model_wrapper,
)
from coc_labeling.model_clients.vlm_wrappers.qwen import QWEN_MODEL_SPECS  # noqa: E402


def test_released_model_registry_excludes_qwen25_aliases() -> None:
    """Keep deprecated Qwen2.5 aliases out of the OSS release surface."""
    deprecated_aliases = {"qwen2.5_vl_32b_awq", "qwen2.5_vl_72b_awq"}

    assert deprecated_aliases.isdisjoint(MODEL_WRAPPER_REGISTRY)
    assert deprecated_aliases.isdisjoint(QWEN_MODEL_SPECS)


def test_released_model_registry_excludes_qwen3_30b_alias() -> None:
    """Keep the lower-quality Qwen3 30B alias out of the OSS release surface."""
    removed_alias = "qwen3_vl_30b_awq"

    assert removed_alias not in MODEL_WRAPPER_REGISTRY
    assert removed_alias not in QWEN_MODEL_SPECS


def test_released_model_registry_includes_gpt55() -> None:
    """Expose only the current GPT aliases through the cloud wrapper."""
    current_gpt_aliases = {
        model_name for model_name in MODEL_WRAPPER_REGISTRY if model_name.startswith("gpt")
    }

    assert current_gpt_aliases == {"gpt5", "gpt5.5"}


def test_unsupported_model_error_lists_supported_values() -> None:
    """Return an actionable error instead of a raw KeyError."""
    with pytest.raises(ValueError, match="Supported values: .*qwen3.5_35b"):
        create_model_wrapper("qwen2.5_vl_32b_awq")
