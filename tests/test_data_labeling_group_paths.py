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

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_data_labeling(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import data_labeling with lightweight stubs for optional runtime deps."""
    hydra_module = ModuleType("hydra")
    hydra_module.main = lambda *args, **kwargs: lambda func: func  # type: ignore[attr-defined]

    omegaconf_module = ModuleType("omegaconf")
    omegaconf_module.DictConfig = object  # type: ignore[attr-defined]

    labeling_agent_module = ModuleType("coc_labeling.agents.labeling_agent")
    labeling_agent_module.LabelingAgent = object  # type: ignore[attr-defined]

    runtime_config_module = ModuleType("coc_labeling.model_clients.runtime_config")
    runtime_config_module.ModelRuntimeConfig = object  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "hydra", hydra_module)
    monkeypatch.setitem(sys.modules, "omegaconf", omegaconf_module)
    monkeypatch.setitem(sys.modules, "coc_labeling.agents.labeling_agent", labeling_agent_module)
    monkeypatch.setitem(
        sys.modules, "coc_labeling.model_clients.runtime_config", runtime_config_module
    )
    sys.modules.pop("coc_labeling.data_labeling", None)
    return importlib.import_module("coc_labeling.data_labeling")


def _cfg(**data_kwargs: Any) -> SimpleNamespace:
    """Build the minimal config shape used by group path rewriting."""
    defaults = {
        "data_dir": "/single/data",
        "cache_dir": "/single/cache",
        "segment_config_path": "/single/segments.json",
        "video_dir": "/single/video",
        "meta_action_dir": "/single/meta",
        "data_dir_root": None,
        "cache_dir_root": None,
        "segment_config_dir": None,
        "video_dir_root": None,
        "meta_action_dir_root": None,
    }
    defaults.update(data_kwargs)
    return SimpleNamespace(data=SimpleNamespace(**defaults))


def test_group_paths_ignore_null_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Leave single-node paths untouched when group roots are null."""
    data_labeling = _load_data_labeling(monkeypatch)
    cfg = _cfg()

    data_labeling._apply_group_paths(cfg, "group_001", "json")

    assert cfg.data.data_dir == "/single/data"
    assert cfg.data.cache_dir == "/single/cache"
    assert cfg.data.segment_config_path == "/single/segments.json"
    assert cfg.data.video_dir == "/single/video"
    assert cfg.data.meta_action_dir == "/single/meta"


def test_timeout_sec_defaults_to_runtime_max(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the shared runtime timeout when config does not define one."""
    data_labeling = _load_data_labeling(monkeypatch)

    assert data_labeling._get_timeout_sec(SimpleNamespace()) == data_labeling.TIMEOUT_MAX


def test_timeout_sec_reads_config_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow Hydra config and CLI overrides to control model request timeout."""
    data_labeling = _load_data_labeling(monkeypatch)

    assert data_labeling._get_timeout_sec(SimpleNamespace(timeout_sec="120")) == 120


def test_group_paths_only_derives_segment_json_for_json_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid replacing parquet segment inputs with per-group JSON paths."""
    data_labeling = _load_data_labeling(monkeypatch)
    cfg = _cfg(segment_config_dir="/group/segments")

    data_labeling._apply_group_paths(cfg, "group_001", "parquet")

    assert cfg.data.segment_config_path == "/single/segments.json"


def test_group_paths_apply_non_null_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build group-specific paths from each configured non-null root."""
    data_labeling = _load_data_labeling(monkeypatch)
    cfg = _cfg(
        data_dir_root="/group/data",
        cache_dir_root="/group/cache",
        segment_config_dir="/group/segments",
        video_dir_root="/group/video",
        meta_action_dir_root="/group/meta",
    )

    data_labeling._apply_group_paths(cfg, "group_001", "json")

    assert cfg.data.data_dir == "/group/data/group_001"
    assert cfg.data.cache_dir == "/group/cache/group_001"
    assert cfg.data.segment_config_path == "/group/segments/group_001.json"
    assert cfg.data.video_dir == "/group/video/group_001"
    assert cfg.data.meta_action_dir == "/group/meta/group_001"


def test_group_data_root_requires_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail with a clear error when paired group roots are incomplete."""
    data_labeling = _load_data_labeling(monkeypatch)
    cfg = _cfg(data_dir_root="/group/data", cache_dir_root=None)

    with pytest.raises(ValueError, match="data.cache_dir_root"):
        data_labeling._apply_group_paths(cfg, "group_001", "json")
