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


def _agent_without_model_init(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Create an agent instance without constructing model wrappers."""
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    coc_agent = importlib.import_module("coc_labeling.agents.vlm.coc_agent")
    return coc_agent.VLMCoCAgent.__new__(coc_agent.VLMCoCAgent)


def _lane_prior(labels: list[str], timestamps: list[int]) -> dict[str, Any]:
    """Build minimal segment data for lateral-prior extraction tests."""
    return {
        "all_meta_actions": [{"Lane": label} for label in labels],
        "all_ts": timestamps,
    }


def test_configured_sampling_params_are_passed_to_vlm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply Hydra sampling overrides to VLM inference calls."""
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    coc_agent = importlib.import_module("coc_labeling.agents.vlm.coc_agent")
    recording_vlm = SimpleNamespace(kwargs=None)

    class _RecordingWrapper:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def infer(self, **kwargs: Any) -> dict[str, Any]:
            recording_vlm.kwargs = kwargs
            return {"content": "ok"}

    monkeypatch.setattr(coc_agent, "VLMWrapper", _RecordingWrapper)
    agent = coc_agent.VLMCoCAgent(
        model_name="dummy",
        agent_config=SimpleNamespace(
            temperature=0.2,
            top_p=0.8,
            repetition_penalty=1.1,
        ),
        vector_config=SimpleNamespace(),
        init_model=False,
    )

    json_schema = object()
    response = agent._infer(
        messages=[{"role": "user", "content": "hello"}],
        json_schema=json_schema,
    )

    assert response == {"content": "ok"}
    assert recording_vlm.kwargs is not None
    assert recording_vlm.kwargs == {
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.2,
        "top_p": 0.8,
        "repetition_penalty": 1.1,
        "json_schema": json_schema,
    }


def test_process_results_persists_model_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist token usage metadata without changing model content."""
    agent = _agent_without_model_init(monkeypatch)
    coc_agent = importlib.import_module("coc_labeling.agents.vlm.coc_agent")
    saved_payload = {}

    monkeypatch.setattr(
        coc_agent,
        "get_vlm_yaml_path",
        lambda save_root, data: f"{save_root}/{data['clip_id']}.yaml",
    )
    monkeypatch.setattr(
        coc_agent,
        "save_yaml",
        lambda path, payload: saved_payload.update(path=path, payload=payload),
    )

    _, output = agent.process_results(
        data={
            "clip_id": "clip-a",
            "event_start_frame": 3,
            "event_start_timestamp": 123,
        },
        messages=[{"role": "user", "m_type": "text", "content": "hello"}],
        response={
            "content": {"label": "ok"},
            "prompt_tokens": 10,
            "response_tokens": 20,
            "reasoning_tokens": 7,
        },
        save_root="/tmp/out",
    )

    assert output["final_content"] == {"label": "ok"}
    assert output["model_usage"] == {
        "prompt_tokens": 10,
        "response_tokens": 20,
        "reasoning_tokens": 7,
    }
    assert saved_payload["payload"] == output


def test_lane_change_prior_requires_adjacent_original_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not treat isolated repeated lane-change labels as sustained lane change."""
    agent = _agent_without_model_init(monkeypatch)

    result = agent._extract_lateral_prior(
        _lane_prior(
            [
                "Left Lane Change",
                "Keep Lane",
                "Keep Lane",
                "Keep Lane",
                "Left Lane Change",
            ],
            [0, 1, 2, 3, 4],
        )
    )

    assert result["has_prior"] is False
    assert result["expected_type"] == ""
    assert "lane_change_onset_left=None" in result["summary"]


def test_adjacent_lane_change_prior_still_triggers_repair_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve sustained lane-change behavior for consecutive original samples."""
    agent = _agent_without_model_init(monkeypatch)

    result = agent._extract_lateral_prior(
        _lane_prior(
            [
                "Left Lane Change",
                "Left Lane Change",
                "Keep Lane",
            ],
            [0, 1, 2],
        )
    )

    assert result["has_prior"] is True
    assert result["expected_type"] == "lane_change"
    assert result["expected_direction"] == "left"
    assert result["expected_onset_ts"] == 0


def test_missing_lateral_direction_triggers_repair_when_prior_has_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Require left/right direction when the lateral prior has one."""
    agent = _agent_without_model_init(monkeypatch)

    trigger, reason = agent._should_trigger_meta_repair(
        "Lane change while maintaining clearance from the target-lane vehicle.",
        {
            "has_prior": True,
            "expected_type": "lane_change",
            "expected_direction": "left",
        },
    )

    assert trigger is True
    assert "predicted=none" in reason
    assert "expected=left" in reason


def test_matching_lateral_direction_does_not_trigger_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep existing behavior when model type and direction match the prior."""
    agent = _agent_without_model_init(monkeypatch)

    trigger, reason = agent._should_trigger_meta_repair(
        "Lane change to the left while maintaining clearance from the target-lane vehicle.",
        {
            "has_prior": True,
            "expected_type": "lane_change",
            "expected_direction": "left",
        },
    )

    assert trigger is False
    assert reason == "Initial output is consistent with onset-based lateral prior."
