# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the public planning contract without dataset or model dependencies."""

import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

import coc_labeling

import numpy as np
import pytest

from coc_labeling.api import (
    ALL_META_ACTIONS,
    DEFAULT_EVENT_TIMING,
    EGO_META_ACTIONS,
    EventTiming,
    OutputSchema,
    prepare_coc_events,
    prompt_identity,
)


def motion(n=200):
    """Return motion with an interior transition from straight to left steering."""
    heading = np.zeros(n)
    heading[40:100] = np.arange(60) * 0.01
    heading[100:] = 0.59
    velocity = 5 * np.column_stack((np.cos(heading), np.sin(heading)))
    xy = np.cumsum(velocity * 0.1, axis=0)
    return xy, heading, velocity


def plan(arrays=None, **settings):
    """Plan all ego actions across a complete synthetic episode."""
    arrays = motion() if arrays is None else arrays
    return prepare_coc_events(
        *arrays,
        **dict(
            selected_actions=None,
            min_duration_ticks=1,
            min_start_tick=0,
            max_start_tick=len(arrays[0]) - 1,
            **settings,
        ),
    )


def test_plan_has_complete_sampling_and_unbound_video():
    result = plan()
    assert result.events
    assert len(EGO_META_ACTIONS) == 14
    assert set(EGO_META_ACTIONS) < set(ALL_META_ACTIONS)
    assert result.num_ticks == 200
    assert (result.prompt_version, result.prompt_sha256) == prompt_identity()
    assert result.events == tuple(
        sorted(result.events, key=lambda e: (e.start_tick, e.end_tick, e.action))
    )
    for event in result.events:
        assert event.frame_ticks == tuple(range(event.start_tick - 20, event.start_tick + 60, 5))
        assert event.speed_ticks == tuple(range(event.start_tick - 20, event.start_tick + 61, 5))
        assert len(event.messages) == 8
        assert event.messages[0]["role"] == "system"
        assert sum(m["m_type"] == "video" for m in event.messages) == 1
        assert next(m["content"] for m in event.messages if m["m_type"] == "video") is None
        assert "lateral: Not Available, lane: Not Available" in event.meta_action_text
        assert "(t=0.0s, v_long=" in event.ego_text
        assert event.start_tick >= 20 and event.start_tick + 60 < result.num_ticks
    for span in result.excluded_candidates:
        assert span.start_tick < 20 or span.start_tick + 60 >= result.num_ticks
        assert span.duration_ticks == span.end_tick - span.start_tick
    assert max(s.end_tick for s in result.longitudinal_spans) == result.num_ticks - 1


def test_no_actions_preserves_full_spans_and_identity():
    arrays = motion()
    empty = prepare_coc_events(
        *arrays, selected_actions=[], min_duration_ticks=1, min_start_tick=0, max_start_tick=199
    )
    assert not empty.events and not empty.excluded_candidates
    assert empty.longitudinal_spans and empty.lateral_spans
    assert empty.prompt_sha256 == plan(arrays).prompt_sha256
    assert len(empty.prompt_sha256) == 64
    assert OutputSchema.model_validate({"ego_behavior_schema": {"effect_on_ego_behavior": "yield"}})


def test_deterministic_parallel_calls_and_heading_wrap():
    arrays = motion()
    wrapped = arrays[1].copy() + np.pi - 0.3
    wrapped = (wrapped + np.pi) % (2 * np.pi) - np.pi
    unwrapped = np.unwrap(wrapped)
    a = plan((arrays[0], wrapped, arrays[2]))
    b = plan((arrays[0], unwrapped, arrays[2]))
    assert a.lateral_spans == b.lateral_spans
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        assert list(pool.map(lambda _: plan(), range(6))) == [plan()] * 6


@pytest.mark.parametrize(
    "settings",
    [
        {"selected_actions": ["keep_lane"]},
        {"selected_actions": ["SteerLeft"]},
        {"selected_actions": "stop"},
        {"selected_actions": [1]},
        {"min_duration_ticks": 0},
        {"min_duration_ticks": True},
        {"min_start_tick": -1},
        {"max_start_tick": 0, "min_start_tick": 1},
        {"max_start_tick": 2.5},
        {"max_start_tick": False},
        {"timing": EventTiming(history_s=1.0)},
        {"timing": None},
        {"timing": EventTiming(sample_interval_s=float("nan"))},
    ],
)
def test_invalid_settings_raise_value_error(settings):
    defaults = dict(
        selected_actions=None, min_duration_ticks=1, min_start_tick=0, max_start_tick=199
    )
    defaults.update(settings)
    with pytest.raises(ValueError):
        prepare_coc_events(*motion(), **defaults)


@pytest.mark.parametrize(
    "arrays",
    [
        (np.zeros((1, 2)), np.zeros(1), np.zeros((1, 2))),
        (np.zeros((2, 3)), np.zeros(2), np.zeros((2, 2))),
        (np.zeros((2, 2)), np.zeros((2, 1)), np.zeros((2, 2))),
        (np.zeros((2, 2)), np.zeros(3), np.zeros((2, 2))),
        (np.full((2, 2), np.nan), np.zeros(2), np.zeros((2, 2))),
        (np.zeros((2, 2)), np.zeros(2), np.full((2, 2), np.inf)),
        (np.full((2, 2), "invalid"), np.zeros(2), np.zeros((2, 2))),
    ],
)
def test_malformed_motion_raises_value_error(arrays):
    with pytest.raises(ValueError):
        plan(arrays)


def test_minimal_motion_and_stop_subtraction():
    result = plan((np.zeros((2, 2)), np.zeros(2), np.zeros((2, 2))))
    assert result.events == ()
    assert result.lateral_spans == ()
    assert [s.action for s in result.longitudinal_spans] == ["stop"]
    assert result.excluded_candidates == result.longitudinal_spans
    timing = DEFAULT_EVENT_TIMING
    assert (timing.history_ticks, timing.future_ticks, timing.sample_interval_ticks) == (20, 60, 5)
    assert (timing.sampled_fps, timing.num_frames, timing.num_speed_samples) == (2, 16, 17)


def test_import_and_planning_do_not_import_runtime_clients_or_do_io(tmp_path):
    code = """
import importlib.abc, sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'trajdata', 'openai', 'transformers', 'cv2', 'hydra', 'omegaconf'}:
            raise AssertionError('runtime import: '+fullname)
sys.meta_path.insert(0, RejectRuntime())
import numpy as np
from coc_labeling.api import prepare_coc_events
def reject_io(event, args):
    # Python/pandas may lazily load their own modules on the first call.
    if event == 'open' and str(args[0]).endswith(('.py', '.pyc', '.so')) and args[1] in {'r', 'rb'}:
        return
    if event in {'open', 'socket.connect', 'subprocess.Popen', 'os.listdir', 'os.scandir'}:
        raise AssertionError('I/O during planning: '+event)
sys.addaudithook(reject_io)
prepare_coc_events(np.zeros((2,2)), np.zeros(2), np.zeros((2,2)),
    selected_actions=None, min_duration_ticks=1, min_start_tick=0, max_start_tick=1)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(Path(coc_labeling.__file__).parent.parent),
        },
    )


def test_prompt_projection_preserves_baseline_signed_zero():
    """Use the same rotated projection as the real standalone vector adapter."""
    from coc_labeling.request import prompt_speed_components

    velocity = np.array([[5.0, 0.0], [0.0, -0.0]], dtype=np.float32)
    longitudinal, lateral = prompt_speed_components(velocity, np.zeros(2, dtype=np.float32))
    assert f"{float(longitudinal[0]):.3f}" == "5.000"
    assert f"{float(lateral[0]):.3f}" == "-0.000"
    assert np.array_equal(velocity, [[5.0, 0.0], [0.0, -0.0]])
