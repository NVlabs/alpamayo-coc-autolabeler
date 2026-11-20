# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan ego CoC requests on prepared 10 Hz motion without loading media or models."""

from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral, Real
from typing import Any

import numpy as np

from coc_labeling.data_loader.meta_action_loader import (
    mapping_action2text,
    meta_action_longitu,
    meta_action_lateral,
    meta_action_lane,
    normalize_meta_action,
)
from coc_labeling.prompts.common import OutputSchema
from coc_labeling.prompts.vlm import coc as prompt
from coc_labeling.request import (
    build_coc_messages,
    context_frame_ticks,
    matches_selection,
    prompt_speed_components,
    render_meta_action_text,
    render_speed_text,
)
from meta_action.data_structures.ego_meta_action import (
    clean_lateral_segments,
    clean_longitudinal_segments,
)
from meta_action.data_structures.ego_meta_action_wrapper import segment_tick_bounds, subtract_stops
from meta_action.processor.lateral.classification import classify_lateral
from meta_action.processor.lateral.segmentation import (
    prepare_lateral_features,
    segment_lateral_features,
)
from meta_action.processor.longitudinal.classification import classify_longitudinal
from meta_action.processor.longitudinal.segmentation import (
    prepare_longitudinal_features,
    segment_longitudinal_features,
)

EGO_META_ACTIONS = tuple(meta_action_longitu + meta_action_lateral)
ALL_META_ACTIONS = EGO_META_ACTIONS + tuple(meta_action_lane)


@dataclass(frozen=True)
class MetaActionSpan:
    """A canonical action over a half-open range of motion ticks."""

    action: str
    start_tick: int
    end_tick: int

    @property
    def duration_ticks(self) -> int:
        """Return span duration in 10 Hz ticks."""
        return self.end_tick - self.start_tick


@dataclass(frozen=True)
class EventTiming:
    """Describe context lengths and sampling on the 10 Hz motion grid."""

    history_s: float = 2.0
    future_s: float = 6.0
    sample_interval_s: float = 0.5

    @property
    def history_ticks(self) -> int:
        """Return historical motion coverage."""
        return round(self.history_s * 10)

    @property
    def future_ticks(self) -> int:
        """Return future motion coverage."""
        return round(self.future_s * 10)

    @property
    def sample_interval_ticks(self) -> int:
        """Return spacing between requested samples."""
        return round(self.sample_interval_s * 10)

    @property
    def sampled_fps(self) -> float:
        """Return requested camera frame rate."""
        return 1 / self.sample_interval_s

    @property
    def num_frames(self) -> int:
        """Return the count in the half-open camera context."""
        return (self.history_ticks + self.future_ticks) // self.sample_interval_ticks

    @property
    def num_speed_samples(self) -> int:
        """Include the additional terminal future speed sample."""
        return self.num_frames + 1


DEFAULT_EVENT_TIMING = EventTiming()


@dataclass(frozen=True)
class CocEventRequest:
    """Describe one event, its media sampling and unbound prompt messages."""

    action: str
    start_tick: int
    end_tick: int
    frame_ticks: tuple[int, ...]
    speed_ticks: tuple[int, ...]
    meta_action_text: str
    ego_text: str
    messages: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CocEpisodePlan:
    """Return full motion spans and selected requests or incomplete-context spans."""

    num_ticks: int
    longitudinal_spans: tuple[MetaActionSpan, ...]
    lateral_spans: tuple[MetaActionSpan, ...]
    events: tuple[CocEventRequest, ...]
    excluded_candidates: tuple[MetaActionSpan, ...]
    timing: EventTiming
    prompt_version: str
    prompt_sha256: str


def _validate_timing(timing):
    """Reject timing settings outside the qualified v1 sampling contract."""
    if not isinstance(timing, EventTiming):
        raise ValueError("timing must be EventTiming")
    values = (timing.history_s, timing.future_s, timing.sample_interval_s)
    if any(isinstance(v, bool) or not isinstance(v, Real) or not np.isfinite(v) for v in values):
        raise ValueError("timing fields must be finite numbers")
    if timing != DEFAULT_EVENT_TIMING:
        raise ValueError("v1 supports only DEFAULT_EVENT_TIMING")


def _prompt_config(timing):
    """Resolve the same timing fields used by the standalone prompt module."""
    return dict(
        hist_length_sec=timing.history_s,
        fut_length_sec=timing.future_s,
        time_interval=timing.sample_interval_s,
    )


def prompt_identity(timing=DEFAULT_EVENT_TIMING) -> tuple[str, str]:
    """Hash static messages, output schema and executable motion-format examples.

    Canonical serialization is UTF-8 JSON with sorted keys, compact separators,
    unescaped Unicode and no non-finite values. Motion examples pin precision,
    signed zero, timestamp conventions and every fixed motion-text heading.
    """
    _validate_timing(timing)
    sample = np.array([-0.0, 0.0, 1.23456, -12.34567, 10.0])
    speeds = {
        f"ego_{window}_{axis}_speed": sample
        for window in ("hist", "fut")
        for axis in ("longitudinal", "lateral")
    }
    actions = [
        {"Longitudinal": "Maintain Speed", "Lateral": "Not Available", "Lane": "Not Available"}
    ]
    policy = {
        "messages": build_coc_messages(prompt, _prompt_config(timing), "", ""),
        "schema": OutputSchema.model_json_schema(),
        "meta_action_text": render_meta_action_text(actions, [-5], 10, True),
        "speed_text": render_speed_text(speeds, timing.sample_interval_s),
        "action_display_text": {name: mapping_action2text[name] for name in EGO_META_ACTIONS},
    }
    encoded = json.dumps(
        policy, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return prompt.PROMPT_VERSION, hashlib.sha256(encoded).hexdigest()


def _motion_arrays(xy, heading, velocity_xy):
    """Validate finite real arrays without changing caller-owned memory."""
    arrays = []
    try:
        for value in (xy, heading, velocity_xy):
            array = np.asarray(value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError("motion must contain finite real numbers")
            arrays.append(array)
    except (TypeError, OverflowError) as error:
        raise ValueError("motion must contain finite real arrays") from error
    xy, heading, velocity_xy = arrays
    n = len(xy) if xy.ndim else 0
    if n < 2 or xy.shape != (n, 2) or heading.shape != (n,) or velocity_xy.shape != (n, 2):
        raise ValueError("motion requires matching (N,2), (N,), (N,2) arrays with N >= 2")
    return xy, heading, velocity_xy


def _spans(classified, stops=()):
    """Apply the standalone time conversion and stop suppression to classified rows."""
    spans = []
    for _, row in classified.iterrows():
        start, end = segment_tick_bounds(row)
        action = normalize_meta_action(row["caption"])
        for start, end in subtract_stops(start, end, stops):
            spans.append(MetaActionSpan(action, start, end))
    return tuple(sorted(spans, key=lambda s: (s.start_tick, s.end_tick, s.action)))


def prepare_coc_events(
    xy: np.ndarray,
    heading: np.ndarray,
    velocity_xy: np.ndarray,
    *,
    selected_actions: Sequence[str] | None,
    min_duration_ticks: int,
    min_start_tick: int,
    max_start_tick: int,
    timing: EventTiming = DEFAULT_EVENT_TIMING,
) -> CocEpisodePlan:
    """Plan ego CoC requests without I/O, dataset handles, media or model clients.

    Args:
        xy: Finite (N, 2) world-frame positions in metres, uniformly sampled at 10 Hz.
        heading: Finite (N,) radians, counterclockwise from world +X, vehicle forward.
        velocity_xy: Finite (N, 2) world-frame metres/second at the same reference point.
        selected_actions: Canonical ego action names; None selects all, empty selects none.
        min_duration_ticks: Minimum span duration, at least one tick.
        min_start_tick: Inclusive non-negative lower bound for span starts.
        max_start_tick: Inclusive upper bound for span starts.
        timing: Qualified camera/motion context; v1 requires DEFAULT_EVENT_TIMING.

    Returns:
        Deterministically ordered requests and selected spans lacking complete motion context.

    Raises:
        ValueError: Motion, selection bounds, action names or timing are malformed.
    """
    _validate_timing(timing)
    for name, value in (
        ("min_duration_ticks", min_duration_ticks),
        ("min_start_tick", min_start_tick),
        ("max_start_tick", max_start_tick),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must be an integer")
    if min_duration_ticks < 1 or min_start_tick < 0 or min_start_tick > max_start_tick:
        raise ValueError(
            "require min_duration_ticks >= 1 and 0 <= min_start_tick <= max_start_tick"
        )
    if selected_actions is None:
        selected_actions = EGO_META_ACTIONS
    if isinstance(selected_actions, (str, bytes)) or not isinstance(selected_actions, Sequence):
        raise ValueError("selected_actions must be a sequence of canonical ego names or None")
    if any(not isinstance(a, str) or a not in EGO_META_ACTIONS for a in selected_actions):
        raise ValueError("selected_actions contains an unknown or non-ego action")
    selected = set(selected_actions)
    xy, heading, velocity_xy = _motion_arrays(xy, heading, velocity_xy)
    speed = np.linalg.norm(velocity_xy, axis=-1)
    longitudinal_speed = velocity_xy[:, 0] * np.cos(heading) + velocity_xy[:, 1] * np.sin(heading)
    long_features = prepare_longitudinal_features(speed, longitudinal_speed)
    long_spans = _spans(
        clean_longitudinal_segments(
            classify_longitudinal(segment_longitudinal_features(long_features))
        )
    )
    stops = [(s.start_tick, s.end_tick) for s in long_spans if s.action == "stop"]
    lat_features = prepare_lateral_features(xy, heading, longitudinal_speed)
    lat_spans = _spans(
        clean_lateral_segments(classify_lateral(segment_lateral_features(lat_features))), stops
    )
    candidates = sorted(
        (
            s
            for s in long_spans + lat_spans
            if matches_selection(
                s.action,
                s.start_tick,
                s.end_tick,
                selected,
                min_duration_ticks,
                min_start_tick,
                max_start_tick,
            )
        ),
        key=lambda s: (s.start_tick, s.end_tick, s.action),
    )
    n = len(xy)
    prompt_longitudinal, prompt_lateral = prompt_speed_components(velocity_xy, heading)
    frames = [
        {"Longitudinal": "Not Available", "Lateral": "Not Available", "Lane": "Not Available"}
        for _ in range(n)
    ]
    for span in long_spans:
        for tick in range(span.start_tick, span.end_tick):
            frames[tick]["Longitudinal"] = mapping_action2text[span.action]
    events, excluded = [], []
    for span in candidates:
        e = span.start_tick
        if e < timing.history_ticks or e + timing.future_ticks >= n:
            excluded.append(span)
            continue
        history_ticks, future_ticks = context_frame_ticks(
            n, e, timing.history_ticks, timing.future_ticks, timing.sample_interval_ticks
        )
        frame_ticks = history_ticks + future_ticks
        speed_ticks = frame_ticks + (e + timing.future_ticks,)
        meta_text = render_meta_action_text(
            [frames[t] for t in frame_ticks], [t - e for t in frame_ticks], 10, True
        )
        history_count = timing.history_ticks // timing.sample_interval_ticks + 1
        speeds = {}
        for axis, values in (("longitudinal", prompt_longitudinal), ("lateral", prompt_lateral)):
            samples = values[list(speed_ticks)]
            speeds[f"ego_hist_{axis}_speed"] = samples[:history_count]
            speeds[f"ego_fut_{axis}_speed"] = samples[history_count:]
        ego_text = render_speed_text(speeds, timing.sample_interval_s)[2]
        messages = tuple(build_coc_messages(prompt, _prompt_config(timing), meta_text, ego_text))
        events.append(
            CocEventRequest(
                span.action,
                e,
                span.end_tick,
                frame_ticks,
                speed_ticks,
                meta_text,
                ego_text,
                messages,
            )
        )
    version, digest = prompt_identity(timing)
    return CocEpisodePlan(
        n, long_spans, lat_spans, tuple(events), tuple(excluded), timing, version, digest
    )
