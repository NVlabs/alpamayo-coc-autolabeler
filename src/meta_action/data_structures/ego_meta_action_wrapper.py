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

from typing import TYPE_CHECKING, Tuple, TypeVar

from meta_action.data_structures.ego_meta_action import (
    get_or_compute_lane_segments,
    get_or_compute_lateral_segments,
    get_or_compute_longitudinal_segments,
)
from meta_action.data_structures.motion import TemporalMotionChunk

from meta_action.utils.constant import DELTA_TIMESTAMP

if TYPE_CHECKING:
    from meta_action.data_structures.scenario import TemporalScenario

Interval = Tuple[int, int]
TTemporalMotion = TypeVar("TTemporalMotion", bound=TemporalMotionChunk)


def subtract_stops(a_start: int, a_end: int, stops: list[Interval]) -> list[Interval]:
    """Subtract a union of stop intervals from [a_start, a_end) and return
    the list of remaining non-overlapping sub-intervals.

    Only if the entire interval is covered by stops will this return [].
    """
    if a_end <= a_start:
        return []
    if not stops:
        return [(a_start, a_end)]

    sorted_stops = sorted(stops, key=lambda x: (x[0], x[1]))
    remaining: list[Interval] = []
    current_start = a_start

    for s_start, s_end in sorted_stops:
        if s_end <= current_start:
            continue
        if s_start >= a_end:
            break
        if s_start > current_start:
            remaining.append((current_start, min(s_start, a_end)))
        current_start = max(current_start, s_end)
        if current_start >= a_end:
            break

    if current_start < a_end:
        remaining.append((current_start, a_end))

    return [(s, e) for (s, e) in remaining if e > s]


def segment_tick_bounds(row):
    """Convert standalone segment start/duration seconds to half-open 10 Hz bounds."""
    start_time = float(row["segment_start_time"])
    end_time = start_time + float(row["duration"])
    return int(round(start_time / DELTA_TIMESTAMP)), int(round(end_time / DELTA_TIMESTAMP))


def _get_stop_intervals(
    scenario: TemporalScenario,
    agent_token: str,
) -> list[Interval]:
    """Build Stop intervals (in ticks) from longitudinal segments.

    Returns an empty list if longitudinal segments are unavailable or malformed.
    """
    stop_intervals: list[Interval] = []
    try:
        long_df = get_or_compute_longitudinal_segments(scenario, agent_token)
        if long_df is not None and len(long_df) > 0:
            for _, lrow in long_df.iterrows():
                if str(lrow.get("caption", "")) != "Stop":
                    continue
                l_start_ts, l_end_ts = segment_tick_bounds(lrow)
                if l_end_ts > l_start_ts:
                    stop_intervals.append((l_start_ts, l_end_ts))
    except Exception:
        # Keep legacy behavior: if longitudinal parsing fails, skip suppression.
        return []

    return stop_intervals


class ReverseTemporal(TemporalMotionChunk):
    """Temporal wrapper for longitudinal reverse segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[ReverseTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="Reverse",
            temp_motion_class=ReverseTemporal,
        )


class GentleAccelerationTemporal(TemporalMotionChunk):
    """Temporal wrapper for gentle-acceleration segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[GentleAccelerationTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="GentleAcceleration",
            temp_motion_class=GentleAccelerationTemporal,
        )


class StrongAccelerationTemporal(TemporalMotionChunk):
    """Temporal wrapper for strong-acceleration segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[StrongAccelerationTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="StrongAcceleration",
            temp_motion_class=StrongAccelerationTemporal,
        )


class GentleDecelerationTemporal(TemporalMotionChunk):
    """Temporal wrapper for gentle-deceleration segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[GentleDecelerationTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="GentleDeceleration",
            temp_motion_class=GentleDecelerationTemporal,
        )


class StrongDecelerationTemporal(TemporalMotionChunk):
    """Temporal wrapper for strong-deceleration segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[StrongDecelerationTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="StrongDeceleration",
            temp_motion_class=StrongDecelerationTemporal,
        )


class MaintainSpeedTemporal(TemporalMotionChunk):
    """Temporal wrapper for maintain-speed segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[MaintainSpeedTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token,
            scenario,
            target_caption="MaintainSpeed",
            temp_motion_class=MaintainSpeedTemporal,
        )


class StopTemporal(TemporalMotionChunk):
    """Temporal wrapper for stop segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(agent_token: str, scenario: TemporalScenario) -> list[StopTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_longitudinal(
            agent_token, scenario, target_caption="Stop", temp_motion_class=StopTemporal
        )


def _build_motions_from_longitudinal(
    agent_token: str,
    scenario: TemporalScenario,
    target_caption: str,
    temp_motion_class: type[TTemporalMotion],
) -> list[TTemporalMotion]:
    """Build temporal motions directly from the precomputed longitudinal segments"""
    motions: list[TTemporalMotion] = []
    df = get_or_compute_longitudinal_segments(scenario, agent_token)
    if df is None or len(df) == 0:
        return motions

    for _, row in df.iterrows():
        if str(row.get("caption", "")) != target_caption:
            continue
        start_ts, end_ts = segment_tick_bounds(row)
        if end_ts <= start_ts:
            continue
        motions.append(temp_motion_class(agent_token, start_ts, end_ts))

    return motions


def _build_motions_from_lateral(
    agent_token: str,
    scenario: TemporalScenario,
    target_caption: str,
    temp_motion_class: type[TTemporalMotion],
) -> list[TTemporalMotion]:
    """Build temporal motions directly from lateral (turning) segments computed via
    heading-change rate per distance.
    """
    motions: list[TTemporalMotion] = []
    class_df = get_or_compute_lateral_segments(scenario, agent_token)
    if class_df is None or len(class_df) == 0:
        return motions

    # Suppress lateral labels while longitudinal classification says Stop.
    stop_intervals = _get_stop_intervals(scenario, agent_token)

    for _, row in class_df.iterrows():
        if str(row.get("caption", "")) != target_caption:
            continue
        start_ts, end_ts = segment_tick_bounds(row)
        if end_ts <= start_ts:
            continue
        # Subtract Stop intervals: keep non-overlapping parts only. If fully covered, skip.
        non_overlapping_parts = subtract_stops(start_ts, end_ts, stop_intervals)
        for s_ts, e_ts in non_overlapping_parts:
            if e_ts > s_ts:
                motions.append(temp_motion_class(agent_token, s_ts, e_ts))

    return motions


def _build_motions_from_lane(
    agent_token: str,
    scenario: TemporalScenario,
    target_caption: str,
    temp_motion_class: type[TTemporalMotion],
) -> list[TTemporalMotion]:
    """Build temporal motions directly from lane segments."""
    motions: list[TTemporalMotion] = []
    class_df = get_or_compute_lane_segments(scenario, agent_token)
    if class_df is None or len(class_df) == 0:
        return motions

    # Suppress lane labels while longitudinal classification says Stop.
    stop_intervals = _get_stop_intervals(scenario, agent_token)

    for _, row in class_df.iterrows():
        if str(row.get("caption", "")) != target_caption:
            continue
        start_ts, end_ts = segment_tick_bounds(row)
        if end_ts <= start_ts:
            continue
        # Subtract Stop intervals: keep non-overlapping parts only. If fully covered, skip.
        non_overlapping_parts = subtract_stops(start_ts, end_ts, stop_intervals)
        for s_ts, e_ts in non_overlapping_parts:
            if e_ts > s_ts:
                motions.append(temp_motion_class(agent_token, s_ts, e_ts))

    return motions


class LeftLaneChangeTemporal(TemporalMotionChunk):
    """Temporal wrapper for left lane-change segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[LeftLaneChangeTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="LeftLaneChange",
            temp_motion_class=LeftLaneChangeTemporal,
        )


class RightLaneChangeTemporal(TemporalMotionChunk):
    """Temporal wrapper for right lane-change segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[RightLaneChangeTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="RightLaneChange",
            temp_motion_class=RightLaneChangeTemporal,
        )


class LaneKeepTemporal(TemporalMotionChunk):
    """Temporal wrapper for lane-keep segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[LaneKeepTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="LaneKeep",
            temp_motion_class=LaneKeepTemporal,
        )


class SlightlyShiftLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for slight-left-shift segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SlightlyShiftLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="SlightlyShiftLeft",
            temp_motion_class=SlightlyShiftLeftTemporal,
        )


class SlightlyShiftRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for slight-right-shift segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SlightlyShiftRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="SlightlyShiftRight",
            temp_motion_class=SlightlyShiftRightTemporal,
        )


class TurnLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for left-turn lane segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[TurnLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="TurnLeft",
            temp_motion_class=TurnLeftTemporal,
        )


class TurnRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for right-turn lane segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[TurnRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="TurnRight",
            temp_motion_class=TurnRightTemporal,
        )


class FollowCurveLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for left follow-curve segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[FollowCurveLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="FollowCurveLeft",
            temp_motion_class=FollowCurveLeftTemporal,
        )


class FollowCurveRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for right follow-curve segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[FollowCurveRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lane(
            agent_token,
            scenario,
            target_caption="FollowCurveRight",
            temp_motion_class=FollowCurveRightTemporal,
        )


class ReverseRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for reverse-right lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[ReverseRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="ReverseRight",
            temp_motion_class=ReverseRightTemporal,
        )


class ReverseLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for reverse-left lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[ReverseLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="ReverseLeft",
            temp_motion_class=ReverseLeftTemporal,
        )


class SteerRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for steer-right lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SteerRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="SteerRight",
            temp_motion_class=SteerRightTemporal,
        )


class SteerLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for steer-left lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SteerLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="SteerLeft",
            temp_motion_class=SteerLeftTemporal,
        )


class SharpSteerRightTemporal(TemporalMotionChunk):
    """Temporal wrapper for sharp-steer-right lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SharpSteerRightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="SharpSteerRight",
            temp_motion_class=SharpSteerRightTemporal,
        )


class SharpSteerLeftTemporal(TemporalMotionChunk):
    """Temporal wrapper for sharp-steer-left lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[SharpSteerLeftTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="SharpSteerLeft",
            temp_motion_class=SharpSteerLeftTemporal,
        )


class GoStraightTemporal(TemporalMotionChunk):
    """Temporal wrapper for go-straight lateral segments."""

    def __init__(self, agent_token: str, start_ts: int, end_ts: int) -> None:
        super().__init__(agent_token, start_ts, end_ts)

    @staticmethod
    def get_motion_for_scenario(
        agent_token: str, scenario: TemporalScenario
    ) -> list[GoStraightTemporal]:
        """Return matching motion chunks for one agent in one scenario."""
        return _build_motions_from_lateral(
            agent_token,
            scenario,
            target_caption="GoStraight",
            temp_motion_class=GoStraightTemporal,
        )


META_ACTION_MAPPING = {
    # longitudinal
    "gentle_acceleration": GentleAccelerationTemporal,
    "strong_acceleration": StrongAccelerationTemporal,
    "gentle_deceleration": GentleDecelerationTemporal,
    "strong_deceleration": StrongDecelerationTemporal,
    "maintain_speed": MaintainSpeedTemporal,
    "stop": StopTemporal,
    "reverse": ReverseTemporal,
    # lateral
    "reverse_right": ReverseRightTemporal,
    "reverse_left": ReverseLeftTemporal,
    "steer_right": SteerRightTemporal,
    "steer_left": SteerLeftTemporal,
    "sharp_steer_right": SharpSteerRightTemporal,
    "sharp_steer_left": SharpSteerLeftTemporal,
    "go_straight": GoStraightTemporal,
    # Lane
    "keep_lane": LaneKeepTemporal,
    "lane_keep": LaneKeepTemporal,
    "left_lane_change": LeftLaneChangeTemporal,
    "right_lane_change": RightLaneChangeTemporal,
    "slightly_shift_left": SlightlyShiftLeftTemporal,
    "slightly_shift_right": SlightlyShiftRightTemporal,
    "turn_left": TurnLeftTemporal,
    "turn_right": TurnRightTemporal,
    # Intentionally disabled: VLM captures curvy-road behavior better than this
    # heuristic, so we do not emit FollowCurve* meta actions in current outputs.
    # "follow_curve_left": FollowCurveLeftTemporal,
    # "follow_curve_right": FollowCurveRightTemporal,
}
