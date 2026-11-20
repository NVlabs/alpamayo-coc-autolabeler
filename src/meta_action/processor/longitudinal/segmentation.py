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

from typing import Any, List

import numpy as np
import pandas as pd

from meta_action.processor.common import Threshold
from meta_action.processor.longitudinal.config import LONGITUDINAL_SEGMENTATION_THRESHOLDS
from meta_action.utils.constant import DELTA_TIMESTAMP


def prepare_longitudinal_data(scenario: Any, agent_token: str) -> pd.DataFrame:
    """Prepare per-timestep longitudinal features for one agent."""
    agent_idx = list(scenario.all_agents_names).index(agent_token)
    speed_t = scenario.agent_trajdata["agent_speed"][agent_idx]
    speed_along_t = scenario.agent_trajdata["agent_speed_along_heading"][agent_idx]

    speed = (
        speed_t.detach().cpu().numpy().astype(float)
        if hasattr(speed_t, "detach")
        else np.asarray(speed_t, dtype=float)
    )
    speed_along = (
        speed_along_t.detach().cpu().numpy().astype(float)
        if hasattr(speed_along_t, "detach")
        else np.asarray(speed_along_t, dtype=float)
    )

    return prepare_longitudinal_features(speed, speed_along)


def prepare_longitudinal_features(speed: np.ndarray, speed_along: np.ndarray) -> pd.DataFrame:
    """Build longitudinal features from speeds on the uniform 10 Hz grid."""
    speed = np.asarray(speed, dtype=float)
    speed_along = np.asarray(speed_along, dtype=float)
    acc = np.diff(speed) / float(DELTA_TIMESTAMP)
    if acc.size == 0:
        acc = np.zeros_like(speed)
    else:
        acc = np.concatenate([[acc[0]], acc])

    data = {
        "agent_speed": speed,
        "agent_speed_along_heading": speed_along,
        "agent_acceleration": acc,
    }
    return pd.DataFrame(data)


def segment_longitudinal(
    scenario: Any,
    agent_token: str,
    segmentation_thresholds: List[Threshold] = LONGITUDINAL_SEGMENTATION_THRESHOLDS,
    accel_smoothing_window: int = 5,
    hysteresis_fraction: float = 0.15,
) -> pd.DataFrame:
    """Segment one agent's longitudinal trajectory into motion-consistent spans."""
    # Prepare per-agent DataFrame with required series
    df_series = prepare_longitudinal_data(scenario, agent_token)
    return segment_longitudinal_features(
        df_series, segmentation_thresholds, accel_smoothing_window, hysteresis_fraction
    )


def segment_longitudinal_features(
    df_series: pd.DataFrame,
    segmentation_thresholds: List[Threshold] = LONGITUDINAL_SEGMENTATION_THRESHOLDS,
    accel_smoothing_window: int = 5,
    hysteresis_fraction: float = 0.15,
) -> pd.DataFrame:
    """Segment prepared longitudinal features using the standalone motion rules."""

    speeds = df_series["agent_speed"].to_numpy(dtype=float)
    speeds_along = df_series["agent_speed_along_heading"].to_numpy(dtype=float)

    if speeds.size < 2:
        return pd.DataFrame(
            {
                "segment_start_time": [],
                "duration": [],
                "delta_velocity": [],
                "average_acceleration": [],
                "average_speed": [],
                "average_speed_along_heading": [],
            }
        )

    # Build per-threshold boolean states (above vs below) and cut when any boolean flips
    # Apply smoothing and 15% hysteresis to acceleration-based thresholds to avoid chattering
    def _smooth_series(x: np.ndarray, window: int) -> np.ndarray:
        """Apply centered rolling-mean smoothing to a 1D series."""
        window = int(max(1, window))
        if window == 1 or x.size <= 1:
            return x.astype(float)
        # centered rolling mean with edge handling via min_periods=1
        return (
            pd.Series(x, dtype=float)
            .rolling(window=window, center=True, min_periods=1)
            .mean()
            .to_numpy(dtype=float)
        )

    def _hysteresis_boolean(series: np.ndarray, threshold: float, frac: float) -> np.ndarray:
        """Convert a numeric series to a boolean using hysteresis around a base threshold.
        State is True when series is above the threshold band and remains True until it
        falls below the lower threshold band.

        For positive threshold T:
          - enter True at >= T*(1+frac), exit to False at < T*(1-frac)
        For negative threshold T:
          - enter True at >= T*(1-frac), exit to False at < T*(1+frac)
        """
        T = float(threshold)
        f = float(max(0.0, frac))
        if T >= 0.0:
            T_on = T * (1.0 + f)
            T_off = T * (1.0 - f)
        else:
            T_on = T * (1.0 - f)
            T_off = T * (1.0 + f)

        out = np.zeros_like(series, dtype=bool)
        if series.size == 0:
            return out

        # Initialize using base threshold to avoid biasing initial state
        state = bool(series[0] >= T)
        out[0] = state
        for i in range(1, series.size):
            val = float(series[i])
            if not state:
                if val >= T_on:
                    state = True
            else:
                if val < T_off:
                    state = False
            out[i] = state
        return out

    # Pre-compute smoothed acceleration series for thresholding
    acc_raw = df_series["agent_acceleration"].to_numpy(dtype=float)
    acc_smooth = _smooth_series(acc_raw, accel_smoothing_window)

    threshold_states = []
    for th in list(segmentation_thresholds or []):
        prop = str(th.property_name)
        thr = float(th.threshold)
        if prop == "agent_acceleration":
            series_np = acc_smooth
            states = _hysteresis_boolean(series_np, thr, hysteresis_fraction)
        else:
            series_np = df_series[prop].to_numpy(dtype=float)
            states = series_np >= thr
        threshold_states.append(states)

    if len(threshold_states) == 0:
        T = int(speeds.shape[0])
        segments = [(0, max(1, T - 1))]
    else:
        T = min(len(s) for s in threshold_states)
        segments = []
        start = 0
        current_states = tuple(bool(states[0]) for states in threshold_states)
        i = 1
        while i < T:
            next_states = tuple(bool(states[i]) for states in threshold_states)
            if next_states != current_states:
                # If we are within the last 3 ticks, don't cut a new segment.
                # Extend the current segment to the end instead.
                if (T - i) <= 3:
                    break
                seg_start_tick = int(start)
                seg_end_tick = int(i)
                segments.append((seg_start_tick, seg_end_tick))
                # Start a new segment at i; skip cutting again for the next 3 ticks
                start = i
                current_states = next_states
                i += 3
                continue
            i += 1
        segments.append((int(start), int(T - 1)))

    rows = []
    dt = float(DELTA_TIMESTAMP)
    for seg_start_tick, seg_end_tick in segments:
        # Clamp to bounds and skip zero-length segments safely
        seg_start_tick = max(0, min(int(seg_start_tick), len(speeds) - 1))
        seg_end_tick = min(int(seg_end_tick), len(speeds) - 1)
        if seg_end_tick <= seg_start_tick:
            continue
        start_time = seg_start_tick * dt
        duration = (seg_end_tick - seg_start_tick) * dt
        dv = float(speeds[seg_end_tick] - speeds[seg_start_tick])
        avg_a = float(dv / duration) if duration > 0 else 0.0
        # Use half-open slicing [start:end) for averages
        avg_speed = float(np.mean(speeds[seg_start_tick:seg_end_tick]))
        avg_speed_along = float(np.mean(speeds_along[seg_start_tick:seg_end_tick]))
        result_dict = {
            "segment_start_time": start_time,
            "duration": duration,
            "delta_velocity": dv,
            "average_acceleration": avg_a,
            "average_speed": avg_speed,
            "average_speed_along_heading": avg_speed_along,
        }
        rows.append(result_dict)

    df = pd.DataFrame(
        rows,
        columns=[
            "segment_start_time",
            "duration",
            "delta_velocity",
            "average_acceleration",
            "average_speed",
            "average_speed_along_heading",
        ],
    )

    return df.sort_values("segment_start_time").reset_index(drop=True)
