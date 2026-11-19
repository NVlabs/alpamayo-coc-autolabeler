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
from meta_action.processor.lateral.config import LATERAL_SEGMENTATION_THRESHOLDS
from meta_action.utils.constant import DELTA_TIMESTAMP


def _moving_average_centered(values: np.ndarray, window_size: int) -> np.ndarray:
    """Compute a centered sliding window average for a 1D array.
    Uses reflective padding so the output length matches the input length.
    Ensures an odd window size; even sizes are incremented by 1.
    """
    if values is None:
        return values
    x = np.asarray(values, dtype=float)
    n = int(window_size) if window_size is not None else 0
    if n <= 1 or x.size == 0:
        return x.copy()
    if (n % 2) == 0:
        n += 1
    pad = n // 2
    x_pad = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(n, dtype=float) / float(n)
    y = np.convolve(x_pad, kernel, mode="valid")
    return y.astype(x.dtype, copy=False)


def prepare_lateral_data(scenario: Any, agent_token: str, smooth_window: int = 5) -> pd.DataFrame:
    """Prepare per-timestep lateral motion features for one agent."""
    agent_idx = list(scenario.all_agents_names).index(agent_token)

    # Build per-agent xy and heading from agent arrays (supports ego and non-ego)
    agent_xyh = scenario.agent_trajdata["agent_xyh"][agent_idx].detach().cpu().numpy()
    xy = agent_xyh[:, :2]
    h = agent_xyh[:, -1]

    # Velocity along heading, align length to L
    speed_along_t = scenario.agent_trajdata["agent_speed_along_heading"][agent_idx]
    speed_along = (
        speed_along_t.detach().cpu().numpy().astype(float)
        if hasattr(speed_along_t, "detach")
        else np.asarray(speed_along_t, dtype=float)
    )
    return prepare_lateral_features(xy, h, speed_along, smooth_window)


def prepare_lateral_features(
    xy: np.ndarray, h: np.ndarray, speed_along: np.ndarray, smooth_window: int = 5
) -> pd.DataFrame:
    """Build wrap-safe lateral features on the uniform 10 Hz grid."""
    # Compute per-tick distance and heading change in radians
    diff_xy = np.diff(xy, axis=0)
    dist = np.linalg.norm(diff_xy, axis=1)  # length L = T-1
    dh = np.diff(h)  # length L
    # Normalize to [-pi, pi]
    dh = (dh + np.pi) % (2 * np.pi) - np.pi
    # Smooth heading change (centered sliding window average)
    dh = _moving_average_centered(dh, smooth_window)
    # Turn rate rad/m; avoid divide by zero
    rate = np.zeros_like(dh)
    nz = dist > 1e-6
    rate[nz] = dh[nz] / dist[nz]

    L = len(rate)
    if len(speed_along) >= L:
        speed_along = speed_along[:L]
    else:
        # pad if unexpectedly shorter
        pad = np.full(L - len(speed_along), speed_along[-1] if len(speed_along) > 0 else 0.0)
        speed_along = np.concatenate([speed_along, pad])

    data = {
        "agent_heading_change_rate": rate,
        "agent_heading_change_rad": dh,
        "agent_distance_m": dist,
        "agent_velocity_along_heading": speed_along,
    }

    return pd.DataFrame(data)


def segment_lateral(
    scenario: Any,
    agent_token: str,
    segmentation_thresholds: List[Threshold] = LATERAL_SEGMENTATION_THRESHOLDS,
) -> pd.DataFrame:
    """Segment the trajectory into chunks by heading-change rate per distance.
    Returns a DataFrame with segment_start_time, duration, total_heading_change_deg,
    average_turn_rate_deg_per_m.
    """
    df_series = prepare_lateral_data(scenario, agent_token)
    return segment_lateral_features(df_series, segmentation_thresholds)


def segment_lateral_features(
    df_series: pd.DataFrame,
    segmentation_thresholds: List[Threshold] = LATERAL_SEGMENTATION_THRESHOLDS,
) -> pd.DataFrame:
    """Segment prepared lateral features using the standalone motion rules."""

    # Build boolean state arrays for each provided threshold (above/below regime)
    threshold_states = []
    for th in list(segmentation_thresholds or []):
        series_np = df_series[th.property_name].to_numpy(dtype=float)
        threshold_states.append((series_np >= float(th.threshold)).astype(np.int8))

    # If no thresholds, single segment
    rate = df_series["agent_heading_change_rate"].to_numpy(dtype=float)
    dh = df_series["agent_heading_change_rad"].to_numpy(dtype=float)
    dist = df_series["agent_distance_m"].to_numpy(dtype=float)
    v_along = df_series["agent_velocity_along_heading"].to_numpy(dtype=float)

    if len(threshold_states) == 0:
        T = int(len(rate))
        # Use exclusive end index semantics consistently
        segments = [(0, max(1, T))]
    else:
        T = min(len(s) for s in threshold_states)
        segments = []
        start = 0
        current_states = tuple(int(states[0]) for states in threshold_states)
        i = 1
        while i < T:
            next_states = tuple(int(states[i]) for states in threshold_states)
            if next_states != current_states:
                # Avoid cutting within last 3 ticks
                if (T - i) <= 3:
                    break
                segments.append((int(start), int(i)))
                start = i
                current_states = next_states
                i += 3
                continue
            i += 1
        # Use an exclusive end index for segments so that the final segment can
        # extend to the full timeline length. This ensures the last tick is
        # covered when converting to time/ticks downstream.
        segments.append((int(start), int(T)))

    # Metric accumulation over segments
    dt = float(DELTA_TIMESTAMP)
    rows = []
    for s0, s1 in segments:
        # Clamp and skip zero-length. Allow s1 to equal len(dh) because s1 is treated
        # as an exclusive bound when slicing series of length len(dh).
        s0 = max(0, min(int(s0), len(dh)))
        s1 = min(int(s1), len(dh))
        if s1 <= s0:
            continue
        start_time = s0 * dt
        duration = (s1 - s0) * dt
        total_heading = float(np.sum(dh[s0:s1]))  # rad signed
        total_dist = float(np.sum(dist[s0:s1]))
        avg_rate = float(total_heading / total_dist) if total_dist > 1e-6 else 0.0  # rad/m
        avg_speed_along = float(np.mean(v_along[s0:s1]))
        rows.append(
            {
                "segment_start_time": start_time,
                "duration": duration,
                "total_heading_change_rad": total_heading,
                "average_turn_rate_rad_per_m": avg_rate,
                "total_distance_m": total_dist,
                "average_speed_along_heading": avg_speed_along,
            }
        )

    return pd.DataFrame(rows).sort_values("segment_start_time").reset_index(drop=True)
