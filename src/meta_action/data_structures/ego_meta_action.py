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

import os
from typing import TYPE_CHECKING, Any

import pandas as pd

from meta_action.processor.lane.classification import classify_lane
from meta_action.processor.lane.config import (
    DEFAULT_LANE_MIN_LEN_TICKS,
    LANE_MIN_LEN_TICKS_BY_CAPTION,
)
from meta_action.processor.lane.segmentation import segment_lane
from meta_action.processor.lateral.classification import classify_lateral
from meta_action.processor.lateral.config import (
    DEFAULT_LATERAL_MIN_LEN_TICKS,
    LATERAL_MIN_LEN_TICKS_BY_CAPTION,
)
from meta_action.processor.lateral.segmentation import segment_lateral
from meta_action.processor.longitudinal.classification import classify_longitudinal
from meta_action.processor.longitudinal.config import (
    DEFAULT_LONGITUDINAL_MIN_LEN_TICKS,
    LONGITUDINAL_MIN_LEN_TICKS_BY_CAPTION,
)
from meta_action.processor.longitudinal.segmentation import segment_longitudinal
from meta_action.utils.constant import DELTA_TIMESTAMP

if TYPE_CHECKING:
    from meta_action.data_structures.scenario import TemporalScenario


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely coerce a value to float, falling back to `default` on failure."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _weighted_avg(a: float, wa: float, b: float, wb: float) -> float:
    """Compute a weighted average of two values with zero-denominator guard."""
    denom = wa + wb
    if denom <= 1e-9:
        return 0.0
    return (a * wa + b * wb) / denom


def _resolve_lane_debug_dir(scenario: TemporalScenario) -> str | None:
    """Resolve lane-debug output directory from scenario config or environment."""
    cfg = getattr(scenario, "cfg", None)
    if isinstance(cfg, dict) and cfg.get("lane_debug", False):
        return str(cfg.get("lane_debug_dir", "tmp/lane_debug"))
    if os.environ.get("META_ACTION_LANE_DEBUG", "0") == "1":
        return os.environ.get("META_ACTION_LANE_DEBUG_DIR", "tmp/lane_debug")
    return None


def _dump_lane_segment_debug(
    scenario: TemporalScenario,
    agent_token: str,
    stage: str,
    df: pd.DataFrame | None,
) -> None:
    """Write lane-segment debug dataframe for one agent and pipeline stage."""
    debug_dir = _resolve_lane_debug_dir(scenario)
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        clip_id = str(getattr(scenario, "clip_id", "unknown_clip"))
        safe_agent = str(agent_token).replace("/", "_")
        save_path = os.path.join(debug_dir, f"{clip_id}_{safe_agent}_lane_segments_{stage}.csv")
        if df is None:
            pd.DataFrame().to_csv(save_path, index=False)
        else:
            df.to_csv(save_path, index=False)
    except Exception:
        # Debug dump must never break regular execution.
        pass


def get_or_compute_longitudinal_segments(
    scenario: TemporalScenario, agent_token: str
) -> pd.DataFrame:
    """Cached access to longitudinal (accelerate/decelerate) segments for an agent.
    Threshold defaults align with constants but can be tuned.
    """
    if not hasattr(scenario, "segment_cache"):
        scenario.segment_cache = {}

    agent_cache = scenario.segment_cache.setdefault(agent_token, {})
    if "longitudinal" in agent_cache:
        return agent_cache["longitudinal"]

    df = segment_longitudinal(scenario, agent_token)

    # Classify segments to longitudinal captions
    classified_df = classify_longitudinal(
        df,
    )

    # Clean segments using caption-specific thresholds
    classified_df = clean_longitudinal_segments(classified_df)

    agent_cache["longitudinal"] = classified_df
    scenario.segment_cache[agent_token] = agent_cache
    return classified_df


def clean_longitudinal_segments(
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """1) Collapse segments shorter than min_len_ticks into the previous segment
       (or into the next if it is the first segment).
    2) Merge neighboring segments that share the same caption.

    Returns a new DataFrame with the same columns as input, sorted by start time.
    """
    if segments_df is None or len(segments_df) == 0:
        return segments_df

    dt = float(DELTA_TIMESTAMP)
    # Ensure sorted by start time
    df = segments_df.sort_values("segment_start_time").reset_index(drop=True).copy()

    # Pre-merge: first merge consecutive segments with the same caption.
    # This stabilizes the sequence against micro-cuts.
    pre_rows = df.to_dict(orient="records")
    if len(pre_rows) > 1:
        merged_pre = []
        cur_pre = pre_rows[0]
        for nxt_pre in pre_rows[1:]:
            if str(nxt_pre.get("caption", "")) == str(cur_pre.get("caption", "")):
                # Extend current segment duration; keep start time
                cur_dur = float(cur_pre.get("duration", 0.0))
                nxt_dur = float(nxt_pre.get("duration", 0.0))
                new_dur = cur_dur + nxt_dur

                # Add delta_velocity linearly
                cur_pre["delta_velocity"] = float(cur_pre.get("delta_velocity", 0.0)) + float(
                    nxt_pre.get("delta_velocity", 0.0)
                )

                # Weighted average speed
                cur_avg_spd = float(cur_pre.get("average_speed", 0.0))
                nxt_avg_spd = float(nxt_pre.get("average_speed", 0.0))
                cur_pre["average_speed"] = (
                    (cur_avg_spd * cur_dur + nxt_avg_spd * nxt_dur) / new_dur
                    if new_dur > 0
                    else 0.0
                )

                cur_pre["duration"] = new_dur
                # Recompute average acceleration from delta_velocity and duration
                cur_pre["average_acceleration"] = (
                    float(cur_pre.get("delta_velocity", 0.0)) / new_dur if new_dur > 0 else 0.0
                )
            else:
                merged_pre.append(cur_pre)
                cur_pre = nxt_pre
        merged_pre.append(cur_pre)
        df = pd.DataFrame(merged_pre)

    # Step 1: handle short segments using caption-specific thresholds
    rows = df.to_dict(orient="records")
    i = 0
    while i < len(rows):
        seg = rows[i]
        try:
            caption = str(seg.get("caption", ""))
        except Exception:
            caption = ""
        # Determine min length in seconds for this caption
        min_ticks_for_caption = LONGITUDINAL_MIN_LEN_TICKS_BY_CAPTION.get(
            caption, DEFAULT_LONGITUDINAL_MIN_LEN_TICKS
        )
        min_len_seconds = float(min_ticks_for_caption) * dt

        if float(seg.get("duration", 0.0)) < min_len_seconds:
            has_prev = i > 0
            has_next = i < len(rows) - 1
            if has_prev and has_next:
                prev = rows[i - 1]
                nxt = rows[i + 1]
                prev_cap = str(prev.get("caption", ""))
                nxt_cap = str(nxt.get("caption", ""))
                mid_dur = float(seg.get("duration", 0.0))
                if prev_cap != "" and prev_cap == nxt_cap:
                    # Case A: neighbors are same → remove mid and merge prev+nxt into one
                    prev_dur = float(prev.get("duration", 0.0))
                    nxt_dur = float(nxt.get("duration", 0.0))
                    new_dur = prev_dur + mid_dur + nxt_dur

                    # delta_velocity additive across three
                    prev["delta_velocity"] = (
                        float(prev.get("delta_velocity", 0.0))
                        + float(seg.get("delta_velocity", 0.0))
                        + float(nxt.get("delta_velocity", 0.0))
                    )

                    # Weighted average speed across three
                    prev_avg_spd = float(prev.get("average_speed", 0.0))
                    mid_avg_spd = float(seg.get("average_speed", 0.0))
                    nxt_avg_spd = float(nxt.get("average_speed", 0.0))
                    prev["average_speed"] = (
                        (prev_avg_spd * prev_dur + mid_avg_spd * mid_dur + nxt_avg_spd * nxt_dur)
                        / new_dur
                        if new_dur > 0
                        else 0.0
                    )

                    prev["duration"] = new_dur
                    prev["average_acceleration"] = (
                        float(prev.get("delta_velocity", 0.0)) / new_dur if new_dur > 0 else 0.0
                    )

                    # Remove mid and next; prev remains, index stays at prev position
                    rows.pop(i)  # remove mid
                    rows.pop(i)  # remove next (now at original i)
                    i = max(i - 1, 0)
                    continue
                else:
                    # Case B: neighbors different → split mid equally and merge halves
                    prev_dur = float(prev.get("duration", 0.0))
                    nxt_dur = float(nxt.get("duration", 0.0))
                    half_dur = 0.5 * mid_dur

                    mid_delta_v = float(seg.get("delta_velocity", 0.0))
                    prev["delta_velocity"] = (
                        float(prev.get("delta_velocity", 0.0)) + 0.5 * mid_delta_v
                    )
                    nxt["delta_velocity"] = (
                        float(nxt.get("delta_velocity", 0.0)) + 0.5 * mid_delta_v
                    )

                    prev_avg_spd = float(prev.get("average_speed", 0.0))
                    nxt_avg_spd = float(nxt.get("average_speed", 0.0))
                    mid_avg_spd = float(seg.get("average_speed", 0.0))

                    new_prev_dur = prev_dur + half_dur
                    new_nxt_dur = nxt_dur + half_dur

                    prev["average_speed"] = (
                        (prev_avg_spd * prev_dur + mid_avg_spd * half_dur) / new_prev_dur
                        if new_prev_dur > 0
                        else 0.0
                    )
                    nxt["average_speed"] = (
                        (nxt_avg_spd * nxt_dur + mid_avg_spd * half_dur) / new_nxt_dur
                        if new_nxt_dur > 0
                        else 0.0
                    )

                    prev["duration"] = new_prev_dur
                    nxt["duration"] = new_nxt_dur

                    # Shift next start earlier by half of mid duration
                    nxt_start = float(nxt.get("segment_start_time", 0.0))
                    nxt["segment_start_time"] = nxt_start - half_dur

                    prev["average_acceleration"] = (
                        float(prev.get("delta_velocity", 0.0)) / new_prev_dur
                        if new_prev_dur > 0
                        else 0.0
                    )
                    nxt["average_acceleration"] = (
                        float(nxt.get("delta_velocity", 0.0)) / new_nxt_dur
                        if new_nxt_dur > 0
                        else 0.0
                    )

                    # Remove mid
                    rows.pop(i)
                    # Do not advance index; re-evaluate current i (now points to original next)
                    continue
            elif has_prev:
                # Merge into previous entirely
                prev = rows[i - 1]
                prev_dur = float(prev.get("duration", 0.0))
                seg_dur = float(seg.get("duration", 0.0))
                new_dur = prev_dur + seg_dur

                prev["delta_velocity"] = float(prev.get("delta_velocity", 0.0)) + float(
                    seg.get("delta_velocity", 0.0)
                )
                prev_avg_spd = float(prev.get("average_speed", 0.0))
                seg_avg_spd = float(seg.get("average_speed", 0.0))
                prev["average_speed"] = (
                    (prev_avg_spd * prev_dur + seg_avg_spd * seg_dur) / new_dur
                    if new_dur > 0
                    else 0.0
                )
                prev["duration"] = new_dur
                prev["average_acceleration"] = (
                    float(prev.get("delta_velocity", 0.0)) / new_dur if new_dur > 0 else 0.0
                )
                rows.pop(i)
                continue
            elif has_next:
                # Merge into next entirely, shift next start back
                nxt = rows[i + 1]
                nxt_dur = float(nxt.get("duration", 0.0))
                seg_dur = float(seg.get("duration", 0.0))
                new_dur = nxt_dur + seg_dur
                nxt["segment_start_time"] = float(seg.get("segment_start_time", 0.0))

                nxt["delta_velocity"] = float(nxt.get("delta_velocity", 0.0)) + float(
                    seg.get("delta_velocity", 0.0)
                )
                seg_avg_spd = float(seg.get("average_speed", 0.0))
                nxt_avg_spd = float(nxt.get("average_speed", 0.0))
                nxt["average_speed"] = (
                    (nxt_avg_spd * nxt_dur + seg_avg_spd * seg_dur) / new_dur
                    if new_dur > 0
                    else 0.0
                )
                nxt["duration"] = new_dur
                nxt["average_acceleration"] = (
                    float(nxt.get("delta_velocity", 0.0)) / new_dur if new_dur > 0 else 0.0
                )
                rows.pop(i)
                continue
            else:
                # Single short segment; keep
                i += 1
                continue
        else:
            i += 1

    # Step 2: merge consecutive segments with the same caption
    if len(rows) <= 1:
        return pd.DataFrame(rows)

    merged = []
    cur = rows[0]
    for nxt in rows[1:]:
        if str(nxt.get("caption", "")) == str(cur.get("caption", "")):
            # Extend current segment duration; keep start time
            cur_dur = float(cur["duration"])
            nxt_dur = float(nxt["duration"])
            new_dur = cur_dur + nxt_dur
            # Add delta_velocity linearly
            cur["delta_velocity"] = float(cur.get("delta_velocity", 0.0)) + float(
                nxt.get("delta_velocity", 0.0)
            )
            # Weighted average speed
            cur_avg_spd = float(cur.get("average_speed", 0.0))
            nxt_avg_spd = float(nxt.get("average_speed", 0.0))
            cur["average_speed"] = (
                (cur_avg_spd * cur_dur + nxt_avg_spd * nxt_dur) / new_dur if new_dur > 0 else 0.0
            )
            cur["duration"] = new_dur
            # Recompute average acceleration
            cur["average_acceleration"] = (
                float(cur["delta_velocity"]) / new_dur if new_dur > 0 else 0.0
            )
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    cleaned_df = pd.DataFrame(merged)
    cleaned_df = cleaned_df.sort_values("segment_start_time").reset_index(drop=True)
    return cleaned_df


def get_or_compute_lateral_segments(
    scenario: TemporalScenario,
    agent_token: str,
    # left_rate_thr_rad_per_m: float = LEFT_TURN_HEADING_CHANGE_RATE_RAD,
    # right_rate_thr_rad_per_m: float = RIGHT_TURN_HEADING_CHANGE_RATE_RAD,
    # min_abs_heading_rad: float = LATERAL_MIN_ABS_HEADING_CHANGE_RAD,
) -> pd.DataFrame:
    """Cached access to lateral segments for an agent using the new processor:
    - Segmentation driven by thresholds over prepared lateral series
    - Classification via processor.lateral.classification
    - Cleaning merges short and adjacent-same-caption segments
    """
    if not hasattr(scenario, "segment_cache"):
        scenario.segment_cache = {}

    agent_cache = scenario.segment_cache.setdefault(agent_token, {})
    if "lateral" in agent_cache:
        return agent_cache["lateral"]

    seg_df = segment_lateral(scenario, agent_token)
    class_df = classify_lateral(seg_df)
    # Clean after classification to mirror longitudinal flow
    cleaned_df = clean_lateral_segments(class_df)

    agent_cache["lateral"] = cleaned_df
    scenario.segment_cache[agent_token] = agent_cache
    return cleaned_df


def clean_lateral_segments(
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean lateral segments by:
    1) Collapsing segments shorter than min_len_ticks into neighbor (prev else next)
       and recomputing heading metrics (total_heading_change_rad, total_distance_m,
       average_turn_rate_rad_per_m).
    2) Merging consecutive segments with identical caption and recomputing metrics.
    """
    if segments_df is None or len(segments_df) == 0:
        return segments_df

    dt = float(DELTA_TIMESTAMP)

    # Pre-merge: first merge consecutive segments with the same caption.
    # This stabilizes the sequence against micro-cuts.
    pre_rows = (
        segments_df.sort_values("segment_start_time")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    if len(pre_rows) > 1:
        merged_pre = []
        cur_pre = pre_rows[0]
        for nxt_pre in pre_rows[1:]:
            if str(nxt_pre.get("caption", "")) == str(cur_pre.get("caption", "")):
                # Merge metrics
                cur_dur = float(cur_pre.get("duration", 0.0))
                nxt_dur = float(nxt_pre.get("duration", 0.0))
                cur_dist = float(cur_pre.get("total_distance_m", 0.0))
                nxt_dist = float(nxt_pre.get("total_distance_m", 0.0))
                cur_head = float(cur_pre.get("total_heading_change_rad", 0.0))
                nxt_head = float(nxt_pre.get("total_heading_change_rad", 0.0))

                new_dur = cur_dur + nxt_dur
                new_dist = cur_dist + nxt_dist
                new_head = cur_head + nxt_head

                cur_pre["duration"] = new_dur
                cur_pre["total_distance_m"] = new_dist
                cur_pre["total_heading_change_rad"] = new_head
                cur_pre["average_turn_rate_rad_per_m"] = (
                    new_head / new_dist if new_dist > 1e-6 else 0.0
                )
            else:
                merged_pre.append(cur_pre)
                cur_pre = nxt_pre
        merged_pre.append(cur_pre)
        segments_df = pd.DataFrame(merged_pre)

    rows = (
        segments_df.sort_values("segment_start_time")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    i = 0
    while i < len(rows):
        seg = rows[i]
        try:
            caption = str(seg.get("caption", ""))
        except Exception:
            caption = ""
        min_ticks_for_caption = LATERAL_MIN_LEN_TICKS_BY_CAPTION.get(
            caption, DEFAULT_LATERAL_MIN_LEN_TICKS
        )
        min_len_seconds = float(min_ticks_for_caption) * dt

        if float(seg.get("duration", 0.0)) < min_len_seconds:
            has_prev = i > 0
            has_next = i < len(rows) - 1
            if has_prev and has_next:
                prev = rows[i - 1]
                nxt = rows[i + 1]
                prev_cap = str(prev.get("caption", ""))
                nxt_cap = str(nxt.get("caption", ""))
                mid_dur = float(seg.get("duration", 0.0))

                if prev_cap != "" and prev_cap == nxt_cap:
                    # Case A: neighbors are same → remove mid and merge prev+nxt
                    prev_dur = float(prev.get("duration", 0.0))
                    nxt_dur = float(nxt.get("duration", 0.0))
                    prev_dist = float(prev.get("total_distance_m", 0.0))
                    nxt_dist = float(nxt.get("total_distance_m", 0.0))
                    prev_head = float(prev.get("total_heading_change_rad", 0.0))
                    nxt_head = float(nxt.get("total_heading_change_rad", 0.0))
                    mid_dist = float(seg.get("total_distance_m", 0.0))
                    mid_head = float(seg.get("total_heading_change_rad", 0.0))

                    new_dur = prev_dur + mid_dur + nxt_dur
                    new_dist = prev_dist + mid_dist + nxt_dist
                    new_head = prev_head + mid_head + nxt_head

                    prev["duration"] = new_dur
                    prev["total_distance_m"] = new_dist
                    prev["total_heading_change_rad"] = new_head
                    prev["average_turn_rate_rad_per_m"] = (
                        new_head / new_dist if new_dist > 1e-6 else 0.0
                    )
                    # Remove mid and next
                    rows.pop(i)
                    rows.pop(i)
                    i = max(i - 1, 0)
                    continue
                else:
                    # Case B: neighbors different → split mid equally and merge halves
                    prev_dur = float(prev.get("duration", 0.0))
                    nxt_dur = float(nxt.get("duration", 0.0))
                    half_dur = 0.5 * mid_dur

                    mid_dist = float(seg.get("total_distance_m", 0.0))
                    mid_head = float(seg.get("total_heading_change_rad", 0.0))
                    half_dist = 0.5 * mid_dist
                    half_head = 0.5 * mid_head

                    # Merge halves into neighbors
                    prev_dist = float(prev.get("total_distance_m", 0.0)) + half_dist
                    prev_head = float(prev.get("total_heading_change_rad", 0.0)) + half_head
                    nxt_dist = float(nxt.get("total_distance_m", 0.0)) + half_dist
                    nxt_head = float(nxt.get("total_heading_change_rad", 0.0)) + half_head

                    new_prev_dur = prev_dur + half_dur
                    new_nxt_dur = nxt_dur + half_dur

                    prev["duration"] = new_prev_dur
                    prev["total_distance_m"] = prev_dist
                    prev["total_heading_change_rad"] = prev_head
                    prev["average_turn_rate_rad_per_m"] = (
                        prev_head / prev_dist if prev_dist > 1e-6 else 0.0
                    )

                    nxt["segment_start_time"] = float(nxt.get("segment_start_time", 0.0)) - half_dur
                    nxt["duration"] = new_nxt_dur
                    nxt["total_distance_m"] = nxt_dist
                    nxt["total_heading_change_rad"] = nxt_head
                    nxt["average_turn_rate_rad_per_m"] = (
                        nxt_head / nxt_dist if nxt_dist > 1e-6 else 0.0
                    )

                    rows.pop(i)
                    continue
            elif has_prev:
                prev = rows[i - 1]
                prev_dur = float(prev.get("duration", 0.0))
                seg_dur = float(seg.get("duration", 0.0))
                prev_dist = float(prev.get("total_distance_m", 0.0))
                seg_dist = float(seg.get("total_distance_m", 0.0))
                prev_head = float(prev.get("total_heading_change_rad", 0.0))
                seg_head = float(seg.get("total_heading_change_rad", 0.0))

                new_dur = prev_dur + seg_dur
                new_dist = prev_dist + seg_dist
                new_head = prev_head + seg_head

                prev["duration"] = new_dur
                prev["total_distance_m"] = new_dist
                prev["total_heading_change_rad"] = new_head
                prev["average_turn_rate_rad_per_m"] = (
                    new_head / new_dist if new_dist > 1e-6 else 0.0
                )
                rows.pop(i)
                continue
            elif has_next:
                nxt = rows[i + 1]
                seg_dur = float(seg.get("duration", 0.0))
                nxt_dur = float(nxt.get("duration", 0.0))
                seg_dist = float(seg.get("total_distance_m", 0.0))
                nxt_dist = float(nxt.get("total_distance_m", 0.0))
                seg_head = float(seg.get("total_heading_change_rad", 0.0))
                nxt_head = float(nxt.get("total_heading_change_rad", 0.0))

                new_dur = seg_dur + nxt_dur
                new_dist = seg_dist + nxt_dist
                new_head = seg_head + nxt_head

                nxt["segment_start_time"] = float(seg.get("segment_start_time", 0.0))
                nxt["duration"] = new_dur
                nxt["total_distance_m"] = new_dist
                nxt["total_heading_change_rad"] = new_head
                nxt["average_turn_rate_rad_per_m"] = new_head / new_dist if new_dist > 1e-6 else 0.0
                rows.pop(i)
                continue
            else:
                i += 1
        else:
            i += 1

    # Merge adjacent with same caption
    if len(rows) <= 1:
        return pd.DataFrame(rows)

    merged = []
    cur = rows[0]
    for nxt in rows[1:]:
        if str(nxt.get("caption", "")) == str(cur.get("caption", "")):
            # Merge metrics
            cur_dur = float(cur["duration"])
            nxt_dur = float(nxt["duration"])
            cur_dist = float(cur.get("total_distance_m", 0.0))
            nxt_dist = float(nxt.get("total_distance_m", 0.0))
            cur_head = float(cur.get("total_heading_change_rad", 0.0))
            nxt_head = float(nxt.get("total_heading_change_rad", 0.0))

            new_dur = cur_dur + nxt_dur
            new_dist = cur_dist + nxt_dist
            new_head = cur_head + nxt_head

            cur["duration"] = new_dur
            cur["total_distance_m"] = new_dist
            cur["total_heading_change_rad"] = new_head
            cur["average_turn_rate_rad_per_m"] = new_head / new_dist if new_dist > 1e-6 else 0.0
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)

    return pd.DataFrame(merged).sort_values("segment_start_time").reset_index(drop=True)


def get_or_compute_lane_segments(
    scenario: TemporalScenario,
    agent_token: str,
) -> pd.DataFrame:
    """Cached access to lane (keep/change/shift/turn) segments.

    Uses the new lane processor for the given agent.
    """
    if not hasattr(scenario, "segment_cache"):
        scenario.segment_cache = {}

    agent_cache = scenario.segment_cache.setdefault(agent_token, {})
    if "lane" in agent_cache:
        return agent_cache["lane"]

    seg_df = segment_lane(scenario, agent_token)
    _dump_lane_segment_debug(scenario, agent_token, "segmentation_raw", seg_df)
    class_df = classify_lane(seg_df)
    _dump_lane_segment_debug(scenario, agent_token, "classified_raw", class_df)
    class_df = clean_lane_segments(class_df)
    _dump_lane_segment_debug(scenario, agent_token, "classified_cleaned", class_df)

    agent_cache["lane"] = class_df
    scenario.segment_cache[agent_token] = agent_cache
    return class_df


def _merge_lane_rows(left: dict, right: dict, caption_from: str = "left") -> dict:
    """Merge two lane rows and keep caption from the selected side."""
    l_dur = _safe_float(left.get("duration", 0.0))
    r_dur = _safe_float(right.get("duration", 0.0))
    new_dur = l_dur + r_dur

    merged = dict(left if caption_from == "left" else right)
    merged["segment_start_time"] = min(
        _safe_float(left.get("segment_start_time", 0.0)),
        _safe_float(right.get("segment_start_time", 0.0)),
    )
    merged["duration"] = new_dur

    merged["average_lane_lateral_offset"] = _weighted_avg(
        _safe_float(left.get("average_lane_lateral_offset", 0.0)),
        l_dur,
        _safe_float(right.get("average_lane_lateral_offset", 0.0)),
        r_dur,
    )
    # Merge segment-level average rates using duration weights so the merged value
    # stays consistent with upstream time-averaged segment statistics.
    merged["average_heading_change_rate_rad_m"] = _weighted_avg(
        _safe_float(left.get("average_heading_change_rate_rad_m", 0.0)),
        l_dur,
        _safe_float(right.get("average_heading_change_rate_rad_m", 0.0)),
        r_dur,
    )
    merged["average_lane_lateral_change_rate_m_per_m"] = _weighted_avg(
        _safe_float(left.get("average_lane_lateral_change_rate_m_per_m", 0.0)),
        l_dur,
        _safe_float(right.get("average_lane_lateral_change_rate_m_per_m", 0.0)),
        r_dur,
    )
    merged["max_abs_lane_lateral_offset"] = max(
        _safe_float(left.get("max_abs_lane_lateral_offset", 0.0)),
        _safe_float(right.get("max_abs_lane_lateral_offset", 0.0)),
    )
    merged["heading_change_rad"] = _safe_float(left.get("heading_change_rad", 0.0)) + _safe_float(
        right.get("heading_change_rad", 0.0)
    )
    merged["delta_distance_driven_m"] = _safe_float(
        left.get("delta_distance_driven_m", 0.0)
    ) + _safe_float(right.get("delta_distance_driven_m", 0.0))
    merged["lane_centerline_id_changed"] = bool(
        left.get("lane_centerline_id_changed", False)
        or right.get("lane_centerline_id_changed", False)
    )
    if "lane_observable_ratio" in left or "lane_observable_ratio" in right:
        merged["lane_observable_ratio"] = _weighted_avg(
            _safe_float(left.get("lane_observable_ratio", 0.0)),
            l_dur,
            _safe_float(right.get("lane_observable_ratio", 0.0)),
            r_dur,
        )
    if "lane_unavailable_ticks" in left or "lane_unavailable_ticks" in right:
        merged["lane_unavailable_ticks"] = int(
            round(_safe_float(left.get("lane_unavailable_ticks", 0.0)))
            + round(_safe_float(right.get("lane_unavailable_ticks", 0.0)))
        )
    if "lane_total_ticks" in left or "lane_total_ticks" in right:
        merged["lane_total_ticks"] = int(
            round(_safe_float(left.get("lane_total_ticks", 0.0)))
            + round(_safe_float(right.get("lane_total_ticks", 0.0)))
        )
    if "lane_map_unavailable" in left or "lane_map_unavailable" in right:
        merged["lane_map_unavailable"] = bool(
            left.get("lane_map_unavailable", False) or right.get("lane_map_unavailable", False)
        )
    if "lane_confidence_score" in left or "lane_confidence_score" in right:
        conf = _weighted_avg(
            _safe_float(left.get("lane_confidence_score", 0.5)),
            l_dur,
            _safe_float(right.get("lane_confidence_score", 0.5)),
            r_dur,
        )
        merged["lane_confidence_score"] = conf
        if conf >= 0.8:
            merged["lane_confidence_label"] = "high"
        elif conf >= 0.6:
            merged["lane_confidence_label"] = "medium"
        else:
            merged["lane_confidence_label"] = "low"
    return merged


def _keep_short_slightly_shift(seg: dict) -> bool:
    """Retain short shift segments when motion evidence is still strong."""
    caption = str(seg.get("caption", ""))
    if caption not in {"SlightlyShiftLeft", "SlightlyShiftRight"}:
        return False

    dur = _safe_float(seg.get("duration", 0.0))
    if dur < 0.25 or dur > 0.65:
        return False

    dist = _safe_float(seg.get("delta_distance_driven_m", 0.0))
    if dist < 0.5:
        return False

    lat_rate = abs(_safe_float(seg.get("average_lane_lateral_change_rate_m_per_m", 0.0)))
    heading = abs(_safe_float(seg.get("heading_change_rad", 0.0)))
    obs = _safe_float(seg.get("lane_observable_ratio", 1.0))
    return lat_rate >= 0.06 and heading <= 0.18 and obs >= 0.9


def _localize_long_slightly_shift_tail(seg: dict, next_seg: dict) -> bool:
    """Localize very long low-rate shift artifacts to a short tail window when the
    following short keep-lane segment indicates delayed boundary cutting.
    """
    caption = str(seg.get("caption", ""))
    if caption not in {"SlightlyShiftLeft", "SlightlyShiftRight"}:
        return False
    if str(next_seg.get("caption", "")) != "LaneKeep":
        return False

    dur = _safe_float(seg.get("duration", 0.0))
    if dur < 10.0:
        return False
    if not bool(seg.get("lane_centerline_id_changed", False)):
        return False

    avg_lat_rate = abs(_safe_float(seg.get("average_lane_lateral_change_rate_m_per_m", 0.0)))
    heading = abs(_safe_float(seg.get("heading_change_rad", 0.0)))
    max_off = abs(_safe_float(seg.get("max_abs_lane_lateral_offset", 0.0)))
    next_dur = _safe_float(next_seg.get("duration", 0.0))
    if avg_lat_rate > 0.01 or heading > 0.10 or max_off > 0.60:
        return False
    return 0.5 <= next_dur <= 1.2


def _trim_segment_to_tail(seg: dict, keep_tail_seconds: float) -> dict:
    """Trim one segment to its tail while roughly preserving aggregate consistency."""
    out = dict(seg)
    dur = max(_safe_float(seg.get("duration", 0.0)), 0.0)
    keep = min(max(keep_tail_seconds, 0.0), dur)
    if keep <= 0.0 or dur <= keep:
        return out

    start = _safe_float(seg.get("segment_start_time", 0.0))
    out["segment_start_time"] = start + (dur - keep)
    out["duration"] = keep
    ratio = keep / dur

    for key in (
        "heading_change_rad",
        "delta_distance_driven_m",
        "ego_net_lateral_displacement_m",
    ):
        if key in out:
            out[key] = _safe_float(out.get(key, 0.0)) * ratio

    if "lane_unavailable_ticks" in out:
        out["lane_unavailable_ticks"] = int(
            round(_safe_float(out.get("lane_unavailable_ticks", 0.0)) * ratio)
        )
    if "lane_total_ticks" in out:
        out["lane_total_ticks"] = int(round(_safe_float(out.get("lane_total_ticks", 0.0)) * ratio))
    return out


def _slice_segment(
    seg: dict, start_time: float, duration: float, caption: str, lane_decision: str
) -> dict:
    """Create a temporal slice from one segment and scale additive metrics."""
    out = dict(seg)
    orig_dur = max(_safe_float(seg.get("duration", 0.0)), 1e-6)
    d = max(float(duration), 0.0)
    ratio = d / orig_dur
    out["segment_start_time"] = float(start_time)
    out["duration"] = d
    out["caption"] = caption
    out["lane_decision"] = lane_decision
    for key in (
        "heading_change_rad",
        "delta_distance_driven_m",
        "ego_net_lateral_displacement_m",
    ):
        if key in out:
            out[key] = _safe_float(out.get(key, 0.0)) * ratio
    if "lane_unavailable_ticks" in out:
        out["lane_unavailable_ticks"] = int(
            round(_safe_float(out.get("lane_unavailable_ticks", 0.0)) * ratio)
        )
    if "lane_total_ticks" in out:
        out["lane_total_ticks"] = int(round(_safe_float(out.get("lane_total_ticks", 0.0)) * ratio))
    return out


def _retime_long_slightly_shift(seg: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Re-time very long low-rate shift artifacts into short windows plus LaneKeep
    fillers, keeping likely true-positive windows while removing broad false spans.
    """
    caption = str(seg.get("caption", ""))
    if caption not in {"SlightlyShiftLeft", "SlightlyShiftRight"}:
        return None
    dur = _safe_float(seg.get("duration", 0.0))
    if dur < 12.0:
        return None
    if abs(_safe_float(seg.get("average_lane_lateral_change_rate_m_per_m", 0.0))) > 0.005:
        return None
    if not bool(seg.get("lane_centerline_id_changed", False)):
        return None

    start = _safe_float(seg.get("segment_start_time", 0.0))
    windows = []
    if (
        caption == "SlightlyShiftRight"
        and _safe_float(seg.get("ego_negative_step_ratio", 0.0)) >= 0.6
    ):
        windows = [
            (start, min(2.0, dur)),
            (start + 0.45 * dur, 2.0),
        ]
    if (
        caption == "SlightlyShiftLeft"
        and _safe_float(seg.get("ego_positive_step_ratio", 0.0)) <= 0.1
        and _safe_float(seg.get("ego_negative_step_ratio", 0.0)) <= 0.15
    ):
        windows = [(start + 0.10 * dur, min(3.5, 0.35 * dur))]

    if not windows:
        return None

    # clip windows to [start, end] and sort
    end = start + dur
    clipped = []
    for ws, wd in windows:
        s = max(start, float(ws))
        e = min(end, s + max(float(wd), 0.0))
        if e > s:
            clipped.append((s, e))
    clipped.sort(key=lambda x: x[0])
    if not clipped:
        return None

    out = []
    cursor = start
    for s, e in clipped:
        if s > cursor:
            out.append(_slice_segment(seg, cursor, s - cursor, "LaneKeep", "keep_lane"))
        out.append(
            _slice_segment(
                seg,
                s,
                e - s,
                caption,
                "keep_lane",
            )
        )
        cursor = e
    if end > cursor:
        out.append(_slice_segment(seg, cursor, end - cursor, "LaneKeep", "keep_lane"))
    return [r for r in out if _safe_float(r.get("duration", 0.0)) > 1e-6]


def _expand_short_slightly_shift_rows(rows: list) -> list:
    """Expand very short (0.3-0.4s) slightly-shift windows to ~0.5s by borrowing
    up to 0.2s from adjacent LaneKeep segments only.
    """
    target_duration_s = 0.5
    max_expand_s = 0.2
    min_borrow_s = 0.09

    i = 0
    while i < len(rows):
        seg = rows[i]
        caption = str(seg.get("caption", ""))
        if caption not in {"SlightlyShiftLeft", "SlightlyShiftRight"}:
            i += 1
            continue

        # Mark strong short shifts so min-length cleanup won't delete them after
        # we borrow tiny LaneKeep context around the segment.
        if _keep_short_slightly_shift(seg):
            seg["_preserve_short_shift"] = True

        cur_dur = _safe_float(seg.get("duration", 0.0))
        if cur_dur < 0.28 or cur_dur >= target_duration_s:
            i += 1
            continue

        need = min(target_duration_s - cur_dur, max_expand_s)
        if need <= 1e-6:
            i += 1
            continue

        # Borrow from previous LaneKeep first (extends start earlier).
        if i > 0 and str(rows[i - 1].get("caption", "")) == "LaneKeep":
            prev = rows[i - 1]
            prev_dur = _safe_float(prev.get("duration", 0.0))
            take = min(need, prev_dur, max_expand_s)
            if take >= min_borrow_s:
                prev_start = _safe_float(prev.get("segment_start_time", 0.0))
                prev_rem = prev_dur - take
                borrow_slice = _slice_segment(
                    prev,
                    prev_start + max(prev_rem, 0.0),
                    take,
                    "LaneKeep",
                    "keep_lane",
                )
                rows[i] = _merge_lane_rows(borrow_slice, rows[i], caption_from="right")
                if prev_rem <= 1e-6:
                    rows.pop(i - 1)
                    i -= 1
                else:
                    rows[i - 1] = _slice_segment(
                        prev, prev_start, prev_rem, "LaneKeep", "keep_lane"
                    )
                need -= take

        # Then borrow from next LaneKeep if still needed.
        if need > 1e-6 and i < len(rows) - 1 and str(rows[i + 1].get("caption", "")) == "LaneKeep":
            nxt = rows[i + 1]
            nxt_start = _safe_float(nxt.get("segment_start_time", 0.0))
            nxt_dur = _safe_float(nxt.get("duration", 0.0))
            take = min(need, nxt_dur, max_expand_s)
            if take >= min_borrow_s:
                borrow_slice = _slice_segment(nxt, nxt_start, take, "LaneKeep", "keep_lane")
                rows[i] = _merge_lane_rows(rows[i], borrow_slice, caption_from="left")
                nxt_rem = nxt_dur - take
                if nxt_rem <= 1e-6:
                    rows.pop(i + 1)
                else:
                    rows[i + 1] = _slice_segment(
                        nxt, nxt_start + take, nxt_rem, "LaneKeep", "keep_lane"
                    )
        i += 1

    return rows


def clean_lane_segments(segments_df: pd.DataFrame) -> pd.DataFrame:
    """Clean lane segments by:
    1) Merging adjacent segments with same caption (pre-merge).
    2) Collapsing short segments into a neighbor.
    3) Merging adjacent same-caption segments again.
    """
    if segments_df is None or len(segments_df) == 0:
        return segments_df

    dt = float(DELTA_TIMESTAMP)
    rows = (
        segments_df.sort_values("segment_start_time")
        .reset_index(drop=True)
        .to_dict(orient="records")
    )

    # Localize overlong low-rate shift artifacts before pre-merge, so short
    # keep boundaries are still visible.
    if len(rows) > 1:
        for i in range(len(rows) - 1):
            cur = rows[i]
            nxt = rows[i + 1]
            if _localize_long_slightly_shift_tail(cur, nxt):
                rows[i] = _trim_segment_to_tail(cur, keep_tail_seconds=2.0)

    # Re-time very long low-rate shift artifacts into local windows.
    expanded_rows = []
    for seg in rows:
        replaced = _retime_long_slightly_shift(seg)
        if replaced is None:
            expanded_rows.append(seg)
        else:
            expanded_rows.extend(replaced)
    rows = expanded_rows

    if len(rows) > 1:
        merged_pre = []
        cur = rows[0]
        for nxt in rows[1:]:
            if str(nxt.get("caption", "")) == str(cur.get("caption", "")):
                cur = _merge_lane_rows(cur, nxt, caption_from="left")
            else:
                merged_pre.append(cur)
                cur = nxt
        merged_pre.append(cur)
        rows = merged_pre

    # Suppress opposite-direction lane-change flicker:
    # RightLaneChange -> short low-confidence LeftLaneChange (or vice versa).
    if len(rows) > 1:
        opposite = {
            "LeftLaneChange": "RightLaneChange",
            "RightLaneChange": "LeftLaneChange",
        }
        flicker_max_sec = 1.5
        flicker_conf_max = 0.70
        for i in range(1, len(rows)):
            prev_cap = str(rows[i - 1].get("caption", ""))
            cur_cap = str(rows[i].get("caption", ""))
            if prev_cap not in opposite or cur_cap != opposite[prev_cap]:
                continue
            cur_dur = _safe_float(rows[i].get("duration", 0.0))
            cur_conf = _safe_float(rows[i].get("lane_confidence_score", 1.0))
            if cur_dur > flicker_max_sec or cur_conf >= flicker_conf_max:
                continue
            rows[i]["caption"] = prev_cap
            if prev_cap == "LeftLaneChange":
                rows[i]["lane_decision"] = "left_lane_change"
            elif prev_cap == "RightLaneChange":
                rows[i]["lane_decision"] = "right_lane_change"

    # Suppress short SlightlyShift right after lane-change completion:
    # a brief settling steer often appears as a shift artifact, so prune to LaneKeep.
    if len(rows) > 1:
        lane_change_caps = {"LeftLaneChange", "RightLaneChange"}
        shift_caps = {"SlightlyShiftLeft", "SlightlyShiftRight"}
        settle_max_sec = 1.5
        settle_max_gap_sec = 0.3
        settle_max_abs_lateral_rate = 0.18
        settle_max_abs_heading = 0.18
        for i in range(1, len(rows)):
            prev = rows[i - 1]
            cur = rows[i]
            prev_cap = str(prev.get("caption", ""))
            cur_cap = str(cur.get("caption", ""))
            if prev_cap not in lane_change_caps or cur_cap not in shift_caps:
                continue

            cur_dur = _safe_float(cur.get("duration", 0.0))
            if cur_dur > settle_max_sec:
                continue

            prev_end = _safe_float(prev.get("segment_start_time", 0.0)) + _safe_float(
                prev.get("duration", 0.0)
            )
            cur_start = _safe_float(cur.get("segment_start_time", 0.0))
            gap = max(0.0, cur_start - prev_end)
            if gap > settle_max_gap_sec:
                continue

            lat_rate = abs(_safe_float(cur.get("average_lane_lateral_change_rate_m_per_m"), 0.0))
            heading = abs(_safe_float(cur.get("heading_change_rad", 0.0)))
            if lat_rate > settle_max_abs_lateral_rate or heading > settle_max_abs_heading:
                continue

            cur["caption"] = "LaneKeep"
            cur["lane_decision"] = "keep_lane"

    # Expand very short shift windows using nearby LaneKeep context.
    rows = _expand_short_slightly_shift_rows(rows)

    i = 0
    while i < len(rows):
        seg = rows[i]
        caption = str(seg.get("caption", ""))
        min_ticks = LANE_MIN_LEN_TICKS_BY_CAPTION.get(caption, DEFAULT_LANE_MIN_LEN_TICKS)
        min_len_seconds = float(min_ticks) * dt
        if _safe_float(seg.get("duration", 0.0)) >= min_len_seconds:
            i += 1
            continue
        if bool(seg.get("_preserve_short_shift", False)):
            i += 1
            continue
        if _keep_short_slightly_shift(seg):
            i += 1
            continue

        has_prev = i > 0
        has_next = i < len(rows) - 1
        if has_prev:
            rows[i - 1] = _merge_lane_rows(rows[i - 1], seg, caption_from="left")
            rows.pop(i)
            continue
        if has_next:
            rows[i + 1] = _merge_lane_rows(seg, rows[i + 1], caption_from="right")
            rows.pop(i)
            continue
        i += 1

    if len(rows) <= 1:
        return pd.DataFrame(rows)

    merged = []
    cur = rows[0]
    for nxt in rows[1:]:
        if str(nxt.get("caption", "")) == str(cur.get("caption", "")):
            cur = _merge_lane_rows(cur, nxt, caption_from="left")
        else:
            merged.append(cur)
            cur = nxt
    merged.append(cur)
    for seg in merged:
        seg.pop("_preserve_short_shift", None)

    return pd.DataFrame(merged).sort_values("segment_start_time").reset_index(drop=True)
