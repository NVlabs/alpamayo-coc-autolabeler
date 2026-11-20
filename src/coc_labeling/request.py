# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared motion-text rendering and media-independent CoC messages."""

import numpy as np


def render_speed_text(data_dict, time_interval):
    """Render the standalone speed wording, including signed-zero formatting."""
    ego_hist_longitudinal_speed: np.ndarray = data_dict["ego_hist_longitudinal_speed"]
    ego_hist_lateral_speed: np.ndarray = data_dict["ego_hist_lateral_speed"]
    ego_fut_longitudinal_speed: np.ndarray = data_dict["ego_fut_longitudinal_speed"]
    ego_fut_lateral_speed: np.ndarray = data_dict["ego_fut_lateral_speed"]

    dt = float(time_interval)
    hist_count = int(ego_hist_longitudinal_speed.shape[0])
    fut_count = int(ego_fut_longitudinal_speed.shape[0])

    hist_timestamps = [-(hist_count - 1 - i) * dt for i in range(hist_count)]
    fut_timestamps = [(i + 1) * dt for i in range(fut_count)]

    def _format_speed_series(timestamps: list[float], values: np.ndarray, axis_name: str) -> str:
        """Render timestamped scalar series in ``[(t=.., axis=..), ...]`` format."""
        pairs = ", ".join(
            f"(t={ts:.1f}s, {axis_name}={float(v):.3f} m/s)" for ts, v in zip(timestamps, values)
        )
        return f"[{pairs}]"

    # Convert to text for separate history and future windows.
    hist_long_series = _format_speed_series(hist_timestamps, ego_hist_longitudinal_speed, "v_long")
    hist_lat_series = _format_speed_series(hist_timestamps, ego_hist_lateral_speed, "v_lat")
    fut_long_series = _format_speed_series(fut_timestamps, ego_fut_longitudinal_speed, "v_long")
    fut_lat_series = _format_speed_series(fut_timestamps, ego_fut_lateral_speed, "v_lat")
    ego_hist_text = (
        "History speed samples:\n"
        f"- Longitudinal (m/s): {hist_long_series}\n"
        f"- Lateral (m/s): {hist_lat_series}"
    )
    ego_fut_text = (
        "Future speed samples:\n"
        f"- Longitudinal (m/s): {fut_long_series}\n"
        f"- Lateral (m/s): {fut_lat_series}"
    )

    # convert to text for the whole segment
    ego_longitudinal_speed = np.concatenate(
        (ego_hist_longitudinal_speed, ego_fut_longitudinal_speed)
    )
    ego_lateral_speed = np.concatenate((ego_hist_lateral_speed, ego_fut_lateral_speed))
    all_timestamps = hist_timestamps + fut_timestamps
    all_long_series = _format_speed_series(all_timestamps, ego_longitudinal_speed, "v_long")
    all_lat_series = _format_speed_series(all_timestamps, ego_lateral_speed, "v_lat")
    ego_text = (
        "Ego speed time series (relative to event time t=0.0s):\n"
        f"- Longitudinal (m/s): {all_long_series}\n"
        f"- Lateral (m/s): {all_lat_series}\n"
        "Positive lateral speed means rightward motion, negative means leftward motion."
    )
    return ego_hist_text, ego_fut_text, ego_text


def render_meta_action_text(action_list, ts_list, fps, add_ts_prefix=False):
    """Render sampled meta-actions using the standalone wording."""
    result: list[str] = []
    use_ts_prefix = bool(add_ts_prefix)
    for idx, action in enumerate(action_list):
        base_action = (
            f"longitudinal: {action['Longitudinal']}, "
            f"lateral: {action['Lateral']}, lane: {action['Lane']}"
        )
        if use_ts_prefix:
            ts_sec = ts_list[idx] / fps
            # Relative timestamp with sign makes history/future separation explicit.
            formatted_action = f"(t={ts_sec:+.1f}s) {base_action}"
        else:
            formatted_action = base_action
        result.append(formatted_action)

    if use_ts_prefix:
        header = "\nMeta-actions by sampled timestamp (relative to event, unit: seconds):"
        return header + "\n" + "\n".join(result)

    return "\n" + "\n".join(result)


def build_coc_messages(prompt_module, data_loader_config, meta_text, ego_text, video=None):
    """Build ordered upstream messages; the caller supplies or binds camera frames."""

    def render(attribute):
        renderer = getattr(prompt_module, f"render_{attribute}", None)
        return (
            renderer(data_loader_config)
            if callable(renderer)
            else getattr(prompt_module, attribute)
        )

    return [
        {"role": "system", "m_type": "text", "content": prompt_module.system},
        {"role": "user", "m_type": "text", "content": render("images_prompt")},
        {"role": "user", "m_type": "video", "content": video},
        {"role": "user", "m_type": "text", "content": prompt_module.meta_action_prompt},
        {"role": "user", "m_type": "text", "content": meta_text},
        {"role": "user", "m_type": "text", "content": prompt_module.ego_state_prompt},
        {"role": "user", "m_type": "text", "content": ego_text},
        {"role": "user", "m_type": "text", "content": render("output_prompts")},
    ]


def matches_selection(action, start, end, selected_actions, min_duration, min_start, max_start):
    """Apply the standalone candidate action, duration and inclusive start bounds."""
    return (
        action in selected_actions
        and end - start >= min_duration
        and min_start <= start <= max_start
    )


def context_frame_ticks(num_ticks, event, history_ticks, future_ticks, interval_ticks):
    """Sample standalone frame context with its existing edge-clamping policy."""
    if interval_ticks <= 0:
        raise ValueError("meta_action_config.time_interval * fps must be > 0 for sampling.")
    event = max(0, min(int(event), num_ticks))
    history = tuple(range(max(event - history_ticks, 0), event, interval_ticks))
    future = tuple(range(event, min(event + future_ticks, num_ticks), interval_ticks))
    return history, future


def project_speed_to_heading(velocity, heading):
    """Project planar velocity with the standalone NumPy arithmetic order."""
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    longitudinal = velocity[:, 0] * cos_heading + velocity[:, 1] * sin_heading
    lateral = -velocity[:, 0] * sin_heading + velocity[:, 1] * cos_heading
    return longitudinal, lateral


def prompt_speed_components(velocity_xy, heading):
    """Preserve the standalone axis rotation, heading wrap and signed-zero text."""
    velocity = velocity_xy[:, [1, 0]].copy()
    velocity[:, 0] = -velocity[:, 0]
    rotated_heading = (heading + np.pi / 2) % (2 * np.pi)
    return project_speed_to_heading(velocity, rotated_heading)
