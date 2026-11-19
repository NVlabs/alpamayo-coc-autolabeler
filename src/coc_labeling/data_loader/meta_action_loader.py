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

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from coc_labeling.request import context_frame_ticks, render_meta_action_text

# V2 canonical meta-action labels (snake_case).
meta_action_longitu: List[str] = [
    "gentle_acceleration",
    "strong_acceleration",
    "gentle_deceleration",
    "strong_deceleration",
    "maintain_speed",
    "stop",
    "reverse",
]
meta_action_lateral: List[str] = [
    "reverse_right",
    "reverse_left",
    "steer_right",
    "steer_left",
    "sharp_steer_right",
    "sharp_steer_left",
    "go_straight",
]
meta_action_lane: List[str] = [
    "keep_lane",
    "left_lane_change",
    "right_lane_change",
    "slightly_shift_left",
    "slightly_shift_right",
    "turn_left",
    "turn_right",
    "follow_curve_left",
    "follow_curve_right",
]


# Human-readable captions for canonical labels.
_ACTION_DISPLAY_TEXT: Dict[str, str] = {
    "gentle_acceleration": "Gentle Acceleration",
    "strong_acceleration": "Strong Acceleration",
    "gentle_deceleration": "Gentle Deceleration",
    "strong_deceleration": "Strong Deceleration",
    "maintain_speed": "Maintain Speed",
    "stop": "Stop",
    "reverse": "Reverse",
    "reverse_right": "Reverse Right",
    "reverse_left": "Reverse Left",
    "steer_right": "Steer Right",
    "steer_left": "Steer Left",
    "sharp_steer_right": "Sharp Steer Right",
    "sharp_steer_left": "Sharp Steer Left",
    "go_straight": "Go Straight",
    "keep_lane": "Keep Lane",
    "left_lane_change": "Left Lane Change",
    "right_lane_change": "Right Lane Change",
    "slightly_shift_left": "Slightly Shift Left",
    "slightly_shift_right": "Slightly Shift Right",
    "turn_left": "Turn Left",
    "turn_right": "Turn Right",
    "follow_curve_left": "Follow Curve Left",
    "follow_curve_right": "Follow Curve Right",
    "none": "Not Available",
}


# Raw token aliases -> canonical V2 labels.
_ACTION_ALIASES: Dict[str, str] = {
    "KeepSpeed": "maintain_speed",
    "Wait": "stop",
    "Accelerate": "strong_acceleration",
    "Decelerate": "strong_deceleration",
    "Straight": "go_straight",
    "KeepLane": "keep_lane",
    "LaneKeep": "keep_lane",
    "LeftLaneChange": "left_lane_change",
    "RightLaneChange": "right_lane_change",
    "LeftTurn": "turn_left",
    "RightTurn": "turn_right",
}


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase/PascalCase token to snake_case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def normalize_meta_action(action: Optional[str]) -> str:
    """Normalize raw action tokens to canonical labels when possible.

    Behavior is intentionally unchanged:
    - None -> "none"
    - explicit alias -> mapped canonical label
    - camel/snake variant known in canonical set -> canonical snake_case
    - unknown token -> returned as-is
    """
    if action is None:
        return "none"
    action = action.strip()
    if action in _ACTION_ALIASES:
        return _ACTION_ALIASES[action]
    snake = _camel_to_snake(action)
    if snake in _ACTION_ALIASES:
        return _ACTION_ALIASES[snake]
    if snake in _ACTION_DISPLAY_TEXT:
        return snake
    return action


# Public mapping used by segment filtering logic; includes canonical + legacy aliases.
mapping_action2text: Dict[str, str] = {k: v for k, v in _ACTION_DISPLAY_TEXT.items()}
for raw, canonical in _ACTION_ALIASES.items():
    mapping_action2text[raw] = _ACTION_DISPLAY_TEXT.get(canonical, raw)
mapping_action2text["None"] = "Not Available"


def parse_line(line: str) -> Optional[Tuple[str, int, int]]:
    """Parse one meta-action line for Agent:<ego>.

    Expected example:
    "GentleAcceleration - Agent:<ego>, Start:0, End:35"
    """
    stripped = line.strip()
    if not stripped:
        return None
    if "Agent:<ego>" not in stripped:
        return None

    parts = stripped.split(" - ")
    if len(parts) < 2:
        return None

    meta_action = parts[0].strip()
    rest = parts[1].split(",")
    start: Optional[int] = None
    end: Optional[int] = None

    for token in rest:
        token = token.strip()
        if token.startswith("Start:"):
            try:
                start = int(token.split("Start:")[1])
            except ValueError:
                pass
        elif token.startswith("End:"):
            try:
                end = int(token.split("End:")[1])
            except ValueError:
                pass

    if start is None or end is None:
        return None
    return meta_action, start, end


class MetaActionLoader:
    """Load, slice, and stringify per-frame rule-based meta actions."""

    def __init__(self, meta_action_config: Any, data_config: Any) -> None:
        """Initialize meta-action loader with runtime/data configuration.

        Args:
            meta_action_config: Meta-action loader configuration (or None to disable).
            data_config: Dataset path configuration used to resolve files.
        """
        self.meta_action_config = meta_action_config
        self.data_config = data_config
        if meta_action_config is None:
            logging.info("Meta-action loader is not activated")

    def _debug_enabled(self) -> bool:
        """Return whether detailed meta-action parsing logs are enabled."""
        if self.meta_action_config is None:
            return False
        return bool(getattr(self.meta_action_config, "debug", False))

    def load_rule_meta_actions(self, meta_action_path: str) -> List[Dict[str, str]]:
        """Load per-frame meta actions across the full clip timeline."""
        max_frames = int(self.meta_action_config.num_total_frames)
        debug = self._debug_enabled()

        frame_meta_action: List[Dict[str, str]] = [
            {
                "Longitudinal": "Not Available",
                "Lateral": "Not Available",
                "Lane": "Not Available",
            }
            for _ in range(max_frames)
        ]

        parsed_line_count = 0
        malformed_line_count = 0
        out_of_bounds_count = 0
        unknown_action_count = 0
        max_end_seen = -1

        with open(meta_action_path, encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                parsed = parse_line(line)
                if parsed is None:
                    if line.strip():
                        malformed_line_count += 1
                    continue

                parsed_line_count += 1
                raw_action, frame_start, frame_end = parsed
                meta_action = normalize_meta_action(raw_action)
                max_end_seen = max(max_end_seen, frame_end)
                if self.meta_action_config.convert2text:
                    meta_action_str = mapping_action2text.get(meta_action, meta_action)
                else:
                    meta_action_str = meta_action

                # Guard against malformed / out-of-range ranges coming from source files.
                original_start, original_end = frame_start, frame_end
                if frame_end < frame_start:
                    if debug:
                        logging.debug(
                            "[MetaActionLoader] skip reversed range "
                            "in %s:%d action=%r start=%d, end=%d",
                            meta_action_path,
                            line_no,
                            raw_action,
                            frame_start,
                            frame_end,
                        )
                    continue
                frame_start = max(0, frame_start)
                frame_end = min(frame_end, max_frames)
                if (
                    original_start != frame_start
                    or original_end != frame_end
                    or frame_start >= max_frames
                    or frame_end <= 0
                ):
                    out_of_bounds_count += 1
                    if debug:
                        logging.debug(
                            "[MetaActionLoader] adjusted range "
                            "in %s:%d action=%r original=(%d, %d) effective=(%d, %d) max_frames=%d",
                            meta_action_path,
                            line_no,
                            raw_action,
                            original_start,
                            original_end,
                            frame_start,
                            frame_end,
                            max_frames,
                        )
                if frame_start >= frame_end:
                    # Range falls fully outside the valid frame window.
                    continue

                # Fill per-frame labels on [start, end), consistent with prior behavior.
                for frame_index in range(frame_start, frame_end):
                    if meta_action in meta_action_longitu:
                        frame_meta_action[frame_index]["Longitudinal"] = meta_action_str
                    # if commented out, currently we do not feed this signal into VLM
                    # elif meta_action in meta_action_lateral:
                    #     frame_meta_action[frame_index]["Lateral"] = meta_action_str
                    elif meta_action in meta_action_lane:
                        frame_meta_action[frame_index]["Lane"] = meta_action_str
                    else:
                        unknown_action_count += 1
                        if debug:
                            logging.debug(
                                "[MetaActionLoader] unknown or unused action token "
                                "in %s:%d raw=%r normalized=%r",
                                meta_action_path,
                                line_no,
                                raw_action,
                                meta_action,
                            )
                        # Keep compatibility for unknown/extra labels: skip silently.
                        continue

        if debug:
            logging.debug(
                "[MetaActionLoader] summary path=%s, parsed_lines=%d, "
                "malformed_nonempty_lines=%d, out_of_bounds_ranges=%d, "
                "unknown_action_hits=%d, max_end_seen=%d, max_frames=%d",
                meta_action_path,
                parsed_line_count,
                malformed_line_count,
                out_of_bounds_count,
                unknown_action_count,
                max_end_seen,
                max_frames,
            )
        elif malformed_line_count > 0 or out_of_bounds_count > 0:
            logging.warning(
                "[MetaActionLoader] detected malformed/out-of-bounds meta "
                "actions in %s (malformed_nonempty_lines=%d, "
                "out_of_bounds_ranges=%d, max_end_seen=%d, max_frames=%d). "
                "Set data_loader.meta_action.debug=true for detailed per-line logs.",
                meta_action_path,
                malformed_line_count,
                out_of_bounds_count,
                max_end_seen,
                max_frames,
            )

        return frame_meta_action

    def extract_segments(
        self, frames: List[Dict[str, str]], event_start_frame: Optional[int]
    ) -> Dict[str, Any]:
        """Extract history/future sampled windows around `event_start_frame`."""
        if len(frames) == 0:
            return {
                "hist_meta_actions": [],
                "fut_meta_actions": [],
                "all_meta_actions": [],
                "hist_ts": [],
                "fut_ts": [],
                "all_ts": [],
            }

        if event_start_frame is None:
            # Full-clip mode: keep all frames as a single sequence when no event anchor is provided.
            return {
                "hist_meta_actions": [],
                "fut_meta_actions": [],
                "all_meta_actions": frames,
                "hist_ts": [],
                "fut_ts": [],
                "all_ts": list(range(len(frames))),
            }

        hist_length_frame = int(
            self.meta_action_config.hist_length_sec * self.meta_action_config.fps
        )
        fut_length_frame = int(self.meta_action_config.fut_length_sec * self.meta_action_config.fps)
        frame_interval = int(self.meta_action_config.time_interval * self.meta_action_config.fps)
        if frame_interval <= 0:
            raise ValueError("meta_action_config.time_interval * fps must be > 0 for sampling.")
        event_start_frame = max(0, min(int(event_start_frame), len(frames)))
        hist_ticks, fut_ticks = context_frame_ticks(
            len(frames), event_start_frame, hist_length_frame, fut_length_frame, frame_interval
        )
        hist_meta_actions = [frames[t] for t in hist_ticks]
        fut_meta_actions = [frames[t] for t in fut_ticks]
        hist_ts = [t - event_start_frame for t in hist_ticks]
        fut_ts = [t - event_start_frame for t in fut_ticks]

        all_meta_actions = hist_meta_actions + fut_meta_actions
        all_ts = hist_ts + fut_ts

        return {
            "hist_meta_actions": hist_meta_actions,
            "fut_meta_actions": fut_meta_actions,
            "all_meta_actions": all_meta_actions,
            "hist_ts": hist_ts,
            "fut_ts": fut_ts,
            "all_ts": all_ts,
        }

    def process_meta_action(self, action_list: List[Dict[str, str]], ts_list: List[int]) -> str:
        """Convert sampled meta-action dicts into prompt-friendly text lines."""
        add_ts_prefix = bool(getattr(self.meta_action_config, "add_ts_prefix", False))
        fps = self.meta_action_config.fps if add_ts_prefix and action_list else None
        return render_meta_action_text(action_list, ts_list, fps, add_ts_prefix)

    def load(self, clip_id: str, event_start_frame: Optional[int] = None) -> Dict[str, Any]:
        """Load and format meta-action features for one clip/segment."""
        if self.meta_action_config is None:
            return {}

        meta_action_path = os.path.join(self.data_config.meta_action_dir, f"{clip_id}.txt")
        if not os.path.exists(meta_action_path):
            raise FileNotFoundError(f"Meta action file not found: {meta_action_path}")
        meta_actions = self.load_rule_meta_actions(meta_action_path)

        output_dict = self.extract_segments(meta_actions, event_start_frame)
        output_dict.update(
            {
                "hist_meta_action_str": self.process_meta_action(
                    output_dict["hist_meta_actions"], output_dict["hist_ts"]
                ),
                "fut_meta_action_str": self.process_meta_action(
                    output_dict["fut_meta_actions"], output_dict["fut_ts"]
                ),
                "all_meta_action_str": self.process_meta_action(
                    output_dict["all_meta_actions"], output_dict["all_ts"]
                ),
            }
        )
        return output_dict
