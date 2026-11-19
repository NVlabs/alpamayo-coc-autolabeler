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

"""Select keyframe from ego meta-action text files.

This script parses per-clip ``.txt`` files, extracts ego actions with start/end
timestamps, filters by target actions and temporal constraints, and writes:
1) all matched segments
2) sampled segments balanced by action type

It supports both legacy action names (e.g. ``LeftTurn``) and current canonical
snake_case action names (e.g. ``turn_left``).
"""

# pylint: disable=import-error

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import MutableMapping

from tqdm import tqdm

from coc_labeling.request import matches_selection

from coc_labeling.data_loader.meta_action_loader import (
    meta_action_lane,
    meta_action_lateral,
    meta_action_longitu,
    normalize_meta_action,
    parse_line,
)

# Canonical meta actions, kept in sync with meta_action_loader.
ALL_META_ACTIONS: list[str] = [
    *meta_action_longitu,
    *meta_action_lateral,
    *meta_action_lane,
]

# For canonical actions, allow using all available segments if under target count.
RESERVE_ALL_ACTIONS: set[str] = set(ALL_META_ACTIONS)

# By default, extract all canonical action types.
DEFAULT_ACTIONS: list[str] = ALL_META_ACTIONS.copy()
DEFAULT_MIN_DURATION = 10


class InsufficientActionSamplesError(ValueError):
    """Raised when non-reserve actions do not meet required sample count."""


@dataclass
class ActionRecord:
    """Structured action span extracted from one clip file."""

    meta_action: str
    clip_id: str
    event_start_frame: int
    event_end_frame: int
    duration: int


def canonicalize_action(action: str) -> str:
    """Map raw/legacy action labels to canonical keys used for filtering."""
    normalized = normalize_meta_action(action)
    if normalized != action.strip():
        return normalized
    # Keep compatibility for unknown actions by using a stable normalized token.
    return action.replace(" ", "").lower()


def parse_txt_file(
    file_path: Path,
    selected_actions: set[str],
    min_duration: int,
    min_start_ts: int,
    max_start_ts: int,
) -> list[ActionRecord]:
    """Extract action spans from one meta-action file under filter constraints."""
    records: list[ActionRecord] = []
    with file_path.open("r", encoding="utf-8") as txt_file:
        for line in txt_file:
            parsed = parse_line(line)
            if parsed is None:
                continue
            action_raw, start, end = parsed
            duration = end - start
            action = canonicalize_action(action_raw)
            if matches_selection(
                action, start, end, selected_actions, min_duration, min_start_ts, max_start_ts
            ):
                records.append(
                    ActionRecord(
                        meta_action=action,
                        clip_id=file_path.stem,
                        event_start_frame=start,
                        event_end_frame=end,
                        duration=duration,
                    )
                )
    return records


def init_action_buckets(selected_actions: set[str]) -> dict[str, list[ActionRecord]]:
    """Create empty record buckets for each target action."""
    return {action: [] for action in sorted(selected_actions)}


def print_duration_stats(action_buckets: dict[str, list[ActionRecord]], stage: str) -> None:
    """Print per-action count and duration summary."""
    logging.info("Printing statistics %s", stage)
    for action, records in action_buckets.items():
        if not records:
            logging.info("%s, 0 actions", action)
            continue
        durations = [record.duration for record in records]
        mean_value = statistics.mean(durations)
        std_dev = statistics.pstdev(durations)
        min_value = min(durations)
        max_value = max(durations)
        logging.info(
            "%s, %d actions, duration: mean %.1f, std_dev %.1f, min %.1f, max %.1f",
            action,
            len(records),
            mean_value,
            std_dev,
            min_value,
            max_value,
        )


def write_action_json(path: Path, action_buckets: dict[str, list[ActionRecord]]) -> None:
    """Serialize action buckets to JSON."""
    payload = {
        action: [asdict(record) for record in records] for action, records in action_buckets.items()
    }
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=4)


def load_action_json(path: Path, selected_actions: set[str]) -> dict[str, list[ActionRecord]]:
    """Load cached action JSON and keep only selected action types."""
    action_buckets = init_action_buckets(selected_actions)
    with path.open("r", encoding="utf-8") as input_file:
        cached_data = json.load(input_file)

    for key, values in cached_data.items():
        if not isinstance(values, list):
            continue
        for item in values:
            if not isinstance(item, MutableMapping):
                continue
            action = canonicalize_action(str(item.get("meta_action", item.get("action", key))))
            if action not in action_buckets:
                continue
            try:
                start_key = "event_start_frame" if "event_start_frame" in item else "start"
                end_key = "event_end_frame" if "event_end_frame" in item else "end"
                clip_id_key = "clip_id" if "clip_id" in item else "filename"
                start = int(item[start_key])
                end = int(item[end_key])
                duration = int(item.get("duration", end - start))
                clip_id = str(item[clip_id_key])
            except (KeyError, TypeError, ValueError):
                continue
            action_buckets[action].append(
                ActionRecord(
                    meta_action=action,
                    clip_id=clip_id,
                    event_start_frame=start,
                    event_end_frame=end,
                    duration=duration,
                )
            )

    return action_buckets


def collect_action_records(
    folder: Path,
    selected_actions: set[str],
    min_duration: int,
    min_start_ts: int,
    max_start_ts: int,
) -> dict[str, list[ActionRecord]]:
    """Scan folder and extract all matching action records from ``.txt`` files."""
    action_buckets = init_action_buckets(selected_actions)
    txt_files = sorted(path for path in folder.iterdir() if path.suffix == ".txt")
    for txt_file in tqdm(txt_files, desc="Parsing meta-action files"):
        for record in parse_txt_file(
            txt_file,
            selected_actions,
            min_duration,
            min_start_ts,
            max_start_ts,
        ):
            action_buckets[record.meta_action].append(record)
    return action_buckets


def balance_action_records(
    action_buckets: dict[str, list[ActionRecord]],
    target_count: int,
) -> dict[str, list[ActionRecord]]:
    """Cap each action type to ``target_count`` by longest duration first."""
    balanced: dict[str, list[ActionRecord]] = {}
    for action, records in action_buckets.items():
        count = len(records)
        if count < target_count:
            if action not in RESERVE_ALL_ACTIONS:
                logging.warning(
                    "Action '%s' is not a built-in meta action and has %d segments "
                    "below --target_count=%d. Check the action name, or lower "
                    "--target_count for an intentional custom action.",
                    action,
                    count,
                    target_count,
                )
                raise InsufficientActionSamplesError(
                    f"Action '{action}' is not a built-in meta action and has {count} "
                    f"segments below --target_count={target_count}. Check the action "
                    "name, or lower --target_count for an intentional custom action."
                )
            balanced[action] = records
            logging.info(
                "Found %d actions for '%s' below target %d; using all available segments.",
                count,
                action,
                target_count,
            )
            continue
        if count > target_count:
            records = sorted(records, key=lambda record: record.duration, reverse=True)
            records = records[:target_count]
        balanced[action] = records
    return balanced


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Parse ego meta-action txt files, select target action spans, and "
            "balance samples across action types."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--meta_action_dir",
        required=True,
        help=(
            "Folder containing per-clip txt files with ego meta-action spans "
            "(for example: ./data/meta_action/final_outputs). "
        ),
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        metavar="ACTION",
        default=DEFAULT_ACTIONS,
        help=(
            "Target actions to extract. Supports canonical names (e.g. turn_left) "
            "and legacy aliases (e.g. LeftTurn)."
        ),
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=DEFAULT_MIN_DURATION,
        help="Minimum action duration in frames.",
    )
    parser.add_argument(
        "--min_start_ts",
        type=int,
        default=20,
        help="Inclusive lower bound for action start frame index.",
    )
    parser.add_argument(
        "--max_start_ts",
        type=int,
        default=135,
        help="Inclusive upper bound for action start frame index.",
    )
    parser.add_argument(
        "--target_count",
        type=int,
        default=500000,
        help="Maximum number of segments to keep per action type after balancing.",
    )
    parser.add_argument(
        "--output_dir",
        default="./experiments",
        help=(
            "Directory where output JSON files are written "
            "(segments_relative_timestamp_all.json and "
            "segments_relative_timestamp_sampled.json)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run keyframe auto-selection pipeline."""
    args = parse_args()
    selected_actions = {canonicalize_action(action) for action in args.actions}

    meta_action_dir = Path(args.meta_action_dir)
    if not meta_action_dir.exists() or not meta_action_dir.is_dir():
        raise FileNotFoundError(
            f"Input meta_action_dir not found or not a directory: {meta_action_dir}"
        )

    output_dir = Path(args.output_dir) if args.output_dir else Path("./")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_segments_path = output_dir / "segments_relative_timestamp_all.json"
    sampled_segments_path = output_dir / "segments_relative_timestamp_sampled.json"

    if all_segments_path.exists():
        logging.info("Loading cached segments from %s", all_segments_path)
        action_buckets = load_action_json(all_segments_path, selected_actions)
    else:
        action_buckets = collect_action_records(
            meta_action_dir,
            selected_actions,
            args.min_duration,
            args.min_start_ts,
            args.max_start_ts,
        )
        print_duration_stats(action_buckets, "before sampling")
        write_action_json(all_segments_path, action_buckets)
        logging.info("Saved all matching segments to %s", all_segments_path)

    try:
        balanced = balance_action_records(action_buckets, args.target_count)
    except InsufficientActionSamplesError as exc:
        logging.error("%s", exc)
        sys.exit(1)
    print_duration_stats(balanced, "after sampling")
    write_action_json(sampled_segments_path, balanced)
    logging.info("Processing complete. Output written to %s", sampled_segments_path)


if __name__ == "__main__":
    main()
