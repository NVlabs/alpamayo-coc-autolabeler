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

"""Segment generators for processing driving scenario data.

This module provides classes for generating and filtering segments from various data sources,
including JSON configuration files and Parquet files. The generators support filtering by
meta-actions, timestamps, and other criteria to create focused datasets for training and evaluation.

Classes:
    SegmentInfo: Dataclass for storing segment information
    SegmentListGenerator: Abstract base class for segment list generators
    ParsedSegmentGenerator: Generator that uses JSON configuration files
    ParquetSegmentGenerator: Generator that loads segments from Parquet files
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

from coc_labeling.data_loader.meta_action_loader import mapping_action2text
from coc_labeling.data_loader.vector_loader import VectorLoader
from coc_labeling.utils.data import get_clip_timing_info, get_scene_batch_from_scene_id_ts
from coc_labeling.utils.general_helpers import get_vlm_yaml_path
from coc_labeling.utils.type_check import is_path_exists


@dataclass
class SegmentInfo:
    """Dataclass for storing segment information.

    Attributes:
        clip_id: Unique identifier for the clip
        event_start_timestamp: Timestamp when the event starts (in microseconds)
        event_start_clipgt_index: Frame index at 10Hz for the event start
    """

    clip_id: str
    event_start_timestamp: int
    event_start_clipgt_index: int


class SegmentListGenerator(ABC):
    """Abstract base class for segment list generators.

    This class defines the interface for generating lists of segments from various data sources.
    Subclasses should implement the generate_segment_list method to provide specific functionality.

    Attributes:
        cfg: Configuration object containing data loading parameters
        verbose: Verbose logging configuration (optional)
        save_root: Root directory for saving outputs (optional)
        clip_id_all: List of all available clip IDs (set after parsing)
        clips_id_w_obstacles: List of clip IDs with obstacle files (optional)
        group: Group identifier for filtering (optional)
    """

    def __init__(
        self,
        cfg: DictConfig,
        verbose: Optional[Any] = None,
        save_root: Optional[str] = None,
    ) -> None:
        """Initialize the segment list generator.

        Args:
            cfg: Configuration object with data loading parameters
            verbose: Verbose logging configuration (optional)
            save_root: Root directory for saving outputs (optional)
        """
        self.cfg = cfg
        self.verbose = verbose
        self.save_root = save_root
        self.clip_id_all: Optional[List[str]] = None
        self.clips_id_w_obstacles: Optional[List[str]] = None
        self.group: Optional[str] = None
        self.vector_loader: Optional[VectorLoader] = None

    @abstractmethod
    def generate_segment_list(self) -> List[SegmentInfo]:
        """Generate a list of segments to process.

        Returns:
            List[SegmentInfo]: List of segment information objects
        """
        pass

    def _compute_event_start_clipgt_index(
        self,
        clip_id: str,
        event_start_timestamp: int,
        vector_loader: Optional[VectorLoader] = None,
    ) -> int:
        """Compute event_start_clipgt_index from event_start_timestamp using clip timing info.

        Args:
            clip_id: Identifier for the clip
            event_start_timestamp: Timestamp when the event starts (microseconds)
            vector_loader: Vector loader object for accessing timing information (optional)

        Returns:
            int: Frame index at 10Hz for the event start, or 0 if timing info unavailable
        """
        if vector_loader is None:
            return 0  # Default if no vector loader available

        start_micros, dt_micros = get_clip_timing_info(vector_loader.dataset, clip_id)
        if start_micros is None or dt_micros is None:
            return 0  # Default if timing info not available
        if dt_micros <= 0:
            raise ValueError(
                f"Invalid clip timing for {clip_id}: dt_micros must be > 0, got {dt_micros}."
            )

        # Compute frame index from timestamp
        try:
            event_start_clipgt_index = int((event_start_timestamp - start_micros) / dt_micros)
        except TypeError as exc:
            raise ValueError(
                "Invalid timestamp inputs for computing event_start_clipgt_index: "
                f"event_start_timestamp={event_start_timestamp}, "
                f"start_micros={start_micros}, dt_micros={dt_micros}"
            ) from exc
        return max(0, event_start_clipgt_index)  # Ensure non-negative frame index

    def _get_keyframe_cfg_value(self, key: str, default: Any = None) -> Any:
        """Read keyframe loader config with backward-compatible fallback."""
        keyframe_cfg = self.cfg.data_loader.get("keyframe", None)
        if keyframe_cfg is not None:
            value = keyframe_cfg.get(key, None)
            if value is not None:
                return value
        return self.cfg.data_loader.get(key, default)

    @staticmethod
    def _is_meaningful_output_file(path: str) -> bool:
        """Return whether path points to an existing output file with valid content."""
        try:
            if not is_path_exists(path) or not os.path.isfile(path):
                return False

            size_bytes = os.path.getsize(path)
            # Treat 0/1-byte files as empty placeholders (common failed-write artifact).
            if size_bytes <= 1:
                return False

            # Treat whitespace-only files as empty/incomplete.
            with open(path, encoding="utf-8") as f:
                file_text = f.read()
            if not file_text.strip():
                return False

            # For YAML outputs, require syntactically valid and non-empty content.
            if path.endswith((".yaml", ".yml")):
                try:
                    yaml_obj = OmegaConf.load(path)
                except Exception:
                    return False
                if yaml_obj is None:
                    return False
                if isinstance(yaml_obj, (DictConfig, ListConfig)):
                    return len(yaml_obj) > 0
            return True
        except (OSError, UnicodeDecodeError):
            return False


class ParsedSegmentGenerator(SegmentListGenerator):
    """Segment generator that uses existing parse_dataset and filter methods.

    This generator loads segments from JSON configuration files and applies various
    filtering operations including meta-action filtering, exclusion lists, and deduplication.

    Constants:
        MAX_EPISODE_START_DELTA_MICROS: Maximum time difference for episode alignment (1 second)
    """

    # Constants
    MAX_EPISODE_START_DELTA_MICROS = (
        1e6  # Max difference in start ts to match meta-action to episode
    )

    @staticmethod
    def _shift_duration_if_available(segment_data: Dict[str, Any], frame_delta: int) -> None:
        """Shift duration by ``frame_delta`` when duration is available and numeric."""
        duration = segment_data.get("duration")
        if duration is None:
            return
        try:
            segment_data["duration"] = int(duration) + int(frame_delta)
        except (TypeError, ValueError):
            logging.debug("Skip duration shift for non-numeric duration=%s", duration)

    @staticmethod
    def _normalize_segment_entry(
        segment_data: Dict[str, Any], fallback_meta_action: str
    ) -> Optional[Dict[str, Any]]:
        """Normalize segment keys across legacy and ActionRecord JSON schemas."""
        if not isinstance(segment_data, dict):
            return None

        normalized = segment_data.copy()
        meta_action = str(
            normalized.get("meta_action", normalized.get("action", fallback_meta_action))
        )
        clip_id = normalized.get("clip_id", normalized.get("filename"))
        start_frame = normalized.get("event_start_frame", normalized.get("start"))
        end_frame = normalized.get("event_end_frame", normalized.get("end"))
        duration = normalized.get("duration")

        if clip_id is None or start_frame is None:
            return None

        try:
            start_frame = int(start_frame)
            if end_frame is not None:
                end_frame = int(end_frame)
            if duration is not None:
                duration = int(duration)
        except (TypeError, ValueError):
            return None

        if end_frame is None and duration is not None:
            end_frame = start_frame + duration
        if duration is None and end_frame is not None:
            duration = end_frame - start_frame

        normalized["meta_action"] = meta_action
        normalized["clip_id"] = str(clip_id)
        normalized["event_start_frame"] = start_frame
        if end_frame is not None:
            normalized["event_end_frame"] = end_frame
        if duration is not None:
            normalized["duration"] = duration

        return normalized

    def __init__(
        self,
        cfg: DictConfig,
        verbose: Optional[bool] = None,
        save_root: Optional[str] = None,
    ) -> None:
        """Initialize the parsed segment generator.

        Args:
            cfg: Configuration object with data loading parameters
            verbose: Verbose logging configuration (optional)
            save_root: Root directory for saving outputs (optional)
        """
        super().__init__(cfg, verbose, save_root)
        # Internal state for parsing
        self.segments: Dict[str, List[Dict[str, Any]]] = {}
        self.episode_timestamps_df: Optional[pd.DataFrame] = None

    def parse_dataset(
        self, cfg: DictConfig, save_root: str, vector_loader: Optional[VectorLoader]
    ) -> None:
        """Parse dataset configuration and load segment data from JSON files.

        Args:
            cfg: Configuration object with updated parameters
            save_root: Root directory for saving outputs
            vector_loader: Vector loader for accessing timing information

        Raises:
            FileNotFoundError: If the segment configuration file doesn't exist
        """
        # update the cfg again
        self.cfg = cfg
        self.save_root = save_root
        self.vector_loader = vector_loader

        # save the group
        self.group = os.path.splitext(os.path.basename(self.cfg.data.segment_config_path))[0]

        # parse segments
        if not os.path.exists(self.cfg.data.segment_config_path):
            raise FileNotFoundError(f"File not found: {self.cfg.data.segment_config_path}")
        with open(self.cfg.data.segment_config_path) as f:
            raw_segments = json.load(f)

        self.segments = {}
        for fallback_meta_action, list_segment in raw_segments.items():
            if not isinstance(list_segment, list):
                continue
            normalized_list_segment = []
            for segment_data in list_segment:
                normalized_segment = self._normalize_segment_entry(
                    segment_data, fallback_meta_action
                )
                if normalized_segment is not None:
                    normalized_list_segment.append(normalized_segment)
            self.segments[fallback_meta_action] = normalized_list_segment

        logging.info(f"Loading segments from {self.cfg.data.segment_config_path}")
        for meta_action, list_segment in self.segments.items():
            logging.info(f"{len(list_segment)} segments found for {meta_action}")

        # parse clips from normalized entries (supports both legacy/new schemas)
        self.clip_id_all = sorted(
            {
                segment_data["clip_id"]
                for list_segment in self.segments.values()
                for segment_data in list_segment
                if "clip_id" in segment_data
            }
        )
        logging.info(f"Total number of clips: {len(self.clip_id_all)}")

        # parse filters for world-model files
        if self.cfg.data.wm_file_check is not None:
            with open(self.cfg.data.wm_file_check) as f:
                self.clips_id_w_obstacles = [line.strip() for line in f if line.strip()]
            logging.info(f"Number of clips with WM: {len(self.clips_id_w_obstacles)}")
        else:
            self.clips_id_w_obstacles = None

    def generate_segment_list(self) -> List[SegmentInfo]:
        """Generate segment list using existing parsing and filtering logic.

        This method applies a series of filters in sequence:
        1. Meta-action filtering
        2. Exclusion list filtering
        3. Inclusion list filtering
        4. Deduplication (if enabled)

        Returns:
            List[SegmentInfo]: Filtered list of segment information objects
        """
        # Dataset should already be parsed via parse_dataset() call

        # sometimes, we are only interested in segments of specific meta actions
        segment_list = self.filter_segment_w_metaaction(
            self._get_keyframe_cfg_value("meta_action_filter")
        )
        if self.verbose and self.verbose.verbose_data:
            logging.info(
                f"Number of segment list after filter with meta action is {len(segment_list)}"
            )

        # sometimes, we need to exclude clips that have already gone through meta-action labeling
        # but are missing necessary files to proceed (e.g. video recordings). exclude these before
        # the inclusion filter since that one checks for timestamps, which are not present if the
        # video is not present.
        segment_list = self.filter_segment_w_exclude_list(
            segment_list, self._get_keyframe_cfg_value("segment_filter_exclude")
        )
        if self.verbose and self.verbose.verbose_data:
            logging.info(
                "Number of segment list after filter with exclude segment list "
                f"is {len(segment_list)}"
            )

        # sometimes, we are only interested in segments with WM files available
        segment_list = self.filter_segment_w_list(
            segment_list, self._get_keyframe_cfg_value("segment_filter_include")
        )
        if self.verbose and self.verbose.verbose_data:
            logging.info(
                "Number of segment list after filter with include segment list "
                f"is {len(segment_list)}"
            )

        # filter out duplicate segments that have matching key:value pairs except for keys
        # that we want to concatenate or ignore.
        if self._get_keyframe_cfg_value("deduplicate_segments", False):
            # Concatenate action-ish fields while ignoring values unused by current output.
            segment_list = self.filter_segment_deduplicate(
                segment_list,
                concatenate_keys={
                    "action",
                    "meta_action",
                    "event_start_frame_unrounded",
                },
                ignore_keys={"end", "duration", "event_start_frame"},
                separator=",",
            )
            if self.verbose and self.verbose.verbose_data:
                logging.info(
                    f"Number of segment list after de-duplication filter is {len(segment_list)}"
                )

        # Convert dict segments to SegmentInfo dataclass
        segment_info_list = []
        for segment in segment_list:
            # Compute event_start_clipgt_index (10hz) from event_start_timestamp
            event_start_clipgt_index = self._compute_event_start_clipgt_index(
                segment["clip_id"],
                segment["event_start_timestamp"],
                self.vector_loader,
            )

            segment_info = SegmentInfo(
                clip_id=segment["clip_id"],  # clip_id is set in filter_segment_w_list
                event_start_timestamp=segment["event_start_timestamp"],
                event_start_clipgt_index=event_start_clipgt_index,
            )
            segment_info_list.append(segment_info)

        return segment_info_list

    def filter_segment_w_metaaction(
        self, meta_action_filter: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter segments by meta-action type.

        Args:
            meta_action_filter: List of meta-action types to include (None for all)

        Returns:
            Filtered list of segment dictionaries
        """
        # keep data with all meta actions if meta_action_filter is none
        if meta_action_filter is None:
            meta_action_filter = list(mapping_action2text.keys())

            # add compatibility to other segments found via other ways
            # instead of using basic rule-based meta actions as the filter
            meta_action_filter.append("Unknown")

        # sample the segments according to the meta actions
        segment_list = []
        for meta_action_filter_tmp in meta_action_filter:
            if meta_action_filter_tmp == "None":
                continue

            if meta_action_filter_tmp in self.segments:
                segment_target_tmp = self.segments[meta_action_filter_tmp]
                segment_list.extend(segment_target_tmp)

        return segment_list

    def filter_segment_w_exclude_list(
        self,
        segment_list: List[Dict[str, Any]],
        segment_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out segments that are in the exclusion list.

        Args:
            segment_list: List of segment dictionaries to filter
            segment_filter: List of clip IDs to exclude (None for no exclusions)

        Returns:
            Filtered list of segment dictionaries
        """
        # loop through all the segments and filter
        segment_list_filtered = []
        for segment_tmp in segment_list:
            clip_id = segment_tmp.get("clip_id", segment_tmp.get("filename"))
            if clip_id is None:
                continue
            if segment_filter is not None and clip_id in segment_filter:
                continue
            segment_list_filtered.append(segment_tmp)
        return segment_list_filtered

    def filter_segment_w_list(
        self,
        segment_list: List[Dict[str, Any]],
        segment_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter segments by inclusion list and apply various processing steps.

        This method performs several operations:
        1. Timestamp alignment with training episodes
        2. Frame rounding for better episode alignment
        3. Segment filtering by inclusion list
        4. World model file availability checking
        5. Finished segment detection
        6. Trajdata cache validation

        Args:
            segment_list: List of segment dictionaries to filter
            segment_filter: List of clip IDs to include (None for all)

        Returns:
            Filtered and processed list of segment dictionaries
        """
        # loop through all the segments and filter
        segment_list_filtered = []
        for segment_tmp in segment_list:
            clip_id = segment_tmp.get("clip_id", segment_tmp.get("filename"))
            key_meta_action = segment_tmp.get("meta_action", segment_tmp.get("action"))
            event_start_frame = segment_tmp.get("event_start_frame", segment_tmp.get("start"))
            if clip_id is None or key_meta_action is None or event_start_frame is None:
                if self.verbose and self.verbose.verbose_data:
                    logging.info("skip malformed segment missing clip/action/start keys")
                continue
            try:
                event_start_frame = int(event_start_frame)
            except (TypeError, ValueError):
                if self.verbose and self.verbose.verbose_data:
                    logging.info("skip malformed segment with non-integer start frame")
                continue

            # Modify timestamp to align with training episodes via one of two methods
            sft_timestamp_alignment_file = self._get_keyframe_cfg_value(
                "sft_timestamp_alignment_file"
            )
            round_event_start_frame_interval = self._get_keyframe_cfg_value(
                "round_event_start_frame_interval", 0
            )
            if sft_timestamp_alignment_file and round_event_start_frame_interval > 0:
                raise ValueError(
                    "Both sft_timestamp_alignment and round_event_start_frame_interval "
                    "should not be set together."
                )
            episode_anchor_timestamps = None
            if sft_timestamp_alignment_file:
                # load possible timestamps to align to from provided episode timestamp file
                episode_anchor_timestamps = self.get_episode_timestamps(
                    clip_id, sft_timestamp_alignment_file
                )
            elif round_event_start_frame_interval > 0:
                # Enforce frame-number rounding to better align with episodes.
                # `event_start_frame` is not an absolute timestamp here.
                segment_tmp["event_start_frame_unrounded"] = event_start_frame
                event_start_frame = round_event_start_frame_interval * round(
                    event_start_frame / round_event_start_frame_interval
                )
                segment_tmp["event_start_frame"] = event_start_frame
                # Keep duration consistent with the shifted end when present.
                self._shift_duration_if_available(
                    segment_tmp,
                    segment_tmp["event_start_frame_unrounded"] - event_start_frame,
                )
                if self.verbose and self.verbose.verbose_data:
                    logging.info(
                        f"{clip_id} {key_meta_action} event_start_frame "
                        f"rounded to {event_start_frame} "
                        f"from {segment_tmp['event_start_frame_unrounded']} using "
                        f"interval {round_event_start_frame_interval}"
                    )

            # minimal to start from event_start 20
            clip_id_w_ts = f"{clip_id}_{event_start_frame:02d}"
            segment_tmp["clip_id_w_ts"] = clip_id_w_ts
            logging.debug(
                "filter_segment_w_list: episode_anchor_timestamps=%s, segment_tmp=%s",
                episode_anchor_timestamps,
                segment_tmp,
            )

            # Populate timestamp
            segment_tmp = self.populate_timestamp(segment_tmp, episode_anchor_timestamps)
            if segment_tmp is None:
                if self.verbose and self.verbose.verbose_data:
                    logging.info(f"skip {clip_id_w_ts} due to the lack of timestamp")
                continue
            clip_id_w_ts = segment_tmp["clip_id_w_ts"]
            event_start_frame = segment_tmp["event_start_frame"]

            # Filter out {clip_id}_{start_frame} segments specified in config segment_filter
            if segment_filter is not None and clip_id_w_ts not in segment_filter:
                if self.verbose and self.verbose.verbose_data:
                    logging.info(f"skip {clip_id_w_ts} due to segment_filter")
                continue

            # skip the ones that do not have obstacle files
            if (
                self.clips_id_w_obstacles is not None
                and self._get_keyframe_cfg_value("filter_clip_missing_wm", False)
                and clip_id not in self.clips_id_w_obstacles
            ):
                if self.verbose and self.verbose.verbose_data:
                    logging.info(f"skip {clip_id_w_ts} due to the lack of obstacle file")
                continue

            # skip the ones that are already finished
            final_save_file_1 = os.path.join(
                self.save_root, key_meta_action, clip_id_w_ts, "behavior_planning.yaml"
            )
            final_save_file_2 = get_vlm_yaml_path(self.save_root, segment_tmp)
            if self._is_meaningful_output_file(
                final_save_file_1
            ) or self._is_meaningful_output_file(final_save_file_2):
                if self.verbose and self.verbose.verbose_data:
                    logging.info(f"skip {clip_id_w_ts} as it is already finished")
                continue
            else:
                if self.vector_loader is not None:
                    dataset = self.vector_loader.dataset
                    scene_ts_idx_map = self.vector_loader.scene_ts_idx_map
                    if dataset is None or scene_ts_idx_map is None:
                        if self.verbose and self.verbose.verbose_data:
                            logging.info(f"skip {clip_id_w_ts} due to uninitialized vector loader")
                        continue
                    # check if this segment falls into this trajdata cache
                    try:
                        _ = get_scene_batch_from_scene_id_ts(
                            f"{clip_id}_{event_start_frame}",
                            dataset,
                            scene_ts_idx_map,
                        )

                    # segment not in this trajdata cache group
                    # Can throw TypeError/KeyError depending on mapping internals.
                    except (TypeError, KeyError):
                        if self.verbose and self.verbose.verbose_data:
                            logging.info(f"skip {clip_id_w_ts} as it is not in the trajdata cache")
                        continue

            segment_list_filtered.append(segment_tmp)

        return segment_list_filtered

    def filter_segment_deduplicate(
        self,
        segment_list: List[Dict[str, Any]],
        concatenate_keys: set[str],
        ignore_keys: set[str],
        separator: str,
    ) -> List[Dict[str, Any]]:
        """Remove duplicate segments by concatenating values in specified keys.

        Args:
            segment_list: List of segment dictionaries to deduplicate
            concatenate_keys: Set of keys whose values should be concatenated for duplicates
            ignore_keys: Set of keys to ignore when comparing segments
            separator: String to use when concatenating values

        Returns:
            Deduplicated list of segment dictionaries
        """
        unique_segments = {}
        for item in segment_list:
            comparison_keys = set(item.keys()) - concatenate_keys - ignore_keys
            comparison_tuple = tuple(sorted((k, item[k]) for k in comparison_keys))
            if comparison_tuple in unique_segments:
                if self.verbose and self.verbose.verbose_data:
                    logging.info(
                        f"  Duplicate clip {item['clip_id']} and timestamp "
                        f"{item['event_start_timestamp']} found"
                    )
                existing = unique_segments[comparison_tuple]
                merged = existing.copy()
                # Update concatenated keys only; keep other values from the first item.
                for key in concatenate_keys:
                    old_val = existing.get(key, "null")
                    new_val = item.get(key, "null")
                    merged[key] = f"{old_val}{separator}{new_val}"
                unique_segments[comparison_tuple] = merged
            else:
                unique_segments[comparison_tuple] = item.copy()
        return list(unique_segments.values())

    def populate_timestamp(
        self, segment_data: Dict[str, Any], episode_timestamps: Optional[List[int]]
    ) -> Optional[Dict[str, Any]]:
        """Compute and populate timestamp information for a segment.

        Args:
            segment_data: Dictionary containing segment information
            episode_timestamps: List of episode timestamps for alignment (optional)

        Returns:
            Updated segment data with timestamp info, or None if timing unavailable
        """
        # Compute the event_start_timestamp from the cached scene start and dt values.
        # If a list of episode_timestamps is provided, find the closest episode timestamp to the
        # calculated event_start_timestamp for the current clip_id.
        if self.vector_loader is None:
            return None

        start_micros, dt_micros = get_clip_timing_info(
            self.vector_loader.dataset, segment_data["clip_id"]
        )
        if start_micros is None or dt_micros is None:
            return None
        idx_in_clip = segment_data["event_start_frame"]

        segment_data["event_start_timestamp"] = int(start_micros + idx_in_clip * dt_micros)
        segment_data["clip_start_micros"] = start_micros
        segment_data["clip_dt_micros"] = dt_micros
        logging.debug(
            "segment_data=%s, event_start_timestamp=%s, clip_start_micros=%s, clip_dt_micros=%s",
            segment_data,
            segment_data["event_start_timestamp"],
            start_micros,
            dt_micros,
        )
        if episode_timestamps is not None:
            if len(episode_timestamps) == 0:
                return None
            evt_start_ts = segment_data["event_start_timestamp"]
            closest_episode_timestamp = self.find_closest_timestamp(
                evt_start_ts, episode_timestamps
            )
            log_message = "Nearest episode found"
            if closest_episode_timestamp is None:
                closest_episode_timestamp = min(episode_timestamps)
                log_message = "All episodes start later, picking lowest"
            elif (
                abs(closest_episode_timestamp - evt_start_ts) > self.MAX_EPISODE_START_DELTA_MICROS
            ):
                closest_episode_timestamp = min(episode_timestamps)
                log_message = "Nearest episode too far, picking lowest"
            if self.verbose and self.verbose.verbose_data:
                logging.info(
                    f"{log_message}: {segment_data['clip_id']}, {evt_start_ts} "
                    f"-- closest={closest_episode_timestamp} "
                    f"delta={evt_start_ts - closest_episode_timestamp}"
                )
            # Save the meta-action event_start info in case we need it later
            segment_data["action_start_timestamp"] = evt_start_ts
            segment_data["action_start_frame"] = segment_data["event_start_frame"]
            segment_data["event_start_timestamp"] = closest_episode_timestamp
            segment_data["event_start_frame"] = int(
                (closest_episode_timestamp - start_micros) / dt_micros
            )
            # End and duration are not currently used; preserve the same end by
            # updating duration for this episode alignment scheme.
            # For now, we can try to preserve the same end as before, so we update duration.
            self._shift_duration_if_available(
                segment_data,
                segment_data["action_start_frame"] - segment_data["event_start_frame"],
            )
            # Set these to ensure old references continue to work
            # segment_data["start"] = segment_data["event_start_frame"]
            segment_data["clip_id_w_ts"] = (
                f"{segment_data['clip_id']}_{segment_data['event_start_frame']:02d}"
            )
        return segment_data

    def get_episode_timestamps(self, clip_id: str, ts_file: str) -> List[int]:
        """Get list of episode timestamps associated with a clip_id.

        Args:
            clip_id: Identifier for the clip
            ts_file: Path to the timestamp file (Parquet format)

        Returns:
            List of timestamps in microseconds
        """
        if self.episode_timestamps_df is None:
            self.episode_timestamps_df = pd.read_parquet(ts_file)
            logging.info(f"Loaded {self.episode_timestamps_df.shape[0]} entries from {ts_file}.")
        filtered = self.episode_timestamps_df.loc[
            self.episode_timestamps_df["clip_id"] == clip_id, "time_us"
        ]
        if len(filtered) == 0:
            logging.error(f"Clip ID {clip_id} not found in {ts_file}.")
        return filtered.tolist()

    def find_closest_timestamp(self, ts: int, ts_list: List[int]) -> Optional[int]:
        """Find closest timestamp in ts_list that is less than or equal to the provided timestamp.

        Args:
            ts: Target timestamp to find closest match for
            ts_list: List of timestamps to search (not necessarily sorted)

        Returns:
            Optional[int]: Closest timestamp <= ts, or None if no valid timestamps found
        """
        values_below = [x for x in ts_list if x <= ts]
        if len(values_below) == 0:
            return None
        return min(values_below, key=lambda x: ts - x)


class ParquetSegmentGenerator(SegmentListGenerator):
    """Segment generator that loads segment list from a parquet file.

    This generator provides a simpler interface for loading segments from Parquet files
    compared to the JSON-based ParsedSegmentGenerator. It supports group-based filtering
    and basic segment validation.
    """

    def __init__(
        self,
        cfg: DictConfig,
        verbose: Optional[bool] = None,
        save_root: Optional[str] = None,
    ) -> None:
        """Initialize the parquet segment generator.

        Args:
            cfg: Configuration object with data loading parameters
            verbose: Verbose logging configuration (optional)
            save_root: Root directory for saving outputs (optional)
        """
        super().__init__(cfg, verbose, save_root)
        # Internal state for parsing
        self.segment_list_df: Optional[pd.DataFrame] = None
        self.segments: Optional[pd.DataFrame] = None
        self._normalized_clip_id_column = "__clip_id"
        self._normalized_timestamp_column = "__event_start_timestamp"

    @staticmethod
    def _to_str_list(value: Any) -> List[str]:
        """Normalize a string or list-like config value to a list of strings."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple, ListConfig)):
            return [str(v) for v in value]
        return [str(value)]

    def _resolve_timestamp_candidates(
        self, timestamp_column_cfg: Any, parquet_files: List[str]
    ) -> List[List[str]]:
        """Resolve timestamp-column candidates per parquet file.

        Supported forms:
        - string: same column name for every parquet file
        - list[str] with len == number of parquet files: order-based one column per file
        - list[str] with len != number of parquet files: same fallback candidates for every file
        - list[list[str]] with len == number of parquet files: explicit fallback candidates per file
        """
        if isinstance(timestamp_column_cfg, str):
            return [[timestamp_column_cfg] for _ in parquet_files]

        if isinstance(timestamp_column_cfg, (list, tuple, ListConfig)):
            values = list(timestamp_column_cfg)
            if len(values) == 0:
                raise ValueError("segment_list_parquet_timestamp_column cannot be empty.")

            # Explicit per-file fallback candidates.
            if all(isinstance(v, (list, tuple, ListConfig)) for v in values):
                if len(values) != len(parquet_files):
                    raise ValueError(
                        "When using nested timestamp columns, "
                        "segment_list_parquet_timestamp_column length must match "
                        "segment_list_parquet_file length."
                    )
                return [[str(candidate) for candidate in per_file] for per_file in values]

            # Flat list[str]: order-based if lengths match, otherwise shared fallback list.
            str_values = [str(v) for v in values]
            if len(str_values) == len(parquet_files):
                return [[col] for col in str_values]
            return [str_values for _ in parquet_files]

        return [[str(timestamp_column_cfg)] for _ in parquet_files]

    def parse_dataset(
        self, cfg: DictConfig, save_root: str, vector_loader: Optional[VectorLoader]
    ) -> None:
        """Parse dataset configuration and load segment data from Parquet file.

        Args:
            cfg: Configuration object with updated parameters
            save_root: Root directory for saving outputs
            vector_loader: Vector loader for accessing timing information

        Raises:
            FileNotFoundError: If the parquet file doesn't exist
        """
        # Update the cfg again in case they have changed
        self.cfg = cfg
        self.save_root = save_root
        self.vector_loader = vector_loader
        # Use shorter names for config options
        parquet_file_from_data = self.cfg.data.get("segment_list_parquet_file", None)
        parquet_files_cfg = (
            parquet_file_from_data
            if parquet_file_from_data is not None
            else self._get_keyframe_cfg_value("segment_list_parquet_file")
        )
        parquet_files = self._to_str_list(parquet_files_cfg)
        if len(parquet_files) == 0:
            raise ValueError("No parquet file configured for segment list loading.")
        clip_id_column = str(self._get_keyframe_cfg_value("segment_list_parquet_clip_id_column"))
        timestamp_column_cfg = self._get_keyframe_cfg_value("segment_list_parquet_timestamp_column")
        timestamp_candidates_per_file = self._resolve_timestamp_candidates(
            timestamp_column_cfg, parquet_files
        )
        # Load the entire segment list parquet file if it has not been loaded already.
        if self.segment_list_df is None:
            loaded_dfs: List[pd.DataFrame] = []
            for parquet_file, ts_candidates in zip(parquet_files, timestamp_candidates_per_file):
                if not os.path.exists(parquet_file):
                    raise FileNotFoundError(f"Parquet file not found: {parquet_file}")

                source_df = cast(pd.DataFrame, pd.read_parquet(parquet_file))
                if clip_id_column not in source_df.columns:
                    raise KeyError(
                        f"Clip id column '{clip_id_column}' not found in {parquet_file}. "
                        f"Available columns: {list(source_df.columns)}"
                    )

                timestamp_column = next(
                    (col for col in ts_candidates if col in source_df.columns), None
                )
                if timestamp_column is None:
                    raise KeyError(
                        f"None of timestamp columns {ts_candidates} found in {parquet_file}. "
                        f"Available columns: {list(source_df.columns)}"
                    )

                normalized_df = pd.DataFrame(
                    {
                        self._normalized_clip_id_column: source_df[clip_id_column].astype(str),
                        self._normalized_timestamp_column: source_df[timestamp_column],
                    }
                )
                loaded_dfs.append(normalized_df)
                logging.info(
                    "Loaded %d entries from %s using timestamp column '%s'.",
                    normalized_df.shape[0],
                    parquet_file,
                    timestamp_column,
                )

            self.segment_list_df = cast(pd.DataFrame, pd.concat(loaded_dfs, ignore_index=True))
            logging.info(
                "Total loaded entries from %d parquet file(s): %d.",
                len(parquet_files),
                self.segment_list_df.shape[0],
            )
        if self.segment_list_df is None:
            raise RuntimeError("Segment parquet data did not load correctly.")
        segment_list_df: pd.DataFrame = cast(pd.DataFrame, self.segment_list_df)
        # Check if we are processing a group folder.
        basename = os.path.basename(self.save_root)
        if len(basename) == 4 and all(c in "0123456789abcdef" for c in basename):
            self.group = basename
            clip_ids = segment_list_df[self._normalized_clip_id_column].astype(str)
            group_mask: list[bool] = clip_ids.str.startswith(self.group).tolist()
            segments_df: pd.DataFrame = cast(pd.DataFrame, segment_list_df.loc[group_mask])
            logging.info(f"Number of segments for group {self.group}: {segments_df.shape[0]}")
        else:
            self.group = None
            segments_df = segment_list_df
        # Remove duplicates
        subset_columns: list[str] = [
            self._normalized_clip_id_column,
            self._normalized_timestamp_column,
        ]
        segments_df = segments_df.drop_duplicates(subset=subset_columns)
        logging.info(f"Number of segments remaining after deduplication: {segments_df.shape[0]}")
        self.segments = segments_df
        self.clip_id_all = (
            segments_df[self._normalized_clip_id_column].astype(str).unique().tolist()
        )

    def check_finished(self, clip_id: str, timestamp: int) -> bool:
        """Check if the segment is already finished by looking for output files.

        Args:
            clip_id: Identifier for the clip
            timestamp: Timestamp for the segment

        Returns:
            bool: True if the segment is finished (output files exist), False otherwise
        """
        if self.save_root is None:
            raise RuntimeError("save_root is not set. Call parse_dataset() first.")
        final_save_file = get_vlm_yaml_path(
            self.save_root, {"clip_id": clip_id, "event_start_timestamp": timestamp}
        )
        return self._is_meaningful_output_file(final_save_file)

    def check_trajdata_exists(self, clip_id: str, clipgt_index: int) -> bool:
        """Check if the segment is available in the trajdata cache.

        Args:
            clip_id: Identifier for the clip
            clipgt_index: Index relative to the cached scene start, assuming sampled at 10Hz

        Returns:
            bool: True if the segment exists in trajdata cache, False otherwise
        """
        if self.vector_loader is None:
            return False

        dataset = self.vector_loader.dataset
        scene_ts_idx_map = self.vector_loader.scene_ts_idx_map
        if dataset is None or scene_ts_idx_map is None:
            return False

        # check if this segment falls into this trajdata cache
        try:
            _ = get_scene_batch_from_scene_id_ts(
                clip_id + f"_{clipgt_index}",
                dataset,
                scene_ts_idx_map,
            )
        # If segment is not found, can throw TypeError/KeyError depending on mapping internals.
        except (TypeError, KeyError):
            return False
        return True

    def generate_segment_list(self) -> List[SegmentInfo]:
        """Generate segment list by loading from parquet file.

        This method processes segments from the loaded parquet data, filtering out
        finished segments and those not available in the trajdata cache.

        Returns:
            List[SegmentInfo]: List of valid segment information objects
        """
        # Use shorter names for config options
        # Generate segment info list
        segment_info_list = []
        if self.segments is None:
            raise RuntimeError("Segments are not initialized. Call parse_dataset() first.")
        for _, row in self.segments.iterrows():
            clip_id = str(row[self._normalized_clip_id_column])
            timestamp = int(row[self._normalized_timestamp_column])
            # Skip the ones that are already finished
            if self.check_finished(clip_id, timestamp):
                if self.verbose and self.verbose.verbose_data:
                    logging.info(f"Skip {clip_id}, {timestamp} as it is already finished")
                continue
            # Compute event_start_clipgt_index (10hz) from timestamp
            event_start_clipgt_index = self._compute_event_start_clipgt_index(
                clip_id, timestamp, self.vector_loader
            )
            # Skip the ones that are not in the trajdata cache
            if not self.check_trajdata_exists(clip_id, event_start_clipgt_index):
                if self.verbose and self.verbose.verbose_data:
                    logging.info(
                        f"Skip {clip_id}, {event_start_clipgt_index}, "
                        f"{timestamp} as it is not in the trajdata cache"
                    )
                continue
            # Create segment info
            segment_info = SegmentInfo(
                clip_id=clip_id,
                event_start_timestamp=timestamp,
                event_start_clipgt_index=event_start_clipgt_index,
            )
            segment_info_list.append(segment_info)
        logging.info(
            f"Number of segments after finished and trajdata checks: {len(segment_info_list)}"
        )
        return segment_info_list
