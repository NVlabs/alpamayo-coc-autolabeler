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

import logging
from typing import Any, cast

from omegaconf import DictConfig
from tqdm import tqdm

from coc_labeling.data_loader.meta_action_loader import MetaActionLoader
from coc_labeling.data_loader.segment_generators import (
    ParquetSegmentGenerator,
    ParsedSegmentGenerator,
)
from coc_labeling.data_loader.vector_loader import VectorLoader
from coc_labeling.data_loader.video_loader import VideoLoader
from coc_labeling.model_clients.runtime_config import ModelRuntimeConfig


class LabelingAgent:
    """Top-level coordinator for loading segment data and running a selected agent."""

    def __init__(
        self,
        cfg: DictConfig,
        save_root: str,
        verbose: DictConfig,
        mode: str = "vlm",
        ip_addr: str | None = None,
        runtime_config: ModelRuntimeConfig | None = None,
    ) -> None:
        """Initialize top-level labeling orchestrator and segment generator.

        Args:
            cfg: Full runtime configuration.
            save_root: Output root directory for generated labels.
            verbose: Verbosity configuration.
            mode: Agent mode, e.g. ``"vlm"``.
        """
        self.save_root = save_root
        self.cfg = cfg
        self.verbose = verbose
        self.runtime_config = runtime_config
        self.ip_addr = runtime_config.ip_addr if runtime_config is not None else ip_addr

        # Initialized here for static analyzers and safer control flow.
        self.video_loader: VideoLoader | None = None
        self.meta_action_loader: MetaActionLoader | None = None
        self.vector_loader: VectorLoader | None = None
        self.agent: Any = None

        # determine which agent to use
        self.mode = mode
        self.init_model()

        # determine which segment generator to use
        keyframe_cfg = self.cfg.data_loader.get("keyframe", None)
        segment_generator_type = (
            keyframe_cfg.get("segment_generator_type")
            if keyframe_cfg is not None and keyframe_cfg.get("segment_generator_type") is not None
            else self.cfg.data_loader.get("segment_generator_type")
        )

        if segment_generator_type == "parquet":
            self.segment_generator = ParquetSegmentGenerator(
                cfg=self.cfg, verbose=self.verbose, save_root=self.save_root
            )
        elif segment_generator_type == "json":
            self.segment_generator = ParsedSegmentGenerator(
                cfg=self.cfg, verbose=self.verbose, save_root=self.save_root
            )
        else:
            raise ValueError(
                "Unsupported data_loader.segment_generator_type "
                f"'{segment_generator_type}'. "
                "Supported values: 'parquet', 'json'."
            )

    def init_model(self) -> None:
        """Instantiate the configured model agent for the selected mode.

        Raises:
            ValueError: If ``self.mode`` is unsupported.
        """
        if self.mode == "vlm":  # VLM agent
            from coc_labeling.agents.vlm.factory import VLMFactory

            timeout_sec = (
                self.runtime_config.timeout_sec
                if self.runtime_config is not None
                else ModelRuntimeConfig().timeout_sec
            )
            self.agent = VLMFactory(
                model_name=self.cfg.model_name,
                agent_config=self.cfg.vlm_agent,
                vector_config=self.cfg.data_loader.vector,
                data_loader_config=self.cfg.data_loader,
                ip_addr=self.ip_addr,
                timeout_sec=timeout_sec,
            )
        else:
            raise ValueError(f"mode: {self.mode} not supported")

    def parse_dataset(self, cfg: DictConfig, save_root: str) -> None:
        """Initialize loaders and parse dataset metadata into segment generator.

        Args:
            cfg: Updated runtime configuration.
            save_root: Output root directory.
        """
        # update the cfg again
        self.cfg = cfg
        self.save_root = save_root

        # create video data loader
        if self.cfg.data_loader.video is not None:
            self.video_loader = VideoLoader(self.cfg.data_loader.video, self.cfg.data)
        else:
            self.video_loader = None

        # create meta action data loader
        if self.cfg.data_loader.meta_action is not None:
            self.meta_action_loader = MetaActionLoader(
                self.cfg.data_loader.meta_action, self.cfg.data
            )
        else:
            self.meta_action_loader = None

        # create world model data loader, used by LLM, and sometimes for VLM as well
        if self.cfg.data_loader.vector is not None:
            self.vector_loader = VectorLoader(self.cfg.data_loader.vector, self.cfg.data)
        else:
            self.vector_loader = None

        # Parse dataset using segment generator if available
        if self.segment_generator is not None:
            # Some generators accept None here; cast avoids false-positive IDE type errors.
            vector_loader = cast(Any, self.vector_loader)
            self.segment_generator.parse_dataset(self.cfg, self.save_root, vector_loader)

    def run(self) -> None:
        """Run labeling over all generated segments and save model outputs."""
        # Generate segment list using the configured segment generator
        segment_info_list = self.segment_generator.generate_segment_list()

        # loop through the remaining segments to be finished
        success_count = 0
        for segment_info in tqdm(segment_info_list, desc="Segment List Remaining"):
            logging.info(segment_info)

            # Build base segment data first so flow is valid even without vector data.
            segment_data: dict[str, Any] = {
                "clip_id": segment_info.clip_id,
                "event_start_timestamp": segment_info.event_start_timestamp,
                # NOTE this is not actually the frame number from the video.
                # It is an interval count at 10Hz from the start of the cached scene.
                "event_start_frame": segment_info.event_start_clipgt_index,
            }

            # Parse vector data (if enabled).
            if self.vector_loader is not None:
                data_dict = self.vector_loader.load(
                    segment_info.clip_id, segment_info.event_start_clipgt_index
                )
                segment_data.update(data_dict)

            # load video data
            try:
                if self.mode in ["vlm", "early_fusion"] and self.video_loader is not None:
                    if self.cfg.data_loader.video.event_start_from_source_fps:
                        # Video indexing is not guaranteed to be the same as clip indexing.
                        # Compute video_start_frame from event_start_timestamp and load
                        # the video using the computed video_start_frame.
                        try:
                            (
                                video_start_frame,
                                video_start_timestamp,
                            ) = self.video_loader.get_index_from_timestamp(
                                segment_data["clip_id"],
                                segment_data["event_start_timestamp"],
                            )
                        except (ValueError, IndexError, FileNotFoundError) as e:
                            logging.error(f"Error converting timestamp to index: {e}")
                            continue
                        video_data = self.video_loader.load(
                            segment_data["clip_id"],
                            video_start_frame,
                            event_start_timestamp=segment_data["event_start_timestamp"],
                        )
                        if self.verbose.verbose_data:
                            timestamp_delta = (
                                segment_data["event_start_timestamp"] - video_start_timestamp
                            )
                            logging.info(
                                "Using frame index %s for %s event_start_frame %s, "
                                "timestamp delta %s-%s = %s usec",
                                video_start_frame,
                                segment_data["clip_id"],
                                segment_data["event_start_frame"],
                                segment_data["event_start_timestamp"],
                                video_start_timestamp,
                                timestamp_delta,
                            )
                    else:
                        # Kept for compatibility with prior usage. Unless the video
                        # is cropped to the cached scene, frames can be misaligned.
                        video_data = self.video_loader.load(
                            segment_data["clip_id"],
                            segment_data["event_start_frame"],
                            event_start_timestamp=segment_data["event_start_timestamp"],
                        )
                    # Convert VideoLoadResult dataclass to dictionary for backward compatibility
                    segment_data.update(
                        {
                            "hist_fpv_frames": video_data.hist_fpv_frames,
                            "fut_fpv_frames": video_data.fut_fpv_frames,
                            "all_fpv_frames": video_data.all_fpv_frames,
                            "all_fpv_frames_info": video_data.all_fpv_frames_info,
                            "hist_bev_frames": video_data.hist_bev_frames,
                            "fut_bev_frames": video_data.fut_bev_frames,
                            "all_bev_frames": video_data.all_bev_frames,
                            "all_bev_frames_info": video_data.all_bev_frames_info,
                        }
                    )
            except (AssertionError, FileNotFoundError) as e:
                logging.error(f"video not found error: {e}")
                continue

            if self.mode in ["vlm", "early_fusion", "llm"] and self.meta_action_loader is not None:
                meta_action_data = self.meta_action_loader.load(
                    segment_info.clip_id, segment_info.event_start_clipgt_index
                )
                segment_data.update(meta_action_data)

            # run agent
            try:
                full_messages, output_dict = self.agent.run(segment_data, save_root=self.save_root)
                success_count += 1
            except Exception:
                logging.exception("An error occurred during agent.run")
                continue

        logging.info(
            f"Processed {success_count} / {len(segment_info_list)} segments successfully "
            f"from {self.cfg.data.segment_config_path}"
        )
