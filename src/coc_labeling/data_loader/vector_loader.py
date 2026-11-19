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
from typing import Any, Dict, Optional, Tuple

import numpy as np
from omegaconf import DictConfig, open_dict

from coc_labeling.request import render_speed_text

from coc_labeling.data_loader.trajdata import convert_trajdata2data_dict
from coc_labeling.utils.data import (
    get_scene_batch_from_scene_id_ts,
    get_scene_ts_idx_map,
    get_trajdata_dataset,
)


class VectorLoader:
    """Load vector-state data and derive text features used by prompting agents."""

    def __init__(self, vector_config: Optional[DictConfig], data_config: DictConfig) -> None:
        """Initialize vector loader and trajdata dataset handles.

        Args:
            vector_config: Vector configuration; ``None`` disables vector loading.
            data_config: Data path configuration used to locate/cache trajdata.
        """
        self.data_config = data_config
        self.vector_config: Optional[DictConfig] = None
        self.dataset: Optional[Any] = None
        self.scene_ts_idx_map: Optional[Dict[str, Any]] = None
        if vector_config is None:
            logging.info("Vector loader is not activated")
            return

        expanded_config = VectorLoader.expand_config(vector_config)
        self.vector_config = expanded_config

        logging.info(f"Loading trajdata from {data_config.cache_dir}.")
        dataset_name_cfg = getattr(data_config, "dataset_name", None)
        dataset_name = "pai" if dataset_name_cfg is None else str(dataset_name_cfg)
        is_pai = dataset_name.lower() == "pai"
        self.dataset = get_trajdata_dataset(
            config=expanded_config.trajdata,
            dataset_name=dataset_name,
            data_dir=data_config.data_dir,
            cache_dir=data_config.cache_dir,
            incl_vector_map=not is_pai,
            require_map_cache=not is_pai,
        )
        self.scene_ts_idx_map = get_scene_ts_idx_map(self.dataset)
        all_scene_ids = [scene_ts.split("_")[0] for scene_ts in list(self.scene_ts_idx_map.keys())]
        # each clip may have multiple segments due to timestamp cuts
        logging.info(
            "%d segments and %d clips loaded in trajdata.",
            len(all_scene_ids),
            len(set(all_scene_ids)),
        )

    def _require_vector_config(self) -> DictConfig:
        """Return initialized vector config or raise a clear error."""
        if self.vector_config is None:
            raise ValueError("Vector loader is not activated (vector_config is None).")
        return self.vector_config

    @staticmethod
    def expand_config(vector_config: DictConfig) -> DictConfig:
        """Expand derived temporal/sampling fields in vector config.

        Args:
            vector_config: Input vector configuration to augment in-place.

        Returns:
            The same configuration object with derived fields populated.
        """
        with open_dict(vector_config):
            # Derive trajdata temporal settings directly from vector settings.
            if "trajdata" in vector_config:
                vector_config.trajdata.desired_dt = 1.0 / float(vector_config.fps)
                vector_config.trajdata.history_sec = float(vector_config.hist_length_sec)
                vector_config.trajdata.future_sec = float(vector_config.fut_length_sec)

            vector_config.hist_length_frame = int(vector_config.hist_length_sec * vector_config.fps)
            vector_config.fut_length_frame = int(vector_config.fut_length_sec * vector_config.fps)
            vector_config.freq_sample = int(1 / vector_config.time_interval)
            vector_config.hist_length_frame_sample = int(
                vector_config.hist_length_sec * vector_config.freq_sample
            )
            vector_config.fut_length_frame_sample = int(
                vector_config.fut_length_sec * vector_config.freq_sample
            )
            vector_config.sample_rate = int(vector_config.fps / vector_config.freq_sample)

        return vector_config

    def extract_segments(self, ego_traj: np.ndarray, event_start: int) -> Tuple[str, str]:
        """Convert ego trajectory slices into history/future speed text.

        Args:
            ego_traj: Ego trajectory array in source rate.
            event_start: Event start frame index in source rate.

        Returns:
            Two strings: historical speed text and future speed text.
        """
        cfg = self._require_vector_config()

        # calculate the hist and fut frame index
        hist_start_frame = event_start - cfg.hist_length_frame
        fut_end_frame = event_start + cfg.fut_length_frame

        # get vx and vy from ego traj
        ego_traj_vxvy = ego_traj[1:, 0:2] - ego_traj[:-1, 0:2]
        ego_traj_vxvy = ego_traj_vxvy * cfg.fps  # scale to speed
        ego_hist_vxvy = ego_traj_vxvy[hist_start_frame : event_start : cfg.freq_sample]
        ego_fut_vxvy = ego_traj_vxvy[event_start : fut_end_frame : cfg.freq_sample]

        # convert to text
        hist_long = ego_hist_vxvy[:, 0]
        hist_lat = ego_hist_vxvy[:, 1]
        fut_long = ego_fut_vxvy[:, 0]
        fut_lat = ego_fut_vxvy[:, 1]
        ego_hist_text = f"The longitudinal speed is: {hist_long}. The lateral speed is {hist_lat}."
        ego_fut_text = f"The longitudinal speed is: {fut_long}. The lateral speed is {fut_lat}."
        return ego_hist_text, ego_fut_text

    def convert_speed_to_text(self, data_dict: Dict[str, Any]) -> Tuple[str, str, str]:
        """Render history, future and combined ego speed text."""
        return render_speed_text(data_dict, self._require_vector_config().time_interval)

    def load(self, clip_id: str, event_start: int) -> Dict[str, Any]:
        """Load vector features for one clip segment and attach speed prompt text.

        Args:
            clip_id: Clip identifier.
            event_start: Event start frame index in clip timeline.

        Returns:
            Dictionary with converted vector/object fields and prompt helper strings.
        """
        if self.vector_config is None:
            return {}

        cfg = self._require_vector_config()
        if self.dataset is None or self.scene_ts_idx_map is None:
            raise ValueError("Vector loader dataset is not initialized.")

        # Historical indices are encoded with 2-digit zero padding to match
        # trajdata keys such as "<clip_id>_10" at dt=0.1s.
        # Keep this formatting for backward compatibility.
        clip_id_w_ts = f"{clip_id}_{event_start:02d}"
        scene_batch = get_scene_batch_from_scene_id_ts(
            clip_id_w_ts, self.dataset, self.scene_ts_idx_map
        )
        data_dict = convert_trajdata2data_dict(scene_batch, cfg)
        ego_hist_text, ego_fut_text, ego_text = self.convert_speed_to_text(data_dict)
        data_dict["ego_hist_text"] = ego_hist_text
        data_dict["ego_fut_text"] = ego_fut_text
        data_dict["ego_text"] = ego_text

        return data_dict
