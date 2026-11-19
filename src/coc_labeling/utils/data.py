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

"""Trajdata dataset helpers used by coc_labeling data loaders."""

import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd

# Compatibility for NumPy 2.x where np.string_ was removed (used by some trajdata releases).
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

from trajdata import UnifiedDataset

DataIndexEntry = Tuple[str, int]


def _extract_scene_id(scene_path: str) -> str:
    """Extract clip/scene id from a trajdata scene path.

    Args:
        scene_path: Scene path string from trajdata data index.

    Returns:
        Scene id extracted from path (second-to-last component).
    """
    return scene_path.split("/")[-2]


def _get_data_index(dataset: UnifiedDataset) -> Iterable[DataIndexEntry]:
    """Return dataset data index as a typed iterable.

    Notes:
        `UnifiedDataset` stores this as a private member (`_data_index`). We centralize
        access here to keep private-API usage localized and easier to maintain.

    Args:
        dataset: Trajdata unified dataset instance.

    Returns:
        Iterable of `(scene_path, ts)` entries.

    Raises:
        AttributeError: If `_data_index` is missing.
        TypeError: If `_data_index` has unexpected type.
    """
    raw_index = getattr(dataset, "_data_index", None)
    if raw_index is None:
        raise AttributeError("UnifiedDataset does not expose `_data_index`.")
    try:
        iter(raw_index)
    except TypeError as exc:
        raise TypeError("UnifiedDataset `_data_index` must be iterable.") from exc
    return cast(Iterable[DataIndexEntry], raw_index)


def get_trajdata_dataset(
    config: Union[SimpleNamespace, Any],
    dataset_name: str = "pai",
    data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_agent_num: int = 50,
    incl_raster_map: bool = False,
    incl_vector_map: bool = True,
    require_map_cache: bool = True,
) -> UnifiedDataset:
    """Create instance for `UnifiedDataset`.

    Args:
        dataset_name: Dataset prefix name used by trajdata.
        data_dir: Root folder containing dataset clips.
        cache_dir: Trajdata cache directory.
        max_agent_num: Max agent count per scene sample.
        num_workers: Number of worker processes for data/map loading.
        incl_raster_map: Whether to include raster maps.
        incl_vector_map: Whether to include vector maps.

    Returns:
        Configured `UnifiedDataset`.
    """
    if data_dir is None:
        raise ValueError("`data_dir` must be provided to build the dataset.")

    data_dirs = {dataset_name: data_dir}
    # Keep acceleration terms to support downstream calls to as_format("xdd,ydd").
    state_format = "x,y,z,xd,yd,xdd,ydd,h"
    dataset_kwargs: Dict[str, Any] = {
        "desired_data": [f"{dataset_name}-all"],
        "centric": "scene",
        "max_agent_num": max_agent_num,
        "desired_dt": config.desired_dt,
        "history_sec": (config.history_sec, config.history_sec),
        "future_sec": (config.future_sec, config.future_sec),
        "state_format": state_format,
        "obs_format": state_format,
        "agent_interaction_distances": defaultdict(lambda: 50.0),
        "incl_robot_future": False,
        "incl_vector_map": incl_vector_map,
        "incl_raster_map": incl_raster_map,
        "raster_map_params": {
            "px_per_m": 4,
            "map_size_px": 448,
            "offset_frac_xy": (0.0, 0.0),
        },
        "vector_map_params": {
            "incl_road_lanes": True,
            "incl_road_areas": True,
            "incl_road_edges": True,
            "incl_ped_crosswalks": False,
            "incl_ped_walkways": False,
            "collate": True,
            "keep_in_memory": True,
            "num_workers": config.num_workers,
        },
        "verbose": True,
        "num_workers": config.num_workers,
        "cache_location": cache_dir,
        "rebuild_maps": False,
        "rebuild_cache": False,
        "require_map_cache": require_map_cache,
        "data_dirs": data_dirs,
    }
    dataset = UnifiedDataset(**dataset_kwargs)

    logging.info("# Data Samples: %d", len(dataset))
    return dataset


def get_scene_ts_idx_map(dataset: UnifiedDataset) -> Dict[str, int]:
    """Build a map from `<scene_id>_<ts>` to dataset index.

    Args:
        dataset: Trajdata unified dataset.

    Returns:
        Dict mapping scene-timestamp key to sample index in dataset.
    """
    scene_ts_idx: Dict[str, int] = {}

    for idx, (scene_path, ts) in enumerate(_get_data_index(dataset)):
        scene_id = _extract_scene_id(scene_path)
        scene_ts_idx[f"{scene_id}_{ts}"] = idx

    return scene_ts_idx


def get_scene_batch_from_scene_id_ts(
    scene_id_ts: str,
    dataset: UnifiedDataset,
    scene_ts_idx: Mapping[str, int],
) -> Any:
    """Fetch and collate a single scene sample by `scene_id_ts` key.

    Args:
        scene_id_ts: Composite key in form `<scene_id>_<ts>`.
        dataset: Trajdata unified dataset.
        scene_ts_idx: Mapping from `scene_id_ts` to dataset index.

    Returns:
        Scene batch object returned by `dataset.get_collate_fn()`.
    """
    idx = scene_ts_idx[scene_id_ts]
    scene_item = dataset[idx]
    scene_batch = dataset.get_collate_fn()([scene_item])
    return scene_batch


def get_clip_timing_info(dataset: Any, clip_id: str) -> Tuple[Optional[int], Optional[float]]:
    """Get clip start timestamp and dt (microseconds) for a clip id."""
    for env in dataset.envs:
        if hasattr(env, "clip_start_micros") and clip_id in env.clip_start_micros:
            return int(env.clip_start_micros[clip_id]), float(env.metadata.dt * 1e6)
    return None, None


def load_cached_pai_ego(cache_dir: Path, clip_id: str) -> tuple[pd.DataFrame, dict]:
    """Read complete 10 Hz ego poses through TrajData's existing cache reader.

    Args:
        cache_dir: Existing trusted TrajData cache root containing the ``pai`` environment.
        clip_id: PAI clip UUID whose scene metadata and agent table are already cached.

    Returns:
        Ego rows limited to the declared scene ticks, with their source checksum and tick rate.

    Raises:
        ValueError: Scene identity, sampling interval or measured pose coverage is inconsistent.
        FileNotFoundError: The existing scene cache is incomplete; no recaching is attempted.
    """
    from trajdata.caching import EnvCache
    from trajdata.caching.df_cache import DataFrameCache

    scene = EnvCache(cache_dir).load_scene("pai", clip_id, 0.1)
    if scene.name != clip_id or scene.env_name != "pai" or scene.dt != 0.1:
        raise ValueError("PAI cache identity or sampling interval differs")
    cache_path = DataFrameCache.scene_cache_dir(
        cache_dir, "pai", clip_id
    ) / DataFrameCache._agent_data_file(0.1)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    ego = DataFrameCache(cache_dir, scene).scene_data_df.xs("ego", level="agent_id").sort_index()
    ego = ego.loc[0 : scene.length_timesteps - 1].copy()
    if not np.array_equal(ego.index.to_numpy(), np.arange(scene.length_timesteps)):
        raise ValueError("PAI cache does not cover its declared scene ticks")
    if not np.isfinite(ego[["x", "y", "z", "heading"]].to_numpy()).all():
        raise ValueError("nonfinite cached PAI ego motion")
    with cache_path.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    return ego, {"cache_sha256": checksum, "num_ticks": len(ego), "tick_us": 100_000}
