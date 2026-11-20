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

import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Mapping, Optional, Sequence

import tqdm

import meta_action.utils.io as io_utils
from meta_action import post_processing
from meta_action.data_structures.ego_meta_action_wrapper import META_ACTION_MAPPING
from meta_action.data_structures.scenario import TemporalScenario
from meta_action.utils.constant import DELTA_TIMESTAMP, HISTORY_SEC
from meta_action.utils.trajdata.dataloader import (
    get_scene_batch_from_scene_id_ts,
    get_scene_ts_idx_map,
    get_trajdata_dataset,
)

logger = logging.getLogger(__name__)


def save_result_clip(save_file: str, results: Sequence[Any]) -> None:
    """Save per-clip meta-action results to a json file.

    Args:
        save_file: Output json path for one clip.
        results: Iterable of meta-action objects (must support `str()`).

    Returns:
        None.
    """
    if len(results) == 0:
        return

    res_strs = [str(meta_action_res) for meta_action_res in results]
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(res_strs, f)


def process_clip(
    clip_id: str,
    save_root: str,
    dataset: Any,
    scene_ts_idx_map: Mapping[str, int],
    meta_action_classes: Sequence[Any],
    use_lane: bool,
) -> str:
    """Process one clip and write its meta-action labels.

    Args:
        clip_id: Clip identifier.
        save_root: Directory where per-clip result files are stored.
        dataset: Trajdata dataset object.
        scene_ts_idx_map: Mapping from `scene_id_ts` string to dataset index.
        meta_action_classes: Selected meta-action class objects.
        use_lane: Whether lane graph dependent labeling is enabled.

    Returns:
        Processed clip id.
    """
    save_file = os.path.join(save_root, f"{clip_id}.json")

    # Retrieve the scene batch from trajdata.
    if clip_id + "_0" in scene_ts_idx_map:
        scene_batch = get_scene_batch_from_scene_id_ts(clip_id + "_0", dataset, scene_ts_idx_map)
    else:
        # We loop the current ts from the start of history timestamps,
        # e.g., 10 denotes current timestamp ts=1.0s with dt=0.1.
        # A minimum of 1s history is required to compute meta actions.
        start_frames = int(HISTORY_SEC / DELTA_TIMESTAMP)
        scene_batch = get_scene_batch_from_scene_id_ts(
            clip_id + f"_{start_frames}", dataset, scene_ts_idx_map
        )

    scenario = TemporalScenario(clip_id=clip_id, scene_batch_data=scene_batch, use_lane=use_lane)

    results: List[Any] = []
    for meta_action in meta_action_classes:
        output = scenario.get_tag_motions(meta_action)
        results += output

    save_result_clip(save_file, results)
    return clip_id


def create_argparser() -> argparse.ArgumentParser:
    """Create and return the CLI parser."""
    argparser = argparse.ArgumentParser(
        description="Run batch meta-action labeling over clips from trajdata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argparser.add_argument(
        "--dataset_name",
        type=str,
        default="pai",
        help="Dataset key used by trajdata (for example: pai).",
    )
    argparser.add_argument(
        "--meta_action_names",
        type=str,
        nargs="+",
        default=["go_straight"],
        help=(
            "Meta-action names to compute. Use 'all_ego' to expand to the default ego-action set."
        ),
    )
    argparser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory of the trajdata parquet clips for the selected dataset.",
    )
    argparser.add_argument(
        "--cache_dir",
        type=str,
        required=True,
        help="Root directory of trajdata cache data.",
    )
    argparser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Output directory for raw and post-processed meta-action results.",
    )
    argparser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of worker threads for data loading and clip processing.",
    )
    argparser.add_argument(
        "--use_lane",
        action="store_true",
        help="Enable lane-graph-dependent meta-action generation.",
    )
    argparser.add_argument(
        "--scene_list",
        type=str,
        default=None,
        help=(
            "Optional path to a text file with clip IDs to process (one per line). "
            "When provided, this clip list determines the final set of clips."
        ),
    )
    argparser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose exception reporting.",
    )
    return argparser


def get_all_ego_meta_action_names(use_lane: bool) -> List[str]:
    """Return default `all_ego` meta-action names.

    Args:
        use_lane: Whether to include lane-level meta-action names.

    Returns:
        Ordered list of meta-action names.
    """
    meta_action_names: List[str] = [
        # Longitudinal
        "gentle_acceleration",
        "strong_acceleration",
        "gentle_deceleration",
        "strong_deceleration",
        "maintain_speed",
        "stop",
        "reverse",
        # Lateral
        "reverse_right",
        "reverse_left",
        "steer_right",
        "steer_left",
        "sharp_steer_right",
        "sharp_steer_left",
        "go_straight",
    ]
    if use_lane:
        # Lane-level meta actions.
        meta_action_names.extend(
            [
                "keep_lane",
                "left_lane_change",
                "right_lane_change",
                "slightly_shift_left",
                "slightly_shift_right",
                "turn_left",
                "turn_right",
                # "follow_curve_left",
                # "follow_curve_right",
            ]
        )
    return meta_action_names


def resolve_clip_ids(
    scene_ts_idx_map: Mapping[str, int], scene_list_path: Optional[str]
) -> List[str]:
    """Build unique clip-id list from scene timestamp mapping.

    Args:
        scene_ts_idx_map: Mapping with keys formatted as `<clip_id>_<ts_idx>`.
        scene_list_path: Optional file path with one clip id per line.

    Returns:
        Shuffled clip-id list, or file-provided order if `scene_list_path` is set.
    """
    # Retrieve the unique number of clips/segments.
    # Each clip may have multiple segments due to timestamp cuts.
    all_clip_ids = [scene_ts.rsplit("_", 1)[0] for scene_ts in list(scene_ts_idx_map.keys())]

    logger.info("total number of segments is %d", len(all_clip_ids))
    all_clip_ids = list(set(all_clip_ids))
    logger.info("total number of clip ids is %d", len(all_clip_ids))
    random.shuffle(all_clip_ids)

    # Only process selected clips.
    if scene_list_path is not None:
        with open(scene_list_path, encoding="utf-8") as f:
            all_clip_ids = f.read().splitlines()

    return all_clip_ids


def main() -> None:
    """CLI entrypoint for batch meta-action labeling."""
    args = create_argparser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
    )

    # Load trajdata dataset.
    dataset = get_trajdata_dataset(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        num_workers=args.num_workers,
        incl_vector_map=bool(args.use_lane),
    )
    scene_ts_idx_map = get_scene_ts_idx_map(dataset)
    all_clip_ids = resolve_clip_ids(scene_ts_idx_map, args.scene_list)

    # Select meta actions to label.
    if args.meta_action_names[0] == "all_ego":
        meta_action_names = get_all_ego_meta_action_names(args.use_lane)
    else:
        meta_action_names = args.meta_action_names

    meta_action_classes = [
        META_ACTION_MAPPING[meta_action_tmp] for meta_action_tmp in meta_action_names
    ]
    logger.info("meta_action_names: %s", meta_action_names)

    # Path to save.
    if args.meta_action_names[0] == "all_ego":
        save_root = os.path.join(args.save_dir, "tmp", "raw_results")
        post_save_root = os.path.join(args.save_dir, "final_outputs")
    else:
        save_names = "-".join(meta_action_names)
        save_root = os.path.join(args.save_dir, save_names)
        post_save_root = save_root + "_post"
    io_utils.mkdir_if_missing(save_root)

    # Parallel processing.
    cnt = 0
    num_workers = min(args.num_workers, len(all_clip_ids))
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_clip,
                clip_id,
                save_root,
                dataset,
                scene_ts_idx_map,
                meta_action_classes,
                args.use_lane,
            )
            for clip_id in all_clip_ids
        ]
        for future in tqdm.tqdm(as_completed(futures), total=len(all_clip_ids)):
            try:
                future.result()
            except Exception as e:
                logger.exception("[data_labeling] Worker raised an exception: %r", e)
                continue

            cnt += 1

    # Post-processing the meta actions:
    # smoothing, resolving conflict, flickering, filling, pruning redundant ones, etc.
    post_processing.process_batch(
        meta_action_dir=save_root,
        save_dir=post_save_root,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()
