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

"""Prepare streaming inputs from the existing PAI cache and meta-actions."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from coc_labeling.data_loader.meta_action_loader import (
    meta_action_lane,
    meta_action_lateral,
    meta_action_longitu,
    normalize_meta_action,
    parse_line,
)


def _sha256(path: Path) -> str:
    """Hash file content without retaining media in memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _action_entry(path: Path, origin_us: int, clock: dict, source: dict) -> dict:
    """Bind existing half-open 10 Hz action spans to their verified physical origin."""
    from ..common.meta_actions import coverage_diagnostics, validate_meta_actions

    groups = {
        action: group
        for group, actions in (
            ("longitudinal", meta_action_longitu),
            ("lateral", meta_action_lateral),
            ("lane", meta_action_lane),
        )
        for action in actions
    }
    actions = []
    for line in path.read_text().splitlines():
        if not line.strip() or "Agent:<ego>" not in line:
            continue
        parsed = parse_line(line)
        if parsed is None:
            raise ValueError("malformed ego action: " + str(path))
        name, start_tick, end_tick = parsed
        action = normalize_meta_action(name)
        if start_tick < 0 or start_tick >= end_tick or action not in groups:
            raise ValueError("invalid ego action span: " + str(path))
        start = max(clock["origin_us"], origin_us + start_tick * 100_000)
        end = min(clock["end_us"], origin_us + end_tick * 100_000)
        if start < end:
            actions.append(
                {"action": action, "group": groups[action], "start_us": start, "end_us": end}
            )
    actions.sort(key=lambda item: (item["group"], item["start_us"], item["end_us"], item["action"]))
    entry = {
        "available": bool(actions),
        "reason": "decoded_existing_pai_meta_actions" if actions else "no_decodable_ego_actions",
        "clock": clock,
        "actions": actions,
        "source": {**source, "kind": "pai_rule_generated_meta_actions", "sha256": _sha256(path)},
    }
    validate_meta_actions(entry)
    entry.update(coverage_diagnostics(entry))
    return entry


def prepare_pai_inputs(
    data_dir: Path,
    cache_dir: Path,
    meta_action_dir: Path,
    video_dir: Path,
    clip_ids: list[str],
    target: Path,
    native_generation: dict,
    video_generation: dict,
    native_compiler: dict,
    video_compiler: dict,
) -> dict:
    """Reuse the standalone cache/actions to write verified inputs for both streaming recipes.

    Dataset caching, meta-action generation and video extraction remain the standalone
    workflow's responsibilities. This preparation performs local I/O and zero model calls.
    The cache must be trusted, as TrajData scene metadata uses Python serialization.
    """
    from coc_labeling.utils.data import load_cached_pai_ego
    from trajdata.dataset_specific.pai.pai_dataset import PAIDataset

    from ..common.meta_actions import SCHEMA
    from ..infra.artifacts import write_json_atomic
    from ..streaming_rsg.env.video import video_timing
    from ..streaming_rsg.runtime.config import RunConfig
    from ..tasks.streaming_rsg.generation.frame_sampling import validate_camera_brackets
    from ..tasks.streaming_rsg.schema.contracts import ClipClock

    target = target.resolve()
    if target.exists():
        raise FileExistsError(target)
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("PAI clip list must be nonempty and unique")
    dataset = PAIDataset("pai", str(data_dir.resolve()), parallelizable=False, has_maps=False)
    target.mkdir(parents=True)
    sensors, videos, actions, sensor_items, video_items = {}, {}, {}, [], []
    for clip_id in clip_ids:
        ego, source = load_cached_pai_ego(cache_dir.resolve(), clip_id)
        raw = (
            dataset._load_ego_df(dataset.clip_dir[clip_id])
            .sort_values("timestamp")
            .drop_duplicates("timestamp")
        )
        origin_us = int(raw["timestamp"].iloc[0])
        ticks = ego.index.to_numpy(dtype=np.int64)
        times = origin_us + ticks * 100_000
        for coordinate in ("x", "y", "z"):
            expected = np.interp(times, raw["timestamp"].to_numpy(), raw[coordinate].to_numpy())
            if not np.allclose(ego[coordinate], expected, rtol=0, atol=1e-4):
                raise ValueError("cached PAI motion does not match the raw clock: " + clip_id)
        source = {
            **source,
            "dataset": "pai",
            "motion": "trajdata_cache",
            "cache_origin_us": origin_us,
            "orientation": "cached_heading_yaw_only",
            "origin_verified_against_raw_positions": True,
        }
        video = (video_dir / "camera" / f"{clip_id}.camera_front_wide_120fov.mp4").resolve()
        sidecar = video.with_suffix(".timestamps.parquet")
        timestamps = pd.read_parquet(sidecar)["timestamp"].to_numpy()
        timing = video_timing(video)
        if (
            not pd.api.types.is_integer_dtype(timestamps.dtype)
            or len(timestamps) != timing["frames"]
        ):
            raise ValueError("PAI camera sidecar differs from video frames: " + clip_id)
        validate_camera_brackets(timestamps)
        period = max(1, int(round(float(np.median(np.diff(timestamps))))))
        clock = {"origin_us": int(timestamps[0]), "end_us": int(timestamps[-1]) + period}
        media_hash = _sha256(video)
        clip_dir = target / "sensors" / clip_id
        clip_dir.mkdir(parents=True)
        capture_path = clip_dir / "camera.timestamps"
        np.savetxt(
            capture_path, np.column_stack((np.arange(len(timestamps)), timestamps)), fmt="%d"
        )
        pose_path = clip_dir / "ego.parquet"
        heading = ego["heading"].to_numpy()
        pd.DataFrame(
            {
                "t_us": times,
                "x": ego["x"].to_numpy(),
                "y": ego["y"].to_numpy(),
                "z": ego["z"].to_numpy(),
                "qw": np.cos(heading / 2),
                "qx": 0.0,
                "qy": 0.0,
                "qz": np.sin(heading / 2),
            }
        ).to_parquet(pose_path, index=False)
        sensors[clip_id] = {
            "clock": clock,
            "ego": {"path": str(pose_path), "sha256": _sha256(pose_path)},
            "camera": {
                "video": str(video),
                "video_sha256": media_hash,
                "timestamps": str(capture_path),
                "timestamps_sha256": _sha256(capture_path),
            },
            "source": source,
        }
        actions[clip_id] = _action_entry(
            meta_action_dir / f"{clip_id}.txt", origin_us, clock, source
        )
        videos[clip_id] = {"video": str(video), "video_sha256": media_hash}
        for cfg, items, extent, verified in (
            (native_generation, sensor_items, clock, True),
            (video_generation, video_items, {"origin_us": 0, "end_us": timing["end_us"]}, False),
        ):
            items.append(
                {
                    "clip_id": clip_id,
                    "clock": extent,
                    "clock_verified": verified,
                    "sample_anchors_us": ClipClock(**extent).sample_anchors(
                        cfg["policy"]["anchor_stride_us"]
                    ),
                }
            )
    write_json_atomic(
        target / "sensors.json", {"schema": "alparsg_sensor_inputs_v1", "clips": sensors}
    )
    write_json_atomic(target / "actions.json", {"schema": SCHEMA, "clips": actions})
    write_json_atomic(
        target / "videos.json", {"schema": "alparsg_video_only_inputs_v1", "clips": videos}
    )
    for name, generation, compiler, items in (
        ("sensor", native_generation, native_compiler, sensor_items),
        ("video", video_generation, video_compiler, video_items),
    ):
        directory = target / name
        directory.mkdir(exist_ok=True)
        compiler, generation = deepcopy(compiler), deepcopy(generation)
        compiler["infra"].update(
            metadb=None, data_dir=None, endpoint_registry=generation["endpoint_registry"]
        )
        env = compiler["playground"]["env"]
        env.update(navigation_manifest=None, meta_action_manifest=None, video_manifest=None)
        if name == "sensor":
            env.update(
                sensor_manifest=str(target / "sensors.json"),
                meta_action_manifest=str(target / "actions.json"),
                sources=["ego_pose", "production_meta_actions"],
            )
        else:
            env.update(video_manifest=str(target / "videos.json"), sources=[])
        RunConfig.model_validate(compiler)
        write_json_atomic(directory / "inputs.json", {"clips": items})
        write_json_atomic(directory / "compiler.json", compiler)
        generation.update(
            input_manifest=str(directory / "inputs.json"),
            compiler_config=str(directory / "compiler.json"),
        )
        write_json_atomic(directory / "generation.json", generation)
    return {"clips": len(clip_ids), "prepared": str(target), "api_calls": 0}


def main() -> None:
    """Prepare PAI streaming inputs after the documented standalone cache/extraction steps."""
    recipes = Path(__file__).resolve().parents[1] / "tasks/streaming_rsg/configs"
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-dir",
        "cache-dir",
        "meta-action-dir",
        "video-dir",
        "clip-list",
        "endpoint-registry",
        "out",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--native-config", type=Path, default=recipes / "wl_readable_context.yaml")
    parser.add_argument(
        "--video-config", type=Path, default=recipes / "video_wl_readable.generation.yaml"
    )
    args = parser.parse_args()
    native = yaml.safe_load(args.native_config.read_text())
    video = yaml.safe_load(args.video_config.read_text())
    for config in (native, video):
        config["endpoint_registry"] = str(args.endpoint_registry.resolve())
    native_compiler = yaml.safe_load((recipes / "wl_readable_context.compiler.yaml").read_text())
    video_compiler = yaml.safe_load((recipes / "video_wl_readable.compiler.yaml").read_text())
    clip_ids = [line.strip() for line in args.clip_list.read_text().splitlines() if line.strip()]
    print(
        json.dumps(
            prepare_pai_inputs(
                args.data_dir,
                args.cache_dir,
                args.meta_action_dir,
                args.video_dir,
                clip_ids,
                args.out,
                native,
                video,
                native_compiler,
                video_compiler,
            )
        )
    )


if __name__ == "__main__":
    main()
