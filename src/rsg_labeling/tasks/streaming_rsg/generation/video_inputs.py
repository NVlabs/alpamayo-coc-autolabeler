"""Prepare immutable camera-only inputs from MP4 content and presentation timestamps."""

import hashlib
from copy import deepcopy
from pathlib import Path

from ....infra.artifacts import write_json_atomic
from ....streaming_rsg.env.video import video_timing
from ....streaming_rsg.runtime.config import RunConfig
from ..schema.contracts import ClipClock
from .frame_sampling import validate_camera_brackets


def prepare_video_inputs(videos_dir: Path, target: Path, compiler: dict, generation: dict) -> dict:
    """Bind MP4 content and clocks, then write a runnable generation configuration.

    Args:
        videos_dir: Directory searched recursively for front-wide MP4 inputs.
        target: New prepared directory; existing directories are rejected.
        compiler: Camera-only compiler configuration, with local endpoint paths resolved.
        generation: Native generation configuration, with approved operational settings.

    Returns:
        Clip count, prepared directory and zero API calls. No sidecars are read.
    """
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError(target)
    paths = sorted(Path(videos_dir).resolve().rglob("*.mp4"))
    if not paths:
        raise ValueError("no MP4 files found; extract the input archive first")
    videos, clips = {}, []
    compiler = deepcopy(compiler)
    compiler["playground"]["env"]["video_manifest"] = str(target / "videos.json")
    RunConfig.model_validate(compiler)
    env = compiler["playground"]["env"]
    for path in paths:
        with path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        cid = "video-" + checksum[:24]
        if cid in videos:
            raise ValueError("duplicate video content: " + str(path))
        timing = video_timing(path)
        if generation["policy"].get("action_camera_sampling") in {"uniform_2hz", "uniform_3hz"}:
            validate_camera_brackets(timing["timestamps_us"])
        if timing["end_us"] / 1e6 * env["max_fps"] > env["max_frames"]:
            raise ValueError("video exceeds configured frame allowance: " + str(path))
        videos[cid] = {"video": str(path), "video_sha256": checksum}
        clips.append(
            {
                "clip_id": cid,
                "clock": {"origin_us": 0, "end_us": timing["end_us"]},
                "clock_verified": False,
                "sample_anchors_us": ClipClock(0, timing["end_us"]).sample_anchors(
                    generation["policy"]["anchor_stride_us"]
                ),
            }
        )
    target.mkdir(parents=True)
    write_json_atomic(
        target / "videos.json", {"schema": "alparsg_video_only_inputs_v1", "clips": videos}
    )
    write_json_atomic(target / "inputs.json", {"clips": clips})
    write_json_atomic(target / "compiler.json", compiler)
    generation = {
        **generation,
        "input_manifest": str(target / "inputs.json"),
        "compiler_config": str(target / "compiler.json"),
    }
    write_json_atomic(target / "generation.json", generation)
    return {"clips": len(clips), "prepared": str(target), "api_calls": 0}
