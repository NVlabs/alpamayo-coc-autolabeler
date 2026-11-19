"""Front-camera video timing without native archives or external sensor timestamps."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np

from .env import AccessPolicy, Anchor, RunBudget, SamplingBounds, build_env
from .loaders import ClipData


def presentation_timing(probe: dict) -> dict:
    """Return presentation-ordered relative microseconds and the actual final display end."""
    if len(probe["streams"]) != 1:
        raise ValueError("exactly one selected video stream is required")
    stream = probe["streams"][0]
    time_base = Fraction(stream["time_base"])
    if time_base <= 0:
        raise ValueError("video time base must be positive")
    packets = probe["packets"]
    if len(packets) < 2 or any(type(row.get("pts")) is not int for row in packets):
        raise ValueError(
            "at least two video packets with exact presentation timestamps are required"
        )
    packets = sorted(packets, key=lambda row: row["pts"])
    if len({row["pts"] for row in packets}) != len(packets):
        raise ValueError("duplicate presentation timestamps")
    last_duration = packets[-1].get("duration")
    if type(last_duration) is not int or last_duration <= 0:
        raise ValueError("the final frame needs a declared positive display duration")
    origin = packets[0]["pts"]
    timestamps = [round((row["pts"] - origin) * time_base * 1_000_000) for row in packets]
    if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("presentation timestamps are ambiguous at microsecond precision")
    intervals = np.diff(timestamps)
    tolerance_us = max(1, float(time_base * 1_000_000))
    if np.any(np.abs(intervals - np.median(intervals)) > tolerance_us):
        raise ValueError("video requires constant frame spacing for indexed OpenCV decoding")
    declared_count = stream.get("nb_frames")
    if declared_count not in (None, "N/A") and int(declared_count) != len(packets):
        raise ValueError("video frame count differs from packet timing; frame-level audit required")
    end = round((packets[-1]["pts"] + last_duration - origin) * time_base * 1_000_000)
    return {
        "timestamps_us": timestamps,
        "end_us": end,
        "source_origin_pts": origin,
        "time_base": stream["time_base"],
        "frames": len(packets),
    }


def video_timing(path: Path) -> dict:
    """Read one MP4's packet timing without consulting any sidecar or native metadata."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=time_base,nb_frames:packet=pts,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return presentation_timing(json.loads(result.stdout))


def build_env_video(path: Path, cache_dir: Path, config, clip_id=None, expected_sha256=None):
    """Bind a bare front-wide video; unavailable sensors stay absent, never zero-valued."""
    camera = "camera_front_wide_120fov"
    if (
        config.sources
        or config.cameras != [camera]
        or config.anchor_mode != "discovery"
        or config.navigation_manifest is not None
        or config.meta_action_manifest is not None
    ):
        raise ValueError("video-only environment requires camera-only discovery configuration")
    with path.open("rb") as handle:
        content_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    if expected_sha256 is not None and content_hash != expected_sha256:
        raise ValueError("registered input video content changed")
    timing = video_timing(path)
    identity = clip_id or "video-" + content_hash[:24]
    cache_dir.mkdir(parents=True, exist_ok=True)
    timestamps = cache_dir / "video_pts.timestamps"
    np.savetxt(
        timestamps,
        np.column_stack((np.arange(timing["frames"]), timing["timestamps_us"])),
        fmt="%d",
    )
    end = timing["end_us"]
    data = ClipData(
        identity,
        0,
        end,
        (),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        {camera: {"mp4": str(path), "timestamps": str(timestamps)}},
    )
    binding = {
        "video_sha256": content_hash,
        "timing_source": "video_presentation_timestamps",
        "clock": {"origin_us": 0, "end_us": end},
        "frames": timing["frames"],
        "time_base": timing["time_base"],
        "externally_verified_sensor_time": False,
    }
    return build_env(
        identity,
        (Anchor(end, end, 0, "discovery"),),
        SamplingBounds(config.max_fps, config.max_frames, (camera,), config.image_max_height_px),
        AccessPolicy(agent_gt_labels=False),
        RunBudget(config.budgets.max_ops, config.budgets.wall_clock_s),
        (),
        data,
        include_external_labels=False,
        reference=config.reference,
        reference_concepts=None
        if config.reference_concepts is None
        else tuple(config.reference_concepts),
        video_binding=binding,
    )


def build_env_video_manifest(manifest_path, clip_id, cache_dir, config):
    """Resolve only registered media; the generation manifest contains no evaluation clock."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != "alparsg_video_only_inputs_v1":
        raise ValueError("unknown video-only input manifest")
    entry = manifest["clips"][clip_id]
    if set(entry) != {"video", "video_sha256"}:
        raise ValueError("video input manifest contains undeclared metadata")
    return build_env_video(Path(entry["video"]), cache_dir, config, clip_id, entry["video_sha256"])
