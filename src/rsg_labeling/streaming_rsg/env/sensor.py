"""Bind prepared camera captures and measured ego poses without dataset-specific I/O."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .loaders import ClipData


def load_sensor_data(manifest_path: str, clip_id: str) -> tuple[ClipData, dict]:
    """Verify a prepared clip's content and physical clock before exposing its sources.

    Args:
        manifest_path: JSON sensor manifest with content hashes for every bound file.
        clip_id: Registered clip identifier.

    Returns:
        Camera/ego data and a content-bound provenance record. Missing sensor coverage
        remains absent; no poses, labels, maps or navigation are synthesized.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest["schema"] != "alparsg_sensor_inputs_v1":
        raise ValueError("unknown sensor input manifest")
    entry = manifest["clips"][clip_id]
    if set(entry) != {"clock", "ego", "camera", "source"}:
        raise ValueError("sensor input manifest contains undeclared sources")
    clock, ego_entry, camera = entry["clock"], entry["ego"], entry["camera"]
    if set(clock) != {"origin_us", "end_us"} or any(type(v) is not int for v in clock.values()):
        raise ValueError("sensor clock requires integer microseconds")
    lo, hi = clock["origin_us"], clock["end_us"]
    if lo >= hi or not isinstance(entry["source"], dict):
        raise ValueError("invalid sensor clock or provenance")
    if set(ego_entry) != {"path", "sha256"} or set(camera) != {
        "video",
        "video_sha256",
        "timestamps",
        "timestamps_sha256",
    }:
        raise ValueError("sensor manifest requires explicit content hashes")
    for path, expected in (
        (ego_entry["path"], ego_entry["sha256"]),
        (camera["video"], camera["video_sha256"]),
        (camera["timestamps"], camera["timestamps_sha256"]),
    ):
        with Path(path).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("registered sensor input content changed: " + str(path))
    captures = np.asarray(np.loadtxt(camera["timestamps"], dtype=str, ndmin=2), dtype=np.int64)
    if (
        captures.shape[1] != 2
        or len(captures) < 2
        or not np.array_equal(captures[:, 0], np.arange(len(captures)))
        or np.any(np.diff(captures[:, 1]) <= 0)
    ):
        raise ValueError("sensor captures require ordered frame indices and timestamps")
    period = max(1, int(round(float(np.median(np.diff(captures[:, 1]))))))
    if int(captures[0, 1]) != lo or int(captures[-1, 1]) + period != hi:
        raise ValueError("sensor clock differs from the physical camera extent")
    capture = cv2.VideoCapture(camera["video"])
    try:
        if not capture.isOpened() or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != len(captures):
            raise ValueError("sensor capture count differs from the video")
    finally:
        capture.release()
    ego = pd.read_parquet(ego_entry["path"])
    columns = {"t_us", "x", "y", "z", "qw", "qx", "qy", "qz"}
    if set(ego.columns) != columns or not pd.api.types.is_integer_dtype(ego["t_us"]):
        raise ValueError("sensor ego table requires integer times, positions and quaternions")
    if not np.isfinite(ego.to_numpy(dtype=float)).all() or np.any(np.diff(ego["t_us"]) <= 0):
        raise ValueError("sensor ego poses must be finite and strictly ordered")
    if not np.allclose(np.linalg.norm(ego[["qw", "qx", "qy", "qz"]], axis=1), 1, atol=1e-5):
        raise ValueError("sensor ego quaternions must have unit norm")
    ego = ego[(ego["t_us"] >= lo) & (ego["t_us"] <= hi)].reset_index(drop=True)
    if len(ego) < 2:
        raise ValueError("sensor clip requires at least two measured ego poses")
    binding = {
        "clock": clock,
        "source": entry["source"],
        "ego_sha256": ego_entry["sha256"],
        "video_sha256": camera["video_sha256"],
        "timestamps_sha256": camera["timestamps_sha256"],
        "ego_coverage_us": [int(ego["t_us"].iloc[0]), int(ego["t_us"].iloc[-1])],
        "frames": len(captures),
    }
    data = ClipData(
        clip_id,
        lo,
        hi,
        (),
        ego,
        None,
        None,
        None,
        None,
        None,
        None,
        {"camera_front_wide_120fov": {"mp4": camera["video"], "timestamps": camera["timestamps"]}},
    )
    return data, binding
