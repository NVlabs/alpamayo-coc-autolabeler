"""Shared clip tables, pose geometry and timestamp-indexed camera decoding.

Dataset adapters provide ClipData; the environment mediates all subsequent reads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ...common.clipgt import obstacle_class_name, quaternion_to_matrix
from ...common.video import decode_bgr_frame


@dataclass(frozen=True)
class ClipData:
    """Raw clip data bound by the env: trajectory, labels, map, and camera media."""

    clip_id: str
    start_us: int
    end_us: int
    keyframes_us: tuple[int, ...]
    ego: pd.DataFrame | None
    objects: pd.DataFrame | None
    objects_raw: pd.DataFrame | None
    crosswalks: pd.DataFrame | None
    lanes: pd.DataFrame | None
    lane_chunks: pd.DataFrame | None
    calibration: pd.DataFrame | None
    cameras: dict[str, dict[str, str]]


def discover_camera_videos(clip_dir: Path) -> dict[str, dict[str, str]]:
    """Find ``{camera}.mp4`` files with a ``.mp4.timestamps`` sidecar under the clip dir.

    Recordings live in per-recording subdirectories (``recordings/<hash>/camera_*.mp4``), so
    the search is recursive.
    """
    cameras: dict[str, dict[str, str]] = {}
    for mp4 in sorted(clip_dir.rglob("*.mp4")):
        sidecar = Path(str(mp4) + ".timestamps")
        if sidecar.is_file():
            cameras[mp4.stem] = {"mp4": str(mp4), "timestamps": str(sidecar)}
    return cameras


def ego_kinematics_at(ego: pd.DataFrame, t_us: int) -> dict[str, Any]:
    """Raw trajectory reading at ``t_us``: rig pose plus finite-difference kinematics.

    This is the sensor stream — pose with transformation + context (speed, yaw rate over a
    +/-0.5s window). Deliberately no maneuver tag: that classification is rule parsing and
    belongs to a parse op, not the sensor source.
    """
    times = ego["t_us"].to_numpy()
    idx = int(np.argmin(np.abs(times - t_us)))
    lo = max(int(times.searchsorted(times[idx] - 500_000)), 0)
    hi = min(int(times.searchsorted(times[idx] + 500_000)), len(ego) - 1)
    yaw = _yaw_from_quat(*ego.iloc[idx][["qw", "qx", "qy", "qz"]].tolist())
    reading: dict[str, Any] = {
        "support_window_us": [int(times[lo]), int(times[hi])],
        "x": float(round(ego["x"].iloc[idx], 2)),
        "y": float(round(ego["y"].iloc[idx], 2)),
        "yaw_rad": float(round(yaw, 3)),
    }
    if hi > lo:
        dt_s = (times[hi] - times[lo]) * 1e-6
        dx = ego["x"].iloc[hi] - ego["x"].iloc[lo]
        dy = ego["y"].iloc[hi] - ego["y"].iloc[lo]
        yaw_lo = _yaw_from_quat(*ego.iloc[lo][["qw", "qx", "qy", "qz"]].tolist())
        yaw_hi = _yaw_from_quat(*ego.iloc[hi][["qw", "qx", "qy", "qz"]].tolist())
        reading["speed_mps"] = float(round(math.hypot(dx, dy) / dt_s, 2))
        reading["yaw_rate_rps"] = float(
            round(math.atan2(math.sin(yaw_hi - yaw_lo), math.cos(yaw_hi - yaw_lo)) / dt_s, 3)
        )
    return reading


def actors_at(
    objects: pd.DataFrame, ego: pd.DataFrame, t_us: int, tolerance_us: int = 100_000
) -> list[dict[str, Any]]:
    """GT obstacle labels near ``t_us``, transformed into the rig frame at that time.

    Rig-frame convention (FLU, right-handed z-up): ``forward_m`` along the ego heading,
    **positive ``lateral_m`` is left** of the ego. A crossing actor whose lateral goes
    negative to positive moves right-to-left.
    """
    mask = np.abs(objects["t_us"].to_numpy() - t_us) <= tolerance_us
    if not mask.any():
        return []
    nearby = objects[mask]
    nearest_t = int(nearby["t_us"].iloc[int(np.argmin(np.abs(nearby["t_us"].to_numpy() - t_us)))])
    snap = nearby[nearby["t_us"] == nearest_t]
    ego_idx = int(np.argmin(np.abs(ego["t_us"].to_numpy() - nearest_t)))
    ego_x, ego_y = float(ego["x"].iloc[ego_idx]), float(ego["y"].iloc[ego_idx])
    yaw = _yaw_from_quat(*ego.iloc[ego_idx][["qw", "qx", "qy", "qz"]].tolist())
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)

    actors: list[dict[str, Any]] = []
    for row in snap.itertuples(index=False):
        speed = math.hypot(float(row.vx), float(row.vy))
        dx, dy = float(row.x) - ego_x, float(row.y) - ego_y
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        actor = {
            "id": f"track_{int(row.track_id)}",
            "type": obstacle_class_name(int(row.class_id)),
            "state": "moving" if speed >= 0.5 else "stopped",
            "forward_m": float(round(forward, 1)),
            "lateral_m": float(round(lateral, 1)),
            "distance_m": float(round(math.hypot(forward, lateral), 1)),
            "speed_mps": float(round(speed, 2)),
        }
        if speed >= 0.5:
            # world-frame velocity heading: the agent's own direction of travel,
            # independent of ego (undefined when near-still)
            actor["heading_rad"] = float(round(math.atan2(float(row.vy), float(row.vx)), 3))
        actors.append(actor)
    actors.sort(key=lambda actor: actor["distance_m"])
    return actors


# dw_lane.position codes that mark a lane's relation to ego (0 = background, dropped).
_LANE_POSITION_NAMES = {
    1: "ego",
    2: "adjacent_left",
    3: "adjacent_right",
    4: "ego_fork_left",
    5: "ego_fork_right",
    6: "merge_from_left",
    7: "merge_from_right",
}


def lanes_at(
    lane_df: pd.DataFrame | None,
    chunk_df: pd.DataFrame | None,
    ego: pd.DataFrame,
    t_us: int,
) -> list[dict[str, Any]]:
    """The ego-centric drivable lanes at the nearest timestamp, in the rig frame.

    ``dw_lane`` gives identity and the ego-relative ``position``; ``lane_chunk`` gives the
    center polylines. Background lanes (position 0) are dropped; each kept lane carries its
    position name, nearest distance, lateral offset, and length.
    """
    if lane_df is None or lane_df.empty or chunk_df is None or chunk_df.empty:
        return []
    lane_ts = lane_df["key"].apply(lambda k: int(k["timestamp_micros"])).to_numpy()
    nearest_t = int(lane_ts[np.argmin(np.abs(lane_ts - t_us))])
    lanes_snap = list(lane_df[lane_ts == nearest_t]["dw_lane"])
    chunk_ts = chunk_df["key"].apply(lambda k: int(k["timestamp_micros"])).to_numpy()
    chunk_by_index: dict[int, Any] = {}
    for chunk in chunk_df[chunk_ts == nearest_t]["lane_chunk"]:
        chunk_by_index.setdefault(int(chunk["lane_chunk_index"]), chunk)
    ego_idx = int(np.argmin(np.abs(ego["t_us"].to_numpy() - nearest_t)))
    ego_loc = ego.iloc[ego_idx][["x", "y", "z"]].to_numpy(dtype=np.float64)
    rotation = quaternion_to_matrix(*ego.iloc[ego_idx][["qw", "qx", "qy", "qz"]].tolist())

    lanes: list[dict[str, Any]] = []
    for lane in lanes_snap:
        position = int(lane["position"])
        if position not in _LANE_POSITION_NAMES:
            continue
        points: list[np.ndarray] = []
        for chunk_index in lane["chunkIndices"]:
            chunk = chunk_by_index.get(int(chunk_index))
            if chunk is None:
                continue
            for point in chunk["center"]:
                world = np.array([float(point["x"]), float(point["y"]), float(point["z"])])
                points.append((world - ego_loc) @ rotation)
        if not points:
            continue
        distances = [math.hypot(point[0], point[1]) for point in points]
        nearest_point = points[int(np.argmin(distances))]
        length = sum(math.dist(points[i][:2], points[i + 1][:2]) for i in range(len(points) - 1))
        lanes.append(
            {
                "id": f"lane_{int(lane['currentId'])}",
                "type": "lane",
                "position": _LANE_POSITION_NAMES[position],
                "distance_m": float(round(min(distances), 1)),
                "lateral_m": float(round(nearest_point[1], 1)),
                "length_m": float(round(length, 1)),
            }
        )
    lanes.sort(key=lambda lane: lane["distance_m"])
    return lanes


def crosswalks_at(
    cross_df: pd.DataFrame | None, ego: pd.DataFrame, t_us: int, top_k: int = 3
) -> list[dict[str, Any]]:
    """The closest crosswalk zones at the nearest timestamp, in the rig frame."""
    if cross_df is None or cross_df.empty:
        return []
    ts = cross_df["key"].apply(lambda k: int(k["timestamp_micros"])).to_numpy()
    nearest_t = int(ts[np.argmin(np.abs(ts - t_us))])
    ego_idx = int(np.argmin(np.abs(ego["t_us"].to_numpy() - nearest_t)))
    ego_loc = np.array(
        [
            float(ego["x"].iloc[ego_idx]),
            float(ego["y"].iloc[ego_idx]),
            float(ego["z"].iloc[ego_idx]),
        ]
    )
    rotation = quaternion_to_matrix(*ego.iloc[ego_idx][["qw", "qx", "qy", "qz"]].tolist())

    scored: list[tuple[float, dict[str, Any]]] = []
    for payload in cross_df[ts == nearest_t]["cf_crosswalks"]:
        area = payload.get("crosswalk_area")
        if area is None or len(area) == 0:
            continue
        world = np.array([[float(p["x"]), float(p["y"]), float(p["z"])] for p in area])
        rig = (world - ego_loc) @ rotation
        distance = float(np.min(np.hypot(rig[:, 0], rig[:, 1])))
        element_id = payload.get("id") or payload.get("track_id") or "unknown"
        scored.append(
            (
                distance,
                {
                    "id": f"crosswalk_{element_id}",
                    "type": "crosswalk",
                    "distance_m": float(round(distance, 1)),
                },
            )
        )
    scored.sort(key=lambda item: item[0])
    return [element for _, element in scored[:top_k]]


def extract_video_segment(
    mp4: str, timestamps: str, window_us: tuple[int, int], image_max_height_px: int, out_path
) -> tuple[int, int, int]:
    """Re-encode the frames inside ``window_us`` as a standalone mp4 segment.

    Returns ``(first_ts, last_ts, frame_count)`` of the frames actually written. The segment
    preserves the source frame spacing (fps derived from the sidecar timestamps).
    """
    table = np.loadtxt(timestamps, dtype=np.int64, ndmin=2)
    frame_timestamps = table[:, 1] if table.shape[1] > 1 else table[:, 0]
    indices = np.where((frame_timestamps >= window_us[0]) & (frame_timestamps <= window_us[1]))[0]
    if len(indices) == 0:
        raise ValueError(f"no frames inside window {window_us} of {mp4}")
    selected = frame_timestamps[indices]
    duration_s = max((int(selected[-1]) - int(selected[0])) / 1e6, 1e-3)
    fps = max((len(indices) - 1) / duration_s, 1.0)
    capture = cv2.VideoCapture(mp4)
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {mp4}")
    writer = None
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))
        for _ in indices:
            ok, image = capture.read()
            if not ok:
                break
            height = image.shape[0]
            if height > image_max_height_px:
                scale = image_max_height_px / height
                image = cv2.resize(image, (int(image.shape[1] * scale), image_max_height_px))
            if writer is None:
                writer = cv2.VideoWriter(
                    str(out_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (image.shape[1], image.shape[0]),
                )
            writer.write(image)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return int(selected[0]), int(selected[-1]), len(indices)


def decode_frame(
    mp4: str, timestamps: str, t_us: int, image_max_height_px: int
) -> tuple[int, bytes]:
    """Decode the frame nearest ``t_us`` and return ``(frame_timestamp_us, jpeg_bytes)``."""
    table = np.loadtxt(timestamps, dtype=np.int64, ndmin=2)
    frame_timestamps = table[:, 1] if table.shape[1] > 1 else table[:, 0]
    index = int(np.argmin(np.abs(frame_timestamps - t_us)))
    image = decode_bgr_frame(mp4, index)
    height = image.shape[0]
    if height > image_max_height_px:
        scale = image_max_height_px / height
        image = cv2.resize(image, (int(image.shape[1] * scale), image_max_height_px))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"jpeg encode failed for frame {index} of {mp4}")
    return int(frame_timestamps[index]), encoded.tobytes()


def _yaw_from_quat(qw: float, qx: float, qy: float, qz: float) -> float:
    """Yaw (Z rotation) in radians from a ``(w, x, y, z)`` quaternion."""
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
