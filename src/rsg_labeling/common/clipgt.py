"""Clipgt table access, obstacle names and calibrated geometry."""

import math
from typing import Any

import numpy as np
import pandas as pd


def obstacle_class_name(class_id: int) -> str:
    """Return the friendly name for an obstacle class id.

    Only an id outside the enum falls back to the numeric class. An import failure is
    NOT caught: it would silently rename every actor (``class_0`` vs ``Invalid``) and
    make golden-based tests depend on which environment froze them.
    """
    from ..infra.obstacles import NATURAL_LANGUAGE_NAME_MAP, ObstacleClass

    try:
        enum_name = ObstacleClass(class_id).name
    except ValueError:
        return f"class_{class_id}"
    return NATURAL_LANGUAGE_NAME_MAP.get(enum_name, enum_name.lower())


def quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """3x3 rig->world rotation; a world point maps to rig via ``(p - ego_loc) @ matrix``."""
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def project_obstacle_box(
    obstacle: dict,
    camera: Any,
    camera_to_rig: np.ndarray,
    ego_location: np.ndarray,
    ego_rotation: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[int, int, int, int, float] | None:
    """Project a world-frame cuboid into clipped pixel bounds and raw rig-frame distance.

    Args:
        obstacle: Clipgt cuboid center, half axes and heading in the world frame.
        camera: Calibrated camera model with width, height and ray projection.
        camera_to_rig: (4, 4) camera-to-rig transform.
        ego_location: (3,) rig origin in world coordinates at the frame timestamp.
        ego_rotation: (3, 3) rig-to-world rotation at the frame timestamp.
        target_w: Target image width in pixels.
        target_h: Target image height in pixels.

    Returns:
        Clipped x0, y0, x1, y1 and unrounded distance in meters, or None when the cuboid
        crosses the camera plane, lies outside the image or has empty pixel bounds.
    """
    import torch

    from ..infra.transformation import box2camera, xyzrot_to_corners

    center = obstacle["cuboid_3D_center"]
    half = obstacle["cuboid_3D_halfAxisXYZ"]
    direction = obstacle["obstacle_direction"]
    yaw = math.atan2(float(direction["y"]), float(direction["x"]))
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    rotation = torch.tensor(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    xyz = torch.tensor(
        [float(center["x"]), float(center["y"]), float(center["z"])], dtype=torch.float64
    )
    dims = torch.tensor(
        [2 * float(half["x"]), 2 * float(half["y"]), 2 * float(half["z"])],
        dtype=torch.float64,
    )
    corners = xyzrot_to_corners(xyz, rotation, dims).numpy().astype(np.float64)
    corners = (corners - ego_location) @ ego_rotation
    distance_m = float(np.linalg.norm(corners.mean(axis=0)))
    uv, depth = box2camera(corners, camera_to_rig, camera)
    if not np.all(depth > 0):
        return None
    xs = uv[:, 0] * (target_w / camera.width)
    ys = uv[:, 1] * (target_h / camera.height)
    if xs.max() < 0 or xs.min() > target_w or ys.max() < 0 or ys.min() > target_h:
        return None
    x0 = int(round(max(0.0, xs.min())))
    y0 = int(round(max(0.0, ys.min())))
    x1 = int(round(min(target_w - 1.0, xs.max())))
    y1 = int(round(min(target_h - 1.0, ys.max())))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1, distance_m


def ego_table(ego_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten egomotion_estimate.parquet into a sorted pose table."""
    location = ego_df["egomotion_estimate"].apply(lambda e: e["location"])
    orientation = ego_df["egomotion_estimate"].apply(lambda e: e["orientation"])
    return (
        pd.DataFrame(
            {
                "t_us": ego_df["key"].apply(lambda k: int(k["timestamp_micros"])).to_numpy(),
                "x": location.apply(lambda loc: float(loc["x"])).to_numpy(),
                "y": location.apply(lambda loc: float(loc["y"])).to_numpy(),
                "z": location.apply(lambda loc: float(loc["z"])).to_numpy(),
                "qw": orientation.apply(lambda q: float(q["w"])).to_numpy(),
                "qx": orientation.apply(lambda q: float(q["x"])).to_numpy(),
                "qy": orientation.apply(lambda q: float(q["y"])).to_numpy(),
                "qz": orientation.apply(lambda q: float(q["z"])).to_numpy(),
            }
        )
        .sort_values("t_us")
        .reset_index(drop=True)
    )


def object_table(obj_df: pd.DataFrame) -> pd.DataFrame:
    """Flatten object_fused.parquet into one row per (timestamp, obstacle)."""
    keys, fused = obj_df["key"], obj_df["object_fused"]
    return pd.DataFrame(
        {
            "t_us": keys.apply(lambda k: int(k["timestamp_micros"])).to_numpy(),
            "track_id": fused.apply(lambda f: int(f["obstacle_id"])).to_numpy(),
            "class_id": fused.apply(lambda f: int(f["obstacle_class"])).to_numpy(),
            "x": fused.apply(lambda f: float(f["cuboid_3D_center"]["x"])).to_numpy(),
            "y": fused.apply(lambda f: float(f["cuboid_3D_center"]["y"])).to_numpy(),
            "vx": fused.apply(lambda f: float(f["obstacle_velocity_mps"]["x"])).to_numpy(),
            "vy": fused.apply(lambda f: float(f["obstacle_velocity_mps"]["y"])).to_numpy(),
        }
    )
