"""Calibrated box projection: an env data-access capability.

Deterministic geometry over the env's own data — the clip calibration (rig_json -> ftheta
camera model + extrinsics, via the shared ``alpamayo`` src) projects obstacle cuboids
world->rig at a frame's timestamp onto the image plane. This is *capability*, not policy:
whether to render enhanced visual context is the agent's decision (the render tool lives in
``agent/render.py``). Heavy imports load lazily so runs without projection never pay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...common.clipgt import project_obstacle_box, quaternion_to_matrix
from .loaders import ClipData


@dataclass(frozen=True)
class ProjectedBox:
    """One obstacle's axis-aligned pixel box on a target image."""

    track_id: str
    x0: int
    y0: int
    x1: int
    y1: int
    distance_m: float


class Projector:
    """Projects a clip's 3D obstacles onto its cameras' frames as labeled pixel boxes."""

    def __init__(self, data: ClipData) -> None:
        """Parse the rig calibration and index obstacles/ego poses by timestamp."""
        if data.calibration is None:
            raise ValueError("clip carries no calibration_estimate; cannot project boxes")
        rig = json.loads(data.calibration["calibration_estimate"].iloc[0]["rig_json"])
        self._sensors = {
            str(sensor["name"]).replace(":", "_"): sensor
            for sensor in rig["rig"]["sensors"]
            if str(sensor.get("protocol", "")).startswith("camera")
        }
        self._models: dict[str, tuple[Any, np.ndarray]] = {}
        self._obstacles_by_t: dict[int, list[dict]] = {}
        for key, obstacle in zip(
            data.objects_raw["key"], data.objects_raw["object_fused"], strict=True
        ):
            self._obstacles_by_t.setdefault(int(key["timestamp_micros"]), []).append(obstacle)
        self._obstacle_ts = np.array(sorted(self._obstacles_by_t), dtype=np.int64)
        self._ego = data.ego

    def boxes_for(
        self,
        camera: str,
        timestamp_us: int,
        target_w: int,
        target_h: int,
        *,
        max_distance_m: float = 40.0,
        max_boxes: int = 8,
    ) -> list[ProjectedBox]:
        """The nearest obstacles visible on ``camera`` at the nearest obstacle timestamp."""
        model = self._camera_model(camera)
        if model is None or len(self._obstacle_ts) == 0:
            return []
        nearest = int(self._obstacle_ts[int(np.argmin(np.abs(self._obstacle_ts - timestamp_us)))])
        if abs(nearest - timestamp_us) > 100_000:
            return []
        ego_idx = int(np.argmin(np.abs(self._ego["t_us"].to_numpy() - timestamp_us)))
        ego_location = self._ego.iloc[ego_idx][["x", "y", "z"]].to_numpy(dtype=np.float64)
        ego_rotation = quaternion_to_matrix(
            *self._ego.iloc[ego_idx][["qw", "qx", "qy", "qz"]].tolist()
        )
        boxes = []
        for obstacle in self._obstacles_by_t[nearest]:
            box = _project_obstacle(obstacle, model, ego_location, ego_rotation, target_w, target_h)
            if box is not None and box.distance_m <= max_distance_m:
                boxes.append(box)
        boxes.sort(key=lambda box: box.distance_m)
        return boxes[:max_boxes]

    def visible_tracks(
        self,
        camera: str,
        timestamp_us: int,
        target_w: int,
        target_h: int,
        *,
        min_area_px: float = 100.0,
        max_distance_m: float = 120.0,
    ) -> set[str]:
        """Track ids whose projected box lands in-image at the nearest label timestamp.

        The env's label-visibility annotation: unlike ``boxes_for`` (a render capability)
        there is no top-k cap -- every actor is tested -- and a minimum projected area
        keeps off-image slivers from counting as visible (the uncculled-projector
        lesson). Occlusion is NOT modeled: a projected-but-occluded object still counts,
        which is documented as imperfect for re-identification (the appearance judge is
        the backstop).
        """
        model = self._camera_model(camera)
        if model is None or len(self._obstacle_ts) == 0:
            return set()
        nearest = int(self._obstacle_ts[int(np.argmin(np.abs(self._obstacle_ts - timestamp_us)))])
        if abs(nearest - timestamp_us) > 100_000:
            return set()
        ego_idx = int(np.argmin(np.abs(self._ego["t_us"].to_numpy() - timestamp_us)))
        ego_location = self._ego.iloc[ego_idx][["x", "y", "z"]].to_numpy(dtype=np.float64)
        ego_rotation = quaternion_to_matrix(
            *self._ego.iloc[ego_idx][["qw", "qx", "qy", "qz"]].tolist()
        )
        visible: set[str] = set()
        for obstacle in self._obstacles_by_t[nearest]:
            box = _project_obstacle(obstacle, model, ego_location, ego_rotation, target_w, target_h)
            if box is None or box.distance_m > max_distance_m:
                continue
            if (box.x1 - box.x0) * (box.y1 - box.y0) >= min_area_px:
                visible.add(box.track_id)
        return visible

    def _camera_model(self, camera: str) -> tuple[Any, np.ndarray] | None:
        """Parse and cache the ftheta model + camera->rig extrinsics for one camera."""
        if camera in self._models:
            return self._models[camera]
        sensor = self._sensors.get(camera)
        if sensor is None:
            return None
        from ...infra.camera import FThetaCamera
        from ...infra.transformation import sensor_to_rig

        camera_to_rig = sensor_to_rig(sensor)
        if camera_to_rig is None:
            return None
        model = (FThetaCamera.from_dict(sensor), np.asarray(camera_to_rig, dtype=np.float64))
        self._models[camera] = model
        return model


def _project_obstacle(
    obstacle: dict,
    model: tuple[Any, np.ndarray],
    ego_location: np.ndarray,
    ego_rotation: np.ndarray,
    target_w: int,
    target_h: int,
) -> ProjectedBox | None:
    """Project one cuboid to a pixel box, or ``None`` when not visibly in front."""
    ftheta, camera_to_rig = model
    projected = project_obstacle_box(
        obstacle, ftheta, camera_to_rig, ego_location, ego_rotation, target_w, target_h
    )
    if projected is None:
        return None
    x0, y0, x1, y1, distance_m = projected
    return ProjectedBox(
        track_id=f"track_{int(obstacle['obstacle_id'])}",
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        distance_m=round(distance_m, 1),
    )
