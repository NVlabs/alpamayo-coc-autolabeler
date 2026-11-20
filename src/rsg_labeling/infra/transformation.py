# SPDX-License-Identifier: Apache-2.0
"""Native camera transforms and cuboid geometry."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import IO, TYPE_CHECKING, Any, Final, TypeVar

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from . import camera as ndas_camera_model

T = TypeVar("T")
RETRY_LOAD_MAX_ATTEMPTS = 5
RETRY_LOAD_INITIAL_BACKOFF_S = 0.1


def retry_load_text(
    path: str,
    parse: Callable[[IO[str]], T],
    retryable_exceptions: tuple[type[BaseException], ...],
    *,
    logger: logging.Logger,
    label: str,
    max_attempts: int = RETRY_LOAD_MAX_ATTEMPTS,
    initial_backoff_s: float = RETRY_LOAD_INITIAL_BACKOFF_S,
) -> T:
    """Retry transient file-load failures with exponential backoff.

    FileNotFoundError is always re-raised immediately (permanent, not transient).
    """
    last_error: BaseException | None = None
    backoff_s = initial_backoff_s
    for attempt in range(1, max_attempts + 1):
        try:
            with open(path) as reader:
                return parse(reader)
        except retryable_exceptions as exc:
            if isinstance(exc, FileNotFoundError):
                raise
            last_error = exc
            if attempt == max_attempts:
                break
            logger.warning(
                "%s retry %d/%d for %s (%s); retrying in %.2fs",
                label,
                attempt,
                max_attempts,
                path,
                exc,
                backoff_s,
            )
            time.sleep(backoff_s)
            backoff_s *= 2
    assert last_error is not None
    raise last_error


CROSS_LEFT_CAMERA_NAME: Final = "camera_cross_left_120fov"
CROSS_RIGHT_CAMERA_NAME: Final = "camera_cross_right_120fov"
FRONT_TELE_CAMERA_NAME: Final = "camera_front_tele_30fov"
FRONT_WIDE_CAMERA_NAME: Final = "camera_front_wide_120fov"
REAR_LEFT_CAMERA_NAME: Final = "camera_rear_left_70fov"
REAR_RIGHT_CAMERA_NAME: Final = "camera_rear_right_70fov"
REAR_TELE_CAMERA_NAME: Final = "camera_rear_tele_30fov"
CAMERA_NAMES_TO_INDICES = {
    CROSS_LEFT_CAMERA_NAME: 0,
    FRONT_WIDE_CAMERA_NAME: 1,
    CROSS_RIGHT_CAMERA_NAME: 2,
    REAR_LEFT_CAMERA_NAME: 3,
    REAR_TELE_CAMERA_NAME: 4,
    REAR_RIGHT_CAMERA_NAME: 5,
    FRONT_TELE_CAMERA_NAME: 6,
}
CAMERA_NAMES = tuple(name for name in CAMERA_NAMES_TO_INDICES)


def euler_2_so3(euler_angles: np.ndarray, degrees: bool = True, seq: str = "xyz") -> np.ndarray:
    """Converts the euler angles representation to the so3 rotation matrix
    Args:
        euler_angles (np.array): euler angles [n,3]
        degrees bool: True if angle is given in degrees else False
        seq string: sequence in which the euler angles are given

    Out:
        (np array): rotations given so3 matrix representation [n,3,3]
    """
    return (
        R.from_euler(seq=seq, angles=euler_angles, degrees=degrees).as_matrix().astype(np.float32)
    )


def _promote_common_dtype(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """Return the input tensors cast to their shared promotion dtype.

    Mixed-dtype matmul (``Float @ BFloat16``) raises ``RuntimeError`` in
    torch. In autograd-active code paths like the trajectory tokenizer,
    a bf16 model-output tensor is frequently combined with an fp32 anchor
    coming from the unaugmented ego-pose history. Rather than forcing
    every caller to remember to cast, centralize the promotion here so
    the transform is robust to callers with mixed dtypes.

    Output dtype is :func:`torch.promote_types` across all inputs. When
    all inputs share a dtype, this is a no-op and the input dtype is
    preserved. Callers that need the common dtype downstream can read
    it off the first returned tensor's ``.dtype``.

    Takes and returns a ``list`` (rather than ``*varargs`` / variadic
    ``tuple``) so that callers can be ``torch.jit.script``-compiled —
    TorchScript does not support variadic tensor arguments.
    """
    common = tensors[0].dtype
    for t in tensors[1:]:
        common = torch.promote_types(common, t.dtype)
    return [t if t.dtype == common else t.to(dtype=common) for t in tensors]


def xyzrot_to_corners(xyz: torch.Tensor, rot: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Get the corners of a 3d bounding box. The implementation interprets the tail dimensions as
    the meaningful ones, and everything else effectively as a batch dimension.

    Args:
        xyz: ...x3 location of the center of the bounding box
        rot: ...x3x3 orientation of the bounding box represented as a rotation matrix
        dims: ...x3 dimensions of the bounding box (length, width, height)

    Returns:
        corns: ...x8x3 corners of the bounding box. The first 4 points are the bottom
               corners and the next 4 are the top corners.
    """
    promoted = _promote_common_dtype([xyz, rot, dims])
    xyz, rot, dims = promoted[0], promoted[1], promoted[2]
    corns = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        device=xyz.device,
        dtype=xyz.dtype,
    )

    # scale
    corns = dims.unsqueeze(-2) * corns

    # rotate
    corns = (rot.unsqueeze(-3) @ corns.unsqueeze(-1)).squeeze(-1)

    # translate
    corns = corns + xyz.unsqueeze(-2)

    return corns


logger = logging.getLogger(__name__)


def get_sensor_to_sensor_flu(sensor: str) -> np.ndarray:
    """Compute a rotation matrix that rotates sensor to Front-Left-Up format.

    Args:
        sensor (str): sensor name.

    Returns:
        np.ndarray: the resulting rotation matrix.
    """
    if "cam" in sensor:
        rot = [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    else:
        rot = np.eye(4, dtype=np.float32)

    return np.asarray(rot, dtype=np.float32)


def sensor_to_rig(sensor: dict) -> np.ndarray | None:
    """Obtain sensor to rig transformation matrix."""
    sensor_name = sensor["name"]
    sensor_to_FLU = get_sensor_to_sensor_flu(sensor_name)

    if "nominalSensor2Rig_FLU" not in sensor:
        # Some sensors (like CAN sensors) don't have an associated sensorToRig
        return None

    nominal_T = sensor["nominalSensor2Rig_FLU"]["t"]
    nominal_R = sensor["nominalSensor2Rig_FLU"]["roll-pitch-yaw"]

    correction_T = np.zeros(3, dtype=np.float32)
    correction_R = np.zeros(3, dtype=np.float32)

    if "correction_rig_T" in sensor.keys():
        correction_T = sensor["correction_rig_T"]

    if "correction_sensor_R_FLU" in sensor.keys():
        assert "roll-pitch-yaw" in sensor["correction_sensor_R_FLU"].keys(), str(
            sensor["correction_sensor_R_FLU"]
        )
        correction_R = sensor["correction_sensor_R_FLU"]["roll-pitch-yaw"]

    nominal_R = euler_2_so3(nominal_R)
    correction_R = euler_2_so3(correction_R)

    R = nominal_R @ correction_R
    T = np.array(nominal_T, dtype=np.float32) + np.array(correction_T, dtype=np.float32)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = R
    transform[:3, 3] = T

    sensor_to_rig = transform @ sensor_to_FLU

    return sensor_to_rig


def box2camera(
    points: np.ndarray,
    camera_to_rig: np.ndarray,
    camera_intrinsic: ndas_camera_model.FThetaCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the 3D bounding box to the camera image.

    Args:
        points: (N, 3) The corner points of 3D bounding box in global coordinate.
        camera_to_rig: (4, 4) The camera extrinsics.
        camera_intrinsic: The camera intrinsics.

    Returns:
        uv: The projected points in pixel coordinate.
        z: The depth of the points.
    """
    # closed-form inversion
    rig_to_camera = np.zeros_like(camera_to_rig)
    rig_to_camera[:3, :3] = camera_to_rig[:3, :3].T
    rig_to_camera[:3, 3] = -camera_to_rig[:3, :3].T @ camera_to_rig[:3, 3]
    camera_xyz = (points @ rig_to_camera[:3, :3].T) + rig_to_camera[:3, 3]
    uv = camera_intrinsic.ray2pixel(camera_xyz)
    z = camera_xyz[:, 2]
    return uv, z


def compute_preprocessing_transform(
    camera_intrinsic: ndas_camera_model.FThetaCamera, maglev_conf: dict
) -> tuple[float, float]:
    """Pre-processing due to crop and resize.

    Args:
        camera_intrinsic: The FTheta camera intrinsic.
        maglev_conf: The maglev configuration from generator_config.yaml.

    Returns:
        xbias: The x-bias due to the crop.
        ybias: The y-bias due to the crop.
        facw: The factor to rescale width.
        fach: The factor to rescale height.
    """
    og_width = int(camera_intrinsic.width)
    og_height = int(camera_intrinsic.height)

    # TODO in the calibration comments, it seems like the
    # original size is 3840x2160? But actually sometimes the
    # calibration says the original size is 3848x2168

    # the pre-processing steps are
    # 1) bottom crop
    # 2) symmetric left-right crop
    # 3) rescaling

    # for transformation parameters, the bottom crop
    # only changes the extent. So in terms of the
    # change to the coordinate frame, we only take
    # into account the left-right crop and rescaling.

    xbias = (og_width - maglev_conf["crop_w"]) / 2
    ybias = (og_height - maglev_conf["crop_h"]) / 2
    # make sure resize is equal
    facw = maglev_conf["crop_w"] / maglev_conf["size_w"]
    fach = maglev_conf["crop_h"] / maglev_conf["size_h"]
    assert facw == fach, f"{facw} {fach}"
    return xbias, ybias, facw, fach


@lru_cache(maxsize=8)
def _load_generator_config_with_retry(datapath: str) -> dict[str, Any]:
    """Load generator config with retries and per-process cache.

    Cached for the lifetime of the process; no automatic invalidation on disk changes.
    """
    loaded = retry_load_text(
        datapath,
        yaml.safe_load,
        (OSError, yaml.YAMLError),
        logger=logger,
        label="generator config",
    )
    if not isinstance(loaded, dict):
        raise TypeError(f"Unexpected generator config type {type(loaded)!r} at {datapath}")
    return loaded


def get_video_parameters(
    camkeys: Iterable[str],
    datapath: str | None = None,
) -> dict[str, int]:
    """Load the pre-processing config file for the dataset.

    Currently we assert that all cameras use the same parameters with the only difference of
    size_w/h, we could loosen that in the future.
    """
    collect = {"crop_h": [], "crop_w": [], "size_h": [], "size_w": [], "fps": []}
    if datapath is None:
        datapath = __file__.replace("transformation.py", "generator_config.yaml")
    if not os.path.isfile(datapath):
        raise FileNotFoundError(
            f"Generator config not found: {datapath}\n"
            "If this is a Lustre path, ensure the filesystem is mounted. "
            "For local/CI, use a test config with a bundled generator_config_path."
        )
    conf = _load_generator_config_with_retry(datapath)

    for camkey in camkeys:
        for key, value_list in collect.items():
            value_list.append(conf["data"][camkey][key])
    # make sure they're all the same
    for key, value_list in collect.items():
        assert len(set(value_list)) == 1, value_list

    # select the first one since they're all the same
    out = {k: v[0] for k, v in collect.items()}

    return out
