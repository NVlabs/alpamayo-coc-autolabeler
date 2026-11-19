# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
"""Calibrated native fisheye camera model."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import brentq, curve_fit

from . import transformation, transformation as constants


class FThetaCamera:
    """Defines an FTheta camera model.

    Rectification depends on a ``generator_config.yaml`` that describes the
    crop/resize preprocessing applied to the source images.  Two pipelines
    use different configs:

    * **WebDataset** -- images are pre-resized to 512x320.  The bundled
      default ``generator_config.yaml`` beside the transformation module
      matches this layout. Used when ``generator_config_path`` is *None*.
    * **Native-resolution images** -- 2MP images (1920x1080).  Requires an
      explicit ``generator_config_path`` pointing to a matching calibration
      supplied by the caller.

    Callers that only ever process WebDataset data (e.g.
    ``from_intrinsics_array``) can omit the path and get correct defaults.
    Callers processing native-resolution camera data **must** pass the appropriate config path
    via ``from_dict(rig_dict, generator_config_path=...)`` or the
    constructor.
    """

    __hash__ = None  # unhashable due to mutable numpy array attributes

    @classmethod
    def from_rig(cls, rig_file: str, sensor_name: str):
        """Initialize a new object using a rig file and the sensor's name.

        Args:
            rig_file (str): the rig file path.
            sensor_name (str): the name of the sensor.

        Returns:
            FThetaCamera: the newly created object.
        """
        with open(rig_file) as fp:
            rig = json.load(fp)

        # Parse the properties from the rig file
        sensors = rig["rig"]["sensors"]
        sensor = None
        sensor_found = False

        for sensor in sensors:
            if sensor["name"] == sensor_name:
                sensor_found = True
                break

        if not sensor_found:
            raise ValueError(f"The camera '{sensor_name}' was not found in the rig!")

        return cls.from_dict(sensor)

    @classmethod
    def from_dict(
        cls,
        rig_dict: dict[str, Any],
        generator_config_path: str | None = None,
    ):
        """Initialize from a rig sensor dictionary.

        Args:
            rig_dict: The sensor dictionary to initialize with.
            generator_config_path: Path to a ``generator_config.yaml`` that
                describes the crop/resize preprocessing applied to the source
                images.  When *None* the bundled WebDataset config is used
                (512x320 targets).  Pass a matching calibration config for 2MP data.

        Returns:
            FThetaCamera: the newly created object.
        """
        (
            cx,
            cy,
            width,
            height,
            poly,
            poly_type,
        ) = FThetaCamera.get_ftheta_parameters_from_json(rig_dict)
        return cls(
            cx,
            cy,
            width,
            height,
            poly,
            poly_type=poly_type,
            generator_config_path=generator_config_path,
        )

    @classmethod
    def from_intrinsics_array(cls, intrinsics: np.ndarray):
        """Initialize from a flat intrinsics array (WebDataset defaults only).

        This classmethod does not accept ``generator_config_path`` and always
        uses the bundled WebDataset generator config (512x320).  Callers
        processing native-resolution 2MP data should use :meth:`from_dict` with an
        explicit path instead.

        Args:
            intrinsics (np.ndarray): the intrinsics array. The ordering is
                expected to be "cx, cy, width, height, bw_poly".
                This is the same ordering as the `intrinsics`
                property of this class.

        Returns:
            FThetaCamera: the newly created object.
        """
        return cls(
            cx=intrinsics[0],
            cy=intrinsics[1],
            width=intrinsics[2],
            height=intrinsics[3],
            bw_poly=intrinsics[4:],
        )

    def __init__(
        self,
        cx: float,
        cy: float,
        width: int,
        height: int,
        bw_poly: np.ndarray,
        focal_length_multiplier: float = 1.0,
        target_width: int | None = None,
        target_height: int | None = None,
        poly_type: str = "pixeldistance-to-angle",
        generator_config_path: str | None = None,
    ):
        """The __init__ method.

        Args:
            cx (float): optical center x.
            cy (float): optical center y.
            width (int): the width of the image.
            height (int): the height of the image.
            bw_poly (np.ndarray): the polynomial of the FTheta model. Despite the name,
                this can be either a backward (pixeldistance-to-angle) or forward
                (angle-to-pixeldistance) polynomial, controlled by ``poly_type``.
            focal_length_multiplier (float): the focal length multiplier, defaults to 1.0.
            target_width: Width of the rectified output image.  When *None*,
                read from ``size_w`` in the generator config.
            target_height: Height of the rectified output image.  When *None*,
                read from ``size_h`` in the generator config.
            poly_type (str): the type of the polynomial. Either "pixeldistance-to-angle"
                (backward, default) or "angle-to-pixeldistance" (forward).
            generator_config_path: Path to a ``generator_config.yaml`` that
                describes the crop/resize preprocessing applied to the source
                images.  When *None* the bundled WebDataset default is used.
        """
        self._center = np.asarray([cx, cy], dtype=np.float32)
        self._width = int(width)
        self._height = int(height)
        self._focal_length_multiplier = focal_length_multiplier

        if poly_type == "angle-to-pixeldistance":
            self._fw_poly = Polynomial(bw_poly)
            self._bw_poly = self._compute_bw_poly(self._fw_poly)
        elif poly_type == "pixeldistance-to-angle":
            self._bw_poly = Polynomial(bw_poly)
            self._fw_poly = self._compute_fw_poly()
        else:
            raise ValueError(f"Unsupported polynomial type: {poly_type}")
        # Other properties that need to be computed
        self._horizontal_fov = None
        self._vertical_fov = None
        self._max_angle = None
        self._max_ray_angle = None
        # Populate the array of intrinsics
        self._intrinsics = np.append([cx, cy, width, height], bw_poly).astype(np.float32)

        self._update_calibrated_camera()

        maglev_conf = transformation.get_video_parameters(
            constants.CAMERA_NAMES, datapath=generator_config_path
        )
        self._target_width = int(
            target_width if target_width is not None else maglev_conf["size_w"]
        )
        self._target_height = int(
            target_height if target_height is not None else maglev_conf["size_h"]
        )

        (
            valid_source_pixels,
            valid_target_pixels,
            rectified_normalized_intrinsics,
            target_resolution,
        ) = self._compute_rectified_intrinsics(
            self._width,
            self._height,
            self._target_width,
            self._target_height,
            self._focal_length_multiplier,
            maglev_conf,
        )

        self.rectified_valid_source_pixels = valid_source_pixels
        self.rectified_valid_target_pixels = valid_target_pixels
        self.target_resolution = target_resolution
        self._rectified_normalized_intrinsics = rectified_normalized_intrinsics

    @property
    def rectified_normalized_intrinsics(self) -> np.ndarray:
        """Obtain an array of the rectified intrinsics of this camera model.

        Returns:
            np.ndarray: an array of rectified intrinsics. The ordering is
                "cx, cy, width, height, bw_poly". dtype is np.float32.
        """
        return self._rectified_normalized_intrinsics

    @staticmethod
    def get_ftheta_parameters_from_json(rig_dict: dict[str, Any]) -> tuple[Any]:
        """Helper method for obtaining FTheta camera model parameters.

        Args:
            rig_dict (Dict[str, Any]): the rig dictionary to parse.

        Raises:
            ValueError: if the provided rig is not supported.
            AssertionError: if the provided model is supported, but cannot be
                parsed properly.

        Returns:
            Tuple[Any]: the values `cx`, `cy`, `width`, `height`, `poly`, and
                `poly_type` that were parsed.
        """
        props = rig_dict["properties"]

        if props["Model"] != "ftheta":
            raise ValueError("The given camera is not an FTheta camera")

        cx = float(props["cx"])
        cy = float(props["cy"])
        width = int(props["width"])
        height = int(props["height"])

        poly_type = "pixeldistance-to-angle"
        if "bw-poly" in props:  # Is this a regular rig?
            poly = props["bw-poly"]
        elif "polynomial" in props:  # Is this a VT rig?
            # VT rigs have a slightly different format, so need to handle these
            # specifically. Refer to the following thread for more details:
            poly_type = props["polynomial-type"]

            linear_c = float(props["linear-c"]) if "linear-c" in props else None
            linear_d = float(props["linear-d"]) if "linear-d" in props else None
            linear_e = float(props["linear-e"]) if "linear-e" in props else None

            # If we had all the terms present, sanity check to make sure
            # they are [1, 0, 0]

            if linear_c is not None and linear_d is not None and linear_e is not None:
                if linear_c != 1.0:
                    raise ValueError(
                        f"Expected `linear-c` term to be 1.0 (got {linear_c}. Rig:\n{rig_dict})"
                    )
                if linear_d != 0.0:
                    raise ValueError(
                        f"Expected `linear-d` term to be 0.0 (got {linear_d}. Rig:\n{rig_dict})"
                    )
                if linear_e != 0.0:
                    raise ValueError(
                        f"Expected `linear-e` term to be 0.0 (got {linear_e}. Rig:\n{rig_dict})"
                    )

            # If we're here, then it means we can parse the rig successfully.
            poly = props["polynomial"]
        else:
            raise ValueError(
                f"Unable to parse the rig. Only FTheta rigs are supported! Rig:\n{rig_dict}"
            )

        parsed_poly = [np.float32(val) for val in poly.split()]
        return cx, cy, width, height, parsed_poly, poly_type

    @property
    def fov(self) -> tuple:
        """Returns a tuple of horizontal and vertical fov of the sensor."""
        if self._vertical_fov is None or self._horizontal_fov is None:
            self._compute_fov()
        return self._horizontal_fov, self._vertical_fov

    @property
    def width(self) -> int:
        """Returns the width of the sensor."""
        return self._width

    @property
    def height(self) -> int:
        """Returns the height of the sensor."""
        return self._height

    @property
    def center(self) -> np.ndarray:
        """Returns the center of the sensor."""
        return self._center

    @property
    def intrinsics(self) -> np.ndarray:
        """Obtain an array of the intrinsics of this camera model.

        Returns:
            np.ndarray: an array of intrinsics. The ordering is
                "cx, cy, width, height, bw_poly". dtype is np.float32.
        """
        return self._intrinsics

    def __str__(self):
        """Returns a string representation of this object."""
        return (
            f"FTheta camera model:\n\t{self._bw_poly}\n\t"
            f"center={self._center}\n\twidth={self._width}\n\theight={self._height}\n\t"
            f"h_fov={np.degrees(self._horizontal_fov)}\n\tv_fov={np.degrees(self._vertical_fov)}"
        )

    def __eq__(self, other):
        """Check if two FThetaCamera objects are equal."""
        if not isinstance(other, FThetaCamera):
            return NotImplemented
        return (
            np.array_equal(self._intrinsics, other._intrinsics)
            and np.array_equal(self._center, other._center)
            and self._width == other._width
            and self._height == other._height
            and self._target_width == other._target_width
            and self._target_height == other._target_height
            and np.array_equal(self._bw_poly.coef, other._bw_poly.coef)
            and np.array_equal(self._fw_poly.coef, other._fw_poly.coef)
        )

    def _update_calibrated_camera(self):
        """Updates the internals of this object after calculating properties."""
        self._compute_fov()
        self._max_ray_angle = (self._max_angle).copy()
        is_fw_poly_slope_negative_in_domain = False
        ray_angle = (np.float32(self._max_ray_angle)).copy()
        deg2rad = np.pi / 180.0
        while ray_angle >= np.float32(0.0):
            temp_dval = self._fw_poly.deriv()(self._max_ray_angle).item()
            if temp_dval < 0:
                is_fw_poly_slope_negative_in_domain = True
            ray_angle -= deg2rad * np.float32(1.0)

        if is_fw_poly_slope_negative_in_domain:
            ray_angle = (np.float32(self._max_ray_angle)).copy()
            while ray_angle >= np.float32(0.0):
                ray_angle -= deg2rad * np.float32(1.0)
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion within image interior is negative"
            )

        # Evaluate the forward polynomial at point (self._max_ray_angle, 0)
        # Also evaluate its derivative at the same point
        val = self._fw_poly(self._max_ray_angle).item()
        dval = self._fw_poly.deriv()(self._max_ray_angle).item()

        if dval < 0:
            raise ArithmeticError(
                "FThetaCamera: derivative of distortion at edge of image is negative"
            )

        self._max_ray_distortion = np.asarray([val, dval], dtype=np.float32)

    def _compute_rectified_intrinsics(
        self, width, height, target_width, target_height, focal_length_multiplier, maglev_conf
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        """Compute the rectification look-up tables and normalized intrinsics.

        Args:
            width: Sensor image width.
            height: Sensor image height.
            target_width: Desired rectified output width.
            target_height: Desired rectified output height.
            focal_length_multiplier: Focal length scaling factor.
            maglev_conf: Preprocessing config dict from ``get_video_parameters``.
        """
        normalized_centers = (self.center[0] / width, self.center[1] / height)

        target_resolution_w = round(target_width * focal_length_multiplier)
        target_resolution_h = round(target_height * focal_length_multiplier)
        target_resolution = (target_resolution_w, target_resolution_h)
        target_principal_point = (
            target_resolution_w * normalized_centers[0],
            target_resolution_h * normalized_centers[1],
        )
        xbias, ybias, facw, _ = transformation.compute_preprocessing_transform(self, maglev_conf)
        target_focal_length = (
            np.array(
                [self._fw_poly.coef[1], self._fw_poly.coef[1]],
                dtype=np.float32,
            )
            / facw
        )

        # get the rectified pixel coords
        # target resolution in width, height
        target_pixels_x, target_pixels_y = np.meshgrid(
            np.arange(target_resolution[0], dtype=np.int16),
            np.arange(target_resolution[1], dtype=np.int16),
        )  # [0, 1, 2, ..., w-1], [0, 1, 2, ..., h-1]
        target_pixels = np.stack([target_pixels_x.flatten(), target_pixels_y.flatten()], axis=1)
        target_image_points = target_pixels.astype(np.float32) + 0.5
        target_rays = (target_image_points - target_principal_point) / target_focal_length
        target_rays = np.concatenate([target_rays, np.ones_like(target_rays[:, 0:1])], axis=1)

        source_pixels = self.ray2pixel(target_rays)
        source_pixels[..., 0] -= xbias
        source_pixels[..., 1] -= ybias
        source_pixels /= facw
        valid = (
            (source_pixels[..., 0] < target_width)
            & (source_pixels[..., 0] >= 0.0)
            & (source_pixels[..., 1] < target_height)
            & (source_pixels[..., 1] >= 0.0)
        )
        valid_source_pixels = source_pixels[valid].astype(np.int16)
        valid_target_pixels = target_pixels[valid]

        rectified_normalized_intrinsics = np.array(
            [
                target_focal_length[0] / target_resolution[0],
                target_focal_length[1] / target_resolution[1],
                target_principal_point[0] / target_resolution[0],
                target_principal_point[1] / target_resolution[1],
            ],
            dtype=np.float32,
        )

        return (
            valid_source_pixels,
            valid_target_pixels,
            rectified_normalized_intrinsics,
            target_resolution,
        )

    def _compute_fw_poly(self):
        """Computes the forward polynomial for this camera.

        This function is a replication of the logic in the following file from
            the DW repo: src/dw/calibration/cameramodel/CameraModels.cpp
        """

        def get_max_value(p0, p1):
            return np.linalg.norm(np.asarray([p0, p1], dtype=self._center.dtype) - self._center)

        max_value = 0.0

        size = (self._width, self._height)
        value = get_max_value(0.0, 0.0)
        max_value = max(max_value, value)
        value = get_max_value(0.0, size[1])
        max_value = max(max_value, value)
        value = get_max_value(size[0], 0.0)
        max_value = max(max_value, value)
        value = get_max_value(size[0], size[1])
        max_value = max(max_value, value)

        SAMPLE_COUNT = 500
        samples_x = []
        samples_b = []
        step = max_value / SAMPLE_COUNT
        x = step

        for _ in range(SAMPLE_COUNT):
            p = np.asarray([self._center[0] + x, self._center[1]], dtype=np.float32)
            ray, _ = self.pixel2ray(p)
            xy_norm = np.linalg.norm(ray[0, :2])
            theta = np.arctan2(float(xy_norm), float(ray[0, 2]))
            samples_x.append(theta)
            samples_b.append(float(x))
            x += step

        x = np.asarray(samples_x, dtype=np.float64)
        y = np.asarray(samples_b, dtype=np.float64)
        # Fit a 4th degree polynomial. The polynomial function is as follows:

        def f(x, b, x1, x2, x3, x4):
            """4th degree polynomial."""
            return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))

        # The constant in the polynomial should be zero, so add the `bounds`
        #  condition.
        # FIXME(mmaghoumi) DW mentions disabling input normalization,
        #   what's that??
        # - the computation is more stable if the data is normalized before the
        #   fitting process.
        coeffs, _ = curve_fit(
            f,
            x,
            y,
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        # Return the polynomial and hardcode the bias value to 0
        return Polynomial([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)])

    def _compute_bw_poly(self, fw_poly: Polynomial) -> Polynomial:
        """Computes the backward polynomial for this camera, given the forward polynomial.

        The forward polynomial (fw_poly) maps: theta (angle) -> radius.
        The backward polynomial (bw_poly) maps: radius -> theta (angle).

        This function mirrors the logic of ``_compute_fw_poly``. While ``_compute_fw_poly``
        samples radius values and uses ``pixel2ray`` (bw_poly) to get theta, this function
        samples theta values and uses ``fw_poly`` directly to get radius, then fits a
        polynomial that maps radius back to theta.

        Args:
            fw_poly (Polynomial): the forward polynomial.

        Returns:
            Polynomial: the backward polynomial.
        """

        def get_max_value(p0, p1):
            return np.linalg.norm(np.asarray([p0, p1], dtype=self._center.dtype) - self._center)

        max_radius = 0.0
        size = (self._width, self._height)
        max_radius = max(max_radius, get_max_value(0.0, 0.0))
        max_radius = max(max_radius, get_max_value(0.0, size[1]))
        max_radius = max(max_radius, get_max_value(size[0], 0.0))
        max_radius = max(max_radius, get_max_value(size[0], size[1]))

        # Find maximum theta by solving fw_poly(theta) = max_radius.
        # Search in range [0, pi] for fisheye cameras.
        try:
            max_theta = brentq(lambda t: fw_poly(t) - max_radius, 0, np.pi)
        except ValueError:
            # If no solution found, use pi as upper bound
            max_theta = np.pi

        SAMPLE_COUNT = 500
        samples_x = []  # radius values
        samples_b = []  # theta values
        step = max_theta / SAMPLE_COUNT
        theta = step

        for _ in range(SAMPLE_COUNT):
            radius = float(fw_poly(theta))
            samples_x.append(radius)
            samples_b.append(float(theta))
            theta += step

        x = np.asarray(samples_x, dtype=np.float64)
        y = np.asarray(samples_b, dtype=np.float64)

        def f(x, b, x1, x2, x3, x4):
            """4th degree polynomial."""
            return b + x * (x1 + x * (x2 + x * (x3 + x * x4)))

        coeffs, _ = curve_fit(
            f,
            x,
            y,
            bounds=(
                [0, -np.inf, -np.inf, -np.inf, -np.inf],
                [np.finfo(np.float64).eps, np.inf, np.inf, np.inf, np.inf],
            ),
        )
        return Polynomial([np.float32(val) if i > 0 else 0 for i, val in enumerate(coeffs)])

    def pixel2ray(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Backproject 2D pixels into 3D rays.

        Args:
            x (np.ndarray): the pixels to backproject. Size of (n_points, 2),
                where the first column contains the `x` values, and the second
                column contains the `y` values.

        Returns:
            rays (np.ndarray): the backprojected 3D rays. Size of (n_points, 3).
            valid (np.ndarray): bool flag indicating the validity of each
                backprojected pixel.
        """
        # Make sure x is n x 2
        if np.ndim(x) == 1:
            x = x[np.newaxis, :]

        # Fix the type
        x = x.astype(np.float32)
        xd = x - self._center
        xd_norm = np.linalg.norm(xd, axis=1, keepdims=True)
        alpha = self._bw_poly(xd_norm)
        sin_alpha = np.sin(alpha)

        rx = sin_alpha * xd[:, 0:1] / xd_norm
        ry = sin_alpha * xd[:, 1:] / xd_norm
        rz = np.cos(alpha)

        rays = np.hstack((rx, ry, rz))
        # special case: ray is perpendicular to image plane normal
        valid = (xd_norm > np.finfo(np.float32).eps).squeeze()
        rays[~valid, :] = (0, 0, 1)  # This is what DW sets these rays to

        # note:
        # if constant coefficient of bwPoly is non-zero,
        # the resulting ray might not be normalized.
        return rays.astype(np.float32), valid

    def ray2pixel(self, rays: np.ndarray) -> np.ndarray:
        """Project 3D rays to 2D pixel coordinates.

        Args:
            rays (np.ndarray): the rays.

        Returns:
            result (np.ndarray): the projected pixel coordinates.
        """
        # Make sure the input shape is (n_points, 3)
        if np.ndim(rays) == 1:
            rays = rays[np.newaxis, :]

        # Fix the type
        rays = rays.astype(np.float32)
        xy_norm = np.linalg.norm(rays[:, :2], axis=1, keepdims=True)
        cos_alpha = rays[:, 2:] / np.linalg.norm(rays, axis=1, keepdims=True)

        alpha = np.empty_like(cos_alpha)
        cos_alpha_condition = np.logical_and(
            cos_alpha > np.float32(-1.0), cos_alpha < np.float32(1.0)
        ).squeeze()
        alpha[cos_alpha_condition] = np.arccos(cos_alpha[cos_alpha_condition])
        alpha[~cos_alpha_condition] = xy_norm[~cos_alpha_condition]

        delta = np.empty_like(cos_alpha)
        alpha_cond = alpha <= self._max_ray_angle
        delta[alpha_cond] = self._fw_poly(alpha[alpha_cond])
        # For outside the model (which need to do linear extrapolation)
        delta[~alpha_cond] = (
            self._max_ray_distortion[0]
            + (alpha[~alpha_cond] - self._max_ray_angle) * self._max_ray_distortion[1]
        )

        # Determine the bad points with a norm of zero, and avoid div by zero
        bad_norm = xy_norm <= 0
        xy_norm[bad_norm] = 1
        delta[bad_norm] = 0
        # compute pixel relative to center
        scale = delta / xy_norm
        pixel = scale * rays

        # Handle the edge cases (ray along image plane normal)
        edge_case_cond = (xy_norm <= np.float32(0.0)).squeeze()
        pixel[edge_case_cond, :] = rays[edge_case_cond, :]
        result = pixel[:, :2] + self._center
        return result

    def _get_pixel_fov(self, pt: np.ndarray) -> float:
        """Gets the FOV for a given point. Used internally for FOV computation.

        Args:
            pt (np.ndarray): 2D pixel.

        Returns:
            fov (float): the FOV of the pixel.
        """
        ray, _ = self.pixel2ray(pt)
        fov = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return fov

    def _compute_fov(self):
        """Computes the FOV of this camera model."""
        max_x = self._width - 1
        max_y = self._height - 1

        point_left = np.asarray([0, self._center[1]], dtype=np.float32)
        point_right = np.asarray([max_x, self._center[1]], dtype=np.float32)
        point_top = np.asarray([self._center[0], 0], dtype=np.float32)
        point_bottom = np.asarray([self._center[0], max_y], dtype=np.float32)

        fov_left = self._get_pixel_fov(point_left)
        fov_right = self._get_pixel_fov(point_right)
        fov_top = self._get_pixel_fov(point_top)
        fov_bottom = self._get_pixel_fov(point_bottom)

        self._vertical_fov = fov_top + fov_bottom
        self._horizontal_fov = fov_left + fov_right
        self._compute_max_angle()

    def _compute_max_angle(self):
        """Computes the maximum ray angle for this camera."""
        max_x = self._width - 1
        max_y = self._height - 1

        p = np.asarray([[0, 0], [max_x, 0], [0, max_y], [max_x, max_y]], dtype=np.float32)

        self._max_angle = max(  # noqa: PLW3301
            max(self._get_pixel_fov(p[0, ...]), self._get_pixel_fov(p[1, ...])),
            max(self._get_pixel_fov(p[2, ...]), self._get_pixel_fov(p[3, ...])),
        )

    def to_pinhole(self) -> np.ndarray:
        """Convert this FTheta camera to a pinhole camera model assuming a linear polynomial.

        Under the linear approximation, the forward polynomial r = f(theta) is reduced to
        its first-order term: r ≈ c_1 * theta. For a pinhole camera, r = f * tan(theta),
        which for small angles gives r ≈ f * theta. Therefore, the focal length is
        approximated as f = c_1 (the linear coefficient of the forward polynomial).

        Since the FTheta model is rotationally symmetric, f_x = f_y = c_1.

        Returns:
            np.ndarray: a 3x3 pinhole intrinsic matrix K of the form::

                [[f,  0, cx],
                 [0,  f, cy],
                 [0,  0,  1]]

            where f is the linear coefficient of the forward polynomial and
            (cx, cy) is the optical center.
        """
        f = float(self._fw_poly.coef[1])
        cx, cy = float(self._center[0]), float(self._center[1])
        return np.array(
            [[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def is_ray_inside_fov(self, ray: np.ndarray) -> bool:
        """Determines whether a given ray is inside the FOV of this camera.

        Args:
            ray (np.ndarray): the 3D ray.

        Returns:
            bool: whether the ray is inside the FOV.
        """
        if np.ndim(ray) == 1:
            ray = ray[np.newaxis, :]

        ray_angle = np.arctan2(np.linalg.norm(ray[:, :2], axis=1), ray[:, 2])
        return ray_angle <= self._max_angle
