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

"""Video loader module for processing FPV and BEV video frames with timestamp mapping.

This module provides the VideoLoader class for loading and processing video frames
from First Person View (FPV) and Bird's Eye View (BEV) video files. It supports
temporal segmentation, frame subsampling, timestamp mapping, and frame identifier
generation.

Key features:
- Video frame loading with FPS subsampling
- Temporal segment extraction (historical and future frames)
- Timestamp mapping and caching for efficient lookups
- Frame identifier generation with metadata
- Support for both FPV and BEV video streams
- Consistent handling of video files and timestamp data

Example:
    from coc_labeling.data_loader.video_loader import VideoLoader

    # Initialize with configuration objects
    loader = VideoLoader(video_config, data_config)

    # Load video frames and frame identifiers
    result = loader.load("clip_123", event_start=1000)

    # Access frame data
    fpv_frames = result["all_fpv_frames"]
    frame_info = result["all_fpv_frames_info"]
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from omegaconf import DictConfig

try:
    import cv2
except ImportError:
    cv2 = None


def _require_cv2() -> Any:
    """Return OpenCV or raise a clear error for video-processing operations."""
    if cv2 is None:
        raise ImportError(
            "OpenCV is required for video processing. Install `opencv-python` "
            "or `opencv-python-headless`."
        )
    return cv2


# Constants
class VideoConstants:
    """Constants used throughout the VideoLoader class."""

    # File extensions
    VIDEO_EXTENSION = ".mp4"
    TIMESTAMP_EXTENSION = ".timestamps"
    TIMESTAMP_PARQUET_EXTENSION = ".timestamps.parquet"

    # Video codec
    VIDEO_CODEC = "mp4v"

    # Default video dimensions
    DEFAULT_FRAME_WIDTH = 1920
    DEFAULT_FRAME_HEIGHT = 1080

    # FPV video file patterns
    FPV_VIDEO_PATTERN = "{clip_id}_fpv.mp4"
    FPV_FALLBACK_PATTERN = "camera_front_wide_120fov.mp4"

    # BEV video file pattern
    BEV_VIDEO_PATTERN = "{clip_id}_bev.mp4"

    # Timestamp file parsing
    TIMESTAMP_LINE_PARTS = 2


@dataclass
class FrameIdentifier:
    """Data class for frame identifier information."""

    video_filename: str
    index_in_video: int  # actual frame index in the video file
    timestamp_micros: int | None = None


@dataclass
class VideoFrameData:
    """Data class for video frame data and metadata."""

    frames: list[np.ndarray]  # List of BGR frames as numpy arrays
    event_start: int | None
    frame_indices: list[int]


@dataclass
class SegmentData:
    """Data class for segment extraction results."""

    hist_frames: list[np.ndarray]
    fut_frames: list[np.ndarray]
    hist_frame_indices: list[int] | None
    fut_frame_indices: list[int] | None


@dataclass
class ProcessedVideoData:
    """Data class for processed video data from FPV or BEV loading."""

    hist_frames: list[np.ndarray]
    fut_frames: list[np.ndarray]
    hist_frame_indices: list[int]
    fut_frame_indices: list[int]
    video_path: str | None


@dataclass
class VideoLoadResult:
    """Data class for the main load method return value."""

    hist_fpv_frames: list[np.ndarray]
    fut_fpv_frames: list[np.ndarray]
    all_fpv_frames: list[np.ndarray]
    all_fpv_frames_info: list[FrameIdentifier]
    hist_bev_frames: list[np.ndarray]
    fut_bev_frames: list[np.ndarray]
    all_bev_frames: list[np.ndarray]
    all_bev_frames_info: list[FrameIdentifier]


class VideoLoader:
    """Video loader for processing FPV and BEV video frames with timestamp mapping.

    This class provides functionality to load video frames from FPV (First Person View)
    and BEV (Bird's Eye View) video files, extract temporal segments, and generate
    frame identifiers with timestamp information.

    The loader supports:
    - Video frame loading with FPS subsampling
    - Temporal segment extraction (historical and future frames)
    - Timestamp mapping and caching
    - Frame identifier generation with metadata
    - Support for both FPV and BEV video streams

    Attributes:
        video_config: Configuration object containing video processing parameters
        data_config: Configuration object containing data directory paths
        _timestamp_mappings: Cache for timestamp data keyed by video path or clip_id
    """

    def __init__(self, video_config: DictConfig | None, data_config: DictConfig) -> None:
        """Initialize the VideoLoader with configuration objects.

        Args:
            video_config: Configuration object containing video processing parameters.
                Expected attributes: fps, hist_length_sec, fut_length_sec, time_interval,
                use_fpv, use_bev, event_start_from_source_fps. Can be None to disable
                video loading functionality.
            data_config: Configuration object containing data directory paths.
                Expected attributes: video_dir for the base video directory path.
        """
        self.video_config = video_config
        self.data_config = data_config
        # Cache for timestamp mappings: {clip_id: [(index, timestamp), ...]}
        self._timestamp_mappings: dict[str, list[tuple[int, int]]] = {}
        if video_config is None:
            logging.info("Video loader is not activated")
            return

    @contextmanager
    def _video_capture(self, video_path: str) -> Generator[Any]:
        """Context manager for video capture to ensure proper resource cleanup.

        Args:
            video_path: Path to the video file to open.

        Yields:
            cv2.VideoCapture: The opened video capture object.

        Raises:
            FileNotFoundError: If the video file doesn't exist or cannot be opened.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cv2_module = _require_cv2()
        cap = cv2_module.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Cannot open video file: {video_path}")

        try:
            yield cap
        finally:
            cap.release()

    def _load_video_frames(self, video_path: str, event_start: int | None) -> VideoFrameData:
        """Load video frames from a video file with optional FPS subsampling.

        Args:
            video_path: Path to the video file to load.
            event_start: Frame index where the event starts in the original video.
                Can be None for full-video mode; used for FPS subsampling calculations.

        Returns:
            Data class containing frames, adjusted event start, and frame indices.

        Raises:
            FileNotFoundError: If the video file doesn't exist or cannot be opened.
            ValueError: If video FPS is not a multiple of config FPS.
        """
        with self._video_capture(video_path) as cap:
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

            # If source video fps differs from configured fps, map source indices to
            # a sampled timeline at configured fps.
            event_start_new = event_start
            frame_offset = 0
            frame_interval = 1
            if frame_rate != self.video_config.fps:
                if frame_rate % self.video_config.fps != 0:
                    raise ValueError(
                        f"Video fps {frame_rate} is not a multiple of config fps "
                        f"{self.video_config.fps}."
                    )
                frame_interval = int(frame_rate / self.video_config.fps)
                if self.video_config.event_start_from_source_fps and event_start is not None:
                    frame_offset = event_start % frame_interval
                    event_start_new = int(event_start / frame_interval)

            # Event mode: only decode the source range needed for the requested
            # history/future window. This avoids loading full videos into memory.
            if event_start is not None and event_start_new is not None:
                hist_length_frame = int(self.video_config.hist_length_sec * self.video_config.fps)
                fut_length_frame = int(self.video_config.fut_length_sec * self.video_config.fps)
                sampled_start = max(0, int(event_start_new) - hist_length_frame)
                sampled_end = int(event_start_new) + fut_length_frame

                source_start = frame_offset + sampled_start * frame_interval
                source_end_exclusive = frame_offset + sampled_end * frame_interval

                if source_start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)

                frames_bgr: list[np.ndarray] = []
                frame_indices: list[int] = []
                source_index = source_start
                while source_index < source_end_exclusive:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if (source_index - frame_offset) % frame_interval == 0:
                        # Keep BGR frames (downstream encoding expects BGR).
                        frames_bgr.append(frame)
                        frame_indices.append(source_index)
                    source_index += 1

                event_start_local = int(event_start_new) - sampled_start
                return VideoFrameData(
                    frames=frames_bgr,
                    event_start=event_start_local,
                    frame_indices=frame_indices,
                )

            frames_bgr = []
            frame_indices = []  # Track original frame indices

            frame_index = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # since we need to do base64 encoding, we should keep BGR all the time
                # frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_bgr.append(frame)
                frame_indices.append(frame_index)
                frame_index += 1

        # Full-video mode: keep previous behavior.
        if frame_interval != 1:
            frames_bgr = frames_bgr[frame_offset::frame_interval]
            frame_indices = frame_indices[frame_offset::frame_interval]

        return VideoFrameData(
            frames=frames_bgr, event_start=event_start_new, frame_indices=frame_indices
        )

    @staticmethod
    def save_video_frames(
        video_path: str,
        frames_bgr: list[np.ndarray],
        frame_rate: float,
        frame_wh: tuple[int, int] | None = None,
        web_compatible: bool = False,
        ffmpeg_bin: str = "ffmpeg",
    ) -> None:
        """Save a list of BGR frames to a video file.

        Args:
            video_path: Path where the output video file will be saved.
            frames_bgr: List of BGR frames as numpy arrays to save.
            frame_rate: Frame rate for the output video.
            frame_wh: Frame width and height as (width, height).
                Defaults to (1920, 1080).

        Note:
            Input frames are expected to be in BGR format. No color conversion is performed.
            If `web_compatible` is True, output is transcoded to H.264/yuv420p.
        """
        # remember that the input frames are in BGR format, no need to do color conversion
        if len(frames_bgr) == 0:
            raise ValueError("No frames provided for video saving.")
        if frame_wh is None:
            frame_wh = (int(frames_bgr[0].shape[1]), int(frames_bgr[0].shape[0]))
        cv2_module = _require_cv2()
        fourcc = cv2_module.VideoWriter_fourcc(*VideoConstants.VIDEO_CODEC)
        out = cv2_module.VideoWriter(video_path, fourcc, frame_rate, frame_wh)
        for frame in frames_bgr:
            out.write(frame)
        out.release()

        # Optional re-encode for browser playback compatibility.
        if web_compatible:
            tmp_path = f"{video_path}.webtmp.mp4"
            command = [
                ffmpeg_bin,
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-profile:v",
                "high",
                "-level",
                "4.1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                tmp_path,
            ]
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0 and os.path.exists(tmp_path):
                    os.replace(tmp_path, video_path)
                else:
                    logging.warning(
                        "ffmpeg transcode failed for %s, keeping original file. stderr: %s",
                        video_path,
                        result.stderr[-1000:],
                    )
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
            except FileNotFoundError:
                logging.warning(
                    "ffmpeg binary '%s' not found; keeping original saved video %s",
                    ffmpeg_bin,
                    video_path,
                )
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def calculate_segment_bounds(
        self, event_start_frame: int, total_frames: int
    ) -> tuple[int, int, int]:
        """Calculate the start and end frame indices for historical and future segments.

        Args:
            event_start_frame: Frame index where the event starts.
            total_frames: Total number of frames available.

        Returns:
            A tuple containing:
                - hist_start_frame: Start frame index for historical segment
                - fut_end_frame: End frame index for future segment
                - frame_interval: Interval between frames for subsampling

        Note:
            Frame segments are calculated based on video_config parameters:
            - hist_length_sec: Length of historical segment in seconds
            - fut_length_sec: Length of future segment in seconds
            - time_interval: Time interval between frames in seconds
        """
        hist_length_frame = int(self.video_config.hist_length_sec * self.video_config.fps)
        fut_length_frame = int(self.video_config.fut_length_sec * self.video_config.fps)
        frame_interval = int(self.video_config.time_interval * self.video_config.fps)

        hist_start_frame = event_start_frame - hist_length_frame
        fut_end_frame = event_start_frame + fut_length_frame

        # Handle corner cases
        hist_start_frame = max(hist_start_frame, 0)
        fut_end_frame = min(fut_end_frame, total_frames)

        return hist_start_frame, fut_end_frame, frame_interval

    def _extract_segments(
        self,
        frames: list[np.ndarray],
        event_start_frame: int | None,
        frame_indices: list[int] | None = None,
    ) -> list | SegmentData:
        """Extract historical and future frame segments around an event.

        Args:
            frames: List of video frames to extract segments from.
            event_start_frame: Frame index where the event starts.
                If None, returns all frames without segmentation.
            frame_indices: List of original frame indices corresponding
                to each frame in the frames list. Used for maintaining frame index mapping.

        Returns:
            SegmentData object containing extracted segments, or all frames if
            event_start_frame is None.

        Note:
            Frame segments are extracted based on video_config parameters:
            - hist_length_sec: Length of historical segment in seconds
            - fut_length_sec: Length of future segment in seconds
            - time_interval: Time interval between frames in seconds
        """
        if event_start_frame is None:
            return frames

        # Calculate segment bounds using the extracted method
        hist_start_frame, fut_end_frame, frame_interval = self.calculate_segment_bounds(
            event_start_frame, len(frames)
        )

        hist_frames = frames[hist_start_frame:event_start_frame:frame_interval]
        fut_frames = frames[event_start_frame:fut_end_frame:frame_interval]

        # Extract corresponding frame indices if provided
        hist_frame_indices = None
        fut_frame_indices = None
        if frame_indices is not None:
            hist_frame_indices = frame_indices[hist_start_frame:event_start_frame:frame_interval]
            fut_frame_indices = frame_indices[event_start_frame:fut_end_frame:frame_interval]

        return SegmentData(
            hist_frames=hist_frames,
            fut_frames=fut_frames,
            hist_frame_indices=hist_frame_indices,
            fut_frame_indices=fut_frame_indices,
        )

    def _get_video_filepath(self, clip_id: str) -> str | None:
        """Get the file path for an FPV video file.

        Args:
            clip_id: Identifier for the video clip.

        Returns:
            Optional[str]: Path to the FPV video file, or None if FPV is disabled.

        Raises:
            FileNotFoundError: If neither the primary FPV file nor the fallback file is found.

        Note:
            First tries to find {clip_id}_fpv.mp4, then falls back to searching for
            camera_front_wide_120fov.mp4 in the clip directory.
        """
        if not self.video_config.use_fpv:
            return None
        fpv_name: str | Path | None = os.path.join(
            self.data_config.video_dir,
            VideoConstants.FPV_VIDEO_PATTERN.format(clip_id=clip_id),
        )
        if not os.path.exists(fpv_name):
            # Flat camera export: <video_dir>/camera/<clip_id>.camera_front_wide_120fov.mp4
            camera_export_path = os.path.join(
                self.data_config.video_dir,
                "camera",
                f"{clip_id}.camera_front_wide_120fov.mp4",
            )
            if os.path.exists(camera_export_path):
                return camera_export_path
            # Fallback 1: <video_dir>/<clip_id>/**/camera_front_wide_120fov.mp4
            fpv_name = next(
                Path(self.data_config.video_dir)
                .joinpath(clip_id)
                .glob(f"**/{VideoConstants.FPV_FALLBACK_PATTERN}"),
                None,
            )
            # Fallback 2: <video_dir>/<clip_id[:4]>/<clip_id>/**/camera_front_wide_120fov.mp4
            if fpv_name is None:
                fpv_name = next(
                    Path(self.data_config.video_dir)
                    .joinpath(clip_id[:4], clip_id)
                    .glob(f"**/{VideoConstants.FPV_FALLBACK_PATTERN}"),
                    None,
                )
            if fpv_name is None:
                raise FileNotFoundError(
                    f"FPV video file not found under: {self.data_config.video_dir} "
                    f"as {VideoConstants.FPV_VIDEO_PATTERN.format(clip_id=clip_id)} "
                    f"or {VideoConstants.FPV_FALLBACK_PATTERN}"
                )
        return str(fpv_name)

    def _get_video_timestamps_filepath(self, clip_id: str) -> str | None:
        """Get the file path for a video's timestamp file.

        Args:
            clip_id: Identifier for the video clip.

        Returns:
            Optional[str]: Path to the timestamp file (.timestamps extension), or
            None if no video file found.

        Note:
            Timestamp files are expected to have the same name as the video file with
            a .timestamps extension.
        """
        video_path = self._get_video_filepath(clip_id)
        if video_path is None:
            return None
        base = Path(video_path).with_suffix("")
        timestamp_candidates = [
            str(base) + VideoConstants.TIMESTAMP_PARQUET_EXTENSION,
            str(base) + VideoConstants.TIMESTAMP_EXTENSION,
            video_path + VideoConstants.TIMESTAMP_PARQUET_EXTENSION,
            video_path + VideoConstants.TIMESTAMP_EXTENSION,
        ]
        for timestamp_path in timestamp_candidates:
            if os.path.exists(timestamp_path):
                return timestamp_path
        logging.warning(
            "No timestamp file found for clip_id=%s. video_path=%s, tried paths=%s",
            clip_id,
            video_path,
            timestamp_candidates,
        )
        return None

    def _load_timestamp_mapping(self, clip_id: str) -> list[tuple[int, int]]:
        """Load timestamp file into memory mapping if not already cached.

        Args:
            clip_id: Identifier for the video clip.

        Returns:
            List of tuples (frame_index, timestamp) parsed from the timestamp file.

        Raises:
            FileNotFoundError: If the timestamp file doesn't exist.
            ValueError: If the timestamp file format is invalid.

        Note:
            Timestamp files are expected to contain lines with two space-separated integers:
            frame_index timestamp_microseconds
        """
        if clip_id in self._timestamp_mappings:
            return self._timestamp_mappings[clip_id]

        timestamp_file = self._get_video_timestamps_filepath(clip_id)
        if timestamp_file is None or not os.path.exists(timestamp_file):
            raise FileNotFoundError(f"Timestamp file {timestamp_file} not found")

        mapping = []
        if timestamp_file.endswith(".parquet"):
            df = pd.read_parquet(timestamp_file)

            if df.index.name is not None:
                df = df.reset_index()

            possible_frame_cols = ["frame_index", "frame_idx", "index_in_video", "frame", "idx"]
            possible_ts_cols = [
                "timestamp_micros",
                "timestamp_us",
                "timestamp",
                "ts",
                "timestamp_microseconds",
            ]

            frame_col = next((c for c in possible_frame_cols if c in df.columns), None)
            ts_col = next((c for c in possible_ts_cols if c in df.columns), None)

            if frame_col is None or ts_col is None:
                raise ValueError(
                    f"Cannot infer frame/timestamp columns from {timestamp_file}, "
                    f"columns={list(df.columns)}"
                )
            mapping = [
                (int(fi), int(ts))
                for fi, ts in df[[frame_col, ts_col]].itertuples(index=False, name=None)
            ]
        else:
            with open(timestamp_file, encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) == VideoConstants.TIMESTAMP_LINE_PARTS:
                        frame_index, ts = list(map(int, parts))
                        mapping.append((frame_index, ts))
                    else:
                        raise ValueError(f"Failed to parse {line} in {timestamp_file}")

        self._timestamp_mappings[clip_id] = mapping
        return mapping

    def get_index_from_timestamp(
        self, clip_id: str, query_ts: int, use_nearest: bool = True
    ) -> tuple[int, int]:
        """Get the frame index and timestamp closest to a given query timestamp.

        Args:
            clip_id: Identifier for the video clip.
            query_ts: Query timestamp in microseconds to find the closest frame for.
            use_nearest: If True, returns the frame with timestamp closest
                to query_ts. If False, returns the frame with timestamp <= query_ts.
                Defaults to True.

        Returns:
            A tuple containing:
                - index_timestamp: Frame index of the closest frame
                - frame_timestamp: Timestamp of the closest frame in microseconds

        Raises:
            FileNotFoundError: If the timestamp file doesn't exist.
            IndexError: If the query timestamp is outside the range of available timestamps.
            ValueError: If the timestamp file format is invalid.

        Note:
            Uses cached timestamp mapping for efficient lookups. The timestamp file must
            exist and contain valid frame index and timestamp pairs.
        """
        # Load timestamp mapping (cached after first load)
        mapping = self._load_timestamp_mapping(clip_id)

        if not mapping:
            raise IndexError(f"No timestamp data found for clip_id: {clip_id}")

        # Check if query timestamp is before first frame
        if query_ts < mapping[0][1]:
            raise IndexError(
                f"Requested timestamp {query_ts} occurs before the first frame "
                f"timestamp {mapping[0][1]}"
            )

        # Check if query timestamp is after last frame
        if query_ts > mapping[-1][1]:
            raise IndexError(
                f"Requested timestamp {query_ts} occurs after the last frame "
                f"timestamp {mapping[-1][1]}"
            )

        # Find the closest frame using binary search approach
        index_below_timestamp = -1
        index_after_timestamp = -1
        frame_timestamp = -1

        for frame_index, ts in mapping:
            if ts <= query_ts:
                index_below_timestamp = frame_index
                frame_timestamp = ts
            else:
                index_after_timestamp = frame_index
                if use_nearest and (ts - query_ts) < (query_ts - frame_timestamp):
                    return index_after_timestamp, ts
                else:
                    return index_below_timestamp, frame_timestamp

        # If we reach here, query_ts is exactly at or after the last frame
        return index_below_timestamp, frame_timestamp

    def _generate_frame_identifiers(
        self,
        clip_id: str,
        hist_frame_indices: list[int],
        fut_frame_indices: list[int],
        video_path: str | None,
    ) -> list[FrameIdentifier]:
        """Generate frame identifiers for frames containing video filename, index, and timestamp.

        Args:
            clip_id: The clip identifier.
            hist_frame_indices: List of historical frame indices.
            fut_frame_indices: List of future frame indices.
            video_path: Path to the video file.

        Returns:
            List of FrameIdentifier objects containing video filename, index, and timestamp.

        Note:
            Attempts to load timestamp mapping for the video. If timestamp file doesn't exist
            or cannot be loaded, timestamp_micros will be None for all frames.
        """
        if not hist_frame_indices and not fut_frame_indices:
            return []

        frame_identifiers = []
        video_filename = os.path.basename(video_path) if video_path else "unknown"

        # Load timestamp mapping for the video
        try:
            mapping = self._load_timestamp_mapping(clip_id)
        except (FileNotFoundError, IndexError):
            mapping = None

        # Preserve the original order: historical frames first, future frames second.
        for frame_index in hist_frame_indices + fut_frame_indices:
            timestamp = None
            if mapping:
                # Find the closest timestamp for this frame index.
                for frame_idx, ts in mapping:
                    if frame_idx >= frame_index:
                        timestamp = ts
                        break

            frame_identifiers.append(
                FrameIdentifier(
                    video_filename=video_filename,
                    index_in_video=frame_index,
                    timestamp_micros=timestamp,
                )
            )

        return frame_identifiers

    def _to_processed_video_data(
        self,
        segment_data: list[np.ndarray] | SegmentData,
        video_path: str | None,
    ) -> ProcessedVideoData:
        """Normalize `_extract_segments` output into ProcessedVideoData."""
        if isinstance(segment_data, list):
            return ProcessedVideoData(
                hist_frames=segment_data,
                fut_frames=[],
                hist_frame_indices=[],
                fut_frame_indices=[],
                video_path=video_path,
            )
        return ProcessedVideoData(
            hist_frames=segment_data.hist_frames,
            fut_frames=segment_data.fut_frames,
            hist_frame_indices=segment_data.hist_frame_indices or [],
            fut_frame_indices=segment_data.fut_frame_indices or [],
            video_path=video_path,
        )

    def _load_fpv_frames(self, clip_id: str, event_start: int | None) -> ProcessedVideoData:
        """Load and process FPV video frames.

        Args:
            clip_id: Identifier for the video clip.
            event_start: Frame index where the event starts.

        Returns:
            Data class containing processed FPV frame data.
        """
        fpv_name = self._get_video_filepath(clip_id)
        if not fpv_name:
            return ProcessedVideoData([], [], [], [], None)

        video_data = self._load_video_frames(fpv_name, event_start)
        segment_data = self._extract_segments(
            video_data.frames, video_data.event_start, video_data.frame_indices
        )

        return self._to_processed_video_data(segment_data, fpv_name)

    def _load_bev_frames(self, clip_id: str, event_start: int | None) -> ProcessedVideoData:
        """Load and process BEV video frames.

        Args:
            clip_id: Identifier for the video clip.
            event_start: Frame index where the event starts.

        Returns:
            Data class containing processed BEV frame data.
        """
        if not self.video_config.use_bev:
            return ProcessedVideoData([], [], [], [], None)

        bev_name = os.path.join(
            self.data_config.video_dir,
            VideoConstants.BEV_VIDEO_PATTERN.format(clip_id=clip_id),
        )
        if not os.path.exists(bev_name):
            raise FileNotFoundError(f"BEV video file not found: {bev_name}")

        video_data = self._load_video_frames(bev_name, event_start)
        segment_data = self._extract_segments(
            video_data.frames, video_data.event_start, video_data.frame_indices
        )

        return self._to_processed_video_data(segment_data, bev_name)

    def _combine_frame_data(
        self, processed_data: ProcessedVideoData, clip_id: str
    ) -> tuple[list[np.ndarray], list[FrameIdentifier]]:
        """Combine frame data and generate identifiers.

        Args:
            processed_data: Processed video data containing frames and indices.
            clip_id: Clip identifier.

        Returns:
            Combined frames and frame identifiers.
        """
        all_frames = processed_data.hist_frames + processed_data.fut_frames
        frame_identifiers = self._generate_frame_identifiers(
            clip_id,
            processed_data.hist_frame_indices,
            processed_data.fut_frame_indices,
            processed_data.video_path,
        )
        return all_frames, frame_identifiers

    def _get_segment_video_output_root(self) -> str | None:
        """Resolve output root for optional segment-video exports.

        Returns:
            Output directory path when enabled; otherwise ``None``.
        """
        if not bool(getattr(self.video_config, "save_segment_videos", False)):
            return None
        configured_dir = getattr(self.video_config, "segment_video_output_dir", None)
        if configured_dir is not None and str(configured_dir).strip() != "":
            return str(configured_dir)
        return os.path.join(self.data_config.video_dir, "segment_videos")

    def _extract_source_fps_segment(
        self, video_path: str, start_idx: int, end_idx: int
    ) -> tuple[list[np.ndarray], float]:
        """Extract contiguous raw frames from source video in [start_idx, end_idx]."""
        if start_idx < 0 or end_idx < start_idx:
            return [], float(self.video_config.fps)
        with self._video_capture(video_path) as cap:
            source_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(self.video_config.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            frames_bgr: list[np.ndarray] = []
            for _ in range(end_idx - start_idx + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                frames_bgr.append(frame)
        return frames_bgr, source_fps

    def _save_segment_video_if_needed(
        self,
        clip_id: str,
        all_frames: list[np.ndarray],
        event_start: int | None,
        event_start_timestamp: int | None,
        video_path: str | None = None,
        sampled_frame_indices: list[int] | None = None,
        sampled_frame_identifiers: list[FrameIdentifier] | None = None,
    ) -> str | None:
        """Optionally persist extracted segment video to disk for later visualization."""
        output_root = self._get_segment_video_output_root()
        if output_root is None or len(all_frames) == 0:
            return None

        if event_start_timestamp is not None:
            segment_tag = str(event_start_timestamp)
        elif event_start is not None:
            segment_tag = f"frame_{int(event_start):06d}"
        else:
            segment_tag = "full"

        out_dir = os.path.join(output_root, clip_id)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{clip_id}_{segment_tag}.mp4")

        if bool(getattr(self.video_config, "segment_video_skip_existing", True)) and os.path.exists(
            out_path
        ):
            return out_path

        frames_to_save = all_frames
        frame_rate_to_save = float(self.video_config.fps)
        if (
            bool(getattr(self.video_config, "save_segment_videos_source_fps", False))
            and video_path is not None
            and sampled_frame_indices is not None
            and len(sampled_frame_indices) > 0
        ):
            start_idx = int(min(sampled_frame_indices))
            end_idx = int(max(sampled_frame_indices))
            source_frames, source_fps = self._extract_source_fps_segment(
                video_path, start_idx, end_idx
            )
            if len(source_frames) > 0:
                frames_to_save = source_frames
                frame_rate_to_save = source_fps

        try:
            self.save_video_frames(
                out_path,
                frames_to_save,
                frame_rate=frame_rate_to_save,
                web_compatible=bool(
                    getattr(self.video_config, "segment_video_web_compatible", True)
                ),
                ffmpeg_bin=str(getattr(self.video_config, "segment_video_ffmpeg_bin", "ffmpeg")),
            )
            if (
                bool(getattr(self.video_config, "save_segment_video_metadata", True))
                and sampled_frame_identifiers is not None
            ):
                metadata_path = out_path.replace(".mp4", ".sampled_frames.json")
                metadata = {
                    "clip_id": clip_id,
                    "event_start_timestamp": event_start_timestamp,
                    "event_start_frame": event_start,
                    "saved_video_fps": frame_rate_to_save,
                    "saved_video_mode": (
                        "source_fps"
                        if bool(
                            getattr(
                                self.video_config,
                                "save_segment_videos_source_fps",
                                False,
                            )
                        )
                        else "sampled_fps"
                    ),
                    "sampled_frames": [
                        {
                            "video_filename": fi.video_filename,
                            "index_in_video": int(fi.index_in_video),
                            "timestamp_micros": (
                                int(fi.timestamp_micros)
                                if fi.timestamp_micros is not None
                                else None
                            ),
                        }
                        for fi in sampled_frame_identifiers
                    ],
                }
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
            return out_path
        except Exception as exc:
            logging.warning(f"Failed to save segment video to {out_path}: {exc}")
            return None

    def load(
        self,
        clip_id: str,
        event_start: int | None = None,
        event_start_timestamp: int | None = None,
    ) -> VideoLoadResult:
        """Load video frames and generate frame identifiers for a given clip.

        This is the main public method that orchestrates the entire video loading process,
        including frame loading, segment extraction, and frame identifier generation.

        Args:
            clip_id: Identifier for the video clip to load.
            event_start: Frame index where the event starts in the original video.
                If None, uses the entire video without temporal segmentation.

        Returns:
            VideoLoadResult: Data class containing loaded video data with FPV and
            BEV frames and identifiers.

        Note:
            Frame identifiers contain:
            - video_filename: Name of the video file
            - index_in_video: Original frame index in the video
            - timestamp_micros: Timestamp in microseconds

            If video_config is None, returns an empty VideoLoadResult.
            BEV frames are only loaded if video_config.use_bev is True.
        """
        if self.video_config is None:
            return VideoLoadResult([], [], [], [], [], [], [], [])

        # Load FPV frames
        fpv_data = self._load_fpv_frames(clip_id, event_start)

        # Load BEV frames
        bev_data = self._load_bev_frames(clip_id, event_start)

        # Combine FPV frame data
        all_fpv_frames, fpv_frame_identifiers = self._combine_frame_data(fpv_data, clip_id)
        sampled_fpv_indices = (fpv_data.hist_frame_indices or []) + (
            fpv_data.fut_frame_indices or []
        )

        # Optional: persist the extracted FPV segment for downstream visualization workflows.
        self._save_segment_video_if_needed(
            clip_id=clip_id,
            all_frames=all_fpv_frames,
            event_start=event_start,
            event_start_timestamp=event_start_timestamp,
            video_path=fpv_data.video_path,
            sampled_frame_indices=sampled_fpv_indices,
            sampled_frame_identifiers=fpv_frame_identifiers,
        )

        # Combine BEV frame data
        all_bev_frames, bev_frame_identifiers = self._combine_frame_data(bev_data, clip_id)

        return VideoLoadResult(
            hist_fpv_frames=fpv_data.hist_frames,
            fut_fpv_frames=fpv_data.fut_frames,
            all_fpv_frames=all_fpv_frames,
            all_fpv_frames_info=fpv_frame_identifiers,
            hist_bev_frames=bev_data.hist_frames,
            fut_bev_frames=bev_data.fut_frames,
            all_bev_frames=all_bev_frames,
            all_bev_frames_info=bev_frame_identifiers,
        )
