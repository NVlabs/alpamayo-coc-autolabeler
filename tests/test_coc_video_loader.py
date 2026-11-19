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

from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coc_labeling.data_loader.video_loader import (  # noqa: E402
    FrameIdentifier,
    ProcessedVideoData,
    VideoLoader,
)


def test_combine_frame_data_keeps_future_frames_without_history() -> None:
    loader = object.__new__(VideoLoader)

    def fake_identifiers(
        self: VideoLoader,
        clip_id: str,
        hist_frame_indices: list[int],
        fut_frame_indices: list[int],
        video_path: str | None,
    ) -> list[FrameIdentifier]:
        return [
            FrameIdentifier(
                video_filename=video_path or "unknown",
                index_in_video=frame_index,
            )
            for frame_index in hist_frame_indices + fut_frame_indices
        ]

    loader._generate_frame_identifiers = MethodType(fake_identifiers, loader)
    fut_frames = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.ones((1, 1, 3), dtype=np.uint8),
    ]
    processed = ProcessedVideoData(
        hist_frames=[],
        fut_frames=fut_frames,
        hist_frame_indices=[],
        fut_frame_indices=[10, 12],
        video_path="clip.mp4",
    )

    all_frames, frame_identifiers = loader._combine_frame_data(processed, "clip-123")

    assert len(all_frames) == 2
    assert all_frames[0] is fut_frames[0]
    assert all_frames[1] is fut_frames[1]
    assert [item.index_in_video for item in frame_identifiers] == [10, 12]
