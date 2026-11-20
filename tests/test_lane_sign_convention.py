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

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from meta_action.processor.lane.segmentation import (  # noqa: E402
    _offset_delta_to_left_positive_signal,
    _realign_zero_evidence_lane_change_runs,
)


class TestLaneSignConvention(unittest.TestCase):
    """Regression tests for lane-change sign and direction normalization logic."""

    def test_offset_signal_normalization(self) -> None:
        """Offset normalization should map left-positive convention consistently."""
        self.assertIsNone(_offset_delta_to_left_positive_signal(None))
        self.assertIsNone(_offset_delta_to_left_positive_signal(float("nan")))
        self.assertAlmostEqual(_offset_delta_to_left_positive_signal(-0.4), 0.4)
        self.assertAlmostEqual(_offset_delta_to_left_positive_signal(0.25), -0.25)

    def test_short_zero_evidence_run_uses_left_for_decreasing_offset(self) -> None:
        """Short zero-evidence runs should infer left change from decreasing offset."""
        decisions = ["left_lane_change"] * 6
        lane_ids = ["lane_a", "lane_a", "lane_b", "lane_b", "lane_b", "lane_b"]
        lane_offsets = [0.55, 0.48, 0.40, 0.31, 0.22, 0.12]
        left_hits = [0] * len(decisions)
        right_hits = [0] * len(decisions)

        out, reasons = _realign_zero_evidence_lane_change_runs(
            decisions=decisions,
            lane_ids=lane_ids,
            lane_offsets=lane_offsets,
            left_hits=left_hits,
            right_hits=right_hits,
            world_xyzh=None,
        )

        self.assertTrue(all(label == "left_lane_change" for label in out))
        self.assertTrue(
            all(reason == "short_zero_evidence_direction_from_displacement" for reason in reasons)
        )

    def test_ego_displacement_has_priority_over_conflicting_offset_signal(self) -> None:
        """Ego displacement should override conflicting offset-only direction cues."""
        decisions = ["right_lane_change"] * 6
        lane_ids = ["lane_a", "lane_a", "lane_b", "lane_b", "lane_b", "lane_b"]
        # Increasing offset would imply right after normalization fallback.
        lane_offsets = [0.10, 0.18, 0.26, 0.34, 0.42, 0.50]
        left_hits = [0] * len(decisions)
        right_hits = [0] * len(decisions)

        # Ego-frame displacement: heading=0 => +Y is "left".
        world_xyzh = np.zeros((len(decisions), 4), dtype=float)
        world_xyzh[:, 1] = np.linspace(0.0, 0.5, len(decisions))
        world_xyzh[:, 3] = 0.0

        out, reasons = _realign_zero_evidence_lane_change_runs(
            decisions=decisions,
            lane_ids=lane_ids,
            lane_offsets=lane_offsets,
            left_hits=left_hits,
            right_hits=right_hits,
            world_xyzh=world_xyzh,
        )

        self.assertTrue(all(label == "left_lane_change" for label in out))
        self.assertTrue(
            all(reason == "short_zero_evidence_direction_from_displacement" for reason in reasons)
        )

    def test_medium_zero_evidence_run_uses_right_for_increasing_offset(self) -> None:
        """Medium zero-evidence runs should infer right change from increasing offset."""
        run_len = 30  # medium run: > SHORT_MAX(24) and < LONG_SUPPRESS(60)
        decisions = ["right_lane_change"] * run_len
        lane_ids = ["lane_a"] * 15 + ["lane_b"] * 15
        lane_offsets = [0.10 + 0.02 * i for i in range(run_len)]  # net increase > 0.35
        left_hits = [0] * run_len
        right_hits = [0] * run_len

        out, reasons = _realign_zero_evidence_lane_change_runs(
            decisions=decisions,
            lane_ids=lane_ids,
            lane_offsets=lane_offsets,
            left_hits=left_hits,
            right_hits=right_hits,
            world_xyzh=None,
        )

        self.assertTrue(all(label == "right_lane_change" for label in out))
        self.assertTrue(
            all(reason == "medium_zero_evidence_direction_from_displacement" for reason in reasons)
        )


if __name__ == "__main__":
    unittest.main()
