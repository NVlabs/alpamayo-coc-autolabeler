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

import json
import logging
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coc_labeling import keyframe_auto_select  # noqa: E402


def test_default_selection_keeps_brief_decisive_maneuvers(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Keep short OOD-style strong-deceleration spans in sample-eval output."""
    meta_action_dir = tmp_path / "meta_action" / "final_outputs"
    meta_action_dir.mkdir(parents=True)
    (meta_action_dir / "clip_a.txt").write_text(
        "\n".join(
            [
                "StrongDeceleration - Agent:<ego>, Start:30, End:50",
                "StrongDeceleration - Agent:<ego>, Start:60, End:69",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "keyframes"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "keyframe_auto_select",
            "--meta_action_dir",
            str(meta_action_dir),
            "--actions",
            "strong_deceleration",
            "--output_dir",
            str(output_dir),
        ],
    )
    with caplog.at_level(logging.WARNING):
        keyframe_auto_select.main()

    sampled = json.loads((output_dir / "segments_relative_timestamp_sampled.json").read_text())
    assert [
        {
            "meta_action": "strong_deceleration",
            "clip_id": "clip_a",
            "event_start_frame": 30,
            "event_end_frame": 50,
            "duration": 20,
        }
    ] == sampled["strong_deceleration"]
    assert "Found only" not in caplog.text


def test_custom_action_under_target_reports_name_or_target_issue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Explain non-canonical under-target actions without reserve-all jargon."""
    records = [
        keyframe_auto_select.ActionRecord(
            meta_action="custom_action",
            clip_id="clip_a",
            event_start_frame=20,
            event_end_frame=40,
            duration=20,
        )
    ]

    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            keyframe_auto_select.InsufficientActionSamplesError,
            match="not a built-in meta action",
        ):
            keyframe_auto_select.balance_action_records({"custom_action": records}, target_count=2)

    assert "Check the action name" in caplog.text
    assert "lower --target_count" in caplog.text
    assert "reserve-all" not in caplog.text


def test_custom_action_at_target_keeps_original_custom_action_behavior() -> None:
    """Keep existing support for custom actions that meet the requested target."""
    records = [
        keyframe_auto_select.ActionRecord(
            meta_action="custom_action",
            clip_id="clip_a",
            event_start_frame=20,
            event_end_frame=40,
            duration=20,
        )
    ]

    balanced = keyframe_auto_select.balance_action_records(
        {"custom_action": records},
        target_count=1,
    )

    assert balanced == {"custom_action": records}
