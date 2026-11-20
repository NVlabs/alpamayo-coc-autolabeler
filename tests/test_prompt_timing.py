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
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from coc_labeling.prompts.vlm import coc  # noqa: E402


def test_prompt_timing_uses_resolved_video_loader_config() -> None:
    """Render timing text from the live loader config instead of packaged defaults."""
    data_loader_config = SimpleNamespace(
        video=SimpleNamespace(
            hist_length_sec=3,
            fut_length_sec=4,
            time_interval=0.25,
        )
    )

    images_prompt = coc.render_images_prompt(data_loader_config)
    output_prompts = coc.render_output_prompts(data_loader_config)

    assert "at **4 Hz**" in images_prompt
    assert "7 seconds, with 3 seconds of historical window and 4 seconds" in images_prompt
    assert "The video length is 7 seconds, with 3 seconds" in output_prompts
    assert "at the 3 to 4 seconds mark" in output_prompts
    assert '"ego_behavior_schema"' in output_prompts
    assert '"effect_on_ego_behavior"' in output_prompts
    assert '{\n    "effect_on_ego_behavior":' not in output_prompts
    assert "The video length is 8 seconds, with 2 seconds" not in output_prompts
