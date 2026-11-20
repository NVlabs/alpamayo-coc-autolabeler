# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve standalone meta-action formatting and sampling edge cases."""

from types import SimpleNamespace

import pytest

from coc_labeling.data_loader.meta_action_loader import MetaActionLoader


@pytest.mark.parametrize("config", [{}, {"add_ts_prefix": False}])
def test_meta_action_text_without_timestamps_does_not_require_fps(config):
    """Untimed text needs only the action fields, as in the standalone workflow."""
    loader = MetaActionLoader(SimpleNamespace(**config), None)
    actions = [{"Longitudinal": "Stop", "Lateral": "Not Available", "Lane": "Not Available"}]

    assert loader.process_meta_action(actions, []) == (
        "\nlongitudinal: Stop, lateral: Not Available, lane: Not Available"
    )


def test_empty_timestamped_meta_actions_do_not_require_fps():
    """An empty history window still renders its timestamp header without reading FPS."""
    loader = MetaActionLoader(SimpleNamespace(add_ts_prefix=True), None)

    assert loader.process_meta_action([], []) == (
        "\nMeta-actions by sampled timestamp (relative to event, unit: seconds):\n"
    )


def test_sampling_rejects_invalid_interval_before_converting_anchor():
    """Keep the standalone sampling error when both settings are invalid."""
    loader = MetaActionLoader(
        SimpleNamespace(hist_length_sec=2, fut_length_sec=6, time_interval=0, fps=10), None
    )

    with pytest.raises(
        ValueError, match=r"meta_action_config.time_interval \* fps must be > 0 for sampling"
    ):
        loader.extract_segments([{}], "invalid")
