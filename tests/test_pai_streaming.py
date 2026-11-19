# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PAI cache/action adaptation preserves physical timing and missing source coverage."""

import pytest

from rsg_labeling.preparation.pai import _action_entry


def test_precomputed_actions_use_ego_origin_and_preserve_gaps_and_overlaps(tmp_path):
    """The first camera capture is not action tick zero, and missing lane data stays absent."""
    path = tmp_path / "clip.txt"
    path.write_text(
        "MaintainSpeed - Agent:<ego>, Start:0, End:7\n"
        "GentleAcceleration - Agent:<ego>, Start:6, End:10\n"
        "SteerLeft - Agent:<ego>, Start:0, End:10\n"
    )
    entry = _action_entry(path, -100_000, {"origin_us": 0, "end_us": 1_000_000}, {})
    longitudinal = [a for a in entry["actions"] if a["group"] == "longitudinal"]
    assert [(a["action"], a["start_us"], a["end_us"]) for a in longitudinal] == [
        ("maintain_speed", 0, 600_000),
        ("gentle_acceleration", 500_000, 900_000),
    ]
    assert entry["coverage_gaps_us"]["longitudinal"] == [[900_000, 1_000_000]]
    assert entry["coverage_overlaps_us"]["longitudinal"] == [[500_000, 600_000]]
    assert entry["coverage_gaps_us"]["lane"] == [[0, 1_000_000]]
    assert not any(a["group"] == "lane" for a in entry["actions"])
    assert entry["source"]["kind"] == "pai_rule_generated_meta_actions"

    from rsg_labeling.tasks.streaming_rsg.generation.pipeline import _round_messages
    from rsg_labeling.tasks.streaming_rsg.schema.contracts import ClipClock

    snapshot = {
        "graph_hash": "fixture",
        "facts": {},
        "edges": [],
        "action_context": {"record_id": "env:production_meta_actions", **entry},
    }
    _, prompt, _ = _round_messages(
        "clip",
        ClipClock(origin_us=0, end_us=1_000_000),
        snapshot,
        0,
        "Return intervals",
        None,
        None,
        None,
        None,
    )
    assert prompt["production_action_context"]["actions"] == entry["actions"]
    assert prompt["production_action_context"]["reason"]


@pytest.mark.parametrize("span", ["Start:3, End:2", "Start:no, End:10"])
def test_malformed_precomputed_actions_fail_before_generation(tmp_path, span):
    """Bad source spans are not silently repaired into apparently valid context."""
    path = tmp_path / "clip.txt"
    path.write_text("MaintainSpeed - Agent:<ego>, " + span + "\n")
    with pytest.raises(ValueError, match="ego action"):
        _action_entry(path, 0, {"origin_us": 0, "end_us": 1_000_000}, {})
