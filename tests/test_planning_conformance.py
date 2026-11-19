# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the pure planner to recorded upstream native adapter requests."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from coc_labeling.api import EGO_META_ACTIONS, prepare_coc_events

FIXTURE = json.loads((Path(__file__).parent / "fixtures/planning_conformance.json").read_text())


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda case: case["name"] + "-" + case["dtype"]
)
def test_planning_matches_native_upstream_requests(case):
    """Preserve spans, context exclusions, motion wording and complete ordered messages."""
    arrays = [np.array(value, dtype=case["dtype"]) for value in case["motion"]]
    plan = prepare_coc_events(
        *arrays,
        selected_actions=None,
        min_duration_ticks=1,
        min_start_tick=0,
        max_start_tick=len(arrays[0]) - 1,
    )
    spans = [asdict(span) for span in plan.longitudinal_spans + plan.lateral_spans]
    assert (
        sorted(spans, key=lambda s: (s["start_tick"], s["end_tick"], s["action"])) == case["spans"]
    )
    expected_exclusions = [
        s for s in case["spans"] if s["start_tick"] < 20 or s["start_tick"] + 60 >= plan.num_ticks
    ]
    assert [asdict(span) for span in plan.excluded_candidates] == expected_exclusions
    assert len(plan.events) == len(case["events"])
    for request, expected in zip(plan.events, case["events"]):
        actual = {name: getattr(request, name) for name in expected}
        assert actual == expected
        messages = json.loads(json.dumps(FIXTURE["static_messages"]))
        messages[4]["content"] = expected["meta_action_text"]
        messages[6]["content"] = expected["ego_text"]
        assert list(request.messages) == messages
        assert request.action in EGO_META_ACTIONS
