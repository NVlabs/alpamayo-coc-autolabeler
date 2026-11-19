"""Shared route-maneuver vocabulary; route intent does not establish an executed action."""

from __future__ import annotations

import math


def maneuver_family(maneuver: str) -> str | None:
    """Normalize the route vocabulary used by exact-anchor and full-clip consumers."""
    if maneuver.startswith(("TURN_", "ROUNDABOUT_")):
        return "turn"
    if maneuver.startswith(("RAMP_", "FORK_")):
        return "split"
    if maneuver.startswith("MERGE_"):
        return "merge"
    return "keep_lane" if maneuver == "STRAIGHT" else None


def maneuver_text(maneuver: str, family: str, direction: str) -> str:
    """Render only intended route direction, omitting street names and inferred causes."""
    if family == "turn":
        modifier = "slight " if "SLIGHT" in maneuver else "sharp " if "SHARP" in maneuver else ""
        return (
            f"follow the roundabout toward the {direction} exit"
            if maneuver.startswith("ROUNDABOUT_")
            else f"{modifier}turn {direction}"
        )
    if family == "split":
        return f"take the {direction} {'ramp' if maneuver.startswith('RAMP_') else 'fork'}"
    if family == "merge":
        return f"merge {direction}"
    return "continue straight"


def validate_navigation(entry: dict) -> None:
    """Validate a bounded source timeline without assuming missing metadata means straight."""
    if type(entry["available"]) is not bool or not entry["source"]:
        raise ValueError("navigation availability and source must be explicit")
    if not entry["available"]:
        if entry["samples"] or not entry["reason"]:
            raise ValueError("unavailable navigation needs a reason and no samples")
        return
    previous = -1
    if not entry["samples"]:
        raise ValueError("available navigation requires samples")
    for sample in entry["samples"]:
        timestamp = sample["timestamp_us"]
        if type(timestamp) is not int or timestamp <= previous:
            raise ValueError("navigation timestamps must be strictly increasing")
        if (
            not isinstance(sample["maneuver"], str)
            or not sample["maneuver"]
            or not math.isfinite(sample["remaining_m"])
            or sample["remaining_m"] < 0
        ):
            raise ValueError("invalid navigation maneuver or remaining distance")
        if set(sample) != {"timestamp_us", "maneuver", "remaining_m", "instruction_status"}:
            raise ValueError("navigation samples may not carry future-derived ETA or labels")
        previous = timestamp
