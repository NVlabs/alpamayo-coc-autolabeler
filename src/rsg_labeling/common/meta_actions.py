"""Contract for compact production action context, never scene-cause evidence."""

from __future__ import annotations

SCHEMA = "alparsg_production_meta_actions_v1"
GROUPS = ("longitudinal", "lateral", "lane")


def validate_meta_actions(entry: dict) -> None:
    """Reject invalid clocks, unordered action axes and fabricated availability."""
    if type(entry.get("available")) is not bool or not isinstance(entry.get("source"), dict):
        raise ValueError("meta-action availability and pinned source are required")
    clock = entry["clock"]
    lo, hi = clock["origin_us"], clock["end_us"]
    if type(lo) is not int or type(hi) is not int or lo >= hi:
        raise ValueError("invalid meta-action physical clock")
    actions = entry["actions"]
    if bool(actions) != entry["available"]:
        raise ValueError("meta-action availability must describe decoded actions")
    previous = {}
    for action in actions:
        if set(action) != {"action", "group", "start_us", "end_us"}:
            raise ValueError("unexpected meta-action fields")
        start, end, group = action["start_us"], action["end_us"], action["group"]
        if (
            group not in GROUPS
            or not isinstance(action["action"], str)
            or not action["action"]
            or type(start) is not int
            or type(end) is not int
            or not lo <= start < end <= hi
        ):
            raise ValueError("invalid half-open production action span")
        if start < previous.get(group, lo):
            raise ValueError("unordered production action axis")
        previous[group] = start


def coverage_diagnostics(entry: dict) -> dict:
    """Preserve gaps and overlaps per axis without moving any production endpoints."""
    validate_meta_actions(entry)
    gaps, overlaps = {}, {}
    for group in GROUPS:
        cursor = entry["clock"]["origin_us"]
        missing = []
        ambiguous = []
        for action in entry["actions"]:
            if action["group"] != group:
                continue
            if cursor < action["start_us"]:
                missing.append([cursor, action["start_us"]])
            if action["start_us"] < cursor:
                ambiguous.append([action["start_us"], min(cursor, action["end_us"])])
            cursor = max(cursor, action["end_us"])
        if cursor < entry["clock"]["end_us"]:
            missing.append([cursor, entry["clock"]["end_us"]])
        gaps[group] = missing
        overlaps[group] = ambiguous
    return {"coverage_gaps_us": gaps, "coverage_overlaps_us": overlaps}
