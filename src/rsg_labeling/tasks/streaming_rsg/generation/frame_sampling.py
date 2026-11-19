"""Select real camera observations without thinning ego, route or production metadata."""

import base64
import hashlib
import math
from bisect import bisect_left
from pathlib import Path


def validate_camera_brackets(timestamps) -> None:
    """Require physical captures that can bracket the independent action evidence."""
    if len(timestamps) < 2:
        raise ValueError("action sampling requires at least two camera captures")
    if any(right - left > 600_000 for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("source camera gap exceeds physical bracket contract")


def physical_focus(context, observation, clock, inspection_windows_us=()):
    """Retain physical support and dense transition/uncertainty/inspection windows.

    These are camera-selection windows, not semantic interval proposals. A continuing
    physical action can contain permission/actor changes; full-clip sparse images, graph
    evidence and the existing semantic inspection requests remain available for those.
    Source-fact coverage endpoints do not independently create dense windows or events.
    """
    records = {row["id"]: row for row in context["records"]}
    phases = observation["phases"]
    starts = [
        clock.origin_us if phase["after_id"] is None else records[phase["after_id"]]["timestamp_us"]
        for phase in phases
    ]
    windows = [tuple(window) for window in inspection_windows_us]
    required = set()
    for phase, start, end in zip(phases, starts, [*starts[1:], clock.end_us], strict=True):
        required.update(phase["support_ids"])
        required.update(key for key in (phase["before_id"], phase["after_id"]) if key is not None)
        if phase["after_id"] is not None:
            windows.append(
                (max(clock.origin_us, start - 1_000_000), min(clock.end_us, start + 1_000_000))
            )
        if phase["confidence"] == "uncertain":
            windows.append((start, end))
    if required - records.keys():
        raise ValueError("physical support missing from delivered sensor records")
    return sorted(set(windows)), sorted(required)


def sample_camera_context(context, clock, target_hz, dense_windows_us=(), required_ids=()):
    """Return matched image/record subsets and a compact, timestamp-exact selection receipt.

    Select the nearest available camera sample to each target time, preserving both clip
    endpoints and any explicitly required support. Dense windows contain all existing
    captures, not newly fetched images. Fill gaps above the physical parser's0.6s bracket
    limit with existing samples. Actual rates can exceed the nominal target on a coarse or
    jittered source grid; the receipt records this explicitly. No timestamp is synthesized.
    """
    if target_hz not in (2, 3, 5):
        raise ValueError("supported camera targets are2,3,5Hz")
    cameras = [row for row in context["records"] if row["stream"].startswith("camera/")]
    stamps = [row["timestamp_us"] for row in cameras]
    ids = [row["id"] for row in cameras]
    if len(stamps) < 2 or any(a >= b for a, b in zip(stamps, stamps[1:])):
        raise ValueError("camera timestamp order must be strictly increasing")
    if len({row["id"] for row in context["records"]}) != len(context["records"]):
        raise ValueError("sensor record IDs must be unique")
    if not clock.origin_us <= stamps[0] < stamps[-1] < clock.end_us:
        raise ValueError("camera observations lie outside the physical clip clock")
    images = context["images"]
    if len(images) != 2 * len(cameras) or any(
        text["type"] != "text" or image["type"] != "image_url"
        for text, image in zip(images[::2], images[1::2], strict=True)
    ):
        raise ValueError("sampling requires paired timed text and camera images")
    known = {row["id"] for row in context["records"]}
    if set(required_ids) - known:
        raise ValueError("required support must be a delivered observation")
    windows = [tuple(window) for window in dense_windows_us]
    if any(
        len(window) != 2 or not clock.origin_us <= window[0] <= window[1] <= clock.end_us
        for window in windows
    ):
        raise ValueError("dense windows must lie inside the physical clip clock")
    selected = {0, len(stamps) - 1}
    if target_hz == 5:
        selected.update(range(len(stamps)))
    else:
        count = math.ceil((clock.end_us - clock.origin_us) * target_hz / 1_000_000)
        for index in range(count):
            target = clock.origin_us + round(index * 1_000_000 / target_hz)
            after = bisect_left(stamps, target)
            candidates = [i for i in (after - 1, after) if 0 <= i < len(stamps)]
            selected.add(min(candidates, key=lambda i: (abs(stamps[i] - target), stamps[i])))
    selected.update(
        index
        for index, row in enumerate(cameras)
        if row["id"] in required_ids or any(lo <= row["timestamp_us"] <= hi for lo, hi in windows)
    )
    gap_fill_ids = []
    for left, right in zip(sorted(selected), sorted(selected)[1:]):
        while stamps[right] - stamps[left] > 600_000:
            middle = bisect_left(stamps, stamps[left] + 600_001) - 1
            if middle <= left:
                raise ValueError("source camera gap exceeds physical bracket contract")
            selected.add(middle)
            gap_fill_ids.append(ids[middle])
            left = middle
    indices = sorted(selected)
    selected_ids = [ids[index] for index in indices]
    selected_set = set(selected_ids)
    filtered = {
        **context,
        "records": [
            row
            for row in context["records"]
            if not row["stream"].startswith("camera/") or row["id"] in selected_set
        ],
        "images": [part for index in indices for part in images[2 * index : 2 * index + 2]],
    }
    gaps = [stamps[b] - stamps[a] for a, b in zip(indices, indices[1:])]
    return filtered, {
        "target_hz": target_hz,
        "source_camera_count": len(cameras),
        "selected_ids": selected_ids,
        "selected_timestamps_us": [stamps[i] for i in indices],
        "selected_camera_count": len(indices),
        "max_gap_us": max(gaps),
        "observed_rate_hz": (len(indices) - 1) * 1_000_000 / (stamps[-1] - stamps[0]),
        "gap_fill_ids": gap_fill_ids,
        "dense_windows_us": windows,
        "required_ids": sorted(set(required_ids)),
        "noncamera_records_unchanged": True,
    }


def encode_timed_records(source_records, clock):
    """Encode exact sensor bytes once, retaining IDs, timestamps and compact hash receipts."""
    content, records = [], []
    for record in source_records:
        receipt = dict(record)
        if record["stream"].startswith("camera/"):
            data = Path(record["path"]).read_bytes()
            receipt.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Front-wide {record['id']}; actual capture "
                        f"{record['timestamp_us']}us, "
                        f"{(record['timestamp_us'] - clock.origin_us) / 1e6:.3f}s from origin.",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            )
        records.append(receipt)
    return content, records
