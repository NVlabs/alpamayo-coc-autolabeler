"""Op-mediated, immutable task snapshot of the live temporal graph and support records."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

from .base import OpResult


class ReadTimingEvidenceOp:
    """Read bounded physical transition evidence without adding or revising graph facts."""

    name = "read:timing_evidence"

    def __init__(
        self,
        clock_us: tuple[int, int],
        windows: list[tuple[int, int]],
        cache_dir: Path,
        visual: bool,
        detail_stride_us: int | None = None,
    ) -> None:
        """Declare clip extent and targeted windows; density comes from the env contract."""
        if clock_us[0] >= clock_us[1] or any(lo >= hi for lo, hi in windows):
            raise ValueError("timing evidence requires positive physical windows")
        self.clock_us, self.windows, self.cache_dir, self.visual = (
            clock_us,
            windows,
            cache_dir,
            visual,
        )
        self.records: list[dict] | None = None
        if detail_stride_us is not None and (
            type(detail_stride_us) is not int or detail_stride_us <= 0
        ):
            raise ValueError("detail stride must be a positive integer")
        self.detail_stride_us = detail_stride_us

    def run(self, rsg, env, anchor, orchestrator) -> OpResult:
        """Keep real capture times and kinematic footprints under the existing access policy."""
        lo, hi = self.clock_us
        readings = [r for r in env.ego_states("agent", anchor) if lo <= r.timestamp_us < hi]
        records = [r.to_dict() for r in readings]
        if self.visual:
            targets = sorted(
                {
                    r.timestamp_us
                    for r in readings
                    if any(a <= r.timestamp_us <= b for a, b in self.windows)
                }
            )
            if env.video_binding is not None:
                step = max(1, round(1_000_000 / env.sampling.max_fps))
                targets = [
                    t for t in range(lo, hi, step) if any(a <= t <= b for a, b in self.windows)
                ]
            if self.detail_stride_us is None:
                frames = env.frames("camera_front_wide_120fov", targets, self.cache_dir)
            else:
                if env.sampling.image_overview_height_px is None:
                    raise ValueError("detail snapshots require a mixed-resolution environment")
                if len(targets) > env.sampling.max_frames:
                    raise ValueError("combined detail/overview request exceeds frame allowance")
                detailed = {targets[0], targets[-1]} if targets else set()
                for target in range(lo, hi, self.detail_stride_us):
                    if targets:
                        detailed.add(min(targets, key=lambda stamp: (abs(stamp - target), stamp)))
                frames = env.frames(
                    "camera_front_wide_120fov",
                    [stamp for stamp in targets if stamp not in detailed],
                    self.cache_dir,
                )
                frames += env.frames(
                    "camera_front_wide_120fov", sorted(detailed), self.cache_dir, high_detail=True
                )
                frames.sort(key=lambda record: record.timestamp_us)
            records.extend(r.to_dict() for r in frames if lo <= r.timestamp_us < hi)
        self.records = list({r["id"]: r for r in records}.values())
        orchestrator.record_parse(
            self.name,
            "retrospective transition timing; no graph mutation",
            reads=tuple(r["id"] for r in self.records),
        )
        return OpResult()


class ReadEvidenceOp:
    """Select bounded, already observed front-camera evidence without new sensor access."""

    name = "read:evidence"

    def __init__(self, window_us: tuple[int, int], max_frames: int) -> None:
        """Fix the physical clip clock and the explicitly configured teacher image limit."""
        if type(max_frames) is not int or not 1 <= max_frames <= 24:
            raise ValueError("interpreter evidence limit must be between 1 and 24")
        if window_us[0] >= window_us[1]:
            raise ValueError("evidence window must be positive")
        self.window_us = window_us
        self.max_frames = max_frames
        self.records: list[dict] | None = None

    def run(self, rsg, env, anchor, orchestrator) -> OpResult:
        """Reuse catalogued captures evenly over time, retaining original IDs and paths."""
        camera = "camera_front_wide_120fov"
        if camera not in env.sampling.cameras:
            raise ValueError("front-wide evidence is not authorized by this environment")
        lo, hi = self.window_us
        available = sorted(
            (
                r
                for r in env.catalog_dict()
                if r["stream"] == f"camera/{camera}"
                and r["source_class"] == "sensor"
                and r.get("path")
                and r.get("timestamp_us") is not None
                and lo <= r["timestamp_us"] < hi
            ),
            key=lambda r: (r["timestamp_us"], r["id"]),
        )
        if not available:
            raise ValueError("no observed front-wide evidence is available")
        count = min(self.max_frames, len(available))
        indices = (
            [round(i * (len(available) - 1) / (count - 1)) for i in range(count)]
            if count > 1
            else [len(available) // 2]
        )
        self.records = [deepcopy(available[i]) for i in indices]
        orchestrator.record_parse(
            self.name,
            "bounded interpreter visual evidence",
            reads=tuple(r["id"] for r in self.records),
        )
        return OpResult()


class ReadSnapshotOp:
    """Read event-level facts without exposing retired entities as live evidence."""

    name = "read:snapshot"

    def __init__(self) -> None:
        """Prepare an empty output; the executor owns invocation and read tracing."""
        self.snapshot: dict | None = None

    def run(self, rsg, env, anchor, orchestrator) -> OpResult:
        """Persist a graph-versioned snapshot; append no components or relations."""
        graph = rsg.to_dict()
        # Executor timing/log records are not graph revisions and are appended after reads.
        graph.pop("log", None)
        graph_hash = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
        action_record = env.meta_action_record("agent", anchor)
        records = {record["id"]: record for record in env.catalog_dict()}
        records.update({record["id"]: record for record in orchestrator.records})
        for component in rsg.all_components:
            records[component.id] = {
                "sources": list(component.sources),
                **component.provenance.to_dict(),
            }
        facts = {}
        live_ids = {component.id for component in rsg.components}
        for component in rsg.components:
            for index, event in enumerate(component.events):
                facts[f"{component.id}@{index}"] = {
                    "component_id": component.id,
                    "category": component.category,
                    "window_us": list(event.window_us),
                    "label": event.label,
                    "attributes": event.attributes,
                    "tier": int(event.tier),
                    "sources": list(event.sources),
                    "read_ids": list(component.provenance.read_ids)
                    + (
                        [component.provenance.model_call] if component.provenance.model_call else []
                    ),
                }
        self.snapshot = deepcopy(
            {
                "graph_hash": graph_hash,
                "env_id": rsg.env_id,
                "facts": facts,
                "edges": [
                    edge.to_dict()
                    for edge in rsg.edges
                    if edge.source in live_ids and edge.target in live_ids
                ],
                "records": records,
                **(
                    {"action_context": {"record_id": action_record.id, **action_record.payload}}
                    if action_record is not None
                    else {}
                ),
            }
        )
        orchestrator.record_parse(
            self.name,
            f"snapshot {graph_hash}",
            reads=tuple(sorted(live_ids)) + ((action_record.id,) if action_record else ()),
        )
        return OpResult()
