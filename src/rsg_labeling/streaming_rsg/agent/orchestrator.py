"""The orchestrator: context management for model-backed ops.

Decides what each model call sees, administers the env's access policy at run time, and owns
the context layer -- every rendered context, model turn, and tool call is persisted
(``context.jsonl``) with the ids it involved, so the RSG stores read-sets while the context
layer stores the material. Renderings are queryable for loop optimization, never exact-replay
(models are nondeterministic).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..data_structure.rsg import Rsg
from ..env.env import Anchor, Env
from .memory import MemoryBlock
from .tools import TOOLS


@dataclass
class Orchestrator:
    """Context manager for one run: renders views, enforces access, persists the context layer."""

    env: Env
    memory: MemoryBlock = field(default_factory=MemoryBlock)
    records: list[dict] = field(default_factory=list)
    # run deadline (time.monotonic terms), set by the executor: internally-parallel
    # ops admit work waves against it instead of running open-loop past the budget
    budget_deadline: float | None = None
    ops_used: int = 0
    budget_exhausted: bool = False
    tools: dict = field(default_factory=lambda: dict(TOOLS))

    def render_context(
        self, rsg: Rsg, anchor: Anchor, role: str, op: str, graph_view=None, component_filter=None
    ) -> tuple[str, list[str], str]:
        """Render the env contract, trajectory, and current components for ``role``.

        Applies the env access policy through the env's mediated reads. Returns the rendered
        text, the read-set (source record ids + component ids), and the persisted context
        record id, which doubles as the ``model_call`` provenance for the op.

        ``graph_view`` (a finished ReadViewOp) replaces the orchestrator's own component/
        relation rendering with the op-mediated view -- the read went through the gateway
        and its record id joins the read set. Ops assembling their own round prompts keep
        the default path (op-internal input prep).
        """
        read_ids: list[str] = []
        sections: list[str] = [self.render_env(anchor, role)]

        ego_records = self.env.ego_states(role, anchor)
        read_ids.extend(record.id for record in ego_records)
        sections.append(
            "EGO TRAJECTORY (pose sequence):\n"
            + "\n".join(f"[{r.id}] {r.summary}" for r in ego_records)
        )
        if self.env.video_binding is not None:
            sections[-1] = (
                "VIDEO-ONLY INPUT: measured ego trajectory, speed, yaw, navigation and "
                "production actions are unavailable. Infer only visibly supported motion; "
                "do not impute missing kinematics as zero or infer route intent "
                "from missing metadata. "
                "All timestamps are relative video presentation time, not absolute sensor time."
            )

        if graph_view is not None:
            sections.append(graph_view.rendered)
            read_ids.append(graph_view.record_id)
            memory_block = self.memory.render()
            if memory_block:
                sections.append("MEMORY:\n" + memory_block)
            text = "\n\n".join(sections)
            record_id = f"context:{len(self.records)}"
            self.records.append(
                {"id": record_id, "op": op, "role": role, "read_ids": read_ids, "text": text}
            )
            return text, read_ids, record_id

        by_category: dict[str, list[str]] = {}
        rendered_components = [
            component
            for component in rsg.components
            if component_filter is None or component_filter(component)
        ]
        for component in rendered_components:
            read_ids.append(component.id)
            tier = f"T{int(component.tier)}"
            entry = f"[{component.id}] {tier} {component.label}"
            timeline = render_event_timeline(component, anchor.anchor_us)
            if timeline:
                entry += f"\n    timeline: {timeline}"
            by_category.setdefault(component.category, []).append(entry)
        for category, lines in by_category.items():
            sections.append(f"{category.upper()}:\n" + "\n".join(lines))

        reference_block = render_reference(self.env, anchor.anchor_us)
        if reference_block:
            sections.append(reference_block)
            read_ids.extend(
                record.id
                for record in self.env.catalog.values()
                if record.stream.startswith("reference/")
            )

        live_ids = {component.id for component in rendered_components}
        relation_lines = [
            f"{edge.source} --{edge.relation}--> {edge.target}: {edge.rationale}"
            for edge in rsg.edges
            if edge.source in live_ids and edge.target in live_ids
        ]
        if relation_lines:
            sections.append("RELATIONS:\n" + "\n".join(relation_lines))

        memory_block = self.memory.render()
        if memory_block:
            sections.append("MEMORY:\n" + memory_block)

        text = "\n\n".join(sections)
        record_id = f"context:{len(self.records)}"
        self.records.append(
            {"id": record_id, "op": op, "role": role, "read_ids": read_ids, "text": text}
        )
        return text, read_ids, record_id

    def render_env(self, anchor: Anchor, role: str) -> str:
        """Render the env contract so the agent knows the playground it operates in."""
        env = self.env
        access = (
            f"gt_labels={'visible' if env.access.agent_gt_labels else 'hidden'}, "
            f"future_ego={'visible' if env.access.agent_future_ego else 'hidden'}"
            if role == "agent"
            else "full (eval role)"
        )
        sources = ", ".join(f"{name}={cls.value}" for name, cls in env.sources.items())
        return (
            "ENV (the playground contract):\n"
            f"env_id={env.env_id} clip={env.clip_id}\n"
            f"anchor={anchor.anchor_us}us, history={anchor.history_us}us, "
            f"future={anchor.future_us}us, trigger={anchor.trigger}\n"
            f"sampling: max_fps={env.sampling.max_fps}, max_frames={env.sampling.max_frames}, "
            f"cameras={list(env.sampling.cameras)}\n"
            f"access ({role}): {access}\n"
            f"budget: max_ops={env.budget.max_ops}, wall_clock={env.budget.wall_clock_s}s\n"
            f"source classes: {sources}"
        )

    def record_model_turn(
        self,
        op: str,
        prompt_record_id: str,
        response: str,
        source_class: str,
        reads: tuple[str, ...] = (),
    ) -> str:
        """Persist one model response with its reliability class; returns the record id.

        ``source_class`` is ``vlm_observation`` for perception turns and ``model_proposal``
        for proposal turns — the class the reliability walk sees when a component cites the
        record.
        """
        record_id = f"context:{len(self.records)}"
        self.records.append(
            {
                "id": record_id,
                "op": op,
                "response_to": prompt_record_id,
                "source_class": source_class,
                "reads": list(reads),
                "text": response,
            }
        )
        return record_id

    def record_parse(self, op: str, description: str, reads: tuple[str, ...] = ()) -> str:
        """Persist one deterministic parse result (class ``rule_parsed``); returns its id."""
        record_id = f"context:{len(self.records)}"
        self.records.append(
            {
                "id": record_id,
                "op": op,
                "source_class": "rule_parsed",
                "reads": list(reads),
                "text": description,
            }
        )
        return record_id

    def focus_requests(self, rsg, anchor, max_requests: int = 4) -> list[tuple[str, tuple, str]]:
        """The adaptive-focus policy: which targets deserve a focused observation round.

        Selecting targets and windows is context management -- this orchestrator's job, not
        an op's. Triggers: diverged claims (recorded disagreement), fragmented signals (a
        traffic_control component whose single claim spans one round -- its phase sequence
        never formed), and unverifiable claims. The cap is filled ROUND-ROBIN across the
        trigger classes in that priority order, so an abundant class (diverged claims
        dominate hot scenes) cannot starve the others out of the budget. One request per
        component, capped by ``max_requests`` (the env op/wall-clock budgets bound the
        rest). Returns ``(component_id, window_us, reason)`` tuples.
        """
        start_us, _ = anchor.window_us
        end_us = anchor.anchor_us
        candidates: list[tuple[int, str, tuple, str]] = []
        for component in rsg.components:
            observation_events = [
                event for event in component.events if "resolution" in event.attributes
            ]
            if not observation_events:
                continue
            resolutions = {event.attributes["resolution"] for event in observation_events}
            windows = [event.window_us for event in observation_events]
            lo = max(min(w[0] for w in windows) - 2_000_000, start_us)
            hi = min(max(w[1] for w in windows) + 2_000_000, end_us)
            if "diverged" in resolutions:
                candidates.append((0, component.id, (lo, hi), "diverged claim"))
            elif component.category == "traffic_control" and len(observation_events) == 1:
                candidates.append((1, component.id, (lo, hi), "fragmented signal"))
            elif "unverifiable" in resolutions:
                candidates.append((2, component.id, (lo, hi), "unverifiable claim"))
        buckets: dict[int, list] = {}
        for priority, component_id, window, reason in candidates:
            buckets.setdefault(priority, []).append((component_id, window, reason))
        ordered_buckets = [buckets[priority] for priority in sorted(buckets)]
        selected: list[tuple[str, tuple, str]] = []
        index = 0
        while len(selected) < max_requests and any(ordered_buckets):
            bucket = ordered_buckets[index % len(ordered_buckets)]
            if bucket:
                selected.append(bucket.pop(0))
            index += 1
            if index > max_requests * len(ordered_buckets) + len(candidates):
                break
        return selected

    def structured_model_turn(
        self,
        op: str,
        prompt_record_id: str,
        complete,
        messages: list[dict],
        parse,
        source_class: str,
        reads: tuple[str, ...] = (),
        retries: int = 2,
    ):
        """One model interaction that must parse: complete -> record -> parse, with retry.

        Eliciting well-formed output from the model is agent-layer interaction protocol,
        not part of any op's contract (ops are the graph gateway; this is the periphery).
        Every attempt's turn is recorded -- failed elicitations stay auditable. Returns
        ``(parsed, record_id)`` of the attempt that parsed; raises after ``retries``
        re-samples (the known fence-truncation flake hit ~16% of study clips before this).

        A retry RE-ASKS rather than re-sending the identical prompt: measured 2026-08-29,
        one clip per generation died on a malformed reply that a bare re-sample reproduced,
        because nothing in the second request told the model what went wrong.
        """
        last_error: Exception | None = None
        attempt_messages = list(messages)
        for _ in range(retries + 1):
            reply = complete(attempt_messages)
            record_id = self.record_model_turn(op, prompt_record_id, reply, source_class, reads)
            try:
                return parse(reply), record_id
            except ValueError as error:
                last_error = error
                attempt_messages = [
                    *messages,
                    {"role": "assistant", "content": reply[:2000]},
                    {
                        "role": "user",
                        "content": (
                            f"That reply could not be parsed: {str(error)[:200]}. Return ONLY "
                            "the JSON, complete and closed, with no prose and no code fence."
                        ),
                    },
                ]
        raise last_error

    def call_tool(
        self,
        op: str,
        tool: str,
        argument: str,
        execute,
        reads: tuple[str, ...] = (),
        summarize=str,
    ):
        """Execute one tool call and record it atomically; returns ``(result, record_id)``.

        Tool execution is agent-layer interaction protocol (the periphery), never op
        contract: ops receive the recorded result and decide what to commit. ``execute``
        is a no-arg thunk; ``summarize`` renders the result for the context record.
        """
        result = execute()
        record_id = self.record_tool_call(op, tool, argument, summarize(result), reads=reads)
        return result, record_id

    def call_registered(self, op: str, requested: str, argument: str, rsg, env, anchor):
        """Dispatch a registered tool or skill by name, executed + recorded here.

        The registry lookup (tools vs skills) is agent-layer routing; ops pass through the
        model's request and consume the recorded result.
        """
        if requested in self.tools:
            execute = lambda: self.tools[requested](rsg, argument)  # noqa: E731
        else:
            from .skills import SKILLS

            execute = lambda: SKILLS[requested](rsg, env, anchor, self, argument)  # noqa: E731
        return self.call_tool(op, requested, argument, execute)

    def record_tool_call(
        self, op: str, tool: str, argument: str, result: str, reads: tuple[str, ...] = ()
    ) -> str:
        """Persist one tool call (class ``tool_output``); returns the citable record id."""
        record_id = f"context:{len(self.records)}"
        self.records.append(
            {
                "id": record_id,
                "op": op,
                "tool": tool,
                "argument": argument,
                "source_class": "tool_output",
                "reads": list(reads),
                "text": result,
            }
        )
        return record_id

    def write_context_log(self, path: Path) -> None:
        """Serialize the context layer, one JSON record per line."""
        with path.open("w") as log_file:
            for record in self.records:
                log_file.write(json.dumps(record) + "\n")


def _evidence_tag(event, anchor_us: int) -> str:
    """The event's epistemics, rendered from what it already carries: tier + sources.

    PRE-DESIGNED, never policy-configurable (epistemic_context_plan review): a renderer
    the loop could configure could learn to hide uncertainty. Buckets by source prefix:
    external labels, decoded frames (with their span -- when the look actually happened),
    ego poses, derived records (model/rule output).
    """
    buckets: dict[str, list[int]] = {}
    for source in event.sources:
        parts = str(source).split(":")
        kind = {
            "label": "labels",
            "frame": "frames",
            "ego": "poses",
            "render": "frames",
            "ref": "ref",
        }.get(parts[1] if parts[0] == "env" and len(parts) > 1 else "", None)
        if kind is None:
            kind = "derived" if parts[0] == "context" else "other"
        stamps = [int(p) for p in parts if p.isdigit() and len(p) >= 6]
        buckets.setdefault(kind, []).append(stamps[0] if stamps else None)
    pieces = []
    for kind in ("labels", "frames", "poses", "ref", "derived", "other"):
        stamps = buckets.get(kind)
        if not stamps:
            continue
        known = [s for s in stamps if s is not None]
        span = (
            f" {(min(known) - anchor_us) / 1e6:+.1f}..{(max(known) - anchor_us) / 1e6:+.1f}s"
            if known and kind == "frames"
            else ""
        )
        pieces.append(f"{kind}{span} x{len(stamps)}")
    return f"{{T{int(event.tier)} " + "; ".join(pieces or ["uncited"]) + "}"


# A break is a step FAST relative to the tracklet's own motion, not a step merely large:
# an absolute metre threshold flagged 81% of tracklets on smoke7, because a car doing
# 15m/s legitimately moves 15m per sighting. Each tracklet is its own baseline (3x its
# median speed), with a floor so a near-stationary track is not broken by depth noise.
_BREAK_SPEED_RATIO = 3.0
_BREAK_SPEED_FLOOR_MPS = 5.0


def tracklet_tags(env, anchor_us: int) -> dict[str, dict]:
    """Per-tracklet motion tag from the DA3 geometry: range profile and break points.

    The STATIC TAG the agent reasons over. A tracklet is perception's claim that these
    sightings are one object; its world positions say whether that can be true. A smooth
    range profile corroborates the claim, a metre-scale discontinuity contradicts it --
    and the geometry sees this where the pixels cannot, because a car at 25m and the same
    car at 6m look like different vehicles while a swap between adjacent cars looks like
    none at all.

    Returns ``{track_id: {"first_s", "last_s", "range_start_m", "range_end_m",
    "sightings", "max_step_m", "breaks": [seconds]}}``.
    """
    import collections

    by_track: dict[str, list] = collections.defaultdict(list)
    for record in env.catalog.values():
        if not (record.stream or "").startswith("geometry/"):
            continue
        for obj in (record.payload or {}).get("objects", {}).values():
            if obj.get("track_id") and obj.get("world_xyz"):
                by_track[str(obj["track_id"])].append(
                    (record.timestamp_us, obj["world_xyz"], obj.get("distance_m"))
                )
    tags = {}
    for track_id, entries in by_track.items():
        entries.sort()
        steps: list[tuple[float, float, int]] = []  # (metres, m/s, timestamp)
        for (prev_ts, prev, _pd), (ts, world, _d) in zip(entries, entries[1:]):
            step = sum((a - b) ** 2 for a, b in zip(prev, world)) ** 0.5
            dt = max((ts - prev_ts) / 1e6, 1e-3)
            steps.append((step, step / dt, ts))
        # LEAVE-ONE-OUT baseline: each step is judged against the median of the OTHERS.
        # A plain median cannot flag anything on a two-step track, because the median IS
        # the larger step -- and two- and three-sighting tracklets are the common case.
        speeds = [speed for _, speed, _ in steps]
        breaks = []
        for index, (_step, speed, ts) in enumerate(steps):
            others = sorted(speeds[:index] + speeds[index + 1 :])
            baseline = others[len(others) // 2] if others else speed
            if speed > max(_BREAK_SPEED_RATIO * baseline, _BREAK_SPEED_FLOOR_MPS):
                breaks.append(round((ts - anchor_us) / 1e6, 1))
        median_speed = sorted(speeds)[len(speeds) // 2] if speeds else 0.0
        tags[track_id] = {
            "first_s": round((entries[0][0] - anchor_us) / 1e6, 1),
            "last_s": round((entries[-1][0] - anchor_us) / 1e6, 1),
            "range_start_m": entries[0][2],
            "range_end_m": entries[-1][2],
            "sightings": len(entries),
            "max_step_m": round(max(s for s, _, _ in steps), 1) if steps else 0.0,
            "median_speed_mps": round(median_speed, 1),
            "breaks": breaks,
        }
    return tags


def _tag_phrase(tag: dict) -> str:
    """The tag as one compact clause the model reads beside the tracklet id."""
    start, end = tag.get("range_start_m"), tag.get("range_end_m")
    span = (
        f"{start:.0f}->{end:.0f}m"
        if isinstance(start, int | float) and isinstance(end, int | float)
        else "range unknown"
    )
    trend = ""
    if isinstance(start, int | float) and isinstance(end, int | float):
        trend = " closing" if end < start - 1 else " receding" if end > start + 1 else " holding"
    if tag["breaks"]:
        marks = ", ".join(f"{b:+.0f}s" for b in tag["breaks"][:3])
        return (
            f"{span}{trend}, {tag['sightings']} sightings, "
            f"typical {tag.get('median_speed_mps', 0)}m/s but POSITION BREAKS at {marks} "
            f"(max step {tag['max_step_m']}m -- these sightings may not be one object)"
        )
    return f"{span}{trend}, {tag['sightings']} sightings, motion coherent"


def render_reference(env, anchor_us: int, window_us: tuple | None = None) -> str:
    """The model-derived reference block: declared ego-path frame + distilled records.

    PRE-DESIGNED rendering (integrity floor, like the evidence tags). The convention is
    stated with the data -- the audit lesson: a reference without its declared frame is
    unadjudicable. Positions are image-frame proxies until depth lands: left/right =
    thirds around the camera center (front camera ~ ego heading); on_road = the anchor
    sits below the drivable boundary at its x.
    """
    refs = sorted(
        (
            record
            for record in env.catalog.values()
            if record.stream.startswith("reference/")
            and (
                window_us is None
                or (record.timestamp_us or 0) >= window_us[0]
                and (record.timestamp_us or 0) <= window_us[1]
            )
        ),
        key=lambda record: (record.timestamp_us or 0, record.stream),
    )
    if not refs:
        return ""
    boundaries = {
        record.timestamp_us: record.payload.get("boundary_xy_norm", [])
        for record in refs
        if record.payload.get("kind") == "road"
    }
    tags = tracklet_tags(env, anchor_us)
    lines = []
    for record in refs:
        t_s = ((record.timestamp_us or 0) - anchor_us) / 1e6
        payload = record.payload
        if payload.get("kind") == "road":
            lines.append(
                f"[{record.id}] road boundary @{t_s:+.1f}s: "
                f"{len(payload.get('boundary_xy_norm', []))} pts"
            )
            continue
        entries = []
        boundary = boundaries.get(record.timestamp_us, [])
        for obj in payload.get("objects", []):
            x, y = obj["anchor_xy"]
            side = "left" if x < 1 / 3 else "right" if x > 2 / 3 else "center"
            boundary_y = min((pt[1] for pt in boundary if abs(pt[0] - x) < 0.06), default=None)
            on_road = boundary_y is not None and y >= boundary_y
            # the tracklet id is what makes this entry REFERENCEABLE: without it the
            # model can see fourteen vehicles and name none of them
            track = obj.get("track_id")
            entries.append(
                (f"[{track}] " if track else "")
                + f"({x:.2f},{y:.2f}) {side}"
                + (", on-road" if on_road else "")
                + (", TRUNCATED (partially out of frame)" if obj["truncated"] else "")
            )
        lines.append(f"[{record.id}] {payload.get('kind')}s @{t_s:+.1f}s: " + "; ".join(entries))
    tag_lines = [
        f"  [{track_id}] {_tag_phrase(tag)}"
        for track_id, tag in sorted(tags.items())
        if tag["sightings"] > 1
    ]
    tag_block = (
        "\nTRACKLET TAGS (depth-derived motion per tracklet, independent of appearance -- "
        "a car at 25m and the same car at 6m look different, and a swap between adjacent "
        "cars looks like none, so trust this over the crop when they disagree):\n"
        + "\n".join(tag_lines)
        if tag_lines
        else ""
    )
    return (
        "MODEL REFERENCE (ego-path frame -- left/right relative to the camera center "
        "= ego heading; on-road = below the drivable boundary; image-frame proxies "
        "until metric depth lands).\n"
        "Entries in [brackets] are TRACKLET IDS proposed by the perception models and "
        "carried on the overlay boxes. They are PROPOSALS, not established facts: cite "
        "one when your observation is about that object, ignore it when it is wrong, and "
        "describe objects it missed with no id at all.\n" + "\n".join(lines) + tag_block
    )


def render_event_timeline(component, anchor_us: int, window_us: tuple | None = None) -> str:
    """One compact per-entity timeline: events with anchor-relative times.

    ``window_us`` clips the rendering to events overlapping that window -- prompt
    consumers (observe/focus rounds) show the dynamics of the segment being looked at,
    nothing else. Every event line carries its epistemics tag (tier + evidence classes
    + frame coverage) so externally-labeled fact, model claim, and sampling density are
    never conflated at the prompt boundary.

    The full RSG renders -- no per-op truncation; staying within the prompt budget is this
    orchestrator's job (compression happens here, visibly, when it is ever needed).
    Divergences surface inline: a decision should see disagreement, not a silent average.
    """
    events = [
        event
        for event in component.events
        if window_us is None
        or (event.window_us[0] <= window_us[1] and event.window_us[1] >= window_us[0])
    ]
    if not events:
        return ""
    parts = []
    for event in events:
        start_s = (event.window_us[0] - anchor_us) / 1e6
        end_s = (event.window_us[1] - anchor_us) / 1e6
        detail = ""
        speeds = event.attributes.get("speed_mps_range")
        if speeds and speeds[0] != speeds[1]:
            detail += f" speed {speeds[0]}->{speeds[1]}m/s"
        if "distance_m_start" in event.attributes and (
            event.attributes["distance_m_start"] != event.attributes.get("distance_m_end")
        ):
            detail += (
                f" dist {event.attributes['distance_m_start']}"
                f"->{event.attributes['distance_m_end']}m"
            )
        if "divergence" in event.attributes:
            detail += f" [DIVERGENCE: {event.attributes['divergence']}]"
        tag = _evidence_tag(event, anchor_us)
        parts.append(f"{event.label}[{start_s:+.1f}..{end_s:+.1f}s]{detail} {tag}")
    return "; ".join(parts)
