"""Observe frames and focus evidence through the graph operation gateway."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

from ...infra.middleware.parsing import parse_json_reply_list as _parse_json_reply_list
from ..agent.orchestrator import Orchestrator, render_event_timeline, render_reference
from ..agent.tools.render_overlays import has_tracklets
from ..data_structure.rsg import Component, Provenance, Rsg, Tier
from ..env.env import Anchor, Env
from .base import OpResult
from .claims import (
    _DEDUP_GAP_US,
    _DEDUP_JACCARD,
    _coalesce_claim,
    _merged_events,
    _observation_event,
    _record_span,
)

# Categories the observe op may assign to claims the labels missed.
OBSERVABLE_CATEGORIES = ("traffic_control", "dynamic_agent", "static_road_element")


CONTEXTUAL_CATEGORIES = ("risk", "environmental_context")


def _enrich_id(component_id: str, all_ids: set) -> str:
    """Compact enrichment id: strip prior #observed/#obsN suffixes, then #observed/#obsK."""
    import re as _re

    root = _re.sub(r"(#observed\d*|#obs\d+)+$", "", component_id)
    candidate = f"{root}#observed"
    ordinal = 2
    while candidate in all_ids:
        candidate = f"{root}#obs{ordinal}"
        ordinal += 1
    return candidate


def observe_segments(anchor: Anchor, windows: str, rsg: Rsg) -> list[tuple[int, int]]:
    """The sub-windows one observe pass tiles the agent window into.

    ``single`` is one segment over the whole window (the pre-windowing behavior; keyframe
    runs keep it). ``tile:<N>s`` tiles the window into fixed N-second segments. ``maneuver``
    aligns segments to the ego's maneuver events (requires parse:maneuver before observe;
    falls back to single when the ego carries no events).
    """
    start_us, _ = anchor.window_us
    end_us = anchor.anchor_us
    if windows == "single":
        return [(start_us, end_us)]
    if windows.startswith("tile:") and windows.endswith("s"):
        tile_us = int(float(windows[len("tile:") : -1]) * 1e6)
        segments = []
        t = start_us
        while t < end_us:
            segments.append((t, min(t + tile_us, end_us)))
            t += tile_us
        return segments
    if windows == "maneuver":
        ego = next((c for c in rsg.components if c.category == "ego_vehicle"), None)
        if ego is None or not ego.events:
            return [(start_us, end_us)]
        # Channel events overlap, so segments come from the union of their boundaries:
        # between any two adjacent boundaries the ego behavior is constant in every channel.
        boundaries = {start_us, end_us}
        for event in ego.events:
            for bound in event.window_us:
                if start_us <= bound <= end_us:
                    boundaries.add(bound)
        ordered = sorted(boundaries)
        return [(left, right) for left, right in zip(ordered, ordered[1:]) if right > left]
    raise ValueError(f"unknown observe windows spec: {windows!r}")


class ObserveFramesOp:
    """VLM observes the camera frames; claims enrich components or ground new ones.

    Rendered overlays (``derived/box_render`` records) are preferred over raw frames when
    present, so the visual prompt carries the labeled track boxes and the observation's
    sources chain to both the media and the labels that augmented it. With ``windows`` other
    than ``single``, observation runs once per sub-window and claims bind to their segment's
    span -- the fix for whole-clip claims collapsing onto one flat window.
    """

    name = "observe:frames"

    def __init__(
        self,
        complete: Callable[[list[dict]], str],
        cache_dir: Path,
        media: str = "frames",
        windows: str = "single",
        render_tool: Callable | None = None,
        round_local: bool = False,
    ) -> None:
        """Bind the chat callable, the media cache, the input form, and the windowing.

        ``media`` selects the input form for the paired ablation: ``frames`` (default --
        timestamped stills, exact claim-to-time attribution) or ``video`` (one mp4 segment
        over the agent window -- native motion, but claims bind to the whole span; only
        valid with ``windows="single"``).
        """
        if media not in ("frames", "video"):
            raise ValueError(f"unknown observe media form: {media!r}")
        if media == "video" and windows != "single":
            raise ValueError("video input binds to the whole span; windowing requires frames")
        self.complete = complete
        self.cache_dir = cache_dir
        self.media = media
        self.windows = windows
        self.render_tool = render_tool
        # round-local context: each round's candidate table lists only dynamic agents
        # whose windows overlap ITS segment -- association candidates the round can
        # actually see. Off = window-global table (measured: 23-108 rows regardless
        # of segment; the text clutter behind unconstrained id-to-pixel association).
        self.round_local = round_local
        self.dedup_jaccard = _DEDUP_JACCARD
        self.dedup_gap_us = _DEDUP_GAP_US
        self.dedup_key = "text"
        self.max_frames: int | None = None
        self.max_fps: float | None = None
        self._focus_target: str | None = None
        self._focus_only: str | None = None
        self._exclude_route_intent = False
        self._require_null_for_new_claim = False

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """One observation round per sub-window; claims bind to their segment's span."""
        # Enhanced visual context is the agent's visible, optimizable choice (review
        # decision): overlay preparation is a recorded tool call, not an op of its own --
        # the derived media lands in the env catalog; rounds below prefer it per frame.
        segments = observe_segments(anchor, self.windows, rsg)
        if (
            self.render_tool is not None
            and self.media == "frames"
            # EITHER scaffold justifies an overlay: labels when the env carries them,
            # VFM tracklets otherwise. Gating on labels alone made the overlay path
            # unreachable in exactly the mode that needs something to point at.
            and ("object_labels" in env.sources or has_tracklets(env))
        ):
            # render at the ROUND targets: overlays substitute by exact timestamp,
            # so they must exist at the frames the rounds below actually consume
            # (the anchor-grid default never matched multi-window sampling)
            round_targets = sorted(
                {t for segment in segments for t in self._segment_targets(env, anchor, segment)}
            )
            orchestrator.call_tool(
                self.name,
                "render_overlays",
                f"{len(round_targets)} round frames, all contract cameras",
                lambda: self.render_tool(rsg, timestamps_us=round_targets),
            )
        totals = {"observations": 0, "enriched": 0, "created": 0, "diverged": 0, "rounds": 0}
        for segment in segments:
            frame_records = self._collect_media(env, anchor, segment)
            totals["rounds"] += 1
            self._observe_round(rsg, env, anchor, orchestrator, frame_records, segment, totals)
        orchestrator.memory.write(
            f"VLM observed {totals['observations']} items over {totals['rounds']} rounds "
            f"({totals['enriched']} enrichments, {totals['created']} new components, "
            f"{totals['diverged']} divergences, {totals.get('coalesced', 0)} coalesced, "
            f"{totals.get('near_miss', 0)} near-miss"
            + (
                f"; round-local candidates {min(totals['candidate_counts'])}"
                f"-{max(totals['candidate_counts'])}"
                if totals.get("candidate_counts")
                else ""
            )
            # the phase-1 reference rate: how often the model actually points at a
            # perception proposal when it has one to point at
            + f"; {totals.get('tracklet_cited', 0)} cited a tracklet"
            + ")"
        )
        return OpResult()

    def _segment_targets(self, env: Env, anchor: Anchor, segment: tuple[int, int]) -> list:
        """The frame timestamps one round samples (shared with overlay rendering)."""
        count = self.max_frames if self.max_frames is not None else env.sampling.max_frames
        fps = self.max_fps if self.max_fps is not None else env.sampling.max_fps
        if (
            type(count) is not int
            or not 1 <= count <= env.sampling.max_frames
            or not 0 < fps <= env.sampling.max_fps
        ):
            raise ValueError("observation sampling policy exceeds environment bounds")
        if self.windows == "single":
            return env.frame_targets(anchor)[:count]
        step_us = max(int(round(1e6 / fps)), 1)
        targets = list(range(segment[0], segment[1] + 1, step_us))
        return targets[:count]

    def _collect_media(self, env: Env, anchor: Anchor, segment: tuple[int, int]) -> list:
        """The media records for one segment: overlay-preferred frames, or the video span."""
        if self.media == "video":
            return [
                env.video_segment(camera, segment, self.cache_dir)
                for camera in env.sampling.cameras
            ]
        targets = self._segment_targets(env, anchor, segment)
        overlays_by_ts = {
            record.timestamp_us: record
            for record in env.catalog.values()
            if record.stream == "derived/box_render"
        }
        frame_records = []
        for camera in env.sampling.cameras:
            for raw in env.frames(camera, targets, self.cache_dir):
                frame_records.append(overlays_by_ts.get(raw.timestamp_us, raw))
        return frame_records

    def _observe_round(
        self, rsg, env, anchor, orchestrator, frame_records, segment, totals
    ) -> None:
        """One VLM round over one segment's media; writes enrichments and observed components."""
        segment_note = (
            f"This round covers {((segment[0] - anchor.anchor_us) / 1e6):+.1f}s to "
            f"{((segment[1] - anchor.anchor_us) / 1e6):+.1f}s relative to the final moment."
        )
        focus_target = self._focus_target
        if focus_target:
            segment_note += f" {focus_target}"
        if self.media == "video":
            media_note = (
                "A video segment from the ego vehicle's front camera over the observation "
                "window (it ends at the decision moment)."
            )
        else:
            media_note = (
                "Frames from the ego vehicle's front camera, oldest to newest. Green boxes "
                "with track ids, when drawn, mark the labeled objects."
            )
        # numbered-pin overlays: the image carries pins, the roster carries the pin ->
        # component mapping (the model reads the pin off the image, names the component
        # from the roster -- id-to-pixel binding with neither corner-text clutter nor
        # long ids on the image)
        pin_by_track: dict[str, int] = {}
        for record in env.catalog.values():
            if record.stream == "derived/box_render":
                for pin, track in record.payload.get("pins", {}).items():
                    pin_by_track[track] = int(pin)
        if pin_by_track:
            media_note = (
                "Frames from the ego vehicle's front camera, oldest to newest. Colored "
                "boxes with NUMBERED PINS mark labeled objects; each pin number maps to "
                "one component in the list below."
            )

        def segment_candidate(component) -> bool:
            """Round-local: dynamic agents render only when their window overlaps the
            segment (they are the association candidates); other categories always.
            """
            if self._exclude_route_intent and component.category == "ego_goal":
                return False
            if not self.round_local or component.category != "dynamic_agent":
                return True
            window = component.event_window_us
            return window is None or (window[0] <= segment[1] and window[1] >= segment[0])

        context, read_ids, record_id = orchestrator.render_context(
            rsg, anchor, "agent", self.name, component_filter=segment_candidate
        )

        # identity label + this round's dynamics: the timeline is clipped to the segment
        # so the model sees the state of the window it is looking at, never a snapshot
        # from another time; components with no events in-window list as identity only.
        # Grounded and model-created components render in SEPARATE sections (epistemic
        # integrity, pre-designed): a model reading its own prior claims as known scene
        # facts is a self-confirmation channel.
        def roster(components) -> str:
            def pin_tag(component) -> str:
                track = component.attributes.get("id")
                number = pin_by_track.get(track)
                return f" [pin {number}]" if number is not None else ""

            return "\n".join(
                f"{component.id} [T{int(component.tier)}]{pin_tag(component)}: {component.label}"
                + (f" | this window: {timeline}" if timeline else "")
                for component in components
                for timeline in (render_event_timeline(component, anchor.anchor_us, segment),)
            )

        created = [c for c in rsg.components if c.id.split("#", 1)[0].startswith("observed:")]
        grounded = [c for c in rsg.components if c not in created]
        if self.round_local or self._exclude_route_intent:
            grounded = [component for component in grounded if segment_candidate(component)]
        if self.round_local:
            totals.setdefault("candidate_counts", []).append(
                sum(1 for c in grounded if c.category == "dynamic_agent")
            )
        reference_block = render_reference(env, anchor.anchor_us, segment)
        component_lines = (
            ((reference_block + "\n\n") if reference_block else "")
            + "Known scene components (label/sensor-grounded):\n"
            + roster(grounded)
        )
        if created:
            component_lines += (
                "\n\nModel-observed components (these originated from YOUR OWN earlier "
                "claims and are unverified -- corroborate or correct them from what you "
                "SEE in the frames; do not restate them as established fact):\n" + roster(created)
            )
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"{media_note} {segment_note} Known scene components:\n"
                    f"{component_lines}\n\n"
                    "Report driving-relevant visual observations as ONLY a JSON list "
                    '[{"component": "<known component id or null>", "tracklet": '
                    '"<tracklet id from the reference block, or null>", "category": '
                    '"traffic_control|dynamic_agent|static_road_element|risk|'
                    'environmental_context", "description": str}]. Use the component id when '
                    "the observation clearly concerns a known component; use null with a "
                    "category for things the labels missed (signals, road state, hazards). "
                    "Set tracklet to the [id] of the perception proposal your observation "
                    "is about, or null if none fits -- a wrong proposal is better ignored "
                    "than cited. "
                    "Identify each dynamic agent by color and specific type "
                    "('the white sedan', never 'the vehicle'), adding its side only when "
                    "clearly off-center -- descriptions are re-grounded verbatim against "
                    "the visual reference."
                ),
            }
        ]
        for record in frame_records:
            media_b64 = base64.b64encode(Path(record.path).read_bytes()).decode()
            if self.media == "video":
                content.append(
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{media_b64}"},
                    }
                )
            else:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{media_b64}"},
                    }
                )
        observations, vlm_record = orchestrator.structured_model_turn(
            self.name,
            record_id,
            self.complete,
            [{"role": "user", "content": content}],
            _parse_json_reply_list,
            "vlm_observation",
            reads=tuple(record.id for record in frame_records),
        )
        totals["observations"] += len(observations)
        frame_ids = tuple(record.id for record in frame_records)
        visual_evidence = _visual_evidence(frame_records, env.catalog)
        all_ids = {component.id for component in rsg.all_components}
        current = {component.id: component for component in rsg.components}
        observed_base = sum(1 for cid in all_ids if cid.startswith("observed:"))
        known_tracklets = {
            str(obj["track_id"])
            for record in env.catalog.values()
            if (record.stream or "").startswith("reference/")
            for obj in (record.payload or {}).get("objects", []) or []
            if obj.get("track_id")
        }
        for index, item in enumerate(observations):
            description = str(item.get("description", ""))
            tracklet = _resolve_tracklet(item.get("tracklet"), known_tracklets)
            if tracklet is not None:
                totals["tracklet_cited"] = totals.get("tracklet_cited", 0) + 1
            elif item.get("tracklet"):
                totals["tracklet_unresolved"] = totals.get("tracklet_unresolved", 0) + 1
            target = item.get("component")
            # models cite stale ids across rounds; resolve through supersession like propose
            if target is not None and target not in current:
                target = next(
                    (resolved for resolved in _resolve_cited(rsg, [target]) if resolved in current),
                    target,
                )
            # BIND AT CREATION: a claim citing a tracklet joins the component that
            # tracklet already has, instead of minting a fresh one for merge to fold later.
            # Routing through the same `target` path reuses the enrichment machinery --
            # supersession, dedup, window widening -- rather than duplicating it.
            if (
                getattr(self, "bind", "per_round") == "per_tracklet"
                and tracklet is not None
                and target not in current
            ):
                target = next(
                    (
                        cid
                        for cid, comp in current.items()
                        if str((comp.attributes or {}).get("tracklet_id")) == str(tracklet)
                    ),
                    target,
                )
            focus_only = self._focus_only
            if focus_only is not None and target != focus_only:
                continue  # off-target claim in a focused round: context-recorded, not written
            if self._require_null_for_new_claim and target is not None and target not in current:
                orchestrator.memory.write(
                    f"dropped open-scene claim with unknown component {target!r}: "
                    f"{description[:80]}"
                )
                continue
            if target in current:
                component = current.pop(target)
                entity_id = component.attributes.get("id")
                observation_event = _observation_event(
                    component,
                    frame_records,
                    description,
                    lambda records, eid=entity_id: _visual_evidence(records, env.catalog)(eid),
                    vlm_record,
                )
                events = component.events
                window = component.event_window_us
                if observation_event is not None:
                    # a windowless component (e.g. a map element grounded at one instant)
                    # adopts the claim's span as its evidence window
                    window = window or observation_event.window_us
                    # the event may reach outside the predecessor's span (a claim whose
                    # frames sit beyond it): the successor's window must contain it
                    window = (
                        min(window[0], observation_event.window_us[0]),
                        max(window[1], observation_event.window_us[1]),
                    )
                    if "divergence" in observation_event.attributes:
                        totals["diverged"] += 1
                        orchestrator.record_parse(
                            self.name,
                            f"divergence on {component.id}: "
                            f"{observation_event.attributes['divergence']}",
                            reads=(vlm_record, *observation_event.sources[:-1]),
                        )
                    # EXPERIMENTAL dedup: a repeated claim folds into its prior event
                    # (docs/arxiv/claim_dedup_plan.md); every decision is audit-recorded.
                    coalesced_events, note = _coalesce_claim(
                        component,
                        observation_event,
                        jaccard_threshold=self.dedup_jaccard,
                        gap_us=self.dedup_gap_us,
                        key=self.dedup_key,
                    )
                    if note["near_misses"] or note.get("salient_vetoes"):
                        totals["near_miss"] = totals.get("near_miss", 0) + len(note["near_misses"])
                        orchestrator.record_parse(
                            self.name,
                            f"claim dedup audit on {component.id}: "
                            f"near-miss jaccards {note['near_misses']}, "
                            f"salient vetoes {note.get('salient_vetoes', [])}",
                            reads=(vlm_record,),
                        )
                    if coalesced_events is not None:
                        totals["coalesced"] = totals.get("coalesced", 0) + 1
                        orchestrator.record_parse(
                            self.name,
                            f"claim dedup on {component.id}: merged repeat claim "
                            f"(jaccard {note['jaccard']}) into the prior event",
                            reads=(vlm_record,),
                        )
                        if note["noop"]:
                            continue  # identical in window and content: nothing to write
                        events = coalesced_events
                    else:
                        events = _merged_events(component.events, (observation_event,))
                new_id = _enrich_id(component.id, all_ids)
                all_ids.add(new_id)
                rsg.add_component(
                    Component(
                        id=new_id,
                        category=component.category,
                        # identity only -- the claim itself lives as a tagged event
                        # (window + tier + evidence); appending it here re-introduced the
                        # meta-less duplicate the epistemic-context fix eliminated
                        label=component.label.split(" | ", 1)[0],
                        tier=component.tier,
                        provenance=Provenance(
                            op=self.name,
                            read_ids=(component.id, *frame_ids),
                            model_call=record_id,
                        ),
                        sources=tuple(
                            dict.fromkeys(
                                (
                                    *component.sources,
                                    *visual_evidence(component.attributes.get("id")),
                                    vlm_record,
                                )
                            )
                        ),
                        # the claim rides in the events below; identity stays identity
                        attributes=dict(component.attributes),
                        timestamp_us=component.timestamp_us,
                        event_window_us=window,
                        supersedes=(component.id,),
                        events=events,
                    )
                )
                totals["enriched"] += 1
                continue
            category = item.get("category")
            if category not in OBSERVABLE_CATEGORIES + CONTEXTUAL_CATEGORIES:
                # one off-vocabulary claim must not kill the run: drop it, auditable
                orchestrator.memory.write(
                    f"dropped observe claim with unknown category {category!r}: "
                    f"{str(item.get('description'))[:80]}"
                )
                continue
            record_spans = [
                span for span in (_record_span(record) for record in frame_records) if span
            ]
            span = (
                (min(s[0] for s in record_spans), max(s[1] for s in record_spans))
                if record_spans
                else None
            )
            observation_event = _observation_event(
                None,
                frame_records,
                description,
                lambda records: _visual_evidence(records, env.catalog)(None),
                vlm_record,
                category=category,
            )
            new_id = f"observed:{observed_base}"
            observed_base += 1
            all_ids.add(new_id)
            rsg.add_component(
                Component(
                    id=new_id,
                    category=category,
                    label=description,
                    tier=Tier.OBSERVABLE if category in OBSERVABLE_CATEGORIES else Tier.CONTEXTUAL,
                    provenance=Provenance(op=self.name, read_ids=frame_ids, model_call=record_id),
                    sources=visual_evidence(None) + (vlm_record,),
                    # identity only: the claim itself is the observation event below,
                    # and the label carries it for display
                    attributes=({"tracklet_id": tracklet} if tracklet else {}),
                    timestamp_us=anchor.anchor_us,
                    event_window_us=span,
                    events=(observation_event,) if observation_event is not None else (),
                )
            )
            totals["created"] += 1


class ObserveFocusOp(ObserveFramesOp):
    """Adaptive focus pass: re-observe targets the bootstrap round left unresolved.

    The orchestrator's focus policy picks the targets and windows (diverged claims,
    fragmented signals, unverifiable claims); this op executes one focused round per
    request -- the same claim-binding machinery as the bootstrap observe, with the prompt
    directed at the target. Budget-bounded by the request cap and the env budgets.
    """

    name = "observe:focus"

    def __init__(
        self,
        complete: Callable[[list[dict]], str],
        cache_dir: Path,
        max_requests: int = 4,
        scope: str = "component",
    ) -> None:
        """Bind the chat callable and the media cache; focus is frames-only."""
        if scope not in {"component", "scene"}:
            raise ValueError(f"unknown focus scope: {scope!r}")
        super().__init__(complete, cache_dir, media="frames", windows="single")
        self.max_requests = max_requests
        self.scope = scope
        self.pending_requests: list | None = None  # planner-driven loop sets, one-shot

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Execute one focused observation round per focus request.

        Requests come from the orchestrator's policy (round-robin baseline) unless
        ``pending_requests`` was set by a planner-driven loop -- the op executes either
        selector's output identically; consumed on use (one-shot).
        """
        if getattr(self, "pending_requests", None) is not None:
            requests = self.pending_requests
            self.pending_requests = None
        else:
            requests = [
                (component_id, window, reason, None)
                for component_id, window, reason in orchestrator.focus_requests(
                    rsg, anchor, self.max_requests
                )
            ]
        if requests:
            # The request list is policy output -- record it so every focused round is
            # auditable from the context layer (target, window, trigger reason).
            orchestrator.record_parse(
                self.name,
                "focus requests: "
                + "; ".join(
                    f"{component_id} [{window[0]}..{window[1]}] ({reason}"
                    + (f", {frames} frames)" if frames else ")")
                    for component_id, window, reason, frames in requests
                ),
                reads=(),
            )
        totals = {"observations": 0, "enriched": 0, "created": 0, "diverged": 0, "rounds": 0}
        for component_id, window, reason, frames in requests:
            if (self.scope == "scene") != (component_id is None):
                raise ValueError("focus scope and component target disagree")
            live = {component.id for component in rsg.components}
            if self.scope == "component" and component_id not in live:
                continue
            if frames:
                # planner-chosen sampling density: `frames` evenly spaced over the window
                count = max(1, min(frames, env.sampling.max_frames))
                span = window[1] - window[0]
                targets = [
                    window[0] + (span * index) // max(count - 1, 1) for index in range(count)
                ]
            else:
                step_us = max(int(round(1e6 / env.sampling.max_fps)), 1)
                targets = list(range(window[0], window[1] + 1, step_us))[: env.sampling.max_frames]
            overlays_by_ts = {
                record.timestamp_us: record
                for record in env.catalog.values()
                if record.stream == "derived/box_render"
            }
            frame_records = []
            cameras = (
                ("camera_front_wide_120fov",) if self.scope == "scene" else env.sampling.cameras
            )
            for camera in cameras:
                detail = {"high_detail": True} if env.sampling.image_overview_height_px else {}
                for raw in env.frames(camera, targets, self.cache_dir, **detail):
                    frame_records.append(overlays_by_ts.get(raw.timestamp_us, raw))
            totals["rounds"] += 1
            if self.scope == "scene":
                self._focus_target = (
                    "OPEN-SCENE FOCUS: inspect this bounded window for visually grounded "
                    f"evidence relevant to this attention question: {reason}. The question "
                    "is not evidence. Report only what the frames show. Enrich a listed "
                    "component when it matches; use a null component only for missing scene "
                    "evidence. Do not infer route permission, an executed maneuver, or a cause."
                )
                self._focus_only = None
                self._exclude_route_intent = True
                self._require_null_for_new_claim = True
            else:
                component = rsg.component(component_id)
                own_claim = component.id.split("#", 1)[0].startswith("observed:")
                self._focus_target = (
                    f"FOCUS: this round is specifically about [{component.id}] "
                    f"{component.label} (requested because of a {reason})."
                    + (
                        " NOTE: this component originated from your own earlier claim and is "
                        "unverified -- judge it fresh from the frames, do not restate it."
                        if own_claim
                        else ""
                    )
                    + " Report observations about it only -- its state over time, transitions, "
                    f"and anything resolving the {reason}."
                )
                # A component-focused round writes only its target: off-target claims stay in
                # the context record (the model turn persists) but never touch the graph.
                self._focus_only = component_id
            try:
                self._observe_round(rsg, env, anchor, orchestrator, frame_records, window, totals)
            finally:
                self._focus_target = None
                self._focus_only = None
                self._exclude_route_intent = False
                self._require_null_for_new_claim = False
        orchestrator.memory.write(
            f"focused {totals['rounds']} rounds on unresolved targets "
            f"({totals['enriched']} enrichments, {totals['created']} new, "
            f"{totals['diverged']} divergences)"
        )
        return OpResult()


def _resolve_tracklet(cited, known: set[str]) -> str | None:
    """Resolve a model-echoed tracklet citation against the ids perception actually emitted.

    A citation crosses a modality boundary: the id is text the model reproduces, and any
    echo can be malformed. Measured on smoke11, 5 of 38 citations arrived as bare numbers
    ("44" for "t44"). Sanitizing the id at emission cannot fix this -- the source is the
    model's echo -- so resolution is tolerant, but only where tolerance is UNAMBIGUOUS:
    an exact hit, else a unique suffix match. A citation naming several ids, or matching
    more than one, resolves to None rather than guessing, because a wrong binding is worse
    than an absent one.
    """
    if not cited:
        return None
    text = str(cited).strip()
    if text in known:
        return text
    if "," in text or " " in text.strip():
        return None  # several ids in one field: which one is not recoverable
    hits = [k for k in known if k.split(":")[-1].lstrip("t") == text.lstrip("t")]
    return hits[0] if len(hits) == 1 else None


def _resolve_cited(rsg: Rsg, cited_ids: list[str]) -> tuple[str, ...]:
    """Resolve cited component ids to their live components, following supersession.

    A citation of a superseded component (``agent:track_1`` after ``agent:track_1#observed``
    replaced it) resolves to its live successor; citations of unknown components drop.
    """
    live = [component.id for component in rsg.components]
    live_set = set(live)
    resolved = []
    for cited in cited_ids:
        if cited in live_set:
            resolved.append(cited)
            continue
        successor = next((live_id for live_id in live if live_id.startswith(f"{cited}#")), None)
        if successor is not None:
            resolved.append(successor)
    return tuple(dict.fromkeys(resolved))


def _visual_evidence(frame_records, catalog):
    """The core visual-attribution rule, per frame and per entity.

    An augmented frame (overlay) counts as evidence for a claim only where its *added
    content* grounded that claim -- i.e. the entity's box was actually drawn on it. For
    every other claim (a different entity, a map element never drawn, a new observation),
    the evidence is the underlying raw frame: the pixels carried the information, the
    augmentation was about other entities.
    """
    frame_info = []
    for record in frame_records:
        if record.stream == "derived/box_render":
            entry = catalog[record.id]
            frame_info.append(
                (record.id, entry.derived_from[0], set(entry.payload.get("tracks", ())))
            )
        else:
            frame_info.append((record.id, record.id, set()))

    def evidence(entity_id: str | None) -> tuple[str, ...]:
        return tuple(
            shown if entity_id is not None and entity_id in drawn else base
            for shown, base, drawn in frame_info
        )

    return evidence
