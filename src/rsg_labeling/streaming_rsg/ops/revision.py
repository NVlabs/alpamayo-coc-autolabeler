"""Associate, revise, merge and retire graph components."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..agent.orchestrator import Orchestrator
from ..data_structure.rsg import Component, Event, Provenance, Rsg, SourceClass, Tier
from ..env.env import Anchor, Env, SourceRecord
from .base import OpResult
from .claims import _merged_events, _rescoped_events


class MergeAssociateOp:
    """Cross-source identity: associate VLM-observed agents with labeled tracks by geometry.

    The observe op grounds claims the labels missed as ``observed:*`` components; when such a
    claim is actually a labeled entity the model failed to echo, the two views of one entity
    sit unlinked. This op closes that gap without trusting either side's text: the external
    grounding model localizes the claim on the anchor frame, the env's calibrated projector
    localizes every labeled track on the same frame, and a box-IoU match merges the pair into
    one component superseding both (union of sources, so neither chain's evidence drops).
    """

    name = "merge:associate"

    def __init__(
        self,
        detect: Callable[[str, str], list[dict]],
        cache_dir: Path,
        iou_threshold: float = 0.3,
    ) -> None:
        """Bind the structured grounding callable (``ground_detections``) and the frame cache."""
        self.detect = detect
        self.cache_dir = cache_dir
        self.iou_threshold = iou_threshold

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Ground each observed agent on the anchor frame; IoU-match against projected tracks."""
        if "object_labels" not in env.sources:
            orchestrator.memory.write("no object_labels stream; nothing to associate against")
            return OpResult()
        observed = [
            component
            for component in rsg.components
            if component.id.startswith("observed:") and component.category == "dynamic_agent"
        ]
        if not observed:
            orchestrator.memory.write("no observed agents to associate")
            return OpResult()
        camera = env.sampling.cameras[0]
        frames = env.frames(camera, [anchor.anchor_us], self.cache_dir)
        if not frames:
            orchestrator.memory.write("no anchor frame to associate on")
            return OpResult()
        frame = frames[-1]
        import cv2

        height, width = cv2.imread(frame.path).shape[:2]
        gt_boxes = env.projector().boxes_for(camera, frame.timestamp_us, width, height)
        agents_by_track = {
            component.attributes.get("id"): component
            for component in rsg.components
            if component.category == "dynamic_agent"
            and not component.id.startswith("observed:")
            and component.attributes.get("id") is not None
        }
        merged = 0
        for component in observed:
            prompt = _grounding_phrase(claim_text(component))
            try:
                detections, tool_record = orchestrator.call_tool(
                    self.name,
                    "ground_frame",
                    prompt,
                    lambda: self.detect(frame.path, prompt),
                    reads=(frame.id,),
                    summarize=lambda found: (
                        "\n".join(
                            f"box_xyxy_norm={[round(v, 3) for v in d['box_xyxy_norm']]} "
                            f"score={round(d.get('score', 0.0), 3)}"
                            for d in found
                        )
                        or f"no detections for {prompt!r}"
                    ),
                )
            except RuntimeError as error:
                orchestrator.memory.write(f"grounding unavailable, association skipped: {error}")
                return OpResult()
            best_iou, best_track = 0.0, None
            for detection in detections:
                x0, y0, x1, y1 = detection["box_xyxy_norm"]
                detected = (x0 * width, y0 * height, x1 * width, y1 * height)
                for gt_box in gt_boxes:
                    iou = _box_iou(detected, (gt_box.x0, gt_box.y0, gt_box.x1, gt_box.y1))
                    if iou > best_iou:
                        best_iou, best_track = iou, gt_box.track_id
            agent = agents_by_track.get(best_track)
            if best_iou < self.iou_threshold or agent is None:
                continue
            agents_by_track.pop(best_track)
            observation = claim_text(component)
            rsg.add_component(
                Component(
                    id=f"{agent.id}#assoc",
                    category=agent.category,
                    # identity only (same rule as enrichment): the associated claim
                    # lives on as merged events; the merge itself is in provenance
                    label=agent.label.split(" | ", 1)[0],
                    tier=agent.tier,
                    provenance=Provenance(
                        op=self.name,
                        read_ids=(agent.id, component.id, frame.id, tool_record),
                    ),
                    sources=tuple(dict.fromkeys((*agent.sources, *component.sources, tool_record))),
                    attributes={
                        **agent.attributes,
                        "observation": observation,
                        "association_iou": round(best_iou, 2),
                    },
                    timestamp_us=agent.timestamp_us,
                    event_window_us=agent.event_window_us,
                    supersedes=(agent.id, component.id),
                    events=_merged_events(
                        agent.events,
                        _rescoped_events(
                            component.events, agent.events, agent.event_window_us, env.catalog
                        ),
                    ),
                )
            )
            merged += 1
        orchestrator.memory.write(
            f"associated {merged}/{len(observed)} observed agents with labeled tracks"
        )
        return OpResult()


# Claim categories the tracklet merge folds. Excludes ego_vehicle (one per run) and
# environmental_context (scene-level, not object-scoped).
_CLAIM_CATEGORIES = ("dynamic_agent", "static_road_element", "traffic_control")


def claim_text(component) -> str:
    """The component's current claim, read from its events (three_layer_plan phase 2).

    Claims are the TEXT modality and live in events; ``attributes`` carries identity. The
    latest observation event wins, because a later round revising a claim supersedes the
    earlier wording. Falls back to the label, which for a model-created component IS the
    claim -- an honest fallback rather than a hidden dependency on the attribute copy this
    phase removed.
    """
    for event in reversed(component.events or ()):
        text = (event.attributes or {}).get("observation")
        if text:
            return str(text)
    return component.label


class ReviseTrackletsOp:
    """The agent's reflect step: break a tracklet the evidence says is not one object.

    The perception stack proposes a tracklet; the geometry tag says whether its sightings
    can be one object (a POSITION BREAK is a step fast relative to that tracklet's own
    motion). This op asks the agent to adjudicate the flagged ones and records the verdict,
    so a wrong correspondence is caught BEFORE claims are folded onto it.

    The question is put as single-object evidence, per the plan's gate: the judge is shown
    ONE crop and asked whether it matches a description taken from the other side of the
    break. Comparing two crops directly would need a capability we have not validated;
    describe-then-check reuses the appearance judge that already adjudicates colour and
    type, and abstains the same way.

    A split is a CLAIM about the reference, not a rewrite of it: env records are immutable
    and the tracklet id stays as perception issued it. The verdict is registered as a
    derived record that merge:tracklets consults, so claims are never folded across a
    boundary the agent rejected.
    """

    name = "revise:tracklets"

    def __init__(self, judge_appearance=None, mode: str = "judged"):
        """Bind the adjudicator and the revision policy (``judged`` | ``off``).

        No judge still disables the op: without an adjudicator a split would rest on the
        geometry alone, and the whole point is that the agent decides. ``off`` is the
        searchable form of the same choice -- the flag is still recorded, nothing splits.
        """
        self.judge_appearance = judge_appearance
        self.mode = mode

    def _sightings(self, env: Env, track_id: str):
        """[(timestamp, frame_path, box)] for one tracklet, in time order."""
        out = []
        for record in env.catalog.values():
            if not (record.stream or "").startswith("reference/") or not record.derived_from:
                continue
            frame = env.catalog.get(record.derived_from[0])
            if frame is None or not frame.path:
                continue
            for obj in (record.payload or {}).get("objects", []) or []:
                if str(obj.get("track_id")) == str(track_id) and obj.get("box_xyxy_norm"):
                    out.append((record.timestamp_us, frame.path, obj["box_xyxy_norm"]))
        return sorted(out)

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Adjudicate every tracklet the geometry flagged; register the split verdicts."""
        from ..agent.orchestrator import tracklet_tags

        if self.mode == "off":
            orchestrator.memory.write("tracklet revision disabled by policy.revise.mode=off")
            return OpResult()
        if self.judge_appearance is None:
            orchestrator.memory.write("tracklet revision disabled: no appearance judge bound")
            return OpResult()
        tags = tracklet_tags(env, anchor.anchor_us)
        flagged = {tid: tag for tid, tag in tags.items() if tag["breaks"]}
        split_count = kept = unchecked = components_split = 0
        for track_id, tag in sorted(flagged.items()):
            sightings = self._sightings(env, track_id)
            if len(sightings) < 2:
                unchecked += 1
                continue
            boundary_us = anchor.anchor_us + int(tag["breaks"][0] * 1e6)
            before = [s for s in sightings if s[0] < boundary_us]
            after = [s for s in sightings if s[0] >= boundary_us]
            if not before or not after:
                unchecked += 1
                continue
            try:
                # describe the LAST sighting before the break, then ask whether the first
                # sighting after it still matches -- one crop per question, either way
                _, description = self.judge_appearance(before[-1][1], before[-1][2], "describe it")
                matched, actual = self.judge_appearance(after[0][1], after[0][2], description)
            except RuntimeError as error:
                unchecked += 1
                orchestrator.record_parse(
                    self.name,
                    f"tracklet {track_id}: judge unreachable ({type(error).__name__})",
                    reads=(),
                )
                continue
            if matched:
                kept += 1
                continue
            split_count += 1
            # registered BEFORE the components are separated: the split event on each
            # successor cites this record, so the claim "these are two objects" points at
            # the evidence that decided it rather than restating it in attributes
            split_record = env.register_derived(
                SourceRecord(
                    id=f"env:tracklet_split:{track_id}",
                    stream="revision/tracklet_split",
                    source_class=SourceClass.TOOL_OUTPUT,
                    summary=(
                        f"tracklet {track_id} split at {tag['breaks'][0]:+.1f}s: "
                        f"{description[:40]!r} -> {actual[:40]!r}"
                    ),
                    timestamp_us=boundary_us,
                    payload={
                        "track_id": track_id,
                        "boundary_us": boundary_us,
                        "before": description,
                        "after": actual,
                        "max_step_m": tag["max_step_m"],
                    },
                    derived_from=(
                        *(
                            frame.id
                            for frame in env.catalog.values()
                            if frame.path in (before[-1][1], after[0][1])
                        ),
                    ),
                )
            )
            components_split += self._split_components(
                rsg, track_id, boundary_us, description, actual, split_record
            )
        orchestrator.memory.write(
            f"tracklet revision: {split_count} split, {kept} held together, "
            f"{unchecked} unchecked, over {len(flagged)} geometry-flagged tracklets; "
            f"{components_split} components separated across a break"
        )
        return OpResult()

    def _split_components(
        self,
        rsg: Rsg,
        track_id: str,
        boundary_us: int,
        before: str,
        after: str,
        split_record: SourceRecord,
    ) -> int:
        """Separate every live component whose claims straddle this tracklet's break.

        A split verdict used to be recorded and consulted later, which is enough while each
        round mints its own component. Once claims BIND to the component a tracklet already
        has, the verdict has to be able to take them apart again -- otherwise a drifting
        tracklet fuses two objects into one component and nothing can separate them, which
        trades a visible defect for an invisible one.

        Successors keep the parent's id lineage (``#split0``/``#split1``) so an audit can
        follow a component back through its splits, and both inherit the parent's full
        sources: evidence never disappears through supersession (invariant 3). A claim whose
        window SPANS the break goes to neither successor -- assigning it would be a guess,
        and a guess here is the conflation this exists to prevent -- so it retires with the
        superseded parent, marked for the reader.
        """
        separated = 0
        for component in list(rsg.components):
            attributes = component.attributes or {}
            tracklets = {str(t) for t in attributes.get("track_ids", ())}
            if attributes.get("tracklet_id"):
                tracklets.add(str(attributes["tracklet_id"]))
            if str(track_id) not in tracklets or not component.events:
                continue
            sides: dict[str, list] = {"split0": [], "split1": []}
            spanning = []
            for event in component.events:
                start, end = event.window_us
                if end < boundary_us:
                    sides["split0"].append(event)
                elif start >= boundary_us:
                    sides["split1"].append(event)
                else:
                    spanning.append(event)
            if not (sides["split0"] and sides["split1"]):
                continue  # nothing straddles the break: the component is already one side
            for suffix, events in sides.items():
                windows = [w for event in events for w in event.window_us]
                rsg.add_component(
                    Component(
                        id=f"{component.id}#{suffix}",
                        category=component.category,
                        label=component.label.split(" | ", 1)[0],
                        tier=component.tier,
                        provenance=Provenance(
                            op=self.name,
                            read_ids=(component.id,),
                        ),
                        # the parent's whole source set, not this side's share: a successor
                        # that dropped evidence would break the append-only chain
                        sources=tuple(component.sources),
                        # identity only (three_layer_plan): which successor this is,
                        # and nothing else. WHY the split happened is a judged claim about
                        # the referent, so it rides in the event below where a reader can
                        # see its tier and its source.
                        attributes={
                            **attributes,
                            "split_side": "before" if suffix == "split0" else "after",
                        },
                        timestamp_us=component.timestamp_us,
                        event_window_us=(min(windows), max(windows)),
                        supersedes=(component.id,),
                        # FIRST, not appended: the revision spans the successor's whole
                        # window, so it starts at or before every claim it separated, and
                        # the validator requires events ordered by start
                        events=(
                            Event(
                                label=(f"split: {before[:40]!r} -> {after[:40]!r}"),
                                window_us=(min(windows), max(windows)),
                                attributes={
                                    "split_at_us": boundary_us,
                                    "split_reason": f"{before[:40]!r} -> {after[:40]!r}",
                                    # the claims that straddled the break went to NEITHER
                                    # successor and retired with the parent; the count
                                    # belongs on the revision that retired them, because
                                    # there is no surviving event to carry it
                                    **(
                                        {"unresolved_span": len(spanning)}
                                        if spanning and suffix == "split0"
                                        else {}
                                    ),
                                },
                                sources=(split_record.id,),
                                tier=Tier.CONTEXTUAL,
                            ),
                            *events,
                        ),
                    )
                )
            separated += 1
        return separated


class MergeTrackletsOp:
    """Fold the observed claims that cite ONE tracklet into one object-centric component.

    Applies to every claim category (agents, road elements, traffic control), because the
    fragmentation is a property of how observe rounds mint components, not of what the
    claim is about.

    Each observe round mints a fresh ``observed:N`` per observation, so one physical car
    described across nineteen rounds becomes nineteen components with two events each --
    a timeline of fragments where there is one object. Measured on smoke5: 60 dynamic-agent
    components carrying only 24 distinct tracklet ids, worst case 11 components for
    ``vehicle:8``.

    The tracklet id is the merge KEY but never the merge WARRANT: a drifting tracklet
    would fuse claims about genuinely different objects, and that error is quieter than
    the wording-driven over-merge the text dedup already risks. So every merge is gated on
    the appearance judge -- each claim is checked against the tracklet's own crop, and only
    claims the judge MATCHES to that crop join. A diverging claim keeps its component and
    says why; a judge outage merges nothing (abstention is cheaper than a wrong binding).

    Complements ``_coalesce_claim``, which folds repeated events WITHIN one component and
    therefore cannot address fragmentation ACROSS components.
    """

    name = "merge:tracklets"

    def __init__(self, judge_appearance=None, mode: str = "judged"):
        """Bind the adjudicator and the merge policy (``judged`` | ``trusted`` | ``off``).

        ``judged`` gates every merge on the appearance judge, as the plan requires. The
        other two exist because the gate has a measured cost -- 41% of tracklets still
        carry several components under it -- and the benchmark now scores both sides of
        the trade (single_component_rate against distinct_recall), so which gate is right
        is a question the optimizer can answer instead of a standing assumption.
        ``trusted`` takes the tracklet id as warrant; ``off`` merges nothing. With no judge
        bound, ``judged`` merges nothing -- an unadjudicated merge is not the fallback.
        """
        self.judge_appearance = judge_appearance
        self.mode = mode

    def _tracklet_crop(self, env: Env, tracklet_id: str):
        """The frame path and box where this tracklet is largest (most legible crop)."""
        best = None
        for record in env.catalog.values():
            if not (record.stream or "").startswith("reference/") or not record.derived_from:
                continue
            for obj in (record.payload or {}).get("objects", []) or []:
                if str(obj.get("track_id")) != str(tracklet_id) or not obj.get("box_xyxy_norm"):
                    continue
                x0, y0, x1, y1 = obj["box_xyxy_norm"]
                area = max(x1 - x0, 0) * max(y1 - y0, 0)
                frame = env.catalog.get(record.derived_from[0])
                if frame is None or not frame.path:
                    continue
                if best is None or area > best[0]:
                    best = (area, frame.path, obj["box_xyxy_norm"])
        return (best[1], best[2]) if best else (None, None)

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Group observed agents by cited tracklet; merge the judge-confirmed ones."""
        if self.mode == "off":
            orchestrator.memory.write("tracklet merge disabled by policy.merge.mode=off")
            return OpResult()
        if self.mode == "judged" and self.judge_appearance is None:
            orchestrator.memory.write("tracklet merge disabled: no appearance judge bound")
            return OpResult()
        # every CLAIM category, not agents only: a crosswalk described in nine rounds
        # fragments exactly as a vehicle does, and gating on dynamic_agent left the static
        # claims unfolded (measured smoke10: 28 static claims, 21 citing a tracklet, none
        # merged). Geometric `element:*` components are excluded -- they are anchored
        # objects, not claims, and binding a claim ONTO one is a different operation.
        groups: dict[str, list] = {}
        for component in rsg.components:
            if component.category not in _CLAIM_CATEGORIES:
                continue
            if component.id.split("#", 1)[0].startswith("element:"):
                continue
            tracklet = component.attributes.get("tracklet_id")
            if tracklet:
                groups.setdefault(str(tracklet), []).append(component)
        # A tracklet the agent SPLIT is not one referent, so its claims must not fold
        # across the boundary -- otherwise revise:tracklets is decoration.
        splits = {
            str((record.payload or {}).get("track_id")): (record.payload or {}).get("boundary_us")
            for record in env.catalog.values()
            if (record.stream or "") == "revision/tracklet_split"
        }
        candidates = {k: v for k, v in groups.items() if len(v) > 1}
        merged_groups = diverged = unchecked = split_blocked = 0
        for tracklet, members in sorted(candidates.items()):
            boundary = splits.get(str(tracklet))
            if boundary is not None:
                # Partition by WHEN THE CLAIM IS ABOUT, not by component.timestamp_us:
                # observe stamps every created component with the ANCHOR time, so that
                # field is constant across claims and both sides of any boundary landed
                # in one bucket -- which is why 198 splits blocked 0 folds on the 50-clip
                # run despite 22 of 79 split tracklets having >=2 claims. The claim's
                # time lives in its events (three_layer_plan).
                def claim_start(component):
                    windows = [
                        event.window_us[0]
                        for event in component.events
                        if (event.attributes or {}).get("observation")
                    ]
                    return min(windows) if windows else (component.event_window_us or (0, 0))[0]

                before = [c for c in members if claim_start(c) < boundary]
                after = [c for c in members if claim_start(c) >= boundary]
                chosen = before if len(before) >= len(after) else after
                if len(chosen) < len(members):
                    split_blocked += len(members) - len(chosen)
                members = chosen
                if len(members) < 2:
                    continue
            frame_path, box = self._tracklet_crop(env, tracklet)
            if frame_path is None:
                unchecked += len(members)
                continue
            confirmed, rejected = [], []
            for component in members:
                claim = claim_text(component)
                try:
                    matched, actual = self.judge_appearance(frame_path, box, claim)
                except RuntimeError as error:
                    orchestrator.record_parse(
                        self.name,
                        f"tracklet {tracklet}: appearance judge unreachable "
                        f"({type(error).__name__})",
                        reads=(component.id,),
                    )
                    confirmed, rejected = [], []
                    unchecked += len(members)
                    break
                (confirmed if matched else rejected).append((component, actual))
            if len(confirmed) < 2:
                diverged += len(rejected)
                continue
            keep = confirmed[0][0]
            ids = tuple(component.id for component, _ in confirmed)
            events = tuple(
                sorted(
                    (event for component, _ in confirmed for event in component.events),
                    key=lambda event: event.window_us[0],
                )
            )
            if not events:
                continue  # nothing to carry: the fold would assert an empty component
            window = (
                min(event.window_us[0] for event in events),
                max(event.window_us[1] for event in events),
            )
            rsg.add_component(
                Component(
                    id=f"{keep.id.split('#', 1)[0]}#trk",
                    category=keep.category,
                    label=keep.label,
                    tier=keep.tier,
                    provenance=Provenance(op=self.name, read_ids=ids),
                    # invariant 3: the successor inherits every predecessor's evidence
                    sources=tuple(
                        dict.fromkeys(
                            source for component, _ in confirmed for source in component.sources
                        )
                    ),
                    # identity only (three_layer_plan): the fold count is recoverable
                    # from `supersedes`, and what the judge SAW is a claim -- both ride
                    # in the merge event below
                    attributes={**keep.attributes, "tracklet_id": tracklet},
                    timestamp_us=keep.timestamp_us,
                    event_window_us=window,
                    supersedes=ids,
                    # the merge event spans the whole window, so it sorts FIRST, not
                    # last: events must stay ordered by start (validator-enforced)
                    events=tuple(
                        sorted(
                            (
                                *events,
                                Event(
                                    label=f"tracklet merge: {len(confirmed)} claims",
                                    window_us=window,
                                    attributes={
                                        "merged_claims": len(confirmed),
                                        "tracklet_appearance": confirmed[0][1],
                                        "tracklet_id": tracklet,
                                    },
                                    sources=(),
                                    tier=Tier.CONTEXTUAL,
                                ),
                            ),
                            key=lambda event: event.window_us[0],
                        )
                    ),
                )
            )
            merged_groups += 1
            diverged += len(rejected)
        orchestrator.memory.write(
            f"tracklet merge: {merged_groups} groups folded over {len(candidates)} "
            f"multi-claim tracklets ({diverged} claims kept apart by the judge, "
            f"{split_blocked} by a tracklet split, {unchecked} unchecked)"
        )
        self._report_residue(rsg, env, anchor, orchestrator)
        return OpResult()

    def _report_residue(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator):
        """Count the references perception offered that no claim took up.

        three_layer_plan Q2, option (b): an unclaimed reference does NOT become a
        component -- the graph is the agent's world model, and a referent it never engaged
        is context, not knowledge. But the count is reported prominently, because it is
        the measure of what the agent is BLIND to, and it is the number that would justify
        materializing them later.
        """
        from ..agent.orchestrator import tracklet_tags

        offered: dict[str, int] = {}
        for record in env.catalog.values():
            if not (record.stream or "").startswith("reference/"):
                continue
            for obj in (record.payload or {}).get("objects", []) or []:
                if obj.get("track_id"):
                    offered[str(obj["track_id"])] = offered.get(str(obj["track_id"]), 0) + 1
        if not offered:
            return
        cited = {
            str(component.attributes["tracklet_id"])
            for component in rsg.components
            if component.attributes.get("tracklet_id")
        }
        cited |= {
            str(track)
            for component in rsg.components
            for track in component.attributes.get("track_ids", ())
        }
        residue = {track: n for track, n in offered.items() if track not in cited}
        if not residue:
            orchestrator.memory.write(f"residue: 0 of {len(offered)} references unclaimed")
            return
        tags = tracklet_tags(env, anchor.anchor_us)
        ranked = sorted(residue.items(), key=lambda kv: -kv[1])[:5]
        detail = ", ".join(
            f"{track}({n} sightings"
            + (
                f", {tags[track]['range_start_m']:.0f}->{tags[track]['range_end_m']:.0f}m)"
                if track in tags
                and isinstance(tags[track].get("range_start_m"), int | float)
                and isinstance(tags[track].get("range_end_m"), int | float)
                else ")"
            )
            for track, n in ranked
        )
        orchestrator.memory.write(
            f"residue: {len(residue)} of {len(offered)} references unclaimed "
            f"({sum(residue.values())} sightings the agent never spoke to) -- {detail}"
        )
        orchestrator.record_parse(
            self.name,
            f"unclaimed references: {sorted(residue)}",
            reads=(),
        )


class PruneUnobservedOp:
    """Prune dynamic agents no visual observation corroborates.

    "Unobserved", not "ungrounded": these agents *are* grounded — in external label records —
    but no observe/merge op bound visual evidence to them, so they are retired as decision
    context. The prune verdict is about observation coverage, not verifiability.
    """

    name = "prune:unobserved"

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Retire agents no observation EVENT corroborates; the reason is recorded."""
        pruned = tuple(
            component.id
            for component in rsg.components
            # The claim lives in events now (three_layer_plan), so corroboration reads
            # from there. The retired `observation` attribute meant "observe touched
            # this", which is NOT the same as "an observation event exists" -- an
            # enrichment whose window is degenerate produces no event. Provenance carries
            # that fact without putting claim material back in identity.
            if component.category == "dynamic_agent"
            and not any((event.attributes or {}).get("observation") for event in component.events)
            and component.provenance.op != "observe:frames"
        )
        if pruned:
            rsg.prune(pruned, self.name, "no visual observation corroborates these agents")
        orchestrator.memory.write(f"pruned {len(pruned)} unobserved agents")
        return OpResult()


def _grounding_phrase(observation: str) -> str:
    """The subject noun phrase of an observation, as a grounding prompt.

    Grounding models localize short referring phrases ("white van"), not narrative sentences;
    a full VLM observation returns zero detections. Cut at the first copula/clause boundary
    and drop the leading article.
    """
    import re

    phrase = re.split(r"\s+(?:is|are|was|were|appears|seems)\s+|[,.;]", observation, 1)[0]
    phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase.strip(), flags=re.IGNORECASE)
    return phrase or observation


def _box_iou(a: tuple, b: tuple) -> float:
    """Intersection-over-union of two (x0, y0, x1, y1) pixel boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if intersection == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)
