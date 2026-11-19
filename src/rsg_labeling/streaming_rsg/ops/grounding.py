"""Ground source-backed components, maneuver events and spatial relations."""

from __future__ import annotations

import math

from ..agent.orchestrator import Orchestrator
from ..data_structure.rsg import Component, Edge, Event, Provenance, Rsg, Tier
from ..env.env import Anchor, Env
from .base import OpResult


class GroundEgoOp:
    """Ground the ego_vehicle component from the trajectory source records."""

    name = "ground:ego"

    def __init__(self, cache_dir=None):
        """Use the run's media cache only when identity is grounded from a camera viewpoint."""
        self.cache_dir = cache_dir

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """One ego component citing every agent-visible pose reading."""
        if env.video_binding is not None:
            if self.cache_dir is None:
                raise ValueError("camera-only ego identity requires a media cache")
            frames = env.frames("camera_front_wide_120fov", [anchor.window_us[0]], self.cache_dir)
            ids = tuple(record.id for record in frames)
            rsg.add_component(
                Component(
                    id="ego",
                    category="ego_vehicle",
                    label=(
                        "ego vehicle carrying the front-wide camera; "
                        "measured kinematics unavailable"
                    ),
                    tier=Tier.OBSERVABLE,
                    provenance=Provenance(op=self.name, read_ids=ids),
                    sources=ids,
                    attributes={"measured_kinematics_available": False},
                    timestamp_us=frames[0].timestamp_us,
                    event_window_us=anchor.window_us,
                )
            )
            orchestrator.memory.write("grounded camera-carrier identity; no measured motion facts")
            return OpResult()
        records = env.ego_states("agent", anchor)
        if not records:
            raise ValueError("no ego poses inside the agent window")
        latest = records[-1]
        span_s = (records[-1].timestamp_us - records[0].timestamp_us) / 1e6
        rsg.add_component(
            Component(
                id="ego",
                category="ego_vehicle",
                # identity only -- nothing in a label may vary over the window; kinematics
                # live in events (maneuver channels) and the cited pose readings
                label=f"ego vehicle, {len(records)} pose readings over {span_s:.1f}s",
                tier=Tier.OBSERVABLE,
                provenance=Provenance(
                    op=self.name, read_ids=tuple(record.id for record in records)
                ),
                sources=tuple(record.id for record in records),
                attributes={"readings": len(records)},
                timestamp_us=latest.timestamp_us,
                event_window_us=(records[0].timestamp_us, records[-1].timestamp_us),
            )
        )
        orchestrator.memory.write(f"grounded ego from {len(records)} pose readings")
        return OpResult()


# Ego maneuver vocabulary, seeded from the v241 meta-action taxonomy: two pose-derivable
# channels (lane channel follows with lane geometry). Kinematic and tier-1 verifiable --
# distinct from *behaviors* (yield, nudge, overtake: semantic interpretations, tier 2/3,
# future work). Thresholds are deliberately coarse first guesses -- the meta-action
# reference check is what calibrates them.
MANEUVER_CHANNELS = ("longitudinal", "heading")


def _longitudinal_action(speed: float, accel: float) -> str:
    """One reading's longitudinal action (v241 names; Reverse needs signed velocity, later)."""
    if speed < 0.5:
        return "Stop"
    if abs(accel) < 0.5:
        return "MaintainSpeed"
    grade = "Strong" if abs(accel) >= 1.5 else "Gentle"
    return f"{grade}{'Acceleration' if accel > 0 else 'Deceleration'}"


def _heading_action(yaw_rate: float) -> str:
    """One reading's heading action from the yaw rate (positive = left, FLU)."""
    magnitude = abs(yaw_rate)
    if magnitude < 0.03:
        return "GoStraight"
    side = "Left" if yaw_rate > 0 else "Right"
    if magnitude < 0.12:
        return f"Steer{side}"
    if magnitude < 0.25:
        return f"SharpSteer{side}"
    return f"Turn{side}"


def ego_maneuver_events(readings) -> tuple:
    """Channelized ego maneuver events from pose readings (v241 vocabulary).

    Each channel segments independently at action-change boundaries, so the two tracks
    natively overlap -- matching the meta-action GT structure, where one moment is e.g.
    MaintainSpeed + GoStraight simultaneously.
    """
    tagged: dict[str, list] = {channel: [] for channel in MANEUVER_CHANNELS}
    for index, record in enumerate(readings):
        speed = record.payload.get("speed_mps", 0.0)
        if index + 1 < len(readings):
            nxt = readings[index + 1]
            dt = max((nxt.timestamp_us - record.timestamp_us) / 1e6, 1e-3)
            accel = (nxt.payload.get("speed_mps", 0.0) - speed) / dt
        else:
            accel = 0.0 if len(readings) < 2 else tagged["longitudinal"][-1][2]
        tagged["longitudinal"].append((record, _longitudinal_action(speed, accel), accel))
        tagged["heading"].append(
            (record, _heading_action(record.payload.get("yaw_rate_rps", 0.0)), None)
        )
    events = []
    for channel in MANEUVER_CHANNELS:
        segment: list = []
        for record, action, _ in tagged[channel]:
            if segment and action != segment[-1][1]:
                events.append(_maneuver_event(channel, segment))
                segment = []
            segment.append((record, action))
        if segment:
            events.append(_maneuver_event(channel, segment))
    events.sort(key=lambda event: event.window_us[0])
    return tuple(events)


def _maneuver_event(channel: str, segment: list) -> Event:
    """One channel event over a run of same-action readings."""
    records = [record for record, _ in segment]
    action = segment[-1][1]
    speeds = [r.payload.get("speed_mps", 0.0) for r in records]
    return Event(
        label=action,
        window_us=(records[0].timestamp_us, records[-1].timestamp_us),
        attributes={
            "channel": channel,
            "action": action,
            "speed_mps_range": [min(speeds), max(speeds)],
        },
        sources=tuple(record.id for record in records),
        tier=Tier.OBSERVABLE,
    )


class ParseManeuverOp:
    """Rule-parse channelized ego maneuver events from the trajectory kinematics.

    The finite-difference thresholds are a heuristic classification, so the result is
    ``rule_parsed`` evidence (a recorded parse), never sensor data. Two overlapping channel
    tracks (longitudinal, heading) at the v241 meta-action vocabulary; the meta-action DB
    is the eval-side reference that calibrates the thresholds, never agent input.
    Maneuvers are kinematic and OBSERVABLE -- semantic *behaviors* (yield, nudge, overtake,
    pass-by) are interpretations over maneuvers + interactions, CONTEXTUAL or LATENT by
    definition, and deliberately not emitted here.
    """

    name = "parse:maneuver"

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Supersede the ego with per-channel behavior events over the pose readings."""
        ego = next(c for c in rsg.components if c.category == "ego_vehicle")
        readings = env.ego_states("agent", anchor)
        events = ego_maneuver_events(readings)
        by_channel: dict[str, list] = {}
        for event in events:
            by_channel.setdefault(event.attributes["channel"], []).append(event.label)
        summary = "; ".join(
            f"{channel}: {' -> '.join(labels)}" for channel, labels in by_channel.items()
        )
        parse_record = orchestrator.record_parse(
            self.name,
            f"segmented ego into {len(events)} maneuver events ({summary}) "
            "from finite-difference thresholds (v241 vocabulary)",
            reads=tuple(record.id for record in readings),
        )
        rsg.add_component(
            Component(
                id=f"{ego.id}#maneuver",
                category=ego.category,
                # identity only (same rule as enrich/merge): the channelized maneuvers
                # ARE the events; a label summary would be their meta-less duplicate
                label=ego.label,
                tier=Tier.OBSERVABLE,
                provenance=Provenance(op=self.name, read_ids=(ego.id, parse_record)),
                sources=tuple(dict.fromkeys((*ego.sources, parse_record))),
                attributes=dict(ego.attributes),
                timestamp_us=ego.timestamp_us,
                event_window_us=ego.event_window_us,
                supersedes=(ego.id,),
                events=events,
            )
        )
        orchestrator.memory.write(f"parsed ego maneuver events: {summary}")
        return OpResult()


def _in_scope(records, scope) -> bool:
    """Relevance scope: any sighting within distance OR ego-TTC of the envelope.

    ``scope`` = (max_distance_m, max_ttc_s); either bound can be 0 (disabled). TTC
    uses the closing rate between consecutive sightings (approaching only). The
    union semantics keep the LARGER envelope: a far-but-fast-closing agent stays in
    via TTC, a near-but-static one via distance.
    """
    if scope is None:
        return True
    max_distance, max_ttc = scope
    distances = [
        (r.timestamp_us, r.payload.get("distance_m"))
        for r in records
        if r.payload.get("distance_m") is not None
    ]
    if max_distance > 0 and any(d <= max_distance for _, d in distances):
        return True
    if max_ttc > 0:
        for (t0, d0), (t1, d1) in zip(distances, distances[1:]):
            dt = max((t1 - t0) / 1e6, 1e-3)
            closing = (d0 - d1) / dt
            if closing > 0 and d1 / closing <= max_ttc:
                return True
    return False


# Vehicle types get the ego maneuver treatment (user direction, 2026-08-09): label
# kinematics are the meta-action SCAFFOLD until a model-based method is reasonable
# (the DA3 candidate graduates by agreement with these). VRU types stay
# observation-represented (future grounding plan).
_VRU_TYPES = ("Pedestrian", "Cyclist")


def agent_maneuver_events(records) -> tuple:
    """Channelized maneuver events for one labeled vehicle track (v241 vocabulary).

    The exact ego machinery over label kinematics: longitudinal from the speed
    series (the agent's own world speed), heading from the world-frame velocity
    direction -- both ego-independent, unlike the retired ego-relative closure
    channels. Needs >= 2 sightings; heading actions only where the direction is
    defined (near-still headings are undefined, not zero).
    """
    if len(records) < 2:
        return ()
    tagged: dict[str, list] = {"longitudinal": [], "heading": []}
    for index, record in enumerate(records):
        speed = record.payload.get("speed_mps", 0.0)
        if index + 1 < len(records):
            nxt = records[index + 1]
            dt = max((nxt.timestamp_us - record.timestamp_us) / 1e6, 1e-3)
            accel = (nxt.payload.get("speed_mps", 0.0) - speed) / dt
        else:
            accel = tagged["longitudinal"][-1][2] if tagged["longitudinal"] else 0.0
        tagged["longitudinal"].append((record, _longitudinal_action(speed, accel), accel))
        heading = record.payload.get("heading_rad")
        if heading is None:
            continue
        yaw_rate = 0.0
        if index + 1 < len(records):
            nxt_heading = records[index + 1].payload.get("heading_rad")
            if nxt_heading is not None:
                dt = max((records[index + 1].timestamp_us - record.timestamp_us) / 1e6, 1e-3)
                delta = (nxt_heading - heading + math.pi) % (2 * math.pi) - math.pi
                yaw_rate = delta / dt
        tagged["heading"].append((record, _heading_action(yaw_rate), None))
    events = []
    for channel, items in tagged.items():
        segment: list = []
        for record, action, _ in items:
            if segment and action != segment[-1][1]:
                events.append(_maneuver_event(channel, segment))
                segment = []
            segment.append((record, action))
        if segment:
            events.append(_maneuver_event(channel, segment))
    events.sort(key=lambda event: event.window_us[0])
    return tuple(events)


def _segment_state_events(records) -> tuple:
    """Segment time-ordered sighting records into events at state-change boundaries.

    The decided v1 rule: consecutive sightings with the same ``state`` fold into one event;
    a change opens a new one. Speed and distance trends live as per-event range attributes
    (no change-point heuristic). Each event cites exactly its in-window sightings, so the
    validator's timestamp-containment check holds by construction.
    """
    events = []
    segment: list = []
    for record in records:
        if segment and record.payload.get("state") != segment[-1].payload.get("state"):
            events.append(_event_from_segment(segment))
            segment = []
        segment.append(record)
    if segment:
        events.append(_event_from_segment(segment))
    return tuple(events)


def _event_from_segment(segment: list) -> Event:
    """One event over a run of same-state sightings, with range attributes."""
    state = segment[-1].payload.get("state")
    speeds = [r.payload["speed_mps"] for r in segment if "speed_mps" in r.payload]
    distances = [r.payload["distance_m"] for r in segment if "distance_m" in r.payload]
    attributes: dict = {"state": state, "sightings": len(segment)}
    if speeds:
        attributes["speed_mps_range"] = [min(speeds), max(speeds)]
    if distances:
        attributes["distance_m_start"] = distances[0]
        attributes["distance_m_end"] = distances[-1]
    last = segment[-1].payload
    for key in ("forward_m", "lateral_m"):
        if key in last:
            attributes[f"{key}_end"] = last[key]
    return Event(
        label=str(state),
        window_us=(segment[0].timestamp_us, segment[-1].timestamp_us),
        attributes=attributes,
        sources=tuple(record.id for record in segment),
        tier=Tier.OBSERVABLE,
    )


class GroundAgentsOp:
    """Ground one dynamic_agent component per labeled track, with state events.

    Identity (id, type) is time-invariant and stays flat; the time-varying state attributes
    are segmented into events (event_attributes plan, completed -- every consumer reads
    events now). State events and the longitudinal maneuver channel overlap in meaning;
    consolidation is a queued cleanup, not a migration in flight.
    """

    name = "ground:agents"

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Group label sightings by track; one component per entity, events per state run."""
        if "object_labels" not in env.sources:
            orchestrator.memory.write("no object_labels stream in this env; agents unground")
            return OpResult()
        sightings: dict[str, list] = {}
        for ego_record in env.ego_states("agent", anchor):
            for record in env.labels(anchor, ego_record.timestamp_us):
                sightings.setdefault(record.payload["id"], []).append(record)
        total_events = 0
        skipped_scope = 0
        for track_id, records in sightings.items():
            records.sort(key=lambda record: record.timestamp_us)
            if not _in_scope(records, env.scope):
                skipped_scope += 1
                continue
            latest = records[-1]
            maneuvers = (
                agent_maneuver_events(records)
                if latest.payload.get("type") not in _VRU_TYPES
                else ()
            )
            events = tuple(
                sorted(
                    (*_segment_state_events(records), *maneuvers),
                    key=lambda event: event.window_us[0],
                )
            )
            total_events += len(events)
            parse_record = orchestrator.record_parse(
                self.name,
                f"segmented {track_id} into {len(events)} state events: "
                + ", ".join(
                    f"{e.label}[{e.attributes.get('sightings', e.attributes.get('channel', '?'))}]"
                    for e in events
                ),
                reads=tuple(record.id for record in records),
            )
            span = (
                (records[0].timestamp_us - anchor.anchor_us) / 1e6,
                (records[-1].timestamp_us - anchor.anchor_us) / 1e6,
            )
            rsg.add_component(
                Component(
                    id=f"agent:{track_id}",
                    category="dynamic_agent",
                    # identity only (user-caught defect: the old latest-sighting summary
                    # showed dynamics at one fixed timestamp); dynamics live in events
                    label=(
                        f"{latest.payload.get('type', 'agent')} ({track_id}), "
                        f"{len(records)} sightings {span[0]:+.1f}..{span[1]:+.1f}s"
                    ),
                    tier=Tier.OBSERVABLE,
                    provenance=Provenance(
                        op=self.name,
                        read_ids=(*(record.id for record in records), parse_record),
                    ),
                    sources=tuple(record.id for record in records),
                    # identity only (three_layer_plan): the sighting COUNT is evidence
                    # and is already recoverable from the events below
                    attributes={
                        "id": latest.payload.get("id"),
                        "type": latest.payload.get("type"),
                    },
                    timestamp_us=latest.timestamp_us,
                    event_window_us=(records[0].timestamp_us, records[-1].timestamp_us),
                    events=events,
                )
            )
        orchestrator.memory.write(
            f"grounded {len(sightings) - skipped_scope} agents ({total_events} state events"
            + (f"; {skipped_scope} out of scope" if skipped_scope else "")
            + ")"
        )
        return OpResult()


class GroundMapOp:
    """Ground static_road_element components from the map source records at the anchor."""

    name = "ground:map"

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """One component per map zone near the anchor."""
        if "map" not in env.sources:
            orchestrator.memory.write("no map stream in this env; map elements unground")
            return OpResult()
        records = env.map_elements(anchor, anchor.anchor_us)
        for record in records:
            rsg.add_component(
                Component(
                    id=f"element:{record.payload['id']}",
                    category="static_road_element",
                    # identity only: the ego-relative distance is anchor-instant geometry
                    # (it stays in attributes with timestamp_us as its reference; the
                    # spatial edge carries it with a window)
                    label=f"{record.payload.get('type', 'element')} ({record.payload['id']})",
                    tier=Tier.OBSERVABLE,
                    provenance=Provenance(op=self.name, read_ids=(record.id,)),
                    sources=(record.id,),
                    attributes=dict(record.payload),
                    timestamp_us=record.timestamp_us,
                )
            )
        orchestrator.memory.write(f"grounded {len(records)} map elements")
        return OpResult()


def _anchor_event(component, anchor_us: int):
    """The component's geometry-bearing OBSERVABLE event at the anchor, else its latest.

    Filters to events carrying distance geometry so non-positional events (claims)
    never starve the spatial relate.
    """
    observed = [
        event
        for event in component.events
        if event.tier is Tier.OBSERVABLE and "distance_m_end" in event.attributes
    ]
    for event in observed:
        if event.window_us[0] <= anchor_us <= event.window_us[1]:
            return event
    return observed[-1] if observed else None


class RelateSpatialOp:
    """Author spatial and motion-interaction edges from rig-frame geometry.

    Geometry is read from the event overlapping the anchor (never a stale flat snapshot);
    the edge carries that event's window, so the relation is scoped to the time its
    evidence actually covers.
    """

    name = "relate:spatial"

    def run(self, rsg: Rsg, env: Env, anchor: Anchor, orchestrator: Orchestrator) -> OpResult:
        """Relate every surviving agent/element to the ego by its anchor-time position."""
        ego = next(c for c in rsg.components if c.category == "ego_vehicle")
        related = 0
        for component in rsg.components:
            if component.id == ego.id:
                continue
            event = _anchor_event(component, anchor.anchor_us)
            # events are the only dynamics source (flat snapshots retired); static map
            # elements keep a flat distance -- their geometry is not event-segmented
            distance = (
                event.attributes.get("distance_m_end")
                if event is not None
                else component.attributes.get("distance_m")
                if component.category == "static_road_element"
                else None
            )
            if distance is None:
                continue
            state = event.attributes.get("state") if event is not None else None
            forward = event.attributes.get("forward_m_end") if event is not None else None
            lateral = event.attributes.get("lateral_m_end") if event is not None else None
            position = (
                f"{distance}m away"
                if forward is None
                else f"forward {forward}m, lateral {lateral}m ({distance}m)"
            ) + (f", {state}" if state else "")
            window = event.window_us if event is not None else None
            rsg.add_edge(
                Edge(
                    source=component.id,
                    target=ego.id,
                    relation="spatial",
                    rationale=f"rig-frame position at the anchor: {position}",
                    provenance=Provenance(op=self.name, read_ids=(component.id, ego.id)),
                    attributes={"distance_m": distance},
                    window_us=window,
                )
            )
            related += 1
            if state == "moving" and forward is not None and forward > 0 and abs(lateral) < 3.0:
                rsg.add_edge(
                    Edge(
                        source=component.id,
                        target=ego.id,
                        relation="motion_interaction",
                        rationale=f"moving inside the ego forward corridor ({position})",
                        provenance=Provenance(op=self.name, read_ids=(component.id, ego.id)),
                        window_us=window,
                    )
                )
        orchestrator.memory.write(f"related {related} components to the ego by geometry")
        return OpResult()
