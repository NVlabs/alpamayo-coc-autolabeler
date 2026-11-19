"""Construct, coalesce and rescope observation claim events."""

from __future__ import annotations

import re

from ..data_structure.rsg import Event, Tier

# Motion-state vocabulary for claim/label divergence checks: a VLM claim mentioning one
# side's words contradicts an overlapping label event carrying the other side's state.
_CLAIM_STATE_WORDS = {
    "stopped": ("stationary", "stopped", "parked", "standing", "halted", "waiting", "idle"),
    "moving": ("moving", "crossing", "driving", "walking", "running", "accelerating", "turning"),
}


# Signal PHASE vocabulary: the checkable state set for traffic_control claims (motion
# states never applied there -- the category guard). Phases structure claim events
# ("observed: red") and keep dedup from flattening a phase CHANGE into one claim;
# adjudication stays honestly unverifiable until a phase reference exists (claims never
# verify claims).
# Scope (review, 2026-08-04): solid circular phases only. Flashing modes and arrow
# phases are IN scope as physical lamp states but need compound-token matching -- the
# first test case for the queued claim_state flexible matcher, not more substring
# lists. What an arrow PERMITS is normative-layer content, never a claim state.
# Vehicle indicators (blinkers/brake lights) are a dynamic_agent channel, not phases.
_CLAIM_PHASE_WORDS = {
    "red": ("red",),
    "green": ("green",),
    "yellow": ("yellow", "amber"),
}


# Position (side) vocabulary: ego-path frame, v1 = left/right only (the observed
# hallucination class: "left lane" claimed for a right-side car). Adjudicated against
# the host's rig-frame lateral geometry (FLU: positive lateral = left); divergence-only
# in v1 -- catching contradictions is the goal, confirmation upgrades wait for
# calibration. Same single-hit ambiguity rule as the other vocabularies.
_CLAIM_SIDE_WORDS = {
    "left": ("left",),
    "right": ("right",),
}


_SIDE_MIN_LATERAL_M = 1.0  # |lateral| below this = center; side claims unadjudicable


def _claim_state(text: str, vocabulary: dict | None = None) -> str | None:
    """Return one lexical state, abstaining on ambiguity or a qualified state clause.

    Clause splitting is a conservative heuristic, not subject-aware entailment. An
    unrelated negative clause must not erase a positive phase/motion assertion. Category
    guards still constrain use; a hint never establishes normative permission.
    """
    lowered = text.lower()
    patterns = {
        state: re.compile(r"\b(?:" + "|".join(re.escape(word) for word in words) + r")\b")
        for state, words in (vocabulary or _CLAIM_STATE_WORDS).items()
    }
    hits = [state for state, pattern in patterns.items() if pattern.search(lowered)]
    if len(hits) != 1:
        return None
    state = hits[0]
    # A rejected alternative is not an asserted state. Keep the raw claim and abstain;
    # do not infer an opposite state from an expanding list of motion synonyms.
    for alternative in re.finditer(r"\b(?:rather than|instead of)\b([^,;:.!?]*)", lowered):
        if patterns[state].search(alternative.group(1)):
            return None
    qualifier = (
        r"\b(?:no|not|never|without|unclear|unknown|uncertain|unconfirmed|"
        r"may|might|could|maybe|possibly|perhaps|probably|likely|appears|seems)\b|n['’]t\b"
    )
    for sentence in re.split(r"[.!?;]", lowered):
        if not patterns[state].search(sentence):
            continue
        # An attached uncertainty clause still qualifies the state: "green, but its
        # state is uncertain". This does not treat "no state transition" as uncertainty.
        if re.search(
            r"\b(?:its|the)\s+(?:state|phase|color|motion)\s+(?:is|remains|seems|appears)"
            r"\s+(?:unclear|unknown|uncertain|unconfirmed)\b",
            sentence,
        ):
            return None
        for clause in re.split(r"[,():]|\b(?:and|but|with|while|whereas)\b", sentence):
            if patterns[state].search(clause) and re.search(qualifier, clause):
                return None
    return state


def _claim_resolution(claim, attributes: dict, host_events, start: int, end: int) -> str:
    """The adaptive-focus trigger state of one claim: checked / diverged / unverifiable.

    ``checked``: the claim asserts a state some overlapping OBSERVABLE event confirms.
    ``diverged``: it contradicts one (recorded). ``unverifiable``: it asserts no checkable
    state, or nothing overlapping can adjudicate it -- the orchestrator's focus policy
    treats diverged/unverifiable claims as revisit candidates.
    """
    if "divergence" in attributes:
        return "diverged"
    if claim is None:
        return "unverifiable"
    confirming = [
        event
        for event in host_events
        if event.tier is Tier.OBSERVABLE
        and event.attributes.get("state") == claim
        and event.window_us[0] <= end
        and event.window_us[1] >= start
    ]
    return "checked" if confirming else "unverifiable"


def _side_divergence(side: str, host_events, start: int, end: int) -> str | None:
    """A side claim contradicting the host's overlapping rig-frame lateral geometry.

    FLU convention: positive lateral = left of ego heading. Only clear cases diverge
    (|lateral| >= _SIDE_MIN_LATERAL_M); near-center geometry adjudicates nothing.
    """
    for event in host_events:
        lateral = event.attributes.get("lateral_m_end")
        if lateral is None:
            continue
        if not (event.window_us[0] <= end and event.window_us[1] >= start):
            continue
        if abs(lateral) < _SIDE_MIN_LATERAL_M:
            continue
        actual = "left" if lateral > 0 else "right"
        if actual != side:
            return (
                f"claims {side} but rig-frame lateral {lateral:+.1f}m places it {actual} "
                f"(ego-path frame, FLU)"
            )
    return None


def _claim_divergence(claim: str, events, start: int, end: int) -> str | None:
    """The divergence message when a claim contradicts an overlapping OBSERVABLE event."""
    conflicting = [
        event
        for event in events
        if event.tier is Tier.OBSERVABLE
        and event.attributes.get("state") not in (None, claim)
        and event.window_us[0] <= end
        and event.window_us[1] >= start
    ]
    if not conflicting:
        return None
    return (
        f"claim asserts {claim!r} but the overlapping label event "
        f"{conflicting[0].label!r} holds state {conflicting[0].attributes['state']!r}"
    )


# Salient vocabulary for the dedup veto: single-token swaps here change the claim's
# meaning while barely moving Jaccard (left/right, colors, motion words, digits).
_SALIENT_TOKENS = frozenset(
    {
        "left",
        "right",
        "ahead",
        "behind",
        "front",
        "rear",
        "toward",
        "away",
        "red",
        "green",
        "yellow",
        "moving",
        "stopped",
        "stationary",
        "parked",
        "walking",
        "crossing",
        "standing",
    }
)


_DEDUP_JACCARD = 0.5


_DEDUP_NEAR_MISS = 0.35


_DEDUP_GAP_US = 2_000_000


def _claim_tokens(text: str) -> set[str]:
    """Normalized token set for the dedup similarity check."""
    import re as _re

    return set(_re.findall(r"[a-z0-9]+", text.lower()))


def _coalesce_claim(
    component,
    new_event,
    jaccard_threshold: float = _DEDUP_JACCARD,
    gap_us: int = _DEDUP_GAP_US,
    key: str = "text",
) -> tuple[tuple | None, dict]:
    """EXPERIMENTAL cross-round claim dedup: fold a repeated claim into its prior event.

    Returns ``(events, note)``: ``events`` is the component's event tuple with the matched
    prior observation event replaced by the merged one (None when nothing merges -- append
    as usual); ``note`` carries the audit record (jaccard, salient diff, near-misses,
    noop). Predicate: same claim_state AND full-text Jaccard >= 0.5 AND temporal adjacency
    within 2s AND the salient-token veto (direction/color/motion words, digits -- a
    differing salient token never merges).

    Under ``key="identity"`` the Jaccard term is DROPPED for a component that carries a
    tracklet: binding already settled that these claims are about one referent, so wording
    is the model's prose rather than a fact about the world (phase 5 of agent_tracks_plan
    -- identity replaces the similarity TERM, never the whole rule). The veto still
    applies, because "turning left" and "turning right" share a claim_state, and an
    unbound component falls back to text however the knob is set.

    Comparison runs against every temporally
    adjacent observation event, not the last-appended one (focus rounds arrive late but
    cover early windows). Sources union so no evidence drops; resolution and divergence
    recompute on the merged window. Validated on 105 pairs (docs/arxiv/claim_dedup_plan.md);
    performance at scale unproven -- monitor the counts this emits.
    """
    import dataclasses

    attributes = component.attributes or {}
    bound = bool(attributes.get("tracklet_id") or attributes.get("track_ids"))
    by_identity = key == "identity" and bound
    note: dict = {"near_misses": [], "merged": False}
    if by_identity:
        note["key"] = "identity"
    new_attrs = new_event.attributes
    new_tokens = _claim_tokens(new_attrs.get("observation", ""))
    best = None
    for index, event in enumerate(component.events):
        attrs = event.attributes
        if "observation" not in attrs:
            continue
        if attrs.get("claim_state") != new_attrs.get("claim_state"):
            continue
        gap_ok = (
            new_event.window_us[0] <= event.window_us[1] + gap_us
            and new_event.window_us[1] >= event.window_us[0] - gap_us
        )
        if not gap_ok:
            continue
        tokens = _claim_tokens(attrs.get("observation", ""))
        union = tokens | new_tokens
        jaccard = len(tokens & new_tokens) / len(union) if union else 0.0
        salient = sorted(t for t in tokens ^ new_tokens if t in _SALIENT_TOKENS or t.isdigit())
        # the referent is settled by binding, so similarity decides nothing: adjacency,
        # the state and the veto carry the whole predicate
        similar = True if by_identity else jaccard >= jaccard_threshold
        if similar and not salient:
            if best is None or jaccard > best[0]:
                best = (jaccard, index, event)
        elif _DEDUP_NEAR_MISS <= jaccard < jaccard_threshold and not salient:
            note["near_misses"].append(round(jaccard, 2))
        elif similar and salient:
            note.setdefault("salient_vetoes", []).append(salient)
    if best is None:
        return None, note
    jaccard, index, prior = best
    window = (
        min(prior.window_us[0], new_event.window_us[0]),
        max(prior.window_us[1], new_event.window_us[1]),
    )
    attributes = dict(prior.attributes)
    attributes["observation"] = new_attrs.get("observation", attributes.get("observation"))
    attributes["coalesced"] = attributes.get("coalesced", 1) + 1
    claim = attributes.get("claim_state")
    host_events = tuple(
        event for event in component.events if "observation" not in event.attributes
    )
    attributes.pop("divergence", None)
    if claim is not None:
        divergence = _claim_divergence(claim, host_events, *window)
        if divergence is not None:
            attributes["divergence"] = divergence
    attributes["resolution"] = _claim_resolution(claim, attributes, host_events, *window)
    merged = dataclasses.replace(
        prior,
        window_us=window,
        attributes=attributes,
        sources=tuple(dict.fromkeys((*prior.sources, *new_event.sources))),
    )
    note.update(
        {
            "merged": True,
            "jaccard": round(jaccard, 2),
            "noop": window == prior.window_us
            and new_attrs.get("observation") == prior.attributes.get("observation"),
        }
    )
    events = list(component.events)
    events[index] = merged
    events.sort(key=lambda event: event.window_us[0])
    return tuple(events), note


def _record_span(record) -> tuple[int, int] | None:
    """A media record's temporal extent: its timestamp, or its payload window for segments."""
    if record.timestamp_us is not None:
        return (record.timestamp_us, record.timestamp_us)
    window = record.payload.get("window_us")
    return (int(window[0]), int(window[1])) if window else None


def _observation_event(
    component,
    frame_records,
    description: str,
    evidence_of,
    vlm_record: str,
    category: str | None = None,
) -> Event | None:
    """Bind one VLM claim to a CONTEXTUAL event over the frames that carried it.

    The window is the claim's frame span clipped to the component's own window; frames
    clipped out are dropped from the event's sources (they stay on the component), so the
    validator's timestamp-containment holds by construction. A claim whose frames lie
    wholly outside that window keeps its own span -- the caller must then widen the
    successor's event_window_us to contain it -- because a claim that binds no event would
    have nowhere to live now that identity carries no claim text. A frameless round still
    binds nothing. A
    claim asserting a motion state that contradicts an overlapping OBSERVABLE label event
    records the divergence on the event.
    """
    spans = [(record, _record_span(record)) for record in frame_records]
    spans = [(record, span) for record, span in spans if span is not None]
    if not spans:
        # A frameless round carries no sensor evidence, but the claim is still TEXT and
        # must live in the claim layer rather than in identity. Bind it over the host's
        # own window, sourced only to the model record -- an unevidenced claim, visibly so.
        if component is None or component.event_window_us is None:
            return None
        return Event(
            label=f"observed: {description[:60]}",
            window_us=component.event_window_us,
            attributes={"observation": description},
            sources=(vlm_record,),
            tier=Tier.CONTEXTUAL,
        )
    start = min(span[0] for _, span in spans)
    end = max(span[1] for _, span in spans)
    window = component.event_window_us if component is not None else None
    if window is not None:
        clipped_start, clipped_end = max(start, window[0]), min(end, window[1])
        if clipped_start <= clipped_end:
            start, end = clipped_start, clipped_end
        # else: the claim's frames lie wholly outside the component's window. Clipping
        # would empty it and the claim would bind NO event -- which is how three claims
        # in smoke12 survived only as an `observation` attribute, i.e. as identity. The
        # claim keeps its OWN frame span instead, and the caller widens the successor's
        # window to contain it (three_layer_plan phase 2: a claim is an event, always).
    in_window = [record for record, span in spans if span[0] <= end and span[1] >= start]
    evidence = evidence_of(in_window)
    attributes: dict = {"observation": description}
    # motion-state tokens are meaningful only on moving entities: extraction is
    # subject-blind ("red light ... explains why the vehicle STOPPED" is not a stopped
    # light), so non-agent hosts bind stateless claims (honest unverifiable, not
    # accidentally adjudicable). Signal PHASE vocabulary is future work.
    host_category = component.category if component is not None else category
    if host_category in ("dynamic_agent", "ego_vehicle"):
        claim = _claim_state(description)
    elif host_category == "traffic_control":
        claim = _claim_state(description, _CLAIM_PHASE_WORDS)
    else:
        claim = None
    host_events = component.events if component is not None else ()
    if claim is not None:
        attributes["claim_state"] = claim
        divergence = _claim_divergence(claim, host_events, start, end)
        if divergence is not None:
            attributes["divergence"] = divergence
    if host_category == "dynamic_agent":
        side = _claim_state(description, _CLAIM_SIDE_WORDS)
        if side is not None:
            attributes["claim_side"] = side
            side_divergence = _side_divergence(side, host_events, start, end)
            if side_divergence is not None:
                attributes["divergence"] = (
                    attributes.get("divergence", "")
                    + ("; " if "divergence" in attributes else "")
                    + side_divergence
                )
    attributes["resolution"] = _claim_resolution(claim, attributes, host_events, start, end)
    return Event(
        label=f"observed: {claim or 'claim'}",
        window_us=(start, end),
        attributes=attributes,
        sources=(*evidence, vlm_record),
        tier=Tier.CONTEXTUAL,
    )


def _rescoped_events(events, host_events, host_window, catalog) -> tuple:
    """Fit one chain's events into a host component's window, re-checking divergence.

    Cross-source merge unions two chains whose events were bound against different windows:
    each incoming event clips to the host window (dropped when disjoint). Clipping narrows
    the window, which can strand timestamped sources outside it -- those are dropped from
    the event (they remain on the merged component), keeping the validator's containment
    rule true. A CONTEXTUAL claim overlapping a conflicting OBSERVABLE host event gains the
    divergence it could not see before the identities were linked.
    """
    import dataclasses

    rescoped = []
    for event in events:
        start = max(event.window_us[0], host_window[0])
        end = min(event.window_us[1], host_window[1])
        if start > end:
            continue
        sources = []
        for ref in event.sources:
            record = catalog.get(ref)
            span = _record_span(record) if record is not None else None
            if span is None or (span[0] <= end and span[1] >= start):
                sources.append(ref)
        attributes = dict(event.attributes)
        claim = attributes.get("claim_state")
        if claim is not None and "divergence" not in attributes:
            divergence = _claim_divergence(claim, host_events, start, end)
            if divergence is not None:
                attributes["divergence"] = divergence
        if "resolution" in attributes:
            attributes["resolution"] = _claim_resolution(claim, attributes, host_events, start, end)
        rescoped.append(
            dataclasses.replace(
                event, window_us=(start, end), attributes=attributes, sources=tuple(sources)
            )
        )
    return tuple(rescoped)


def _merged_events(*event_groups) -> tuple:
    """Union event groups into one start-ordered tuple (validator ordering invariant)."""
    merged = [event for group in event_groups for event in group]
    merged.sort(key=lambda event: event.window_us[0])
    return tuple(merged)
