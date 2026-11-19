"""The epistemic reasoning scene graph: north-star components, provenance-complete.

The RSG holds **components** -- the nodes and edges of the north-star taxonomy (dynamic agents,
static road elements, traffic controls, rules, ego vehicle/goal, risks, decisions, connected by
typed relations). Sources (frames, trajectory readings, label sightings, map zones) are env
records with ``env:*`` ids; intermediates are context-layer records with ``context:*`` ids;
components cite both through their provenance read-sets, so provenance terminates at the env
and a component's reliability derives from the source classes of what it cites.

Nothing is rewritten or deleted: a revision/merge supersedes, and a prune retires components
under a recorded reason. Every component names the op, read-set, and model call that produced
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

# North-star node categories (graph_taxonomy_north_star.md section 3).
CATEGORIES = frozenset(
    {
        "ego_vehicle",
        "dynamic_agent",
        "static_road_element",
        "traffic_control",
        "traffic_rule",
        "ego_goal",
        "risk",
        "environmental_context",
        "decision",
    }
)

# North-star edge categories (section 4).
RELATIONS = frozenset(
    {
        "spatial",
        "temporal",
        "motion_interaction",
        "rule_applicability",
        "intent",
        "causal",
        "risk_propagation",
        "decision_support",
    }
)

# Provenance read-ids with these prefixes reference env source records or context-layer
# records rather than components; both stores are outside the graph.
EXTERNAL_REF_PREFIXES = ("env:", "context:")


class SourceClass(StrEnum):
    """Reliability class of an env source record, ordered by verifiability.

    Source classes attach to *source records*, never to components -- a component derives its
    reliability by walking provenance to the records it cites. The env's data-source manifest
    (``env/sources.md``) defines the mapping.
    """

    SENSOR = "sensor"
    EXTERNAL_LABEL = "external_label"
    AUGMENTED_MEDIA = "augmented_media"
    RULE_PARSED = "rule_parsed"
    TOOL_OUTPUT = "tool_output"
    VLM_OBSERVATION = "vlm_observation"
    MODEL_PROPOSAL = "model_proposal"
    # A reviewer's verdict or correction, recorded by the review server and materialized by
    # ``revise:human``: external to the run, traceable to a person and a timestamp.
    HUMAN_REVIEW = "human_review"


class Tier(IntEnum):
    """Epistemic tier of a component's claim, named by how its reward is anchored.

    OBSERVABLE facts (presence, position, motion) are anchored by verifiable rewards;
    CONTEXTUAL claims (behavior prediction, rule applicability) are calibrated through
    preference/ranking rewards; LATENT hypotheses (intent) accrue transient rewards evaluated
    over the data distribution, never individual grades.
    """

    OBSERVABLE = 1
    CONTEXTUAL = 2
    LATENT = 3


@dataclass(frozen=True)
class Provenance:
    """How a component came to exist: the op, what it read, and the model call if any.

    ``read_ids`` mixes component ids with external references (``env:*`` source records,
    ``context:*`` renderings); ``model_call`` names the context record of the model
    interaction for model-backed ops, ``None`` for deterministic ones.
    """

    op: str
    read_ids: tuple[str, ...] = ()
    model_call: str | None = None

    def to_dict(self) -> dict:
        """Serialize to plain Python types."""
        result: dict[str, Any] = {"op": self.op}
        if self.read_ids:
            result["read_ids"] = list(self.read_ids)
        if self.model_call is not None:
            result["model_call"] = self.model_call
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> Provenance:
        """Rebuild from a ``to_dict`` mapping."""
        return cls(
            op=raw["op"],
            read_ids=tuple(raw.get("read_ids", ())),
            model_call=raw.get("model_call"),
        )


@dataclass(frozen=True)
class Event:
    """One time-scoped state segment of a component: attributes valid over a sub-window.

    Identity attributes (type, track id, dimensions) stay flat on the component -- the
    component *is* the entity across its window. State attributes (position, speed, motion
    state, observation claims) are functions of time and live here, each segment citing the
    records timestamped inside its window. Tier is per event: an observed segment is
    OBSERVABLE, a model characterization CONTEXTUAL, a predicted segment LATENT (schema
    supports prediction; emission is the outer loop's, never v1's).
    """

    label: str
    window_us: tuple[int, int]
    attributes: dict[str, Any] = field(default_factory=dict)
    sources: tuple[str, ...] = ()
    tier: Tier = Tier.OBSERVABLE

    def to_dict(self) -> dict:
        """Serialize to plain Python types."""
        result: dict[str, Any] = {"label": self.label, "window_us": list(self.window_us)}
        if self.attributes:
            result["attributes"] = self.attributes
        if self.sources:
            result["sources"] = list(self.sources)
        result["tier"] = int(self.tier)
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> Event:
        """Rebuild from a ``to_dict`` mapping."""
        return cls(
            label=raw["label"],
            window_us=tuple(raw["window_us"]),
            attributes=dict(raw.get("attributes", {})),
            sources=tuple(raw.get("sources", ())),
            tier=Tier(raw.get("tier", Tier.OBSERVABLE)),
        )


@dataclass(frozen=True)
class Component:
    """One north-star reasoning-graph node: a categorized, attributed scene component.

    ``label`` is the short human-readable line; structured facts live in ``attributes``
    (north-star attribute tables -- new attributes render without schema changes).

    ``sources`` is the direct-evidence metadata: the env records, context records, and
    components whose content this node actually uses -- narrower than
    ``provenance.read_ids`` (everything the producing op consulted, e.g. a full rendered
    context). Revisions inherit and extend their predecessor's sources, so every node can be
    traced back to its evidence locally, without walking the graph.

    ``supersedes`` names the components this one revises (one for a revision, several for a
    merge); superseded components stay in the graph but drop out of the current view.

    ``events`` scopes the time-varying state attributes into ordered sub-windows of
    ``event_window_us`` (see ``Event``); ``attributes`` then carries identity only. Decision
    components carry no events -- they are anchor-scoped by definition.
    """

    id: str
    category: str
    label: str
    tier: Tier
    provenance: Provenance
    sources: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    timestamp_us: int | None = None
    event_window_us: tuple[int, int] | None = None
    supersedes: tuple[str, ...] = ()
    events: tuple[Event, ...] = ()

    def to_dict(self) -> dict:
        """Serialize to plain Python types."""
        result: dict[str, Any] = {
            "id": self.id,
            "category": self.category,
            "label": self.label,
            "tier": int(self.tier),
            "provenance": self.provenance.to_dict(),
        }
        if self.sources:
            result["sources"] = list(self.sources)
        if self.attributes:
            result["attributes"] = self.attributes
        if self.timestamp_us is not None:
            result["timestamp_us"] = self.timestamp_us
        if self.event_window_us is not None:
            result["event_window_us"] = list(self.event_window_us)
        if self.supersedes:
            result["supersedes"] = list(self.supersedes)
        if self.events:
            result["events"] = [event.to_dict() for event in self.events]
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> Component:
        """Rebuild from a ``to_dict`` mapping; pre-event artifacts load with no events."""
        window = raw.get("event_window_us")
        return cls(
            id=raw["id"],
            category=raw["category"],
            label=raw["label"],
            tier=Tier(raw["tier"]),
            provenance=Provenance.from_dict(raw["provenance"]),
            sources=tuple(raw.get("sources", ())),
            attributes=dict(raw.get("attributes", {})),
            timestamp_us=raw.get("timestamp_us"),
            event_window_us=tuple(window) if window is not None else None,
            supersedes=tuple(raw.get("supersedes", ())),
            events=tuple(Event.from_dict(e) for e in raw.get("events", ())),
        )


@dataclass(frozen=True)
class Edge:
    """One north-star relation between two components, with the same provenance discipline.

    Edge tier is derived, never stored: see ``Rsg.edge_tier``. ``window_us`` optionally
    scopes the relation in time (the event overlap it was derived from); a relation without
    one is anchor-instant.
    """

    source: str
    target: str
    relation: str
    rationale: str
    provenance: Provenance
    attributes: dict[str, Any] = field(default_factory=dict)
    window_us: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        """Serialize to plain Python types."""
        result: dict[str, Any] = {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "rationale": self.rationale,
            "provenance": self.provenance.to_dict(),
        }
        if self.attributes:
            result["attributes"] = self.attributes
        if self.window_us is not None:
            result["window_us"] = list(self.window_us)
        return result

    @classmethod
    def from_dict(cls, raw: dict) -> Edge:
        """Rebuild from a ``to_dict`` mapping."""
        window = raw.get("window_us")
        return cls(
            source=raw["source"],
            target=raw["target"],
            relation=raw["relation"],
            rationale=raw["rationale"],
            provenance=Provenance.from_dict(raw["provenance"]),
            attributes=dict(raw.get("attributes", {})),
            window_us=tuple(window) if window is not None else None,
        )


def _validate_events(component: Component) -> None:
    """Event invariants at the write: containment, ordering, and the decision exclusion.

    Source-timestamp containment (each event source's record timed inside its window) needs
    the stores and lives in the artifact validator; everything checkable from the component
    alone is enforced here.
    """
    if not component.events:
        return
    if component.category == "decision":
        raise ValueError(f"decision component {component.id} carries events (anchor-scoped)")
    if component.event_window_us is None:
        raise ValueError(f"component {component.id} carries events but no event_window_us")
    previous_start = None
    for event in component.events:
        start, end = event.window_us
        if start > end:
            raise ValueError(f"component {component.id} event {event.label!r} window is inverted")
        if start < component.event_window_us[0] or end > component.event_window_us[1]:
            raise ValueError(
                f"component {component.id} event {event.label!r} window exceeds the "
                "component's event_window_us"
            )
        if previous_start is not None and start < previous_start:
            raise ValueError(f"component {component.id} events are not ordered by start")
        previous_start = start


def _validate_read_ids(read_ids: tuple[str, ...], components: dict, owner: str) -> None:
    """Component references must exist; ``env:``/``context:`` references are external."""
    for read_id in read_ids:
        if read_id.startswith(EXTERNAL_REF_PREFIXES):
            continue
        if read_id not in components:
            raise ValueError(f"{owner} provenance reads unknown component {read_id}")


class Rsg:
    """Append-only component graph bound to one env instance.

    Writes validate at the cheapest point: unknown categories/relations, duplicate ids,
    dangling component references, and supersession of unknown components all raise on
    ``add_component`` / ``add_edge`` / ``prune``, so a constructed graph is
    provenance-complete by construction and a loaded one is revalidated by replaying through
    the same writes.
    """

    def __init__(self, env_id: str) -> None:
        """Bind the graph to the env instance whose source records terminate its provenance."""
        self.env_id = env_id
        self._components: dict[str, Component] = {}
        self._edges: list[Edge] = []
        self._superseded: set[str] = set()
        self._prunes: list[dict] = []
        self.log: list[str] = []

    def add_component(self, component: Component) -> None:
        """Append one component, enforcing category, uniqueness, provenance, supersession."""
        if component.category not in CATEGORIES:
            raise ValueError(f"unknown component category: {component.category}")
        if component.id in self._components:
            raise ValueError(f"duplicate component id: {component.id}")
        _validate_read_ids(component.provenance.read_ids, self._components, component.id)
        _validate_read_ids(component.sources, self._components, f"{component.id} sources")
        for superseded_id in component.supersedes:
            if superseded_id not in self._components:
                raise ValueError(
                    f"component {component.id} supersedes unknown component {superseded_id}"
                )
        _validate_events(component)
        self._superseded.update(component.supersedes)
        self._components[component.id] = component

    def add_edge(self, edge: Edge) -> None:
        """Append one edge, enforcing the relation set, endpoints, and provenance reads."""
        if edge.relation not in RELATIONS:
            raise ValueError(f"unknown edge relation: {edge.relation}")
        for endpoint in (edge.source, edge.target):
            if endpoint not in self._components:
                raise ValueError(f"edge endpoint references unknown component {endpoint}")
        _validate_read_ids(
            edge.provenance.read_ids, self._components, f"edge {edge.source}->{edge.target}"
        )
        if edge.window_us is not None and edge.window_us[0] > edge.window_us[1]:
            raise ValueError(f"edge {edge.source}->{edge.target} window is inverted")
        self._edges.append(edge)

    def prune(self, component_ids: tuple[str, ...], op: str, reason: str) -> None:
        """Retire components from the current view under a recorded reason.

        Pruning is a supersession without a successor: the components stay in history, the
        prune record carries the op and reason (the verdict text itself is context-layer
        material).
        """
        for component_id in component_ids:
            if component_id not in self._components:
                raise ValueError(f"prune references unknown component {component_id}")
        self._superseded.update(component_ids)
        self._prunes.append({"op": op, "ids": list(component_ids), "reason": reason})

    def component(self, component_id: str) -> Component:
        """Return one component by id."""
        return self._components[component_id]

    @property
    def components(self) -> list[Component]:
        """Current view: components in insertion order, superseded/pruned ones excluded."""
        return [c for c in self._components.values() if c.id not in self._superseded]

    @property
    def all_components(self) -> list[Component]:
        """Full history: every component in insertion order."""
        return list(self._components.values())

    @property
    def edges(self) -> list[Edge]:
        """All edges in insertion order."""
        return list(self._edges)

    @property
    def prunes(self) -> list[dict]:
        """All prune records in order."""
        return list(self._prunes)

    def edge_tier(self, edge: Edge) -> Tier:
        """Derived edge tier: max of the endpoint tiers (an edge is only as verifiable as its
        least-verifiable endpoint).
        """
        return max(self._components[edge.source].tier, self._components[edge.target].tier)

    def to_dict(self) -> dict:
        """Serialize the full graph history to plain Python types."""
        return {
            "env_id": self.env_id,
            "components": [c.to_dict() for c in self._components.values()],
            "edges": [edge.to_dict() for edge in self._edges],
            "prunes": list(self._prunes),
            "log": list(self.log),
        }

    @classmethod
    def from_dict(cls, raw: dict) -> Rsg:
        """Rebuild from a ``to_dict`` mapping, revalidating every write."""
        rsg = cls(env_id=raw["env_id"])
        for component_raw in raw["components"]:
            rsg.add_component(Component.from_dict(component_raw))
        for edge_raw in raw["edges"]:
            rsg.add_edge(Edge.from_dict(edge_raw))
        for prune_raw in raw.get("prunes", []):
            rsg.prune(tuple(prune_raw["ids"]), prune_raw["op"], prune_raw["reason"])
        rsg.log = list(raw.get("log", []))
        return rsg


# --- Layer discipline (three_layer_plan) -------------------------------------------
# A component is a REFERENCE: an id for uniqueness and a category for search. Everything
# else is one of two other MODALITIES and belongs in events:
#   evidence (sensor)  -- bbox, mask, world position, range; cites env:ref:*/geometry/*
#   claims   (text)    -- what was said about the referent; cites the model record
# Storing either on the component puts a modality in the identity slot, which is how one
# crossing came to exist as two components (an evidence-shaped one and a claim-shaped one)
# with nothing representing the reference they share.
IDENTITY_ATTRIBUTES = frozenset(
    {
        "kind",  # the concept the reference denotes -- the searchable category
        "id",  # external-label track id, where the env carries one
        "type",  # map-stream element type (lane, etc.)
        "tracklet_id",  # the perception reference this component is about
        "track_ids",  # several, when a group binds more than one reference
        "readings",  # ego stream identity
        "split_side",  # which side of a revision break this successor is: lineage
    }
)


def identity_violations(component) -> list[str]:
    """Attribute keys on ``component`` that carry evidence or claims, not identity.

    Empty for a conformant component. This is the specification for the layer split, so
    it is deliberately an allow-list: a NEW attribute is a violation until someone decides
    which modality it belongs to.
    """
    return sorted(set(component.attributes) - IDENTITY_ATTRIBUTES)
