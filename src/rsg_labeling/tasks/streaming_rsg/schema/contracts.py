"""Persisted task contracts; all times are absolute integer microseconds."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClipClock:
    """Physical clip extent, with the verified first-camera timestamp as origin."""

    origin_us: int
    end_us: int

    def __post_init__(self) -> None:
        """Reject lossy clock conversion and nonpositive extents."""
        if any(type(t) is not int for t in (self.origin_us, self.end_us)):
            raise ValueError("clock values must be integer microseconds")
        if self.end_us <= self.origin_us:
            raise ValueError("clip must have positive duration")

    def sample_anchors(self, stride_us: int) -> list[int]:
        """Return anchors with two seconds of history and six seconds of future coverage."""
        if type(stride_us) is not int or stride_us <= 0:
            raise ValueError("anchor_stride_us must be a positive integer")
        return list(range(self.origin_us + 2_000_000, self.end_us - 6_000_000 + 1, stride_us))


@dataclass(frozen=True)
class Interval:
    """A semantic interval and its committed action/cause, not a new graph node.

    Cause references identify event-level facts in the parent graph view. Outcome
    references may use later ego motion; they must not be promoted into cause references.
    Validated means mechanically admissible, not independently proven correct.
    """

    interval_id: str
    interval_start_us: int
    interval_end_us: int
    action: str
    cause: str
    cause_refs: tuple[str, ...]
    outcome_refs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    status: str = "validated"

    def __post_init__(self) -> None:
        """Validate the explicit semantic packet before any export."""
        ClipClock(self.interval_start_us, self.interval_end_us)
        if not self.interval_id or not self.action.strip():
            raise ValueError("interval requires an id and committed action")
        if self.status not in {"validated", "review_needed"}:
            raise ValueError("unknown interval status")
        if self.status == "validated":
            if self.cause == "<NOMINAL>":
                if self.cause_refs or not self.outcome_refs:
                    raise ValueError(
                        "nominal interval needs actual-action evidence, not invented causes"
                    )
            elif not self.cause.strip() or not self.cause_refs:
                raise ValueError("validated interval requires a cause and cause references")
        if len(set(self.cause_refs)) != len(self.cause_refs):
            raise ValueError("duplicate cause references")


@dataclass(frozen=True)
class Timeline:
    """An exact half-open partition of a clip's physical extent."""

    clip_id: str
    clock: ClipClock
    intervals: tuple[Interval, ...]

    def __post_init__(self) -> None:
        """Reject gaps, overlaps, duplicate ids, and truncated annotation."""
        if not self.clip_id or not self.intervals:
            raise ValueError("timeline requires a clip id and intervals")
        ids = [item.interval_id for item in self.intervals]
        if len(set(ids)) != len(ids):
            raise ValueError("interval ids must be unique")
        cursor = self.clock.origin_us
        for item in self.intervals:
            if item.interval_start_us != cursor:
                raise ValueError("timeline must be contiguous and ordered")
            cursor = item.interval_end_us
        if cursor != self.clock.end_us:
            raise ValueError("timeline must cover the full clip")

    def at(self, sample_anchor_us: int) -> Interval:
        """Return the active interval; an interior boundary belongs to its successor."""
        for item in self.intervals:
            if item.interval_start_us <= sample_anchor_us < item.interval_end_us:
                return item
        raise ValueError("sample anchor is outside the clip")

    def to_dict(self) -> dict:
        """Serialize the timeline without changing timestamps or interval identities."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> Timeline:
        """Read a persisted timeline and revalidate its complete partition."""
        intervals = []
        for raw in payload["intervals"]:
            item = dict(raw)
            for name in ("cause_refs", "outcome_refs", "tags"):
                if name in item:
                    item[name] = tuple(item[name])
            intervals.append(Interval(**item))
        return cls(payload["clip_id"], ClipClock(**payload["clock"]), tuple(intervals))
