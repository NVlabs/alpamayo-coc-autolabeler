"""Operation results shared by graph compiler passes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OpResult:
    """Outcome of one op: ``rerun`` names an op the self-critic wants re-planned."""

    status: str = "ok"
    rerun: str | None = None
