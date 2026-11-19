"""Declare which RSG artifact families a task or evaluator accepts.

``ANCHORING_STREAMING`` identifies the temporal clip graphs produced by
the ``streaming_rsg`` engine and interpreted by the ``tasks.streaming_rsg`` package.

``ANCHORING_KEYFRAME`` identifies event-centered graphs and CoCs. Keyframe generation and
evaluation tools are separate development components, outside the streaming release.
Selecting a keyframe from a streaming graph does not change its artifact family.

``"streaming_based"`` and ``"keyframe_based"`` are anchoring protocol values, independent of
Python package names. They identify artifact granularity, not import paths.

Import ``ANCHORING_STREAMING`` and ``applies_to`` from the package's ``common.provenance``
module, then declare supported families at module scope::

    applies_to(ANCHORING_STREAMING)

Check the declaration where an artifact family and a task/evaluator are paired::

    require_applicable(module, ANCHORING_STREAMING)

These helpers validate declared applicability. They do not inspect artifact contents,
record source hashes or evidence lineage, or determine training eligibility.
"""

from __future__ import annotations

from types import ModuleType

# Protocol labels describe artifact families, not package paths. Shared table adapters may
# accept both families; a keyframe selected within a clip graph still uses ANCHORING_STREAMING.
ANCHORING_STREAMING = "streaming_based"
ANCHORING_KEYFRAME = "keyframe_based"
ANCHORINGS = (ANCHORING_STREAMING, ANCHORING_KEYFRAME)

# Shared by the declaration and lookup helpers; it stores module applicability metadata.
ATTRIBUTE = "__alparsg_anchorings__"


def applies_to(*anchorings: str) -> tuple[str, ...]:
    """Record the caller module's supported RSG artifact families.

    Args:
        *anchorings: One or more of ``ANCHORINGS``.

    Returns:
        The declared anchorings, so a module may also bind them for its own use.

    Raises:
        ValueError: If no anchoring is given, or one is not a known anchoring.
    """
    if not anchorings:
        raise ValueError("applies_to() needs at least one anchoring; see ANCHORINGS")
    unknown = sorted(set(anchorings) - set(ANCHORINGS))
    if unknown:
        raise ValueError(f"unknown anchoring(s) {unknown}; known: {list(ANCHORINGS)}")
    frame = __import__("inspect").currentframe()
    caller = frame.f_back if frame is not None else None
    if caller is not None:
        caller.f_globals[ATTRIBUTE] = tuple(anchorings)
    return tuple(anchorings)


def applicable_anchorings(module: ModuleType) -> tuple[str, ...]:
    """Return the module's declared artifact families, or () for an undeclared module."""
    return tuple(getattr(module, ATTRIBUTE, ()))


def require_applicable(module: ModuleType, anchoring: str) -> None:
    """Require the module to declare support for the requested artifact family.

    Raises:
        ValueError: If ``anchoring`` is unknown, if ``module`` declared nothing, or if it
            declared anchorings that do not include this one.
    """
    if anchoring not in ANCHORINGS:
        raise ValueError(f"unknown anchoring {anchoring!r}; known: {list(ANCHORINGS)}")
    declared = applicable_anchorings(module)
    name = getattr(module, "__name__", repr(module))
    if not declared:
        raise ValueError(
            f"{name} declares no anchoring; add `applies_to(...)` from {__name__} "
            "so a wrong pairing fails here rather than scoring a number that is not valid"
        )
    if anchoring not in declared:
        raise ValueError(
            f"{name} applies to {list(declared)}, not {anchoring!r} -- its artifacts do not "
            "carry the fields this module reads"
        )
