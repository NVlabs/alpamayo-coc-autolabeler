"""Persisted sub-RSG selectors and conservative transitive causal eligibility checks."""

from __future__ import annotations


def interval_view(snapshot: dict, start_us: int, end_us: int, fact_ids: list[str]) -> dict:
    """Select time-overlapping facts and relations from one frozen op-produced snapshot.

    A view references its parent's event ids and edges; it never rewrites that graph.
    Inclusive RSG point events at start are allowed, but at the half-open end are not.
    Untimed relations are not extended across the interval.
    """
    if start_us >= end_us:
        raise ValueError("view requires a positive half-open interval")
    ids = sorted(set(fact_ids))
    if set(ids) - set(snapshot["facts"]):
        raise ValueError("view references unknown or retired facts")
    for fact_id in ids:
        lo, hi = snapshot["facts"][fact_id]["window_us"]
        if not (lo < end_us and hi >= start_us):
            raise ValueError(f"fact outside interval: {fact_id}")
    components = {snapshot["facts"][fid]["component_id"] for fid in ids}
    edges = [
        i
        for i, edge in enumerate(snapshot["edges"])
        if edge["source"] in components
        and edge["target"] in components
        and edge.get("window_us") is not None
        and edge["window_us"][0] < end_us
        and edge["window_us"][1] >= start_us
    ]
    return {
        "parent_graph_hash": snapshot["graph_hash"],
        "window_us": [start_us, end_us],
        "fact_ids": ids,
        "edge_indices": edges,
    }


def dependency_issues(
    snapshot: dict,
    fact_ids: list[str],
    history_start_us: int,
    cutoff_us: int,
    allowed_source_classes: list[str],
) -> list[str]:
    """Fail closed on unresolved, cyclic, future, or out-of-history dependencies.

    The walk includes the full consultation set, not only chosen citations. A whole-clip
    model claim is not causal merely because it cites an earlier image. Kinematic streams
    require their actual finite-difference support window. No implicit long-term memory
    or external-label permission is supplied by this function.
    """
    issues: set[str] = set()
    visited: set[str] = set()
    active: set[str] = set()
    records = snapshot["records"]

    def visit(ref: str) -> None:
        """Walk the complete evidence/consultation chain with cycle detection."""
        if ref in active:
            issues.add(f"dependency cycle: {ref}")
            return
        if ref in visited:
            return
        if ref not in records:
            issues.add(f"unresolved dependency: {ref}")
            return
        active.add(ref)
        record = records[ref]
        dependencies = set(record.get("derived_from", ())) | set(record.get("reads", ()))
        dependencies.update(record.get("read_ids", ()))
        dependencies.update(record.get("sources", ()))
        for key in ("response_to", "model_call"):
            if record.get(key):
                dependencies.add(record[key])
        source_class = record.get("source_class")
        if source_class in {"sensor", "external_label"}:
            if source_class not in allowed_source_classes:
                issues.add(f"disallowed source class: {ref}")
            support = record.get("payload", {}).get("support_window_us")
            if support is None and record.get("stream", "").startswith("camera/"):
                support = record.get("payload", {}).get("window_us")
            if support is None and record.get("stream") == "ego_pose":
                issues.add(f"unknown kinematic footprint: {ref}")
            if support is None:
                timestamp = record.get("timestamp_us")
                support = (timestamp, timestamp)
            lo, hi = support
            if type(lo) is not int or type(hi) is not int or lo > hi:
                issues.add(f"unknown evidence time: {ref}")
            else:
                if lo < history_start_us:
                    issues.add(f"outside history: {ref}")
                if hi > cutoff_us:
                    issues.add(f"future evidence: {ref}")
        elif not dependencies:
            issues.add(f"unsupported derived record: {ref}")
        for dependency in sorted(dependencies):
            visit(dependency)
        active.remove(ref)
        visited.add(ref)

    for fact_id in fact_ids:
        if fact_id not in snapshot["facts"]:
            issues.add(f"unknown fact: {fact_id}")
            continue
        fact = snapshot["facts"][fact_id]
        dependencies = set(fact["sources"]) | set(fact["read_ids"])
        if not dependencies:
            issues.add(f"unsupported fact: {fact_id}")
        for ref in sorted(dependencies):
            visit(ref)
    return sorted(issues)


def policy_relations(
    snapshot: dict,
    eligible_facts: list[str],
    history_start_us: int,
    cutoff_us: int,
    allowed_source_classes: list[str],
) -> list[dict]:
    """Project timed relations only when both endpoints and their full provenance are causal."""
    components = {snapshot["facts"][fid]["component_id"] for fid in eligible_facts}
    result = []
    for index, edge in enumerate(snapshot["edges"]):
        window = edge.get("window_us")
        if edge["source"] not in components or edge["target"] not in components or window is None:
            continue
        lo, hi = window
        if lo > cutoff_us or hi < history_start_us:
            continue
        provenance = edge["provenance"]
        reads = list(provenance.get("read_ids", ()))
        if provenance.get("model_call"):
            reads.append(provenance["model_call"])
        probe = {**snapshot, "facts": {"relation": {"sources": [], "read_ids": reads}}}
        if dependency_issues(
            probe, ["relation"], history_start_us, cutoff_us, allowed_source_classes
        ):
            continue
        result.append(
            {
                "edge_index": index,
                "source": edge["source"],
                "target": edge["target"],
                "relation": edge["relation"],
                "attributes": edge.get("attributes", {}),
                "window_us": [max(lo, history_start_us), min(hi, cutoff_us)],
            }
        )
    return result
