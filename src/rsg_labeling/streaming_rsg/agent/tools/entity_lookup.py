"""Entity lookup: full component history for one entity, superseded revisions included."""

from __future__ import annotations

from ...data_structure.rsg import Rsg


def entity_lookup(rsg: Rsg, entity_id: str) -> str:
    """Return every component (history included) whose id or label mentions the entity."""
    lines = [
        f"{component.id} [{component.category} T{int(component.tier)}] {component.label}"
        for component in rsg.all_components
        if entity_id in component.id or entity_id in component.label
    ]
    if not lines:
        return f"no records for {entity_id}"
    return "\n".join(lines)
