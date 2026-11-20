"""The agent's tool registry: one dedicated module per tool.

Tools are callables the agent invokes -- from a programmed op (policy) or on the model's
request in a tool round. Each tool lives in its own file for separation and evolvability;
factories (``make_*``) bind run-scoped state (env, caches, endpoints) and are registered by
the run entry point. Register with one dict entry, the same name-keyed pattern as ops and
skills.
"""

from .entity_lookup import entity_lookup
from .ground_frame import make_grounding_tool
from .render_overlays import make_render_tool

TOOLS: dict = {
    "entity_lookup": entity_lookup,
}

__all__ = ["TOOLS", "entity_lookup", "make_grounding_tool", "make_render_tool"]
