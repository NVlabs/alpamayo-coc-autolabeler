"""Agent middleware: the interaction protocol between models/tools and the op gateway.

Everything here is periphery mechanics -- eliciting well-formed output, parsing replies,
(future: retry policies, structured tool I/O) -- never op contract and never graph
semantics. Ops consume what the middleware returns; the orchestrator drives it
(``structured_model_turn``, ``call_tool``).
"""

from .parsing import parse_json_reply, parse_json_reply_list

__all__ = ["parse_json_reply", "parse_json_reply_list"]
