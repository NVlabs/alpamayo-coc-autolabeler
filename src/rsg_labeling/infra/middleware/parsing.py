"""Middleware parsing: model replies into structured payloads.

The parse side of ``structured_model_turn`` -- interaction protocol, never op contract.
Tolerates fenced code blocks; raises ``ValueError`` on unparsable replies so the
elicitation layer can retry.
"""

from __future__ import annotations

import json


def parse_json_reply_list(reply: str) -> list[dict]:
    """Parse a model reply as a JSON list, tolerating a fenced code block."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    start, end = text.find("["), text.rfind("]")
    if start == -1:
        raise ValueError(f"model reply carries no JSON list: {reply[:200]}")
    # A truncated reply has no closing bracket at all, which is the case worth salvaging.
    span = text[start : end + 1] if end > start else text[start:]
    try:
        parsed = json.loads(span)
    except json.JSONDecodeError:
        # A truncated reply loses its tail, not its earlier elements: keep the complete
        # objects and drop the fragment, rather than discarding an entire observation pass.
        # Raising here costs the whole clip, which is how one clip per generation died.
        parsed = _salvage_objects(span)
    if not isinstance(parsed, list):
        raise ValueError("model reply is not a JSON list")
    return parsed


def _salvage_objects(span: str) -> list[dict]:
    """Every complete top-level object in a broken JSON list, in order.

    Scans for balanced braces outside strings; a trailing fragment is dropped. Raises when
    nothing survives, so an unusable reply still reaches the elicitation layer's retry.
    """
    objects: list[dict] = []
    depth, start, in_string, escaped = 0, None, False, False
    for index, character in enumerate(span):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    objects.append(json.loads(span[start : index + 1]))
                except json.JSONDecodeError:
                    pass
                start = None
    if not objects:
        raise ValueError(f"model reply holds no complete JSON object: {span[:200]}")
    return objects


def parse_json_reply(reply: str) -> dict:
    """Parse a model reply as JSON, tolerating a fenced code block."""
    text = reply.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"model reply carries no JSON object: {reply[:200]}")
    return json.loads(text[start : end + 1])
