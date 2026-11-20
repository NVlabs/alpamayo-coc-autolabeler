"""Compact exact replay traces; transport recovery and billing remain with the caller."""

import json
from datetime import datetime, timezone


def recorded_completion(complete, path, image_receipts):
    """Wrap a completion boundary without changing its arguments, output or retry policy.

    Args:
        complete: Existing metered callable accepting the unmodified message list.
        path: One node-local JSONL trace later retained in the compact clip publication.
        image_receipts: Delivered image URLs mapped to immutable source/hash receipts.

    Returns:
        Callable preserving successful and invalid raw replies before downstream parsing.
        Transport errors retain only their type, never credential-bearing exception text.
    """

    def recorded(messages):
        """Replace pixels only in the saved trace, never in the actual model request."""
        compact = []
        for message in messages:
            if isinstance(message["content"], str):
                compact.append(message)
                continue
            parts = []
            for part in message["content"]:
                if part.get("type") != "image_url":
                    parts.append(part)
                    continue
                url = part["image_url"]["url"]
                if url not in image_receipts:
                    raise ValueError("unregistered image cannot enter a compact completion trace")
                parts.append({"type": "image_reference", "record": image_receipts[url]})
            compact.append({**message, "content": parts})
        row = {
            "utc": datetime.now(timezone.utc).isoformat(),
            "messages": compact,
            "response": None,
            "error_type": None,
        }
        try:
            row["response"] = complete(messages)
            return row["response"]
        except Exception as error:
            row["error_type"] = type(error).__name__
            raise
        finally:
            with path.open("a") as handle:
                handle.write(json.dumps(row) + "\n")

    return recorded
