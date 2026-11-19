"""Persistent inference-key request blocks shared by transports and accounting."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def resolve_inference_state_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve the owner-level accounting directory shared by projects and worktrees."""
    source = os.environ if environ is None else environ
    value = source.get("ALPAMAYO_INFERENCE_STATE_DIR", "").strip()
    if not value:
        raise ValueError("ALPAMAYO_INFERENCE_STATE_DIR is not configured")
    return Path(value).expanduser().resolve()


def exhausted_inference_fingerprints(store_dir: Path) -> set[str]:
    """Read the persistent request blocklist and exhausted entries in the current snapshot."""
    path = store_dir / "exhausted_fingerprints.txt"
    fingerprints = set(path.read_text().splitlines()) if path.exists() else set()
    snapshot = _read_json(store_dir / "current.json") or {}
    fingerprints.update(
        record["fingerprint"]
        for record in snapshot.get("keys", [])
        if record["status"] == "exhausted"
    )
    return fingerprints


def mark_inference_key_exhausted(store_dir: Path, fingerprint: str) -> None:
    """Persist a depletion signal before any later refresh or inference request."""
    store_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (store_dir / "refresh.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        fingerprints = exhausted_inference_fingerprints(store_dir)
        fingerprints.add(fingerprint)
        _write_fingerprints(store_dir / "exhausted_fingerprints.txt", list(fingerprints))


def inference_quota_depleted(status_code: int, body: bytes | str) -> bool:
    """Recognize explicit budget depletion, not throttling or ambiguous resource limits.

    Only structured error codes/types authorize a persistent credential block. Unknown
    errors remain subject to each caller's bounded failure/retry policy.
    """
    if status_code != 429:
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    if not isinstance(payload, dict):
        return False
    error = payload.get("error", payload)
    return isinstance(error, dict) and any(
        error.get(field)
        in ("insufficient_quota", "budget_exceeded", "budget_exhausted", "out_of_budget")
        for field in ("code", "type")
    )


def require_non_exhausted_inference_key(value: str) -> str:
    """Return a safe fingerprint, refusing a blocked key when shared state is configured."""
    fingerprint = "sha256:" + hashlib.sha256(value.encode()).hexdigest()[:12]
    if os.environ.get("ALPAMAYO_INFERENCE_STATE_DIR", "").strip():
        if fingerprint in exhausted_inference_fingerprints(resolve_inference_state_dir()):
            raise RuntimeError(f"exhausted inference key {fingerprint} is blocked")
    return fingerprint


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read one optional secret-free snapshot containing last-known key details."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_fingerprints(
    path: Path, fingerprints: list[str], *, preserve_order: bool = False
) -> None:
    """Replace one fingerprint list without persisting credentials."""
    values = fingerprints if preserve_order else sorted(fingerprints)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("".join(f"{fingerprint}\n" for fingerprint in values))
    tmp_path.replace(path)
