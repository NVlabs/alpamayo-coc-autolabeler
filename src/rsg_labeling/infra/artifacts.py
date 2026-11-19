"""Write and read bounded-inode experiment artifact publications."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

PUBLICATION_SCHEMA = "alparsg_compact_records_v1"
_PARQUET_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string(), nullable=False),
        pa.field("payload_json", pa.large_string(), nullable=False),
        pa.field("payload_sha256", pa.string(), nullable=False),
    ]
)


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path: Path) -> None:
    """Flush directory entries after installing their completed files."""
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_text_atomic(path: Path, contents: str) -> None:
    """Flush text before a same-directory atomic rename, then sync the directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        output.write(contents)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Install a complete, synced JSON mapping through an atomic rename."""
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def event_record_id(key: str, event_t0: int) -> str:
    """Return the canonical compact record id for one event."""
    return json.dumps([key, event_t0], separators=(",", ":"))


class CompactArtifactJournal:
    """Persist resumable records in one journal and finalize them as Parquet."""

    def __init__(self, root: Path, payload_schema: str, shard_size: int = 512) -> None:
        """Open one output root for a single process with any number of worker threads."""
        if shard_size < 1:
            raise ValueError("compact publication shard_size must be positive")
        self.root = Path(root)
        self.payload_schema = payload_schema
        self.shard_size = shard_size
        self.root.mkdir(parents=True, exist_ok=True)
        self._journal_path = self.root / ".resume.jsonl"
        self._manifest_path = self.root / "manifest.json"
        self._finalized = self._manifest_path.is_file()
        self._lock = threading.Lock()
        self._repair_tail = False
        self._latest = self._load_existing()

    def _load_existing(self) -> dict[str, dict[str, Any]]:
        """Read complete records without modifying a journal that another process may append."""
        if self._manifest_path.is_file():
            records = load_compact_records(self.root, payload_schema=self.payload_schema)
            return {
                record_id: {"record_id": record_id, "status": "complete", "payload": payload}
                for record_id, payload in records.items()
            }
        if not self._journal_path.is_file():
            return {}
        journal = self._journal_path.read_bytes()
        if journal and not journal.endswith(b"\n"):
            self._repair_tail = True
            journal = journal[: journal.rfind(b"\n") + 1]
        latest: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(journal.splitlines(), start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, dict) or not isinstance(entry.get("record_id"), str):
                raise ValueError(f"invalid compact resume entry at line {line_number}")
            if entry.get("payload_schema") != self.payload_schema:
                raise ValueError(f"wrong compact resume schema at line {line_number}")
            if entry.get("status") not in {"complete", "error"}:
                raise ValueError(f"invalid compact resume status at line {line_number}")
            latest[entry["record_id"]] = entry
        return latest

    def completed_payload(self, record_id: str) -> dict[str, Any] | None:
        """Return a previously completed payload, or ``None`` when it needs work."""
        with self._lock:
            entry = self._latest.get(record_id)
            if entry is None or entry["status"] != "complete":
                return None
            payload = entry["payload"]
            if not isinstance(payload, dict):
                raise TypeError(f"compact payload must be a mapping: {record_id}")
            return dict(payload)

    def completed_records(self) -> dict[str, dict[str, Any]]:
        """Return all durable successful rows currently available for reading."""
        with self._lock:
            return {
                record_id: dict(entry["payload"])
                for record_id, entry in self._latest.items()
                if entry["status"] == "complete"
            }

    def record(self, record_id: str, payload: Mapping[str, Any]) -> None:
        """Append one successful record and make it available for future resume."""
        self._append(
            {
                "record_id": record_id,
                "status": "complete",
                "payload": dict(payload),
            }
        )

    def record_error(
        self,
        record_id: str,
        error: Exception,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one retryable failure without creating a per-record error file."""
        self._append(
            {
                "record_id": record_id,
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
                "context": dict(context or {}),
            }
        )

    def _append(self, entry: dict[str, Any]) -> None:
        entry["payload_schema"] = self.payload_schema
        encoded = (_canonical_json(entry) + "\n").encode()
        with self._lock:
            if self._finalized:
                raise FileExistsError(f"compact publication is already finalized: {self.root}")
            if self._repair_tail:
                # Only the owning writer repairs an interrupted append; readers never truncate.
                with self._journal_path.open("r+b") as output:
                    current = output.read()
                    if current and not current.endswith(b"\n"):
                        output.truncate(current.rfind(b"\n") + 1)
                        output.flush()
                        os.fsync(output.fileno())
                self._repair_tail = False
            with self._journal_path.open("ab") as journal:
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())
            self._latest[str(entry["record_id"])] = entry

    def finalize(
        self,
        expected_record_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Verify exact successful coverage and publish bounded Parquet shards."""
        expected = list(expected_record_ids)
        if len(expected) != len(set(expected)):
            raise ValueError("compact publication expected record ids are not unique")
        if self._manifest_path.is_file():
            records = load_compact_records(self.root, payload_schema=self.payload_schema)
            if list(records) != expected:
                raise ValueError("completed compact publication record order differs")
            return json.loads(self._manifest_path.read_text())

        with self._lock:
            if self._finalized:
                raise RuntimeError(f"compact publication finalization is in progress: {self.root}")
            extra = set(self._latest) - set(expected)
            missing = [record_id for record_id in expected if record_id not in self._latest]
            failed = [
                record_id
                for record_id in expected
                if record_id in self._latest and self._latest[record_id]["status"] != "complete"
            ]
            if extra or missing or failed:
                raise ValueError(
                    "compact publication is incomplete: "
                    f"missing={len(missing)} failed={len(failed)} extra={len(extra)}"
                )
            rows = []
            for record_id in expected:
                payload_json = _canonical_json(self._latest[record_id]["payload"])
                rows.append(
                    {
                        "record_id": record_id,
                        "payload_json": payload_json,
                        "payload_sha256": _sha256_bytes(payload_json.encode()),
                    }
                )
            # Stop worker appends before releasing the lock for Parquet serialization. Callers
            # normally finalize after joining workers; this also makes accidental overlap safe.
            self._finalized = True

        try:
            # A failed earlier finalization can leave only derived shards behind. The journal
            # remains authoritative until the manifest is atomically installed, so discard and
            # rebuild those unreferenced files before retrying.
            for stale_shard in self.root.glob("part-*.parquet"):
                stale_shard.unlink()
            for stale_temporary in self.root.glob("part-*.parquet.tmp"):
                stale_temporary.unlink()
            shards = []
            for shard_index, start in enumerate(range(0, len(rows), self.shard_size)):
                shard_rows = rows[start : start + self.shard_size]
                shard_path = self.root / f"part-{shard_index:05d}.parquet"
                temporary_shard = shard_path.with_suffix(".parquet.tmp")
                pq.write_table(
                    pa.Table.from_pylist(shard_rows, schema=_PARQUET_SCHEMA),
                    temporary_shard,
                    compression="zstd",
                )
                stored_rows = pq.read_table(temporary_shard).to_pylist()
                if stored_rows != shard_rows:
                    raise ValueError("compact publication Parquet round trip differs")
                with temporary_shard.open("rb") as source:
                    os.fsync(source.fileno())
                temporary_shard.replace(shard_path)
                shards.append(
                    {
                        "path": shard_path.name,
                        "rows": len(shard_rows),
                        "bytes": shard_path.stat().st_size,
                        "sha256": _sha256_file(shard_path),
                    }
                )
            manifest: dict[str, Any] = {
                "schema_version": PUBLICATION_SCHEMA,
                "payload_schema": self.payload_schema,
                "complete": True,
                "records": len(rows),
                "shard_size": self.shard_size,
                "ordered_record_ids_sha256": _sha256_bytes("\n".join(expected).encode()),
                "shards": shards,
                "metadata": dict(metadata),
            }
            write_json_atomic(self._manifest_path, manifest)
            if json.loads(self._manifest_path.read_text()) != manifest:
                raise ValueError("compact publication manifest JSON round trip differs")
        except BaseException:
            with self._lock:
                self._finalized = self._manifest_path.is_file()
            raise
        self._journal_path.unlink(missing_ok=True)
        sync_directory(self.root)
        return manifest


def load_compact_records(
    root: Path,
    payload_schema: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load and verify one finalized compact publication in manifest order."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != PUBLICATION_SCHEMA or manifest.get("complete") is not True:
        raise ValueError(f"invalid compact publication manifest: {root}")
    if payload_schema is not None and manifest.get("payload_schema") != payload_schema:
        raise ValueError(
            f"expected compact payload schema {payload_schema!r}, "
            f"found {manifest.get('payload_schema')!r}"
        )
    records: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    declared_paths = {str(shard["path"]) for shard in manifest["shards"]}
    actual_paths = {path.name for path in root.glob("part-*.parquet")}
    if actual_paths != declared_paths:
        raise ValueError(f"compact publication shard set differs: {root}")
    for shard in manifest["shards"]:
        path = (root / str(shard["path"])).resolve()
        if path.parent != root:
            raise ValueError(f"compact publication shard escapes root: {path}")
        if path.stat().st_size != int(shard["bytes"]) or _sha256_file(path) != shard["sha256"]:
            raise ValueError(f"compact publication shard hash or size mismatch: {path}")
        rows = pq.read_table(path, schema=_PARQUET_SCHEMA).to_pylist()
        if len(rows) != int(shard["rows"]):
            raise ValueError(f"compact publication shard row count mismatch: {path}")
        for row in rows:
            record_id = str(row["record_id"])
            if record_id in records:
                raise ValueError(f"duplicate compact publication record: {record_id}")
            payload_json = str(row["payload_json"])
            if _sha256_bytes(payload_json.encode()) != row["payload_sha256"]:
                raise ValueError(f"compact publication payload hash mismatch: {record_id}")
            payload = json.loads(payload_json)
            if not isinstance(payload, dict):
                raise TypeError(f"compact publication payload is not a mapping: {record_id}")
            ordered_ids.append(record_id)
            records[record_id] = payload
    if len(records) != int(manifest["records"]):
        raise ValueError("compact publication manifest record count differs")
    if _sha256_bytes("\n".join(ordered_ids).encode()) != manifest["ordered_record_ids_sha256"]:
        raise ValueError("compact publication record order digest differs")
    return records


def load_compact_collection(root: Path, payload_schema: str) -> dict[str, dict[str, Any]]:
    """Load one publication or an immediate directory of publication shards."""
    root = Path(root)
    publication_roots = (
        [root]
        if (root / "manifest.json").is_file()
        else [path.parent for path in sorted(root.glob("*/manifest.json"))]
    )
    if not publication_roots:
        raise FileNotFoundError(f"no compact publications found under {root}")
    records: dict[str, dict[str, Any]] = {}
    for publication_root in publication_roots:
        for record_id, payload in load_compact_records(
            publication_root, payload_schema=payload_schema
        ).items():
            if record_id in records:
                raise ValueError(f"duplicate compact collection record: {record_id}")
            records[record_id] = payload
    return records


def load_compact_or_json_payloads(
    root: Path,
    payload_schema: str,
    loose_pattern: str = "shard_*/records/*/*.json",
) -> list[dict[str, Any]]:
    """Read a compact collection, falling back to a historical loose-JSON tree."""
    root = Path(root)
    if (root / "manifest.json").is_file() or any(root.glob("*/manifest.json")):
        return list(load_compact_collection(root, payload_schema).values())
    paths = sorted(root.glob(loose_pattern))
    if not paths:
        raise FileNotFoundError(f"no compact or historical JSON records found under {root}")
    payloads = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"historical JSON record is not a mapping: {path}")
        payloads.append(payload)
    return payloads
