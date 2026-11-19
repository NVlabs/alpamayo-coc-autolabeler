"""Bounded-inode full-clip publications using the existing shared compact journal."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

from .....infra.artifacts import CompactArtifactJournal, load_compact_records

# Producer and readers must agree on the persisted payload contract.
CLIP_SCHEMA = "alparsg_coc_clip_publication_v1"


def load_clip_records(root: Path, allow_empty: bool = False) -> dict[str, dict]:
    """Read finalized shards or durable partial-run outcomes without loose-file extraction.

    Publication completeness means every expected clip has an outcome record, not that
    every annotation succeeded. Per-clip status distinguishes complete/review/failed/not-run.
    """
    if (root / "manifest.json").is_file():
        records = load_compact_records(root, payload_schema=CLIP_SCHEMA)
    elif (root / ".resume.jsonl").is_file():
        records = CompactArtifactJournal(root, CLIP_SCHEMA).completed_records()
    elif allow_empty and root.is_dir() and not any(root.iterdir()):
        records = {}
    else:
        raise FileNotFoundError(f"no compact clip publication: {root}")
    for clip_id, record in records.items():
        if record["clip_id"] != clip_id:
            raise ValueError("compact clip identity mismatch")
        if record["status"] not in {"complete", "review_needed", "failed", "not_run"}:
            raise ValueError("unknown compact clip outcome status")
    return records


def summarize_outcomes(records: dict[str, dict]) -> dict:
    """Report the fixed population, separating review annotations from execution failure."""
    counts = dict.fromkeys(("complete", "review_needed", "failed", "not_run"), 0)
    for row in records.values():
        counts[row["status"]] += 1
    return {
        "outcome_counts": counts,
        "expected_clips": len(records),
        "successful_clips": counts["complete"] + counts["review_needed"],
        "success": bool(records) and not (counts["failed"] or counts["not_run"]),
    }


def archived_paths(value, scratch: Path, clip_id: str):
    """Replace worker-local paths with tar-member references before publication.

    Payloads stay structured; no artifacts or media are extracted to per-clip directories
    by readers. Original sensor paths outside scratch are left unchanged.
    """
    if isinstance(value, str):
        # The compiler trace uses playground-relative paths, while its source catalog
        # and context use absolute paths. Bind both to retained media, not scratch.
        for prefix in (str(scratch) + "/", scratch.name + "/"):
            value = value.replace(prefix, f"media.tar::{clip_id}/")
        return value
    if isinstance(value, list):
        return [archived_paths(item, scratch, clip_id) for item in value]
    if isinstance(value, dict):
        return {key: archived_paths(item, scratch, clip_id) for key, item in value.items()}
    return value


def media_manifest(run_dir: Path) -> dict | None:
    """Digest the optional run-owned media archive without loading it into memory."""
    path = run_dir / "media.tar"
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": digest}


def read_media(publication: Path, references: list[str]) -> dict[str, bytes]:
    """Verify an archive once and read a batch of members without extracting files.

    Args:
        publication: Native run's finalized ``clips/`` publication.
        references: ``media.tar::member`` references from its structured payloads.

    Returns:
        Reference-to-bytes mapping for the requested regular files only.

    Raises:
        ValueError: Missing or changed archive, unsafe reference, or non-regular member.
    """
    manifest = json.loads((publication / "manifest.json").read_text())
    if manifest["complete"] is not True or manifest["payload_schema"] != CLIP_SCHEMA:
        raise ValueError("media requires a finalized clip publication")
    expected = manifest["metadata"]["media"]
    if expected is None or expected != media_manifest(publication.parent):
        raise ValueError("media archive hash or size mismatch")
    images = {}
    with tarfile.open(publication.parent / "media.tar", "r") as archive:
        for reference in references:
            prefix, separator, name = reference.partition("::")
            if (
                prefix != "media.tar"
                or not separator
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise ValueError("unsafe media reference")
            member = archive.getmember(name)
            if not member.isfile():
                raise ValueError("media member must be a regular file")
            with archive.extractfile(member) as handle:
                images[reference] = handle.read()
    return images


def storage_summary(run_dir: Path, expected_clips: int) -> dict:
    """Measure generation artifacts, excluding this measurement's own report file.

    Files/bytes and actual unique inodes include the root and its directories. This is
    a generation-time snapshot; later evaluation files are not silently folded into it.
    """
    if expected_clips <= 0:
        raise ValueError("storage denominator must be positive")
    paths = [run_dir] + [
        path for path in run_dir.rglob("*") if path != run_dir / "storage_summary.json"
    ]
    files = [path for path in paths if path.is_file()]
    size = sum(path.stat().st_size for path in files)
    return {
        "scope": "generation artifacts excluding storage_summary.json",
        "expected_clips": expected_clips,
        "files": len(files),
        "directories": sum(path.is_dir() for path in paths),
        "inodes": len({(path.lstat().st_dev, path.lstat().st_ino) for path in paths}),
        "bytes": size,
        "bytes_per_expected_clip": size / expected_clips,
        "parquet_shards": len(list((run_dir / "clips").glob("part-*.parquet"))),
    }
