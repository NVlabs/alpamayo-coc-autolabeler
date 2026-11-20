"""Validate and publish streaming labels in the public Parquet/CSV format."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .....infra.artifacts import sync_directory
from ...schema.contracts import Timeline


def _public_tags(tags: str | None) -> tuple[str, ...]:
    """Resolve the interchange vocabulary consistently during validation and writing."""
    return tuple(
        dict.fromkeys(
            "nav_not_default" if tag.strip() == "nav_missing_not_default" else tag.strip()
            for tag in (tags or "").split(",")
            if tag.strip()
        )
    )


@contextmanager
def _export_directory(destination: Path):
    """Publish complete export files together, removing temporary files after errors."""
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as name:
        temporary = Path(name)
        yield temporary
        for path in temporary.iterdir():
            with path.open("rb") as source:
                os.fsync(source.fileno())
        sync_directory(temporary)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        sync_directory(destination.parent)


def public_label_issues(rows: list[dict], tolerance_us: int = 1000) -> list[str]:
    """Validate human-compatible interval rows without normalizing their semantic text."""
    issues = []
    if not rows:
        return ["empty clip"]
    previous = None
    origins = []
    for index, row in enumerate(rows):
        start, end = row["start_timestamp_seconds"], row["end_timestamp_seconds"]
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)):
            issues.append(f"nonfinite time at {index}")
            continue
        if end <= start or start < 0:
            issues.append(f"invalid interval at {index}")
        if type(row["start_us"]) is not int or type(row["end_us"]) is not int:
            issues.append(f"noninteger clock at {index}")
        origins.extend((row["start_us"] - round(start * 1e6), row["end_us"] - round(end * 1e6)))
        # Human UI indices are edit identities, not chronological ranks after splits.
        # Sort on time, as the maintained reader does; do not rewrite those identities.
        if not isinstance(row["cot"], str) or not row["cot"].strip():
            issues.append(f"empty CoC at {index}")
        tags = _public_tags(row["tags"])
        unknown = set(tags) - {
            "nominal",
            "interesting",
            "challenging",
            "unsafe",
            "nav_not_default",
            "unknown",
            "egomotion errors",
            "nav error",
            "exogenous_action_change",
        }
        if unknown:
            issues.append(f"unknown tags at {index}: {sorted(unknown)}")
        if "exogenous_action_change" in tags and (
            index == 0 or "nominal" in tags or row["cot"].strip() == "<NOMINAL>"
        ):
            issues.append(f"invalid exogenous marker at {index}")
        if previous is None:
            if abs(start * 1e6) > tolerance_us:
                issues.append("first interval does not start at zero")
        else:
            if abs((previous["end_timestamp_seconds"] - start) * 1e6) > tolerance_us:
                issues.append(f"gap/overlap at {index}")
            previous_tags = _public_tags(previous["tags"])
            if previous["cot"] == row["cot"] and previous_tags == tags:
                issues.append(f"adjacent identical public intervals at {index}")
        previous = row
    if origins and max(origins) - min(origins) > 2:
        issues.append("inconsistent absolute/relative clock")
    return issues


def write_annotation(record: dict, out_dir: Path, diagnostics: bool = False) -> None:
    """Write application annotations with an optional, separate internal audit record.

    Args:
        record: A saved successful clip record, including its graph snapshot.
        out_dir: New destination directory; existing destinations are refused.
        diagnostics: Include the unchanged internal record in diagnostics.json.

    Raises:
        ValueError: Generation failed, or the saved annotation is structurally inconsistent.
    """
    result = record["result"]
    if record["status"] not in {"complete", "review_needed"} or result["status"] not in {
        "complete",
        "review_needed",
    }:
        raise ValueError("cannot export an unsuccessful generation")
    timeline = Timeline.from_dict(result["timeline"])
    if (
        record["clip_id"] != timeline.clip_id
        or record["snapshot"]["graph_hash"] != result["parent_graph_hash"]
    ):
        raise ValueError("annotation identity does not match its saved graph")
    bundles = {bundle["interval_id"]: bundle for bundle in result["bundles"]}
    if len(bundles) != len(result["bundles"]) or set(bundles) != {
        item.interval_id for item in timeline.intervals
    }:
        raise ValueError("annotation bundles do not match its intervals")
    intervals = []
    for item in result["timeline"]["intervals"]:
        bundle = bundles[item["interval_id"]]
        if not isinstance(bundle["coc"], str) or not bundle["coc"].strip():
            raise ValueError("annotation requires nonempty captions")
        intervals.append(
            {
                **{key: value for key, value in item.items() if key != "status"},
                "coc": bundle["coc"],
                "evidence_view": bundle["view"],
            }
        )
    annotation = {
        "schema": "alparsg_annotation_v1",
        "clip_id": timeline.clip_id,
        "generation_status": "complete",
        "evidence_scope": "full_clip_teacher",
        "clock": result["timeline"]["clock"],
        "clock_source": result["clock_source"],
        "clock_verified": result["clock_verified"],
        "parent_graph_hash": result["parent_graph_hash"],
        "graph_snapshot": record["snapshot"],
        "intervals": intervals,
    }
    payload = json.dumps(annotation, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    audit = (
        json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if diagnostics
        else None
    )
    with _export_directory(out_dir) as temporary:
        (temporary / "annotation.json").write_text(payload)
        if audit is not None:
            (temporary / "diagnostics.json").write_text(audit)


def write_public_store(result: dict, out_dir: Path) -> None:
    """Export completed interval bundles in the human tool's Parquet/CSV format.

    Refuse unsuccessful generation, invalid interchange rows or existing destinations.
    Internal review verdicts are not caption-quality gates. Input stores are never modified.
    """
    if result["status"] not in {"complete", "review_needed"}:
        raise ValueError("cannot export an unsuccessful generation")
    timeline = result["timeline"]
    clip_id = timeline["clip_id"]
    origin = timeline["clock"]["origin_us"]
    revision = result["parent_graph_hash"]
    updated = datetime.now(timezone.utc).isoformat()
    cots = {bundle["interval_id"]: bundle["coc"] for bundle in result["bundles"]}
    stem, _, suffix = clip_id.rpartition("_")
    key = stem if stem and suffix.isdigit() else clip_id
    rows, boundaries = [], []
    for index, item in enumerate(timeline["intervals"]):
        start, end = item["interval_start_us"], item["interval_end_us"]
        rows.append(
            {
                "key": key,
                "start_us": start,
                "end_us": end,
                "cot": cots[item["interval_id"]],
                "clipid": clip_id,
                "interval_idx": index,
                "start_timestamp_seconds": (start - origin) / 1e6,
                "end_timestamp_seconds": (end - origin) / 1e6,
                "revision": revision,
                "tags": ",".join(_public_tags(",".join(item["tags"]))),
                "updated_at": updated,
            }
        )
        if index:
            boundaries.append(
                {
                    "clipid": clip_id,
                    "keyframe_idx": index - 1,
                    "timestamp_seconds": (start - origin) / 1e6,
                    "absolute_timestamp_us": start,
                    "revision": revision,
                    "updated_at": updated,
                }
            )
    issues = public_label_issues(rows, tolerance_us=0)
    if issues:
        raise ValueError(f"invalid public labels: {issues}")
    schema = pa.schema(
        [
            (name, dtype)
            for name, dtype in (
                ("key", pa.string()),
                ("start_us", pa.int64()),
                ("end_us", pa.int64()),
                ("cot", pa.string()),
                ("clipid", pa.string()),
                ("interval_idx", pa.int64()),
                ("start_timestamp_seconds", pa.float64()),
                ("end_timestamp_seconds", pa.float64()),
                ("revision", pa.string()),
                ("tags", pa.string()),
                ("updated_at", pa.string()),
            )
        ]
    )
    with _export_directory(out_dir) as temporary:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temporary / "labels_cot_intervals.parquet"
        )
        with (temporary / "labels_cot_keyframes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "clipid",
                    "keyframe_idx",
                    "timestamp_seconds",
                    "absolute_timestamp_us",
                    "revision",
                    "updated_at",
                ],
            )
            writer.writeheader()
            writer.writerows(boundaries)
