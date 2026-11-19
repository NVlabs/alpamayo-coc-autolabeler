"""Bounded streaming label workers and compact resumable publications."""

from __future__ import annotations

import copy
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def read_attempts(root: Path) -> list[dict]:
    """Read actual transport receipts once, whether serial or worker-sharded."""
    return [
        json.loads(line)
        for name in ("usage.jsonl",)
        for path in sorted(root.rglob(name))
        for line in path.read_text().splitlines()
    ]


def run_parallel(
    generation: dict,
    root: Path,
    workers: int,
    *,
    refresh_accounting: bool = False,
    recovery_source: Path | None = None,
    allocation_owned: bool = False,
) -> dict:
    """Run fixed strided shards with one controller-owned request admission per phase.

    Each worker owns one compact run, never a persistent directory per clip. The parent
    publication retains prediction/snapshot records and points to full worker provenance.
    Worker failures remain fixed-population outcomes; they never select a survivor cohort.
    Successful receipts are reused on explicit resume by the underlying task functions.
    """
    if refresh_accounting:
        raise ValueError(
            "Public runs require caller-approved budgets; provider refresh is unavailable"
        )
    import numpy as np
    import yaml

    from ....infra.artifacts import CompactArtifactJournal, write_json_atomic
    from ....infra.model import ready_endpoints
    from .configuration import digest, generation_identity, validate_setup
    from .cost import (
        AllocationGateError,
        RequestAdmission,
        generation_allocation,
        resolved_attempts,
        summarize_attempts,
    )
    from .export.storage import CLIP_SCHEMA, load_clip_records, storage_summary, summarize_outcomes
    from .interface import run

    validate_setup(generation)
    if workers < 1:
        raise ValueError("workers must be positive")
    ready_endpoints(generation["endpoint_registry"])
    items = sorted(
        json.loads(Path(generation["input_manifest"]).read_text())["clips"],
        key=lambda row: row["clip_id"],
    )
    ids = [item["clip_id"] for item in items]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("parallel population must contain unique clips")
    workers = min(workers, len(items))
    sources = (
        load_clip_records(Path(generation["source_run"])) if generation["mode"] != "native" else {}
    )
    root.mkdir(parents=True, exist_ok=True)
    identity = {
        "workers": workers,
        "ids": ids,
        "input_hash": digest(items),
        "generation": generation_identity(generation, {"clips": items}, sources)["sha256"],
        "recovery_source": str(recovery_source.resolve()) if recovery_source else None,
    }
    identity_path = root / "parallel_identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("parallel membership or worker assignment changed")
    write_json_atomic(identity_path, identity)

    generation = copy.deepcopy(generation)
    shards = []
    for index in range(workers):
        directory = root / "workers" / f"worker_{index:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        selected = items[index::workers]
        selected_ids = [row["clip_id"] for row in selected]
        write_json_atomic(directory / "inputs.json", {"clips": selected})
        config = copy.deepcopy(generation)
        config["input_manifest"] = str(directory / "inputs.json")
        if sources:
            source = CompactArtifactJournal(directory / "source", CLIP_SCHEMA)
            if not (directory / "source/manifest.json").exists():
                for cid in selected_ids:
                    source.record(cid, sources[cid])
                source.finalize(selected_ids, {"parent": generation["source_run"]})
            elif source.completed_records() != {cid: sources[cid] for cid in selected_ids}:
                raise ValueError("parallel source content changed")
            config["source_run"] = str(directory / "source")
        (directory / "generation.yaml").write_text(yaml.safe_dump(config))
        shards.append(directory)
    failures = {}

    def generate(directory):
        """Contain worker exceptions while preserving all available compact outcomes."""
        out = directory / "run"
        try:
            run(
                directory / "generation.yaml",
                out_dir=out,
                resume=(out / "generation_identity.json").exists(),
                admission=admission,
                recovery_source=recovery_source / "workers" / directory.name / "run"
                if recovery_source is not None
                else None,
            )
            return None
        except AllocationGateError:
            raise
        except Exception as error:
            return type(error).__name__

    generation_started = time.monotonic()
    admission = RequestAdmission(
        root / "generation_admission.json",
        generation["budget"]["dispatch_envelope_usd"],
        1.0,
        initial_attempts=[
            a
            for path in sorted(recovery_source.rglob("usage.jsonl"))
            for a in resolved_attempts(path, generation["budget"])
            if a["stage"] == "generation"
        ]
        if recovery_source is not None
        else (),
    )
    with generation_allocation(
        root, generation["budget"], request_admission=admission, allocation_owned=allocation_owned
    ) as admission:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for directory, failure in zip(shards, pool.map(generate, shards), strict=True):
                if failure:
                    failures[directory.name] = {"generation": failure}
    records = {}
    for directory in shards:
        publication = directory / "run/clips"
        if publication.exists():
            try:
                records.update(load_clip_records(publication))
            except FileNotFoundError:
                pass  # Failure before publication is an explicit missing outcome below.
    merged = {}
    for cid in ids:
        row = records.get(cid, {"clip_id": cid, "status": "not_run"})
        merged[cid] = {
            k: v
            for k, v in row.items()
            if k
            in {
                "clip_id",
                "status",
                "result",
                "snapshot",
                "error_type",
                "error_trace",
                "external_intervals",
                "benchmark_clock_projection",
            }
        }
    if (root / "clips/manifest.json").exists() and load_clip_records(root / "clips") != merged:
        archive = root / "prior_aggregate_clips"
        if archive.exists():
            raise RuntimeError("bounded aggregate retry exhausted")
        (root / "clips").rename(archive)
    if not (root / "clips/manifest.json").exists():
        publication = CompactArtifactJournal(root / "clips", CLIP_SCHEMA)
        for cid, row in merged.items():
            publication.record(cid, row)
        publication.finalize(ids, {"worker_runs": [str(p / "run") for p in shards], "media": None})
    generation_attempts = [a for a in read_attempts(root) if a["stage"] == "generation"]
    # Deterministic exports and exact cache reuse can complete without an HTTP request.
    attempted_ids = {a["clip_id"] for a in generation_attempts} | {
        cid for cid, row in records.items() if row["status"] != "not_run"
    }
    costs = summarize_attempts(
        generation_attempts,
        len(attempted_ids),
        sum(r["status"] == "complete" for r in records.values()),
    )
    costs["expected_clips"] = len(ids)
    durations = [row["elapsed_s"] for row in records.values() if "elapsed_s" in row]
    costs["clip_wall_time_s"] = {
        "p50": float(np.quantile(durations, 0.5)) if durations else None,
        "p95": float(np.quantile(durations, 0.95)) if durations else None,
        "measured_clips": len(durations),
    }
    costs["controller_wall_time_s"] = time.monotonic() - generation_started
    costs["workers"] = workers
    write_json_atomic(root / "cost_summary.json", costs)
    write_json_atomic(root / "worker_failures.json", failures)
    write_json_atomic(root / "storage_summary.json", storage_summary(root, len(ids)))
    return {"workers": workers, "failures": failures, **summarize_outcomes(merged)}


def main() -> None:
    """Run bounded parallel generation with the supplied local generation config."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="local generation YAML or JSON")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="parent run directory; rerun unchanged to resume its workers",
    )
    parser.add_argument(
        "--workers", type=int, required=True, help="fixed worker count retained across resumes"
    )
    parser.add_argument(
        "--recovery-source",
        type=Path,
        help="prior parallel parent for recovery into a new output directory",
    )
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"generation config file does not exist: {args.config}")
    from .configuration import load_config, validate_setup

    try:
        generation = load_config(args.config)
        validate_setup(generation)
    except ValueError as error:
        parser.error(str(error))

    summary = run_parallel(
        generation,
        args.out,
        args.workers,
        recovery_source=args.recovery_source,
    )
    print(json.dumps(summary))
    if not summary["success"] or summary["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
