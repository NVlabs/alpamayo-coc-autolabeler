"""Streaming RSG/CoC generation and resumable publication."""

from __future__ import annotations

import json
import math
import shutil
import tarfile
import tempfile
import time
import traceback
from copy import deepcopy
from functools import partial
from pathlib import Path

import yaml

from ....common.provenance import ANCHORING_STREAMING, applies_to
from ....infra.artifacts import CompactArtifactJournal, write_json_atomic, write_text_atomic
from ....infra.model import ChatClient, ready_endpoints
from ..schema.contracts import ClipClock, Timeline
from .configuration import (
    digest,
    endpoint_settings,
    generation_identity,
    load_config,
    validate_setup,
)
from .cost import (
    AllocationGateError,
    RequestAccounting,
    attempt_budget_cost,
    authorize_dispatch,
    clip_generation_spend,
    generation_allocation,
    resolved_attempts,
    summarize_attempts,
    timeout_reservations_path,
)
from .export.coc_copyedit import copyedit_interval_cocs
from .export.export_policy import apply_nominal_output_policy
from .export.storage import (
    CLIP_SCHEMA,
    archived_paths,
    load_clip_records,
    media_manifest,
    storage_summary,
    summarize_outcomes,
)
from .pipeline import DEFAULT_INTERVAL_PROMPT, export_timeline, label_snapshot, validate_partition
from .recorded_completion import recorded_completion

REWARD_SOURCE = "clip_interval_semantics"
applies_to(ANCHORING_STREAMING)


def config_fingerprint(config_path, *, overrides=None) -> dict:
    """Expose the same content-bound recipe identity to ordinary Alpaloop run indexing."""
    config = load_config(Path(config_path), overrides)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    sources = load_clip_records(Path(config["source_run"])) if config["mode"] != "native" else {}
    return generation_identity(config, manifest, sources)


def run(
    config_path, *, out_dir, overrides=None, resume=False, admission=None, recovery_source=None
) -> dict:
    """Validate, restore and execute clips inside one controller-owned allocation.

    Args:
        config_path: Generation recipe to load before validating its population and policy.
        out_dir: Persistent journals, accounting and compact publication directory.
        overrides: Explicit recipe overrides applied by the existing config loader.
        resume: Reuse successful records from the same generation identity.
        admission: Owning parallel controller's admission, or None for a standalone owner.
        recovery_source: Registered prior run whose successful records and receipts are reused.

    Returns:
        Fixed-population counts distinguishing successful, failed and undispatched clips.
    """
    config = load_config(Path(config_path), overrides)
    manifest, policy = _validate_generation_inputs(config)
    root = Path(out_dir)
    with generation_allocation(
        root, config["budget"], request_admission=admission, allocation_owned=admission is not None
    ) as controller_admission:
        source_records = _load_source_records(config, manifest["clips"])
        prior, identity = _restore_generation(
            root, config, manifest, source_records, resume, recovery_source
        )
        ids = [item["clip_id"] for item in manifest["clips"]]
        if (
            resume
            and set(prior) == set(ids)
            and all(row["status"] in {"complete", "review_needed"} for row in prior.values())
        ):
            if (
                not (root / "storage_summary.json").is_file()
                or not (root / "clips/manifest.json").is_file()
            ):
                # Durable clips need publication metadata, without renewed dispatch authorization.
                usage_path = root / "usage.jsonl"
                attempts = (
                    [json.loads(line) for line in usage_path.read_text().splitlines()]
                    if usage_path.exists()
                    else []
                )
                return _finalize_publication(
                    root,
                    manifest["clips"],
                    CompactArtifactJournal(root / "clips", CLIP_SCHEMA),
                    [
                        {"clip_id": cid, "status": row["status"], "reused": True}
                        for cid, row in prior.items()
                    ],
                    attempts,
                )
            return summarize_outcomes(prior)
        from ....infra.budget import CostRecord

        journal, run_costs = _open_publication(
            root, config, manifest, identity, prior, resume, recovery_source
        )
        budget = config["budget"]
        outcomes = [
            {"clip_id": cid, "status": r["status"], "reused": True}
            for cid, r in prior.items()
            if r["status"] in {"complete", "review_needed"}
        ]
        usage_path = root / "usage.jsonl"
        all_attempts = resolved_attempts(usage_path, budget)
        if any(attempt_budget_cost(row) is None for row in all_attempts):
            raise RuntimeError("reconcile unknown run costs before resuming dispatch")
        for item in manifest["clips"]:
            if any(row["clip_id"] == item["clip_id"] for row in outcomes):
                continue
            outcome, attempts = _run_clip(
                item,
                config,
                policy,
                source_records,
                identity,
                controller_admission,
                root,
                all_attempts,
                journal,
            )
            outcomes.append(outcome)
            all_attempts.extend(attempts)
            write_json_atomic(root / "outcomes.json", {"clips": outcomes})
            summary = summarize_attempts(attempts, 1, int(outcome["status"] == "complete"))
            run_costs.append(
                CostRecord(
                    owner=budget["owner"],
                    project="alparsg",
                    campaign=budget["campaign"],
                    stage="generation",
                    candidate=root.name,
                    provider="configured_registry",
                    model=",".join(sorted({row["model"] for row in attempts})),
                    estimated_usd=summary["budget_cost_usd"],
                    confirmed_usd=summary["total_cost_usd"],
                    requests=len(attempts),
                    status="completed" if outcome["status"] == "complete" else "failed",
                )
            )
            # Unknown billing permits the already authorized one-clip measurement pilot,
            # but does not propagate into an unpriced multi-clip run.
            if summary["unreserved_unknown_requests"]:
                break
            if (
                sum(attempt_budget_cost(row) for row in all_attempts)
                >= budget["dispatch_envelope_usd"]
            ):
                break
        return _finalize_publication(root, manifest["clips"], journal, outcomes, all_attempts)


def _validate_generation_inputs(config):
    """Validate the input population and policy before allocation or output creation."""
    if (
        config.get("allow_paid_dispatch") is not True
        or not config.get("input_manifest")
        or str(config["input_manifest"]).startswith(("/absolute/path/to/", "/path/to/"))
    ):
        validate_setup(config)
    manifest = json.loads(Path(config["input_manifest"]).read_text())
    clips = manifest["clips"]
    ids = [item["clip_id"] for item in clips]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("input manifest must contain unique clips")
    partition_diagnostic = config["mode"] == "partition_diagnostic"
    allowed_fields = {"clip_id", "clock", "clock_verified", "sample_anchors_us"}
    if partition_diagnostic:
        allowed_fields.add("diagnostic_partition_us")
        if config.get("diagnostic_partition_source") not in {"human", "saved_prediction"}:
            raise ValueError("partition diagnostic requires a declared timestamp source")
        for item in clips:
            if "diagnostic_partition_us" not in item:
                raise ValueError("partition diagnostic is missing timestamp pairs")
            validate_partition(item["diagnostic_partition_us"], ClipClock(**item["clock"]))
    elif "diagnostic_partition_source" in config:
        raise ValueError("partition source is only allowed in partition_diagnostic mode")
    if any(set(item) - allowed_fields for item in clips):
        raise ValueError(
            "generation manifest contains undeclared fields (possibly reference labels)"
        )
    if config["mode"] not in {
        "frozen_graph",
        "frozen_intervals",
        "native",
        "boundary_replay",
        "partition_diagnostic",
        "episode_replay",
        "constraint_replay",
        "wording_replay",
        "wording_edit",
        "context_replay",
    }:
        raise ValueError("unknown generation mode")
    optional_modules = (
        (
            "boundary replay",
            config["mode"] == "boundary_replay",
            "interpretation/boundary_evidence.py",
        ),
        (
            "constraint replay",
            config["mode"] == "constraint_replay",
            "interpretation/constraint_review.py",
        ),
        (
            "boundary localization",
            bool(config["policy"].get("boundary_localization")),
            "interpretation/boundary_localization.py",
        ),
        (
            "constraint review",
            bool(config["policy"].get("constraint_review")),
            "interpretation/constraint_review.py",
        ),
    )
    for feature, requested, module in optional_modules:
        if requested and not Path(__file__).resolve().parent.joinpath(module).is_file():
            raise ValueError(f"{feature} is not included in this package")
    wording_mode = config["mode"] in {"wording_replay", "wording_edit"}
    if (
        config["mode"] != "native"
        and not wording_mode
        and any(not item["clock_verified"] for item in clips)
    ):
        raise ValueError("frozen-graph generation requires verified physical clocks")
    policy = {**config["policy"]}
    output_nominal = policy.get("output_nominal", not policy.get("coc_public_contract", False))
    if type(output_nominal) is not bool:
        raise ValueError("output_nominal must be Boolean")
    if not output_nominal and not policy.get("coc_public_contract", False):
        raise ValueError("descriptive nominal output requires the public-action contract")
    if type(policy.get("coc_copyedit", False)) is not bool:
        raise ValueError("coc_copyedit must be Boolean")
    qualified_context = policy.get("coc_qualified_context", False)
    if type(qualified_context) is not bool:
        raise ValueError("qualified context must be Boolean")
    if qualified_context and config["mode"] != "native":
        raise ValueError("qualified context on fresh outputs requires native generation")
    copyedit_style = policy.get("coc_copyedit_style", "exact_action_composition_v1")
    if copyedit_style not in {
        "exact_action_composition_v1",
        "verified_fluent_v1",
        "verified_fluent_exact_v1",
        "verified_cause_first_v1",
        "verified_source_components_v1",
        "verified_event_account_v1",
        "verified_readable_account_v1",
        "verified_readable_decision_account_v1",
        "restore_selected_original_v1",
        "verified_local_deletion_v1",
        "verified_local_reorder_v1",
    }:
        raise ValueError("unknown CoC copyedit style")
    if "coc_copyedit_style" in policy and not policy.get("coc_copyedit", False):
        raise ValueError("copyedit style requires explicit copyediting")
    omit_clock_times = policy.get("coc_omit_clock_times", False)
    if type(omit_clock_times) is not bool or (
        omit_clock_times
        and (
            not policy.get("coc_copyedit", False)
            or copyedit_style
            not in {"verified_readable_account_v1", "verified_readable_decision_account_v1"}
        )
    ):
        raise ValueError("clock-free prose requires explicit readable copyediting")
    if (config["mode"] == "context_replay" or qualified_context) and (
        not policy.get("coc_copyedit", False)
        or copyedit_style
        not in {"verified_readable_account_v1", "verified_readable_decision_account_v1"}
    ):
        raise ValueError("qualified context requires the complete READABLE contract")
    if (
        config["mode"] == "wording_replay"
        and copyedit_style
        not in {
            "exact_action_composition_v1",
            "verified_fluent_exact_v1",
            "restore_selected_original_v1",
        }
    ) or (
        config["mode"] == "wording_edit"
        and copyedit_style
        not in {
            "verified_fluent_v1",
            "verified_fluent_exact_v1",
            "verified_cause_first_v1",
            "verified_source_components_v1",
            "verified_local_deletion_v1",
            "verified_local_reorder_v1",
            "verified_event_account_v1",
            "verified_readable_account_v1",
            "verified_readable_decision_account_v1",
        }
    ):
        raise ValueError("wording mode and generation-call contract disagree")
    if type(policy.get("export_coc", True)) is not bool:
        raise ValueError("export_coc must be a boolean")
    if not policy.get("export_coc", True) and (
        policy.get("coc_copyedit", False)
        or policy.get("coc_repair", False)
        or policy.get("reuse_grounded_draft", False)
        or config["mode"] in {"wording_replay", "wording_edit", "context_replay", "boundary_replay"}
    ):
        raise ValueError("RSG-only export cannot run narration or wording/timing replay")
    if policy.get("coc_copyedit", False) and (
        policy.get("coc_renderer") != "grounded" or not policy.get("coc_public_contract", False)
    ):
        raise ValueError("copyediting requires the existing grounded public-action contract")
    if config["mode"] in {"wording_replay", "wording_edit", "context_replay"} and (
        not policy.get("coc_copyedit", False)
        or any(
            policy.get(key)
            for key in (
                "coverage_review",
                "constraint_review",
                "boundary_localization",
                "episode_consolidation",
                "action_reconstruction",
                "coc_repair",
                "reuse_grounded_draft",
                "retain_refinement_context",
                "interpreter_max_frames",
                "coc_prompt",
            )
        )
    ):
        raise ValueError("public export replay cannot relabel actions or intervals")
    policy["interval_prompt"] = policy["interval_prompt"] or DEFAULT_INTERVAL_PROMPT
    construction = policy.get("interval_construction", "joint_v1")
    if construction not in ("joint_v1", "skeleton_v1"):
        raise ValueError("interval_construction must be joint_v1 or skeleton_v1")
    if config["mode"] != "native" and construction != "joint_v1":
        raise ValueError("opt-in interval construction requires native generation")
    action_mode = policy.get("action_reconstruction")
    if action_mode not in {None, "joint_dense", "independent_dense"}:
        raise ValueError("unknown action reconstruction mode")
    if action_mode and (
        config["mode"] != "native"
        or policy.get("interval_construction", "joint_v1") != "joint_v1"
        or policy.get("interpreter_max_frames", 0)
    ):
        raise ValueError(
            "dense action reconstruction requires native joint mode and dedicated evidence"
        )
    image_detail = policy.get("action_image_detail")
    if image_detail not in (None, "sentinel_2s"):
        raise ValueError("unknown action image detail policy")
    if image_detail and (
        action_mode != "independent_dense" or policy.get("action_camera_sampling") != "uniform_2hz"
    ):
        raise ValueError("detail snapshots require independent uniform2Hz action reconstruction")
    if type(policy.get("coverage_review", False)) is not bool:
        raise ValueError("coverage_review must be Boolean")
    if type(policy.get("retain_refinement_context", False)) is not bool:
        raise ValueError("retain_refinement_context must be Boolean")
    if type(policy.get("constraint_review", False)) is not bool:
        raise ValueError("constraint_review must be Boolean")
    if policy.get("constraint_review", False) and config["mode"] not in {
        "native",
        "frozen_graph",
        "constraint_replay",
    }:
        raise ValueError("constraint review requires native/frozen_graph/constraint_replay mode")
    if config["mode"] == "constraint_replay" and (
        not policy.get("constraint_review", False)
        or policy.get("coverage_review", False)
        or policy.get("boundary_localization")
        or policy.get("episode_consolidation")
        or policy.get("reuse_grounded_draft", False)
    ):
        raise ValueError("constraint replay changes causes only on a saved action partition")
    if config["mode"] in {"frozen_intervals", "boundary_replay"} and policy.get(
        "coverage_review", False
    ):
        raise ValueError("frozen-interval export cannot revise decision coverage")
    if policy.get("episode_consolidation") is not None and config["mode"] in {
        "frozen_intervals",
        "boundary_replay",
        "partition_diagnostic",
    }:
        raise ValueError("consolidation requires native/frozen_graph/episode_replay mode")
    if config["mode"] == "episode_replay" and (
        policy.get("episode_consolidation") not in {"short", "semantic", "combined", "state"}
        or policy.get("coverage_review", False)
        or policy.get("boundary_localization")
    ):
        raise ValueError("episode replay only consolidates the saved timeline")
    if policy.get("coc_renderer", "legacy") not in {"legacy", "action_first", "grounded"}:
        raise ValueError("unknown CoC renderer")
    if policy.get("coc_renderer", "legacy") != "legacy" and policy["coc_prompt"] is not None:
        raise ValueError("explicit renderer cannot combine an arbitrary narrator prompt")
    if policy.get("episode_consolidation") not in {None, "short", "semantic", "combined", "state"}:
        raise ValueError("unknown episode consolidation mode")
    if type(policy.get("coc_repair", False)) is not bool:
        raise ValueError("coc_repair must be Boolean")
    if type(policy.get("coc_public_contract", False)) is not bool:
        raise ValueError("coc_public_contract must be Boolean")
    if policy.get("coc_factor_contract", "clause") not in {"span", "clause", "span_readable"} or (
        policy.get("coc_factor_contract", "clause") != "clause"
        and policy.get("coc_renderer") != "grounded"
    ):
        raise ValueError("unknown coc_factor_contract or non-grounded renderer")
    if policy.get("coc_public_contract", False) and (
        policy.get("coc_renderer") != "grounded" or policy.get("reuse_grounded_draft", False)
    ):
        raise ValueError("public contract requires fresh grounded narration")
    if policy.get("coc_repair", False) and policy.get("coc_renderer") != "grounded":
        raise ValueError("causal repair requires grounded rendering")
    if type(policy.get("reuse_grounded_draft", False)) is not bool:
        raise ValueError("reuse_grounded_draft must be Boolean")
    if policy.get("reuse_grounded_draft", False) and (
        config["mode"] != "frozen_intervals" or not policy.get("coc_repair", False)
    ):
        raise ValueError("exact draft repair requires frozen intervals and grounded repair")
    if config["mode"] != "native" and policy["max_refinements"] != 0:
        raise ValueError("frozen graph arm cannot request compiler refinements")
    if partition_diagnostic and (
        policy.get("coverage_review", False)
        or policy.get("retain_refinement_context", False)
        or policy.get("interpreter_max_frames", 0)
        or policy.get("boundary_localization")
        or policy["coc_prompt"] is not None
        or policy.get("coc_renderer", "legacy") != "legacy"
    ):
        raise ValueError("partition diagnosis requires frozen evidence and template export")
    validate_setup(config)
    ready_endpoints(config["endpoint_registry"])
    return manifest, policy


def _load_source_records(config, clips):
    """Load saved graph records and enforce the declared wording-replay clock contract."""
    wording_mode = config["mode"] in {"wording_replay", "wording_edit", "context_replay"}
    source_records = (
        load_clip_records(Path(config["source_run"])) if config["mode"] != "native" else {}
    )
    if wording_mode:
        for item in clips:
            if not item["clock_verified"]:
                source = source_records[item["clip_id"]]
                projection = source.get("benchmark_clock_projection", {})
                saved = source["result"]
                clock_source = saved.get("clock_source", {})
                projected = (
                    projection.get("kind") == "post_generation_frame_identity_alignment"
                    and projection.get("physical_alignment_visible_to_model") is False
                    and projection.get("raw_graph_timeline_and_text_unchanged") is True
                    and bool(projection.get("alignment_sha256"))
                    and len(source.get("external_intervals", []))
                    == len(saved["timeline"]["intervals"])
                )
                application_video = (
                    "benchmark_clock_projection" not in source
                    and "external_intervals" not in source
                    and clock_source.get("kind") == "video_presentation_timestamps"
                    and clock_source.get("presentation_timing_verified") is True
                    and clock_source.get("externally_verified_sensor_time") is False
                    and all(clock_source.get(k) == v for k, v in item["clock"].items())
                    and all(
                        s.get("trajectory_status") == "unavailable_video_only"
                        and "future_trajectory" not in s
                        for s in saved["samples"]
                    )
                )
                if (
                    not (projected or application_video)
                    or saved.get("clock_verified") is not False
                    or saved["timeline"]["clock"] != item["clock"]
                    or item["clock"]["origin_us"] != 0
                    or any(
                        s["status"] != "excluded" or s["training_ready"] or s["coc"] is not None
                        for s in saved["samples"]
                    )
                ):
                    raise ValueError(
                        "video wording replay requires the saved private clock projection "
                        "or verified application presentation timing with excluded samples"
                    )
    return source_records


def _restore_generation(root, config, manifest, source_records, resume, recovery_source):
    """Validate recovery lineage and load prior records without rewriting publications."""
    identity = generation_identity(config, manifest, source_records)
    recovery = Path(recovery_source) if recovery_source is not None else None
    recovery_records: dict[str, dict] = {}
    if recovery is not None:
        previous_identity = json.loads((recovery / "generation_identity.json").read_text())
        if (
            previous_identity["components"]["endpoint_settings"]
            != identity["components"]["endpoint_settings"]
        ):
            raise ValueError("recovery endpoint request semantics changed")
        previous_config = yaml.safe_load((recovery / "config.yaml").read_text())
        previous_manifest = json.loads((recovery / "input_manifest.json").read_text())
        if previous_manifest != manifest:
            raise ValueError("recovery input membership or clocks changed")
        ignored = {
            "budget",
            "input_manifest",
            "reference_manifest",
            "compiler_config",
            "scratch_dir",
            "max_requests_per_clip",
            "max_generation_cost_per_clip_usd",
            "source_run",
            "max_resume_attempts",
        }
        if {k: v for k, v in previous_config.items() if k not in ignored} != {
            k: v for k, v in config.items() if k not in ignored
        }:
            raise ValueError("recovery must preserve generation semantics")
        if (
            config["mode"] != "native"
            and load_clip_records(Path(previous_config["source_run"])) != source_records
        ):
            raise ValueError("recovery source graph records changed")
        if config["mode"] in {"native", "boundary_replay"} and yaml.safe_load(
            Path(previous_config["compiler_config"]).read_text()
        ) != yaml.safe_load(Path(config["compiler_config"]).read_text()):
            raise ValueError("recovery compiler configuration changed")
        recovery_records = load_clip_records(recovery / "clips", allow_empty=True)
        identity["recovery"] = {
            "source": str(recovery.resolve()),
            "identity": digest(json.loads((recovery / "generation_identity.json").read_text())),
            "publication": digest(recovery_records),
        }
    identity_path = root / "generation_identity.json"
    prior = {}
    if identity_path.exists():
        if not resume:
            raise FileExistsError("generation exists; use explicit resume")
        previous_identity = json.loads(identity_path.read_text())
        if previous_identity != identity:
            previous = previous_identity.get("components")
            changed = (
                [key for key, value in identity["components"].items() if previous.get(key) != value]
                if isinstance(previous, dict)
                else ["metadata"]
            )
            if previous_identity.get("recovery") != identity.get("recovery"):
                changed.append("recovery")
            raise ValueError(
                f"generation identity drift ({', '.join(changed or ['metadata'])}); use a new run"
            )
        prior = (
            load_clip_records(root / "clips", allow_empty=True) if (root / "clips").exists() else {}
        )
        missing = {item["clip_id"] for item in manifest["clips"]} - set(prior)
        if missing:
            for archived in sorted(
                (root / "prior_publications").glob("attempt_*/clips"), reverse=True
            ):
                for clip_id, payload in load_clip_records(archived).items():
                    if clip_id in missing and payload["status"] in {"complete", "review_needed"}:
                        prior[clip_id] = payload
                        missing.remove(clip_id)
                if not missing:
                    break
    elif resume:
        raise ValueError("resume requires an existing generation identity")
    elif recovery is not None:
        prior = recovery_records
    return prior, identity


def _open_publication(root, config, manifest, identity, prior, resume, recovery_source):
    """Authorize publication, preserve prior attempts and open the compact journal."""
    identity_path = root / "generation_identity.json"
    recovery = Path(recovery_source) if recovery_source is not None else None
    # A reviewed allocation is required even for the first measurement pilot. It is a
    # conservative dispatch envelope, not a promise of exact per-request dollar capping.
    from ....infra.budget import CostLedger

    budget = config["budget"]
    authorize_dispatch(budget)
    root.mkdir(parents=True, exist_ok=True)
    if not resume and {p.name for p in root.iterdir()} - {"manifest.yaml"}:
        raise FileExistsError("generation directory is not empty")
    if recovery is not None and not resume:
        for name in ("usage.jsonl", "media.tar"):
            if (recovery / name).exists():
                temporary = root / f".{name}.recovery"
                shutil.copyfile(recovery / name, temporary)
                temporary.replace(root / name)
    write_json_atomic(identity_path, identity)
    # Preserve every published attempt. Retry only failed/not-run clips; no per-clip
    # persistent directories and no deletion/rewrite of earlier publications.
    if (root / "clips/manifest.json").exists():
        archive_root = root / "prior_publications"
        archive_root.mkdir(exist_ok=True)
        attempt_index = len(list(archive_root.iterdir()))
        if attempt_index >= config.get("max_resume_attempts", 1):
            raise RuntimeError("bounded run retry exhausted")
        archived = archive_root / f"attempt_{attempt_index:03d}"
        archived.mkdir()
        if (root / "media.tar").exists():
            shutil.copyfile(root / "media.tar", archived / "media.tar")
        (root / "clips").rename(archived / "clips")
    # The final write marks completion; a retry cannot reuse an earlier attempt's marker.
    (root / "storage_summary.json").unlink(missing_ok=True)
    run_costs = CostLedger(root / "cost_records.jsonl")
    journal = CompactArtifactJournal(root / "clips", CLIP_SCHEMA)
    for clip_id, payload in prior.items():
        if (
            payload["status"] in {"complete", "review_needed"}
            and journal.completed_payload(clip_id) != payload
        ):
            journal.record(clip_id, payload)
    write_text_atomic(root / "config.yaml", yaml.safe_dump(config, sort_keys=False))
    write_text_atomic(root / "input_manifest.json", json.dumps(manifest, sort_keys=True))
    return journal, run_costs


def _run_clip(
    item, config, policy, source_records, identity, admission, root, all_attempts, journal
):
    """Execute one clip and publish available artifacts before releasing its scratch directory."""
    clip_id = item["clip_id"]
    if Path(clip_id).name != clip_id or clip_id in {".", ".."}:
        raise ValueError("unsafe clip id")
    budget = config["budget"]
    envelope = budget["dispatch_envelope_usd"]
    with tempfile.TemporaryDirectory(
        prefix="alparsg-coc-clip-", dir=config["scratch_dir"]
    ) as directory:
        clip_out = Path(directory)
        accounting = RequestAccounting(
            root / "usage.jsonl",
            clip_id,
            "generation",
            config["max_requests_per_clip"],
            admission=admission,
            timeout_reservations=timeout_reservations_path(budget),
            max_cost_usd=config.get("max_generation_cost_per_clip_usd"),
        )
        accounting.reserved = sum(row["clip_id"] == clip_id for row in all_attempts)
        accounting.inherited_cost_usd = sum(
            clip_generation_spend(row) for row in all_attempts if row["clip_id"] == clip_id
        )

        def before_attempt():
            """Recheck model identity and the run envelope before every HTTP attempt."""
            if config["mode"] == "wording_replay":
                raise RuntimeError("wording replay must not make generation API calls")
            if (
                endpoint_settings(config["endpoint_registry"])
                != identity["components"]["endpoint_settings"]
            ):
                raise RuntimeError("registry model changed during the run")
            if any(
                attempt_budget_cost(row) is None for row in [*all_attempts, *accounting.attempts]
            ):
                raise AllocationGateError(
                    "reconcile unknown generation cost before further dispatch"
                )
            spent = sum(attempt_budget_cost(row) for row in [*all_attempts, *accounting.attempts])
            if spent >= envelope:
                raise AllocationGateError("run allocation exhausted")
            accounting.before_attempt()

        client = ChatClient(
            config["endpoint_registry"],
            max_tokens=config["max_output_tokens"],
            temperature=config["temperature"],
            on_attempt=accounting.record,
            before_attempt=before_attempt,
            transient_retries=2,
        )
        started = time.monotonic()
        payload = {"clip_id": clip_id}
        try:
            status = _generate_clip(item, config, policy, source_records, client, clip_out, payload)
            outcome = {"clip_id": clip_id, "status": status}
        except AllocationGateError as error:
            outcome = {"clip_id": clip_id, "status": "not_run", "allocation_gate": str(error)}
            _publish_clip(root, clip_out, payload, outcome, accounting.attempts, journal)
            raise
        except Exception as error:
            # Error text from a provider may contain private prompts; keep the type here.
            outcome = {
                "clip_id": clip_id,
                "status": "failed",
                "error_type": type(error).__name__,
            }
            outcome["error_trace"] = [
                {
                    "file": Path(frame.filename).name,
                    "function": frame.name,
                    "line": frame.lineno,
                }
                for frame in traceback.extract_tb(error.__traceback__)
            ]
        outcome["elapsed_s"] = time.monotonic() - started
        _publish_clip(root, clip_out, payload, outcome, accounting.attempts, journal)
    return outcome, accounting.attempts


def _generate_clip(item, config, policy, source_records, client, clip_out, payload):
    """Generate or replay one result and apply the declared public text/context policy."""
    clip_id = item["clip_id"]
    wording_mode = config["mode"] in {"wording_replay", "wording_edit", "context_replay"}
    qualified_context = policy.get("coc_qualified_context", False)
    copyedit_style = policy.get("coc_copyedit_style", "exact_action_composition_v1")
    if config["mode"] == "native":
        result = _generate_native_clip(item, config, policy, client, clip_out)
    else:
        result, snapshot = _label_saved_clip(
            item, config, policy, source_records, client, clip_out, payload
        )
    if result["timeline"]["clock"] != item["clock"]:
        raise ValueError("generated physical clock differs from frozen input manifest")
    if [r["sample_anchor_us"] for r in result["samples"]] != item["sample_anchors_us"]:
        raise ValueError("generated anchors differ from frozen input manifest")
    result = apply_nominal_output_policy(
        result,
        output_nominal=policy.get("output_nominal", not policy.get("coc_public_contract", False)),
    )
    if config["mode"] != "context_replay" and policy.get("coc_copyedit", False):
        result = copyedit_interval_cocs(
            result,
            style=copyedit_style,
            complete=recorded_completion(client.complete, clip_out / "copyedit_calls.jsonl", {}),
            omit_clock_times=policy.get("coc_omit_clock_times", False),
        )
    if config["mode"] == "context_replay" or qualified_context:
        from .export.qualified_context import export_qualified_context

        if config["mode"] == "native":
            snapshot = json.loads((clip_out / "snapshot.json").read_text())
        if snapshot["graph_hash"] != result["parent_graph_hash"]:
            raise ValueError("qualified context graph differs from the labeled graph")
        result = export_qualified_context(
            result,
            snapshot,
            recorded_completion(client.complete, clip_out / "interpreter_calls.jsonl", {}),
            copyedit_style,
            omit_clock_times=policy.get("coc_omit_clock_times", False),
        )
    payload["result"] = result
    if wording_mode and not item["clock_verified"]:
        source = source_records[clip_id]
        if result["timeline"] != source["result"]["timeline"]:
            raise ValueError("wording replay changed the saved video timeline")
        annotations = {b["interval_id"]: b["coc"] for b in result["bundles"]}
        if "benchmark_clock_projection" in source:
            payload["external_intervals"] = deepcopy(source["external_intervals"])
            for public, interval in zip(
                payload["external_intervals"],
                result["timeline"]["intervals"],
                strict=True,
            ):
                public["cot"] = annotations[interval["interval_id"]]
            payload["benchmark_clock_projection"] = deepcopy(source["benchmark_clock_projection"])
    return result["status"]


def _generate_native_clip(item, config, policy, client, clip_out):
    """Compile and interpret one native clip with the existing client and policy."""
    clip_id = item["clip_id"]
    action_mode = policy.get("action_reconstruction")
    from ....streaming_rsg.runtime.config import load_config as compiler_config
    from ....streaming_rsg.runtime.run import run as compile_clip
    from .runtime import interpret_clip, validate_native_inputs

    compiler = compiler_config(
        Path(config["compiler_config"]),
        [
            f"infra.endpoint_registry={config['endpoint_registry']}",
            f"policy.generation.max_tokens={config['max_output_tokens']}",
            f"policy.generation.temperature={config['temperature']}",
        ],
    )
    if compiler.playground.env.anchor_mode != "discovery" or compiler.playground.task != "none":
        raise ValueError("native clip task needs discovery and compiler task: none")
    if policy.get("boundary_localization") and compiler.playground.env.max_fps < 5:
        raise ValueError("native timing localization requires a declared >=5Hz environment")
    if action_mode and (
        compiler.playground.env.max_fps < 5
        or compiler.playground.env.max_frames
        < math.ceil(
            (item["clock"]["end_us"] - item["clock"]["origin_us"])
            / 1e6
            * compiler.playground.env.max_fps
        )
    ):
        raise ValueError("dense action reconstruction needs a full-clip 5Hz/frame allowance")
    if compiler.infra.grounding_registry is not None:
        raise ValueError("pilot accounting does not yet cover separate grounding endpoints")
    compile_clip(
        clip_id,
        clip_out,
        compiler,
        client=client,
        publish_viewer=False,
        interpreter=partial(interpret_clip, out_dir=clip_out, policy=policy),
        validate_inputs=partial(validate_native_inputs, item=item, policy=policy),
    )
    result = yaml.safe_load((clip_out / "task.yaml").read_text())
    return result


def _label_saved_clip(item, config, policy, source_records, client, clip_out, payload):
    """Interpret or replay a saved graph without changing its declared source identity."""
    clip_id = item["clip_id"]
    partition_diagnostic = config["mode"] == "partition_diagnostic"
    wording_mode = config["mode"] in {"wording_replay", "wording_edit", "context_replay"}
    complete = recorded_completion(client.complete, clip_out / "interpreter_calls.jsonl", {})
    snapshot = source_records[clip_id]["snapshot"]
    payload["snapshot"] = snapshot
    payload["source"] = {
        "run": config["source_run"],
        "record_id": clip_id,
        "graph_hash": snapshot["graph_hash"],
    }
    if config["mode"] in {"wording_replay", "wording_edit", "context_replay"}:
        result = source_records[clip_id]["result"]
        timeline = Timeline.from_dict(result["timeline"])
        if (
            result["parent_graph_hash"] != snapshot["graph_hash"]
            or timeline.clock != ClipClock(**item["clock"])
            or result.get("clock_verified") is not item["clock_verified"]
        ):
            raise ValueError("wording replay graph/clock identity mismatch")
    elif config["mode"] in {
        "frozen_intervals",
        "boundary_replay",
        "episode_replay",
        "constraint_replay",
    }:
        source = source_records[clip_id]["result"]
        timeline = Timeline.from_dict(source["timeline"])
        if source["parent_graph_hash"] != snapshot["graph_hash"] or timeline.clock != ClipClock(
            **item["clock"]
        ):
            raise ValueError("frozen interval graph/clock identity mismatch")
        timing_receipt = None
        consolidation_receipt = None
        constraint_receipt = None
        if config["mode"] == "constraint_replay":
            from .interpretation.constraint_review import (
                attach_constraint_receipt,
                reconcile_constraints,
            )

            timeline, constraint_receipt = reconcile_constraints(timeline, snapshot, complete)
        if config["mode"] == "episode_replay":
            from .interpretation.episode_consolidation import (
                attach_consolidation_receipt,
                consolidate_timeline,
            )

            timeline, consolidation_receipt = consolidate_timeline(
                timeline, snapshot, complete, policy["episode_consolidation"]
            )
        if config["mode"] == "boundary_replay":
            from .interpretation.boundary_evidence import localize_saved
            from .interpretation.boundary_localization import attach_timing_receipt

            if policy["coc_prompt"] is not None:
                raise ValueError("timing isolation must preserve the template CoC export")
            timeline, timing_receipt = localize_saved(
                timeline,
                snapshot,
                client.complete,
                Path(config["compiler_config"]),
                clip_out,
                policy["boundary_localization"],
            )
        result = export_timeline(
            timeline,
            snapshot,
            complete,
            item["sample_anchors_us"],
            coc_prompt=policy["coc_prompt"],
            allowed_source_classes=tuple(policy["allowed_source_classes"]),
            unresolved_inspections=any(
                "requested evidence refinement is unresolved" in bundle["issues"]
                for bundle in source["bundles"]
            ),
            include_neighbors=policy.get("include_neighbors", False),
            coc_renderer=policy.get("coc_renderer", "legacy"),
            coc_repair=policy.get("coc_repair", False),
            coc_public_contract=policy.get("coc_public_contract", False),
            coc_factor_contract=policy.get("coc_factor_contract", "clause"),
            export_coc=policy.get("export_coc", True),
            coc_drafts={b["interval_id"]: b for b in source["bundles"]}
            if policy.get("reuse_grounded_draft", False)
            else None,
        )
        result["upstream_rounds"] = source["rounds"]
        result["teacher_inputs"] = source.get("teacher_inputs", {}).copy()
        if timing_receipt is not None:
            attach_timing_receipt(result, snapshot, timing_receipt)
            if [b["coc"] for b in result["bundles"]] != [b["coc"] for b in source["bundles"]]:
                raise ValueError("timing replay changed public CoC content")
        trajectories = {r["sample_anchor_us"]: r for r in source["samples"]}
        for sample in result["samples"]:
            previous = trajectories.get(sample["sample_anchor_us"], {})
            for key in ("future_trajectory", "trajectory_status"):
                if key in previous:
                    sample[key] = previous[key]
            if sample["trajectory_status"] == "incomplete":
                sample["status"], sample["coc"] = "excluded", None
                sample["issues"].append("future trajectory does not cover target window")
            sample["training_ready"] = (
                sample["status"] == "eligible"
                and sample["trajectory_status"] == "attached_world_poses"
            )
            if previous.get("status") == "excluded":
                sample["issues"].extend(previous["issues"])
                sample["status"], sample["coc"], sample["training_ready"] = (
                    "excluded",
                    None,
                    False,
                )
        if consolidation_receipt is not None:
            attach_consolidation_receipt(result, consolidation_receipt)
        if constraint_receipt is not None:
            attach_constraint_receipt(result, constraint_receipt)
    else:
        result = label_snapshot(
            clip_id,
            ClipClock(**item["clock"]),
            snapshot,
            complete,
            item["sample_anchors_us"],
            max_refinements=0,
            interval_prompt=policy["interval_prompt"],
            coc_prompt=policy["coc_prompt"],
            allowed_source_classes=tuple(policy["allowed_source_classes"]),
            include_neighbors=policy.get("include_neighbors", False),
            coverage_review=policy.get("coverage_review", False),
            retain_refinement_context=policy.get("retain_refinement_context", False),
            fixed_partition_us=item["diagnostic_partition_us"] if partition_diagnostic else None,
            episode_consolidation=policy.get("episode_consolidation"),
            constraint_review=policy.get("constraint_review", False),
            coc_renderer=policy.get("coc_renderer", "legacy"),
            coc_repair=policy.get("coc_repair", False),
            coc_public_contract=policy.get("coc_public_contract", False),
            coc_factor_contract=policy.get("coc_factor_contract", "clause"),
            export_coc=policy.get("export_coc", True),
        )
        if partition_diagnostic:
            result["diagnostic_only"] = True
            result["teacher_inputs"]["diagnostic_partition"] = {
                "source": config["diagnostic_partition_source"],
                "partition_us": item["diagnostic_partition_us"],
                "usage": "conditional_analysis_not_training_or_promotion",
            }
            for sample in result["samples"]:
                sample["status"] = "excluded"
                sample["coc"] = None
                sample["training_ready"] = False
                sample["issues"].append("fixed-partition diagnostic is not training data")
    if not wording_mode:
        result["clock_verified"] = True
    return result, snapshot


def _publish_clip(root, clip_out, payload, outcome, attempts, journal):
    """Archive available clip artifacts and receipts before scratch cleanup."""
    clip_id = payload["clip_id"]
    # Preserve compiler outputs in the clip record; only media go in the run's tar.
    # A worker has at most one clip's loose artifacts, on node-local scratch.
    # task.yaml is already stored as result; do not duplicate every interval/sample.
    for name in ("rsg.yaml", "config.yaml", "trace.json", "snapshot.json"):
        path = clip_out / name
        if path.is_file():
            payload[name.split(".")[0]] = yaml.safe_load(path.read_text())
    for name in ("context", "interpreter_calls", "copyedit_calls"):
        path = clip_out / f"{name}.jsonl"
        if path.is_file():
            payload[name] = [json.loads(line) for line in path.read_text().splitlines()]
    media_root = clip_out / "frames"
    media_members = []
    if media_root.is_dir():
        with tarfile.open(root / "media.tar", "a") as archive:
            for path in sorted(media_root.rglob("*")):
                if path.is_file():
                    member = f"{clip_id}/{path.relative_to(clip_out).as_posix()}"
                    archive.add(path, arcname=member, recursive=False)
                    media_members.append(member)
    payload.update({**outcome, "usage": attempts, "media_members": media_members})
    journal.record(clip_id, archived_paths(payload, clip_out, clip_id))


def _finalize_publication(root, clips, journal, outcomes, all_attempts):
    """Publish fixed-population outcomes and accounting, including clips left undispatched."""
    ids = [item["clip_id"] for item in clips]
    summary = summarize_attempts(
        all_attempts,
        len(outcomes),
        sum(row["status"] == "complete" for row in outcomes),
    )
    summary["expected_clips"] = len(ids)
    summary["complete_coverage"] = summary["valid_clips"] == len(ids)
    import numpy as np

    durations = [row["elapsed_s"] for row in outcomes if "elapsed_s" in row]
    summary["clip_wall_time_s"] = {
        "p50": float(np.quantile(durations, 0.5)) if durations else None,
        "p95": float(np.quantile(durations, 0.95)) if durations else None,
        "measured_clips": len(durations),
        "reused_clips": sum(row.get("reused", False) for row in outcomes),
    }
    produced = journal.completed_records()
    samples = [
        sample
        for row in produced.values()
        if row.get("result")
        for sample in row["result"]["samples"]
    ]
    eligible = sum(sample["status"] == "eligible" for sample in samples)
    summary["eligible_anchors"] = eligible
    summary["expected_anchors"] = sum(len(row["sample_anchors_us"]) for row in clips)
    summary["usd_per_eligible_anchor"] = (
        summary["total_cost_usd"] / eligible
        if eligible and summary["total_cost_usd"] is not None
        else None
    )
    for clip_id in ids:
        if any(row["clip_id"] == clip_id for row in outcomes):
            continue
        journal.record(
            clip_id,
            {
                "clip_id": clip_id,
                "status": "not_run",
                "reason": "dispatch stopped at accounting/allocation gate",
            },
        )
    publication = journal.finalize(
        ids,
        metadata={
            "annotation_complete_clips": summary["valid_clips"],
            "expected_clips": len(ids),
            "cost_summary": summary,
            "media": media_manifest(root),
        },
    )
    # A durable manifest retains its original accounting when metadata finalization resumes.
    summary = publication["metadata"]["cost_summary"]
    write_json_atomic(root / "cost_summary.json", summary)
    records = journal.completed_records()
    by_id = {row["clip_id"]: row for row in outcomes}
    write_json_atomic(
        root / "outcomes.json",
        {
            "clips": [
                by_id.get(cid, {"clip_id": cid, "status": records[cid]["status"]}) for cid in ids
            ],
            **summarize_outcomes(records),
        },
    )
    coverage = [json.dumps({"clips_total": len(ids)})]
    for cid in ids:
        row = records[cid]
        coverage.append(
            json.dumps(
                {
                    "clip_id": cid,
                    "status": "ok" if row["status"] == "complete" else "partial",
                    "error_type": row.get("error_type", row["status"]),
                }
            )
        )
    write_text_atomic(root / "clips.jsonl", "\n".join(coverage) + "\n")
    write_json_atomic(root / "storage_summary.json", storage_summary(root, len(ids)))
    return summarize_outcomes(records)
