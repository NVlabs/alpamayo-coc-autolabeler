"""The full-clip interpreter's bridge to the existing compiler and its operation gateway."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ....streaming_rsg.ops.executor import run_program
from ....streaming_rsg.ops.observation import ObserveFocusOp
from ....streaming_rsg.ops.read_snapshot import ReadEvidenceOp, ReadSnapshotOp
from ..schema.contracts import ClipClock
from .frame_sampling import physical_focus, sample_camera_context, validate_camera_brackets
from .interpretation.action_input_context import read_full_action_context
from .interpretation.action_phases import (
    action_semantic_context,
    action_semantic_prompt,
    dense_semantic_context,
    observe_action_phases,
)
from .pipeline import label_snapshot
from .recorded_completion import recorded_completion


def physical_clock(env, anchor) -> ClipClock:
    """Derive the camera extent, including the final frame's physical display period."""
    camera = env.data.cameras["camera_front_wide_120fov"]
    timestamps = np.loadtxt(camera["timestamps"], dtype=np.int64, ndmin=2)
    captures = timestamps[:, -1]
    if len(captures) < 2 or np.any(np.diff(captures) <= 0):
        raise ValueError("camera timestamps must be strictly increasing")
    # Public label-store convention: include the display period of the final frame.
    period_us = max(1, int(round(float(np.median(np.diff(captures))))))
    clock = ClipClock(int(captures[0]), int(captures[-1]) + period_us)
    if env.video_binding is not None:
        clock = ClipClock(**env.video_binding["clock"])
    if not anchor.window_us[0] <= clock.origin_us < int(captures[-1]) <= anchor.window_us[1]:
        raise ValueError("compiler extent does not cover physical camera clock")
    return clock


def validate_native_inputs(env, anchor, item: dict, policy: dict) -> None:
    """Reject clock or sample-anchor drift before any compiler request can be dispatched."""
    clock = physical_clock(env, anchor)
    if policy.get("action_camera_sampling") in {"uniform_2hz", "uniform_3hz"}:
        camera = env.data.cameras["camera_front_wide_120fov"]
        captures = np.loadtxt(camera["timestamps"], dtype=np.int64, ndmin=2)[:, -1]
        validate_camera_brackets(captures)
    if {"origin_us": clock.origin_us, "end_us": clock.end_us} != item["clock"]:
        raise ValueError("generated physical clock differs from frozen input manifest")
    if clock.sample_anchors(policy["anchor_stride_us"]) != item["sample_anchors_us"]:
        raise ValueError("generated anchors differ from frozen input manifest")


def interpret_clip(  # noqa: PLR0917 -- compiler interpreter callback plus task configuration
    env, anchor, orchestrator, rsg, client, out_dir: Path, policy: dict
):
    """Interpret/refine a discovery graph in-place through existing ops, then freeze it.

    Clock comes from physical camera captures, never a human reference or metadb offset.
    The original compiler must cover that range. Graph reads and refinements share its
    cumulative budget. The task adds no decision components or new graph representation.
    """
    if anchor.trigger != "discovery":
        raise ValueError("full-clip interpretation requires a discovery compiler anchor")
    construction = policy.get("interval_construction", "joint_v1")
    if construction not in ("joint_v1", "skeleton_v1"):
        raise ValueError("interval_construction must be joint_v1 or skeleton_v1")
    action_mode = policy.get("action_reconstruction")
    if action_mode not in {None, "joint_dense", "independent_dense"}:
        raise ValueError("unknown action reconstruction mode")
    if action_mode and (construction != "joint_v1" or policy.get("interpreter_max_frames", 0)):
        raise ValueError(
            "dense action reconstruction uses joint intervals and its own declared evidence"
        )
    camera_sampling = policy.get("action_camera_sampling")
    if camera_sampling not in (None, "uniform_3hz", "uniform_2hz", "selective_semantic"):
        raise ValueError("unknown action camera sampling mode")
    if camera_sampling is not None and action_mode != "independent_dense":
        raise ValueError("camera sampling requires independent action reconstruction")
    image_detail = policy.get("action_image_detail")
    if image_detail not in (None, "sentinel_2s"):
        raise ValueError("unknown action image detail policy")
    if image_detail and (
        camera_sampling != "uniform_2hz" or env.sampling.image_overview_height_px is None
    ):
        raise ValueError("detail snapshots require uniform2Hz and a mixed-resolution environment")
    camera_selection = {"mode": camera_sampling, "physical": None, "semantic": []}
    inspection_windows = []
    clock = physical_clock(env, anchor)
    all_steps = []
    visual_limit = policy.get("interpreter_max_frames", 0)
    if type(visual_limit) is not int or not 0 <= visual_limit <= 24:
        raise ValueError("interpreter_max_frames must be an integer between 0 and 24")
    image_receipts = {}

    def read_evidence():
        """Deliver timestamped existing captures; persist references instead of duplicate pixels."""
        nonlocal rsg
        op = ReadEvidenceOp((clock.origin_us, clock.end_us), visual_limit)
        rsg, steps = run_program(env, anchor, [op], orchestrator, rsg)
        all_steps.extend(steps)
        if op.records is None:
            raise RuntimeError("compiler budget exhausted before image evidence read")
        content, records = [], []
        for record in op.records:
            data = Path(record["path"]).read_bytes()
            receipt = {**record, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            url = "data:image/jpeg;base64," + base64.b64encode(data).decode()
            image_receipts[url] = receipt
            records.append(receipt)
            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"Front-wide capture {record['id']} at "
                        f"{record['timestamp_us']} us (absolute physical clip clock).",
                    },
                    {"type": "image_url", "image_url": {"url": url}},
                ]
            )
        return content, records

    def read_snapshot():
        """Capture a versioned live graph using the operation gateway."""
        nonlocal rsg
        op = ReadSnapshotOp()
        rsg, steps = run_program(env, anchor, [op], orchestrator, rsg)
        all_steps.extend(steps)
        if op.snapshot is None:
            raise RuntimeError("compiler budget exhausted before graph snapshot")
        # Retain a valid compiler product even if the downstream interval parser fails.
        (out_dir / "snapshot.json").write_text(json.dumps(op.snapshot, sort_keys=True))
        return op.snapshot

    def refine(requests):
        """Convert interval uncertainty into existing focused observation operations."""
        nonlocal rsg
        if camera_sampling == "selective_semantic":
            inspection_windows.extend(tuple(request["window_us"]) for request in requests)
        for request in requests:
            op = ObserveFocusOp(
                client.complete,
                out_dir / "frames",
                max_requests=1,
                scope=request["scope"],
            )
            op.pending_requests = [
                (
                    request["component_id"],
                    tuple(request["window_us"]),
                    request["reason"],
                    request["frames"],
                )
            ]
            rsg, steps = run_program(env, anchor, [op], orchestrator, rsg)
            all_steps.extend(steps)
            if not steps:
                raise RuntimeError("compiler budget exhausted during requested refinement")
        return read_snapshot()

    def invoke(messages):
        """Preserve the compiler wall-clock and existing metered client boundary."""
        if (
            orchestrator.budget_deadline is not None
            and time.monotonic() >= orchestrator.budget_deadline
        ):
            raise RuntimeError("compiler wall-clock budget exhausted before interpreter call")
        return client.complete(messages)

    complete = recorded_completion(invoke, out_dir / "interpreter_calls.jsonl", image_receipts)

    def localize_timing(timeline, snapshot):
        """Localize after semantic commitment using the same op gateway and cumulative budget."""
        from .interpretation.boundary_evidence import read_timing_context
        from .interpretation.boundary_localization import localize_boundaries

        nonlocal rsg
        mode = policy["boundary_localization"]
        if mode not in {"motion", "visual"}:
            raise ValueError("boundary localization must be motion or visual")
        rsg, steps, context = read_timing_context(
            env, anchor, orchestrator, rsg, timeline, out_dir / "frames", mode == "visual"
        )
        all_steps.extend(steps)
        urls = [p["image_url"]["url"] for p in context["images"] if p["type"] == "image_url"]
        frames = [r for r in context["records"] if r["stream"].startswith("camera/")]
        image_receipts.update(zip(urls, frames, strict=True))
        localized, receipt = localize_boundaries(timeline, snapshot, context, complete)
        receipt["environment_contract"] = context.get("environment_contract")
        return localized, receipt

    anchors = clock.sample_anchors(policy["anchor_stride_us"])
    snapshot = read_snapshot()
    initial_graph_hash = snapshot["graph_hash"]
    resolved_prompt = policy["interval_prompt"]
    evidence_reader = read_evidence if visual_limit else None
    action_receipt = None
    dense_context = None
    if action_mode:
        rsg, steps, dense_context = read_full_action_context(
            env,
            anchor,
            orchestrator,
            rsg,
            clock,
            out_dir / "frames",
            **({"detail_stride_us": 2_000_000} if image_detail else {}),
        )
        all_steps.extend(steps)
        urls = [p["image_url"]["url"] for p in dense_context["images"] if p["type"] == "image_url"]
        frames = [r for r in dense_context["records"] if r["stream"].startswith("camera/")]
        image_receipts.update(zip(urls, frames, strict=True))
        physical_context = dense_context
        if camera_sampling is not None:
            physical_hz = {"uniform_3hz": 3, "uniform_2hz": 2, "selective_semantic": 5}[
                camera_sampling
            ]
            physical_context, camera_selection["physical"] = sample_camera_context(
                dense_context, clock, physical_hz
            )
        semantic_context = dense_semantic_context(
            physical_context, dense_context["production_action_context"]
        )
        if action_mode == "independent_dense":
            physical, action_receipt = observe_action_phases(
                env.clip_id,
                clock,
                physical_context,
                dense_context["production_action_context"],
                complete,
            )
            resolved_prompt, evidence_reader = action_semantic_context(
                physical, action_receipt, semantic_context, resolved_prompt
            )
            if camera_sampling is not None:
                ordinary_reader = evidence_reader

                def sampled_semantic_evidence():
                    """Select exact semantic frames while retaining full physical observations."""
                    if camera_sampling == "selective_semantic":
                        windows, required = physical_focus(
                            dense_context, action_receipt["observation"], clock, inspection_windows
                        )
                        selected, receipt = sample_camera_context(
                            dense_context, clock, 2, windows, required
                        )
                    else:
                        selected = physical_context
                        receipt = camera_selection["physical"]
                    semantic_context.clear()
                    semantic_context.update(
                        dense_semantic_context(selected, dense_context["production_action_context"])
                    )
                    camera_selection["semantic"].append(receipt)
                    return ordinary_reader()

                evidence_reader = sampled_semantic_evidence
        else:
            resolved_prompt = action_semantic_prompt(resolved_prompt)

            def evidence_reader():
                """Reuse the same prepared semantic evidence without another source read."""
                return semantic_context["images"], semantic_context["records"]

    result = label_snapshot(
        env.clip_id,
        clock,
        snapshot,
        complete,
        anchors,
        max_refinements=policy["max_refinements"],
        refine=refine,
        interval_prompt=resolved_prompt,
        coc_prompt=policy["coc_prompt"],
        allowed_source_classes=tuple(policy["allowed_source_classes"]),
        include_neighbors=policy.get("include_neighbors", False),
        read_evidence=evidence_reader,
        coverage_review=policy.get("coverage_review", False),
        localize_timing=localize_timing if policy.get("boundary_localization") else None,
        retain_refinement_context=policy.get("retain_refinement_context", False),
        commit_semantic_skeleton=construction == "skeleton_v1",
        inspection_max_frames=min(8, env.sampling.max_frames)
        if construction == "skeleton_v1"
        else 8,
        episode_consolidation=policy.get("episode_consolidation"),
        constraint_review=policy.get("constraint_review", False),
        coc_renderer=policy.get("coc_renderer", "legacy"),
        coc_repair=policy.get("coc_repair", False),
        coc_public_contract=policy.get("coc_public_contract", False),
        coc_factor_contract=policy.get("coc_factor_contract", "clause"),
        export_coc=policy.get("export_coc", True),
    )
    if dense_context is not None:
        result["teacher_inputs"]["dense_action_evidence"] = {
            "env_id": dense_context["env_id"],
            "environment_contract": dense_context["environment_contract"],
            "records": dense_context["records"],
            "independent_pass": action_mode == "independent_dense",
            "initial_graph_hash": initial_graph_hash,
            "final_graph_hash": result["parent_graph_hash"],
            "graph_rebound": False,
            "usage": "full_clip_teacher_not_policy_observation",
        }
        if camera_sampling is not None:
            result["teacher_inputs"]["dense_action_evidence"]["camera_selection"] = camera_selection
        if action_receipt is not None:
            result["teacher_inputs"]["actual_action_observation"] = action_receipt
        for sample in result["samples"]:
            sample["issues"].append(
                "dense action interpretation consumed full-clip teacher evidence"
            )
            sample["status"], sample["coc"], sample["training_ready"] = "excluded", None, False
    # Snapshot the final graph for downstream-only experiments and audit/review.
    final_snapshot = read_snapshot()
    if final_snapshot["graph_hash"] != result["parent_graph_hash"]:
        raise ValueError("graph changed after interval freeze")
    (out_dir / "snapshot.json").write_text(json.dumps(final_snapshot, sort_keys=True))
    for sample in result["samples"]:
        lo, hi = sample["future_window_us"]
        trajectory = env.data.ego
        if trajectory is None:
            sample["status"], sample["coc"] = "excluded", None
            sample["trajectory_status"] = "unavailable_video_only"
            sample["issues"].append("video-only input provides no measured future world poses")
        elif int(trajectory["t_us"].min()) > lo or int(trajectory["t_us"].max()) < hi:
            sample["status"] = "excluded"
            sample["coc"] = None
            sample["issues"].append("future trajectory does not cover target window")
            sample["trajectory_status"] = "incomplete"
        else:
            sample["trajectory_status"] = "attached_world_poses"
            sample["future_trajectory"] = trajectory[
                (trajectory["t_us"] > lo) & (trajectory["t_us"] <= hi)
            ][["t_us", "x", "y", "z", "qw", "qx", "qy", "qz"]].to_dict("records")
        sample["training_ready"] = (
            sample["status"] == "eligible" and sample["trajectory_status"] == "attached_world_poses"
        )
    result["clock_verified"] = env.video_binding is None
    result["clock_source"] = {
        "camera": "camera_front_wide_120fov",
        "origin_us": clock.origin_us,
        "end_us": clock.end_us,
    }
    if env.video_binding is not None:
        result["clock_source"].update(
            kind="video_presentation_timestamps",
            presentation_timing_verified=True,
            externally_verified_sensor_time=False,
        )
    return rsg, all_steps, result
