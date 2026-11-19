"""Full-clip sensor reader for matched joint/independent action reconstruction."""

import numpy as np

from .....streaming_rsg.ops.executor import run_program
from .....streaming_rsg.ops.read_snapshot import ReadTimingEvidenceOp
from ..frame_sampling import encode_timed_records


def read_full_action_context(
    env, anchor, orchestrator, rsg, clock, cache_dir, detail_stride_us=None
):
    """Use the existing read-only evidence op; never rebind or write a canonical graph.

    The caller owns the declared environment and all model/API accounting. This function
    returns physical observations plus route/production-label availability, not semantic cuts.
    Source paths remain node-local until the normal compact media publication.
    """
    if anchor.trigger != "discovery" or env.sampling.max_fps < 5:
        raise ValueError("full action evidence needs declared >=5Hz discovery context")
    camera = env.data.cameras["camera_front_wide_120fov"]
    captures = np.loadtxt(camera["timestamps"], dtype=np.int64, ndmin=2)[:, -1]
    if len(captures) < 2 or np.any(np.diff(captures) <= 0):
        raise ValueError("physical capture times must increase")
    end_us = (
        env.video_binding["clock"]["end_us"]
        if env.video_binding is not None
        else int(captures[-1]) + int(round(float(np.median(np.diff(captures)))))
    )
    extent = (int(captures[0]), end_us)
    if np.any(captures >= end_us):
        raise ValueError("physical captures must precede the bound clip end")
    if extent != (clock.origin_us, clock.end_us):
        raise ValueError("action environment differs from verified physical clip clock")
    op = ReadTimingEvidenceOp(extent, [extent], cache_dir, True, detail_stride_us=detail_stride_us)
    rsg, steps = run_program(env, anchor, [op], orchestrator, rsg)
    if not op.records:
        raise ValueError("full action evidence unavailable under environment budget")
    images, records = encode_timed_records(op.records, clock)
    if not images:
        raise ValueError("full action evidence requires actual camera captures")
    navigation = env.navigation_records("agent", anchor)
    records.extend(row.to_dict() for row in navigation)
    production = env.meta_action_record("agent", anchor)
    return (
        rsg,
        steps,
        {
            "env_id": env.env_id,
            "environment_contract": env.contract_dict(),
            "records": records,
            "images": images,
            "production_action_context": None
            if production is None
            else {"record_id": production.id, **production.payload},
            "navigation_context_available": bool(navigation),
            "usage": "full_clip_teacher_not_policy_observation",
        },
    )
