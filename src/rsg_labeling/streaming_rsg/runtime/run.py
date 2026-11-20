"""The native run: env source records through the component op program.

ground (ego, agents, map; state/maneuver events) -> observe:frames (VLM over decoded frames,
enriching components or grounding observed ones) -> prune:unobserved -> propose:action ->
verify:proposal, executed under the env budgets with the agent layer. The artifact carries the
env contract, the source catalog the run actually read, and the component graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from ...infra.model import ChatClient
from ..agent.orchestrator import Orchestrator
from ..agent.tools import make_grounding_tool, make_render_tool
from ..agent.tools.ground_frame import (
    agent_ground_regions,
    da3_geometry,
    ground_detections,
    judge_appearance,
    track_regions,
    track_window,
)
from ..data_structure.rsg import Rsg
from ..env.env import build_env_native
from ..ops.executor import run_program
from ..ops.grounding import (
    GroundAgentsOp,
    GroundEgoOp,
    GroundMapOp,
    ParseManeuverOp,
    RelateSpatialOp,
)
from ..ops.navigation import GroundNavigationOp
from ..ops.observation import ObserveFocusOp, ObserveFramesOp
from ..ops.revision import MergeAssociateOp, MergeTrackletsOp, PruneUnobservedOp, ReviseTrackletsOp
from .config import RunConfig, write_resolved


def _configured_observe(client, out_dir: Path, config: RunConfig, render_tool=None):
    """The bootstrap observe op with every policy knob applied."""
    op = ObserveFramesOp(
        client.complete,
        out_dir / "frames",
        media=config.policy.observe.media,
        windows=config.policy.observe.windows,
        render_tool=render_tool,
        round_local=config.policy.observe.round_local_context,
    )
    op.bind = config.policy.observe.bind
    op.max_frames = config.policy.observe.max_frames
    op.max_fps = config.policy.observe.max_fps
    op.dedup_jaccard = config.policy.dedup.jaccard
    op.dedup_gap_us = int(config.policy.dedup.gap_s * 1e6)
    op.dedup_key = config.policy.dedup.key
    return op


def _agentic_refinement(env, anchor, orchestrator, rsg, client, out_dir: Path, config):
    """The planner-scheduled refinement loop (focus / relate / stop), one action per round.

    Each chosen action executes through ``run_program`` -- the gateway, budget log, and
    per-step trace snapshots apply to planned actions exactly as to static ones. The loop
    is bounded by ``policy.focus.max_rounds`` on top of env budgets; planner turns are
    middleware (budget-exempt, recorded). Outcome lines feed the next round's digest.
    """
    from ...infra.middleware.planner import plan_next

    focus_op = ObserveFocusOp(client.complete, out_dir / "frames", max_requests=1)
    relate_ops = {"spatial": RelateSpatialOp}
    steps: list[dict] = []
    history: list[str] = []
    max_rounds = config.policy.focus.max_rounds

    def resolution_mix(component_id: str) -> dict:
        counts: dict[str, int] = {}
        for component in rsg.components:
            if component.id.split("#", 1)[0] == component_id.split("#", 1)[0]:
                for event in component.events:
                    if "resolution" in event.attributes:
                        key = event.attributes["resolution"]
                        counts[key] = counts.get(key, 0) + 1
        return counts

    for round_index in range(max_rounds):
        decision = plan_next(
            orchestrator,
            rsg,
            anchor,
            client.complete,
            {"rounds_left": max_rounds - round_index},
            history,
        )
        if decision.action == "stop":
            history.append(f"stop: {decision.detail}")
            break
        # A refinement round is contained AND atomic: the graph snapshots before the
        # action, and any model-backed failure (elicitation exhausted, endpoint outage,
        # oversized reply) restores it -- the op either commits fully or not at all, so
        # the planner retries against the exact prior state. One round lost, recorded,
        # fed to the next digest -- never the run. Context records of the failed attempt
        # persist (audit layer). The static prefix keeps fail-fast semantics.
        snapshot = rsg.to_dict()
        try:
            if decision.action == "focus":
                before = resolution_mix(decision.component)
                focus_op.pending_requests = [
                    (decision.component, decision.window_us, "planner request", decision.frames)
                ]
                rsg, action_steps = run_program(env, anchor, [focus_op], orchestrator, rsg=rsg)
                after = resolution_mix(decision.component)
                base = decision.component.split("#", 1)[0]
                outcome = (
                    f"resolutions {before or '{}'} => {after or '{}'}"
                    if after != before
                    else "NO CHANGE (saturated -- do not re-request)"
                )
                history.append(f"focus {base} ({decision.frames} frames) -> {outcome}")
            else:
                rsg, action_steps = run_program(
                    env, anchor, [relate_ops[decision.kind]()], orchestrator, rsg=rsg
                )
                history.append(f"relate {decision.kind} -> {len(rsg.edges)} edges total")
        except (ValueError, RuntimeError) as error:
            rsg = Rsg.from_dict(snapshot)  # atomic: no partial commits survive
            failure = f"{decision.action} FAILED ({str(error)[:120]}) -- round lost, run continues"
            history.append(failure)
            orchestrator.record_parse("plan:focus", f"round failure: {failure}")
            continue
        steps += action_steps
    orchestrator.memory.write(
        f"planner loop: {len(history)} entries; " + (history[-1] if history else "no rounds")
    )
    return rsg, steps


def _evidence_entry(component) -> dict:
    """One component as the viewer sees it at a step: the FULL contract minus events.

    Derived from ``Component.to_dict()`` rather than re-listed, so a field added to the
    component reaches the viewer with no change here. The hand-maintained allow-list this
    replaced silently dropped ``attributes`` and ``supersedes`` -- the graph was complete
    and the view of it was not, which is unfalsifiable from the viewer side.

    Events are the one omission, and deliberately: they live once in the trace's
    ``provenance`` map keyed by id, so repeating them in each of ~16 steps would multiply
    the payload for nothing. The viewer resolves them by id.
    """
    record = component.to_dict()
    record.pop("events", None)
    return {
        **record,
        # presentation aliases the renderer reads by these names
        "source": f"{component.category} - T{int(component.tier)}",
        "source_class": component.category,
        "media": False,
        "summary": component.label,
    }


def viewer_state(
    rsg: Rsg,
    component_ids: list[str],
    log: list[str],
    edge_count: int = 0,
    sources: list[dict] | None = None,
    context_ids: list[str] | None = None,
) -> dict:
    """One viewer state snapshot over the step-time current view (``component_ids``)."""
    return {
        "sources": sources or [],
        "context_ids": context_ids or [],
        "edges": [edge.to_dict() for edge in rsg.edges[:edge_count]],
        "evidence": [
            _evidence_entry(component)
            for component in (rsg.component(cid) for cid in component_ids)
        ],
        # (six always-empty fields removed 2026-08-24: nothing populated them and the
        # viewer never read them -- an always-empty "divergences" reads as "this run had
        # none", which is worse than the field not existing)
        "log": log,
    }


def _catalog_summary(env) -> list[dict]:
    """All read records grouped per user-defined stream, with class, count, and time range."""
    return _stream_summary(env, list(env.catalog))


def _stream_summary(env, record_ids: list[str]) -> list[dict]:
    """Stream/class/count/time-range summary over one step's snapshot of record ids."""
    groups: dict[str, dict] = {}
    for record_id in record_ids:
        record = env.catalog[record_id]
        group = groups.setdefault(
            record.stream,
            {
                "stream": record.stream,
                "source_class": record.source_class.value,
                "count": 0,
                "t_min_us": None,
                "t_max_us": None,
            },
        )
        group["count"] += 1
        if record.timestamp_us is not None:
            group["t_min_us"] = (
                record.timestamp_us
                if group["t_min_us"] is None
                else min(group["t_min_us"], record.timestamp_us)
            )
            group["t_max_us"] = (
                record.timestamp_us
                if group["t_max_us"] is None
                else max(group["t_max_us"], record.timestamp_us)
            )
    return list(groups.values())


def _track_ids(record) -> dict:
    """``{"track_ids": [...]}`` for a REFERENCE record, index-aligned with its masks.

    Scoped to reference streams on purpose: ``objects`` is not one shape across the
    catalog. Reference records carry a list of detections, geometry records carry a dict
    keyed by index (ops/geometry.py), and iterating that dict yields ints -- which killed
    a 50-clip sweep on every clip, while a local probe passed because its geometry pass
    happened to produce nothing.
    """
    if not (record.stream or "").startswith("reference/") or not record.payload:
        return {}
    objects = record.payload.get("objects")
    if not isinstance(objects, list):
        return {}
    ids = [obj.get("track_id") if isinstance(obj, dict) else None for obj in objects]
    return {"track_ids": ids} if any(ids) else {}


def _served_path(path: str, root: Path) -> str | None:
    """A media path relative to the served playground root, or None when outside it."""
    try:
        return str(Path(path).resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def _refresh_playground(out_dir: Path) -> None:
    """Keep the served playground root current: the run manifest and the viewer page.

    The viewer at the root reads ``runs.json`` for clip selection; each entry points at a
    sibling run directory's trace. Refreshing on every run keeps the served copy in sync
    with the source viewer (a stale copy is a known blank-page trap).
    """
    viewer = (Path(__file__).parents[1] / "viewer" / "viewer.html").read_text()
    build = hashlib.md5(viewer.encode()).hexdigest()[:8]

    def refresh(root: Path, depth: str) -> None:
        # Names are root-relative paths, so an entry keeps its name no matter which run
        # regenerated the manifest -- a renamed/vanished entry broke ?run= selection.
        runs = [
            {
                "name": str(trace.parent.relative_to(root)),
                "trace": f"{trace.parent.relative_to(root)}/trace.json",
            }
            for pattern in depth.split()
            for trace in sorted(root.glob(pattern))
            # archived runs stay served (deep-linkable) but leave the selector
            if trace.parent.relative_to(root).parts[0] not in ("arxiv", "archive")
            # Only current-contract traces (pre-provenance artifacts predate the flows).
            if "provenance" in json.loads(trace.read_text())
        ]
        (root / "runs.json").write_text(json.dumps(runs, indent=1))
        (root / "viewer.html").write_text(viewer.replace("__VIEWER_BUILD__", f"build {build}"))

    refresh(out_dir.parent, "*/trace.json")
    # exp-dir runs also index into the playground root's manifest (recursive), so the
    # root viewer lists every run under stable exp-qualified names
    root = out_dir.parent.parent
    if (root / "viewer.html").is_file():
        refresh(root, "*/trace.json */*/trace.json")


def run(
    clip_id: str,
    out_dir: Path,
    config: RunConfig,
    *,
    interpreter=None,
    client=None,
    publish_viewer=True,
    appearance_complete=None,
    validate_inputs=None,
) -> Path:
    """Run the component op program for one clip under one resolved configuration.

    The config's three scopes drive everything: playground -> the env contract; policy ->
    the op program's knobs; infra -> where endpoints and raw data live. The resolved
    config is serialized beside the artifacts with both scope hashes, and the validator
    checks it against the produced contract (config drift is a validation error).
    Batch consumers disable ``publish_viewer`` to keep writes inside ``out_dir``.
    """
    playground = config.playground.env
    external = {"object_labels", "map"} & set(playground.sources)
    if external not in (set(), {"object_labels", "map"}):
        raise ValueError("partial external-label manifests are not supported yet")
    if config.infra.endpoint_registry is None:
        raise ValueError("infra.endpoint_registry is required (config or CLI)")
    grounding_registry = (
        Path(config.infra.grounding_registry) if config.infra.grounding_registry else None
    )
    if playground.video_manifest is not None:
        from ..env.video import build_env_video_manifest

    env = (
        build_env_video_manifest(playground.video_manifest, clip_id, out_dir / "frames", playground)
        if playground.video_manifest is not None
        else build_env_native(
            config.infra.metadb,
            config.infra.data_dir,
            clip_id,
            cameras=tuple(playground.cameras),
            max_fps=playground.max_fps,
            image_max_height_px=playground.image_max_height_px,
            image_overview_height_px=playground.image_overview_height_px,
            anchor_mode=playground.anchor_mode,
            include_external_labels=bool(external),
            reference=playground.reference,
            reference_concepts=(
                tuple(playground.reference_concepts) if playground.reference_concepts else None
            ),
            scope=(
                (playground.scope_distance_m or 0.0, playground.scope_ttc_s or 0.0)
                if playground.scope_distance_m is not None or playground.scope_ttc_s is not None
                else None
            ),
            history_us=int(playground.history_s * 1e6),
            future_us=int(playground.future_s * 1e6),
            max_frames=playground.max_frames,
            max_ops=playground.budgets.max_ops,
            wall_clock_s=playground.budgets.wall_clock_s,
            navigation_manifest=playground.navigation_manifest,
            meta_action_manifest=playground.meta_action_manifest,
            **(
                {"sensor_manifest": playground.sensor_manifest}
                if playground.sensor_manifest is not None
                else {}
            ),
        )
    )
    anchor = env.anchors[0]
    if validate_inputs is not None:
        validate_inputs(env, anchor)
    orchestrator = Orchestrator(env=env)
    if client is not None and appearance_complete is None:
        appearance_complete = ChatClient(
            client.registry_dir,
            timeout_s=120,
            max_tokens=256,
            temperature=0.0,
            on_attempt=client.on_attempt,
            before_attempt=client.before_attempt,
        ).complete
    if client is None:
        client = ChatClient(
            Path(config.infra.endpoint_registry),
            max_tokens=config.policy.generation.max_tokens,
            temperature=config.policy.generation.temperature,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_resolved(config, out_dir)
    render_tool = make_render_tool(
        env, env.anchors[0], out_dir / "frames", style=config.policy.observe.render_style
    )
    orchestrator.tools["render_overlays"] = render_tool
    task_name = config.playground.task
    if interpreter is not None and task_name != "none":
        raise ValueError("a downstream interpreter requires compiler task: none")
    if task_name != "none" and playground.anchor_mode != "keyframe":
        raise ValueError(
            f"task {task_name!r} designates decision moments; it requires keyframe anchors "
            "(discovery compiles with task: none)"
        )
    if grounding_registry is not None:
        orchestrator.tools["ground_frame"] = make_grounding_tool(
            grounding_registry,
            lambda: next((r.path for r in reversed(list(env.catalog.values())) if r.path), None),
        )
    agentic = config.policy.focus.mode == "agentic"
    if playground.reference != "labels" and grounding_registry is not None:
        from ..ops.reference import GroundReferenceOp

    reference_pass = (
        []
        if playground.reference == "labels" or grounding_registry is None
        else [
            GroundReferenceOp(
                lambda paths, prompt: track_regions(grounding_registry, paths, prompt),
                out_dir / "frames",
                **(
                    {"prompts": tuple(playground.reference_concepts)}
                    if playground.reference_concepts
                    else {}
                ),
            )
        ]
    )
    da3_registry = Path(grounding_registry).parent / "da3_local" if grounding_registry else None
    if playground.reference != "labels" and da3_registry is not None and da3_registry.is_dir():
        from ..ops.geometry import GroundElementsOp, GroundGeometryOp, ParseAgentManeuverOp

    geometry_pass = (
        []
        if playground.reference == "labels" or da3_registry is None or not da3_registry.is_dir()
        else [
            GroundGeometryOp(lambda paths: da3_geometry(da3_registry, paths), out_dir / "frames"),
            # anchored map elements: the consistent reference spatial relations need.
            # Face resolution is DEFERRED (2026-08-23 review): a box crop around one
            # signal head also contains its neighbours, so the VLM captions two lamps
            # as one and the "which member faces ego" question is malformed at the
            # evidence level. GroundElementsOp keeps the judge_face hook (and its
            # tests) for when mask-tight, neighbour-masked crops make one head one
            # crop; until then groups anchor without a role rather than pay for a
            # judgement that landed once in twelve and landed on a billboard.
            GroundElementsOp(
                tolerance_scale=config.policy.elements.tolerance_scale,
                on_disagreement=config.policy.elements.track_disagreement,
            ),
        ]
    )
    prefix = [
        GroundEgoOp(out_dir / "frames"),
        *([GroundNavigationOp()] if playground.navigation_manifest is not None else []),
        *([ParseManeuverOp()] if "ego_pose" in env.sources else []),
        GroundAgentsOp(),
        GroundMapOp(),
        *reference_pass,
        *geometry_pass,
        _configured_observe(client, out_dir, config, render_tool),
        # the agent's reflect step runs BEFORE folding: a tracklet it breaks
        # must not have claims folded onto it
        ReviseTrackletsOp(
            judge_appearance=(
                None
                if config.infra.endpoint_registry is None
                else lambda path, box, claim: judge_appearance(
                    Path(config.infra.endpoint_registry),
                    path,
                    box,
                    claim,
                    complete=appearance_complete,
                )
            ),
            mode=config.policy.revise.mode,
        ),
        # object-centric folding BEFORE label association: claims citing one tracklet
        # become one component, judge-gated, so association and identity downstream see
        # objects rather than per-round fragments
        MergeTrackletsOp(
            judge_appearance=(
                None
                if config.infra.endpoint_registry is None
                else lambda path, box, claim: judge_appearance(
                    Path(config.infra.endpoint_registry),
                    path,
                    box,
                    claim,
                    complete=appearance_complete,
                )
            ),
            mode=config.policy.merge.mode,
        ),
        *(
            []
            if grounding_registry is None
            else [
                MergeAssociateOp(
                    lambda path, prompt: ground_detections(grounding_registry, path, prompt),
                    out_dir / "frames",
                )
            ]
        ),
        PruneUnobservedOp(),
    ]
    if (
        config.policy.identity.mode != "skip"
        and playground.reference != "labels"
        and grounding_registry is not None
    ):
        from ..ops.identity import GroundIdentityOp

    identity_pass = (
        []
        if config.policy.identity.mode == "skip"
        or playground.reference == "labels"
        or grounding_registry is None
        else [
            GroundIdentityOp(
                lambda path, text: agent_ground_regions(
                    grounding_registry, config.infra.endpoint_registry, path, text
                ),
                out_dir / "frames",
                judge_appearance=lambda path, box, claim: judge_appearance(
                    config.infra.endpoint_registry,
                    path,
                    box,
                    claim,
                    complete=appearance_complete,
                ),
                track_window=lambda paths, box: track_window(grounding_registry, paths, box),
                concurrency=config.infra.identity_concurrency,
                per_claim=config.policy.identity.per_claim,
            )
        ]
    )
    suffix = (
        []
        if agentic
        else [
            *identity_pass,
            *([] if not geometry_pass else [ParseAgentManeuverOp()]),
            ObserveFocusOp(
                client.complete, out_dir / "frames", max_requests=config.policy.focus.max_requests
            ),
            RelateSpatialOp(),
        ]
    )
    # Compile: sources -> IR (the op passes). Then interpret: the selected task's steps run
    # on the finished graph -- interpreters vary; the compile prefix does not. In agentic
    # mode the refinement suffix (focus + relate) is planner-scheduled instead of static.
    try:
        rsg, steps = run_program(env, anchor, prefix + suffix, orchestrator)
        if agentic:
            rsg, planner_steps = _agentic_refinement(
                env, anchor, orchestrator, rsg, client, out_dir, config
            )
            steps += planner_steps
        task_output = None
        if interpreter is not None:
            rsg, interpreter_steps, task_output = interpreter(
                env, anchor, orchestrator, rsg, client
            )
            steps += interpreter_steps
        if task_name != "none":
            from ...tasks.registry import TASKS

            task = TASKS[task_name]
            rsg, task_steps = run_program(
                env, anchor, task.program(client.complete), orchestrator, rsg=rsg
            )
            steps += task_steps
            task_output = task.output(rsg)
        if task_output is not None:
            task_output["playground_id"] = config.playground_id
            task_output["policy_id"] = config.policy_id
            with (out_dir / "task.yaml").open("w") as task_file:
                yaml.safe_dump(task_output, task_file, sort_keys=False)

        # Round-trip revalidation: a loaded artifact replays every write through the same checks.
        Rsg.from_dict(rsg.to_dict())
    finally:
        orchestrator.write_context_log(out_dir / "context.jsonl")

    with (out_dir / "rsg.yaml").open("w") as out_file:
        yaml.safe_dump(
            {
                "env": env.contract_dict(),
                "config_ids": {
                    "playground_id": config.playground_id,
                    "policy_id": config.policy_id,
                },
                "sources": env.catalog_dict(),
                "rsg": rsg.to_dict(),
            },
            out_file,
            sort_keys=False,
        )
    trace = {
        "env": env.contract_dict(),
        # Full provenance map (all components, history included) and the env source catalog
        # summary, so the viewer can trace information flow from env sources to each node.
        "provenance": {
            # the COMPLETE component record (op/read_ids spread first so the existing
            # flat reads still resolve), history included -- this is what the viewer
            # resolves events and attributes from
            c.id: {
                **c.provenance.to_dict(),
                **c.to_dict(),
                **({"events": [e.to_dict() for e in c.events]} if c.events else {}),
            }
            for c in rsg.all_components
        },
        "source_catalog": _catalog_summary(env),
        "record_streams": {record_id: r.stream for record_id, r in env.catalog.items()},
        # Per-record detail so the viewer can show any source's own derivation: env-carried
        # records are origins; derived records carry the chain back to them.
        "records": {
            r.id: {
                "stream": r.stream,
                "source_class": r.source_class.value,
                "summary": r.summary,
                "timestamp_us": r.timestamp_us,
                "derived_from": list(r.derived_from),
                # media paths serve relative to the playground root (the viewer's origin)
                **(
                    {"path": _served_path(r.path, out_dir.parent)}
                    if r.path and _served_path(r.path, out_dir.parent)
                    else {}
                ),
                # Index-aligned with the mask file this record points at (both come from
                # one detections list, ops/reference.py). The viewer binds a mask to the
                # component that cites its tracklet; without this the only mask->component
                # link is the identity pass's ref_object, so every agent mask reads as
                # unbound whenever identity is skipped -- which is now the common case.
                **_track_ids(r),
            }
            for r in env.catalog.values()
        },
        "context_catalog": {
            r["id"]: {
                "source_class": r["source_class"],
                "op": r["op"],
                **({"response_to": r["response_to"]} if "response_to" in r else {}),
                **({"tool": r["tool"], "argument": r["argument"]} if "tool" in r else {}),
                **({"reads": r["reads"]} if r.get("reads") else {}),
            }
            for r in orchestrator.records
            if "source_class" in r
        },
        "clip_id": env.clip_id,
        "segment_id": f"{clip_id}_{anchor.anchor_us}_native",
        "time_window_us": list(anchor.window_us),
        "strategy": f"alparsg component op program (env {env.env_id})",
        "config": {"playground_id": config.playground_id, "policy_id": config.policy_id},
        "model": "qwen via native vision client",
        "export": {
            "review_status": "native_run",
            "selected_action_index": None,
            "final_graph": {"nodes": len(rsg.components), "edges": len(rsg.edges)},
        },
        "steps": [{"op": "init", "elapsed_s": 0.0, "state": viewer_state(rsg, [], [])}]
        + [
            {
                "op": step["op"],
                "elapsed_s": step["elapsed_s"],
                "state": viewer_state(
                    rsg,
                    step["current_ids"],
                    step["log_after"],
                    edge_count=step["edge_count"],
                    sources=_stream_summary(env, step["catalog_ids"]),
                    context_ids=step["context_ids"],
                ),
            }
            for step in steps
        ],
    }
    (out_dir / "trace.json").write_text(json.dumps(trace, indent=1))
    if publish_viewer:
        _refresh_playground(out_dir)

    decision = next((c for c in reversed(rsg.components) if c.category == "decision"), None)
    print(
        f"env {env.env_id} -- {len(rsg.all_components)} components total, "
        f"{len(rsg.components)} current, {len(rsg.edges)} edges, "
        f"{len(env.catalog)} source records, {len(steps)} op executions"
    )
    for line in rsg.log:
        print(f"  {line}")
    if decision is not None:
        print(f"decision: {decision.label[:300]}")
    print(f"wrote {out_dir}/rsg.yaml, context.jsonl, trace.json")
    return out_dir / "rsg.yaml"
