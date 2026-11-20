"""The op executor: runs a program of component ops under the env's run budgets.

The static program is the v1 plan (the degenerate planner). A self-critic op returning
``rerun`` re-queues the named op followed by the critic itself -- the re-planning path -- and
every execution counts against the env budget, so the in-run loop terminates on budget, not on
a separate rule.
"""

from __future__ import annotations

import time

from ..agent.orchestrator import Orchestrator
from ..data_structure.rsg import Rsg
from ..env.env import Anchor, Env


def run_program(
    env: Env, anchor: Anchor, program: list, orchestrator: Orchestrator, rsg: Rsg | None = None
) -> tuple[Rsg, list[dict]]:
    """Execute the op program against a fresh RSG bound to the env (or continue ``rsg``).

    Returns the graph and a per-execution step list for the evolution trace. Stops -- with a
    log entry, never an exception -- when the env's op-count or wall-clock budget is exhausted.
    Passing ``rsg`` continues an existing graph: how a task's interpreter step runs after
    compilation while staying inside the gateway and the trace.
    """
    if rsg is None:
        rsg = Rsg(env_id=env.env_id)
    steps: list[dict] = []
    queue = list(program)
    by_name = {op.name: op for op in program}
    # ops that parallelize internal work admit waves against the true run deadline
    # (between-op checks alone let a long pass eat the suffix's budget unnoticed)
    if orchestrator.budget_deadline is None:
        orchestrator.budget_deadline = time.monotonic() + env.budget.wall_clock_s
    while queue:
        if orchestrator.ops_used >= env.budget.max_ops:
            orchestrator.budget_exhausted = True
            rsg.log.append(
                f"budget exhausted: {orchestrator.ops_used} ops (max {env.budget.max_ops})"
            )
            break
        if time.monotonic() >= orchestrator.budget_deadline:
            orchestrator.budget_exhausted = True
            rsg.log.append(f"budget exhausted: wall clock (max {env.budget.wall_clock_s}s)")
            break
        op = queue.pop(0)
        op_started = time.monotonic()
        # Failed operations consume work too; retries cannot renew the run allowance.
        orchestrator.ops_used += 1
        result = op.run(rsg, env, anchor, orchestrator)
        elapsed = time.monotonic() - op_started
        rsg.log.append(f"{op.name}: {elapsed:.1f}s, {len(rsg.components)} current components")
        steps.append(
            {
                "op": op.name,
                "elapsed_s": round(elapsed, 1),
                # Membership is retroactive under supersession, so the per-step view must be
                # captured now, not reconstructed from the final graph.
                "current_ids": [component.id for component in rsg.components],
                "edge_count": len(rsg.edges),
                # Catalogs grow as ops read; snapshot membership so the trace shows each
                # step's records, not the final state's.
                "catalog_ids": list(env.catalog),
                "context_ids": [r["id"] for r in orchestrator.records if "source_class" in r],
                "log_after": list(rsg.log),
            }
        )
        if result.rerun is not None:
            queue.insert(0, op)
            queue.insert(0, by_name[result.rerun])
    return rsg, steps
