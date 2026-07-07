"""STEP 6 — one task's pipeline (WM buffer + parallel sub-agents + success-gate).

Flow (the two concurrency dimensions + the write-timing rule):
   1. ORCHESTRATOR reads PM by similarity (past trajectories) -> makes a plan of dependency WAVES.
   2. For each wave, the M sub-agents run in PARALLEL (asyncio.gather); each reads TOOL MEMORY
      (how to call its tool) then executes. Steps accumulate in WM (a per-task PRIVATE list, no lock).
   3. SUCCESS-GATE: success is known only NOW (at the end). On success the orchestrator appends the
      FULL trajectory (plan + WM steps) to PM once; on failure WM is discarded (PM never sees a
      failing plan).

Dimensions: N tasks (caller) x M sub-agents (here) -> Tool Memory sees N*M reads, PM sees N reads
+ (successes) appends.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from concurrent_mm.procedural_memory import Trajectory


@dataclass
class SubtaskSpec:
    app: str
    text: str


@dataclass
class TaskSpec:
    text: str
    waves: list                        # list[list[SubtaskSpec]] — dependency waves
    will_succeed: bool = True          # modeled outcome (real mode: from evaluation)


@dataclass
class TaskResult:
    text: str
    latency_s: float
    committed: bool                    # did it write to PM?
    pm_read_wait_s: float
    pm_write_wait_s: float
    n_subagents: int


async def run_task(spec: TaskSpec, mm, executor) -> TaskResult:
    t0 = time.perf_counter()
    wm: list = []                                          # WM: per-task PRIVATE buffer, no lock

    # 1. ORCHESTRATOR — read PM (by similarity), then plan (modeled: plan == spec.waves)
    _past, pm_read_wait = await mm.orchestrator_read_pm(spec.text)
    plan = spec.waves

    # 2. SUB-AGENTS — each wave runs its sub-agents in PARALLEL; steps buffered in WM
    n_sub = 0
    for wave in plan:
        async def run_sub(sub: SubtaskSpec):
            usage = mm.subagent_read_tool_memory(sub.app)     # READ-ONLY tool memory (sub-agent side)
            return await executor.run_action(sub.app, sub.text, usage)
        results = await asyncio.gather(*[run_sub(s) for s in wave])
        wm.extend(results)                                    # accumulate LOCALLY
        n_sub += len(wave)

    # 3. SUCCESS-GATE — commit to PM only if the task succeeded (known only now)
    pm_write_wait = 0.0
    committed = False
    if spec.will_succeed:
        traj = Trajectory(task=spec.text, plan=[w for w in plan], subagent_steps=wm)
        pm_write_wait = await mm.orchestrator_write_pm(traj)  # one atomic append
        committed = True
    # else: discard wm (nothing to PM)

    return TaskResult(text=spec.text, latency_s=time.perf_counter() - t0, committed=committed,
                      pm_read_wait_s=pm_read_wait, pm_write_wait_s=pm_write_wait, n_subagents=n_sub)
