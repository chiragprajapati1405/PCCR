#!/usr/bin/env python3
"""Latency benchmark: SEQUENTIAL vs PARALLEL PCCR architecture on 100 tasks.

Both runs drive the SAME real framework (per-task fork with own WM, Phase-3
ρ-gate, the orchestrator re-planning loop, dependency-wave agent execution,
lock-guarded Phase-9 writes, shared stores). The ONLY modeled quantity is the
external call cost: each orchestrator/agent step costs `L` seconds, set to the
*measured* Cerebras gpt-oss-120b per-call latency (median 0.516 s over real
calls; pass --latency to override). asyncio overlaps a real network await
exactly as it overlaps `asyncio.sleep(L)`, so the wall-clock speedup measured
here is what real parallel API calls would achieve (the work is I/O-bound).

What differs between the two runs is ONLY the execution schedule:
  SEQUENTIAL : tasks run one after another; within a task, agents run one at a
               time (the original sync lifecycle's behaviour).
  PARALLEL   : tasks run concurrently (AsyncTaskQueue, max_concurrency=C);
               within a task, independent subtasks in a wave run concurrently.

Note: parallelism changes ONLY latency. Call count / token cost are identical
in both runs — reducing *consults* is the ρ-gate's job (orthogonal).

Run:
    .venv/bin/python bench_parallel.py                 # 100 tasks, C=4, L=0.516
    .venv/bin/python bench_parallel.py --tasks 100 --concurrency 8 --latency 0.516
"""
from __future__ import annotations

import argparse
import asyncio
import time

from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.parallel import AgentResult, Delegation, DependencyAnalyzer
from memory_manager.task_queue import AsyncTaskQueue
from run_task import (AGENT_CAPABILITY_PROFILES, AGENT_PROMPTS,
                      ORCHESTRATOR_PROMPT, TASK_TYPE_RULES)
from environment.tasks import USERNAME, seed_entities

# --- 100-task generator: realistic pattern mix with delegation structure ----
# Each task is a list of orchestrator GROUPS (the re-planning rounds). Within a
# group, the dependency analyzer decides which subtasks run in parallel.
def _make_tasks(n: int) -> list[dict]:
    R = lambda a, s: Delegation(a, s)
    shapes = {
        # pattern: (weight, description, [groups...])
        "single": (0.25, "Add a dentist appointment",
                   [[R("calendar", "create dentist appointment")]]),
        "lookup": (0.20, "How many meetings tomorrow?",
                   [[R("calendar", "list events for tomorrow")]]),
        "recurring": (0.10, "Send the weekly status update as usual",
                      [[R("email", "send the weekly update")]]),
        "coordination": (0.30, "Schedule team sync and notify everyone",
                         [[R("calendar", "list team events"),       # wave: 3 parallel reads
                           R("email", "read the sync thread"),
                           R("search", "find the attendee list")],
                          [R("calendar", "create the sync event")],  # then create
                          [R("email", "send the invite")]]),         # then notify
        "exploratory": (0.15, "Investigate the budget overrun and report",
                        [[R("search", "search finance docs"),        # wave: 2 parallel reads
                          R("email", "read budget threads")],
                         [R("email", "send the findings report")]]),
    }
    order = list(shapes.items())
    tasks = []
    i = 0
    while len(tasks) < n:
        key, (w, desc, groups) = order[i % len(order)]
        # honor weights by interleaving proportionally
        reps = max(1, round(w * 10))
        for _ in range(reps):
            if len(tasks) >= n:
                break
            tasks.append({"id": f"{key}-{len(tasks)}", "pattern": key,
                          "description": f"{desc} #{len(tasks)}", "groups": groups})
        i += 1
    return tasks[:n]


def _build_base() -> MemoryManager:
    base = MemoryManager(config.SETTINGS)
    base.bootstrap(ORCHESTRATOR_PROMPT, AGENT_PROMPTS, TASK_TYPE_RULES,
                   AGENT_CAPABILITY_PROFILES, entity_seed=seed_entities())
    return base


def _stats(tasks: list[dict]) -> tuple[int, int]:
    """Total LLM calls (orch rounds + agents) and ideal parallel call-depth
    (orch rounds + waves) — identical work, different critical path."""
    calls = depth = 0
    for t in tasks:
        for g in t["groups"]:
            calls += 1 + len(g)                          # 1 orch round + its agents
            depth += 1 + len(DependencyAnalyzer().analyze(list(g)))  # 1 orch + #waves
    return calls, depth


async def _run(base: MemoryManager, tasks: list[dict], L: float, C: int, parallel: bool):
    async def agent_runner(agent_type, subtask, snap):
        await asyncio.sleep(L)                            # the external agent call (LLM+tool)
        return AgentResult(agent_type, subtask, observation=f"{agent_type} ok")

    def make_planner(groups):
        st = {"i": 0}
        async def planner(_wm):
            await asyncio.sleep(L)                         # the orchestrator LLM call
            if st["i"] >= len(groups):
                return []
            g = groups[st["i"]]; st["i"] += 1
            return g
        return planner

    async def task_pipeline(spec):
        m = base.fork_for_task()
        m.ingest_task(spec["description"], USERNAME, "2026-06-08")     # Phase 2
        m.retrieve()                                                  # Phase 3 ρ-gate (once)
        if parallel:
            # parallel agents within the task (waves overlap)
            await m.run_planned_task(make_planner(spec["groups"]), agent_runner)
        else:
            # sequential within the task: agents one at a time
            for g in spec["groups"]:
                await asyncio.sleep(L)                                 # orchestrator round
                for d in g:
                    res = await agent_runner(d.agent_type, d.subtask, m.wm.snapshot())
                    m.wm.merge_staging([(res.agent_type, res.subtask, res.observation)])
        plan = [f"{d.agent_type}" for g in spec["groups"] for d in g]
        await m.complete_task_async(True, plan, [], {"last": spec["id"]})  # Phase 9
        return {"task": spec["id"]}

    t0 = time.perf_counter()
    if parallel:
        await AsyncTaskQueue(task_pipeline, max_concurrency=C,
                             is_consolidating=base.is_consolidating).run(tasks)
    else:
        for spec in tasks:                                            # tasks one after another
            await task_pipeline(spec)
    return time.perf_counter() - t0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--latency", type=float, default=0.516,
                    help="per-call latency (s); default = measured gpt-oss-120b median")
    args = ap.parse_args()

    tasks = _make_tasks(args.tasks)
    calls, depth = _stats(tasks)
    from collections import Counter
    mix = Counter(t["pattern"] for t in tasks)

    print(f"{'=' * 68}\nPCCR latency benchmark — sequential vs parallel\n{'=' * 68}")
    print(f"tasks={len(tasks)}  per-call latency L={args.latency:.3f}s (measured)  "
          f"task-concurrency C={args.concurrency}")
    print(f"pattern mix: {dict(mix)}")
    print(f"total LLM calls (same for both runs): {calls}  |  "
          f"ideal parallel critical-path depth: {depth} calls")

    base = _build_base()
    print("\nrunning SEQUENTIAL ...", flush=True)
    seq = await _run(base, tasks, args.latency, args.concurrency, parallel=False)
    base2 = _build_base()
    print("running PARALLEL  ...", flush=True)
    par = await _run(base2, tasks, args.latency, args.concurrency, parallel=True)

    print(f"\n{'-' * 68}\nRESULTS (wall-clock over {len(tasks)} tasks)\n{'-' * 68}")
    print(f"  sequential : {seq:8.2f} s")
    print(f"  parallel   : {par:8.2f} s   (C={args.concurrency})")
    print(f"  speedup    : {seq / par:7.2f}x   (saved {seq - par:.1f}s, "
          f"{100 * (seq - par) / seq:.1f}%)")
    print(f"\n  per-task avg : sequential {seq / len(tasks) * 1000:6.0f} ms  ->  "
          f"parallel {par / len(tasks) * 1000:6.0f} ms")
    print(f"  theory check : seq≈calls×L={calls * args.latency:.1f}s ; "
          f"parallel lower bound≈depth×L/C-ish (overlap of {calls}→{depth} critical calls)")
    print(f"\n  NOTE: identical call count ({calls}) and token cost in both runs — "
          f"parallelism changes\n        only latency; the ρ-gate is what cuts consults.")


if __name__ == "__main__":
    asyncio.run(main())
