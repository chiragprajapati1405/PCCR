#!/usr/bin/env python3
"""Correctness check: does the PARALLEL path produce the SAME outcomes as the
SEQUENTIAL path?  (i.e. is accuracy preserved — parallelism changes only latency)

We run the SAME task set through both execution schedules and compare, per task:
  - success flag
  - plan (which agents did what)
  - the full set of agent OBSERVATIONS (normalised to drop volatile artifact
    filenames/uuids), sorted — so order-of-completion cannot matter
and then compare the FINAL shared-memory state (STM/EM/SM/PM sizes + ENT facts),
since that is what carries forward to affect any *future* task's retrieval.

Why this is the right test:
  * Parallel AGENTS within a task: independent subtasks land in the same wave;
    a dependent subtask is in a later wave. Independent subtasks query external
    state (calendar/email/search), not each other, so their results don't depend
    on sibling order -> identical observations; the merge is deterministic.
  * Parallel TASKS: each held-out task is scored independently, and writes go
    through per-resource locks (no lost updates). Concurrent tasks may see a
    different STM/EM *cache* state (a recurring task might miss a not-yet-written
    cache), but a cache miss only triggers a recompute that yields the SAME plan
    -> same answer. Cross-task concurrency changes cache-hit timing (efficiency),
    not correctness.

Run:
    .venv/bin/python verify_equivalence.py            # 60 tasks, seq vs parallel
"""
from __future__ import annotations

import asyncio
import re
import tempfile

from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.parallel import AgentResult, DependencyAnalyzer
from memory_manager.task_queue import AsyncTaskQueue
from environment import World
from environment.tasks import USERNAME, seed_entities, seed_events, seed_inbox
from run_task import (AGENT_CAPABILITY_PROFILES, AGENT_PROMPTS,
                      ORCHESTRATOR_PROMPT, TASK_TYPE_RULES)
from run_parallel_demo import _make_action
from bench_parallel import _make_tasks

DATE = "2026-06-08"
_STRIP = re.compile(r"\s*Saved to \S+\.?\s*$")        # drop volatile uuid/timestamp filenames


def _norm(obs: str) -> str:
    return _STRIP.sub("", obs).strip()


def _build_base() -> MemoryManager:
    base = MemoryManager(config.SETTINGS)
    base.bootstrap(ORCHESTRATOR_PROMPT, AGENT_PROMPTS, TASK_TYPE_RULES,
                   AGENT_CAPABILITY_PROFILES, entity_seed=seed_entities())
    return base


def _fresh_world() -> World:
    """Each TASK gets its own freshly-seeded external environment — the held-out
    evaluation model (a task is scored against its own sandbox, so concurrent
    tasks never mutate a shared filesystem under each other)."""
    from pathlib import Path
    p = Path(tempfile.mkdtemp(prefix="pccr_eq_"))
    return World(eml_dir=p / "eml", ics_dir=p / "ics", answers_dir=p / "ans",
                 seed_inbox=seed_inbox(), seed_events=seed_events())


def _make_runner(world: World):
    async def runner(agent_type, subtask, snap):
        action = _make_action(agent_type, subtask, USERNAME, DATE)
        await asyncio.sleep(0)                          # cooperative yield
        obs, _s, _a = world.execute(action, username=USERNAME, date=DATE, subtask=subtask)
        return AgentResult(agent_type, subtask, action=action, observation=obs)
    return runner


def _outcome(spec, wm_ctx) -> tuple:
    obs = tuple(sorted(_norm(r.observation) for r in wm_ctx.step_history))
    plan = tuple(sorted(f"{d.agent_type}:{d.subtask[:20]}"
                        for g in spec["groups"] for d in g))
    success = bool(wm_ctx.step_history)
    return (spec["id"], success, plan, obs)


def _make_planner(groups):
    st = {"i": 0}
    async def planner(_wm):
        if st["i"] >= len(groups):
            return []
        g = groups[st["i"]]; st["i"] += 1
        return g
    return planner


async def run_sequential(tasks) -> tuple[list, MemoryManager]:
    base = _build_base()
    outcomes = []
    for spec in tasks:                                  # tasks one after another
        runner = _make_runner(_fresh_world())           # per-task isolated environment
        m = base.fork_for_task()
        m.ingest_task(spec["description"], USERNAME, DATE)
        m.retrieve()
        for g in spec["groups"]:
            for wave in DependencyAnalyzer().analyze(list(g)):
                for d in wave:                          # one agent at a time, live WM
                    res = await runner(d.agent_type, d.subtask, m.wm.snapshot())
                    m.wm.merge_staging([(res.agent_type, res.subtask, res.observation)])
        outcomes.append(_outcome(spec, m.wm.current))
        await m.complete_task_async(True, [f"{d.agent_type}" for g in spec["groups"] for d in g],
                                    [], {"last": spec["id"]})
    return outcomes, base


async def run_parallel(tasks, C=4) -> tuple[list, MemoryManager]:
    base = _build_base()

    async def pipeline(spec):
        runner = _make_runner(_fresh_world())           # per-task isolated environment
        m = base.fork_for_task()
        m.ingest_task(spec["description"], USERNAME, DATE)
        m.retrieve()
        await m.run_planned_task(_make_planner(spec["groups"]), runner)   # parallel waves
        out = _outcome(spec, m.wm.current)
        await m.complete_task_async(True, [f"{d.agent_type}" for g in spec["groups"] for d in g],
                                    [], {"last": spec["id"]})
        return out

    q = AsyncTaskQueue(pipeline, max_concurrency=C, is_consolidating=base.is_consolidating)
    outcomes = await q.run(tasks)
    return outcomes, base


def _memstate(m: MemoryManager) -> dict:
    prof = m.ent.get(USERNAME)
    return {"STM": len(m.stm), "EM": len(m.em), "SM": len(m.sm),
            "PM_rules": len(m.pm.learned_rules),
            "ENT_facts": dict(prof.facts) if prof else {}}


async def main():
    tasks = _make_tasks(60)
    print(f"{'=' * 66}\nEquivalence check — parallel outcomes == sequential?\n{'=' * 66}")
    print(f"tasks={len(tasks)}  (same set through both schedules)")

    seq, seq_mgr = await run_sequential(tasks)
    par, par_mgr = await run_parallel(tasks, C=4)

    seq_by = {o[0]: o for o in seq}
    par_by = {o[0]: o for o in par}
    assert set(seq_by) == set(par_by)

    mismatches = []
    for tid in seq_by:
        if seq_by[tid][1:] != par_by[tid][1:]:          # compare (success, plan, obs)
            mismatches.append(tid)

    same = len(tasks) - len(mismatches)
    print(f"\nPer-task outcome (success + plan + observations):")
    print(f"  IDENTICAL : {same}/{len(tasks)}")
    if mismatches:
        print(f"  MISMATCH  : {mismatches[:8]}")
        t = mismatches[0]
        print(f"\n  e.g. {t}:\n    seq={seq_by[t]}\n    par={par_by[t]}")
    else:
        print("  -> every task produced byte-identical outcomes in both schedules.")

    s_mem, p_mem = _memstate(seq_mgr), _memstate(par_mgr)
    print(f"\nFinal shared-memory state (carries forward to future-task retrieval):")
    print(f"  sequential : {s_mem}")
    print(f"  parallel   : {p_mem}")
    print(f"  IDENTICAL  : {s_mem == p_mem}")

    ok = not mismatches and s_mem == p_mem
    print(f"\n{'-' * 66}")
    print(f"RESULT: accuracy/outcomes are {'PRESERVED — parallel ≡ sequential' if ok else 'DIFFERENT (investigate)'}.")
    print(f"{'-' * 66}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
