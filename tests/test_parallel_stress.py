"""Stress / property tests for the parallelism layer.

A single equivalence run cannot prove the absence of a race (races are
nondeterministic). These tests attack the invariants harder:

  I1/I2  determinism under RANDOMIZED finish orders, repeated many times, and
         IDENTICAL outcomes across every concurrency level C in {1,2,4,8,16}.
  I3     worst-case SAME-KEY contention (all tasks hammer one user / one
         pattern) must lose no write; consolidation interleaved with load must
         never tear or deadlock.
  Liveness: injected failures must not abort the batch nor leak a lock
         (a leaked lock would deadlock the next same-key task).

All deterministic-by-construction: each agent's observation depends only on its
(agent, subtask), never on a sibling or on shared state, and each task's ENT
write uses a DISJOINT key — so the merged result is schedule-independent and any
divergence is a real bug, not benign last-write-wins.
"""
from __future__ import annotations

import asyncio
import random

import pytest

from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.types import Pattern
from memory_manager.parallel import AgentResult, Delegation, ParallelExecutor
from memory_manager.task_queue import AsyncTaskQueue
from memory_manager.stores.working import WorkingMemoryStore


def _base() -> MemoryManager:
    return MemoryManager(config.Settings(embedding_backend="stub", llm_backend="stub"))


# A small, varied workload with multi-agent (parallelizable) tasks.
def _tasks(n: int, *, same_user: bool = False):
    shapes = [
        [Delegation("calendar", "list events"), Delegation("email", "read thread"),
         Delegation("search", "find people"), Delegation("calendar", "create event"),
         Delegation("email", "send invite")],                       # coordination
        [Delegation("calendar", "create event")],                   # single
        [Delegation("search", "search docs"), Delegation("email", "read budget")],  # exploratory
    ]
    out = []
    for i in range(n):
        dels = shapes[i % len(shapes)]
        out.append({"id": f"t{i}", "desc": f"task {i}", "user": "Bob" if same_user else f"u{i % 5}",
                    "pattern": Pattern.SINGLE_ACTION, "dels": dels,
                    "plan": [f"{d.agent_type}:{d.subtask}:{i}" for d in dels]})  # distinct signatures
    return out


async def _run(tasks, concurrency, *, rng=None, jitter=0.0, fail_ids=frozenset()):
    """Run the task set either sequentially (concurrency='seq') or via the queue
    at integer concurrency C. Returns (outcomes:dict, mem:tuple)."""
    base = _base()

    async def runner(agent_type, subtask, snap):
        if rng is not None and jitter:
            await asyncio.sleep(rng.uniform(0, jitter))   # randomize interleavings
        return AgentResult(agent_type, subtask, observation=f"{agent_type}|{subtask}")

    async def pipeline(spec):
        if spec["id"] in fail_ids:
            raise ValueError(f"injected failure in {spec['id']}")
        m = base.fork_for_task()
        m.wm.start_task(spec["id"], spec["desc"], spec["user"], "2026-06-08")
        m.wm.set_pattern(spec["pattern"])
        await m.run_parallel_agents(spec["dels"], runner)             # parallel waves
        obs = tuple(sorted(r.observation for r in m.wm.current.step_history))
        await m.complete_task_async(True, spec["plan"], [], {spec["id"]: 1})  # DISJOINT ent key
        return {"id": spec["id"], "obs": obs}

    if concurrency == "seq":
        results = [await pipeline(t) for t in tasks]
    else:
        results = await AsyncTaskQueue(pipeline, max_concurrency=concurrency,
                                       is_consolidating=base.is_consolidating).run(tasks)
    outcomes = {r["id"]: r["obs"] for r in results if "obs" in r}
    prof = base.ent.get(tasks[0]["user"]) if tasks else None
    mem = (len(base.stm), len(base.em), len(base.sm),
           tuple(sorted(prof.facts.keys())) if prof else ())
    return outcomes, mem, base


# ── I2: deterministic merge under randomized finish orders, many reps ────────
def test_wave_merge_is_deterministic_under_200_random_finish_orders():
    canonical = None
    for seed in range(200):
        rng = random.Random(seed)

        async def runner(agent_type, subtask, snap):
            await asyncio.sleep(rng.uniform(0, 0.003))      # random completion order
            return AgentResult(agent_type, subtask, observation=f"{agent_type}|{subtask}")

        async def go():
            wm = WorkingMemoryStore(); wm.start_task("t", "x", "Bob", "d")
            group = [Delegation("email", "b send"), Delegation("calendar", "a list"),
                     Delegation("search", "c find"), Delegation("calendar", "a create")]
            ex = ParallelExecutor(runner)
            res = await ex.execute_group(group, ex.snapshot(wm.current))
            ex.merge(res, wm)
            return tuple((r.agent, r.observation) for r in wm.current.step_history)

        order = asyncio.run(go())
        canonical = canonical or order
        assert order == canonical, f"merge order diverged at seed {seed}"


# ── I1/I2: identical outcomes + memory state across ALL concurrency levels ───
def test_outcomes_identical_across_concurrency_levels():
    tasks = _tasks(40)
    seq_out, seq_mem, _ = asyncio.run(_run(tasks, "seq"))
    for C in (1, 2, 4, 8, 16):
        out, mem, _ = asyncio.run(_run(tasks, C))
        assert out == seq_out, f"outcomes diverged at C={C}"
        assert mem == seq_mem, f"memory state diverged at C={C}"


# ── I1/I2: equivalence stable over many randomized-timing seeds ──────────────
def test_equivalence_stable_over_30_random_timing_seeds():
    tasks = _tasks(24)
    seq_out, seq_mem, _ = asyncio.run(_run(tasks, "seq"))
    for seed in range(30):
        rng = random.Random(1000 + seed)
        out, mem, _ = asyncio.run(_run(tasks, 4, rng=rng, jitter=0.002))
        assert out == seq_out, f"outcomes diverged under random timing seed {seed}"
        assert mem == seq_mem, f"memory diverged under random timing seed {seed}"


# ── I3: worst-case SAME-USER contention loses no ENT write ───────────────────
def test_max_contention_same_user_all_ent_writes_land():
    for rep in range(10):
        tasks = _tasks(30, same_user=True)                  # every task writes "Bob"
        rng = random.Random(rep)
        _out, mem, base = asyncio.run(_run(tasks, 8, rng=rng, jitter=0.001))
        keys = mem[3]
        assert set(keys) == {t["id"] for t in tasks}, f"lost an ENT write (rep {rep}): {keys}"


# ── I3: worst-case SAME-PATTERN contention loses no STM write ────────────────
def test_max_contention_same_pattern_all_stm_writes_land():
    tasks = _tasks(30, same_user=True)                      # all SINGLE_ACTION + distinct plans
    _out, mem, base = asyncio.run(_run(tasks, 8))
    # distinct signatures -> every successful task adds one STM bundle
    assert mem[0] == len(tasks), f"STM lost a write: {mem[0]} != {len(tasks)}"


# ── Liveness: injected failures don't abort the batch or leak a lock ─────────
def test_failure_injection_isolates_and_does_not_leak_locks():
    tasks = _tasks(20, same_user=True)
    fail = {"t3", "t7", "t11"}
    out, mem, base = asyncio.run(_run(tasks, 6, fail_ids=fail))
    # surviving tasks all produced outcomes
    assert set(out) == {t["id"] for t in tasks} - fail
    # every ENT lock that was created is released (no leak)
    for lk in base.lock_mgr._ent.values():
        assert not lk.locked(), "an ENT lock leaked after a failure"

    # and the system is still LIVE: a fresh same-user task completes promptly
    async def follow_up():
        m = base.fork_for_task()
        m.wm.start_task("after", "x", "Bob", "d"); m.wm.set_pattern(Pattern.SINGLE_ACTION)
        await asyncio.wait_for(
            m.complete_task_async(True, ["calendar:x"], [], {"after": 1}), timeout=2.0)
    asyncio.run(follow_up())  # raises TimeoutError if a lock had deadlocked the user


# ── I3: consolidation interleaved with load, repeated, no tear / deadlock ────
def test_consolidation_interleaved_with_load_repeatedly():
    for rep in range(8):
        base = _base()
        tasks = _tasks(12)

        async def runner(a, s, snap):
            await asyncio.sleep(0)
            return AgentResult(a, s, observation=f"{a}|{s}")

        async def pipeline(spec):
            m = base.fork_for_task()
            m.wm.start_task(spec["id"], spec["desc"], spec["user"], "d")
            m.wm.set_pattern(Pattern.SINGLE_ACTION)
            await m.run_parallel_agents(spec["dels"], runner)
            await m.complete_task_async(True, spec["plan"], [], {spec["id"]: 1})
            return {"id": spec["id"], "ok": True}

        q = AsyncTaskQueue(pipeline, max_concurrency=4, is_consolidating=base.is_consolidating)

        async def both():
            # consolidate concurrently with a fresh load; queue must hold tasks
            # out of phase-3 until the window closes — never tear, never hang.
            cons = asyncio.create_task(base.consolidate_async(min_cluster_size=1))
            await asyncio.sleep(0)
            res = await asyncio.wait_for(q.run(tasks), timeout=5.0)   # deadlock -> TimeoutError
            await cons
            return res

        res = asyncio.run(both())
        assert all(r.get("ok") for r in res), f"a task failed under consolidation (rep {rep})"
        assert base.is_consolidating() is False, "consolidation flag leaked"
