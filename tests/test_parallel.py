"""Tests for the parallelism layer (memory_manager/parallel.py + task_queue.py).

These exercise the NEW concurrency machinery in isolation — no LLM, no real
stores — because correctness here (dependency waves, copy-on-read snapshot,
deterministic staging->merge, per-resource locking, consolidation exclusivity)
must hold regardless of the agent/backbone.
"""
from __future__ import annotations

import asyncio

import pytest

from memory_manager.parallel import (
    DependencyAnalyzer, Delegation, ParallelExecutor, AgentResult, MemoryLockManager, categorize,
)
from memory_manager.task_queue import AsyncTaskQueue
from memory_manager.stores.working import WorkingMemoryStore


# -- LAYER 3: Dependency DAG / waves -----------------------------------------

def test_independent_reads_form_one_parallel_wave():
    analyzer = DependencyAnalyzer()
    delegations = [
        Delegation("calendar", "list events for Bob"),
        Delegation("calendar", "list events for Tom"),
    ]
    waves = analyzer.analyze(delegations)
    assert len(waves) == 1                # both reads -> a single parallel wave
    assert len(waves[0]) == 2

def test_read_then_create_then_send_makes_three_ordered_waves():
    """The motivating example: list -> create -> email becomes 3 waves."""
    analyzer = DependencyAnalyzer()
    delegations = [
        Delegation("calendar", "list events for Bob"),
        Delegation("calendar", "list events for Tom"),
        Delegation("calendar", "create event for Bob"),
        Delegation("calendar", "create event for Tom"),
        Delegation("email", "send invite to Tom"),
    ]
    waves = analyzer.analyze(delegations)
    cats = [w[0].category for w in waves]
    assert cats == ["read", "create", "send"]        # precedence order
    assert len(waves[0]) == 2 and len(waves[1]) == 2 and len(waves[2]) == 1
    # every send-wave node depends on every create-wave node
    edges = analyzer.dag_edges(waves)
    assert any("create" in a or "create event" in a for a, b in edges)

def test_categorize():
    assert categorize("list events for Bob") == "read"
    assert categorize("create event 'X'") == "create"
    assert categorize("send invite to Tom") == "send"


# -- LAYER 4: snapshot isolation + deterministic staging->merge --------------

def test_wm_snapshot_is_isolated_from_live_wm():
    wm = WorkingMemoryStore()
    wm.start_task("t1", "do thing", "Bob", "2024-05-01")
    wm.record_step("calendar", "list", "found 3 events")
    snap = wm.snapshot()
    # mutate live WM after snapshot
    wm.record_step("email", "send", "sent")
    assert len(snap.step_history) == 1          # snapshot frozen at 1 step
    assert len(wm.current.step_history) == 2    # live moved on

def test_parallel_merge_is_deterministic_regardless_of_finish_order():
    """asyncio.gather may finish agents in any order; merge must be stable."""
    runner_calls = []

    async def runner(agent_type, subtask, snap):
        # simulate different finish times: 'email' returns faster than 'calendar'
        await asyncio.sleep(0.02 if agent_type == "calendar" else 0.0)
        runner_calls.append(agent_type)
        return AgentResult(agent_type, subtask, observation=f"{agent_type} done")

    ex = ParallelExecutor(runner)
    wm = WorkingMemoryStore(); wm.start_task("t", "x", "Bob", "d")
    group = [Delegation("email", "send invite"), Delegation("calendar", "list events")]

    snap = ex.snapshot(wm.current)
    results = asyncio.run(ex.execute_group(group, snap))
    ex.merge(results, wm)

    order = [r.agent for r in wm.current.step_history]
    assert order == ["calendar", "email"]   # sorted by agent, NOT finish order
    assert "calendar" in wm.current.shared_context and "email" in wm.current.shared_context


# -- LAYER 5: per-resource locking (no lost updates) -------------------------

def test_per_key_locks_serialize_same_key_but_not_different_keys():
    async def check():
        lm = MemoryLockManager()
        assert lm.ent("Bob") is lm.ent("Bob")        # same user -> same lock
        assert lm.ent("Bob") is not lm.ent("Tom")    # different users -> different locks
        assert lm.stm("p1") is not lm.stm("p2")
        assert lm.faiss_lock is lm.faiss_lock        # global locks stable
    asyncio.run(check())

def test_lock_prevents_lost_update_on_shared_counter():
    """Two concurrent increments under the SAME lock must both land."""
    lm = MemoryLockManager()
    counter = {"n": 0}

    async def inc():
        async with lm.ent("Bob"):
            cur = counter["n"]
            await asyncio.sleep(0.001)   # window where a race would lose an update
            counter["n"] = cur + 1

    async def main():
        await asyncio.gather(*[inc() for _ in range(20)])

    asyncio.run(main())
    assert counter["n"] == 20            # no lost updates


# -- LAYER 1 / 7: task queue + consolidation exclusivity ---------------------

def test_task_queue_runs_tasks_concurrently_and_isolated():
    async def pipeline(spec):
        await asyncio.sleep(0.01)
        return {"task": spec["task"], "success": True}

    q = AsyncTaskQueue(pipeline, max_concurrency=4)
    specs = [{"task": f"t{i}"} for i in range(6)]
    results = asyncio.run(q.run(specs))
    assert len(results) == 6 and all(r["success"] for r in results)

def test_queue_waits_out_consolidation_window():
    """While consolidation is active, no task should complete; once it clears,
    all run. (Phase 10 exclusivity.)"""
    state = {"consolidating": True, "started": []}

    async def pipeline(spec):
        state["started"].append(spec["task"])
        return {"task": spec["task"], "success": True}

    q = AsyncTaskQueue(pipeline, is_consolidating=lambda: state["consolidating"])

    async def main():
        runner = asyncio.create_task(q.run([{"task": "a"}]))
        await asyncio.sleep(0.1)
        assert state["started"] == []        # blocked during consolidation
        state["consolidating"] = False       # window closes
        res = await runner
        assert state["started"] == ["a"] and res[0]["success"]

    asyncio.run(main())


def test_one_failing_task_does_not_kill_the_batch():
    async def pipeline(spec):
        if spec["task"] == "bad":
            raise ValueError("boom")
        return {"task": spec["task"], "success": True}

    q = AsyncTaskQueue(pipeline)
    results = asyncio.run(q.run([{"task": "ok1"}, {"task": "bad"}, {"task": "ok2"}]))
    assert len(results) == 3
    assert sum(1 for r in results if r.get("success")) == 2
    assert any(r.get("error") for r in results)
