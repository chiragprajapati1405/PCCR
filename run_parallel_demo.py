#!/usr/bin/env python3
"""End-to-end demo of the PARALLEL PCCR architecture — all 7 layers, offline.

This drives the standalone `memory_manager/` package through its *async* path
to show the two dimensions of concurrency the architecture adds, while the
ρ-gate router (router.py) and the 10-phase lifecycle stay exactly as they are:

    PART A  Parallel sub-agents within ONE task  (Layers 3 + 4)
            A coordination task is decomposed into delegations, turned into a
            dependency DAG, and executed wave-by-wave: independent subtasks in a
            wave run concurrently against a frozen WM snapshot, then merge.

    PART B  Parallel tasks across the system      (Layers 1 + 2 + 5 + 6)
            N tasks run concurrently through AsyncTaskQueue. Each task forks a
            per-task manager (its OWN Working Memory) that shares the cross-task
            stores; Phase-3 routing runs once per task; Phase-9 STM/ENT writes
            go through per-resource locks (no lost updates).

    PART C  Consolidation exclusivity window      (Layer 7)
            Phase 10 raises is_consolidating(); the queue holds new tasks out of
            Phase-3 FAISS reads until the window closes.

Run:
    .venv/bin/python run_parallel_demo.py

Uses the deterministic stub backends (no API key, no network).
"""
from __future__ import annotations

import asyncio

from environment import World
from environment.tasks import USERNAME, seed_entities, seed_events, seed_inbox
from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.parallel import AgentResult, Delegation, DependencyAnalyzer, categorize
from run_task import (
    AGENT_CAPABILITY_PROFILES,
    AGENT_PROMPTS,
    ORCHESTRATOR_PROMPT,
    TASK_TYPE_RULES,
)

# agent_type -> (world app, read-action, write-action)
_APP = {
    "calendar": ("calendar_agent", "read_calendar", "create_event"),
    "email": ("email_agent", "read_email", "send_email"),
    "search": ("search_agent", "search", "search"),
}


def _make_action(agent_type: str, subtask: str, username: str, date: str) -> dict:
    app, read_act, write_act = _APP.get(agent_type, _APP["search"])
    cat = categorize(subtask)
    action = write_act if cat in ("create", "send") else read_act
    return {"app": app, "action": action, "user": username, "summary": subtask, "time": date}


def build_runner(world: World, username: str, date: str):
    """Async agent runner: (agent_type, subtask, wm_snapshot) -> AgentResult.

    The World call is synchronous and fast, so this returns quickly; in the real
    API harness each agent has seconds of LLM latency, and THAT is when the
    snapshot/staging/lock machinery earns its keep. The yield point keeps the
    event loop cooperative so a wave's agents genuinely interleave."""
    async def runner(agent_type: str, subtask: str, wm_snapshot) -> AgentResult:
        action = _make_action(agent_type, subtask, username, date)
        await asyncio.sleep(0)  # cooperative yield: let the wave interleave
        obs, _structured, _artifact = world.execute(
            action, username=username, date=date, subtask=subtask)
        return AgentResult(agent_type, subtask, action=action, observation=obs)

    return runner


def build_base() -> tuple[MemoryManager, World]:
    world = World(
        eml_dir=config.EML_DIR, ics_dir=config.ICS_DIR, answers_dir=config.ANSWERS_DIR,
        seed_inbox=seed_inbox(), seed_events=seed_events(),
    )
    base = MemoryManager(config.SETTINGS)
    base.bootstrap(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        agent_prompts=AGENT_PROMPTS,
        task_type_rules=TASK_TYPE_RULES,
        agent_capability_profiles=AGENT_CAPABILITY_PROFILES,
        entity_seed=seed_entities(),
    )
    return base, world


def _stores(m: MemoryManager) -> str:
    return (f"STM={len(m.stm)} EM={len(m.em)} SM={len(m.sm)} "
            f"PM.learned_rules={len(m.pm.learned_rules)}")


# ════════════════════════════════════════════════════════════════════
#  PART A — Parallel sub-agents within ONE task (Layers 3 + 4)
# ════════════════════════════════════════════════════════════════════
async def part_a(base: MemoryManager, world: World) -> None:
    print(f"\n{'=' * 72}\nPART A — Parallel sub-agents within ONE task (Layers 3 + 4)\n{'=' * 72}")
    date = "2026-06-08"
    base.ingest_task(
        "Schedule a sync with the team and email everyone the invite", USERNAME, date)
    base.retrieve()  # Phase 3 ρ-gate runs once for the task

    # The orchestrator stays in the loop: it emits one GROUP at a time and is
    # re-invoked after each group merges (diagram: 'More groups? -> back to
    # orchestrator'). Each group is itself split into a dependency DAG.
    rounds = [
        [Delegation("calendar", "list events for the team this week"),   # round 1: gather
         Delegation("search", "find everyone on the project team"),
         Delegation("email", "read recent thread about the sync")],
        [Delegation("calendar", "create the team sync event")],          # round 2: act
        [Delegation("email", "send the invite to the team")],            # round 3: notify
    ]

    async def planner(wm_ctx):
        # orchestrator inspects live (merged) WM, then emits the next group
        done_groups = getattr(planner, "_i", 0)
        print(f"\n  [orchestrator round {done_groups + 1}] WM has "
              f"{len(wm_ctx.step_history)} merged step(s) so far")
        if done_groups >= len(rounds):
            return []                                  # FINISH
        planner._i = done_groups + 1
        group = rounds[done_groups]
        for w in DependencyAnalyzer().analyze(list(group)):
            print(f"      group->wave [{w[0].category:6s}] ({len(w)} parallel): "
                  + ", ".join(f"{d.agent_type}:{d.subtask[:24]}" for d in w))
        return group

    runner = build_runner(world, USERNAME, date)
    waves, results = await base.run_planned_task(planner, runner)

    print(f"\nExecuted {len(results)} agents across {len(waves)} waves "
          f"({len(rounds)} orchestrator rounds). WM step history "
          f"(merged in deterministic order):")
    for rec in base.wm.current.step_history:
        print(f"  [{rec.agent:8s}] {rec.observation[:80]}")
    plan = [f"{d.agent_type}:{d.subtask[:20]}" for grp in rounds for d in grp]
    await base.complete_task_async(True, plan, [], {"last_demo": "part_a"})
    print(f"\nstores after Part A: {_stores(base)}")


# ════════════════════════════════════════════════════════════════════
#  PART B — Parallel tasks across the system (Layers 1 + 2 + 5 + 6)
# ════════════════════════════════════════════════════════════════════
def _demo_tasks() -> list[dict]:
    date = "2026-06-08"
    return [
        {"id": "t-cal", "description": "Add a dentist appointment to my calendar",
         "username": USERNAME, "date": date,
         "delegations": [Delegation("calendar", "create dentist appointment")]},
        {"id": "t-mail", "description": "Email the quarterly update to my manager",
         "username": USERNAME, "date": date,
         "delegations": [Delegation("email", "read the latest quarterly numbers"),
                         Delegation("email", "send the quarterly update to manager")]},
        {"id": "t-coord", "description": "Find the budget doc and forward it to finance",
         "username": USERNAME, "date": date,
         "delegations": [Delegation("search", "find the budget document"),
                         Delegation("email", "forward the budget doc to finance")]},
        {"id": "t-look", "description": "How many meetings do I have tomorrow?",
         "username": USERNAME, "date": date,
         "delegations": [Delegation("calendar", "list events for tomorrow")]},
    ]


async def part_b(base: MemoryManager, world: World) -> None:
    print(f"\n{'=' * 72}\nPART B — Parallel tasks across the system (Layers 1 + 2 + 5 + 6)\n{'=' * 72}")
    from memory_manager.task_queue import AsyncTaskQueue

    runner = build_runner(world, USERNAME, "2026-06-08")
    started, finished = [], []

    async def pipeline(spec: dict) -> dict:
        m = base.fork_for_task()                                    # own WM, SHARED stores+locks
        started.append(spec["id"])
        m.ingest_task(spec["description"], spec["username"], spec["date"])   # Phase 2
        bundle = m.retrieve()                                       # Phase 3 (ρ-gate, once)
        consulted = [s.name for s in bundle.decision.consulted_stores]
        waves, results = await m.run_parallel_agents(spec["delegations"], runner)  # Phase 6-8
        plan = [f"{d.agent_type}:{d.subtask[:20]}" for d in spec["delegations"]]
        metrics = await m.complete_task_async(                      # Phase 9 (locked writes)
            success=bool(results), plan=plan, subtask_memories=[],
            profile_updates={"last_task": spec["id"]})
        finished.append(spec["id"])
        return {"task": spec["id"], "success": metrics.success, "waves": len(waves),
                "consulted_phase3": consulted}

    queue = AsyncTaskQueue(pipeline, max_concurrency=4, is_consolidating=base.is_consolidating)
    results = await queue.run(_demo_tasks())

    print(f"\nDispatched {len(results)} tasks concurrently (max_concurrency=4), each with its "
          f"OWN Working Memory:")
    for r in sorted(results, key=lambda r: r["task"]):
        print(f"  {r['task']:8s} success={r['success']!s:5s} waves={r['waves']} "
              f"phase3_consulted={r['consulted_phase3']}")
    print(f"\nshared stores after Part B (written under per-resource locks): {_stores(base)}")
    print(f"ENT profile for {USERNAME}: "
          f"{base.ent.get(USERNAME).facts if base.ent.get(USERNAME) else None}")


# ════════════════════════════════════════════════════════════════════
#  PART C — Consolidation exclusivity window (Layer 7)
# ════════════════════════════════════════════════════════════════════
async def part_c(base: MemoryManager, world: World) -> None:
    print(f"\n{'=' * 72}\nPART C — Consolidation exclusivity window (Layer 7)\n{'=' * 72}")
    from memory_manager.task_queue import AsyncTaskQueue

    runner = build_runner(world, USERNAME, "2026-06-08")
    order: list[str] = []

    async def pipeline(spec: dict) -> dict:
        m = base.fork_for_task()
        m.ingest_task(spec["description"], spec["username"], spec["date"])
        m.retrieve()
        await m.run_parallel_agents(spec["delegations"], runner)
        await m.complete_task_async(True, ["calendar:x"], [], None)
        order.append("late-task")
        return {"task": spec["id"], "success": True}

    late = {"id": "t-late", "description": "Add a reminder for Friday", "username": USERNAME,
            "date": "2026-06-08", "delegations": [Delegation("calendar", "create a Friday reminder")]}
    queue = AsyncTaskQueue(pipeline, is_consolidating=base.is_consolidating)

    async def consolidate_job():
        order.append("consolidate-start")
        report = await base.consolidate_async(min_cluster_size=1)   # exclusive window
        order.append("consolidate-end")
        return report

    print(f"\nbefore consolidation: {_stores(base)}  is_consolidating={base.is_consolidating()}")
    # Launch consolidation and a new task concurrently; the task must wait out the window.
    consolidate_task = asyncio.create_task(consolidate_job())
    await asyncio.sleep(0)                       # let consolidation grab the flag first
    queue_task = asyncio.create_task(queue.run([late]))
    report, _ = await asyncio.gather(consolidate_task, queue_task)

    print(f"event order: {order}")
    assert order.index("consolidate-end") < order.index("late-task"), \
        "the queued task must run only AFTER the consolidation window closes"
    print("  -> the new task ran ONLY after consolidation closed (Phase-10 exclusivity holds).")
    print(f"consolidation report: {report}")
    print(f"after consolidation:  {_stores(base)}  is_consolidating={base.is_consolidating()}")


async def main() -> None:
    base, world = build_base()
    print("PCCR PARALLEL ARCHITECTURE DEMO (offline, stub backends)")
    print(f"router unchanged | 10 phases preserved | {_stores(base)}")
    await part_a(base, world)
    await part_b(base, world)
    await part_c(base, world)
    print(f"\n{'=' * 72}\nDONE — all 7 layers exercised: "
          f"\n  L1 AsyncTaskQueue  L2 per-task pipeline  L3 DependencyAnalyzer"
          f"\n  L4 ParallelExecutor (snapshot/merge)  L5 MemoryLockManager"
          f"\n  L6 async Phase-9 writes  L7 consolidation window\n{'=' * 72}")


if __name__ == "__main__":
    asyncio.run(main())
