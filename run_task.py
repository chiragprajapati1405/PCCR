#!/usr/bin/env python3
"""End-to-end demo: runs the task suite through the full ten-phase lifecycle,
prints the router's decision trace, and verifies all memory types get
exercised. This is the script to run to *see* the Phase-Conditioned Cascading
Memory Router in action.

Usage:
    .venv/bin/python run_task.py            # one batch + consolidation + a
                                             # second batch (to show learning)
"""
from __future__ import annotations

import json

from agents import BaseAgent, Orchestrator
from environment import World
from environment.tasks import USERNAME, iter_tasks, seed_entities, seed_events, seed_inbox
from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.types import MemoryType

AGENT_NAMES = ["search_agent", "calendar_agent", "email_agent"]

ORCHESTRATOR_PROMPT = (
    "You coordinate a team of specialist agents to complete a user's task. "
    "At each step, decide which ONE agent should act next (or 'FINISH' if the "
    "task is complete), and give them a precise subtask instruction. "
    "Respond as JSON: {\"thought\": str, \"agent\": <name|FINISH>, \"subtask\": str, "
    "\"plan_driven\": bool, \"loop_detected\": bool}."
)

AGENT_PROMPTS = {
    "search_agent": (
        "You search across the user's email and calendar to gather context. "
        "Respond as JSON: {\"app\": \"search_agent\", \"action\": \"search\", "
        "\"user\": <username>, \"summary\": <what you searched for>, \"time\": <date>}."
    ),
    "calendar_agent": (
        "You manage the user's calendar: you can look up events or create new ones. "
        "Respond as JSON: {\"app\": \"calendar_agent\", \"action\": \"create_event\"|\"read_calendar\", "
        "\"user\": <username>, \"summary\": <event/query description>, \"time\": <date>}."
    ),
    "email_agent": (
        "You manage the user's email: you can read the inbox or send messages. "
        "Respond as JSON: {\"app\": \"email_agent\", \"action\": \"send_email\"|\"read_email\", "
        "\"user\": <username>, \"summary\": <subject/body or query>, \"time\": <date>}."
    ),
}

TASK_TYPE_RULES = {
    "A": "LOOKUP: prefer the search agent once; do not take side-effecting actions.",
    "B": "SINGLE_ACTION: delegate to exactly the one agent the action concerns, then FINISH.",
    "C": "COORDINATION: search for context first, then act (e.g. calendar), then notify (email).",
    "D": "RECURRING: this matches a known template -- follow the cached plan if one is provided.",
    "E": "EXPLORATORY: search broadly first; the right next step may only become clear after that.",
}

AGENT_CAPABILITY_PROFILES = {
    "search_agent": {"reads": ["email", "calendar"], "writes": ["answers"]},
    "calendar_agent": {"reads": ["calendar"], "writes": ["calendar"]},
    "email_agent": {"reads": ["email"], "writes": ["email"]},
}


def build_system() -> tuple[MemoryManager, Orchestrator]:
    world = World(
        eml_dir=config.EML_DIR, ics_dir=config.ICS_DIR, answers_dir=config.ANSWERS_DIR,
        seed_inbox=seed_inbox(), seed_events=seed_events(),
    )
    manager = MemoryManager(config.SETTINGS)
    manager.bootstrap(
        orchestrator_prompt=ORCHESTRATOR_PROMPT,
        agent_prompts=AGENT_PROMPTS,
        task_type_rules=TASK_TYPE_RULES,
        agent_capability_profiles=AGENT_CAPABILITY_PROFILES,
        entity_seed=seed_entities(),
    )
    agents = {name: BaseAgent(name, manager.llm, manager) for name in AGENT_NAMES}
    orchestrator = Orchestrator(manager, world, manager.llm, agents)
    return manager, orchestrator


def run_batch(manager: MemoryManager, orchestrator: Orchestrator, label: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    for description, username, date, hint in iter_tasks():
        metrics, trace = orchestrator.run_task(description, username, date)
        print(f"\n--- task {metrics.task_id} (hinted pattern={hint}, classified={trace['pattern']}) ---")
        print(f"  description : {description}")
        print(f"  plan        : {trace['plan']}")
        print(f"  artifacts   : {trace['artifacts']}")
        print(f"  success     : {metrics.success}  steps={metrics.steps}  "
              f"orch_calls={metrics.orchestrator_calls}  duration={metrics.duration_s * 1000:.1f}ms")
        if trace["forced_finish"]:
            print(f"  forced_finish: {trace['forced_finish']}")


def print_routing_trace(manager: MemoryManager, last_n: int) -> None:
    print(f"\n{'-' * 70}\nRouter decision trace (most recent {last_n})\n{'-' * 70}")
    for line in manager.routing_trace()[-last_n:]:
        print(line)
        print()


def verify_all_memory_types_exercised(manager: MemoryManager) -> None:
    touched: set[MemoryType] = set()
    for decision in manager.router.decision_log:
        touched.update(decision.consulted_stores)
    touched.add(MemoryType.PM)   # bootstrap + every agent/orchestrator call
    touched.add(MemoryType.WM)   # every task
    if len(manager.stm):
        touched.add(MemoryType.STM)
    if len(manager.em):
        touched.add(MemoryType.EM)
    if len(manager.sm):
        touched.add(MemoryType.SM)
    if any(manager.ent.get(u) for u in {USERNAME}):
        touched.add(MemoryType.ENT)
    touched.add(MemoryType.EXT)  # .eml/.ics/.answer files written by World

    print(f"\n{'-' * 70}\nMemory-type coverage check\n{'-' * 70}")
    for mt in MemoryType:
        if mt is MemoryType.NONE:
            continue
        mark = "[x]" if mt in touched else "[ ]"
        print(f"  {mark} {mt.name:5s} ({mt.value})")
    missing = {mt for mt in MemoryType if mt is not MemoryType.NONE} - touched
    if missing:
        print(f"  WARNING: not exercised: {[m.name for m in missing]}")
    else:
        print("  All memory types were exercised at least once. <-")


def print_store_sizes(manager: MemoryManager, label: str) -> None:
    print(f"\n[{label}] store sizes -- STM={len(manager.stm)} bundles, "
          f"EM={len(manager.em)} episodes, SM={len(manager.sm)} facts, "
          f"PM.learned_rules={len(manager.pm.learned_rules)}")


def main() -> None:
    manager, orchestrator = build_system()

    run_batch(manager, orchestrator, "BATCH 1 (cold start -- stores mostly empty)")
    print_store_sizes(manager, "after batch 1")

    print(f"\n{'-' * 70}\nConsolidation (Phase 10 -- batch, after-the-fact distillation)\n{'-' * 70}")
    report = manager.consolidate(min_cluster_size=1)
    print(f"  {json.dumps(report)}")
    print_store_sizes(manager, "after consolidation")

    run_batch(manager, orchestrator, "BATCH 2 (warm -- STM/EM/SM now populated)")
    print_store_sizes(manager, "after batch 2")

    print_routing_trace(manager, last_n=8)
    verify_all_memory_types_exercised(manager)

    print(f"\n{'-' * 70}\nTask metrics (all tasks)\n{'-' * 70}")
    for m in manager.metrics_summary():
        print(f"  {m}")


if __name__ == "__main__":
    main()
