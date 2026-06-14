"""Orchestrator: drives one task through Phases 2-9 of the lifecycle.

This is the "outer loop" -- it owns ingestion, repeatedly calls
retrieve -> infer -> delegate -> (sub-agent runs) -> observe, and finally
calls complete_task. Every phase boundary is a direct call into MemoryManager
so the router's decisions are the only thing governing what data crosses
between memory types; the orchestrator itself never touches a store directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from environment.world import World
from memory_manager.llm import LLM
from memory_manager.manager import MemoryManager, TaskMetrics
from memory_manager.types import SubtaskMemory

MAX_STEPS = 6


class Orchestrator:
    def __init__(self, manager: MemoryManager, world: World, llm: LLM, agents: dict[str, "BaseAgent"]):
        self.manager = manager
        self.world = world
        self.llm = llm
        self.agents = agents  # name -> BaseAgent

    def run_task(self, description: str, username: str, date: str) -> tuple[TaskMetrics, dict]:
        """Run one full task end to end and return (metrics, trace) where
        `trace` is a small dict useful for printing/inspection in the demo."""
        ctx = self.manager.ingest_task(description, username, date)   # Phase 2
        plan: list[str] = []
        subtask_memories: list[SubtaskMemory] = []
        artifacts: list[Path] = []
        any_real_action = False

        for _step in range(MAX_STEPS):
            bundle = self.manager.retrieve()                                          # Phase 3
            output = self.manager.run_orchestrator(bundle, available_agents=list(self.agents))  # Phase 4
            self.manager.record_delegation(output)                                    # Phase 5

            agent_name = output.get("agent", "FINISH")
            if agent_name == "FINISH" or ctx.forced_finish is not None:
                break

            agent = self.agents.get(agent_name)
            if agent is None:
                # Orchestrator named an agent we don't have -- treat as a dead end.
                self.manager.record_observation(agent_name, "unknown_agent", f"OBSERVATION: no agent named '{agent_name}' is available.")
                continue

            plan.append(f"{agent_name}:{output.get('subtask', '')[:24]}")
            action, subtask, _inner = agent.act()                                     # Phase 6 + 7
            obs_str, structured, artifact = self.world.execute(                       # Phase 8 (env side)
                action, username=ctx.username, date=ctx.date, subtask=subtask)
            self.manager.record_observation(agent_name, action.get("action", "noop"), obs_str, structured)  # Phase 8 (WM side)

            success_step = action.get("action") != "noop"
            any_real_action = any_real_action or success_step
            subtask_memories.append(agent.to_subtask_memory(subtask, action, obs_str, success_step))
            if artifact is not None:
                artifacts.append(artifact)

        # Capture WM fields BEFORE complete_task() clears WM ('WM: DIES NOW').
        pattern_value = ctx.pattern.value if ctx.pattern else None
        forced_finish = ctx.forced_finish

        success = any_real_action and forced_finish is None and bool(subtask_memories)
        profile_updates = self._derive_profile_updates(ctx.shared_context)
        metrics = self.manager.complete_task(success, plan, subtask_memories, profile_updates)  # Phase 9

        trace = {
            "task_id": metrics.task_id,
            "pattern": pattern_value,
            "plan": plan,
            "artifacts": [str(p) for p in artifacts],
            "forced_finish": forced_finish,
        }
        return metrics, trace

    @staticmethod
    def _derive_profile_updates(shared_context: dict) -> Optional[dict]:
        """Tiny illustrative example of 'new info -> ENT': if a search step
        surfaced an event with an external attendee, remember them as a known
        contact. (Kept deliberately simple -- a real system would have an LLM
        decide what's worth persisting.)"""
        updates: dict = {}
        for _agent, obs in shared_context.items():
            for ev in obs.get("events", []) if isinstance(obs, dict) else []:
                for attendee in ev.get("attendees", []):
                    if "@" in attendee and "company.com" not in attendee:
                        updates.setdefault("known_external_contacts", [])
                        if attendee not in updates["known_external_contacts"]:
                            updates["known_external_contacts"].append(attendee)
        return updates or None
