"""BaseAgent: implements Phase 6 (Agent Inference) and Phase 7 (Agent Output).

Per the lifecycle table, agent inference reads three things every call --
PM (this agent's OWN procedural prompt), EM (pre-loaded subtask memories, "no
new FAISS query"), WM (the orchestrator's subtask instruction) -- and runs a
short inner loop with its own (shortest-lived) history and repeat-detection:
"Inner history: List of {action, obs} / Inner: READ+WRITE / Inner: max 2x /
Repeat: COMPARE / Repeat: every step".

The inner loop here is the agent drafting -> self-checking -> possibly
revising its action *before* committing to the one action that actually gets
executed in the environment (which is what produces the "real",
externally-visible observation recorded into WM/step_history at Phase 8).
This keeps "inner" (per-agent-call, shortest-lived, max 2x) cleanly distinct
from "outer" (per-task, step_history, lives until task end).
"""
from __future__ import annotations

import json
from typing import Optional

from memory_manager.llm import LLM
from memory_manager.manager import MemoryManager
from memory_manager.types import SubtaskMemory


class BaseAgent:
    def __init__(self, name: str, llm: LLM, manager: MemoryManager):
        self.name = name
        self.llm = llm
        self.manager = manager

    def act(self) -> tuple[dict, str, list[SubtaskMemory]]:
        """Run inference (with inner loop) -> commit one action (Phase 7).

        Returns (action_json, subtask_text, draft_subtask_memories) where the
        last element is what the orchestrator distills into this step's
        SubtaskMemory once the real-world observation is known.
        """
        prompt, memories, delegation = self.manager.agent_context(self.name)
        subtask = delegation.get("subtask", "")
        ctx = self.manager.wm.current

        system = self._system_prompt(prompt, memories)
        user_base = self._user_prompt(subtask, ctx)

        inner_history: list[dict] = []
        action: Optional[dict] = None
        for _ in range(2):  # 'Inner: max 2x'
            raw = self.llm.complete(system, user_base + self._render_inner(inner_history))
            draft = self._parse_action(raw)
            self_obs = self._self_check(draft, inner_history)
            inner_history.append({"action": draft, "obs": self_obs})
            if self_obs != "repeat_detected":   # 'Repeat: COMPARE' -- stop early on repeat
                action = draft
                break
        if action is None:
            action = inner_history[-1]["action"]

        self.manager.record_agent_action(action)
        return action, subtask, inner_history

    # -- prompt assembly ------------------------------------------------------

    def _system_prompt(self, pm_prompt: str, memories: list[SubtaskMemory]) -> str:
        lines = [f"ROLE: agent\nAGENT_NAME: {self.name}", pm_prompt]
        if memories:
            lines.append("PAST_SUBTASK_MEMORIES (pre-loaded from EM, no new FAISS query):")
            for mem in memories:
                lines.append(f"  - subtask='{mem.subtask}' action='{mem.action}' -> {mem.outcome}: {mem.observation_summary}")
        return "\n".join(lines)

    def _user_prompt(self, subtask: str, ctx) -> str:
        return f"USERNAME: {ctx.username}\nDATE: {ctx.date}\nSUBTASK: {subtask}"

    @staticmethod
    def _render_inner(inner_history: list[dict]) -> str:
        if not inner_history:
            return ""
        last = inner_history[-1]
        return f"\nPRIOR_DRAFT: {json.dumps(last['action'])} (self-check: {last['obs']}) -- refine if needed."

    # -- inner-loop helpers ----------------------------------------------------

    def _parse_action(self, raw: str) -> dict:
        try:
            action = json.loads(raw)
            assert isinstance(action, dict) and "action" in action
            return action
        except Exception:
            return {"app": self.name, "action": "noop", "user": "", "summary": "", "time": ""}

    @staticmethod
    def _self_check(draft: dict, inner_history: list[dict]) -> str:
        """'Repeat: COMPARE, every step' -- is this draft the same as the last one?"""
        if inner_history and inner_history[-1]["action"] == draft:
            return "repeat_detected"
        if draft.get("action") == "noop":
            return "draft looks like a no-op, consider a concrete action"
        return "draft looks actionable"

    # -- distillation (consumed by the orchestrator at Phase 9 prep) ----------

    def to_subtask_memory(self, subtask: str, action: dict, observation: str, success: bool) -> SubtaskMemory:
        return SubtaskMemory(
            agent=self.name, subtask=subtask, action=action.get("action", "noop"),
            observation_summary=observation[:160], outcome="success" if success else "failure",
        )
