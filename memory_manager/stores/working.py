"""Working Memory (WM): the volatile per-task scratchpad.

Per the lifecycle table, WM is the universal landing zone for "all current
task results that die when task ends" -- task ingestion fields, orchestrator
delegation output, agent actions, environment observations, loop-detection
state, and the final success/failure flag, all funnel through here, and it is
unconditionally cleared at Memory Storage ("WM: DIES NOW").
"""
from __future__ import annotations

from typing import Optional

from ..types import Pattern, StepRecord, TaskContext


class WorkingMemoryStore:
    def __init__(self):
        self._current: Optional[TaskContext] = None

    # -- Task Ingestion: WRITE for ALL items ("once per task, at task start") -

    def start_task(self, task_id: str, description: str, username: str, date: str) -> TaskContext:
        self._current = TaskContext(task_id=task_id, description=description, username=username, date=date)
        return self._current

    @property
    def current(self) -> TaskContext:
        if self._current is None:
            raise RuntimeError("No active task: call start_task() first")
        return self._current

    # -- Memory Retrieval / Orchestrator phases: READ + WRITE ---------------

    def set_pattern(self, pattern: Pattern) -> None:
        self.current.pattern = pattern

    def record_step(self, agent: str, action: str, observation: str) -> StepRecord:
        record = StepRecord(step=len(self.current.step_history), agent=agent, action=action, observation=observation)
        self.current.step_history.append(record)
        return record

    def share_context(self, agent: str, observation: dict) -> None:
        """Cross-agent shared context: 'Agent obs -> shared_context -> other agents see'."""
        self.current.shared_context[agent] = observation

    def history_text(self) -> str:
        """Render step history the way prompts consume it ('OBSERVATION: ...')."""
        lines = []
        for rec in self.current.step_history:
            lines.append(f"[{rec.agent}] ACTION: {rec.action}\nOBSERVATION: {rec.observation}")
        return "\n".join(lines)

    def set_last_action(self, action: dict) -> None:
        self.current.last_action = action

    def set_forced_finish(self, payload: Optional[dict]) -> None:
        self.current.forced_finish = payload

    def mark_outcome(self, success: bool) -> None:
        self.current.success = success

    # -- Memory Storage: CLEAR ("WM: DIES NOW") -----------------------------

    def clear(self) -> None:
        if self._current is not None:
            self._current.clear()
        self._current = None
