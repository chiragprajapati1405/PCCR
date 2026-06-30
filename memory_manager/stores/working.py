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

    # -- Parallel agents: snapshot (copy-on-read) + staging merge -----------

    def snapshot(self) -> TaskContext:
        """Frozen deep copy of the current task context. All agents in a
        parallel wave read from this snapshot, never from live WM, so they
        cannot race on step_history / shared_context (CompArch copy-on-read)."""
        import copy
        return copy.deepcopy(self.current)

    def merge_staging(self, staged: list) -> None:
        """Merge a wave's staged results into LIVE WM, in deterministic sorted
        order (by agent), after ALL agents in the wave have finished. Each
        staged item is (agent, subtask, observation) (AISAC blackboard append)."""
        for agent, subtask, observation in sorted(staged, key=lambda x: (x[0], x[1])):
            self.record_step(agent=agent, action=subtask, observation=observation)
            self.share_context(agent, {"last_subtask": subtask, "last_observation": observation})

    # -- Memory Storage: CLEAR ("WM: DIES NOW") -----------------------------

    def clear(self) -> None:
        if self._current is not None:
            self._current.clear()
        self._current = None
