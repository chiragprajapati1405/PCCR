"""Parallelism layer for PCCR — two dimensions of concurrency:

  1. Parallel sub-agents : N agents run simultaneously within ONE task
                           (Phase 6-8), grouped into dependency waves.
  2. Parallel tasks      : N tasks run simultaneously across the system
                           (see task_queue.AsyncTaskQueue), each with its own
                           isolated Working Memory.

This module holds the three new building blocks; the ρ-gate router
(router.py) is unchanged — routing happens once per task in Phase 3, before any
parallel execution, so the execution model is orthogonal to the routing model.

Design references (for the methodology):
  - DynTaskMAS (arXiv:2503.07675): SACMS shared context + APEE async execution
  - GAP (arXiv:2510.25320): DAG-based dependency modeling of subtasks
  - SPOQ (arXiv:2606.03115): wave-based topological dispatch
  - MIRIX (arXiv:2507.07957): asyncio.gather parallel writes + per-resource locks
  - CompArch Memory (arXiv:2603.10062): copy-on-read snapshot consistency
  - AISAC (arXiv:2511.14043): blackboard append/patch merge
  - AdaptOrch (arXiv:2602.16873): parallel/sequential executor switching
"""
from __future__ import annotations

import asyncio
import copy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional


# ════════════════════════════════════════════════════════════════════
#  LAYER 5: Memory Lock Manager (per-resource locks for parallel tasks)
# ════════════════════════════════════════════════════════════════════
class MemoryLockManager:
    """Per-resource asyncio locks so N parallel tasks can write shared stores
    without lost updates (MIRIX per-agent/per-resource lock pattern).

    Granularity matters: STM is locked PER PATTERN KEY and ENT PER USERNAME, so
    two tasks touching *different* patterns/users never block each other; only
    genuine contention on the same key serializes. FAISS and PM use a single
    lock each because their writes (Phase 10) are global, batch, and rare.
    """

    def __init__(self):
        # Locks are created LAZILY on first access (must be inside a running
        # event loop on Python 3.9, where asyncio.Lock() binds to the loop).
        self._stm: dict[str, asyncio.Lock] = {}    # per pattern key
        self._ent: dict[str, asyncio.Lock] = {}    # per username
        self._named: dict[str, asyncio.Lock] = {}  # faiss / pm / history

    @staticmethod
    def _get(d: dict, key: str) -> asyncio.Lock:
        lk = d.get(key)
        if lk is None:
            lk = asyncio.Lock()
            d[key] = lk
        return lk

    def stm(self, pattern: str) -> asyncio.Lock:
        return self._get(self._stm, pattern)

    def ent(self, username: str) -> asyncio.Lock:
        return self._get(self._ent, username)

    @property
    def faiss_lock(self) -> asyncio.Lock:   # global: EM/SM vector writes (Phase 10)
        return self._get(self._named, "faiss")

    @property
    def pm_lock(self) -> asyncio.Lock:      # global: learned-rule writes (Phase 10)
        return self._get(self._named, "pm")

    @property
    def history_lock(self) -> asyncio.Lock:  # routing-history atomic updates
        return self._get(self._named, "history")


# ════════════════════════════════════════════════════════════════════
#  LAYER 3: Dependency Analyzer (build a DAG of subtasks -> waves)
# ════════════════════════════════════════════════════════════════════
@dataclass
class Delegation:
    """One orchestrator delegation: an agent + its subtask instruction."""
    agent_type: str
    subtask: str
    category: str = ""        # filled by the analyzer: read | create | send | finish | other


# Action precedence: reads gather info first; creates/writes depend on reads;
# sends/notifications depend on creates; finish is last. Subtasks in the SAME
# category are independent -> one parallel wave; categories are sequential.
_CATEGORY_ORDER = {"read": 0, "create": 1, "send": 2, "finish": 3, "other": 1}


def categorize(subtask: str) -> str:
    s = subtask.lower()
    if any(k in s for k in ("list", "read", "search", "find", "fetch", "look up", "get ", "what", "how many")):
        return "read"
    if any(k in s for k in ("send", "email", "notify", "remind", "reply", "forward")):
        return "send"
    if any(k in s for k in ("create", "add", "schedule", "write", "set ", "book", "delete", "update")):
        return "create"
    if "finish" in s:
        return "finish"
    return "other"


class DependencyAnalyzer:
    """Turn a flat list of delegations into ordered PARALLEL groups (waves).

    A wave = subtasks with no dependency on each other (run concurrently). Waves
    are ordered by action precedence (read -> create -> send -> finish), so a
    'create' never starts before the 'read's it might depend on, and a 'send'
    (which needs created details) waits for the creates. This is a coarse but
    sound DAG: every node in wave k+1 conservatively depends on all of wave k.
    """

    def analyze(self, delegations: list[Delegation]) -> list[list[Delegation]]:
        for d in delegations:
            if not d.category:
                d.category = categorize(d.subtask)
        buckets: dict[int, list[Delegation]] = defaultdict(list)
        for d in delegations:
            buckets[_CATEGORY_ORDER.get(d.category, 1)].append(d)
        # waves in precedence order; within a wave, order deterministically by agent
        waves = []
        for level in sorted(buckets):
            wave = sorted(buckets[level], key=lambda d: (d.agent_type, d.subtask))
            waves.append(wave)
        return waves

    def dag_edges(self, waves: list[list[Delegation]]) -> list[tuple[str, str]]:
        """Explicit edges (for inspection/tests): every node depends on every
        node in the previous wave."""
        edges = []
        for i in range(1, len(waves)):
            for prev in waves[i - 1]:
                for cur in waves[i]:
                    edges.append((f"{prev.agent_type}:{prev.subtask[:20]}",
                                  f"{cur.agent_type}:{cur.subtask[:20]}"))
        return edges


# ════════════════════════════════════════════════════════════════════
#  LAYER 4: Parallel Agent Execution (snapshot -> staging -> gather -> merge)
# ════════════════════════════════════════════════════════════════════
@dataclass
class AgentResult:
    agent_type: str
    subtask: str
    action: dict = field(default_factory=dict)
    observation: str = ""


# An agent runner: (agent_type, subtask, wm_snapshot) -> AgentResult (awaitable).
AgentRunner = Callable[[str, str, object], Awaitable[AgentResult]]


class ParallelExecutor:
    """Runs one wave of agents concurrently with copy-on-read isolation and a
    deterministic staging->merge, so parallel agents never race on Working
    Memory (CompArch snapshot + AISAC blackboard-merge patterns).

    Per wave:
      1. SNAPSHOT  — freeze WM; all agents in the wave read this frozen copy.
      2. STAGING   — each agent writes to its own staging slot, never live WM.
      3. GATHER    — asyncio.gather runs the wave concurrently.
      4. MERGE     — after ALL finish, merge staging -> live WM in sorted order
                     (deterministic, so results are reproducible regardless of
                     which agent finished first).
    """

    def __init__(self, runner: AgentRunner):
        self._runner = runner

    @staticmethod
    def snapshot(working_memory) -> object:
        """Copy-on-read: a frozen WM the whole wave reads from."""
        return copy.deepcopy(working_memory)

    async def execute_group(self, group: list[Delegation], wm_snapshot) -> list[AgentResult]:
        """Run a wave concurrently against the frozen snapshot; return staging."""
        coros = [self._runner(d.agent_type, d.subtask, wm_snapshot) for d in group]
        results = await asyncio.gather(*coros)
        # deterministic order independent of completion order
        return sorted(results, key=lambda r: (r.agent_type, r.subtask))

    @staticmethod
    def merge(results: list[AgentResult], wm_store) -> None:
        """Merge staged results into LIVE working memory, in sorted order, so
        the next wave sees them. Appends one step per agent + updates
        shared_context (blackboard append/patch)."""
        for r in sorted(results, key=lambda r: (r.agent_type, r.subtask)):
            wm_store.record_step(agent=r.agent_type, action=r.subtask, observation=r.observation)
            wm_store.share_context(r.agent_type, {"last_subtask": r.subtask,
                                                   "last_observation": r.observation})
