"""STEP 4 — MemoryManager: the single front door all N tasks call.

Enforces the role/store split:
  * orchestrator side  -> PM  (read past trajectories, write on success)   [the read-WRITE store]
  * sub-agent side     -> Tool Memory (how to call a tool)                 [the READ-ONLY store]

The manager is deliberately thin: the concurrency behaviour lives in ProceduralMemory's strategy.
It aggregates the instrumentation so the benchmark can report per-store wait/latency.
"""
from __future__ import annotations

import time

from concurrent_mm.procedural_memory import ProceduralMemory, Trajectory
from concurrent_mm.tool_memory import ToolMemory


class MemoryManager:
    def __init__(self, tool_memory: ToolMemory, pm: ProceduralMemory):
        self.tool = tool_memory
        self.pm = pm
        self.tool_read_time_s = 0.0        # cumulative tool-memory read time (should stay tiny/flat)

    # -- orchestrator side (PM: read + write) --------------------------------
    async def orchestrator_read_pm(self, query: str, k: int = 3):
        return await self.pm.read(query, k)              # (hits, lock_wait)

    async def orchestrator_write_pm(self, traj: Trajectory):
        return await self.pm.write(traj)                 # lock_wait

    # -- sub-agent side (Tool Memory: read-only, lock-free) ------------------
    def subagent_read_tool_memory(self, app: str) -> dict:
        t0 = time.perf_counter()
        spec = self.tool.get(app)                        # O(1), no lock, immutable
        self.tool_read_time_s += time.perf_counter() - t0
        return spec

    # -- sync (thread-safe) API for the real threaded runner -----------------
    def orchestrator_read_pm_sync(self, query, k: int = 3):
        return self.pm.read_sync(query, k)

    def orchestrator_read_pm_scored(self, query, k: int = 1):
        return self.pm.read_sync_scored(query, k)      # [(cosine_score, trajectory), ...]

    def orchestrator_write_pm_sync(self, traj):
        self.pm.write_sync(traj)
