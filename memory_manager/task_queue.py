"""LAYER 1: Async task queue — accept N tasks and run their full 10-phase
pipelines concurrently, each with its OWN isolated Working Memory.

Per-task isolation is the key invariant: WM is per-task (it 'dies' at task end),
so each concurrent task pipeline owns a fresh WM and never shares it. The shared
stores (PM/STM/EM/SM/ENT) ARE cross-task and are protected by MemoryLockManager.

Consolidation (Phase 10) is an EXCLUSIVE window: while it rewrites FAISS/PM, the
queue must not start new tasks (which would read stale/half-written vectors in
Phase 3). The queue checks `is_consolidating()` before dispatching.

Reference: DynTaskMAS APEE async execution; AdaptOrch parallel dispatch.
"""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable, Optional


# A task pipeline: (task_spec) -> result (awaitable). Each call must build and
# use its OWN per-task Working Memory (isolation).
TaskPipeline = Callable[[dict], Awaitable[dict]]


class AsyncTaskQueue:
    def __init__(self, pipeline: TaskPipeline, max_concurrency: int = 4,
                 is_consolidating: Optional[Callable[[], bool]] = None):
        self._pipeline = pipeline
        self._max_concurrency = max_concurrency
        self._sem: Optional[asyncio.Semaphore] = None   # created inside the loop (3.9-safe)
        self._is_consolidating = is_consolidating or (lambda: False)

    async def _run_one(self, task_spec: dict) -> dict:
        # Wait out any active consolidation window before entering Phase 3.
        while self._is_consolidating():
            await asyncio.sleep(0.05)
        async with self._sem:
            try:
                return await self._pipeline(task_spec)
            except Exception as e:  # one task failing must not kill the batch
                return {"task": task_spec.get("task", "?"), "error": str(e), "success": False}

    async def run(self, task_specs: list[dict]) -> list[dict]:
        """Accept N tasks, run their pipelines concurrently, return all results.
        Each task gets its own pipeline invocation (and thus its own WM)."""
        self._sem = asyncio.Semaphore(self._max_concurrency)   # 3.9: create in-loop
        coros = [asyncio.create_task(self._run_one(t)) for t in task_specs]
        return await asyncio.gather(*coros)
