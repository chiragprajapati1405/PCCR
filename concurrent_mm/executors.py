"""STEP 5 — Executors (pluggable).

The research contribution (the concurrent memory manager) is pure software, so we measure it with a
ModeledExecutor first: one tool/LLM step = an `await asyncio.sleep(call_latency)`. Because asyncio
overlaps a real network await exactly as it overlaps a sleep, the modeled wall-clock is what a real
I/O-bound parallel run would achieve — with ZERO API cost and no rate limits.

DockerExecutor (Step 8, later) swaps in the real one-container OfficeBench tool execution.
"""
from __future__ import annotations

import asyncio


class ModeledExecutor:
    """One sub-agent action = a fixed modeled latency (measured gpt-oss-120b p50 = 0.516s/call)."""

    def __init__(self, call_latency: float = 0.516):
        self.call_latency = call_latency
        self.actions = 0

    async def run_action(self, app: str, subtask: str, usage: dict) -> dict:
        await asyncio.sleep(self.call_latency)           # the I/O-bound step (overlaps across tasks)
        self.actions += 1
        # a modeled "successful" step record (this is what accumulates in WM)
        return {"agent": app, "action": f"{app}.<call>", "subtask": subtask, "obs": "ok"}


class DockerExecutor:
    """STEP 8 placeholder — real one-container OfficeBench execution with per-task workdirs.
    Implemented in Phase 3; kept here so the executor interface is stable."""

    def __init__(self, container: str = "cmm-bench"):
        self.container = container

    async def run_action(self, app: str, subtask: str, usage: dict) -> dict:
        raise NotImplementedError("DockerExecutor is Phase 3 (Step 8).")
