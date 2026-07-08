"""STEP 2 — Procedural Memory (PM): shared, read + WRITE.

ONE append-only log of successful FULL trajectories (plan + sub-agent steps). No pattern key —
retrieval is by similarity over the task text (general, benchmark-agnostic).

The whole point of the study lives here: how much do locks hamper concurrency? PM supports three
strategies so the benchmark can MEASURE it rather than guess:

  * "lockfree"  — append-only + immutable entries + snapshot reads. Reads grab the current length
                  and scan that prefix (a consistent snapshot); writes append atomically. No lock is
                  held across the modeled access latency, so reads AND writes overlap freely.
                  Consistency = snapshot isolation (a reader may miss a just-appended entry — benign
                  for an optimization store). THIS is the max-concurrency design.
  * "global"    — one lock around every read and write. Serializes all PM access (the naive baseline).
  * "rwlock"    — readers share, writer exclusive. Reads overlap; only the (rare) write serializes.

`access_latency` models the store-access cost (e.g. a scan / remote fetch). A strategy "hampers
concurrency" exactly to the extent it holds a lock across that latency.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Trajectory:
    task: str
    plan: list                         # orchestrator-level plan
    subagent_steps: list               # [{agent, action, obs}, ...]  ← sub-agent memory lives here
    embedding: object = None           # 384-d normalized vector (filled by the embedder on write)


def _jaccard(a: str, b: str) -> float:
    """Dependency-free fallback similarity (token Jaccard) — used when no embedder is set."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return (len(sa & sb) / len(sa | sb)) if (sa or sb) else 0.0


class _AsyncRWLock:
    """Minimal readers-writer lock: many concurrent readers, exclusive writer."""

    def __init__(self):
        self._readers = 0
        self._rmutex = asyncio.Lock()      # protects the reader count
        self._wlock = asyncio.Lock()       # held while any reader OR the writer is active

    async def acquire_read(self):
        async with self._rmutex:
            self._readers += 1
            if self._readers == 1:
                await self._wlock.acquire()

    async def release_read(self):
        async with self._rmutex:
            self._readers -= 1
            if self._readers == 0:
                self._wlock.release()

    async def acquire_write(self):
        await self._wlock.acquire()

    def release_write(self):
        self._wlock.release()


class ProceduralMemory:
    def __init__(self, strategy: str = "lockfree", access_latency: float = 0.02, embedder=None):
        assert strategy in ("lockfree", "global", "rwlock")
        self.strategy = strategy
        self.access_latency = access_latency
        self.embedder = embedder                    # callable text -> normalized 384-d np.array (or None)
        self._elock = threading.Lock()              # serialize encode() across worker threads
        self._log: list[Trajectory] = []            # append-only; entries never mutated
        self._global = asyncio.Lock()
        self._rw = _AsyncRWLock()
        # instrumentation
        self.read_count = 0
        self.write_count = 0
        self.read_wait_s = 0.0
        self.write_wait_s = 0.0

    def _embed(self, text: str):
        with self._elock:                           # torch/ST forward pass isn't thread-safe
            return self.embedder(text)

    def _score(self, query, upto: int, k: int, qvec=None):
        """Rank the first `upto` trajectories against the query. VECTOR search (cosine on normalized
        embeddings) when an embedder is set; token-Jaccard fallback otherwise."""
        if self.embedder is not None:
            import numpy as np
            if qvec is None:
                qvec = self._embed(query)
            scored = [(float(np.dot(qvec, t.embedding)) if t.embedding is not None else -1.0, t)
                      for t in self._log[:upto]]
        else:
            scored = [(_jaccard(query, t.task), t) for t in self._log[:upto]]
        scored.sort(key=lambda x: -x[0])
        return [t for _s, t in scored[:k]]

    async def read(self, query: str, k: int = 3):
        """Return the top-k most similar past trajectories. Records lock-wait separately from work."""
        self.read_count += 1
        if self.strategy == "lockfree":
            wait = 0.0
            snapshot_len = len(self._log)                    # snapshot: consistent prefix, no lock
            await asyncio.sleep(self.access_latency)         # modeled access cost — overlaps freely
            hits = self._score(query, snapshot_len, k)
        elif self.strategy == "global":
            t0 = time.perf_counter()
            await self._global.acquire()
            wait = time.perf_counter() - t0
            try:
                await asyncio.sleep(self.access_latency)     # held UNDER the lock -> serializes
                hits = self._score(query, len(self._log), k)
            finally:
                self._global.release()
        else:  # rwlock
            t0 = time.perf_counter()
            await self._rw.acquire_read()
            wait = time.perf_counter() - t0
            try:
                await asyncio.sleep(self.access_latency)     # readers share -> overlap
                hits = self._score(query, len(self._log), k)
            finally:
                await self._rw.release_read()
        self.read_wait_s += wait
        return hits, wait

    async def write(self, traj: Trajectory):
        """Append one successful trajectory. Records lock-wait."""
        self.write_count += 1
        if self.strategy == "lockfree":
            wait = 0.0
            await asyncio.sleep(self.access_latency)         # modeled write cost — overlaps
            self._log.append(traj)                           # atomic under the GIL, no lock
        elif self.strategy == "global":
            t0 = time.perf_counter()
            await self._global.acquire()
            wait = time.perf_counter() - t0
            try:
                await asyncio.sleep(self.access_latency)
                self._log.append(traj)
            finally:
                self._global.release()
        else:  # rwlock — writer is exclusive
            t0 = time.perf_counter()
            await self._rw.acquire_write()
            wait = time.perf_counter() - t0
            try:
                await asyncio.sleep(self.access_latency)
                self._log.append(traj)
            finally:
                self._rw.release_write()
        self.write_wait_s += wait
        return wait

    # -- sync (thread-safe) API for the real threaded runner (lock-free: GIL-atomic) --
    def read_sync(self, query: str, k: int = 3):
        snapshot_len = len(self._log)                    # consistent prefix, no lock
        self.read_count += 1
        return self._score(query, snapshot_len, k)       # VECTOR search (or Jaccard fallback)

    def read_sync_scored(self, query: str, k: int = 1):
        """Like read_sync but returns [(cosine_score, trajectory), ...] so the trace can log the
        similarity score of the retrieved trajectory."""
        snapshot_len = len(self._log)
        self.read_count += 1
        if self.embedder is not None:
            import numpy as np
            qvec = self._embed(query)
            scored = [(float(np.dot(qvec, t.embedding)) if t.embedding is not None else -1.0, t)
                      for t in self._log[:snapshot_len]]
        else:
            scored = [(_jaccard(query, t.task), t) for t in self._log[:snapshot_len]]
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def write_sync(self, traj: "Trajectory"):
        if self.embedder is not None and traj.embedding is None:
            traj.embedding = self._embed(traj.task)      # embed once, at write time
        self._log.append(traj)                           # atomic under the GIL
        self.write_count += 1

    def __len__(self):
        return len(self._log)
