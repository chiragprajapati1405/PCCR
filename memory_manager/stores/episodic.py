"""Episodic Memory (EM): specific past experiences ("what happened").

Per the lifecycle table EM stores `FullTaskMemory` objects, is FAISS-indexed,
"persists on disk", and is read in two distinct ways:
  1. On-demand similarity search during Memory Retrieval (a STM miss + the
     task pattern says episodic context would help).
  2. "Plan-driven loading" during Orchestrator Output: when the orchestrator's
     delegation reveals which agents will run, EM is queried for THOSE agents'
     past subtask memories and "pre-loaded" -- "no new FAISS query" at Agent
     Inference time (see manager.py for how these two paths differ).

Note what is conspicuously absent: per the lifecycle table, `on_task_complete`
(Memory Storage) does NOT write to EM -- only to STM (+ENT). EM is populated
exclusively in bulk, after the fact, by Consolidation's "LLM curator -> FAISS
+ JSON". That is a deliberate quality gate: raw per-task traces land in the
cheap STM cache immediately, while only curated, distilled experiences earn a
permanent slot in episodic memory. `add`/`add_many` below are consolidation
operations -- see manager.py `consolidate()`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..types import FullTaskMemory, Pattern, SubtaskMemory
from ._faiss_index import PersistentVectorIndex


def _to_dict(mem: FullTaskMemory) -> dict:
    payload = {
        "task_id": mem.task_id,
        "description": mem.description,
        "pattern": mem.pattern.value,
        "plan": mem.plan,
        "step_signature": mem.step_signature,
        "outcome": mem.outcome,
        "timestamp": mem.timestamp,
        "subtask_memories": [vars(sm) for sm in mem.subtask_memories],
    }
    return payload


def _from_dict(payload: dict) -> FullTaskMemory:
    return FullTaskMemory(
        task_id=payload["task_id"],
        description=payload["description"],
        pattern=Pattern.from_code(payload["pattern"]),
        plan=payload["plan"],
        step_signature=payload["step_signature"],
        outcome=payload.get("outcome", "success"),
        timestamp=payload.get("timestamp", 0.0),
        subtask_memories=[SubtaskMemory(**sm) for sm in payload.get("subtask_memories", [])],
    )


class EpisodicStore:
    def __init__(self, embedder, directory: Path):
        self._embedder = embedder
        self._index = PersistentVectorIndex(directory, "episodic", embedder.dim)

    def __len__(self) -> int:
        return len(self._index)

    # -- Bootstrap: LOAD ("Disk -> FAISS RAM") -------------------------------
    # (handled by PersistentVectorIndex._load_or_create at construction time)

    # -- Retrieval path 1: on-demand similarity search -----------------------

    def search(self, query_text: str, k: int) -> list[tuple[float, FullTaskMemory]]:
        query_vec = self._embedder.encode([query_text])[0]
        return [(score, _from_dict(payload)) for score, payload in self._index.search(query_vec, k)]

    # -- Retrieval path 2: plan-driven batch pre-load ------------------------

    def preload_for_agents(self, agent_names: list[str], query_text: str, k_per_agent: int) -> dict[str, list[SubtaskMemory]]:
        """Batch-load each named agent's past subtask memories in ONE pass over
        the already-loaded FAISS index ('no new FAISS query' beyond this)."""
        query_vec = self._embedder.encode([query_text])[0]
        hits = self._index.search(query_vec, k=max(k_per_agent * len(agent_names), k_per_agent))
        by_agent: dict[str, list[SubtaskMemory]] = {name: [] for name in agent_names}
        for _score, payload in hits:
            mem = _from_dict(payload)
            for sm in mem.subtask_memories:
                if sm.agent in by_agent and len(by_agent[sm.agent]) < k_per_agent:
                    by_agent[sm.agent].append(sm)
        return by_agent

    # -- Consolidation: WRITE (curated distillation, NOT per-task storage) ---

    def add(self, memory: FullTaskMemory) -> None:
        if memory.embedding is None:
            memory.embedding = self._embedder.encode([memory.description])[0]
        self._index.add(np.asarray([memory.embedding], dtype=np.float32), [_to_dict(memory)])

    # -- Consolidation: bulk distillation ------------------------------------

    def add_many(self, memories: list[FullTaskMemory]) -> None:
        vectors, payloads = [], []
        for mem in memories:
            if mem.embedding is None:
                mem.embedding = self._embedder.encode([mem.description])[0]
            vectors.append(mem.embedding)
            payloads.append(_to_dict(mem))
        if vectors:
            self._index.add(np.asarray(vectors, dtype=np.float32), payloads)

    def save(self) -> None:
        self._index.save()
