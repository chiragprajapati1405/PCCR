"""Integration test: run real tasks through the full ten-phase lifecycle and
check that every memory type gets exercised and the router's bookkeeping is
internally consistent. Uses the stub LLM/embedder backends (see
memory_manager.config) so this runs fast and offline -- no API key needed.
"""
from __future__ import annotations

import pytest

from agents import BaseAgent, Orchestrator
from environment import World
from environment.tasks import iter_tasks, seed_entities, seed_events, seed_inbox
from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.types import MemoryType
from run_task import (
    AGENT_CAPABILITY_PROFILES, AGENT_NAMES, AGENT_PROMPTS, ORCHESTRATOR_PROMPT, TASK_TYPE_RULES,
)


@pytest.fixture
def system(tmp_path):
    eml, ics, ans, faiss_dir = (tmp_path / d for d in ("eml", "ics", "answers", "faiss"))
    world = World(eml_dir=eml, ics_dir=ics, answers_dir=ans, seed_inbox=seed_inbox(), seed_events=seed_events())

    settings = config.Settings(embedding_backend="stub", llm_backend="stub")
    manager = MemoryManager(settings)
    # Point FAISS-backed stores at a scratch dir so tests don't touch real data/.
    manager.em._index._dir = faiss_dir
    manager.em._index._index_path = faiss_dir / "episodic.faiss"
    manager.em._index._payload_path = faiss_dir / "episodic.json"
    manager.sm._index._dir = faiss_dir
    manager.sm._index._index_path = faiss_dir / "semantic.faiss"
    manager.sm._index._payload_path = faiss_dir / "semantic.json"
    faiss_dir.mkdir(parents=True, exist_ok=True)

    manager.bootstrap(ORCHESTRATOR_PROMPT, AGENT_PROMPTS, TASK_TYPE_RULES, AGENT_CAPABILITY_PROFILES, seed_entities())
    agents = {name: BaseAgent(name, manager.llm, manager) for name in AGENT_NAMES}
    orchestrator = Orchestrator(manager, world, manager.llm, agents)
    return manager, orchestrator


def test_one_task_runs_end_to_end_and_clears_working_memory(system):
    manager, orchestrator = system
    description, username, date, _hint = next(iter_tasks())

    metrics, trace = orchestrator.run_task(description, username, date)

    assert metrics.success is True
    assert metrics.steps > 0
    assert trace["pattern"] is not None
    with pytest.raises(RuntimeError):
        _ = manager.wm.current   # 'WM: DIES NOW' -- cleared at task end


def test_full_suite_exercises_every_memory_type_at_least_once(system):
    manager, orchestrator = system
    for description, username, date, _hint in iter_tasks():
        orchestrator.run_task(description, username, date)
    manager.consolidate(min_cluster_size=1)
    for description, username, date, _hint in iter_tasks():   # second pass: warms STM/EM/SM
        orchestrator.run_task(description, username, date)

    consulted_anywhere = {s for d in manager.router.decision_log for s in d.consulted_stores}
    assert consulted_anywhere >= {MemoryType.PM, MemoryType.WM, MemoryType.STM, MemoryType.ENT}
    assert len(manager.em) > 0   # populated by consolidation
    assert len(manager.sm) > 0   # populated by consolidation
    assert len(manager.stm) > 0  # populated by on_task_complete


def test_repeat_task_triggers_stm_short_circuit(system):
    """The cascading router's headline cost-saving: a near-identical task seen
    again should hit STM and skip EM/SM/ENT entirely."""
    manager, orchestrator = system
    description, username, date, _hint = next(iter_tasks())

    orchestrator.run_task(description, username, date)
    n_before = len(manager.router.decision_log)
    orchestrator.run_task(description, username, date)
    new_decisions = manager.router.decision_log[n_before:]

    retrieval_decisions = [d for d in new_decisions if d.cache_hit]
    assert retrieval_decisions, "expected at least one STM cache hit on the repeat run"
    for d in retrieval_decisions:
        assert MemoryType.EM not in d.consulted_stores
        assert MemoryType.SM not in d.consulted_stores
        assert MemoryType.ENT not in d.consulted_stores


def test_failure_clears_working_memory_without_writing_stm_or_entity(system):
    manager, orchestrator = system
    ctx = manager.ingest_task("an empty task with no agents available", "chirag", "2026-06-08")
    metrics = manager.complete_task(success=False, plan=[], subtask_memories=[], profile_updates={"x": 1})

    assert metrics.success is False
    storage_decisions = [d for d in manager.router.decision_log if d.phase.value == "memory_storage"]
    assert storage_decisions, "expected a storage routing decision to be logged"
    last = storage_decisions[-1]
    assert MemoryType.STM not in last.consulted_stores
    assert MemoryType.ENT not in last.consulted_stores
    with pytest.raises(RuntimeError):
        _ = manager.wm.current
