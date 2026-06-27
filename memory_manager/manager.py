"""MemoryManager: wires the six stores + router across the ten lifecycle phases.

Each `phase_*` method corresponds 1:1 to a row of the lifecycle table and
delegates the *decision* of what to touch to `MemoryRouter`, then performs the
reads/writes the decision calls for. Keeping decision (router) and execution
(manager) separate is what lets the router be swapped for a learned policy
later without rewriting the lifecycle plumbing.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from . import config
from .embeddings import build_embedder
from .llm import build_llm
from .router import MemoryRouter
from .stores import (
    EntityStore, EpisodicStore, ProceduralStore, SemanticStore, ShortTermStore, WorkingMemoryStore,
)
from .types import (
    FullTaskMemory, MemoryBundle, MemoryType, Pattern, Phase, RoutingDecision,
    SubtaskMemory, TaskContext,
)

CLASSIFIER_SYSTEM = (
    "ROLE: classifier\n"
    "Classify the task into one pattern code and respond as JSON {\"pattern\": \"<A|B|C|D|E>\"}.\n"
    "A=lookup (read-only question), B=single action by one agent, "
    "C=multi-agent coordination, D=recurring/templated, E=exploratory/ambiguous."
)


@dataclass
class RetrievalBundle:
    """Everything Memory Retrieval (route_query) assembles for the orchestrator.

    Mirrors the table's "Type of Data" column for that phase exactly: orch
    prompt (PM), task context (WM), cached bundle (STM), past experiences
    (EM), meaning vectors / facts (SM), user profile (ENT).
    """

    orchestrator_prompt: str
    task_context: TaskContext
    cached_bundle: Optional[MemoryBundle] = None
    episodes: list[tuple[float, FullTaskMemory]] = field(default_factory=list)
    facts: list[tuple[float, str]] = field(default_factory=list)
    profile: Optional[dict] = None
    decision: Optional[RoutingDecision] = None


@dataclass
class TaskMetrics:
    task_id: str
    success: bool
    steps: int
    orchestrator_calls: int
    routing_decisions: int
    consulted_total: int
    skipped_total: int
    duration_s: float


class MemoryManager:
    def __init__(self, settings: config.Settings = config.SETTINGS):
        self.settings = settings
        self.embedder = build_embedder(settings.embedding_backend)
        self.llm = build_llm(settings.llm_backend, model=settings.llm_model) \
            if settings.llm_backend != "stub" else build_llm("stub")
        self.classifier_llm = build_llm("stub")  # cheap, deterministic pattern classification

        self.pm = ProceduralStore()
        self.wm = WorkingMemoryStore()
        self.stm = ShortTermStore(self.embedder, settings.stm_cache_threshold,
                                  capacity=(getattr(settings, "stm_capacity", 0) or None))
        self.em = EpisodicStore(self.embedder, config.FAISS_DIR)
        self.sm = SemanticStore(self.embedder, config.FAISS_DIR)
        self.ent = EntityStore()

        self.router = MemoryRouter(consult_threshold=1.0)

        self._preloaded_episodic: dict[str, list[SubtaskMemory]] = {}
        self._task_log: list[FullTaskMemory] = []   # raw completed-task traces, fed to consolidation
        self._metrics_log: list[TaskMetrics] = []
        self._task_start_time: float = 0.0
        self._orchestrator_calls: int = 0

        # Parallelism (see memory_manager/parallel.py). The sync lifecycle is
        # unchanged; these power the OPTIONAL async path (parallel agents +
        # parallel tasks). The rho-gate router is untouched.
        from .parallel import MemoryLockManager
        self.lock_mgr = MemoryLockManager()
        self._consolidation_active = False   # Phase-10 exclusivity flag (Layer 7)

    def is_consolidating(self) -> bool:
        """Task queue checks this before entering Phase 3 (FAISS read) so no
        task reads vectors while Phase 10 is rewriting them."""
        return self._consolidation_active

    def fork_for_task(self) -> "MemoryManager":
        """Per-task manager for PARALLEL TASKS: shares the cross-task stores
        (PM/STM/EM/SM/ENT), the router, and the lock manager, but gets its OWN
        Working Memory (per-task isolation). Bootstrap (Phase 1) is NOT redone.
        Lock-guarded writes through the shared stores keep concurrent tasks
        consistent (Layer 5/6)."""
        m = MemoryManager.__new__(MemoryManager)
        m.settings = self.settings
        m.embedder, m.llm, m.classifier_llm = self.embedder, self.llm, self.classifier_llm
        m.pm, m.stm, m.em, m.sm, m.ent = self.pm, self.stm, self.em, self.sm, self.ent  # SHARED
        m.router, m.lock_mgr = self.router, self.lock_mgr                                # SHARED
        m.wm = type(self.wm)()                       # OWN, isolated
        m._preloaded_episodic = {}
        m._task_log, m._metrics_log = self._task_log, self._metrics_log   # shared logs
        m._task_start_time, m._orchestrator_calls = 0.0, 0
        m._consolidation_active = False
        return m

    # =====================================================================
    # Phase 1 -- System Bootstrap
    # =====================================================================

    def bootstrap(
        self,
        orchestrator_prompt: str,
        agent_prompts: dict[str, str],
        task_type_rules: dict[str, str],
        agent_capability_profiles: dict[str, dict],
        entity_seed: dict[str, dict] | None = None,
    ) -> None:
        """'WRITE (init)' + 'LOAD' -- runs once, never again.

        Routes each developer-supplied item through `router.route_bootstrap`
        purely to keep bootstrap honest with the lifecycle table's
        classification rule (static rules -> PM, facts -> ENT, ...); the
        actual writes go straight to the target store since there is no
        "decision" to make at this phase, only classification.
        """
        assert self.router.route_bootstrap("prompt") is MemoryType.PM
        assert self.router.route_bootstrap("entity_fact") is MemoryType.ENT
        assert self.router.route_bootstrap("embedding_index") is MemoryType.SM

        self.pm.bootstrap(orchestrator_prompt, agent_prompts, task_type_rules, agent_capability_profiles)
        for username, facts in (entity_seed or {}).items():
            self.ent.seed(username, facts)
        # FAISS indexes + embedding model are loaded from disk inside
        # build_embedder()/EpisodicStore()/SemanticStore() constructors above
        # ("Disk -> FAISS RAM", "Disk -> Model RAM" -- once at startup).

    # =====================================================================
    # Phase 2 -- Task Ingestion
    # =====================================================================

    def ingest_task(self, description: str, username: str, date: str) -> TaskContext:
        """'WRITE for ALL items -> WM, once per task'. Also classifies the
        task pattern, which the router then conditions all later decisions on."""
        self._task_start_time = time.perf_counter()
        self._orchestrator_calls = 0
        self._preloaded_episodic = {}

        task_id = uuid.uuid4().hex[:12]
        for kind in ("description", "username", "pattern", "step_history", "shared_context"):
            assert self.router.route_ingestion(kind) is MemoryType.WM

        ctx = self.wm.start_task(task_id, description, username, date)
        pattern = self._classify_pattern(description)
        self.wm.set_pattern(pattern)
        return ctx

    def _classify_pattern(self, description: str) -> Pattern:
        raw = self.classifier_llm.complete(CLASSIFIER_SYSTEM, description)
        try:
            code = json.loads(raw)["pattern"]
        except Exception:
            code = "B"
        return Pattern.from_code(code)

    # =====================================================================
    # Phase 3 -- Memory Retrieval (route_query) -- the cascading router
    # =====================================================================

    def retrieve(self) -> RetrievalBundle:
        """The phase the router is named for. Cascade:

          1. Always grab PM (orchestrator prompt) + WM (task context) -- table
             says these are read on every single call, unconditionally.
          2. Check STM first. Hit -> short-circuit (skip EM, SM, ENT).
          3. Miss -> ask the router which of EM/SM/ENT clear the
             utility/cost threshold for this task's pattern, and read only
             those.
        """
        ctx = self.wm.current
        pattern = ctx.pattern or Pattern.SINGLE_ACTION

        cache_hit = self.stm.lookup(ctx.description, pattern=pattern)   # C3 pattern guard
        stm_hit = cache_hit is not None
        k = self.settings.max_episodes_per_query

        # A1: per-query retrieval-confidence gate. Pre-search each optional store
        # (cheap FAISS, ms) so the router can gate on this query's top-k similarity;
        # the EXPENSIVE part (injecting tokens into the prompt) still happens only for
        # consulted stores. The pre-fetched hits are reused below (no double search).
        confidence = None
        pre_em = pre_sm = None
        pre_profile = None
        if getattr(self.settings, "confidence_gate", False) and not stm_hit:
            pre_em = self.em.search(ctx.description, k=k)
            pre_sm = self.sm.search(ctx.description, k=k)
            ent_conf = 0.0
            prof = self.ent.get(ctx.username)
            if prof and prof.facts:
                ptxt = ", ".join(f"{a}={b}" for a, b in prof.facts.items())
                qv = self.embedder.encode([ctx.description])[0]
                pv = self.embedder.encode([ptxt])[0]
                ent_conf = float(sum(x * y for x, y in zip(qv, pv)))   # cosine (normalized vectors)
                pre_profile = dict(prof.facts)
            confidence = {
                MemoryType.EM: float(pre_em[0][0]) if pre_em else 0.0,
                MemoryType.SM: float(pre_sm[0][0]) if pre_sm else 0.0,
                MemoryType.ENT: ent_conf,
            }

        decision = self.router.plan_retrieval(ctx.task_id, pattern, stm_hit=stm_hit,
                                              confidence=confidence)

        bundle = RetrievalBundle(
            orchestrator_prompt=self.pm.get_orchestrator_prompt(),
            task_context=ctx,
            decision=decision,
        )
        if cache_hit is not None:
            bundle.cached_bundle = cache_hit[0]
        if MemoryType.EM in decision.consulted_stores:
            bundle.episodes = pre_em if pre_em is not None else self.em.search(ctx.description, k=k)
        if MemoryType.SM in decision.consulted_stores:
            bundle.facts = pre_sm if pre_sm is not None else self.sm.search(ctx.description, k=k)
        if MemoryType.ENT in decision.consulted_stores:
            if pre_profile is not None:
                bundle.profile = pre_profile
            else:
                profile = self.ent.get(ctx.username)
                bundle.profile = dict(profile.facts) if profile else None
        return bundle

    # =====================================================================
    # Phase 4 -- Orchestrator Inference (consume + log; nothing persists)
    # =====================================================================

    def run_orchestrator(self, bundle: RetrievalBundle, available_agents: list[str]) -> dict:
        """'NONE (consumed by LLM, not stored)' + 'CONSUME / LOG'.

        Assembles the system+user prompt from PM+WM+STM+EM+ENT, calls the
        LLM, and logs latency -- nothing here is written to any memory store;
        the table is explicit that this data "dies after this single LLM
        call" and is "not stored anywhere".
        """
        ctx = bundle.task_context
        system = (
            "ROLE: orchestrator\n"
            f"{bundle.orchestrator_prompt}\n"
            f"AVAILABLE_AGENTS: {available_agents}\n"
            f"TASK_PATTERN_RULE: {self.pm.get_rule_for_pattern(ctx.pattern.value if ctx.pattern else '')}\n"
        )
        user = self._assemble_user_prompt(bundle)

        t0 = time.perf_counter()
        raw = self.llm.complete(system, user)
        latency = time.perf_counter() - t0
        self._orchestrator_calls += 1

        try:
            output = json.loads(raw)
        except json.JSONDecodeError:
            output = {"thought": raw, "agent": "FINISH", "subtask": "parse_error", "plan_driven": False, "loop_detected": False}
        output["_latency_s"] = latency
        return output

    def _assemble_user_prompt(self, bundle: RetrievalBundle) -> str:
        ctx = bundle.task_context
        parts = [
            f"USERNAME: {ctx.username}",
            f"DATE: {ctx.date}",
            f"TASK: {ctx.description}",
        ]
        if bundle.cached_bundle:
            parts.append(f"CACHED_PLAN (from STM, similar past task): {bundle.cached_bundle.plan}")
        if bundle.episodes:
            parts.append("RELEVANT_PAST_EPISODES (from EM):")
            for score, mem in bundle.episodes:
                parts.append(f"  - (sim={score:.2f}) '{mem.description}' -> plan={mem.plan} outcome={mem.outcome}")
        if bundle.facts:
            parts.append("RELEVANT_FACTS (from SM): " + "; ".join(f for _s, f in bundle.facts))
        if bundle.profile:
            parts.append(f"USER_PROFILE (from ENT): {bundle.profile}")
        history = self.wm.history_text()
        if history:
            parts.append("STEP_HISTORY:\n" + history)
        if ctx.shared_context:
            parts.append(f"SHARED_CONTEXT (cross-agent): {ctx.shared_context}")
        return "\n".join(parts)

    # =====================================================================
    # Phase 5 -- Orchestrator Output (delegation)
    # =====================================================================

    def record_delegation(self, output: dict) -> dict[str, list[SubtaskMemory]]:
        """'WM for ALL items' + loop detection (READ+COMPARE) + plan-driven
        EM pre-load trigger.

        Returns the (possibly newly expanded) pre-loaded-episodic-memory dict;
        Agent Inference reads from this dict rather than issuing its own FAISS
        query ("EM: pre-loaded ... no new FAISS query").
        """
        ctx = self.wm.current
        agent = output.get("agent", "FINISH")
        subtask = output.get("subtask", "")
        ctx.pending_delegation = {"thought": output.get("thought", ""), "agent": agent, "subtask": subtask}

        loop_detected = self._detect_loop(agent, subtask)
        ctx.loop_detected = loop_detected
        if loop_detected:
            ctx.forced_finish = {"reason": "repeated_agent_subtask", "agent": agent, "subtask": subtask}

        if output.get("plan_driven") and agent != "FINISH":
            newly_detected = {agent} - ctx.detected_agents
            if self.router.should_preload_episodic(newly_detected):
                ctx.detected_agents |= newly_detected
                fresh = self.em.preload_for_agents(
                    list(newly_detected), ctx.description, k_per_agent=self.settings.max_episodes_per_query)
                self._preloaded_episodic.update(fresh)
        return self._preloaded_episodic

    def _detect_loop(self, agent: str, subtask: str, *, repeat_threshold: int = 2) -> bool:
        history = self.wm.current.step_history
        if len(history) < repeat_threshold:
            return False
        recent = history[-repeat_threshold:]
        return all(rec.agent == agent and rec.action.startswith(subtask[:30]) for rec in recent)

    # =====================================================================
    # Phase 6 -- Agent Inference (mostly executed inside agents/*; MM supplies
    # the routed context: PM prompt, pre-loaded EM, WM instruction)
    # =====================================================================

    def agent_context(self, agent_name: str) -> tuple[str, list[SubtaskMemory], dict]:
        """'PM: READ / EM: READ (pre-loaded) / WM: READ' -- assembled once per
        agent call; see agents/base_agent.py for how it's consumed."""
        prompt = self.pm.get_agent_prompt(agent_name)
        memories = self._preloaded_episodic.get(agent_name, [])
        delegation = self.wm.current.pending_delegation or {}
        return prompt, memories, delegation

    # =====================================================================
    # Phase 7 -- Agent Output (action)
    # =====================================================================

    def record_agent_action(self, action: dict) -> None:
        """'WM: WRITE (action JSON)'. File artifacts are routed to EXT by the
        environment layer itself (see environment/apps/*) -- the manager only
        ever sees the action JSON, matching 'Action JSON -> WM (consumed by
        executor) / Files -> EXTERNAL'."""
        assert self.router.route_agent_output("action_json") is MemoryType.WM
        self.wm.set_last_action(action)

    # =====================================================================
    # Phase 8 -- Environment Observation
    # =====================================================================

    def record_observation(self, agent: str, action_summary: str, observation: str, structured: dict | None = None) -> None:
        """'WM (Working) for ALL items' + cross-agent shared_context fan-out."""
        self.wm.record_step(agent=agent, action=action_summary, observation=observation)
        self.wm.share_context(agent, structured if structured is not None else {"observation": observation})

    # =====================================================================
    # Phase 9 -- Memory Storage (on_task_complete)
    # =====================================================================

    def complete_task(self, success: bool, plan: list[str], subtask_memories: list[SubtaskMemory],
                      profile_updates: Optional[dict] = None) -> TaskMetrics:
        """'WM: READ (flag) / STM: WRITE x3 (success only) / ENT: WRITE
        (success + new info only) / WM: CLEAR (always) / Metrics: LOG'."""
        ctx = self.wm.current
        self.wm.mark_outcome(success)
        pattern = ctx.pattern or Pattern.SINGLE_ACTION
        signature = self._step_signature(plan)
        embedding = self.embedder.encode([ctx.description])[0]

        decision = self.router.plan_storage(ctx.task_id, pattern, success)

        if MemoryType.STM in decision.consulted_stores:
            self.stm.put(signature, plan, subtask_memories, embedding, pattern)
        if MemoryType.ENT in decision.consulted_stores and profile_updates:
            self.ent.merge(ctx.username, profile_updates)

        # Raw trace always logged for the consolidation curator to draw from
        # later (this is NOT a memory-store write -- see episodic.py docstring
        # on why EM is populated only at Consolidation, not here).
        self._task_log.append(FullTaskMemory(
            task_id=ctx.task_id, description=ctx.description, pattern=pattern, plan=list(plan),
            step_signature=signature, subtask_memories=list(subtask_memories), embedding=embedding,
            outcome="success" if success else "failure",
        ))

        metrics = TaskMetrics(
            task_id=ctx.task_id, success=success, steps=len(ctx.step_history),
            orchestrator_calls=self._orchestrator_calls,
            routing_decisions=len(self.router.decision_log),
            consulted_total=sum(len(d.consulted_stores) for d in self.router.decision_log),
            skipped_total=sum(len(d.skipped_stores) for d in self.router.decision_log),
            duration_s=time.perf_counter() - self._task_start_time,
        )
        self._metrics_log.append(metrics)  # 'Metrics -> not stored' in memory; logged to our own JSON log

        self.wm.clear()  # 'WM: DIES NOW'
        return metrics

    @staticmethod
    def _step_signature(plan: list[str]) -> str:
        return " -> ".join(plan) if plan else "noop"

    # =====================================================================
    # Phase 6-8 (PARALLEL sub-agents) + Phase 9 (parallel lock-guarded writes)
    # Optional async path; the sync methods above are unchanged.
    # =====================================================================

    async def run_agent_wave(self, delegations, agent_runner):
        """One parallel wave (Phase 6-8): snapshot WM (copy-on-read), run the
        wave's agents concurrently against the frozen snapshot, then merge the
        staged results into live WM in deterministic order. `agent_runner` is
        async (agent_type, subtask, wm_snapshot) -> parallel.AgentResult."""
        from .parallel import ParallelExecutor
        executor = ParallelExecutor(agent_runner)
        snap = self.wm.snapshot()                       # frozen WM for the whole wave
        results = await executor.execute_group(delegations, snap)
        self.wm.merge_staging([(r.agent_type, r.subtask, r.observation) for r in results])
        return results

    async def run_parallel_agents(self, delegations, agent_runner):
        """Phase 6-8 for a full plan: build the dependency DAG, then run each
        wave in order — independent subtasks in a wave run concurrently, and a
        later wave only starts after the previous wave's results are merged."""
        from .parallel import DependencyAnalyzer
        waves = DependencyAnalyzer().analyze(list(delegations))
        results = []
        for wave in waves:
            results += await self.run_agent_wave(wave, agent_runner)
        return waves, results

    async def run_planned_task(self, planner, agent_runner, *, max_rounds: int = 8):
        """Iterative Phase 4-8 loop — the diagram's 'More groups? -> back to
        orchestrator'. Each round:

          1. the orchestrator (`planner`) inspects LIVE WM and returns the next
             group of delegations (or an empty list / None to FINISH);
          2. that group is split into a dependency DAG and its waves run
             concurrently (run_parallel_agents);
          3. results are merged into WM (Phase 8);
          4. control returns to the orchestrator for the next round, which now
             sees the merged results of every prior group.

        `planner` is async: (wm_task_context) -> list[Delegation] | None. This
        keeps the orchestrator in the loop (re-planning between groups) rather
        than decomposing the whole plan up front."""
        all_waves, all_results = [], []
        for _round in range(max_rounds):
            group = await planner(self.wm.current)        # Phase 4-5: orchestrator
            if not group:
                break                                     # orchestrator said FINISH
            waves, results = await self.run_parallel_agents(group, agent_runner)
            all_waves += waves
            all_results += results
        return all_waves, all_results

    async def complete_task_async(self, success: bool, plan: list[str],
                                  subtask_memories: list[SubtaskMemory],
                                  profile_updates: Optional[dict] = None) -> TaskMetrics:
        """Async Phase 9: the success-only STM/ENT writes run concurrently via
        asyncio.gather, each acquiring its own per-resource lock (no lost
        updates across parallel tasks). Same semantics as complete_task."""
        ctx = self.wm.current
        self.wm.mark_outcome(success)
        pattern = ctx.pattern or Pattern.SINGLE_ACTION
        signature = self._step_signature(plan)
        embedding = self.embedder.encode([ctx.description])[0]
        username = ctx.username
        # W6: routing-history update. The router (and its decision_log) is SHARED
        # across parallel tasks via fork_for_task, so the append is serialized
        # under history_lock to keep routing history atomic (diagram Phase-9 W6).
        async with self.lock_mgr.history_lock:
            decision = self.router.plan_storage(ctx.task_id, pattern, success)

        writes = []
        if MemoryType.STM in decision.consulted_stores:
            writes.append(self._write_stm_async(pattern, signature, plan, subtask_memories, embedding))
        if MemoryType.ENT in decision.consulted_stores and profile_updates:
            writes.append(self._write_entity_async(username, profile_updates))
        if writes:
            await asyncio.gather(*writes)               # Layer 6: parallel writes

        self._task_log.append(FullTaskMemory(
            task_id=ctx.task_id, description=ctx.description, pattern=pattern, plan=list(plan),
            step_signature=signature, subtask_memories=list(subtask_memories), embedding=embedding,
            outcome="success" if success else "failure",
        ))
        metrics = TaskMetrics(
            task_id=ctx.task_id, success=success, steps=len(ctx.step_history),
            orchestrator_calls=self._orchestrator_calls,
            routing_decisions=len(self.router.decision_log),
            consulted_total=sum(len(d.consulted_stores) for d in self.router.decision_log),
            skipped_total=sum(len(d.skipped_stores) for d in self.router.decision_log),
            duration_s=time.perf_counter() - self._task_start_time,
        )
        self._metrics_log.append(metrics)
        self.wm.clear()
        return metrics

    async def _write_stm_async(self, pattern, signature, plan, subtask_memories, embedding):
        async with self.lock_mgr.stm(pattern.value):    # per-pattern lock
            self.stm.put(signature, plan, subtask_memories, embedding, pattern)

    async def _write_entity_async(self, username, profile_updates):
        async with self.lock_mgr.ent(username):         # per-user lock
            self.ent.merge(username, profile_updates)

    # =====================================================================
    # Phase 10 -- Memory Consolidation (after training; batch, not per-task)
    # =====================================================================

    def consolidate(self, *, min_cluster_size: int = 2, similarity_threshold: float = 0.5) -> dict[str, int]:
        """'WRITE for ALL items: EM distill, SM embed, PM consolidate, STM
        pre-fill. Once after ALL training completes (batch operation).'

        Plays the role of the table's "LLM curator": clusters successful raw
        traces by embedding similarity, writes the curated set to EM, distills
        clusters of >= min_cluster_size into a learned PM rule, extracts short
        factual statements into SM, and pre-fills STM with the most-repeated
        successful bundle per cluster. (A real system would use an LLM to do
        the clustering/distillation/fact-extraction judgment calls; this
        rule-based stand-in keeps the *routing* fully exercised and testable
        without requiring model calls -- swap the clustering/extraction
        functions below for LLM calls to scale this up for real experiments.)
        """
        successful = [m for m in self._task_log if m.outcome == "success"]
        if not successful:
            return {"em_added": 0, "sm_facts": 0, "pm_rules": 0, "stm_prefilled": 0}

        clusters = self._cluster_by_similarity(successful, similarity_threshold)

        em_added = 0
        for mem in successful:
            self.em.add(mem)
            em_added += 1

        sm_facts = 0
        facts = self._extract_facts(successful)
        if facts:
            self.sm.add_facts(facts)
            sm_facts = len(facts)

        pm_rules = 0
        stm_prefilled = 0
        for cluster in clusters:
            if len(cluster) >= min_cluster_size:
                rule = self._distill_rule(cluster)
                self.pm.consolidate_rule(rule)
                pm_rules += 1
                exemplar = max(cluster, key=lambda m: len(m.subtask_memories))
                self.stm.prefill(exemplar.step_signature, exemplar.plan, exemplar.subtask_memories,
                                 exemplar.embedding, exemplar.pattern)
                stm_prefilled += 1

        self.em.save()
        self.sm.save()
        return {"em_added": em_added, "sm_facts": sm_facts, "pm_rules": pm_rules, "stm_prefilled": stm_prefilled}

    async def consolidate_async(self, **kwargs) -> dict[str, int]:
        """Phase 10 as an EXCLUSIVE window (Layer 7): raises the consolidation
        flag (so AsyncTaskQueue stops admitting new tasks into Phase-3 FAISS
        reads) and holds the global faiss_lock while it rewrites EM/SM/PM/STM.
        Reuses the synchronous consolidate() body."""
        self._consolidation_active = True
        try:
            async with self.lock_mgr.faiss_lock:
                return self.consolidate(**kwargs)
        finally:
            self._consolidation_active = False

    @staticmethod
    def _cluster_by_similarity(memories: list[FullTaskMemory], threshold: float) -> list[list[FullTaskMemory]]:
        """Greedy single-link clustering over normalized embeddings (stand-in
        for an LLM curator's "these look like the same kind of task" judgment)."""
        clusters: list[list[FullTaskMemory]] = []
        cluster_vecs: list[np.ndarray] = []
        for mem in memories:
            vec = np.asarray(mem.embedding, dtype=np.float32)
            placed = False
            for cluster, centroid in zip(clusters, cluster_vecs):
                if float(centroid @ vec) >= threshold:
                    cluster.append(mem)
                    placed = True
                    break
            if not placed:
                clusters.append([mem])
                cluster_vecs.append(vec)
        return clusters

    @staticmethod
    def _distill_rule(cluster: list[FullTaskMemory]) -> dict:
        signatures = [m.step_signature for m in cluster]
        common_signature = max(set(signatures), key=signatures.count)
        return {
            "pattern": cluster[0].pattern.value,
            "rule": f"For tasks like '{cluster[0].description[:60]}...', prefer plan: {common_signature}",
            "support": len(cluster),
        }

    @staticmethod
    def _extract_facts(memories: list[FullTaskMemory]) -> list[str]:
        facts = []
        for mem in memories:
            for sm in mem.subtask_memories:
                if sm.outcome == "success" and sm.observation_summary:
                    facts.append(f"When {sm.agent} does '{sm.action}', a typical result is: {sm.observation_summary}")
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for fact in facts:
            if fact not in seen:
                seen.add(fact)
                unique.append(fact)
        return unique

    # =====================================================================
    # Introspection / reporting
    # =====================================================================

    def metrics_summary(self) -> list[dict]:
        return [vars(m) for m in self._metrics_log]

    def routing_trace(self) -> list[str]:
        return [self.router.explain(d) for d in self.router.decision_log]
