"""The Phase-Conditioned Cascading Memory Router (PCCR) -- the core contribution.

WHY THIS EXISTS
---------------
Surveying current agent-memory routing work (MemRouter, "Did You Check the
Right Pocket?", FluxMem, RCR-Router -- see project notes) shows each optimizes
ONE edge of the routing problem in isolation: write-side admission, OR
read-side store selection, OR which memory *structure* to use, OR context
budget allocation for one phase of one (usually single-agent, conversational)
loop. None of them route across a full six-store taxonomy (PM/WM/STM/EM/SM/
ENT) spanning a *multi-agent orchestrator/sub-agent* lifecycle, and none make
the routing policy itself a function of *which lifecycle phase* is executing.

This router does three things no single prior system combines:

  1. PHASE CONDITIONING -- routing is not one global policy. Each of the ten
     lifecycle phases gets its own rule set (see PHASE_POLICY), because "what
     should I consult" means something different during retrieval than during
     storage than during consolidation. A request arriving during
     ORCH_INFERENCE is never routed to disk; the same kind of request during
     CONSOLIDATION always is.

  2. CASCADING, COST-EFFECTIVENESS-RATIO GATING -- within the retrieval phase,
     the router first tries the free short-circuit (STM cache hit -> skip
     everything else), and on a miss does not fall back to "query everything"
     (expensive) or "query nothing" (brittle). Instead it computes, per
     candidate store, a ratio rho = expected_utility(pattern, store) /
     relative_cost(store) and consults only stores that clear a threshold.
     This makes the historically ad-hoc "if pattern needs X" rule from the
     lifecycle table into an explicit, tunable, *inspectable* quantity.

  3. DECISION LOGGING AS A FIRST-CLASS OUTPUT -- every routing call returns
     and records a RoutingDecision (candidates considered, stores consulted,
     stores skipped + WHY, cache hit/miss, latency). This is not telemetry
     bolted on after the fact: it is the substrate a follow-on closed-loop
     layer needs to ask "did consulting EM for pattern-C tasks actually
     correlate with success?" and rewrite PATTERN_UTILITY accordingly --
     turning today's hand-set utility table into tomorrow's learned policy
     without changing a single call site. (That closed-loop layer is staged
     as the project's phase 2; this router is built so it slots in cleanly --
     see `MemoryRouter.adjust_utility`.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .types import MemoryType, Pattern, Phase, RoutingDecision

# ---------------------------------------------------------------------------
# 1. Relative cost model
# ---------------------------------------------------------------------------
# Coarse, hand-set weights reflecting real latency/compute differences: WM/PM
# are in-process dict lookups; STM is one cached-embedding comparison; ENT is
# a dict lookup keyed by username; SM/EM both require a FAISS similarity
# search (and EM additionally deserializes whole task traces). These numbers
# are deliberately *not* measured per-call latencies -- they are priors the
# router starts from. A learned/closed-loop layer would replace them with
# measured p50 latencies from RoutingDecision.latency_s logs.

STORE_COST: dict[MemoryType, float] = {
    MemoryType.PM: 0.02,
    MemoryType.WM: 0.02,
    MemoryType.STM: 0.08,
    MemoryType.ENT: 0.15,
    MemoryType.SM: 0.45,
    MemoryType.EM: 0.55,
}

# ---------------------------------------------------------------------------
# 2. Pattern -> expected utility of consulting each *optional* store
# ---------------------------------------------------------------------------
# PM and WM are always read (the lifecycle table is unconditional about this);
# STM is always checked first. What's actually in question on a cache MISS is
# whether EM, SM, and ENT are worth their cost for *this kind* of task. These
# values encode hypotheses about the use case (see types.Pattern docstring):
#   - LOOKUP tasks lean on durable facts (ENT) and general knowledge (SM),
#     rarely need a specific past episode (EM).
#   - SINGLE_ACTION / COORDINATION tasks benefit most from EM ("how did I
#     sequence this kind of multi-step action before").
#   - RECURRING tasks are STM's whole reason to exist; on a rare miss, EM is
#     the most likely fallback to find a near-identical past trace.
#   - EXPLORATORY tasks are exactly where similarity search over general
#     knowledge (SM) and loosely-related episodes (EM) earns its cost; ENT
#     (specific durable facts) is least likely to match an open-ended ask.

PATTERN_UTILITY: dict[Pattern, dict[MemoryType, float]] = {
    Pattern.LOOKUP:        {MemoryType.EM: 0.15, MemoryType.SM: 0.45, MemoryType.ENT: 0.80},
    Pattern.SINGLE_ACTION: {MemoryType.EM: 0.55, MemoryType.SM: 0.25, MemoryType.ENT: 0.55},
    Pattern.COORDINATION:  {MemoryType.EM: 0.70, MemoryType.SM: 0.40, MemoryType.ENT: 0.65},
    Pattern.RECURRING:     {MemoryType.EM: 0.75, MemoryType.SM: 0.15, MemoryType.ENT: 0.30},
    Pattern.EXPLORATORY:   {MemoryType.EM: 0.55, MemoryType.SM: 0.75, MemoryType.ENT: 0.20},
}

ALWAYS_READ = (MemoryType.PM, MemoryType.WM)
OPTIONAL_ON_MISS = (MemoryType.EM, MemoryType.SM, MemoryType.ENT)


@dataclass
class PhasePolicy:
    """Declarative description of one lifecycle phase's routing behavior.

    `mode` documents the *shape* of the decision the router makes in this
    phase (useful for the paper's architecture diagram and for asserting the
    router covers every phase); the actual logic lives in the `route_*`
    methods below, keyed by phase.
    """

    phase: Phase
    mode: str
    note: str


PHASE_POLICY: dict[Phase, PhasePolicy] = {
    Phase.BOOTSTRAP: PhasePolicy(
        Phase.BOOTSTRAP, "classify-by-kind",
        "Static rules -> PM, facts -> ENT, vectors -> SM, infra -> NONE. One-time, unconditional."),
    Phase.INGESTION: PhasePolicy(
        Phase.INGESTION, "fixed-sink",
        "Everything new about this task -> WM. No conditionality: it's all volatile by construction."),
    Phase.RETRIEVAL: PhasePolicy(
        Phase.RETRIEVAL, "cascading-cost-gate",
        "PM+WM always; STM short-circuit; on miss, utility/cost ratio gates EM/SM/ENT. The core contribution."),
    Phase.ORCH_INFERENCE: PhasePolicy(
        Phase.ORCH_INFERENCE, "consume-and-log",
        "Assembled prompt is consumed by the LLM and logged; nothing persists -> MemoryType.NONE."),
    Phase.ORCH_OUTPUT: PhasePolicy(
        Phase.ORCH_OUTPUT, "fixed-sink-plus-trigger",
        "Delegation output -> WM; detecting a not-yet-seen agent triggers a batched EM pre-load."),
    Phase.AGENT_INFERENCE: PhasePolicy(
        Phase.AGENT_INFERENCE, "fixed-read-set",
        "PM (this agent's own rules) + pre-loaded EM + WM instruction. No new FAISS query here."),
    Phase.AGENT_OUTPUT: PhasePolicy(
        Phase.AGENT_OUTPUT, "classify-by-kind",
        "Action JSON -> WM (consumed by executor); created files -> EXT (not memory)."),
    Phase.OBSERVATION: PhasePolicy(
        Phase.OBSERVATION, "fixed-sink",
        "Action results and cross-agent observations -> WM (shared_context)."),
    Phase.STORAGE: PhasePolicy(
        Phase.STORAGE, "outcome-conditional",
        "Success -> write STM x3 + ENT merge; failure -> only clear WM. WM always cleared."),
    Phase.CONSOLIDATION: PhasePolicy(
        Phase.CONSOLIDATION, "batch-distribute",
        "Batch-distribute distilled experience: EM (episodes), SM (facts), PM (rules), STM (prefill)."),
}


@dataclass
class MemoryRouter:
    """Stateful router: holds the (mutable, learnable) policy + a decision log."""

    consult_threshold: float = 1.0
    pattern_utility: dict[Pattern, dict[MemoryType, float]] = field(
        default_factory=lambda: {p: dict(v) for p, v in PATTERN_UTILITY.items()})
    store_cost: dict[MemoryType, float] = field(default_factory=lambda: dict(STORE_COST))
    decision_log: list[RoutingDecision] = field(default_factory=list)

    # -- Phase 1: BOOTSTRAP --------------------------------------------------

    @staticmethod
    def route_bootstrap(item_kind: str) -> MemoryType:
        """'Static rules -> PM / Facts about things -> ENT / Meaning vectors ->
        SM / Infrastructure -> not memory' -- a pure classification, no state."""
        return {
            "prompt": MemoryType.PM,
            "task_type_rule": MemoryType.PM,
            "agent_capability_profile": MemoryType.PM,
            "entity_fact": MemoryType.ENT,
            "embedding_index": MemoryType.SM,
            "infra": MemoryType.NONE,
        }.get(item_kind, MemoryType.NONE)

    # -- Phase 2: INGESTION ---------------------------------------------------

    @staticmethod
    def route_ingestion(_item_kind: str) -> MemoryType:
        """'All data changes between tasks and dies when task ends -> WM'."""
        return MemoryType.WM

    # -- Phase 3: RETRIEVAL (the cascading cost-gate -- core contribution) ---

    def plan_retrieval(self, task_id: str, pattern: Pattern, stm_hit: bool) -> RoutingDecision:
        """Decide which stores to consult for this task's memory bundle.

        Step 1 (cascade head): STM is checked first by the caller; if it was a
        hit, we skip EM/SM/ENT outright -- the cheapest possible path and the
        majority of the cost savings versus "always query everything".

        Step 2 (cost-gated cascade body): on a miss, don't fall back to
        querying every store. For each optional store, compute
            rho = expected_utility(pattern, store) / relative_cost(store)
        and consult it only if rho clears `consult_threshold`. This is the
        piece that makes the routing rule a *ratio* rather than a hard-coded
        boolean per pattern -- and the piece a closed-loop layer can retune
        from outcome data via `adjust_utility` without touching this method.
        """
        start = time.perf_counter()
        candidates = list(ALWAYS_READ) + [MemoryType.STM] + list(OPTIONAL_ON_MISS)
        consulted: list[MemoryType] = [MemoryType.PM, MemoryType.WM, MemoryType.STM]
        skipped: dict[MemoryType, str] = {}

        if stm_hit:
            for store in OPTIONAL_ON_MISS:
                skipped[store] = "stm_cache_hit: cascade short-circuited"
        else:
            utilities = self.pattern_utility.get(pattern, {})
            for store in OPTIONAL_ON_MISS:
                utility = utilities.get(store, 0.0)
                cost = self.store_cost.get(store, 1.0)
                ratio = utility / cost if cost > 0 else float("inf")
                if ratio >= self.consult_threshold:
                    consulted.append(store)
                else:
                    skipped[store] = (
                        f"utility/cost ratio {ratio:.2f} < threshold {self.consult_threshold:.2f} "
                        f"(utility={utility:.2f}, cost={cost:.2f}) for pattern {pattern.value}"
                    )

        decision = RoutingDecision(
            phase=Phase.RETRIEVAL, task_id=task_id, pattern=pattern,
            candidate_stores=candidates, consulted_stores=consulted, skipped_stores=skipped,
            cache_hit=stm_hit, latency_s=time.perf_counter() - start,
        )
        self.decision_log.append(decision)
        return decision

    # -- Phase 5: ORCHESTRATOR OUTPUT (plan-driven EM pre-load trigger) ------

    def should_preload_episodic(self, newly_detected_agents: set[str]) -> bool:
        """'Plan-driven agent detection triggers EPISODIC read'.

        Routing rule: only trigger the (relatively costly, EM.cost=0.55) batch
        pre-load when the orchestrator's plan reveals agents we haven't
        already pre-loaded for in this task -- avoids redundant FAISS hits
        when the orchestrator re-confirms the same delegation across steps.
        """
        return bool(newly_detected_agents)

    # -- Phase 7: AGENT OUTPUT (classify action vs. file artifact) -----------

    @staticmethod
    def route_agent_output(item_kind: str) -> MemoryType:
        """'Action JSON -> WM (consumed by executor) / Files -> EXTERNAL'."""
        if item_kind == "action_json":
            return MemoryType.WM
        if item_kind in ("ics_file", "eml_file", "answer_file"):
            return MemoryType.EXT
        return MemoryType.NONE

    # -- Phase 9: STORAGE (outcome-conditional fan-out) ----------------------

    def plan_storage(self, task_id: str, pattern: Pattern, success: bool) -> RoutingDecision:
        """'Success -> write STM + ENT / Failure -> only clear WM'.

        Encoded as a routing decision (not just an if-statement) so the
        success/failure branch is itself auditable -- e.g. to later notice
        "pattern-D tasks fail at 3x the rate of others, and when they do we
        write nothing to STM, so the cache never warms up for them".
        """
        start = time.perf_counter()
        consulted = [MemoryType.WM]  # always: read flag, then clear
        skipped: dict[MemoryType, str] = {}
        if success:
            consulted += [MemoryType.STM, MemoryType.ENT]
        else:
            skipped[MemoryType.STM] = "task failed: bundle would shortcut future tasks toward a failing plan"
            skipped[MemoryType.ENT] = "task failed: avoid merging unverified facts into the user profile"

        decision = RoutingDecision(
            phase=Phase.STORAGE, task_id=task_id, pattern=pattern,
            candidate_stores=[MemoryType.WM, MemoryType.STM, MemoryType.ENT],
            consulted_stores=consulted, skipped_stores=skipped, cache_hit=False,
            latency_s=time.perf_counter() - start,
        )
        self.decision_log.append(decision)
        return decision

    # -- Closed-loop hook (staged for phase 2 of the project) ----------------

    def adjust_utility(self, pattern: Pattern, store: MemoryType, delta: float, *, lo: float = 0.0, hi: float = 1.0) -> None:
        """Nudge a learned utility weight toward what outcome data suggests.

        Not wired into the main loop yet (that requires the
        `contributed_to_success` feedback pass over `decision_log` plus an
        evaluation harness -- the project's closed-loop phase). Provided now,
        and exercised by tests, so the eventual learning pass has a stable,
        already-reviewed point of entry rather than needing to reach into
        `pattern_utility` directly.
        """
        table = self.pattern_utility.setdefault(pattern, {})
        current = table.get(store, PATTERN_UTILITY.get(pattern, {}).get(store, 0.5))
        table[store] = min(hi, max(lo, current + delta))

    # -- Introspection --------------------------------------------------------

    def explain(self, decision: RoutingDecision) -> str:
        lines = [f"[{decision.phase.value}] task={decision.task_id} pattern={decision.pattern and decision.pattern.value}"]
        lines.append(f"  consulted: {[s.value for s in decision.consulted_stores]}")
        for store, reason in decision.skipped_stores.items():
            lines.append(f"  skipped {store.value}: {reason}")
        if decision.cache_hit:
            lines.append("  (cascade short-circuited on STM cache hit)")
        lines.append(f"  routing latency: {decision.latency_s * 1000:.3f} ms")
        return "\n".join(lines)
