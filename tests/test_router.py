"""Unit tests for the Phase-Conditioned Cascading Memory Router (PCCR).

These exercise the router in isolation -- no LLM, no embeddings, no stores --
because the router's job is purely to *decide* what to consult given a
(phase, pattern, cache-state) triple, and that decision should be testable
without standing up the rest of the lifecycle.
"""
from __future__ import annotations

import pytest

from memory_manager.router import MemoryRouter, OPTIONAL_ON_MISS, PATTERN_UTILITY, STORE_COST
from memory_manager.types import MemoryType, Pattern, Phase


@pytest.fixture
def router() -> MemoryRouter:
    return MemoryRouter(consult_threshold=1.0)


# -- Cascade head: STM hit short-circuits everything optional ----------------

def test_stm_hit_skips_all_optional_stores(router):
    decision = router.plan_retrieval("t1", Pattern.EXPLORATORY, stm_hit=True)

    assert decision.cache_hit is True
    assert MemoryType.PM in decision.consulted_stores
    assert MemoryType.WM in decision.consulted_stores
    assert MemoryType.STM in decision.consulted_stores
    for store in OPTIONAL_ON_MISS:
        assert store not in decision.consulted_stores
        assert "cascade short-circuited" in decision.skipped_stores[store]


def test_stm_hit_short_circuit_holds_for_every_pattern(router):
    """The cascade head's behavior must not depend on the pattern -- a cache
    hit is a cache hit regardless of what kind of task it is."""
    for pattern in Pattern:
        decision = router.plan_retrieval("t", pattern, stm_hit=True)
        assert set(decision.consulted_stores) == {MemoryType.PM, MemoryType.WM, MemoryType.STM}


# -- Cascade body: cost-gated, pattern-conditioned selection on a miss -------

def test_miss_consults_only_stores_clearing_the_utility_cost_ratio(router):
    decision = router.plan_retrieval("t2", Pattern.EXPLORATORY, stm_hit=False)

    assert decision.cache_hit is False
    for store in OPTIONAL_ON_MISS:
        utility = PATTERN_UTILITY[Pattern.EXPLORATORY][store]
        cost = STORE_COST[store]
        expected_consult = (utility / cost) >= router.consult_threshold
        assert (store in decision.consulted_stores) == expected_consult
        if not expected_consult:
            assert "ratio" in decision.skipped_stores[store]


def test_recurring_pattern_skips_semantic_but_lookup_consults_entity(router):
    """Sanity-check two of the use-case-specific routing hypotheses encoded in
    PATTERN_UTILITY: RECURRING tasks shouldn't pay for a semantic-memory
    search (low expected payoff relative to its cost), while LOOKUP tasks
    should consult Entity memory (durable user facts directly answer them)."""
    recurring = router.plan_retrieval("tD", Pattern.RECURRING, stm_hit=False)
    assert MemoryType.SM not in recurring.consulted_stores

    lookup = router.plan_retrieval("tA", Pattern.LOOKUP, stm_hit=False)
    assert MemoryType.ENT in lookup.consulted_stores


def test_threshold_is_a_single_lever_that_shifts_every_pattern(router):
    """Raising the threshold should only ever shrink (never grow) the
    consulted set -- this is what makes it a meaningful cost/quality dial
    rather than an arbitrary knob."""
    loose = MemoryRouter(consult_threshold=0.1)
    strict = MemoryRouter(consult_threshold=5.0)
    for pattern in Pattern:
        loose_set = set(loose.plan_retrieval("x", pattern, stm_hit=False).consulted_stores)
        strict_set = set(strict.plan_retrieval("x", pattern, stm_hit=False).consulted_stores)
        assert strict_set <= loose_set


# -- Storage phase: outcome-conditional fan-out ------------------------------

def test_successful_task_writes_stm_and_entity(router):
    decision = router.plan_storage("t3", Pattern.SINGLE_ACTION, success=True)
    assert MemoryType.STM in decision.consulted_stores
    assert MemoryType.ENT in decision.consulted_stores
    assert not decision.skipped_stores


def test_failed_task_writes_neither_stm_nor_entity(router):
    decision = router.plan_storage("t4", Pattern.SINGLE_ACTION, success=False)
    assert MemoryType.STM not in decision.consulted_stores
    assert MemoryType.ENT not in decision.consulted_stores
    assert MemoryType.WM in decision.consulted_stores
    assert "failed" in decision.skipped_stores[MemoryType.STM]
    assert "failed" in decision.skipped_stores[MemoryType.ENT]


# -- Bootstrap / ingestion / agent-output classification ---------------------

@pytest.mark.parametrize("kind, expected", [
    ("prompt", MemoryType.PM),
    ("task_type_rule", MemoryType.PM),
    ("entity_fact", MemoryType.ENT),
    ("embedding_index", MemoryType.SM),
    ("infra", MemoryType.NONE),
])
def test_route_bootstrap_classification(router, kind, expected):
    assert router.route_bootstrap(kind) is expected


def test_route_ingestion_always_lands_in_working_memory(router):
    for kind in ("description", "username", "pattern", "step_history", "shared_context"):
        assert router.route_ingestion(kind) is MemoryType.WM


@pytest.mark.parametrize("kind, expected", [
    ("action_json", MemoryType.WM),
    ("ics_file", MemoryType.EXT),
    ("eml_file", MemoryType.EXT),
    ("answer_file", MemoryType.EXT),
])
def test_route_agent_output_classification(router, kind, expected):
    assert router.route_agent_output(kind) is expected


# -- Decision log: the substrate the closed-loop layer will need -------------

def test_every_decision_is_logged_for_later_audit(router):
    router.plan_retrieval("a", Pattern.LOOKUP, stm_hit=False)
    router.plan_retrieval("b", Pattern.RECURRING, stm_hit=True)
    router.plan_storage("a", Pattern.LOOKUP, success=True)

    assert len(router.decision_log) == 3
    assert [d.phase for d in router.decision_log] == [Phase.RETRIEVAL, Phase.RETRIEVAL, Phase.STORAGE]
    assert all(d.latency_s >= 0 for d in router.decision_log)


# -- Closed-loop hook (staged for phase 2) -----------------------------------

def test_adjust_utility_nudges_and_clamps(router):
    before = router.pattern_utility[Pattern.LOOKUP][MemoryType.EM]
    router.adjust_utility(Pattern.LOOKUP, MemoryType.EM, +0.1)
    assert router.pattern_utility[Pattern.LOOKUP][MemoryType.EM] == pytest.approx(before + 0.1)

    router.adjust_utility(Pattern.LOOKUP, MemoryType.EM, +10.0)
    assert router.pattern_utility[Pattern.LOOKUP][MemoryType.EM] == 1.0   # clamped to hi

    router.adjust_utility(Pattern.LOOKUP, MemoryType.EM, -10.0)
    assert router.pattern_utility[Pattern.LOOKUP][MemoryType.EM] == 0.0   # clamped to lo


def test_adjust_utility_changes_future_routing_decisions(router):
    """The whole point of logging decisions is that a learned policy can feed
    back into THIS table and immediately change behavior -- prove it does."""
    pattern, store = Pattern.RECURRING, MemoryType.SM
    before = router.plan_retrieval("p", pattern, stm_hit=False)
    assert store not in before.consulted_stores   # low prior utility -> skipped

    router.adjust_utility(pattern, store, +0.5)   # simulate "this turned out to help a lot"
    after = router.plan_retrieval("p", pattern, stm_hit=False)
    assert store in after.consulted_stores        # router now consults it
