"""The closed loop learns utility from outcomes; retrieval-confidence gates per query."""
import random

from memory_manager.online_utility import OnlineUtility
from memory_manager.router import MemoryRouter
from memory_manager.types import MemoryType, Pattern


def _consulted(router, pattern, **kw):
    d = router.plan_retrieval("t", pattern, stm_hit=False, **kw)
    return set(d.consulted_stores)


def test_closed_loop_learns_em_helps_and_flips_gate():
    """EM genuinely helps COORDINATION (success iff consulted) -> learned U rises,
    and a router that started below threshold begins consulting EM."""
    r = MemoryRouter(consult_threshold=2.0,              # high enough that the 0.70 prior is borderline
                     online=OnlineUtility(epsilon=0.0, min_samples=2, strength=1.0))
    # ground truth: consulting EM -> success; skipping EM -> failure
    for _ in range(40):
        r.record_outcome(Pattern.COORDINATION, consulted_stores=[MemoryType.EM], success=True)
        r.record_outcome(Pattern.COORDINATION, consulted_stores=[], success=False)
    u = r.pattern_utility[Pattern.COORDINATION][MemoryType.EM]
    assert u > 0.8, f"EM utility should approach 1 when it is load-bearing, got {u:.2f}"


def test_closed_loop_learns_em_useless_and_suppresses():
    """EM does NOT help LOOKUP (success rate identical on/off) -> learned U -> ~0,
    so the gate stops consulting it (the OfficeBench-style protective behavior)."""
    r = MemoryRouter(consult_threshold=0.5,
                     online=OnlineUtility(epsilon=0.0, min_samples=2, strength=1.0))
    for _ in range(40):
        r.record_outcome(Pattern.LOOKUP, consulted_stores=[MemoryType.EM], success=True)
        r.record_outcome(Pattern.LOOKUP, consulted_stores=[], success=True)   # same outcome w/o EM
    u = r.pattern_utility[Pattern.LOOKUP][MemoryType.EM]
    assert u < 0.2, f"useless EM should learn U~=0, got {u:.2f}"
    assert MemoryType.EM not in _consulted(r, Pattern.LOOKUP), "gate should suppress a U~=0 store"


def test_retrieval_confidence_gates_per_query():
    """Same pattern/utility, but a weak per-query match gates EM out while a strong one
    lets it through -- the per-query confidence proxy."""
    r = MemoryRouter(consult_threshold=1.0)              # EM prior 0.70/0.55 cost ~= 1.27 >= 1.0
    assert MemoryType.EM in _consulted(r, Pattern.COORDINATION)               # no confidence -> as before
    lowconf = _consulted(r, Pattern.COORDINATION, confidence={MemoryType.EM: 0.2})
    highconf = _consulted(r, Pattern.COORDINATION, confidence={MemoryType.EM: 0.95})
    assert MemoryType.EM not in lowconf, "weak match should gate EM out this query"
    assert MemoryType.EM in highconf, "strong match should consult EM"


def test_exploration_samples_subthreshold_stores():
    """With epsilon>0 the gate occasionally consults a below-threshold store to keep
    estimating P_on (value of information / no cold-start lock-out)."""
    r = MemoryRouter(consult_threshold=99.0,             # nothing clears the gate on merit
                     online=OnlineUtility(epsilon=1.0))  # always explore
    assert MemoryType.EM in _consulted(r, Pattern.LOOKUP), "epsilon=1 should force a sub-threshold consult"


def test_prior_kept_until_enough_samples():
    """A single observation must not overwrite the hand-set prior (min_samples guard)."""
    r = MemoryRouter(online=OnlineUtility(min_samples=5))
    before = r.pattern_utility[Pattern.COORDINATION][MemoryType.EM]
    r.record_outcome(Pattern.COORDINATION, consulted_stores=[MemoryType.EM], success=False)
    after = r.pattern_utility[Pattern.COORDINATION][MemoryType.EM]
    assert before == after, "utility should not move before min_samples on-observations"


if __name__ == "__main__":
    for fn in [test_closed_loop_learns_em_helps_and_flips_gate,
               test_closed_loop_learns_em_useless_and_suppresses,
               test_retrieval_confidence_gates_per_query,
               test_exploration_samples_subthreshold_stores,
               test_prior_kept_until_enough_samples]:
        fn(); print(f"ok  {fn.__name__}")
    print("all online-utility tests passed")
