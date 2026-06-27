"""Online utility estimation for the PCCR gate -- the closed loop, wired.

Instead of precomputing per-pattern utility with an expensive counterfactual
leave-one-out sweep (which also degenerates to ~0 on a weak backbone), this
estimates utility *online* from the task stream the gate already produces:

  For each (pattern, store) we keep two recency-weighted success-rate trackers:
    on  = P(task succeeded | this store WAS consulted)
    off = P(task succeeded | this store was SKIPPED)
  and define   U(pattern, store) = clamp(P_on - P_off, 0, 1)   -- the marginal
  value of consulting the store, harvested from the natural on/off variation in
  the decision log (no dedicated A/B runs). epsilon-exploration occasionally
  consults a below-threshold store so P_on keeps getting samples (value of
  information), avoiding the cold-start trap where a store stuck at U=0 is never
  tried again.

This turns the router's hand-set PATTERN_UTILITY into a learned policy with
zero calibration runs; see MemoryRouter.record_outcome / refresh_utilities.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .types import MemoryType, Pattern


@dataclass
class _Rate:
    """Recency-weighted (EWMA) Laplace-smoothed success rate."""
    succ: float = 0.0
    n: float = 0.0

    def update(self, success: bool, decay: float) -> None:
        self.succ = self.succ * decay + (1.0 if success else 0.0)
        self.n = self.n * decay + 1.0

    def rate(self, prior: float, strength: float) -> float:
        return (self.succ + prior * strength) / (self.n + strength)


@dataclass
class OnlineUtility:
    """Tracks marginal utility U=P_on-P_off per (pattern, store) from outcomes."""
    decay: float = 0.98              # per-observation EWMA decay (recency weighting)
    prior: float = 0.5               # Laplace prior success rate (cold start)
    strength: float = 3.0            # prior pseudo-count: how long to trust the prior
    epsilon: float = 0.10            # exploration: prob. of consulting a sub-threshold store
    min_samples: float = 2.0         # don't override the hand-set prior until this many on-samples
    _on: dict = field(default_factory=dict)
    _off: dict = field(default_factory=dict)

    def observe(self, pattern: Pattern, store: MemoryType, consulted: bool, success: bool) -> None:
        key = (pattern, store)
        bank = self._on if consulted else self._off
        bank.setdefault(key, _Rate()).update(success, self.decay)

    def p_on(self, pattern: Pattern, store: MemoryType) -> float:
        return self._on.get((pattern, store), _Rate()).rate(self.prior, self.strength)

    def p_off(self, pattern: Pattern, store: MemoryType) -> float:
        return self._off.get((pattern, store), _Rate()).rate(self.prior, self.strength)

    def samples_on(self, pattern: Pattern, store: MemoryType) -> float:
        return self._on.get((pattern, store), _Rate()).n

    def utility(self, pattern: Pattern, store: MemoryType) -> float | None:
        """Learned marginal utility, or None if we lack enough on-samples to trust it."""
        if self.samples_on(pattern, store) < self.min_samples:
            return None
        return max(0.0, min(1.0, self.p_on(pattern, store) - self.p_off(pattern, store)))

    def explore(self, rng: random.Random | None = None) -> bool:
        return (rng or random).random() < self.epsilon
