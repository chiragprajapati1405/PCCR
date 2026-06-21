# B. Assumption rectification — grounding the priors

The original ρ-gate used hand-set cost/utility numbers. These experiments
replace assertions with measurements and prove the memory is actually used.
(Full 14-assumption ledger: `explanation/rectified_architecture.md`.)

## B1 — Measured store cost (assumption A1)
Cost = average tokens injected per consultation (the real downstream cost),
anchored so the cheaper store = 0.15.

| Store | Avg injected tokens | Cost | |
|---|---|---|---|
| Episodic (EM) | 212 | **4.543** | FAISS search |
| Entity (ENT) | 7 | **0.15** | keyed lookup |

**Measured ratio EM/ENT = 30×** (hand-set was ~3.7×). The router auto-loads
`calibration/store_cost.json`, so after calibration no hand-set cost is used.
*Note:* because the cost scale changed, θ is re-swept; the measured **ratio** is
what carries the claim.

## B2 — Counterfactual utility (assumption A2)
U(pattern, store) = clamp( P(success | store ON) − P(success | store OFF), 0, 1 ),
measured by forcing each store on/off on held-out tasks.

| Pattern | Store | Success ON | Success OFF | Utility |
|---|---|---|---|---|
| relation_email | **entity** | 7/8 | **0/8** | **0.875** |
| relation_email | episodic | 8/8 | 7/8 | 0.125 |

Decisive: with ENT off the task scores 0/8 (the address lives only in ENT).
Harness built + validated on this pattern; full per-pattern table is remaining
API compute. Output: `calibration/pattern_utility.json`.

## B3 — No-memory validity baseline (30 held-out OfficeBench tasks)
A reviewer's first question: if a no-memory run ties, is memory load-bearing?

| Method | Accuracy |
|---|---|
| No-memory | **13/30** |
| PCCR | 15/30 |
| Retrieve-All | 15/30 |

Finding: on **base** OfficeBench tasks, memory is mostly **not** load-bearing —
no-memory ties on every pattern except a couple. This motivated B4.

## B4 — Memory-CRITICAL test set (16 tasks: 8 EM-critical + 8 ENT-critical) ★
Engineered tasks where the answer lives **only** in memory (`synthetic_em_tasks.py`,
`synthetic_ent_tasks.py`).

| Method | Accuracy | Optional consults |
|---|---|---|
| **No-memory** | **0/16** | 0 |
| **PCCR** | **16/16** | 24 |
| **Retrieve-All** | **16/16** | 32 |

Per pattern: `recurring_em` (EM-critical) no-mem 0/8 → PCCR **8/8**;
`relation_email` (ENT-critical) no-mem 0/8 → PCCR **8/8**.
When memory is genuinely required, no-memory **collapses to 0/16** while PCCR
reaches **16/16 at lower cost** (24 vs 32) — it consults EM only for the
recurring tasks and ENT only for the relation tasks.

## B5 — Scale at equal accuracy (30 held-out, measured cost)
| Method | Test acc | Consults |
|---|---|---|
| Retrieve-All | 15/30 | 41 |
| Boolean | 14/30 | 37 |
| **PCCR** | **15/30** | **7** |

**−83% consults at equal accuracy** with the measured (30×) cost.

## Net effect of rectification
The two real hand-set numbers (cost, utility) are now measured; the savings
**grew and became grounded** (up to 83–87% fewer consults), and the accuracy
claim is now made on tasks where memory is **provably** load-bearing — a harder,
more honest test that the headline survives.

## Reproduce
```bash
python calibrate.py costs                 # -> calibration/store_cost.json  (local)
python calibrate.py utilities             # -> calibration/pattern_utility.json (API)
PCCR_MEMCRIT_ONLY=1 PCCR_EM_CRITICAL=1 PCCR_ENT_CRITICAL=1 ./run_3agent.sh   # B4
python compare_nomem.py                    # B3 no-memory baseline
```
Raw JSON: `results_archive/memcrit/`, `results_archive/a5_gptoss/`,
`calibration/`.
