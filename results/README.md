# Results & Experiments — master index

All experiments for the PCCR project and their results, in one place.
Four areas (each has its own detail file):

1. [`01_officebench_rho_gate.md`](01_officebench_rho_gate.md) — the ρ-gate memory
   router on OfficeBench (cost at equal accuracy).
2. [`02_assumption_rectification.md`](02_assumption_rectification.md) — grounding
   the priors (measured cost/utility, no-memory & memory-critical baselines, scale).
3. [`03_parallel_architecture.md`](03_parallel_architecture.md) — two-dimensional
   parallelism (latency + proof of accuracy-equivalence).
4. [`04_longmemeval.md`](04_longmemeval.md) — the real LongMemEval benchmark
   (retrieval-routing + real-LLM QA + real parallel latency).

Raw artifacts (logs, calibration JSON, traces) are indexed in
[`raw_artifacts.md`](raw_artifacts.md).

---

## Headline results

| Claim | Evidence | Result |
|---|---|---|
| ρ-gate cuts cost at equal accuracy | OfficeBench 3-agent frontier | same 11/32 acc, **−78% consults, −89% tokens** (θ=1.0→1.2) |
| Memory is provably load-bearing | engineered memory-critical set | no-memory **0/16** → PCCR **16/16** |
| Priors are measured, not asserted | calibration | cost ratio **30×** (was 3.7×); utility **0.875** (counterfactual) |
| Parallelism: latency win, no accuracy cost | 100-task bench + equivalence | **3.73× @ C=4** (→13.5× @ C=16), **60/60 byte-identical** |
| Holds on a real memory benchmark | LongMemEval QA (real gpt-oss) | **same 0.733 acc, −17% consults, ½ SM searches, −5% tokens** |

---

## Full experiment ledger

### A. ρ-gate routing — OfficeBench (gpt-oss-120b)
| ID | Experiment | Result |
|---|---|---|
| A1 | 2-agent held-out (θ=1.0, 11 tasks) | retrieve_all 5/11 (20) · boolean 5/11 (14) · **PCCR 5/11 (12)** → −40% consults |
| A2 | 3-agent held-out (θ=1.0, 32 tasks) | retrieve_all 10/32 (46) · boolean 10/32 (39) · **PCCR 11/32 (27)** → −41% consults |
| A3 | 3-agent θ-frontier | θ=1.0 11/32 (27 consults, 837 tok) → **θ=1.2 11/32 (6, 96 tok)** = −78%/−89%; θ=1.4 → 8/32 |
| A4 | Per-pattern routing (mechanism) | EM pruned on lookups, ENT pruned where irrelevant, both for coordination — as designed |

### B. Assumption rectification
| ID | Experiment | Result |
|---|---|---|
| B1 | Measured store cost (A1) | 3.7× (hand-set) → **30×** (measured; EM 4.543 / ENT 0.15) |
| B2 | Counterfactual utility (A2) | `relation_email` ENT = **0.875** (off 0/8 → on 7/8) |
| B3 | No-memory validity (30 held-out) | no-mem **13/30** vs PCCR/retrieve-all 15/30 |
| B4 | Memory-critical (8 EM + 8 ENT) | no-mem **0/16** · **PCCR 16/16** · retrieve-all 16/16 (PCCR 24 vs 32 consults) |
| B5 | Scale @ equal acc (30, measured cost) | retrieve_all 15/30 (41) · boolean 14/30 (37) · **PCCR 15/30 (7)** → −83% consults |

### C. Parallel architecture
| ID | Experiment | Result |
|---|---|---|
| C1 | Latency, 100 tasks (0.516 s/call) | seq 227.7 s → **par 61.1 s = 3.73×** (C=4, −73%) |
| C2 | Concurrency scaling | C=1 0.96× · C=2 1.89× · C=4 3.73× · C=8 7.15× · **C=16 13.47×** |
| C3 | Accuracy equivalence (60 tasks) | **60/60 byte-identical** + identical memory state |
| C4 | Stress suite (7 tests) | determinism across C, randomized interleavings, contention, fault/lock-leak, consolidation — all pass |

### D. LongMemEval (real benchmark)
| ID | Experiment | Result |
|---|---|---|
| D1 | Retrieval-routing, 90 Q (deterministic) | retrieve_all 0.759 (270) · boolean 0.735 (180) · **PCCR 0.735 (225, SM 45 vs 90)**; parallel==seq True |
| D2 | **Real-LLM QA, 30 Q** (gpt-oss-120b) | retrieve_all 0.733 (181,896 tok, 90) · **PCCR 0.733 (172,558 tok, 75; EM30/SM15/ENT30)** → same acc, −17% consults, −5% tok |
| D3 | Real parallel latency (30 Q) | seq 165.6 s → par 138.6 s = **1.19×** (rate-limited; ceiling 3.73×) |

---

## Honest caveats (apply across the board)
- **LongMemEval grader** is a deterministic lexical proxy for the GPT-4o judge →
  `single-session-preference` 0/5 for *both* methods (subjective golds); equal
  across methods, not a routing effect.
- **Real latency is rate-limit-bound** on 4 Cerebras keys; the controlled ceiling
  is the modeled 3.73× (C1).
- **Cross-model (A8)** is an open evidence gap — only gpt-oss-120b is usable on
  the account.
- **Utility tables are priors**; A2-style counterfactual calibration is the
  principled next step (a mis-set LongMemEval prior was caught: knowledge-update
  4/12 → 11/12 after fixing).

## Reproduce
See each area file's "Reproduce" section. Quick map:
- OfficeBench: `./run_3agent.sh`, `./run_sweep_3agent.sh`, `python analyze_results.py`
- Calibration: `python calibrate.py costs` / `utilities`
- Parallel: `python bench_parallel.py`, `python verify_equivalence.py`, `pytest tests/`
- LongMemEval: `python -m longmemeval.run --n 90`, `python -m longmemeval.run_qa --n 30`
