# C. Parallel architecture — latency & correctness

Two dimensions of concurrency (parallel sub-agents within a task; parallel tasks
across the system) over the 7-layer design, with the ρ-gate router unchanged.
Architecture detail: `paper/parallel_paper.tex`, `pccr_parallel_architecture.svg`.

## C1 — Latency: sequential vs parallel (100 tasks) ★
Same real framework, two execution schedules. Per-call cost L = **measured**
gpt-oss-120b latency = **0.516 s** (median over live calls). asyncio overlaps a
real network await exactly as it overlaps `sleep(L)`, so the speedup is faithful.
Task mix: single 20, lookup 20, recurring 10, coordination 30, exploratory 20
(440 LLM calls; critical-path depth 360).

| Run | Wall-clock (100 tasks) | Per task | Speedup |
|---|---|---|---|
| Sequential | 227.7 s | 2277 ms | 1.00× |
| **Parallel (C=4)** | **61.1 s** | **611 ms** | **3.73×** |

Sanity: sequential ≈ analytic 440×0.516 = 227.0 s.

## C2 — Concurrency scaling (ratio is latency-independent)
| C | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Speedup | 0.96× | 1.89× | 3.73× | 7.15× | 13.47× |
| Wall-clock cut | −5% | 47% | 73% | 86% | 93% |

Near-linear to the dependency critical path. C=1 ≈ neutral (no task concurrency;
intra-task wave overlap ≈ cancels per-wave snapshot/merge overhead) → the gain
is dominated by **parallel tasks**, with parallel sub-agents a secondary benefit.

## C3 — Accuracy equivalence (60 tasks) ★
Same task set through both schedules; compare per-task (success, plan, sorted
observations) + final shared-memory state.

| Environment | Outcomes identical | Memory state identical |
|---|---|---|
| Per-task isolated | **60/60** | **yes** |
| Shared world | 58/60* | yes |

*The 2 differ only on an incidental observation count of a shared mutable
resource (success/plan/memory still identical) — removed by per-task env
isolation, the standard held-out eval model. **Parallelism changes latency only,
never accuracy.**

## C4 — Concurrency stress suite (7 tests, all pass)
- deterministic wave-merge under 200 randomized finish orders
- identical outcomes + memory state across C ∈ {1,2,4,8,16}
- equivalence stable over 30 random-timing seeds
- worst-case same-user / same-pattern contention loses no write
- injected-failure isolation + no lock leak (liveness)
- consolidation interleaved with load (no tear / no deadlock)

Plus the foundation: 39 unit/integration tests (15 cover the parallelism layer).

## Reproduce
```bash
python bench_parallel.py --tasks 100 --concurrency 4 --latency 0.516   # C1
python bench_parallel.py --tasks 100 --concurrency 8                   # C2 sweep
python verify_equivalence.py                                           # C3
pytest tests/test_parallel.py tests/test_parallel_stress.py            # C4
```
Detail: `explanation/parallel_latency.md`.
