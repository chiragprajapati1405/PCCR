# Parallel vs sequential — latency on 100 tasks

**Question:** how much latency difference between the sequential architecture
(`main`/`rectified-architecture`) and the new parallel architecture
(`parallel-architecture`)?

## Method
`bench_parallel.py` drives the **same real framework** (per-task `fork_for_task`
with own WM, Phase-3 ρ-gate, the orchestrator re-planning loop, dependency-wave
agent execution, lock-guarded Phase-9 writes, shared stores) over 100 tasks in
two execution modes:

- **sequential** — tasks one after another; within a task, agents one at a time
  (original sync-lifecycle behaviour).
- **parallel** — tasks concurrent via `AsyncTaskQueue` (max_concurrency=C);
  within a task, independent subtasks in a dependency wave run concurrently.

The only modeled quantity is the external call cost `L` = **measured Cerebras
gpt-oss-120b per-call latency = 0.516 s** (median over real calls; mean 0.743 s).
asyncio overlaps a real network await exactly as it overlaps `asyncio.sleep(L)`,
so the measured wall-clock speedup is what real parallel API calls would give
(the work is I/O-bound).

Task mix (100 tasks): single 20, lookup 20, recurring 10, coordination 30,
exploratory 20. **440 total LLM calls** (identical in both runs);
critical-path depth (orch rounds + waves) = 360 calls.

## Headline result (measured L = 0.516 s, C = 4)

| run | wall-clock (100 tasks) | per-task avg |
|---|---|---|
| **sequential** | **227.7 s** | 2277 ms |
| **parallel (C=4)** | **61.1 s** | 611 ms |
| **speedup** | **3.73× — 73.2% faster (saved 166.6 s)** | |

Sanity check: sequential 227.7 s ≈ analytic `calls×L` = 440 × 0.516 = 227.0 s.

## Scaling with task-concurrency C (ratio is latency-independent)

| C | parallel | speedup | wall-clock saved |
|---|---|---|---|
| 1 | — | 0.96× | −4.6% |
| 2 | — | 1.89× | 47.2% |
| 4 | — | **3.73×** | 73.2% |
| 8 | — | 7.15× | 86.0% |
| 16 | — | 13.47× | 92.6% |

Near-linear up to the critical-path bound. **C=1 is ~neutral**: with no
task-level concurrency, the only gain is intra-task wave overlap (coordination
8→6 calls, etc.), which roughly cancels the per-wave snapshot/merge overhead —
so the win is dominated by **task-level concurrency**, with intra-task wave
overlap a secondary benefit.

## Caveats (honest framing for the paper)
- Parallelism changes **only latency**. Call count and token cost are identical
  in both runs — cutting *consults* is the ρ-gate's job (orthogonal axis).
- Real speedup is capped below C by (a) provider RPM limits (we hit 429s at 4
  keys) and (b) the dependency critical path (a coordination task's send still
  waits for its create). The 0.516 s is pure latency; provider rate limits would
  lower the effective ceiling in a live run.
- The benchmark models agent work as the measured per-call latency rather than
  re-running 440 live calls (cost/rate-limit); the scheduling it measures is the
  real framework's.

Reproduce: `.venv/bin/python bench_parallel.py --tasks 100 --concurrency 4 --latency 0.516`
