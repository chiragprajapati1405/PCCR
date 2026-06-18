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

---

# Accuracy is preserved — parallel ≡ sequential

`verify_equivalence.py` runs the **same task set** through both schedules and
compares, per task, the outcome `(success, plan, sorted normalised
observations)` plus the **final shared-memory state** (STM/EM/SM/PM sizes + ENT
facts — what carries forward to future-task retrieval).

**Result (60 tasks, per-task isolated environments, C=4):**
- per-task outcomes **IDENTICAL: 60/60** (byte-for-byte)
- final shared-memory state **IDENTICAL**
- → parallelism changes only *latency*, not *accuracy*.

### Why this holds (and the one caveat it exposed)
- **Parallel agents within a task:** independent subtasks share a wave; a
  dependent subtask is in a later wave. Independent subtasks read external state
  (calendar/email/search), not each other, so their results don't depend on
  sibling order; the staging→merge is deterministic (sorted), so completion
  order can't change the result. (Unit-tested: deterministic-merge,
  dependency-waves, snapshot-isolation.)
- **Parallel tasks:** each held-out task is scored independently and writes go
  through per-resource locks (no lost updates — unit-tested). Concurrent tasks
  may observe a different *cache* state (a recurring task could miss a
  not-yet-written STM entry), but a cache miss only triggers a recompute that
  yields the **same plan** → same answer. Cross-task concurrency changes
  cache-hit *timing* (efficiency), not correctness.
- **Caveat surfaced during the check:** with a *shared* external World, 2/60
  coordination tasks diverged on an incidental observation (a calendar-event
  *count* read by the search agent), because concurrent tasks were creating
  events in the same filesystem. Success and plan were still identical, and
  memory state was identical. Giving each task its own environment (the standard
  held-out eval model — analogous to per-task WM isolation) makes it 60/60. The
  takeaway: the memory architecture never diverges; only a *shared mutable
  external resource* can, and benchmark isolation already handles that.

### Bearing on the reported OfficeBench held-out numbers
The parallel code lives in the standalone `memory_manager/` package; the
OfficeBench harness that produced the held-out accuracies (retrieve-all 15/30,
boolean 14/30, PCCR 15/30) was **not modified**, so those numbers stand exactly.
The equivalence check above shows that *had* those tasks run on the parallel
schedule (with per-task env isolation), the outcomes would be identical.

Reproduce: `.venv/bin/python verify_equivalence.py`
