# 2D (parallel sub-agents) on OfficeBench — full-152 results

> **UPDATE (clean N=4 measured run, `twod_par152_traces.json`):** pure-2D = **18/152**, real tokens **1.25M**, successful calls **1636** (raw 4356 incl. 2252 429-retries + 468 dead-key skips), tok/call **765**, injected **0**, **measured continuous N=4 wall = 15.6 min** (seq-equiv 25.5 min). This is the row now in Table 18 (tab:final). The earlier 17/152 run is archived in `twod_par152_traces_run1.json`.


Backbone gpt-oss-120b (Cerebras). Accuracy / tokens / calls / cost are per-task, so they are
**concurrency-independent** (identical at task-level N=1 or N=4); only wall-clock changes with N.

## Headline table (N=4 task-level run, `run_2d_par152.py`)

| Arm | Accuracy | Real tok | tok/call | calls | cost $ |
|-----|:--------:|:--------:|:--------:|:-----:|:------:|
| retrieve-all (1D seq)              | 72/152 | 4.47M | 1955 | 2284 | 1.46 |
| improved PCCR (1D)                 | 74/152 | 7.15M | 2273 | 3146 | 2.51 |
| **PURE-2D (all 152, fair)**        | **17/152** | 1.34M | 677 | 1974 | 0.66 |
| 2D-router — general structural (fires 62) | 60/152 | 3.83M | 1454 | 2633 | 1.46 |
| 2D-router — keyword niche (fires 8)       | **75/152** | — | — | — | — |

## The honest verdict (answers "why can't pure-2D on 152 match retrieve-all?")

1. **Pure-2D on all 152 = 17/152** — catastrophically below retrieve-all (72). Running the
   decomposition architecture on every task destroys accuracy on the sequential / non-fan-out
   majority. By level: L1 7/47, L2 9/48, **L3 1/57** (multi-step pipelines collapse). Paths taken:
   planner 131, single 15, fastpath 6.
2. **A GENERAL, keyword-free structural router = 60/152 — still below 1D.** Routing to 2D wherever a
   parallel decomposition materialises (`fanout_fired`, an act wave ≥2 leaves) fires on 62 tasks and
   **regresses 16 that 1D passed** — because "a parallel decomposition exists" ≠ "2D will succeed";
   the planner over-decomposes sequential tasks into parallel-looking waves that then fail.
3. **Only the narrow keyword-niche gate (8 calendar-create tasks) beats 1D: 75/152 vs 74.** The +1
   (solving 2-14/0, top-three students) is real but requires task-structure-specific gating — the
   overfit fairness concern is valid: 2D is a *niche* optimisation, not a general agent.

### Why (mechanisms, confirmed by the traces)
Only ~8/152 tasks are per-item fan-outs; 47 are multi-app SEQUENTIAL pipelines (no parallelism to
exploit) and 97 are single/other. Decomposition (a) loses the whole-task context needed for
cross-step reasoning, (b) adds the planner as a new per-task failure point, (c) passes a lossy ~2KB
blackboard between waves, (d) compounds sub-agent errors with no shared-context recovery. Example:
`traces/twod_par152/1-12_2_2d.txt` — trivial lookup "What is X's HW2 score?" split into two
sub-agents that both re-read the file and never answered.

## Wall-clock / timing
The N=4 `_run.total_wall_s` (205.5s) is **unreliable** — the watchdog restarted the run twice
(healing stalls), resetting the wall timer; it covers only the final segment. Reconstruct the true
wall from per-task `worker_elapsed_s` (monotonic within a segment, drops at each restart -> sum each
segment's max; this excludes the idle hang before a restart):

| Time metric (pure-2D N=4) | value |
|---|---|
| N=4 parallel wall (active, segment-reconstructed) | **25.4 min** |
| seq-equiv compute (sum of per-task wall) | 49.4 min |
| speedup (seq-equiv / parallel) | **1.9×** |
| retrieve-all (1D seq) wall | 69.7 min |
| improved PCCR (1D) wall | 128.3 min |

Speedup is only ~1.9× (not ~4×) because 4 concurrent tasks × parallel sub-agents saturate the
Cerebras API (429 contention) and contend on concurrent Docker setup. And pure-2D's low wall/tokens
is partly because sub-agents **fail fast** (1.34M tokens vs retrieve-all 4.47M) — not real efficiency.
`build_2d_table.py` prints the reconstructed wall from the traces.

## Artifacts
- `twod_par152_traces.json` — full machine traces (all fields) + `_run` meta.
- `traces/twod_par152/<task>_<sub>_2d.{json,txt}` — per-task machine + readable pair (152 each).
- `build_2d_table.py` (`TWOD_TRACES=... `), `routed_152.py` — reproduce the numbers.
- N=1 run traces (reached 80 tasks before the N=4 restart): `traces/twod_all152/`.

## Exact-diagram arm (DAG-first + orchestrator re-plan + PCCR-agent leaves), N=4
Measured full 152 (`par_dagreplan_progress.json`, traces in `traces/par_dagreplan/`, 152 pairs):
**64/152** (L1 29/47, L2 24/48, L3 11/57) | real tokens **8.10M** | tok/call 1687 | calls **4802 (~49/parallelised task)** | injected **1350K** | N=4 wall (LPT est.) **51.6 min** (seq-equiv 205.8 min).
This is the arm that realises fig:arch literally (Phase 4-5 orchestrator re-plan loop → DAG → parallel PCCR sub-agents → merge → "more groups?"; sequential chains → single PCCR fallback). Verdict: **most faithful but most expensive AND least accurate** of the memory arms — 1.5-3× the cost of the keyword-gated PCCR+2D (68) for -4 accuracy, because the LLM orchestrator re-plans one group at a time and mis-decomposes (and injects memory every round). The deterministic gated fast-path is the better engineering choice. (Run survived ~6 colima docker-socket crashes; watchdog now restarts colima on DEAD too.)
