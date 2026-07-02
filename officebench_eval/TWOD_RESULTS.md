# 2D (parallel sub-agents) on OfficeBench — full-152 results

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

## Timing note
The N=4 `_run.total_wall_s` (205.5s) is **unreliable** — the watchdog restarted the run twice
(healing stalls), resetting the wall timer; it covers only the final segment. Use the
**sequential-equivalent compute** (sum of per-task wall) = **2963s ≈ 49 min**. Note pure-2D's low
compute/tokens is because sub-agents **fail fast**, not real efficiency.

## Artifacts
- `twod_par152_traces.json` — full machine traces (all fields) + `_run` meta.
- `traces/twod_par152/<task>_<sub>_2d.{json,txt}` — per-task machine + readable pair (152 each).
- `build_2d_table.py` (`TWOD_TRACES=... `), `routed_152.py` — reproduce the numbers.
- N=1 run traces (reached 80 tasks before the N=4 restart): `traces/twod_all152/`.
