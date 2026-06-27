# T1 Improved-PCCR — N=4 Parallel Run Results

**Date:** 2026-06-28 · **Bank:** 62 training procedures · **Test:** 152 held-out subtasks
**Backbone:** Cerebras gpt-oss-120b · **Concurrency:** C=4 (4 Docker containers `ob-test-0..3`)
**Raw data:** `t1_improved_n4_final.json` (per-task records) · traces in `traces/t1_par/`

## Configuration

- **Improved arm (`--improved`)** = A1 confidence gate (θ_sim=0.55) + B1 verified replay
  (threshold 0.75) + D1 LLM-curated PM + D3 output convention + B3 step-hint + B4 L3 cap=45.
  (B2 plan-then-execute and A2 online-loop NOT in the bundle.)
- **Baseline** = frozen-priors PCCR (the published T1 `t1_progress.json`), run **sequentially**.
- **Concurrency dimension:** parallel **tasks** only (Dim-1). Within-task sub-agent parallelism
  (Dim-2) is NOT exercised on OfficeBench (the agent loop is one-action-at-a-time).

## Accuracy (152 matched tasks)

| Level | Improved | Baseline | Δ |
|---|---|---|---|
| **Overall** | **0.368** | **0.421** | **−0.053** ❌ |
| L1 (n=47) | 0.468 | 0.532 | −0.064 |
| L2 (n=48) | 0.479 | 0.562 | −0.083 |
| L3 (n=57) | 0.193 | 0.211 | −0.018 |

Win/Loss/Tie vs baseline: **W13 / L21 / T118**.

### By pattern
| Pattern | n | Improved | Baseline | Δ |
|---|---|---|---|---|
| lookup | 16 | 0.625 | 0.625 | 0.000 (tie) |
| doc_process | 25 | 0.600 | 0.600 | 0.000 (tie) |
| multi_app | 78 | 0.218 | 0.256 | −0.038 |
| single_action | 23 | 0.391 | 0.478 | −0.087 |
| data_compute | 10 | 0.500 | 0.800 | −0.300 |

## Accuracy vs RETRIEVE-ALL (strongest baseline arm; LegoMem-style always-inject)

| Level | Improved | Retrieve-all | Δ |
|---|---|---|---|
| **Overall** | **0.368** | **0.467** | **−0.099** ❌ |
| L1 (n=47) | 0.468 | 0.723 | −0.255 |
| L2 (n=48) | 0.479 | 0.583 | −0.104 |
| L3 (n=57) | 0.193 | 0.158 | **+0.035 ✅** (gate protects; retrieve-all collapses) |

W/L/T = 11/26/115. By pattern: doc_process tie; multi_app −0.051; lookup −0.188;
single_action −0.217; data_compute −0.300. Tokens: improved 878K vs retrieve_all 734K (**+20%**).
NB: both the accuracy gap and the +20% tokens are amplified by the parallel malformed-loops
(re-inject per step); the clean offline A1 token figure vs the full cascade was −26%.

## Cost

| Metric | Improved | Baseline (pccr) | Δ |
|---|---|---|---|
| **Injected tokens (total)** | **878,470** | **1,031,111** | **−15%** ✅ |
| vs retrieve-all | 878,470 | 734,295 | +20% (parallel-inflated; offline A1 −26%) |
| (offline per-retrieval vs cascade) | — | — | −26% (A1 suppresses SM/ENT) |

## Latency / throughput

| Metric | Value |
|---|---|
| Parallel speedup @ C=4 | **3.32×** (83% of ideal 4×) |
| Parallel makespan | 24.1 min (1447 s) |
| Sequential-equivalent (sum per-task wall) | 80.0 min (4801 s) |
| Compute time, 429 neglected | 63.0 min (3780 s) |
| Rate-limit (429) share of wall | 21% |
| Per-task: compute(no-429) / wall | 30.1 s / 37.7 s |
| Baseline per-task wall (sequential) | 53.1 s |
| Replay fired | 27 / 152 tasks |
| Total 429 hits | 1,328 |

## ⚠️ Confounds (why this comparison is NOT apples-to-apples)

1. **Parallel vs sequential execution.** Improved ran in **parallel** (worker threads);
   baseline ran **sequential** (main thread). The parallel run has a residual
   **malformed-action rate of 21% vs baseline's 11%** — the worker-thread timeout is a
   no-op (see fix below), so the parallel execution loses a few tasks/level that the
   sequential baseline doesn't. This penalizes "improved" for reasons unrelated to the features.
2. **LLM-calls/task (26.5 vs 15.0) is NOT real reasoning calls.** The counter counts every
   API *attempt*; at C=4 each request retries across keys under 429 pressure → retry
   inflation. Steps (13.7 vs 12.0) and run-level tokens are likewise inflated by the
   parallel malformed-loops.
3. **Docker crash + resume.** colima crashed at 110/152 (`ConnectionRefusedError`); the
   run was resumed for the final 42 (mostly L1). The 3.32× speedup is from the 110-task
   pre-crash segment.

## Key finding (harness bug, fixed)

The earlier catastrophic "regression" (89% malformed) was **not** the architecture — it was
OfficeBench's signal-based `timeout` (intercode), which only works in the main thread.
Under `asyncio.to_thread` (worker thread) it raised on every action → caught as "Malformed".
**Fix:** `runner._patch_intercode_timeout()` makes it main-thread-aware (committed). Verified
89% → 0% malformed in the worker-thread context. All features (A1/B1/B2/D1/D3) were innocent.

## Verdict & next step

- **Cost-efficient (−15% tokens) but accuracy −5.3pp behind.** A cost-for-accuracy trade,
  not a win. Biggest losses: data_compute (−0.30), single_action (−0.087).
- The comparison is **unfair to improved** (parallel malformed + retry inflation).
- **For a clean, publishable number:** re-run improved **sequentially** (malformed → ~11%,
  calls/tokens become real) and try **θ_sim=0.45** (0.55 is too strict — it skips
  medium-confidence EM that still helps, which is the likely cause of the L2/data_compute losses).
