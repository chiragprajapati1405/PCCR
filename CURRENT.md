# PCCR — Current State (as implemented now)

Snapshot of what the architecture **currently does**, mirroring the section layout of
`NEXT_STEPS.md` so the two are directly comparable (CURRENT vs what we will change).

---

## 0. Summary table (current → planned)

| Area | CURRENT (now) | NEXT_STEPS (planned) |
|---|---|---|
| **Utility** | hand-set per-pattern priors (`PATTERN_UTILITY`) | per-query **retrieval-confidence** + online loop |
| **Gate** | `rho = U/C >= theta`, U from priors | same gate, U from confidence (label-free, general) |
| **STM** | real bounded store, prefill 1/cluster, thresh 0.85 | + LRU/LFU eviction, recurrence-workload eval |
| **PM** | rule-based distill (step-signature + reflection) | LLM-curated actionable rules + cautions |
| **SM/ENT** | live + ρ-gated (noise on OfficeBench) | suppressed by confidence gate (kept on LongMemEval) |
| **Steps** | step-by-step loop, 1 LLM call/action, memory = hint | adaptive+verified **replay**, plan-then-execute |
| **Closed loop** | wired (`online_utility.py`) but NOT used in eval | run as an arm (learned utility) |
| **Parallelism** | implemented + proven equivalent (modeled speedup) | unchanged (orthogonal) |

---

## A. Utility mechanism — CURRENT

- `MemoryRouter` computes `rho = expected_utility(pattern, store) / store_cost(store)`; consults a
  store iff `rho >= consult_threshold (theta)`.
- **`U` is HAND-SET** per (pattern, store) in `router.py: PATTERN_UTILITY` (e.g. LOOKUP→EM 0.15,
  SM 0.45, ENT 0.80). `C` is hand-set in `STORE_COST` (EM 0.55, …).
- T1 runs with **frozen priors** (`--priors`): the calibrated values were confounded, so we use
  the hand-set table directly.
- **Online closed-loop EXISTS** (`memory_manager/online_utility.py`, `MemoryRouter.record_outcome`):
  learns `U = P_on − P_off` per (pattern, store) with EWMA + epsilon-exploration + min-sample guard,
  and a per-query confidence scaler is supported in `plan_retrieval(confidence=...)`.
  **But it is NOT active in the T1 eval** (frozen priors only).
- **Limitation:** utility is assumed (hand-set) → not general, no novelty vs LegoMem. (This is the
  thing NEXT_STEPS A1/A2 fixes.)

## B. Step execution — CURRENT

- **Step-by-step agent loop**: 1 LLM call decides each action, then 1 tool exec. An L3 task ≈ 20–30
  LLM calls.
- **Memory is injected as a HINT** into the prompt (orchestrator plans + agent actions + PM rule +
  SM/ENT) — the agent still re-derives every action with its own LLM call.
- **No replay, no batching, no procedure execution.** (NEXT_STEPS B1/B2 adds adaptive+verified
  replay and plan-then-execute.)
- Action-pipeline fixes already in: `max_tokens=2048` (no truncation) + first-balanced-JSON
  `proc_action` (no merged-JSON) → malformed actions ~47%→~10%.

## C. STM — CURRENT

- Real `ShortTermStore` (`memory_manager/stores/short_term.py`), wired via `RealMem` (the real
  `MemoryManager.retrieve()` path).
- **Prefilled by `consolidate()` with ONE exemplar per cluster** (~13 entries from the 62-proc bank)
  — already bounded by #clusters (scalable), NOT prefill-all.
- Lookup threshold **0.85** in T1 (default 0.92). **Hit → short-circuit:** returns the cached plan
  and **skips EM/SM/ENT** (`retrieve()` honors `stm_hit`).
- **No eviction policy / usage-based refresh yet** (NEXT_STEPS C1 adds LRU/LFU + fixed budget).
- Index = `IndexFlatIP` (exact). Fine at 62 procs; NEXT_STEPS C4 = HNSW/IVFPQ at 1M+.

## D. PM — CURRENT

- `consolidate()` distills **rule-based** PM rules: most-common cluster step-signature + a truncated
  reflection (`"For tasks like X, prefer plan: shell→excel. Lesson: …"`). ~13 rules.
- Injected as an always-read `##PROCEDURAL RULE` block on the gate arm.
- **Lookup bug FIXED**: `get_rule_for_pattern` now reads `learned_rules` (was write-only). PM fires
  on gate tasks from the fix forward.
- **Rules are generic** (plan shape, not actionable cautions) → weak. (NEXT_STEPS D1 = LLM-curated.)

## E. SM / ENT — CURRENT

- **Live and ρ-gated** via `RealMem` (the real cascade). Per-pattern routing: lookup → SM+ENT,
  data_compute → EM+ENT, etc. SM has ~180 rule-extracted facts; ENT seeded minimally.
- **On OfficeBench they are NOISE** — rule-based facts + sparse entity profile distract the model,
  dragging pccr below retrieve-all on single_action/lookup and adding +26% tokens.
- (NEXT_STEPS A1/E: confidence gate suppresses them on OB; they stay consulted on LongMemEval.)

## F. Parallel architecture — CURRENT (done, unchanged in NEXT_STEPS)

- 2-D concurrency (parallel tasks + parallel sub-agents) as 7 layers; ρ-gate + 10-phase lifecycle
  unchanged.
- **Proven equivalent:** 60/60 byte-identical outcomes + identical memory state on the deterministic
  harness (`verify_equivalence.py`) via WM isolation, deterministic wave-merge, per-resource locks.
- **Latency:** modeled **3.73× at C=4** (measured 0.516 s/call), ~13.5× at C=16; real LongMemEval
  combined gate+parallel run 1.19× (rate-limit-bound).
- Honest gap: sub-agent parallelism proven on the deterministic harness, not on a real-LLM
  multi-agent benchmark (needs per-task container isolation).

---

## G. Current experimental results (T1, real architecture, frozen priors)

**Status: 375/456 runs (incomplete — rate-limit-blocked), 123 matched tasks.**

3-arm (no_memory / retrieve_all=EM-always / pccr=full real cascade):

```
                acc       note
no_memory       ~0.46
retrieve_all    ~0.53     EM-always; the current accuracy leader
pccr (real)     ~0.48     full cascade; below retrieve-all (SM/ENT noise on L1)

by level:        L1            L2            L3
  no_memory      0.55          0.51          0.17
  retrieve_all   0.72          0.60          0.00   ← collapses on hard tasks
  pccr           0.53          0.57          0.21   ← BEST on L3

by pattern (EM effect = retrieve_all − no_memory):
  multi_app     EM neutral; pccr ~ no_mem ~ retrieve_all
  single_action EM +0.13, but pccr dragged to no_mem (SM/ENT noise)
  doc_process   EM +0.15, pccr MATCHES retrieve_all
  lookup        EM +0.19, but pccr dragged to no_mem (SM/ENT noise)
  data_compute  EM +0.10, pccr MATCHES retrieve_all

cost:   pccr injects +26% MORE tokens than retrieve_all (SM/ENT bloat), 19% fewer EM consults
latency (median): all ~17-18s on L1/L2; on L3 pccr FASTEST (25s vs retr 28s vs no_mem 38s)
STM short-circuits: ~7 (held-out → rare near-dups, as expected)
PM injected: yes (post-fix), on the later gate tasks
```

**What this shows now:** EM helps, but the **hand-set priors over-consult SM/ENT** → pccr loses on
easy tasks (L1) and costs more tokens; pccr already **wins on hard tasks (L3)** where retrieve-all
collapses. The fix is to make utility **measured/learned (confidence-based)** and to **execute
memory** rather than just hint it — i.e. everything in `NEXT_STEPS.md`.

---

## H. What is already in the codebase vs not

**Implemented now:** 6 stores + router ρ-gate (hand-set U) · real `MemoryManager.retrieve()` cascade
(STM short-circuit → ρ-gated EM/SM/ENT, PM always-read) · `consolidate()` (EM/SM/PM/STM) · online
closed-loop utility (wired, not used in eval) · per-query confidence scaler in `plan_retrieval`
(supported, not used) · parallel arch + equivalence proof · action-pipeline fixes · PM lookup fix.

**NOT yet (the NEXT_STEPS work):** confidence-as-primary-utility · adaptive+verified procedure replay
· plan-then-execute · STM LRU/LFU eviction · LLM-curated PM rules · SM/ENT suppression run ·
generalization run (same gate OB+LME) · the ablation suite.
