# PCCR — Next Steps (post-T1 changes)

Plan for the changes after the T1 run completes. Goal: turn PCCR from "looks like LegoMem +
a tuning knob" into a **general, label-free routing mechanism** that **matches/beats
retrieve-all accuracy at lower cost and much lower latency.**

---

## 0. Where we are — the T1 findings that motivate this

3-arm test comparison (no_memory / retrieve_all=EM-always / pccr=full real cascade), real
`MemoryManager`, frozen priors:

- **Overall:** retrieve_all ~0.53 > pccr ~0.48 > no_memory ~0.46.
- **By level:** L1 retrieve_all dominates (0.72); L2 pccr competitive; **L3 PCCR is BEST
  (0.17–0.21) while retrieve_all collapses to ~0** (always-injecting EM distracts on hard
  multi-step tasks).
- **By pattern:** EM helps single_action/doc_process/lookup/data_compute (+0.10..+0.19),
  **neutral on multi_app**. pccr matches retrieve_all on doc_process/data_compute but is
  dragged to no-memory level on single_action/lookup — **because SM/ENT injections are noise**
  on OfficeBench (rule-based facts + sparse entity profile distract the model).
- **Cost:** pccr injects **+26% MORE tokens** than retrieve_all (SM/ENT bloat) despite 19%
  fewer EM consults.
- **Root problems:** (a) utility is **hand-set** → not general → no novelty vs LegoMem;
  (b) SM/ENT consulted because priors over-rate them → hurts accuracy AND cost;
  (c) memory is injected as a *hint* but the agent still runs the full step-by-step loop.

Achievable ceilings from the data: pattern-routed EM-only = 0.537 (> retrieve_all 0.529);
oracle EM-confidence gate = 0.570 (+4pp over retrieve_all).

---

## 0b. L3 autopsy — why hard tasks fail (and what recovers them)

L3 is ~0.10–0.21 across ALL three arms, so it is **not** a memory-routing problem — it is a
**task-completion** problem that hits every arm. Traced all 170 L3 runs + every `file_exist`
failure to its actual missing file:

```
L3 failed predicates (170 runs):  file_exist 67 · contain 59 · exact_match 5 · eval-CRASH 5
Step cap (30): only 21% hit it, only 7 of the 67 file_exist fails are cap-bound → NOT the cap.

Root cause of the 67 file_exist fails:
  36  output file NEVER created   ← agent finishes the 1st sub-goal, skips the rest
  17  path/name mistake           ← wrote it, but wrong dir (no data/) or renamed
  14  all files made; save/subdir edge case
46 of 67 are MULTI-output tasks (2–6 required files). Outputs never produced:
  .pdf 20 · .ics 15 · .docx 7 · .jpg 7 · .eml 5 · .txt 3 · .xlsx 3   (mostly the 2nd app)
```

**Mechanism:** L3 tasks are multi-app, multi-output (build the spreadsheet AND schedule the
calendar `.ics` AND convert-to-`.pdf` AND send the `.eml`). The agent completes the **first**
output and calls `finish_task`, never switching to the second app. Excel/PDF tools DO save
correctly (`wb.save`) — the files are simply never *attempted*. This is **incomplete multi-app
execution**, not reasoning difficulty, not the step cap, not a save bug.

**Why this is an architecture opportunity (memory should fix it, today's hint-injection doesn't):**
the EM episode holds the *full* multi-app sequence; today it is injected as a hint and the agent
still drops the tail outputs. The fixes are queued below — they target the 67 directly and hit all
three arms, with PCCR best-positioned (it already wins L3):
- **B1 replay → multi-output completeness** (recovers the 36 never-made): replaying the full cached
  sequence reproduces ALL outputs, including the 2nd-app ones. Replay is an **accuracy** lever here,
  not just latency.  See **B5**.
- **D3 PM completion-checklist + `data/` convention rule** (recovers the 17 path/name + reinforces
  the 36): tell the agent every required output + where/what to name it; don't finish until all exist.
- **Harness: L3 cap 30→45 + batch cell-writes** (recovers the ~7 cap-bound + frees budget for the
  2nd app).  See **B2**.

---

## A. Utility mechanism — the novelty / generalization fix  (TOP PRIORITY)

**A1. Per-query retrieval-confidence utility.** Replace the per-pattern hand-set table with
`U(store, query) = retrieval_confidence(store, query)` = the top-k embedding similarity from
that store's FAISS search. Gate: `rho = U/cost >= theta`; consult a store iff it clears theta.
- *Why it's the novelty:* label-free, **calibration-free, benchmark-agnostic** (one global
  theta, no per-task utility table). LegoMem always retrieves; PCCR consults a store only when
  *that store's retrieval is confident enough to be worth its cost*, decided fresh per query.
- *Effect:* automatically **suppresses SM/ENT** on OfficeBench (their hits are low-similarity)
  and **keeps EM** (high-similarity) — with NO per-benchmark tuning.
- *Use the embedding similarity, NOT an LLM-judged score* (an LLM judge adds a call → latency).
  The FAISS score is already produced by the retrieval — free.
- *Already exposed:* `RealArch.search` / `RealMem` give per-store top-k sims.

**A2. Online closed-loop refinement** (already wired: `memory_manager/online_utility.py` +
`MemoryRouter.record_outcome`). Layer on A1 to *learn* the confidence→utility mapping / theta
per store from observed outcomes (EWMA P_on vs P_off, epsilon-exploration, min-sample guard).
- *Why:* the gate self-corrects in deployment with no calibration runs.
- *Effect:* converges to suppress SM/ENT, sharpens per-store thresholds.

**A3. Keep phase-conditioning** (already there): a store is only *eligible* in the phases the
lifecycle table allows — bounds which stores are even searched per step.

> Utility is never "assumed": **computed per query (confidence, general) → refined online
> (outcomes, adaptive) → gated by phase.** That is the generalizing mechanism LegoMem lacks.

---

## B. Reduce LLM CALLS — the biggest latency lever  (HIGH PRIORITY)

**Key distinction:** a "step" today = **1 LLM call** (decide the action, ~0.5–1 s) **+ 1 tool
exec** (do it, ~0.1 s). Latency is ~99% the LLM calls. We do NOT remove the tool executions
(the real work must run); we collapse the **per-action LLM reasoning** into 1–2 calls, and we
cut the **wasted/exploratory actions** memory lets us skip (observed: 3-90/0 = 9 steps with
PCCR vs 18 no-memory).

```
no-memory:  ~18 actions, 1 LLM call each → 18 LLM calls (slow + flailing)
PCCR loop:   ~9 actions, 1 LLM call each →  9 LLM calls (memory cut the flailing)
PCCR replay:  9 actions, planned in 1 call → ~1–2 LLM calls (plan once, execute 9)
```

**B1. ADAPTIVE + VERIFIED procedure replay (NEVER blind) on high-confidence cache hits.**
The cached procedure already holds the action sequence. On a high-confidence STM/near-dup hit,
plan it once and execute without per-step LLM reasoning — but with two safeguards so it stays
correct when the current task differs (e.g. cached=10 steps, current needs 11):

1. **Adapt call (1 LLM call) reasons about the DIFFERENCE.** Given the cached procedure AND the
   current task, the LLM outputs the *complete adapted plan*: rewrite params (files/values/cells),
   **ADD** steps the current task needs (e.g. the 11th), **REMOVE** steps not needed. So the LLM
   *does* know about the extra step — it compares the two tasks once, upfront.
2. **Execute the adapted sequence; verify each action by the tool's OWN signal (no LLM).**
   OfficeBench returns explicit success/failure per action:
   `"Successfully ..."` (ok) vs `"Malformed action!"` / `"... does not exist"` / nonzero exit (fail).
   ```
   for action in adapted_plan:
       obs = env.exec(action)
       if is_failure(obs):  → HALT, hand back to the LLM loop to recover from here (1 LLM call)
       else:                → continue            (NO LLM call)
   ```
3. **End-state goal check** (catches "succeeded but wrong"): does the required file/answer exist?
   If not → hand back to the LLM loop to finish/fix. (Programmatic, or 1 cheap LLM check.)

- *Cost:* all-ok common case ≈ **2 LLM calls** (plan + goal check) vs N per-step calls; on a
  failure, recover from that point (worst case = step-by-step from the failure → no worse than
  today). Verification is by **tool signal + goal check**, with the LLM as the *recovery*
  fallback only — not a per-step verifier.
- *Novelty:* memory becomes an **executable, monitored procedure** (a learned skill/macro with
  a trip-wire), not just retrieved context. LegoMem/retrieve-all only hint, then still loop.
- *Gating:* replay only on **high similarity** (small, safe delta). Low similarity → full loop.

**B2. Plan-then-execute (batched actions)** for medium-confidence consults: the agent commits to
a multi-step plan from the injected procedure and emits several actions per LLM call → ~2–3×
fewer calls than step-by-step (with the same tool-signal + goal verification).

**B3. Memory-guided step budgeting / early stop** — the retrieved procedure indicates expected
step count → cut exploratory flailing.

**B4. Raise the L3 step cap 30→~45 + batch cell-writes** (harness, cheap). `set_cell` is 1 cell
= 1 LLM step, so a spreadsheet eats ~20 steps and starves the 2nd-app output. Plan-then-execute
(B2) emits multiple cell-writes per call; the higher cap covers the genuinely longer pipelines.
Recovers the ~7 cap-bound L3 fails and frees budget for the 2nd app. *(Re-confirm a few capped
tasks pass at cap=45 before committing.)*

**B5. Multi-output completeness (the 36 never-made L3 fails — accuracy lever).** L3 tasks need
2–6 output files across apps; the agent finishes output #1 and quits. Two memory mechanisms close
this, both already in scope:
- **Full-sequence replay (B1):** on a high-confidence EM/STM hit, the adapt-call rewrites params
  but **preserves the full multi-app sequence** (excel→…→calendar→…→pdf), so executing it produces
  *every* output, not just the first. Add an explicit instruction to the adapt prompt: *"keep ALL
  output-producing steps; do not drop the 2nd/3rd app."*
- **Completion gate before `finish_task`:** programmatically refuse `finish_task` until every
  required output the plan named exists on disk (reuse the B1 end-state goal check over the full
  output list). If missing → hand back to the loop to produce the remaining outputs.

> Tiering: high-conf near-dup → **adaptive+verified replay** (~10× fewer LLM calls) ·
> medium-conf → **plan-then-execute** (~2–3× fewer) · low-conf/novel → full loop.

---

## C. STM — keep it a bounded hot-cache (scalable); demo on recurring workloads

**C1. Bounded cache, NOT prefill-all.** Prefilling all N successes defeats STM's purpose (it
becomes a duplicate of EM, no cascade benefit) and does not scale to 1000+ tasks. Design:
- Seed by **cluster exemplar** (current `consolidate()` — size ~ #clusters, grows sub-linearly:
  1000 tasks → ~100 clusters → ~100 STM entries; EM holds the full 1000).
- **LRU/LFU eviction with a fixed budget** (e.g. 50–100): frequently-reused procedures get
  promoted, cold ones evicted. STM always = the hot set.

**C2. Demonstrate the latency win on a RECURRING workload** (replay/repeat stream), not the
held-out single-pass test. On held-out, near-dup hits are inherently rare (~7 at >=0.85) — STM's
short-circuit + replay (B1) pays off when task types **recur**. Report hit-rate↑ → latency↓ there.

**C3. (optional) Lower threshold 0.85→0.80 with a pattern-match safety guard** to get more hits
only where the cached plan is safe.

**C4. Scale note (1M+ vectors):** switch `IndexFlatIP → HNSW/IVFPQ` (FAISS is built for this) —
search stays ~1–10 ms/query; STM short-circuit lets most queries skip the big-index search; PQ
compresses memory (1.5 GB → ~50–150 MB/store).

---

## D. PM — LLM-curated, actionable rules (accuracy)

**D1. Replace rule-based distillation with LLM curation.** Current PM rules are just the
most-common step signature (`"prefer plan: shell→excel"`) + a truncated reflection — weak.
LLM-distil each cluster (1 cheap offline call) into a **crisp, actionable rule with cautions**:
> *"data_compute (Excel): 1) shell `ls /testbed/data` to find the file; 2) read it; 3) compute;
> 4) write to the EXACT cell; 5) **save the file**. Common failures: forgetting to save,
> off-by-one (row 1 is the header)."*
- *Effect:* targets the discriminating-detail failures we found (didn't save, off-by-one,
  missing sort). Modest, real bump on **habit-driven** failures (not task-specific reasoning).
- *Scalable:* #rules grows with task **diversity**, not task count.

**D2. PM lookup bug is fixed** (`get_rule_for_pattern` now reads `learned_rules`) — keep wired.

**D3. Output-convention + completion-checklist rules (targets the 17 path/name + 36 never-made
L3 fails).** Two PM/prompt rules the agent currently never sees (grep of the prompt = empty):
- **`data/` convention:** *"Write every output file under `/testbed/data/` using the EXACT
  filename named in the task — no prefixes, no renames."* Fixes the 17 wrote-to-wrong-dir /
  renamed failures (e.g. wrote `/testbed/report.docx` for `data/report.docx`; `Anderson_admission.pdf`
  for `admission.pdf`). Cheap, recovers across all arms.
- **Completion checklist (per task type, LLM-curated like D1):** *"multi_app calendar+excel tasks
  emit N outputs: the `.xlsx` AND one `.ics` per attendee — produce all before finishing."*
  Reinforces B5's programmatic completion gate with a planning-time reminder.

---

## E. SM/ENT — suppress on OfficeBench (falls out of A)

Subsumed by the confidence gate (A1): low-similarity SM/ENT hits are skipped automatically. As
an explicit ablation, also test SM/ENT utility = 0. Expected: recovers single_action 0.48→~0.61
and lookup 0.62→~0.81, removes the +26% token bloat (→ ~19% *fewer* tokens than retrieve-all).
NB: SM/ENT are *relevant* on LongMemEval — the *same* confidence gate consults them there. That
contrast IS the generalization demo (F2).

---

## F. Experiments (the paper results)

**F1. The 3-arm utility story (core narrative):** frozen-priors (fails, honest negative) →
**confidence/measured-utility** (matches/beats retrieve-all at ~19% lower cost, wins L3) →
**online-loop** (learns the same without calibration).

**F2. Generalization figure (the novelty claim):** the **same confidence gate, same theta, no
re-tuning** on **OfficeBench AND LongMemEval** → one mechanism wins on both (SM/ENT suppressed
on OB, consulted on LME — by the same rule).

**F3. Latency/step result:** adaptive replay + plan-then-execute → report **LLM-calls/task** and
median latency vs retrieve-all; show ~10× fewer LLM calls on near-dup/recurring tasks.

**F4. STM on a recurring workload:** hit-rate vs cache budget; latency↓ from short-circuit+replay.

**F5. Ablations (attribute each gain):** confidence-gate · +LLM-PM · +bounded-STM · +adaptive-
replay · +online-loop.

**F6. L3 recovery table (from the 0b autopsy):** report `file_exist` fails before/after, split by
bucket (36 never-made / 17 path-name / 7 cap), attributing recovery to B5 replay+completion-gate,
D3 convention rule, and B4 cap — shows L3 lift is a *completion* fix, and that PCCR (already L3-best)
gains most. This is the concrete answer to "why is L3 low and does our architecture fix it."

> Target end state: **PCCR (confidence-gated, EM-focused, LLM-PM, bounded-STM, adaptive replay):**
> ≥ retrieve-all accuracy (wins L3 where retrieve-all = 0), **~19% fewer tokens**, **~10× fewer
> LLM calls / lower latency on recurring tasks**, via **one generalizing, label-free mechanism.**

---

## G. Implementation order (after T1 finishes)

0. **L3 quick wins first (cheap, large, all-arm)** — confirmed by the 0b autopsy:
   **D3 `data/` convention + completion-checklist rule** (17 path/name) · **B4 cap 30→45 +
   batch cell-writes** (~7 cap-bound). Validate on the 67 file_exist tasks before the deeper work.
1. **A1 retrieval-confidence gate** — kills SM/ENT noise; the generalization core.
2. **B1 adaptive + verified replay** + **B5 multi-output completeness gate** — the big latency win
   AND the recover-36-never-made accuracy win (never blind; preserve full multi-app sequence;
   tool-signal + goal-check + completion verification; LLM recovers on failure).
3. **D1 LLM-curated PM** — accuracy.
4. **A2 online-loop arm** — deployment / learned-utility claim.
5. **C bounded-STM + recurrence eval**, **B2 plan-then-execute**, **F2 generalization run**.
6. **F5 ablations** + write-up into `paper/combine.tex`.

## H. Run the improved architecture on the PARALLEL arch (N=4) — branch + adapter

Plan: let T1 finish on this branch (baseline), **branch from here**, implement A–E
(sequential, confirm accuracy/cost), then run the improved PCCR through the parallel
architecture at **N=4** to get **real-LLM parallel latency** on a real multi-agent
benchmark (currently we only have modeled 3.73× + LongMemEval 1.19×). Bonus: the N=4 run
also serves as a real-LLM check of parallel≡sequential (accuracy should match the
sequential improved run).

### H1. Experiment design (no fixed batches)
- N=4 = **4 concurrent WORKERS pulling from one queue of all 152 tasks** — NOT a chosen
  subset of 4. Each worker grabs the next task when it finishes its current one
  (`AsyncTaskQueue(max_concurrency=4)`, a semaphore over the full list).
- **4 workers <-> 4 Docker containers** (`ob-test-0..3`): worker *i* uses container *i*; each
  container processes a stream one-at-a-time; 4 streams run in parallel.
- **Run only the improved PCCR arm** through the queue; compare wall-clock to sequential
  (1 worker). Speedup = sequential_total / parallel_total.
- **LPT ordering:** sort tasks longest-first (L3→L2→L1) before queueing so the long L3 tasks
  don't tail and stall a worker (minimizes makespan).

### H2. Parallel architecture readiness — what's READY (do NOT rebuild)
- `AsyncTaskQueue` (`task_queue.py`): worker pool / semaphore, per-task failure isolation,
  consolidation-window wait. ✓
- `MemoryLockManager` (`parallel.py`): per-resource locks (stm/ent/faiss/pm/history). ✓
- `DependencyAnalyzer` + `ParallelExecutor`: DAG→waves, parallel sub-agents, snapshot/merge. ✓
- Invariants proven: `verify_equivalence.py` = 60/60 byte-identical + identical memory state. ✓
- `CerebrasLLM`: thread-safe multi-key round-robin (`_lock`), so 4 concurrent tasks naturally
  spread across the 22 keys — already parallelism-friendly. ✓

### H3. Modifications NEEDED (the OfficeBench parallel adapter — ~1 new file + guards)
The queue was only ever driven by **stub agents** (bench_parallel = asyncio.sleep,
verify_equivalence = deterministic World). To drive **real OfficeBench tasks** in parallel:

1. **Async pipeline wrapping the BLOCKING runner (critical).** `run_task` is synchronous and
   blocking (Docker exec + LLM HTTP). The `_pipeline` must do
   `await asyncio.to_thread(run_task, ...)` — else the 4 "concurrent" tasks run **serially**
   (the event loop blocks on each Docker/LLM call). This is the make-or-break piece.
2. **Per-worker container pool.** `run_task` currently uses one shared `ob-test`. Create
   `ob-test-0..3` and an **acquire/release pool (size 4)**: each running task grabs a free
   container, releases on finish. (`run_task` already takes `container=`, so it's just routing.)
3. **LPT ordering** of the 152 tasks (L3→L2→L1) before feeding the queue.
4. **Read-safety guard.** Concurrent FAISS searches are fine (bank frozen during eval, no
   writes); wrap `embedder.encode()` in a `threading.Lock` to be safe under 4 threads.
5. **Driver `run_t1_parallel.py`:** build `RealMem` once (shared, read-only), set up the
   container pool, LPT-order tasks, run `AsyncTaskQueue(pipeline, max_concurrency=4)`,
   measure wall-clock vs sequential.

### H4. Caveats to plan for
- **4 containers = ~4× Docker/colima RAM+CPU.** On a laptop this can be heavy — ensure colima
  has enough RAM, or test **N=2 first**, then N=4.
- **Rate limits ×N:** 4 concurrent tasks ~ 4× request rate. Mitigated by the fixed paced/rotate
  multi-key client (assign different keys to different streams). The just-fixed `cerebras_llm.py`
  (max_retries=0, skip dead keys, paced rotation) is required for this to work.
- **Memory consistency:** during the frozen-bank eval there are no writes, so concurrent reads
  are safe; if the online-loop (A2) writes utilities during a parallel run, route those writes
  through `MemoryLockManager` (history_lock) — the locks already exist.

---

## Honest caveats / guardrails (carry into the paper)
- Confidence ≠ correctness perfectly — high-sim tasks can still differ in a discriminating detail;
  hence replay is **adaptive + verified**, not blind.
- STM/replay wins concentrate in **recurring** workloads; on held-out single-pass they are rare
  (report honestly; demo on a recurrence stream).
- Backbone limit (gpt-oss-120b) caps absolute accuracy; the claims are **relative** (vs
  retrieve-all/no-memory) + **mechanism generalization**, which are backbone-independent.
- Keep `cost` real: report injected tokens + LLM-calls (rate-limit-independent), use **median**
  latency (the mean is contaminated by 429 backoffs).
