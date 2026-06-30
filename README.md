# PCCR — Phase-Conditioned Cascading Memory Router (OfficeBench 72/152 reproduction)

This branch is a self-contained snapshot that reproduces the full-152 **N=4** OfficeBench result
**72/152 = 0.474** (parity with an always-retrieve baseline at lower per-call cost), together with the
**per-task N=4 traces** for that run. For the exact run command and environment setup see
[`REPRODUCE.md`](REPRODUCE.md); this file explains the **architecture**.

---

## 1. What PCCR is

PCCR is a memory router for LLM agents. Instead of dumping all available memory into every prompt
(*retrieve-everything*), it decides **which memory store to consult** per task, conditioned on the
agent's **lifecycle phase** and the **task pattern**, and gates each optional store by an explicit
**cost-effectiveness ratio**

```
        ρ = U / C ≥ θ      →  consult store s ;  otherwise skip it
```

where `U` is the store's estimated utility for this (phase, pattern) and `C` its retrieval cost. The
goal: keep the accuracy of retrieve-everything while injecting far less, and decide *what to consult*
as a tunable, measured decision rather than a reflex.

### The six memory stores

| Store | Role | Read policy |
|---|---|---|
| **WM** — Working Memory | per-task scratchpad (current goal, plan, intermediate state) | always read; **isolated per task** |
| **PM** — Procedural Memory | learned procedural *rules* per task pattern ("how to do this kind of task") | always read |
| **STM** — Short-Term Memory | cache of recent **plan bundles** (a shortcut for a near-identical task) | **checked first**; a confident hit *short-circuits* the cascade |
| **EM** — Episodic Memory | specific past task **trajectories** (FAISS-indexed by description) | ρ-gated |
| **SM** — Semantic Memory | facts distilled from episodes | ρ-gated |
| **ENT** — Entity Memory | per-user / per-entity profiles | ρ-gated |

### The cascading gate (read path)

1. **STM first.** If the current task is a near-duplicate of a cached bundle (similarity ≥ 0.85,
   same pattern), inject just that cached plan and **skip EM/SM/ENT entirely** — the cheapest,
   highest-precision path.
2. **Otherwise, ρ-gate the optional stores.** PM and WM are always read; EM/SM/ENT are each consulted
   only if `ρ = U/C ≥ θ`. In the OfficeBench runs the gate is the **confidence gate**: consult a store
   iff its top-k FAISS similarity ≥ `θ_sim = 0.45` (label-free, benchmark-agnostic — it suppresses
   low-similarity SM/ENT noise automatically).

### The ten-phase lifecycle (where memory is read and written)

Retrieval is **Phase 3** (the ρ-gate, run once per task). Writing is **Phase 9** (success-gated:
`pattern → STM` and `person → ENT`, under locks; EM/SM deferred to consolidation). **Consolidation**
is an exclusive window that rebuilds EM/SM/PM and pre-fills STM. PCCR conditions the Phase-3 decision
on the phase + pattern; the rest of the lifecycle is unchanged.

---

## 2. The two-dimensional parallel architecture

The reference router is sequential (one task, one sub-agent at a time). It is extended with **two
orthogonal dimensions of concurrency** as a seven-layer design that leaves the ρ-gate and the
ten-phase lifecycle **intact**:

| Layer | Adds |
|---|---|
| L1 Async task queue | **parallel tasks** (many tasks at once) |
| L2 Per-task pipeline | **WM isolation** (each task gets its own scratchpad) |
| L3 Dependency analyzer | sub-tasks → dependency **waves** (a DAG) |
| L4 Parallel agent execution | **parallel sub-agents** within a task |
| L5 Memory lock manager | STM locked **per pattern key**, ENT **per username** |
| L6 Phase-9 parallel writes | success-gated writes under those locks |
| L7 Consolidation window | exclusive global window (holds the FAISS lock) |

These make the parallel schedule **provably and empirically outcome-equivalent** to the sequential
one (same answers, byte-identical on the equivalence suite); concurrency buys throughput, not
different results. The OfficeBench run here uses **L1 (parallel tasks) at N=4** with per-worker Docker
containers; a task may read STM before a concurrent task's write lands, an ordering L5 governs.

---

## 3. The four execution-fidelity mechanisms (what produced 72/152)

With the backbone fixed (gpt-oss-120b), accuracy turned out to be gated not by *which* store is
consulted but by the **fidelity of what the cascade injects** and by whether the agent is allowed to
stop prematurely. These four mechanisms — all read/write-path extensions that do **not** touch the
ρ-gate or the lifecycle — moved OfficeBench from 0.421 to **0.474**:

- **F1 — Schema-validated episodic injection.** Banked trajectories can contain *invalid* tool calls
  (a malformed step the agent later corrected). Injected verbatim as "past successful actions," they
  teach the backbone to imitate invalid calls. A sanitiser (`real_mem._clean_past_action`) parses each
  candidate, keeps it only if its `(app, action)` is schema-valid, remaps known aliases
  (e.g. non-existent `shell/run` → real `shell/command`), and drops the rest. *Storage unchanged; only
  what reaches the prompt is constrained to the tool schema.*

- **F2 — Inject-once context scheduling.** The cascade output has a *static* part (the PM rule + action
  exemplars, most useful while planning) and a *task roadmap* (the matched plan steps). The reference
  design re-sends the whole block every step. F2 splits them (`real_mem.retrieve` →
  `heavy_block`/`light_block`; scheduled in `pccr_policy.build_prompt`): heavy part injected only for
  the first **K=3** steps, light roadmap every step. Cuts per-call memory injection **709 → 146 tokens**.

- **F3 — Completion gate over both stop signals.** A procedural agent ends with `finish_task` ("done")
  or `got_stuck` ("give up"). The reference completion check intercepted only the former, but the
  dominant OfficeBench multi-app failure is **abandonment** (quit with required outputs unmade). F3
  (`pccr_policy._finish_gate`) intercepts **both**: if a stop is requested while a required output file
  is missing, it blocks and re-prompts. It excludes task *inputs* from the required set, and gives
  **schema-aware** guidance for `.eml`/`.ics` outputs (the exact `send_email`/`create_event` fields)
  since those come from tool actions, not file writes.

- **F4 — Continual short-term cache.** Not a new mechanism — it **exercises the lifecycle's own**
  Phase-9 `success → STM` write (`real_mem.cache_success`) *during* evaluation, not only at
  consolidation. Each successful task's executed plan is cached (success-only, so a failed plan never
  poisons future tasks). STM stays short-term via its bounded capacity (cap 24, LFU/LRU) and high
  short-circuit threshold (0.85); one-offs are evicted unused, only true near-duplicates fire. This is
  the architecture's **continual** arm (memory accumulates within a deployment).

Also active in `--improved`: **verified replay** (on a high-confidence EM hit, adapt the cached action
sequence in one LLM call and execute it with per-action tool-signal verification, aborting and
recovering on any failure — never blind), an output-path **convention**, a **self-verification** pass,
and a **malformed-action recovery** aid.

---

## 4. Result

| Arm | Accuracy | Injected-memory tokens | tok/call |
|---|---|---|---|
| no-memory | 63/152 = 0.414 | 0 | — |
| retrieve-all | 71/152 = 0.467 | 734,295 | 335 |
| **PCCR improved (this branch)** | **72/152 = 0.474** | **628,570** | **146** |

By pattern, the substantive story: **`multi_app` recovers 0.179 → 0.244 → 0.295** (now matches
no-memory and beats retrieve-all's 0.269) and **`doc_process` reaches 0.680** (> retrieve-all 0.600).
STM short-circuit hits 12 (vs 8 frozen). Full breakdown: `officebench_eval/FULL152_IMPROVED_RESULTS.md`;
per-task records: `officebench_eval/full152_improved_final.json`.

> **Honest cost caveat.** The 628K is *injected-memory* tokens only. The tightened gate (F3) roughly
> *doubles* LLM calls (28.3 vs 14.4 / task, almost all on L3 retries). Because each call re-sends the
> full prompt (system + observation history + memory) and we instrument only the memory slice, the
> **net** API-token bill is unresolved and may be higher. The defensible claim is **parity accuracy
> with multi_app repaired**, plus a 5× cut in per-call memory injection — *not* a net total-cost win.

---

## 5. File map

```
memory_manager/                    # the PCCR architecture (standalone, no harness deps)
  router.py        manager.py      # ρ-gate decision + ten-phase lifecycle
  stores/          types.py        # WM/PM/STM/EM/SM/ENT stores + dataclasses
  config.py        online_utility.py
  task_queue.py    parallel.py     # AsyncTaskQueue + the 7-layer parallel wrappers

officebench_eval/                  # the OfficeBench harness + F1–F4
  real_mem.py                      # cascade wiring; F1 sanitiser, F2 split, F4 cache_success
  pccr_policy.py                   # agent policy; F2 schedule, F3 completion gate, verified replay
  real_arch.py  gate.py            # FAISS EM store + ρ-gate plumbing
  runner.py                        # single-task run (Docker env + agent loop)
  run_t1_parallel.py               # the N=4 driver (--improved enables F1–F4 + the rest)
  cerebras_llm.py                  # gpt-oss-120b client (multi-key round-robin, 429 handling)
  em_bank.json                     # the memory bank (62 curated procedures)
  pm_curated.json                  # schema-valid curated PM rules (one per pattern)
  split.json patterns.json         # test split + per-task pattern labels
  full152.json missing27.json      # task lists (the run completed in two N=4 segments)
  full152_improved_final.json      # per-task results for 72/152
  FULL152_IMPROVED_RESULTS.md      # the result tables
  traces/par_full152/              # N=4 traces, 125-task segment (.json full record + .txt steps)
  traces/par_missing27/            # N=4 traces, 27-task completion segment

requirements.txt  cerebras.env.example  REPRODUCE.md
```

**External dependency (not in this repo):** the **OfficeBench** benchmark + Docker harness — clone it
into `OfficeBench/` and provide a Docker daemon with the `officebench` image. See `REPRODUCE.md`.

---

## 6. Reproduce

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp cerebras.env.example cerebras.env        # fill in CEREBRAS_KEY_1..N
git clone <officebench-upstream> OfficeBench

set -a; source cerebras.env; set +a
python -m officebench_eval.run_t1_parallel --improved \
    --tasks-file officebench_eval/full152.json --tag full152 --concurrency 4
```

Backbone nondeterminism means expect ~70–74/152; **72** is the recorded run, with its traces preserved
here. If a worker hangs (Docker OOM under N=4 load), kill, restart the daemon/containers, and re-launch
the same `--tag` — completed tasks are skipped.
