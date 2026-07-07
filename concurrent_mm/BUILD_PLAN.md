# Concurrent Memory Manager — Build Plan

**Branch:** `concurrent-memory-manager`
**Goal:** measure whether N concurrent tasks access shared memory *efficiently*, and the *latency* of
completing 100 parallel tasks — through **one** Docker container, deferring accuracy.

---

## ★ FINAL DESIGN (converged) ★

Three memory levels — only two are shared:

```
   WM  (per-task, PRIVATE)   in-progress step buffer · no lock · discarded on failure
   PM  (shared)              successful FULL trajectories (plan + sub-agent steps)
                             append-only · lock-free · retrieved by embedding similarity
   TOOL MEMORY (shared)      how to call each tool (schema/format) · READ-ONLY
```

Rules that make it lock-free + consistent + general:
- **No classifier, no gate, no patterns** — general to any benchmark (PM retrieval = similarity).
- **2 concurrency dimensions:** N parallel tasks × M parallel sub-agents each.
  → Tool Memory takes **N×M** reads (read-only, zero contention);
  → PM takes **N** appends (atomic, once per task).
- **Success-gate:** a task buffers its steps in WM; on SUCCESS the orchestrator appends the full
  trajectory to PM once; on failure WM is discarded (PM never sees a failing plan).
- **Lock-free PM:** append-only + immutable entries + snapshot reads (relax freshness → benign staleness).

## ★ SIMPLE STEP PLAN ★

```
   STEP 1  Tool Memory   — read-only dict {app: {action: usage}}          (from ARG_SCHEMA/ACTION_HINTS)
   STEP 2  PM            — append-only log; read=snapshot+similarity, write=atomic append (lock-free)
   STEP 3  WM            — a per-task local list (the step buffer); no store, just a variable
   STEP 4  MemoryManager — read_pm(text) / write_pm(traj) / read_tool_memory(app) + counters
   STEP 5  Executor      — ModeledExecutor: run_action = sleep(0.516s)     (no Docker, no API)
   STEP 6  Task pipeline — read PM → plan → gather(M sub-agents) → success? append WM→PM : discard
   STEP 7  Benchmark     — fire N∈{1,10,50,100} tasks; measure wall / throughput / PM-wait / tool-read
   STEP 8  (later) swap ModeledExecutor → DockerExecutor (1 container, per-task workdirs), save outputs
```

Deliverable of steps 1–7: the scaling table (runs in seconds, zero cost) proving lock-free memory
access. Step 8 = real latency. Accuracy deferred (optional Phase 4).

---

## 0. Goal & Non-Goals

**We ARE measuring:**
- **Memory-access efficiency** under concurrency: does the read-write store (PM) bottleneck as N grows,
  while the read-only store (Tool Memory) stays contention-free?
- **Latency / throughput:** wall-clock to complete 10 / 50 / 100 tasks in parallel; per-task latency;
  scaling curve vs concurrency.

**We are NOT (yet) measuring:**
- Accuracy. Each task's output is saved to its own directory; evaluation is deferred and optional.

**Design pillars (from the mentor):**
- **2 stores only:** `ProceduralMemory` (PM, read+write, stores successful trajectories) and
  `ToolMemory` (read-only, per-app tool-use knowledge for sub-agents).
- **1 memory manager** all N tasks call into.
- **1 Docker container** for all N tasks, isolated by per-task working directories.

---

## 1. Architecture

```
   task_0  task_1  ...  task_99          (N async coroutines)
      │       │            │
      └───────┴─────┬──────┘
                    ▼
         ┌─────────────────────────────┐
         │   MemoryManager             │   ← the concurrency contract lives here
         │                             │
         │   PM  (read + WRITE)  ▓lock │   orchestrator-level, successful trajectories
         │   Tool Memory (READ only)   │   sub-agent-level, lock-free shared
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌─────────────────────────────┐
         │  Executor (pluggable)       │
         │  • ModeledExecutor (sleep)  │   ← Phase 1-2: no Docker, no API, clean numbers
         │  • DockerExecutor (1 cont.) │   ← Phase 3: real tools, per-task workdir
         └─────────────────────────────┘
                    │
         per-task output dir  →  saved for OPTIONAL later eval
```

**Why the executor is pluggable:** the *research contribution* is the memory manager's concurrency,
which is pure software. We measure it cleanly with a **ModeledExecutor** (LLM latency = a measured
`sleep(0.516s)`, zero API cost, no rate limits). Only Phase 3 swaps in the real **DockerExecutor**.

---

## 2. Component Design

### 2.1  `ToolMemory`  (read-only)  — `concurrent_mm/tool_memory.py`
- **Content:** *how to USE each tool* — NOT past episodes. Keyed by `(app, action)`:
  ```
  {"calendar.create_event": {args: [user, summary, time_start, time_end],
                             format: "times = YYYY-MM-DD HH:MM:SS",
                             note: "..."},
   "email.send_email":      {args: [sender, recipient, subject, content],
                             note: "sender/recipient are names, not addresses"}, ...}
  ```
  It is a **tool manual**: the schema + format rules + gotchas a sub-agent needs to emit a correct
  call. Static — a tool's usage doesn't change. (Raw material already exists in
  `officebench_eval/subagent_parallel.py`: `VALID_ACTIONS` + `ARG_SCHEMA` + `ACTION_HINTS`.)
- **API:**
  - `get(app: str) -> dict[action -> usage]`  — a keyed lookup; the sub-agent reads its app's tool spec.
- **Concurrency:** **none needed, ever.** Read-only + static → N sub-agents read freely, no locks, no
  writes. This is the "cheap, scales perfectly" store; the benchmark should show flat read latency
  regardless of N. (This is the whole point of the read/write split.)
- **NOT stored here:** past subtask episodes / `subtask_memories`. Task-level history lives in PM as
  successful trajectories; tool memory is pure how-to.

### 2.2  `ProceduralMemory`  (read + WRITE)  — `concurrent_mm/procedural_memory.py`
- **Content:** ONE append-only log of successful **FULL trajectories** (no pattern key — general;
  retrieval by embedding similarity over task text). Each Trajectory =
  orchestrator plan **+** the sub-agent action steps that succeeded:
  ```
  Trajectory{ task, pattern, plan:[...],
              subagent_steps:[ {agent, action_json, obs}, ... ] }   ← sub-agent memory lives HERE
  ```
  Grows as tasks succeed. The orchestrator reads a matching trajectory to reuse BOTH the plan and the
  concrete sub-agent actions. (Sub-agent execution memory lives here, NOT in tool memory — tool memory
  is static how-to only.)
- **Benchmark note:** because a written entry is a whole trajectory (not a one-line plan), PM writes are
  larger → longer lock-hold → sharper contention curve as N grows. Good — it stresses the read-write
  store, which is the point of the study.
- **API:**
  - `read(key, query, k=3) -> list[Trajectory]`  — orchestrator reads relevant past trajectories.
  - `write(key, trajectory)`  — orchestrator appends a successful trajectory.
- **Concurrency:** **the interesting part.** N orchestrators read+write concurrently. Three lock
  strategies to benchmark (pick one as default, keep others behind a flag):
  - **(a) per-key lock** — lock PM per key; different keys never block. *Default — simplest.*
  - **(b) read-write lock** — many concurrent readers, exclusive writer per key.
  - **(c) snapshot-read + staged-merge** — readers see a frozen snapshot, writers stage + merge.
- **Instrumentation:** every `read`/`write` records `lock_wait_s` so we can plot contention vs N.

### 2.3  `MemoryManager`  — `concurrent_mm/manager.py`
The single object all tasks call. Thin API matching the two roles:
- `orchestrator_read_pm(key, query)` / `orchestrator_write_pm(key, trajectory)`  → PM (locked)
- `subagent_read_tool_memory(app, subtask)`  → Tool Memory (lock-free)
- Holds counters: total PM reads/writes, total tool reads, cumulative lock-wait, per-store latency.

### 2.4  Executors  — `concurrent_mm/executors.py`
- `ModeledExecutor(call_latency=0.516)` — `async run_action(...)` = `await asyncio.sleep(latency)`.
  Represents one LLM/tool step. **No Docker, no API.**
- `DockerExecutor(container="cmm-bench")` — real OfficeBench tool exec:
  - one container; per-task `env.workdir = /testbed/run_<task_id>/` (copy testbed in on setup).
  - `run_action(app, action, workdir)` → `container.exec_run(cmd, workdir=workdir)`.
  - save each task's output dir → `results/run_<task_id>/` for deferred eval.

### 2.5  The task model  — `concurrent_mm/task.py`
A single task's async pipeline. NO classifier — PM retrieval is by embedding similarity (general,
benchmark-agnostic). Sub-agents run in PARALLEL (2nd dimension):
```
async def run_task(spec, mm, executor):
    wm = []                                                  # WM: per-task PRIVATE buffer (no lock)
    # ORCHESTRATOR phase
    past = mm.orchestrator_read_pm(spec.text)                # read PM by SIMILARITY (no pattern key)
    plan = decide_plan(spec, past)                           # (modeled: fixed; docker: LLM)
    # SUB-AGENT phase — PARALLEL sub-agents (2nd dimension)
    async def run_sub(sub):
        usage = mm.subagent_read_tool_memory(sub.app)        # read Tool Memory (lock-free)
        return await executor.run_action(sub, usage)         # execute (sleep OR docker)
    for wave in plan.waves:                                  # dependency waves
        results = await asyncio.gather(*[run_sub(s) for s in wave])   # M sub-agents at once
        wm.extend(results)                                   # accumulate steps LOCALLY
    # SUCCESS-GATE — commit only if the task succeeded
    if success(wm):                                          # known only NOW, at the end
        mm.orchestrator_write_pm(Trajectory(spec, plan, wm))  # 1 atomic append
    # else: discard wm (PM never sees a failing plan)
    return TaskResult(latency, lock_waits, ...)
```
**Two concurrency levels:** N tasks (dim 1) × M sub-agents each (dim 2) → tool memory sees **N×M**
concurrent reads (read-only, no contention); PM sees **N** appends (atomic). Sub-agents never write PM.

### 2.6  The benchmark harness  — `concurrent_mm/bench_concurrent.py`
- Fire N tasks via `asyncio.gather` (or a bounded semaphore for the Docker case).
- Sweep concurrency: **N ∈ {1, 10, 50, 100}**.
- Collect + print:
  ```
  N     wall_s   throughput(tasks/s)   avg PM lock-wait   avg tool-read   speedup vs N=1
  1     ...
  10    ...
  50    ...
  100   ...
  ```
- Emit `results.json` + a readable table. **This is the deliverable graph.**

---

## 3. What we prove

1. **Latency/throughput:** the scaling curve — how wall-clock for 100 tasks drops with concurrency,
   and where it saturates (the memory manager's ceiling, not the provider's).
2. **Memory efficiency:** PM lock-wait rises with N (read-write contention) while Tool-Memory read time
   stays flat (read-only) — the quantitative case that the **role-split** (r/w PM + r/o tool mem) is
   the right design: put the shared knowledge sub-agents need in a lock-free read-only store, and only
   pay concurrency cost on the small read-write PM.
3. **Lock-strategy comparison (optional):** per-key vs read-write vs snapshot — which keeps PM
   contention lowest at N=100.

---

## 4. Phased Build (milestones)

**Phase 1 — Core + modeled harness (no Docker, no API).**  ← *start here, runs in seconds, zero cost*
- `tool_memory.py`, `procedural_memory.py` (per-key lock), `manager.py`, `executors.py:ModeledExecutor`,
  `task.py`, `bench_concurrent.py`.
- Deliverable: the N ∈ {1,10,50,100} scaling table on modeled latency. Proves the concurrency design.

**Phase 2 — Lock-strategy sweep + instrumentation polish.**
- Add read-write lock + snapshot-merge behind a flag; compare PM contention at N=100.
- Deliverable: the "which lock strategy scales best" table.

**Phase 3 — Real one-container Docker executor + deferred output saving.**
- `executors.py:DockerExecutor` with per-task workdir isolation; save `results/run_<id>/`.
- Real-LLM anchor run at modest N (rate-limit aware).
- Deliverable: real end-to-end latency for a real 100-task parallel run through one container.

**Phase 4 (optional, later) — Deferred accuracy.**
- Point OfficeBench evaluators at each saved `results/run_<id>/`; produce the accuracy table if wanted.

---

## 5. File layout

```
concurrent_mm/
├── BUILD_PLAN.md          (this file)
├── tool_memory.py         read-only store
├── procedural_memory.py   read-write store + lock strategies + instrumentation
├── manager.py             the single MemoryManager API
├── executors.py           ModeledExecutor + DockerExecutor
├── task.py                one task's orchestrator→subagent→writeback pipeline
├── bench_concurrent.py    the N-sweep harness → results.json + table
└── build_tool_memory.py   one-off: VALID_ACTIONS + ARG_SCHEMA + ACTION_HINTS → tool_memory.json
```

---

## 6. Open decisions (confirm before/while building)
1. **PM key:** what do orchestrators key PM reads/writes on — task `pattern` (5 buckets, more sharing/
   contention) or a per-task signature (less contention, less reuse)? *Default: pattern.*
2. **Default lock strategy:** per-key lock (simplest). Others behind a flag.
3. **Run mode order:** Phase 1 modeled first (recommended), Docker in Phase 3.
4. **Task source:** reuse `full152.json` specs (real task texts) even in modeled mode, so numbers are
   grounded. *Default: yes.*
```
```



 The simple steps

  ┌──────┬───────────────────────────────────────────────────────────────────────────────────┬──────────────────────┐
  │ Step │                                   What we build                                   │     Key property     │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 1    │ Tool Memory — {app: {action: usage}}                                              │ read-only, lock-free │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 2    │ PM — append-only log; read = snapshot + similarity, write = atomic append         │ lock-free            │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 3    │ WM — a per-task local list (step buffer)                                          │ private, no lock     │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 4    │ MemoryManager — read_pm / write_pm / read_tool_memory + counters                  │ the front door       │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 5    │ Executor — modeled: run_action = sleep(0.516s)                                    │ no Docker, no API    │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 6    │ Task pipeline — read PM → plan → gather(M sub-agents) → success? WM→PM : discard  │ the two dimensions   │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 7    │ Benchmark — fire N∈{1,10,50,100}; measure wall / throughput / PM-wait / tool-read │ the result           │
  ├──────┼───────────────────────────────────────────────────────────────────────────────────┼──────────────────────┤
  │ 8    │ (later) swap in DockerExecutor — 1 container, per-task workdirs, save outputs     │ real latency         │
  └──────┴───────────────────────────────────────────────────────────────────────────────────┴──────────────────────┘