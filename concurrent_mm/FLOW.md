# Concurrent Memory Manager — Detailed Flow Chart

Every step of the one-container run (`run_real_one.py`), from startup to result. Each box cites
`file.py:function`. Three memory levels: **PM** (shared, read+write, vector search), **Tool Memory**
(shared, read-only, per-app), **WM** (per-task, private).

```
LEGEND   ┌─┐ step    <?> decision   ║ ║ store   ═► data   ──► control   ∥ parallel
```

---

## PART 1 — STARTUP  (main_async, runs once)

```
   python -m concurrent_mm.run_real_one --n 20 --concurrency 20
        │
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ S1. resolve pm_bank path (abspath) BEFORE chdir              │  run_real_one.main_async
   │ S2. _lazy_imports() -> _setup() chdir into OfficeBench;      │  imports CerebrasLLM, subagent_parallel
   │     import helpers (VALID_ACTIONS, plan helpers, evaluators) │  helpers, utils.evaluate
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ S3. embedder = SentenceTransformer('all-MiniLM-L6-v2')       │  384-d, normalized
   │ S4. pm = ProceduralMemory(strategy=lockfree, embedder)       │  procedural_memory.py
   │ S5. SEED PM from pm_bank.json (62 trajectories):             │
   │       for r in bank:                                         │
   │         traj = Trajectory(r.task, r.plan, r.steps)           │
   │         traj.embedding = embed(r.task)   ═════════════════►  ║ PM ║  (62 vectors)
   │         pm._log.append(traj)                                 │
   │ S6. _load_pm(pm) — also warm from pm_store.json (persistent) │
   │ S7. mm = MemoryManager(default_tool_memory(), pm)            │  manager.py
   │       └─ Tool Memory built from VALID_ACTIONS/ARG_SCHEMA ══► ║ TOOL MEM ║ (7 apps, read-only)
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ S8. box = OneBox()  — create/reuse ONE container 'cmm-onebox'│  run_real_one.OneBox
   │       └─ docker cp apps -> container:/apps   (ONCE)          │
   │ S9. sem = Semaphore(concurrency);  t_start = now             │
   └───────────────────────────┬─────────────────────────────────┘
                               ▼
                    asyncio.gather( run(it) for it in the N tasks )     ◄══ DIMENSION 1: N parallel tasks
                               │
                               ▼  (each task: acquire sem -> run_one_task)
```

---

## PART 2 — ONE TASK  (run_one_task — runs N of these concurrently in the one container)

```
   run_one_task(it, box, mm, model, ob):
        │
        ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T1. ns = box.setup_task(tid,sid)                             │  ISOLATION
   │       mkdir /testbed/run_<tid>_<sid>                          │
   │       docker cp tasks/<tid>/testbed/. -> container:ns         │  each task its OWN namespace
   │ T2. llm = CerebrasLLM(); t0 = now; wm = []   ◄══════════════  ║ WM ║ (per-task, private, no lock)
   │ T3. files = ls ns/data  ->  "/testbed/data/<name>, ..."      │  FULL paths (exact case)
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T4. ORCHESTRATOR reads PM  (VECTOR SEARCH)          → PART 3   │
   │     pm_hits = mm.orchestrator_read_pm_sync(task, k=1)         │
   │     pm_hit  = pm_hits[0]  (most similar past trajectory)      │  used_pm = pm_hit is not None
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T5. PLAN  _plan(task, files, pm_hit, llm)   [1 LLM call]      │  run_real_one._plan
   │     prompt = "A SIMILAR PAST TASK succeeded with this plan —  │  ◄── RETRIEVED plan injected here
   │              ADAPT it: <pm_hit.plan>"                         │
   │            + "use ONLY these files: <files>"                  │  ◄── real filenames (no hallucination)
   │            + "TASK: <task>"                                   │
   │     -> [ Delegation(app, subtask), ... ]                      │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T6. split by category (dependency):                          │  categorize()
   │       reads = [d for d if categorize == "read"]              │
   │       acts  = [d for d if categorize != "read"]              │
   │     blackboard = { files, data:"" }                          │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T7. READ WAVE — sub-agents in PARALLEL                        │  DIMENSION 2 (within a task)
   │       await gather( _run_subagent(d) for d in reads )   ∥∥∥   │  → PART 4
   │       wm += steps;  blackboard.data += observations           │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T8. FAN-OUT EXPANSION  _expand_fanout(acts, blackboard.data)  │  run_real_one._expand_fanout
   │     "create event for EACH participant"                       │
   │       -> parse item list from read data (structural grid)     │  _extract_items_structural
   │       -> ONE Delegation per item (create for Alice, Bob, ...) │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T9. ACT WAVE — sub-agents in PARALLEL                         │  DIMENSION 2
   │       await gather( _run_subagent(d) for d in acts )    ∥∥∥   │  → PART 4
   │       wm += steps                                             │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T10. SAVE + EVAL  box.save_and_eval(tid,sid,ns,eval_spec)    │
   │       docker cp ns/. -> one_results/<task>/testbed  (persist) │  deferred-eval filesystem
   │       for predicate in eval_spec: run evaluate_*()   -> ok,fp │  utils.evaluate
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
                        <? task succeeded (ok) ?>
                          │yes                │no
                          ▼                   ▼
   ┌──────────────────────────────────┐   (discard WM — PM never sees a failing plan)
   │ T11. SUCCESS-GATE — WRITE PM      │
   │  Trajectory(task, plan, wm)       │
   │  mm.orchestrator_write_pm_sync()  ═══════════════════════════► ║ PM ║ (embed + atomic append)
   └───────────────────────────┬──────┘
                               ▼
   ┌──────────────────────────────────────────────────────────────┐
   │ T12. return { latency, success, used_pm, pm_hit_task, calls,  │
   │               steps, plan, trajectory }                       │
   │      -> _save_trace()  ->  one_traces/<task>.txt/.json        │
   └──────────────────────────────────────────────────────────────┘
```

---

## PART 3 — PM READ (vector search)  procedural_memory.read_sync / _score

```
   orchestrator_read_pm_sync(task_text, k=1):
        │
        ▼
   snapshot_len = len(PM._log)          ◄── consistent snapshot, NO lock (lock-free read)
        │
        ▼
   qvec = embed(task_text)              ◄── embed the query (thread-safe under _elock)
        │
        ▼
   for each traj in PM._log[:snapshot_len]:
        score = cosine(qvec, traj.embedding)      ◄── dot product (normalized vectors)
        │
        ▼
   sort by score desc -> return top-k trajectories
        │
        ▼
   pm_hit = the most similar past successful trajectory  ═══► used in T5 (plan injection)
```

---

## PART 4 — ONE SUB-AGENT  _run_subagent (runs in parallel within a wave)

```
   _run_subagent(box, llm, d, mm, ns, blackboard, ob):
        │
        ▼
   ┌──────────────────────────────────────────────────────────┐
   │ A1. READ TOOL MEMORY (per app)                            │
   │     usage = mm.subagent_read_tool_memory(d.app)   ◄══════ ║ TOOL MEM ║  (O(1), read-only, no lock)
   │       -> {create_event:{args,format}, ...}  for THIS app  │
   └───────────────────────┬──────────────────────────────────┘
                           ▼   loop up to 6 steps:
   ┌──────────────────────────────────────────────────────────┐
   │ A2. prompt = _subagent_prompt(app, subtask, usage,        │
   │        last_obs, blackboard)                              │
   │      = "SUB-TASK + available files + DATA so far +        │  ◄── real files + read data (blackboard)
   │         tool usage (how to call it)"                      │  ◄── Tool Memory usage
   │ A3. action = parse( llm.generate(prompt) )   [1 LLM call] │  (to_thread — overlaps siblings)
   │      if action in {done,finish,none}: break               │
   │ A4. obs = box.exec_action(app, action, ns)      → PART 5  │  (to_thread)
   │ A5. wm-step = {agent, action, obs}                        │  ═══► appended to WM
   │ A6. sig = json(action)                                    │
   │      if sig already seen: break   (repeated action guard) │  ◄── no 6x-identical loops
   │      if action ok AND not fan-out: break  (single done)   │
   └──────────────────────────────────────────────────────────┘
        │
        ▼
   return steps  ═══► merged into WM by the wave
```

---

## PART 5 — ONE-CONTAINER ISOLATION  box.exec_action

```
   exec_action(app, action, ns):    ns = /testbed/run_<id>
        │
        ▼
   command = apps.AVAILABLE_ACTIONS[app][act].construct_action(ns, args)
        │   e.g.  "python3 /apps/excel_app/excel_read_file.py --file_path /testbed/data/x.xlsx"
        ▼
   command = command.replace("/testbed", ns)        ◄── FILE apps: rewrite arg paths -> namespace
        │   ->  "... --file_path /testbed/run_<id>/data/x.xlsx"   ( /apps stays )
        ▼
   container.exec_run(["/bin/bash","-c",command],
                      workdir=ns,
                      environment={"TESTBED_ROOT": ns})  ◄── calendar/email: patched apps read this
        │                                                     (patch_officebench.py)
        ▼
   return observation
```

Two namespacing mechanisms (why isolation works in ONE container):
- **File apps** (excel/word/pdf) — path is a command ARG → **string rewrite** `/testbed → ns`.
- **calendar/email** — path hardcoded in the app → **`TESTBED_ROOT` env var** (patched apps honor it).

---

## PART 6 — THE STORES (summary)

```
   ║ PM ║  procedural_memory.py   READ + WRITE   shared
      read  = VECTOR SEARCH (embed query -> cosine top-k), lock-free snapshot        (PART 3)
      write = success-gated append (embed + list.append), lock-free/atomic           (T11)
      seed  = pm_bank.json (62 past trajectories) at startup                         (S5)

   ║ TOOL MEM ║  tool_memory.py    READ-ONLY      shared
      read  = get(app) -> {action: usage}, O(1) dict lookup, no lock                 (A1)

   ║ WM ║  a Python list            per-task      PRIVATE
      accumulates sub-agent steps; discarded on failure; -> PM trajectory on success (T2,T5,T11)
```

---

## PART 7 — CONCURRENCY MODEL

```
   DIMENSION 1  (across tasks):   asyncio.gather + Semaphore(N)   — N tasks, ONE container
   DIMENSION 2  (within a task):  read wave ∥ , act wave ∥        — sub-agents via gather
   MEMORY:      PM lock-free (snapshot read + atomic append) · Tool Mem read-only · WM private
   ISOLATION:   per-task namespace /testbed/run_<id> (rewrite + TESTBED_ROOT)
   RESULT:      per-task latency + TOTAL parallel wall (≈ longest task) + used_pm + traces
```
