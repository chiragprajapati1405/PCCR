# Concurrent Memory Manager — 20-task one-container run

A general **2-store memory manager** for multi-agent LLM systems, run against real OfficeBench tasks
**in a single Docker container** with per-task isolation. Measures **per-task latency** and **total
parallel completion time** for N tasks sharing one memory manager and one container.

## Result (20 real OfficeBench test tasks, one container, concurrency 20)

**Total parallel completion time: 13.7s** (≈ the single longest task, 13.1s → near-optimal scaling;
the memory manager adds no serialization). Per-task latency 6.3–13.1s (avg 8.9s). Full per-task
numbers and traces are under [`one_traces/`](one_traces/) (`_summary.json` + one `.txt`/`.json` per task).

## Architecture

```
   N tasks  ──►  ONE MemoryManager  ──►  PM (read + write)      + Tool Memory (read-only)
                                          vector search, lock-free  per-app how-to, O(1)
                        │
                        └──►  ONE Docker container, per-task workdir /testbed/run_<id>
```

- **PM** (`procedural_memory.py`) — append-only log of successful **full trajectories** (plan +
  sub-agent steps). Retrieval is **vector search** (sentence-transformers `all-MiniLM-L6-v2`, cosine
  top-k). Lock-free (snapshot reads + atomic appends). Persisted to `pm_store.json` so it warms across
  runs.
- **Tool Memory** (`tool_memory.py`) — read-only, per-app tool manual (schema/format), looked up by app
  name (O(1), no locks). Injected into the sub-agent prompt.
- **WM** — each task's private in-progress step buffer; discarded on failure (success-gate: only
  successful trajectories are appended to PM).
- **Concurrency** — N tasks run in parallel; within a task, sub-agents run in parallel waves
  (read wave ∥ act wave). All share one container, isolated by `TESTBED_ROOT`/path-rewrite.

## Files

| File | Role |
|---|---|
| `manager.py` | the single `MemoryManager` (orchestrator↔PM, sub-agent↔Tool Memory) |
| `procedural_memory.py` | PM: append-only, vector search, lock-free |
| `tool_memory.py` | Tool Memory: read-only per-app tool manual |
| `run_real_one.py` | the runner: real OfficeBench tasks, one container, per-task isolation, traces + latency |
| `eval_real.py` | deferred eval: score the saved outputs against OfficeBench evaluators |
| `patch_officebench.py` | **required** — patches OfficeBench apps for one-container isolation (`TESTBED_ROOT`) |
| `one_traces/` | the 20-task run: per-task plan, steps, latency (`_summary.json` + per-task files) |
| `BUILD_PLAN.md` | design notes |

## Reproduce

```bash
# 1. deps
python -m venv .venv && source .venv/bin/activate
pip install sentence-transformers docker openai fire icalendar openpyxl python-docx

# 2. clone OfficeBench and APPLY THE PATCH (required for one-container isolation)
git clone https://github.com/zlwang-cs/OfficeBench.git
python -m concurrent_mm.patch_officebench

# 3. Docker (one container is created automatically as 'cmm-onebox')
#    macOS/colima: export DOCKER_HOST=unix://$HOME/.colima/default/docker.sock

# 4. Cerebras API keys (gpt-oss-120b)
#    export CEREBRAS_KEY_1=...   (one or more)

# 5. run 20 tasks in parallel through ONE container
python -m concurrent_mm.run_real_one --n 20 --concurrency 20
```

Outputs: per-task latency table + total wall-clock to stdout; traces to `one_traces/`; task outputs to
`one_results/<task>/testbed/` (for deferred eval via `python -m concurrent_mm.eval_real`).

## Notes
- Accuracy is **not** the focus here — this run measures **latency + memory-access concurrency**. The
  memory retrieval (PM vector search, Tool Memory lookup) is verified working (`used_pm=True`); task
  accuracy is a separate agent-quality effort.
- The OfficeBench patch is **backward-compatible**: apps default to `/testbed` when `TESTBED_ROOT` is
  unset, so normal single-task-per-container runs are unaffected.
