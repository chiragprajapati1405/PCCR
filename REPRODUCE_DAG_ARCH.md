# PCCR + PERF + 2D + DAG re-plan — reproducible architecture (65/152 @ 4.90M)

This branch (`pccr-perf-2d-dag-repro`) pins the exact architecture and result:

**Measured result (full 152, N=4, gpt-oss-120b via Cerebras):**
| Accuracy | Real tokens | tok/call | calls | injected | Real N=4 wall |
|---|---|---|---|---|---|
| **65/152** (L1 26/47, L2 24/48, L3 15/57) | **4.90M** | 1908 | 2568 | 471K | **36.2 min** |

Result file: `officebench_eval/par_dag_io2_progress.json` · Traces: `officebench_eval/traces/par_dag_io2/` (152 `.json`+`.txt` pairs).

---

## The architecture (what runs per task)
1. **Task-level parallelism (N=4):** `AsyncTaskQueue` runs 4 task pipelines concurrently, each on its own Docker container `ob-test-0..3`.
2. **ρ-gate / memory injected ONCE per task** (`_run_2d_dag` → `real_mem.retrieve`), seeded into the shared blackboard (diagram Phase 3; inject-once).
3. **DAG-first routing** (`subagent_parallel.dag_decision`): a single planner pass + dependency DAG + structural (file-disjoint) independence test decides if the task has genuine parallel work.
4. **Parallel → orchestrator RE-PLAN loop** (`run_task_2d(replan=True)`): each round the orchestrator (`plan_next_group`) emits the next group → `DependencyAnalyzer` waves it → `ParallelExecutor` runs the wave with **PCCR-agent leaves** (each a policy on its own env attached to the shared container) → merge into the blackboard → repeat until FINISH (cap `MAX_REPLAN_ROUNDS=4`).
5. **Sequential chains → single PCCR agent** (`run_task`) with **plan-then-execute batching** (`--plan --batch-size 3`) — the accuracy floor.

Three architecture-preserving efficiency passes vs. the first build (64/152 @ 8.10M): **inject-once** (1350K→471K), **round-cap** 8→4, **fallback batching**.

## Exact command
```bash
set -a; source cerebras.env; set +a       # 29 Cerebras API keys (gitignored)
export CEREBRAS_MAX_INFLIGHT=10            # global in-flight cap (avoids org-wide 429 storm)
.venv/bin/python -m officebench_eval.run_t1_parallel \
    --improved --twod --plan --batch-size 3 \
    --concurrency 4 \
    --tasks-file officebench_eval/full152.json --tag dag_io2
```
Watchdog (self-heals colima docker-socket death + resumes): `officebench_eval/watchdog_dagio2.sh`.

## Key source files (all on this branch)
- `officebench_eval/run_t1_parallel.py` — N=4 harness; `--twod` DAG router (`_run_2d_dag`, `_make_pccr_leaf`); batching on fallback.
- `officebench_eval/subagent_parallel.py` — `dag_decision`, `run_task_2d` (re-plan loop, inject-once `mem_seed`, structural independence `_refine_wave`, `MAX_REPLAN_ROUNDS`), `plan_next_group`.
- `officebench_eval/pccr_policy.py` — `make_pccr_policy` (ρ-gate, EM injection, slim-history).
- `officebench_eval/real_mem.py`, `real_arch.py`, `gate.py` — the memory cascade + calibration.
- `officebench_eval/cerebras_llm.py` — backbone + `CEREBRAS_MAX_INFLIGHT` global semaphore.
- `officebench_eval/runner.py` — `run_task` (single PCCR agent, fallback).
- `memory_manager/parallel.py` — `DependencyAnalyzer` (DAG), `ParallelExecutor` (snapshot→gather→merge).

## Data / config (force-added to this branch for reproducibility)
- `officebench_eval/em_bank.json` — the FAISS EM procedure bank (memory).
- `officebench_eval/calibration/` — ρ-gate cost/utility calibration.
- `officebench_eval/patterns.json`, `split.json`, `full152.json`, `pm_curated.json` — task patterns, the 152-task test set, curated PM rules.

## External dependencies (NOT committed — set up locally)
- **OfficeBench** clone at `./OfficeBench/` (tasks + apps + intercode env) — the benchmark itself.
- **Docker image** `officebench` (built per OfficeBench README) + Colima running.
- **`cerebras.env`** — `CEREBRAS_KEY_1..N` API keys (gitignored).
- Python venv `.venv/` with the OfficeBench requirements + `openai`, `faiss`.

## Recompute the aggregate from the committed result
```python
import json
d = json.load(open('officebench_eval/par_dag_io2_progress.json'))
t = [v for v in d.values() if 'level' in v]
p = sum(v['success'] for v in t)
real = sum(v['prompt_tokens']+v['completion_tokens'] for v in t)
calls = sum(v['llm_calls'] for v in t)
t1 = [v['worker_t1'] for v in t if v.get('worker_t1') is not None]
print(f'{p}/152  {real/1e6:.2f}M  {real/calls:.0f} tok/call  {calls} calls  '
      f'REAL N=4 wall {max(t1)/60:.1f} min')   # -> 65/152 4.90M 1908 tok/call 2568 calls 36.2 min
```
The Wall is the **true concurrent elapsed** (`max(worker_t1)`), not an estimate.
