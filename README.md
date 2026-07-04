# PCCR: Phase-Conditioned Cascading Memory Routing for Multi-Agent LLM Systems

> ## 📌 This branch: `pccr-perf-2d-dag-repro` — reproducible DAG re-plan architecture
> A self-contained snapshot of the **PCCR + PERF + 2D + DAG re-plan** architecture with its
> **traces and memory bank committed**, so the result reproduces.
>
> **Result (full 152, N=4, gpt-oss-120b):** **65/152 · 4.90 M real tokens · 2568 calls · 471 K injected · 36.2 min real N=4 wall** (L1 26/47, L2 24/48, L3 15/57).
>
> Per task at N=4: inject-once ρ-gate/memory (once/task) → DAG-first routing (structural
> file-disjoint independence) → orchestrator RE-PLAN loop with parallel PCCR-agent leaves →
> sequential chains fall back to a batched single PCCR agent. Three efficiency passes
> (inject-once, round-cap 8→4, fallback batching) cut it from 64/152 @ 8.10 M to 65 @ 4.90 M,
> architecture unchanged.
>
> - **Reproduce / source-file map:** [`REPRODUCE_DAG_ARCH.md`](REPRODUCE_DAG_ARCH.md)
> - **Traces:** `officebench_eval/traces/par_dag_io2/` (152 `.json`+`.txt` pairs) · **Result:** `officebench_eval/par_dag_io2_progress.json` · **Memory bank:** `officebench_eval/em_bank.json` + `officebench_eval/calibration/`
> - **Run:** `CEREBRAS_MAX_INFLIGHT=10 python -m officebench_eval.run_t1_parallel --improved --twod --plan --batch-size 3 --concurrency 4 --tasks-file officebench_eval/full152.json --tag dag_io2`
>
> The full PCCR project overview follows.

---

A central **memory manager** for multi-agent LLM agents that routes reads and
writes across six cognitive memory types — **P**rocedural, **W**orking,
**S**hort-**T**erm, **E**pisodic, **S**emantic, and **Ent**ity — over a
ten-phase agentic lifecycle.

Instead of querying *every* memory store on *every* step (the common default),
PCCR consults a store only when it is worth its cost. On a cache miss it gates
each optional store by an explicit cost-effectiveness ratio

```
ρ(pattern, store) = expected_utility(pattern, store) / access_cost(store)
consult store iff ρ ≥ θ        (θ is a single, tunable cost/quality dial)
```

with a short-term-cache short-circuit, phase-conditioned policies, and an
outcome-driven closed loop that adapts the utilities.

**Headline result** (OfficeBench, gpt-oss-120b, held-out): at **equal task
accuracy**, PCCR issues **41–87% fewer store consultations** and injects up to
**89% fewer retrieved tokens** than an indiscriminate retrieve-everything policy.
See [`paper/`](paper/) for the write-up and [`results_archive/`](results_archive/)
for the raw results.

### Parallel execution (two dimensions of concurrency)

PCCR also supports concurrency without touching the router or the lifecycle:

1. **Parallel sub-agents** — independent sub-tasks *within* one task run
   simultaneously (dependency-DAG → waves, copy-on-read WM snapshot,
   deterministic staging→merge).
2. **Parallel tasks** — many tasks run simultaneously *across* the system
   (async queue, per-task Working-Memory isolation, per-resource store locks,
   an exclusive consolidation window).

The architecture is the 7-layer design in
[`pccr_parallel_architecture.svg`](pccr_parallel_architecture.svg) (full write-up:
[`paper/parallel_paper.tex`](paper/parallel_paper.tex)).

**Headline results** — parallelism changes *latency only, never accuracy*:

- **Latency**: 100 tasks at the measured gpt-oss-120b latency (0.516 s/call),
  **3.73× faster** at concurrency `C=4` (227.7 s → 61.1 s, −73% wall-clock),
  scaling to **13.5×** at `C=16`.
- **Accuracy**: parallel outcomes are **60/60 byte-identical** to sequential
  (and final memory state identical) — verified, not just claimed.

The `ρ`-gate router (`router.py`) and the 10-phase lifecycle are **unchanged**;
the parallel path is additive (async methods beside the sync ones).

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── cerebras.env.example         # copy → cerebras.env, add your keys (gitignored)
│
├── pccr_on_top_of_legomem.py    # ★ PCCR router (the contribution): ρ-gate,
│                                #   phase conditioning, closed loop, frozen/
│                                #   multimode/threshold-sweep experiment driver
├── mm_on_top_of_legomem.py      # base LEGOMem+MemoryManager harness (6 stores,
│                                #   10 phases, agents) — PCCR builds on this
├── run_officebench_local.py     # no-Docker OfficeBench runner + evaluators
├── free_legomem.py              # local sentence-transformer embedder
│
├── analyze_results.py           # build comparison / frontier / per-pattern tables
├── measure_tokens.py            # measure retrieved-token savings per threshold
├── run_3agent.sh                # 3-agent comparison (train once, test all modes)
├── run_sweep_3agent.sh          # 3-agent threshold sweep (cost/quality frontier)
├── run_full.sh / run_comparison.sh / run_sweep.sh   # 2-agent drivers
│
├── memory_manager/             # standalone reference implementation of the
│   ├── router.py               #   ★ the ρ-gate router (UNCHANGED by parallelism)
│   ├── manager.py              #   10-phase lifecycle + async parallel methods
│   ├── parallel.py             #   ★ DependencyAnalyzer, ParallelExecutor, LockManager
│   ├── task_queue.py           #   ★ AsyncTaskQueue (parallel tasks)
│   └── stores/                 #   the six memory stores
├── agents/                     #   orchestrator + sub-agents (offline stub backends,
├── environment/                #   simulated email/calendar/search world)
├── run_task.py                 #   sequential end-to-end demo
├── run_parallel_demo.py        # ★ all 7 parallel layers end-to-end (offline)
├── bench_parallel.py           # ★ sequential vs parallel latency (3.73× @ C=4)
├── verify_equivalence.py       # ★ proves parallel ≡ sequential (60/60 identical)
├── tests/                      #   unit + integration tests (pytest, 39 tests)
│
├── pccr_parallel_architecture.svg   # the 7-layer architecture diagram
├── paper/                      # paper.tex (PCCR core) + parallel_paper.tex (parallel)
├── explanation/               # parallel_latency.md, design tables, cheatsheet
├── results_archive/            # archived result JSONs (2-agent & 3-agent)
└── docs/                       # lifecycle spec PDF, memory schema
```

There are **two** code paths:

1. **Reference implementation** (`memory_manager/`, `agents/`, `environment/`,
   `run_task.py`, `tests/`) — a clean, fully-tested, **offline** implementation of
   the 10-phase lifecycle and the router using deterministic stub LLM/embedder
   backends. Run it with no API key:
   ```bash
   python run_task.py            # sequential end-to-end demo
   python run_parallel_demo.py   # all 7 parallel layers end-to-end
   python bench_parallel.py      # sequential vs parallel latency (3.73× @ C=4)
   python verify_equivalence.py  # parallel ≡ sequential (60/60 byte-identical)
   pytest tests/                 # 39 tests (15 cover the parallelism layer)
   ```

2. **OfficeBench experiments** (`pccr_on_top_of_legomem.py` + the `*_legomem.py`
   files and `run_*.sh`) — the real evaluation on the OfficeBench benchmark with a
   live LLM that produced the paper's numbers.

---

## Setup (for the OfficeBench experiments)

```bash
# 1. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install python-docx        # for the document (3rd) agent

# 2. Clone the OfficeBench benchmark into ./OfficeBench
git clone https://github.com/zlwang-cs/OfficeBench.git

# 3. Add your Cerebras API key(s)
cp cerebras.env.example cerebras.env
# edit cerebras.env and paste your key(s)
```

> `cerebras.env`, `.venv/`, `OfficeBench/`, and all generated artifacts
> (`logs_memory_manager/`, `data/`, `legomem_bank_mm/`) are gitignored.

---

## Reproducing the results

```bash
# 3-agent comparison: PCCR vs Boolean vs Retrieve-All (train once, test all)
./run_3agent.sh

# 3-agent threshold sweep → cost/quality frontier (θ = 1.0, 1.2, 1.4)
./run_sweep_3agent.sh

# Build the tables from the result JSONs
python analyze_results.py
python measure_tokens.py
```

Key environment knobs (read by `pccr_on_top_of_legomem.py`):

| Var | Meaning | Default |
|---|---|---|
| `PCCR_AGENTS` | 2 = calendar+email, 3 = +document | 2 |
| `PCCR_MODES` | comma list to compare (`pccr,boolean,retrieve_all`) | — |
| `PCCR_THRESHOLDS` | comma list for a pccr threshold sweep | — |
| `PCCR_THRESHOLD` | single θ (cost dial) | 1.0 |
| `PCCR_TRAIN_N` | absolute #train tasks (rest held out for test) | split frac |
| `PCCR_FREEZE_TEST` | 1 = no memory writes during test (clean isolation) | 0 |
| `PCCR_STRICT_STM` | 1 = semantic-only cache (so the router actually runs) | 1 |
| `PCCR_RESUME` | 1 = reload consolidated bank from disk, skip training | 0 |

---

## Method summary

- **Six memory types** with differing access cost (entity lookup ≪ FAISS search).
- **Ten lifecycle phases**, each with its own routing mode (the policy is *phase-
  conditioned*, not global).
- **Cascading retrieval**: short-term cache hit short-circuits everything; on a
  miss, the ρ-gate selects cost-effective optional stores per task pattern.
- **Outcome-driven closed loop**: utilities are nudged up/down from task
  success/failure; every routing decision is logged for audit.
- **Two-dimensional parallelism** (additive, router unchanged): parallel
  sub-agents within a task (dependency waves + snapshot/merge) and parallel
  tasks across the system (async queue + per-task WM isolation + per-resource
  locks + exclusive consolidation window). Outcome-equivalent to sequential.

See [`paper/paper.tex`](paper/paper.tex) for the PCCR core and
[`paper/parallel_paper.tex`](paper/parallel_paper.tex) for the parallel
architecture (full details, related work, and results).

## License

[Add a license, e.g. MIT.] The OfficeBench benchmark is a separate project under
its own license; clone it from its upstream repository.
