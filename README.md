# PCCR on AppWorld — memory & orchestration generality study

Testing whether the **PCCR** architecture (memory router + DAG re-plan orchestration), developed on
OfficeBench, **generalizes to a different agentic benchmark**: [AppWorld](https://appworld.dev) — an
interactive coding agent over 9 apps (spotify, amazon, phone, venmo, gmail, …) where the agent acts by
writing Python that calls `apis.<app>.<method>`. Backbone: **gpt-oss-120b** (Cerebras).

This branch is **self-contained** (`appworld_eval/` only; no `officebench_eval`/`memory_manager` deps).

## Headline results (dev split, gpt-oss-120b)

| PCCR dimension | Result | Transfers? |
|---|---|---|
| **Memory** (ρ-gate + episodic procedures, inject-once) | no-mem **10%** → PCCR-mem **60%** (+10, 0 regressions) | ✅ **yes, strongly** |
| **DAG re-plan orchestration** | flat-mem **57%** vs DAG-orch **14%**, ~**3× the calls** | ❌ **no** (coherence loss) |

**Takeaway:** PCCR's *memory* is the general, transferable contribution — a large, regression-free lift
on a completely different benchmark. The *DAG/orchestration* dimension does **not** help (a good flat
memory agent beats it here as on OfficeBench). AppWorld's tasks are highly procedural (login → paginate
→ filter/sort → answer), so retrieved solved procedures are almost directly adaptable — the memory lift
is larger here than on OfficeBench.

## Layout (`appworld_eval/`)
| File | Role |
|---|---|
| `probe.py` | Phase 0 — confirms the AppWorld env + agent loop work end-to-end |
| `cerebras_llm.py` | gpt-oss-120b Cerebras client (multi-key, rate-limit handling) |
| `aw_runner.py` | Phase 1 — code-action agent step loop; `run_task(..., em=)` inject-once |
| `build_em_bank.py` | Phase 2 — harvest 90 EM procedures from train ground-truth solutions → `em_bank_appworld.json` |
| `aw_memory.py` | Phase 2 — `AppWorldEM`: all-MiniLM-L6-v2 retriever + `inject_block()` |
| `aw_compare.py` | Phase 3.1 — no-memory vs PCCR-memory on dev → `dev_nomem_vs_mem.json` |
| `aw_dag.py` | Phase 3.2 — DAG re-plan orchestrator (memory-backed) vs flat-memory → `dev_dag_orch.json` |

Full setup, the AppWorld interface, and the phase log: **`APPWORLD_SETUP.md`**.

## Setup & run
```bash
# 1. AppWorld needs Python 3.11+
brew install python@3.11
python3.11 -m venv .venv311
.venv311/bin/pip install appworld openai sentence-transformers
.venv311/bin/appworld install && .venv311/bin/appworld download data   # -> ./data (gitignored)

# 2. Cerebras keys
cp cerebras.env.example cerebras.env   # add CEREBRAS_KEY_1..N   (gitignored)

# 3. run
set -a; source cerebras.env; set +a
.venv311/bin/python -m appworld_eval.build_em_bank          # build the EM bank
.venv311/bin/python -m appworld_eval.aw_compare 20          # no-mem vs PCCR-mem (dev)
.venv311/bin/python -m appworld_eval.aw_dag 12              # DAG-orch vs flat-memory (dev)
```

## Not committed (external / gitignored)
`.venv311/`, `data/`, `.appworld/`, `cerebras.env`, and the `all-MiniLM-L6-v2` embedder (auto-downloads).
