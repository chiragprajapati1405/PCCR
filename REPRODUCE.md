# Reproducing OfficeBench 72/152 (N=4 improved PCCR)

This branch contains **only** the files needed to reproduce the full-152 N=4 result
(**72/152 = 0.474**, parity with retrieve-all 71/152, injected-memory 628K tokens) plus the
**N=4 per-task traces** for that run.

## Result
| Arm | Accuracy | Injected tokens | tok/call |
|---|---|---|---|
| retrieve-all | 71/152 = 0.467 | 734,295 | 335 |
| **PCCR improved (this branch)** | **72/152 = 0.474** | **628,570** | **146** |

multi_app 0.295 (beats retrieve-all 0.269), doc_process 0.680, STM hits 12.
Full breakdown: `officebench_eval/FULL152_IMPROVED_RESULTS.md`.
Per-task results: `officebench_eval/full152_improved_final.json`.

## What's here
- `memory_manager/` — the PCCR memory architecture (router, ρ-gate, lifecycle, stores, parallel queue).
- `officebench_eval/` — the OfficeBench harness + the four execution-fidelity mechanisms (F1–F4):
  - `real_mem.py` — F1 (schema-validated past-action injection, `_clean_past_action`), F4
    (continual STM `cache_success`), inject-once split (`heavy_block`/`light_block`).
  - `pccr_policy.py` — F2 (inject-once scheduling in `build_prompt`), F3 (completion gate over
    `finish_task`+`got_stuck`, schema-aware `.eml`/`.ics`).
  - `real_arch.py`, `runner.py`, `run_t1_parallel.py`, `cerebras_llm.py`, `gate.py`.
  - Data: `em_bank.json` (memory bank), `pm_curated.json` (fixed curated rules), `split.json`,
    `patterns.json`, `full152.json`, `missing27.json`.
- `officebench_eval/traces/par_full152/` + `traces/par_missing27/` — the **N=4 traces** (152 tasks
  × `.json` full record + `.txt` step-by-step), the 125-task segment and the 27-task completion.

## External dependency (NOT in this repo)
- **OfficeBench** (the benchmark + Docker harness) is large and lives upstream — clone it into
  `OfficeBench/` at the repo root: `git clone <officebench-upstream> OfficeBench`. The harness
  expects `OfficeBench/tasks/`, `OfficeBench/apps/`, and the `utils.env`/`utils.policies` modules.
- A **Docker daemon** (Colima or Docker Desktop) with the `officebench` image built. Give the VM
  enough memory — N=4 OOM-crashed Colima during our run; restart and resume on hang.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp cerebras.env.example cerebras.env     # then fill in CEREBRAS_KEY_1..N
```

## Run (N=4, all fixes)
```bash
set -a; source cerebras.env; set +a
python -m officebench_eval.run_t1_parallel --improved \
    --tasks-file officebench_eval/full152.json --tag full152 --concurrency 4
```
`--improved` enables: confidence gate (θ_sim=0.45) + verified replay + curated PM + convention +
completion gate + self-verify + **inject-once (F2)** + **continual STM cap 24 (F4)**; F1 (sanitiser)
and F3 (got_stuck-intercepting gate) are always on in the policy. Output: `par_full152_progress.json`
and traces under `traces/par_full152/`.

## Notes / caveats
- **Backbone is gpt-oss-120b** (Cerebras). Run-to-run varies (~19/125 differ vs sequential), so
  expect 70–74/152; 72 is the recorded run preserved here.
- **Reliability:** if a worker hangs (Docker OOM), kill, restart the daemon/containers, and re-launch
  the same `--tag` — completed tasks are skipped. We completed in two N=4 segments (hence the two
  trace dirs).
- **Cost honesty:** the 628K is *injected-memory* tokens only; the tightened gate ~doubles LLM calls
  (28.3 vs 14.4/task), so the *net* API token bill is unresolved — do not quote a total-cost win
  without real `usage` accounting.
- Origin of this snapshot: tag `officebench-72of152` on branch `pccr-improvements`.
