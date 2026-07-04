# AppWorld integration — Phase 0 (setup verified) + plan

**Status:** ✅ Phase 0 (env) + ✅ Phase 1 (adapter + code-action agent) done — one task runs end-to-end through our loop (`appworld_eval/aw_runner.py`).

## Confirmed working setup
- **Python 3.11** (AppWorld needs ≥3.11; the OfficeBench venv is 3.9): `brew install python@3.11`.
- **AppWorld** in a dedicated venv: `python3.11 -m venv .venv311 && .venv311/bin/pip install appworld`.
- **Benchmark + data:** `.venv311/bin/appworld install` then `.venv311/bin/appworld download data` (fast, ~15s → `./data/`).
- **Splits:** train 90 · dev 57 · test_normal 168 · test_challenge 417.
- Docker/Colima up (AppWorld can sandbox execution).

Probe: `.venv311/bin/python -m appworld_eval.probe`

## AppWorld interface (what the adapter bridges)
```python
from appworld import AppWorld, load_task_ids
tids = load_task_ids("train")
with AppWorld(task_id=tids[0], experiment_name="pccr_dag") as w:
    instruction = w.task.instruction          # the task text
    while not w.task_completed():
        code = agent(instruction, history)    # AGENT ACTION = Python code string
        output = w.execute(code)              # runs code calling apis.<app>.<method>(...)
        history.append((code, output))
    metrics = w.evaluate()                     # score
```
**Key difference from OfficeBench:** the agent action is a **Python code string** (calling `apis.<app>.<method>`), not a JSON `{app, action, args}`. The adapter maps our step protocol to `w.execute(code)`.

## Integration plan (this branch: `appworld-integration`)
- **Phase 1 — adapter:** ✅ DONE. `appworld_eval/aw_runner.py` = per-task driver: load task → code-action agent loop (LLM emits ```python``` → `w.execute` → observation) → `w.evaluate()`. Metrics: success / pass_count / num_tests / steps / calls / wall. Bare-agent smoke on train[0]: 1/2 tests, 8 steps, 8.9s.
- **Phase 2 — memory:** build an EM bank + pattern classifier from the 90 train tasks.
- **Phase 3 — full arch:** wire the DAG re-plan orchestrator + PCCR sub-agent leaves + N=4 parallel tasks (reuse `subagent_parallel.py`, `memory_manager/parallel.py`, `pccr_policy.py`).
- **Phase 4 — run + eval** dev (57), then test_normal.

**Why AppWorld suits the 2D/DAG dimension:** tasks span multiple apps (spotify, amazon, phone, venmo, …) with real cross-app dependencies and batch/fan-out operations — far more parallel structure than OfficeBench's mostly-sequential tasks.
