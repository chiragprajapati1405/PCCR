"""Run REAL OfficeBench test tasks through the concurrent_mm 2-store memory manager.

Each task uses its OWN real /testbed (real eval needs it), so tasks run across a small CONTAINER POOL
(dimension-1 parallelism). Per task: the orchestrator reads PM (concurrent_mm) + plans with the real
LLM; sub-agents read Tool Memory (concurrent_mm) and run a real LLM->action->exec loop in the container;
WM buffers the steps; on success the full trajectory is appended to PM. OfficeBench evaluators score
each task. Full traces + per-task latency are logged.

  export DOCKER_HOST=unix:///Users/<you>/.colima/default/docker.sock
  set -a; source cerebras.env; set +a
  python -m concurrent_mm.run_real --n 15 --concurrency 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time

from concurrent_mm.manager import MemoryManager
from concurrent_mm.procedural_memory import ProceduralMemory, Trajectory
from concurrent_mm.tool_memory import default_tool_memory

_PKG_OB = None
TRACE_DIR = os.path.join(os.path.dirname(__file__), "real_traces")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "real_results")   # separate filesystem for outputs


def _lazy_imports():
    """Import OfficeBench machinery AFTER chdir (runner._setup)."""
    global _PKG_OB
    from officebench_eval.runner import _setup, cap_for_level
    _setup()
    from officebench_eval.cerebras_llm import CerebrasLLM
    from officebench_eval.subagent_parallel import (
        _exec_direct, VALID_ACTIONS, ARG_SCHEMA, _arg_hint, _parse_action,
        plan_delegations)
    import utils.evaluate as ev
    from utils.env import OfficeAgentEnv
    _PKG_OB = dict(CerebrasLLM=CerebrasLLM, _exec_direct=_exec_direct, VALID_ACTIONS=VALID_ACTIONS,
                   _arg_hint=_arg_hint, _parse_action=_parse_action, plan_delegations=plan_delegations,
                   ev=ev, OfficeAgentEnv=OfficeAgentEnv, cap_for_level=cap_for_level)
    return _PKG_OB


def _subagent_prompt(app, subtask, usage, last_obs, ob):
    hint = ob["_arg_hint"](app, ob["VALID_ACTIONS"][app])
    tool_note = ""
    for act, spec in (usage or {}).items():
        if spec.get("format") or spec.get("note"):
            tool_note += f"  {act}: {spec.get('format','')} {spec.get('note','')}\n"
    return (
        f"You are the {app} sub-agent. Do ONLY this sub-task with ONE tool call.\n"
        f"SUB-TASK: {subtask}\n"
        f"Actions (exact args): {hint}\n"
        + (f"Tool notes:\n{tool_note}" if tool_note else "")
        + "Files are under /testbed/data. Reply with ONE JSON {\"action\": <name>, <args>}. "
          "If the sub-task is already done, reply {\"action\": \"done\"}.\n"
        + (f"Result of last action: {last_obs[:200]}\n" if last_obs else "") + "JSON:")


def run_subagent(env, llm, app, subtask, usage, max_steps, ob):
    """Real LLM->action->exec loop for one sub-task, grounded by Tool Memory usage. Returns steps."""
    obs, steps = "", []
    for _ in range(max_steps):
        action = ob["_parse_action"](llm.generate(_subagent_prompt(app, subtask, usage, obs, ob)))
        if not action or str(action.get("action")).lower() in ("done", "finish", "none", ""):
            break
        obs = ob["_exec_direct"](env, app, action)
        steps.append({"agent": app, "action": action, "obs": obs[:200]})
        ok = "Malformed" not in obs and "Fail" not in obs and "does not exist" not in obs
        s = subtask.lower()
        if ok and not any(k in s for k in ("each", "all", "every")):
            break                                          # single-item sub-task done
    return steps


def run_real_task(it, container, mm, model, ob):
    """One REAL OfficeBench task through concurrent_mm memory. Blocking (runs in a worker thread)."""
    tid, sid = it["task"], it["subtask"]
    cfg = json.load(open(f"tasks/{tid}/subtasks/{sid}.json"))
    env = ob["OfficeAgentEnv"](image_name="officebench", container_name=container,
                               task=cfg["task"], verbose=False)
    env.reset()
    env.prepare_docker_env(testbed_dir=f"tasks/{tid}/testbed/", app_dir="apps/")
    env.cache_docker_status(local_cache_dir=f"tasks/{tid}/cache/{sid}/")
    llm = ob["CerebrasLLM"](model_name=model)
    llm.max_tokens = 4096
    t0 = time.perf_counter()
    wm = []                                                # WM: per-task private buffer

    # ORCHESTRATOR — read PM (concurrent_mm), then plan the real task
    past = mm.orchestrator_read_pm_sync(cfg["task"])       # exercises PM read (empty until successes)
    try:
        _ec, _out = env.container.exec_run("ls /testbed/data", workdir="/")
        files = _out.decode().replace("\n", ", ").strip()
    except Exception:
        files = ""
    dels = ob["plan_delegations"](cfg["task"], llm, files_hint=files) or []
    if not dels:                                           # fall back to one whole-task sub-agent
        from memory_manager.parallel import Delegation
        dels = [Delegation(agent_type="shell", subtask=cfg["task"])]

    # SUB-AGENTS — each reads Tool Memory (concurrent_mm), runs a real loop; steps buffered in WM
    for d in dels:
        usage = mm.subagent_read_tool_memory(d.agent_type)
        wm.extend(run_subagent(env, llm, d.agent_type, d.subtask, usage, max_steps=6, ob=ob))

    # SAVE OUTPUTS to the separate results filesystem (inside concurrent_mm), for deferred eval.
    results_dir = os.path.join(RESULTS_DIR, f"{tid}_{sid}")
    shutil.rmtree(results_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=results_dir)              # PERSIST /testbed -> results_dir
    json.dump(cfg["evaluation"], open(os.path.join(results_dir, "_eval_spec.json"), "w"))  # for compare
    testbed = os.path.join(results_dir, "testbed")

    # EVALUATE (also runs standalone later via eval_real.py). Inline result drives the success-gate.
    ok, fp = True, None
    for item in cfg["evaluation"]:
        try:
            if not getattr(ob["ev"], item["function"])(testbed, item["args"]):
                ok, fp = False, item["function"]; break
        except Exception as e:
            ok, fp = False, f"{item['function']}:ERR:{str(e)[:40]}"; break
    latency = time.perf_counter() - t0

    # SUCCESS-GATE — append full trajectory to PM only on success
    if ok:
        mm.orchestrator_write_pm_sync(Trajectory(task=cfg["task"],
                                                 plan=[d.subtask for d in dels], subagent_steps=wm))
    env.close()
    shutil.rmtree(f"tasks/{tid}/cache/{sid}", ignore_errors=True)     # keep results_dir (persisted)
    return {"task": f"{tid}/{sid}", "level": it["level"], "success": ok, "failed_predicate": fp,
            "latency_s": round(latency, 1), "llm_calls": getattr(llm, "calls", 0),
            "prompt_tokens": getattr(llm, "prompt_tokens", 0),
            "completion_tokens": getattr(llm, "completion_tokens", 0),
            "n_subtasks": len(dels), "steps": len(wm),
            "plan": [f"{d.agent_type}: {d.subtask}" for d in dels], "trajectory": wm,
            "committed_to_pm": ok}


def _save_trace(r):
    os.makedirs(TRACE_DIR, exist_ok=True)
    base = os.path.join(TRACE_DIR, r["task"].replace("/", "_"))
    json.dump(r, open(base + ".json", "w"), indent=2, default=str)
    with open(base + ".txt", "w") as fh:
        fh.write(f"TASK {r['task']} L{r['level']}  success={r['success']}  latency={r['latency_s']}s  "
                 f"calls={r['llm_calls']}  tokens={r['prompt_tokens']+r['completion_tokens']}\n")
        fh.write(f"failed_predicate={r['failed_predicate']}\n" + "=" * 60 + "\nPLAN:\n")
        for p in r["plan"]:
            fh.write(f"  - {p}\n")
        fh.write("STEPS:\n")
        for st in r["trajectory"]:
            fh.write(f"  [{st['agent']}] {json.dumps(st['action'])[:120]}  ->  {st['obs'][:100]}\n")


async def main_async(args):
    ob = _lazy_imports()
    split = json.load(open(os.path.join(os.path.dirname(__file__), "..", "officebench_eval", "split.json")))
    test = split["test"]
    # stratified sample: a representative MIX across L1/L2/L3 (not just the hardest)
    by_level = {1: [], 2: [], 3: []}
    for it in test:
        by_level.get(int(it["level"]), by_level[3]).append(it)
    tasks = []
    i = 0
    while len(tasks) < args.n and any(by_level.values()):
        for L in (1, 2, 3):
            if by_level[L] and len(tasks) < args.n:
                tasks.append(by_level[L].pop(0))
        i += 1
    tasks.sort(key=lambda it: -int(it["level"]))           # LPT within the sample (long tasks first)

    mm = MemoryManager(default_tool_memory(), ProceduralMemory(strategy="lockfree"))
    pool: asyncio.Queue = asyncio.Queue()
    for i in range(args.concurrency):
        os.system(f"docker rm -f cmm-real-{i} >/dev/null 2>&1")
        pool.put_nowait(f"cmm-real-{i}")
    write_lock = asyncio.Lock()
    results = []
    t_start = time.perf_counter()

    async def run_one(it):
        c = await pool.get()
        try:
            r = await asyncio.to_thread(run_real_task, it, c, mm, args.model, ob)
        finally:
            pool.put_nowait(c)
        async with write_lock:
            _save_trace(r)
            results.append(r)
            print(f"  [{len(results)}/{len(tasks)}] {r['task']} L{r['level']}  "
                  f"success={r['success']}  latency={r['latency_s']}s  calls={r['llm_calls']}  "
                  f"steps={r['steps']}  PM_size={len(mm.pm)}", flush=True)
        return r

    print(f"\nREAL OfficeBench through concurrent_mm — {len(tasks)} tasks, concurrency {args.concurrency}\n")
    await asyncio.gather(*[run_one(it) for it in tasks])
    wall = time.perf_counter() - t_start

    print(f"\n{'='*64}\nPER-TASK LATENCY\n{'='*64}")
    print(f"{'task':>10} {'L':>2} {'success':>8} {'latency_s':>10} {'calls':>6} {'steps':>6}")
    for r in sorted(results, key=lambda x: x["task"]):
        print(f"{r['task']:>10} {r['level']:>2} {str(r['success']):>8} {r['latency_s']:>10} "
              f"{r['llm_calls']:>6} {r['steps']:>6}")
    succ = sum(int(r["success"]) for r in results)
    lat = [r["latency_s"] for r in results]
    print(f"\naccuracy: {succ}/{len(results)}   wall(parallel): {wall:.1f}s   "
          f"sum per-task latency: {sum(lat):.1f}s   avg: {sum(lat)/len(lat):.1f}s   "
          f"min/max: {min(lat):.1f}/{max(lat):.1f}s")
    print(f"traces: {TRACE_DIR}")
    json.dump(results, open(os.path.join(TRACE_DIR, "_summary.json"), "w"), indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of real test tasks")
    ap.add_argument("--concurrency", type=int, default=4, help="container pool size (parallel tasks)")
    ap.add_argument("--model", default="gpt-oss-120b")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
