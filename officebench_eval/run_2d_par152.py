"""PURE-2D on the FULL 152 at TASK-LEVEL N=4 (1st dimension) STACKED with the 2D sub-agent
parallelism (2nd dimension). Four tasks run concurrently, each on its OWN Docker container
(ob-2dpar-0..3) so they never collide on /testbed/data; inside each task run_task_2d fans its
sub-agents out in parallel. Measures the COMBINED-dimension wall-clock vs the N=4 PCCR baseline.

Accuracy / tokens / calls / cost are per-task (identical to the N=1 run) -- only wall-clock differs.
Saves the same FULL per-task traces (+ cost/latency) AND the run-level parallel wall to
twod_par152_traces.json (key "_run" holds N, total_wall_s, started/'). Resumable.
"""
import json, os, time, asyncio, shutil
from concurrent.futures import ThreadPoolExecutor
REPO = "/Users/chirag/Documents/Agentic_MM"
from officebench_eval.runner import _setup
_setup()
from utils.env import OfficeAgentEnv
import utils.evaluate as ev
from officebench_eval.cerebras_llm import CerebrasLLM
import officebench_eval.subagent_parallel as sp

PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.25, 0.69
N = 4
TASKS = json.load(open(os.path.join(REPO, "officebench_eval/full152.json")))
# LPT order (L3 -> L2 -> L1) so long tasks don't tail a worker
TASKS = sorted(TASKS, key=lambda it: -int(it["level"]))
OUT = os.path.join(REPO, "officebench_eval/twod_par152_traces.json")
prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
print("PURE-2D N=%d full-152: %d tasks (%d done)" % (N, len(TASKS), len([k for k in prog if k != "_run"])), flush=True)


def _agent_row(r):
    return {"app": r.agent_type, "subtask": r.subtask,
            "action": str(getattr(r, "action", ""))[:800],
            "observation": (getattr(r, "observation", "") or "")[:800]}


def _setup_env(it, container):
    tid, sid = it["task"], it["subtask"]
    cfg = json.load(open("tasks/%s/subtasks/%s.json" % (tid, sid)))
    env = OfficeAgentEnv(image_name="officebench", container_name=container, task=cfg["task"], verbose=False)
    env.reset(); env.prepare_docker_env(testbed_dir="tasks/%s/testbed/" % tid, app_dir="apps/")
    env.cache_docker_status(local_cache_dir="tasks/%s/cache/%s/" % (tid, sid))
    return env, cfg


def _evaluate(env, cfg, it, container):
    tid, sid = it["task"], it["subtask"]
    out_dir = "tasks/%s/outputs/%s/2dpar_%s" % (tid, sid, container)
    shutil.rmtree(out_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out_dir); testbed = os.path.join(out_dir, "testbed")
    ok, fp = True, None
    for item in cfg["evaluation"]:
        try:
            if not getattr(ev, item["function"])(testbed, item["args"]): ok, fp = False, item["function"]; break
        except Exception as e:
            ok, fp = False, "%s:ERR:%s" % (item["function"], str(e)[:40]); break
    shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree("tasks/%s/cache/%s" % (tid, sid), ignore_errors=True)
    return ok, fp


async def run_one(it, pool, run_start):
    key = "%s/%s" % (it["task"], it["subtask"])
    if key in prog:
        return
    container = await pool.get()
    try:
        env, cfg = await asyncio.to_thread(_setup_env, it, container)
        llm = CerebrasLLM(model_name="gpt-oss-120b")
        t0 = time.perf_counter()
        results, waves, dels, meta = None, [], [], {"path": "ERR", "fanout_fired": False, "leaves": 0, "items": []}
        err = None
        try:
            results, waves, dels, meta = await sp.run_task_2d(cfg["task"], env, llm, force=True)
        except Exception as e:
            err = str(e)[:200]
        wall = round(time.perf_counter() - t0, 1)
        ok, fp = await asyncio.to_thread(_evaluate, env, cfg, it, container)
        await asyncio.to_thread(env.close)
        prog[key] = {
            "task": it["task"], "subtask": it["subtask"], "level": it["level"], "task_text": cfg["task"],
            "success": ok, "failed_predicate": fp, "error": err, "container": container,
            "path": meta.get("path"), "fanout_fired": meta.get("fanout_fired"),
            "leaves": meta.get("leaves"), "items": meta.get("items", []),
            "delegations": [{"app": d.agent_type, "subtask": d.subtask, "category": getattr(d, "category", "")} for d in (dels or [])],
            "wave_sizes": [len(w) for w in (waves or [])],
            "agents": [_agent_row(r) for r in (results or [])],
            "eval_spec": [e["function"] for e in cfg["evaluation"]],
            "prompt_tokens": getattr(llm, "prompt_tokens", 0), "completion_tokens": getattr(llm, "completion_tokens", 0),
            "real_tokens": getattr(llm, "prompt_tokens", 0) + getattr(llm, "completion_tokens", 0),
            "calls": getattr(llm, "calls", 0),
            "cost_usd": round(getattr(llm, "prompt_tokens", 0) / 1e6 * PRICE_IN_PER_M
                              + getattr(llm, "completion_tokens", 0) / 1e6 * PRICE_OUT_PER_M, 6),
            "wall_s": wall, "rate_limit_wait_s": round(getattr(llm, "rate_limit_wait_s", 0.0), 1),
            "rate_limit_hits": getattr(llm, "rate_limit_hits", 0),
            "worker_elapsed_s": round(time.perf_counter() - run_start, 1),   # wall since run start
        }
        prog["_run"] = {"N": N, "total_wall_s": round(time.perf_counter() - run_start, 1),
                        "start_epoch": prog.get("_run", {}).get("start_epoch"),
                        "done": len([k for k in prog if k != "_run"])}
        json.dump(prog, open(OUT, "w"), indent=1)
        npass = sum(1 for k, v in prog.items() if k != "_run" and v["success"])
        print("  [%d/152] %s: %s path=%s fire=%s (real=%d calls=%d wall=%.1fs) [%s] | pure2D %d" % (
            prog["_run"]["done"], key, "PASS" if ok else "fail", meta.get("path"), meta.get("fanout_fired"),
            prog[key]["real_tokens"], prog[key]["calls"], wall, container, npass), flush=True)
    finally:
        pool.put_nowait(container)


async def main():
    loop = asyncio.get_running_loop()
    # Cap concurrent blocking sub-agent LLM calls: 64 workers let N=4 x fan-out burst 40+ simultaneous
    # Cerebras calls -> 429 storm -> multi-minute hangs (which corrupted the overnight wall). 16 keeps
    # the request rate smooth so the run finishes in one continuous pass = a clean, real wall.
    loop.set_default_executor(ThreadPoolExecutor(max_workers=16))
    # fresh per-worker containers
    for i in range(N):
        os.system("docker rm -f ob-2dpar-%d >/dev/null 2>&1" % i)
    pool = asyncio.Queue()
    for i in range(N):
        pool.put_nowait("ob-2dpar-%d" % i)
    # persist a single wall-clock START epoch so total wall is the TRUE continuous elapsed even if the
    # process is resumed -- no more per-restart timer resets.
    start_epoch = prog.get("_run", {}).get("start_epoch") or time.time()
    prog["_run"] = {"N": N, "start_epoch": start_epoch, "done": len([k for k in prog if k != "_run"])}
    json.dump(prog, open(OUT, "w"), indent=1)                          # persist epoch before any work
    run_start = time.perf_counter() - (time.time() - start_epoch)     # anchor perf_counter to epoch
    await asyncio.gather(*[run_one(it, pool, run_start) for it in TASKS])
    total = round(time.time() - start_epoch, 1)
    prog["_run"] = {"N": N, "total_wall_s": total, "start_epoch": start_epoch,
                    "done": len([k for k in prog if k != "_run"])}
    json.dump(prog, open(OUT, "w"), indent=1)
    d = [v for k, v in prog.items() if k != "_run"]
    print("\nPURE-2D N=%d FULL-152: %d/%d passed | real tokens %d | PARALLEL wall %.1fs (%.1f min) | "
          "seq-equiv wall %.1fs" % (N, sum(v["success"] for v in d), len(d), sum(v["real_tokens"] for v in d),
          total, total / 60.0, sum(v["wall_s"] for v in d)), flush=True)
    print("DONE", flush=True)


asyncio.run(main())
