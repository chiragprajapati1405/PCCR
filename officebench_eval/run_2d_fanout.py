"""Run the 2nd-dimension (parallel sub-agent) architecture on the fan-out task subset.
Pure 2D (no 1D hybrid). For each task: plan -> read wave -> per-item fan-out expansion ->
parallel act wave, evaluate natively, record real tokens + wall. Reports accuracy + real-token
cost, to compare against the 1D improved+PERF baseline on the same tasks."""
import json, os, time, asyncio, shutil
REPO = "/Users/chirag/Documents/Agentic_MM"
from officebench_eval.runner import _setup
_setup()
from utils.env import OfficeAgentEnv
import utils.evaluate as ev
from officebench_eval.cerebras_llm import CerebrasLLM
import officebench_eval.subagent_parallel as sp

TASKS = json.load(open(os.path.join(REPO, "officebench_eval/fanout_tasks.json")))
OUT = os.path.join(REPO, "officebench_eval/twod_fanout_progress.json")
prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
print("2D fan-out run: %d tasks" % len(TASKS), flush=True)

for it in TASKS:
    tid, sid = it["task"], it["subtask"]
    key = "%s/%s" % (tid, sid)
    if key in prog:
        continue
    cfg = json.load(open("tasks/%s/subtasks/%s.json" % (tid, sid)))
    env = OfficeAgentEnv(image_name="officebench", container_name="ob-2dtest", task=cfg["task"], verbose=False)
    env.reset(); env.prepare_docker_env(testbed_dir="tasks/%s/testbed/" % tid, app_dir="apps/")
    env.cache_docker_status(local_cache_dir="tasks/%s/cache/%s/" % (tid, sid))
    llm = CerebrasLLM(model_name="gpt-oss-120b")
    t0 = time.perf_counter()
    try:
        results, waves, dels = asyncio.run(sp.run_task_2d(cfg["task"], env, llm))
    except Exception as e:
        results, waves, dels = None, [], "ERR:%s" % str(e)[:60]
    wall = round(time.perf_counter() - t0, 1)
    # native evaluation
    out_dir = "tasks/%s/outputs/%s/2d" % (tid, sid); shutil.rmtree(out_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out_dir); testbed = os.path.join(out_dir, "testbed")
    ok, fp = True, None
    for item in cfg["evaluation"]:
        try:
            if not getattr(ev, item["function"])(testbed, item["args"]): ok, fp = False, item["function"]; break
        except Exception as e:
            ok, fp = False, "%s:ERR" % item["function"]; break
    env.close(); shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree("tasks/%s/cache/%s" % (tid, sid), ignore_errors=True)
    real = llm.prompt_tokens + llm.completion_tokens
    prog[key] = {"success": ok, "level": it["level"], "failed": fp, "real_tokens": real,
                 "calls": llm.calls, "wall_s": wall, "waves": [len(w) for w in (waves or [])],
                 "parallelised": results is not None}
    json.dump(prog, open(OUT, "w"), indent=2)
    p = sum(1 for v in prog.values() if v["success"])
    print("  [%d/%d] %s: %s (real=%d calls=%d waves=%s) running pass %d" % (
        len(prog), len(TASKS), key, "PASS" if ok else "fail", real, llm.calls, prog[key]["waves"], p), flush=True)

d = list(prog.values())
print("\n2D FAN-OUT: %d/%d passed | total real tokens %d | avg %d/task" % (
    sum(v["success"] for v in d), len(d), sum(v["real_tokens"] for v in d), sum(v["real_tokens"] for v in d)//max(1,len(d))), flush=True)
print("DONE", flush=True)
