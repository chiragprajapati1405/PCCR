"""PURE-2D on the FULL 152 (fair comparison vs the sequential baselines: the 2D architecture is
applied to EVERY task, no keyword gate, no 1D fall-back -- force=True degrades an undecomposable
task to a single whole-task sub-agent). ONE run yields:
  * pure-2D accuracy (raw success over all 152)
  * the STRUCTURAL-router number (accept 2D iff meta.fanout_fired, else the 1D ledger) -- computed
    later by routed_all152.py from these traces + par_rt_improved_progress.json.

Saves the FULL trace of every task (delegations, each sub-agent's action+observation, extracted
items, decomposition path, tokens, eval result + failed predicate) so we NEVER re-run the 152 and
can mine the failures offline. Resumable: a task already in the progress file is skipped.
"""
import json, os, time, asyncio, shutil
REPO = "/Users/chirag/Documents/Agentic_MM"
from officebench_eval.runner import _setup
_setup()
from utils.env import OfficeAgentEnv
import utils.evaluate as ev
from officebench_eval.cerebras_llm import CerebrasLLM
import officebench_eval.subagent_parallel as sp

# Cerebras gpt-oss-120b published rates (USD per 1M tokens) -- edit here if the price changes; every
# trace stores the raw token counts too, so cost is always recomputable.
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.25, 0.69

TASKS = json.load(open(os.path.join(REPO, "officebench_eval/full152.json")))
OUT = os.path.join(REPO, "officebench_eval/twod_all152_traces.json")
prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
print("PURE-2D full-152 run: %d tasks (%d already done)" % (len(TASKS), len(prog)), flush=True)


def _agent_row(r):
    return {"app": r.agent_type, "subtask": r.subtask,
            "action": str(getattr(r, "action", ""))[:800],
            "observation": (getattr(r, "observation", "") or "")[:800]}


for it in TASKS:
    tid, sid = it["task"], it["subtask"]
    key = "%s/%s" % (tid, sid)
    if key in prog:
        continue
    cfg = json.load(open("tasks/%s/subtasks/%s.json" % (tid, sid)))
    env = OfficeAgentEnv(image_name="officebench", container_name="ob-2dall", task=cfg["task"], verbose=False)
    env.reset(); env.prepare_docker_env(testbed_dir="tasks/%s/testbed/" % tid, app_dir="apps/")
    env.cache_docker_status(local_cache_dir="tasks/%s/cache/%s/" % (tid, sid))
    llm = CerebrasLLM(model_name="gpt-oss-120b")
    t0 = time.perf_counter()
    results, waves, dels, meta = None, [], [], {"path": "ERR", "fanout_fired": False, "leaves": 0, "items": []}
    err = None
    try:
        results, waves, dels, meta = asyncio.run(sp.run_task_2d(cfg["task"], env, llm, force=True))
    except Exception as e:
        err = str(e)[:200]
    wall = round(time.perf_counter() - t0, 1)
    # native evaluation
    out_dir = "tasks/%s/outputs/%s/2dall" % (tid, sid); shutil.rmtree(out_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out_dir); testbed = os.path.join(out_dir, "testbed")
    ok, fp = True, None
    for item in cfg["evaluation"]:
        try:
            if not getattr(ev, item["function"])(testbed, item["args"]): ok, fp = False, item["function"]; break
        except Exception as e:
            ok, fp = False, "%s:ERR:%s" % (item["function"], str(e)[:40]); break
    env.close(); shutil.rmtree(out_dir, ignore_errors=True); shutil.rmtree("tasks/%s/cache/%s" % (tid, sid), ignore_errors=True)
    prog[key] = {
        "task": tid, "subtask": sid, "level": it["level"], "task_text": cfg["task"],
        "success": ok, "failed_predicate": fp, "error": err,
        "path": meta.get("path"), "fanout_fired": meta.get("fanout_fired"),
        "leaves": meta.get("leaves"), "items": meta.get("items", []),
        "delegations": [{"app": d.agent_type, "subtask": d.subtask, "category": getattr(d, "category", "")} for d in (dels or [])],
        "wave_sizes": [len(w) for w in (waves or [])],
        "agents": [_agent_row(r) for r in (results or [])],
        "eval_spec": [e["function"] for e in cfg["evaluation"]],
        "prompt_tokens": getattr(llm, "prompt_tokens", 0), "completion_tokens": getattr(llm, "completion_tokens", 0),
        "real_tokens": getattr(llm, "prompt_tokens", 0) + getattr(llm, "completion_tokens", 0),
        "calls": getattr(llm, "calls", 0),
        # cost (USD) from real billed tokens
        "cost_usd": round(getattr(llm, "prompt_tokens", 0) / 1e6 * PRICE_IN_PER_M
                          + getattr(llm, "completion_tokens", 0) / 1e6 * PRICE_OUT_PER_M, 6),
        # RAW timing only (derived compute_s / per-call latency are ill-defined per-task under
        # concurrency -- rate_limit_wait_s is SUMMED across parallel sub-agents so it can exceed the
        # parallel wall; the paper-column aggregates are built by build_2d_table.py from these raws).
        "wall_s": wall,                                                     # true elapsed (the latency)
        "rate_limit_wait_s": round(getattr(llm, "rate_limit_wait_s", 0.0), 1),  # summed 429 time
        "rate_limit_hits": getattr(llm, "rate_limit_hits", 0),
    }
    json.dump(prog, open(OUT, "w"), indent=1)
    npass = sum(1 for v in prog.values() if v["success"])
    nfire = sum(1 for v in prog.values() if v.get("fanout_fired"))
    print("  [%d/%d] %s: %s path=%s fire=%s leaves=%s (real=%d calls=%d) | pure2D %d, fanout-fired %d" % (
        len(prog), len(TASKS), key, "PASS" if ok else "fail", meta.get("path"), meta.get("fanout_fired"),
        meta.get("leaves"), prog[key]["real_tokens"], prog[key]["calls"], npass, nfire), flush=True)

d = list(prog.values())
print("\nPURE-2D FULL-152: %d/%d passed | total real tokens %d | fanout-fired on %d tasks" % (
    sum(v["success"] for v in d), len(d), sum(v["real_tokens"] for v in d),
    sum(1 for v in d if v.get("fanout_fired"))), flush=True)
print("DONE", flush=True)
