"""Option B — run REAL OfficeBench test tasks in ONE container, isolated per task.

How isolation works (OfficeBench hardcodes /testbed, so we namespace it):
  * each task gets a root /testbed/run_<id>; its testbed is copied there.
  * FILE apps (excel/word/pdf) take absolute /testbed/data paths as ARGS -> we rewrite the command
    string  /testbed -> /testbed/run_<id>.
  * calendar/email/... hardcode /testbed INSIDE the app script -> we patched them to honor the env var
    TESTBED_ROOT (default /testbed, backward-compatible), which we set per exec.
Result: N real tasks share ONE container, never colliding, outputs saved per task for deferred eval.

  export DOCKER_HOST=unix:///Users/<you>/.colima/default/docker.sock
  set -a; source cerebras.env; set +a
  python -m concurrent_mm.run_real_one --n 15
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

TRACE_DIR = os.path.join(os.path.dirname(__file__), "one_traces")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "one_results")


def _docker_client():
    import docker
    if os.environ.get("DOCKER_HOST"):
        return docker.from_env()
    colima = os.path.expanduser("~/.colima/default/docker.sock")
    return docker.DockerClient(base_url=f"unix://{colima}") if os.path.exists(colima) else docker.from_env()


class OneBox:
    """The single shared container. Namespaces each task under /testbed/run_<id>."""

    def __init__(self, ob, container="cmm-onebox", image="officebench"):
        import docker
        self.ob = ob
        self.client = _docker_client()
        try:
            self.container = self.client.containers.get(container)
            if self.container.status != "running":
                self.container.start()
        except docker.errors.NotFound:
            self.container = self.client.containers.run(image, name=container, command="sleep infinity",
                                                        detach=True, tty=True)
        self.name = container
        os.system(f"docker cp apps {self.name}:/ >/dev/null 2>&1")   # apps once (cwd=OfficeBench)

    def setup_task(self, tid, sid) -> str:
        ns = f"/testbed/run_{tid}_{sid}"
        self.container.exec_run(f'/bin/bash -c "rm -rf {ns} && mkdir -p {ns}"')
        os.system(f"docker cp tasks/{tid}/testbed/. {self.name}:{ns} >/dev/null 2>&1")   # data -> ns
        return ns

    def exec_action(self, app, action, ns):
        """Build the real app command, rewrite /testbed -> ns, set TESTBED_ROOT=ns, exec."""
        import apps
        ob = self.ob
        act = action.get("action")
        if act not in ob["VALID_ACTIONS"].get(app, []):
            return f"Malformed action! Unknown '{act}' for '{app}'."
        need = ob["ARG_SCHEMA"].get(f"{app}.{act}", [])
        payload = {}
        for k, v in action.items():
            payload[ob["ARG_ALIAS"].get(k, k) if ob["ARG_ALIAS"].get(k, k) in need else k] = v
        if "contents" in need and "content" in action and "contents" not in payload:
            payload["contents"] = action["content"]
        payload["action"], payload["app"] = act, app
        try:
            command = apps.AVAILABLE_ACTIONS[app][act].construct_action(ns, args=payload)
        except Exception as e:
            return f"Malformed action! ({str(e)[:60]})"
        command = command.replace("/testbed", ns)                    # rewrite ARG paths -> namespace
        ec, out = self.container.exec_run(["/bin/bash", "-c", command], workdir=ns,
                                          environment={"TESTBED_ROOT": ns})
        return out.decode("utf-8", "ignore").split("OBSERVATION:")[-1].strip()

    def save_and_eval(self, tid, sid, ns, eval_spec):
        results_dir = os.path.join(RESULTS_DIR, f"{tid}_{sid}")
        shutil.rmtree(results_dir, ignore_errors=True)
        os.makedirs(os.path.join(results_dir, "testbed"), exist_ok=True)
        os.system(f'docker cp {self.name}:{ns}/. "{os.path.join(results_dir, "testbed")}" >/dev/null 2>&1')
        json.dump(eval_spec, open(os.path.join(results_dir, "_eval_spec.json"), "w"))
        testbed = os.path.join(results_dir, "testbed")
        ok, fp = True, None
        for item in eval_spec:
            try:
                if not getattr(self.ob["ev"], item["function"])(testbed, item["args"]):
                    ok, fp = False, item["function"]; break
            except Exception as e:
                ok, fp = False, f"{item['function']}:ERR:{str(e)[:40]}"; break
        self.container.exec_run(f'/bin/bash -c "rm -rf {ns}"')       # free the namespace
        return ok, fp


def _lazy_imports():
    from officebench_eval.runner import _setup
    _setup()
    from officebench_eval.cerebras_llm import CerebrasLLM
    from officebench_eval.subagent_parallel import (
        VALID_ACTIONS, ARG_SCHEMA, _ARG_ALIAS, _arg_hint, _parse_action, _parse_json_block,
        _is_fanout, _extract_items_structural)
    from memory_manager.parallel import Delegation, categorize
    import utils.evaluate as ev
    return dict(CerebrasLLM=CerebrasLLM, VALID_ACTIONS=VALID_ACTIONS, ARG_SCHEMA=ARG_SCHEMA,
                ARG_ALIAS=_ARG_ALIAS, _arg_hint=_arg_hint, _parse_action=_parse_action,
                _parse_json_block=_parse_json_block, _is_fanout=_is_fanout,
                _extract_items_structural=_extract_items_structural,
                Delegation=Delegation, categorize=categorize, ev=ev)


# ---- PLANNER: uses the retrieved PM trajectory + the REAL file list -------------------------------
def _plan(task, files, pm_hit, llm, ob):
    """Orchestrator plan. FIX: injects the retrieved PM trajectory (reuse past plan) AND the real
    filenames, so the plan grounds on memory + actual data (not hallucinated names)."""
    hint = ""
    if pm_hit is not None:
        hint = ("A SIMILAR PAST TASK succeeded with this plan — ADAPT it:\n"
                + "\n".join(f"  - {p}" for p in pm_hit.plan[:8]) + "\n\n")
    prompt = (hint +
              "Decompose this office-automation task into the SMALLEST independent sub-tasks. "
              "Return ONLY a JSON array of {\"app\": <shell,excel,word,pdf,ocr,calendar,email>, "
              "\"subtask\": <one concrete instruction>}. RULES: (1) a sub-task that READS/lists source "
              "data MUST come before sub-tasks that use it; (2) if the task repeats an operation over "
              "each item (each member/row/person), phrase the subtask with 'for each'; (3) use ONLY the "
              "EXACT filenames below — never invent one.\n"
              f"Available files: {files}\n\nTASK: {task}\n\nJSON:")
    parsed = ob["_parse_json_block"](llm.generate(prompt)) or []
    dels = []
    for d in parsed:
        app = str(d.get("app", "")).lower().strip()
        sub = str(d.get("subtask", "")).strip()
        if app in ob["VALID_ACTIONS"] and sub:
            dels.append(ob["Delegation"](agent_type=app, subtask=sub))
    if not dels:
        dels = [ob["Delegation"](agent_type="shell", subtask=task)]
    return dels


def _subagent_prompt(app, subtask, usage, last_obs, blackboard, ob):
    """FIX: the sub-agent now SEES the real file list + data read so far (from the blackboard), so it
    stops hallucinating filenames, plus the Tool Memory usage (how to call the tool)."""
    hint = ob["_arg_hint"](app, ob["VALID_ACTIONS"][app])
    notes = "".join(f"  {a}: {s.get('format','')} {s.get('note','')}\n"
                    for a, s in (usage or {}).items() if s.get("format") or s.get("note"))
    data = (blackboard.get("data", "") or "")[:900]
    return (f"You are the {app} sub-agent. Do ONLY this sub-task with ONE tool call.\n"
            f"SUB-TASK: {subtask}\n"
            f"Available files (use EXACT names): {blackboard.get('files','')}\n"
            + (f"DATA read so far:\n{data}\n" if data else "")
            + f"Actions (exact args): {hint}\n"
            + (f"Tool notes:\n{notes}" if notes else "")
            + "Reply with ONE JSON {\"action\": <name>, <args>}. If already done, {\"action\": \"done\"}.\n"
            + (f"Result of last action: {last_obs[:200]}\n" if last_obs else "") + "JSON:")


async def _run_subagent(box, llm, d, mm, ns, blackboard, ob):
    """One sub-agent: reads ITS tool memory, runs an LLM->action->exec loop. FIX: breaks on a repeated
    identical failing action (no more 6x identical fails). Blocking calls go through to_thread so
    same-wave sub-agents overlap."""
    usage = mm.subagent_read_tool_memory(d.agent_type)          # per-app Tool Memory (read-only)
    steps, obs, sigs = [], "", []
    fan = ob["_is_fanout"](d.subtask)
    for _ in range(6):
        prompt = _subagent_prompt(d.agent_type, d.subtask, usage, obs, blackboard, ob)
        action = ob["_parse_action"](await asyncio.to_thread(llm.generate, prompt))
        if not action or str(action.get("action")).lower() in ("done", "finish", "none", ""):
            break
        obs = await asyncio.to_thread(box.exec_action, d.agent_type, action, ns)
        steps.append({"agent": d.agent_type, "action": action, "obs": obs[:200]})
        failed = ("Malformed" in obs or "Fail" in obs or "does not exist" in obs)
        sig = json.dumps(action, sort_keys=True)
        if sig in sigs:                                         # repeated identical action (ok OR fail) -> stop
            break
        sigs.append(sig)
        if (not failed) and (not fan):                          # single-item sub-task done
            break
    return steps


def _expand_fanout(acts, data, ob):
    """FIX: expand a 'for each <item>' act delegation into ONE sub-agent per real item parsed from the
    read data (structural grid extraction) — so N events get created, not one for a literal '<name>'."""
    out = []
    for d in acts:
        if ob["_is_fanout"](d.subtask):
            items = ob["_extract_items_structural"](data, d.subtask)
            if len(items) >= 1:
                import re as _re
                base = _re.sub(r"\b(for\s+)?(each|every|all)\b.*$", "", d.subtask, flags=_re.I).strip(" ,.")
                for it in items[:25]:
                    out.append(ob["Delegation"](agent_type=d.agent_type, subtask=f"{base} for {it}"))
                continue
        out.append(d)
    return out


async def run_one_task(it, box, mm, model, ob):
    tid, sid = it["task"], it["subtask"]
    cfg = json.load(open(f"tasks/{tid}/subtasks/{sid}.json"))
    ns = await asyncio.to_thread(box.setup_task, tid, sid)
    llm = ob["CerebrasLLM"](model_name=model); llm.max_tokens = 4096
    t0 = time.perf_counter()
    wm = []
    _ec, _out = await asyncio.to_thread(box.container.exec_run, ["/bin/bash", "-c", f"ls {ns}/data"])
    names = [n for n in _out.decode("utf-8", "ignore").split() if n]
    files = ", ".join(f"/testbed/data/{n}" for n in names)   # FULL paths + exact case (agent copies verbatim)

    # ORCHESTRATOR: PM VECTOR search (top-1) -> USE the hit in planning
    pm_hits = mm.orchestrator_read_pm_sync(cfg["task"], k=1)
    pm_hit = pm_hits[0] if pm_hits else None
    dels = await asyncio.to_thread(_plan, cfg["task"], files, pm_hit, llm, ob)

    # split into a READ wave and an ACT wave (dependency: read before act)
    reads = [d for d in dels if ob["categorize"](d.subtask) == "read"]
    acts = [d for d in dels if ob["categorize"](d.subtask) != "read"]
    blackboard = {"files": files, "data": ""}

    # READ WAVE — sub-agents in PARALLEL (2nd dimension)
    if reads:
        rres = await asyncio.gather(*[_run_subagent(box, llm, d, mm, ns, blackboard, ob) for d in reads])
        for steps in rres:
            wm.extend(steps)
            blackboard["data"] += "\n".join(s["obs"] for s in steps) + "\n"

    # FAN-OUT EXPANSION using the read data, then ACT WAVE in PARALLEL
    acts = _expand_fanout(acts, blackboard["data"], ob)
    if acts:
        ares = await asyncio.gather(*[_run_subagent(box, llm, d, mm, ns, blackboard, ob) for d in acts])
        for steps in ares:
            wm.extend(steps)

    ok, fp = await asyncio.to_thread(box.save_and_eval, tid, sid, ns, cfg["evaluation"])
    latency = time.perf_counter() - t0
    used_pm = pm_hit is not None
    if ok:                                                       # SUCCESS-GATE -> append to PM
        mm.orchestrator_write_pm_sync(Trajectory(task=cfg["task"],
                                                 plan=[d.subtask for d in dels], subagent_steps=wm))
    return {"task": f"{tid}/{sid}", "level": it["level"], "success": ok, "failed_predicate": fp,
            "latency_s": round(latency, 1), "llm_calls": getattr(llm, "calls", 0), "used_pm": used_pm,
            "steps": len(wm), "plan": [f"{d.agent_type}: {d.subtask}" for d in dels], "trajectory": wm}


PM_STORE = os.path.join(os.path.dirname(__file__), "pm_store.json")


def _load_pm(pm, embedder):
    """Warm PM from the persistent bank so retrieval has content (re-embeds task text on load)."""
    if not os.path.exists(PM_STORE):
        return
    for t in json.load(open(PM_STORE)):
        traj = Trajectory(task=t["task"], plan=t["plan"], subagent_steps=t.get("steps", []))
        traj.embedding = embedder(t["task"])
        pm._log.append(traj)


def _save_pm(pm):
    json.dump([{"task": t.task, "plan": t.plan, "steps": t.subagent_steps} for t in pm._log],
              open(PM_STORE, "w"), indent=1, default=str)


def _save_trace(r):
    os.makedirs(TRACE_DIR, exist_ok=True)
    base = os.path.join(TRACE_DIR, r["task"].replace("/", "_"))
    json.dump(r, open(base + ".json", "w"), indent=2, default=str)
    with open(base + ".txt", "w") as fh:
        fh.write(f"TASK {r['task']} L{r['level']} success={r['success']} latency={r['latency_s']}s "
                 f"calls={r['llm_calls']} failed={r['failed_predicate']}\nPLAN:\n")
        for p in r["plan"]:
            fh.write(f"  - {p}\n")
        fh.write("STEPS:\n")
        for st in r["trajectory"]:
            fh.write(f"  [{st['agent']}] {json.dumps(st['action'])[:110]} -> {st['obs'][:90]}\n")


async def main_async(args):
    ob = _lazy_imports()
    split = json.load(open(os.path.join(os.path.dirname(__file__), "..", "officebench_eval", "split.json")))
    test = split["test"]
    by = {1: [], 2: [], 3: []}
    for it in test:
        by.get(int(it["level"]), by[3]).append(it)
    tasks = []
    while len(tasks) < args.n and any(by.values()):
        for L in (1, 2, 3):
            if by[L] and len(tasks) < args.n:
                tasks.append(by[L].pop(0))
    tasks.sort(key=lambda it: -int(it["level"]))

    shutil.rmtree(RESULTS_DIR, ignore_errors=True); shutil.rmtree(TRACE_DIR, ignore_errors=True)
    box = OneBox(ob)
    from sentence_transformers import SentenceTransformer
    _st = SentenceTransformer("all-MiniLM-L6-v2")          # PM retrieval = VECTOR search
    def _embed(text):
        return _st.encode([text], normalize_embeddings=True)[0]
    pm = ProceduralMemory(strategy="lockfree", embedder=_embed)
    _load_pm(pm, _embed)                                    # WARM PM from prior runs (persistent bank)
    print(f"PM loaded with {len(pm)} past trajectories (vector search)")
    mm = MemoryManager(default_tool_memory(), pm)
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    results = []
    t_start = time.perf_counter()

    async def run(it):
        async with sem:
            r = await run_one_task(it, box, mm, args.model, ob)   # async: parallel sub-agent waves
        async with lock:
            _save_trace(r); results.append(r)
            print(f"  [{len(results)}/{len(tasks)}] {r['task']} L{r['level']} success={r['success']} "
                  f"latency={r['latency_s']}s calls={r['llm_calls']} used_pm={r['used_pm']} PM={len(mm.pm)}", flush=True)
        return r

    print(f"\nONE-CONTAINER real OfficeBench (Option B) — {len(tasks)} tasks, in-container concurrency {args.concurrency}\n")
    await asyncio.gather(*[run(it) for it in tasks])
    _save_pm(mm.pm)                                         # persist PM so it warms the NEXT run
    wall = time.perf_counter() - t_start

    print(f"\n{'='*60}\nPER-TASK LATENCY (one container)\n{'='*60}")
    print(f"{'task':>10} {'L':>2} {'success':>8} {'latency_s':>10} {'calls':>6}")
    for r in sorted(results, key=lambda x: x["task"]):
        print(f"{r['task']:>10} {r['level']:>2} {str(r['success']):>8} {r['latency_s']:>10} {r['llm_calls']:>6}")
    succ = sum(int(r["success"]) for r in results)
    lat = [r["latency_s"] for r in results]
    print(f"\naccuracy: {succ}/{len(results)}   TOTAL wall(all in ONE container): {wall:.1f}s   "
          f"max task: {max(lat):.1f}s   avg: {sum(lat)/len(lat):.1f}s")
    print(f"traces: {TRACE_DIR}   results: {RESULTS_DIR}")
    json.dump(results, open(os.path.join(TRACE_DIR, "_summary.json"), "w"), indent=2, default=str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--concurrency", type=int, default=15, help="in-container parallel tasks")
    ap.add_argument("--model", default="gpt-oss-120b")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
