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
        # pass the LLM's args straight through (tool memory now gives verified arg names, so no
        # buggy re-aliasing that used to turn a correct 'username' into 'user' against a stale schema)
        payload = {k: v for k, v in action.items() if k != "action"}
        if isinstance(payload.get("args"), dict):                    # LLM sometimes nests args in "args"
            payload.update(payload.pop("args"))
        if "content" in payload and "contents" not in payload:       # common convenience only
            payload["contents"] = payload["content"]
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
    return dels, prompt                                          # prompt returned for the trace


def _subagent_prompt(app, subtask, usage, last_obs, blackboard, ob):
    """FIX: the sub-agent now SEES the real file list + data read so far (from the blackboard), so it
    stops hallucinating filenames, plus the Tool Memory usage (how to call the tool). The action hint
    is built FROM tool memory (verified arg names), not the stale ARG_SCHEMA."""
    hint = "; ".join(f"{a}({', '.join(s.get('args', []))})" for a, s in (usage or {}).items())
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
    """One sub-agent: reads ITS tool memory, runs an LLM->action->exec loop. Returns (steps, sa_trace)
    where sa_trace captures the tool memory retrieved + the system prompt + every step (for the trace)."""
    usage = mm.subagent_read_tool_memory(d.agent_type)          # per-app Tool Memory (read-only)
    steps, obs, sigs = [], "", []
    fan = ob["_is_fanout"](d.subtask)
    first_prompt = None
    for _ in range(6):
        prompt = _subagent_prompt(d.agent_type, d.subtask, usage, obs, blackboard, ob)
        if first_prompt is None:
            first_prompt = prompt                               # the sub-agent's system prompt (step 1)
        action = ob["_parse_action"](await asyncio.to_thread(llm.generate, prompt))
        if not action or str(action.get("action")).lower() in ("done", "finish", "none", ""):
            break
        obs = await asyncio.to_thread(box.exec_action, d.agent_type, action, ns)
        steps.append({"agent": d.agent_type, "action": action, "obs": obs[:220]})
        failed = ("Malformed" in obs or "Fail" in obs or "does not exist" in obs)
        sig = json.dumps(action, sort_keys=True)
        if sig in sigs:                                         # repeated identical action (ok OR fail) -> stop
            break
        sigs.append(sig)
        if (not failed) and (not fan):                          # single-item sub-task done
            break
    sa_trace = {"app": d.agent_type, "subtask": d.subtask,
                "tool_memory_retrieved": usage,                 # what THIS sub-agent read from Tool Mem
                "system_prompt": first_prompt, "steps": steps}
    return steps, sa_trace


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

    tr = {"task": f"{tid}/{sid}", "level": it["level"], "task_text": cfg["task"], "files": files}

    # ORCHESTRATOR: PM VECTOR search (top-1, WITH similarity score) -> USE the hit in planning
    scored = mm.orchestrator_read_pm_scored(cfg["task"], k=1)
    pm_sim, pm_hit = (scored[0][0], scored[0][1]) if scored else (None, None)
    tr["pm"] = {"query": cfg["task"], "retrieved": (pm_hit.task if pm_hit else None),
                "sim_score": (round(pm_sim, 3) if pm_sim is not None else None),
                "retrieved_plan": (pm_hit.plan if pm_hit else [])}
    dels, orch_prompt = await asyncio.to_thread(_plan, cfg["task"], files, pm_hit, llm, ob)
    tr["orchestrator_system_prompt"] = orch_prompt
    tr["orchestrator_delegations"] = [{"app": d.agent_type, "subtask": d.subtask,
                                       "category": ob["categorize"](d.subtask)} for d in dels]

    # WAVE COMPUTATION: split by dependency category (read before act)
    reads = [d for d in dels if ob["categorize"](d.subtask) == "read"]
    acts = [d for d in dels if ob["categorize"](d.subtask) != "read"]
    blackboard = {"files": files, "data": ""}
    tr["waves"] = {"read_wave": [f"[{d.agent_type}] {d.subtask}" for d in reads],
                   "act_wave_planned": [f"[{d.agent_type}] {d.subtask}" for d in acts]}
    subagents = []

    # READ WAVE — sub-agents in PARALLEL (2nd dimension)
    if reads:
        rres = await asyncio.gather(*[_run_subagent(box, llm, d, mm, ns, blackboard, ob) for d in reads])
        for steps, sa in rres:
            wm.extend(steps); subagents.append({**sa, "wave": "read"})
            blackboard["data"] += "\n".join(s["obs"] for s in steps) + "\n"

    # FAN-OUT EXPANSION using the read data, then ACT WAVE in PARALLEL
    acts_before = [f"[{d.agent_type}] {d.subtask}" for d in acts]
    acts = _expand_fanout(acts, blackboard["data"], ob)
    tr["waves"]["fanout_expanded"] = {"before": acts_before,
                                      "after": [f"[{d.agent_type}] {d.subtask}" for d in acts]}
    if acts:
        ares = await asyncio.gather(*[_run_subagent(box, llm, d, mm, ns, blackboard, ob) for d in acts])
        for steps, sa in ares:
            wm.extend(steps); subagents.append({**sa, "wave": "act"})
    tr["subagents"] = subagents

    ok, fp = await asyncio.to_thread(box.save_and_eval, tid, sid, ns, cfg["evaluation"])
    latency = time.perf_counter() - t0
    if ok:                                                       # SUCCESS-GATE -> append to PM
        mm.orchestrator_write_pm_sync(Trajectory(task=cfg["task"],
                                                 plan=[d.subtask for d in dels], subagent_steps=wm))
    tr.update({"success": ok, "failed_predicate": fp, "latency_s": round(latency, 1),
               "llm_calls": getattr(llm, "calls", 0), "used_pm": pm_hit is not None,
               "pm_hit_task": (pm_hit.task if pm_hit else None), "steps": len(wm)})
    return tr


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


def _blk(fh, title, body):
    fh.write(f"\n{'-'*78}\n{title}\n{'-'*78}\n{body.rstrip()}\n")


def _save_trace(r):
    """Write the FULL flow so a task is understandable from the trace alone: PM retrieval + sim score,
    orchestrator system prompt + delegations + wave computation, and per sub-agent the tool memory
    retrieved + its system prompt + every step."""
    os.makedirs(TRACE_DIR, exist_ok=True)
    base = os.path.join(TRACE_DIR, r["task"].replace("/", "_"))
    json.dump(r, open(base + ".json", "w"), indent=2, default=str)
    with open(base + ".txt", "w") as fh:
        fh.write("=" * 78 + "\n")
        fh.write(f"TASK {r['task']}  L{r['level']}  success={r['success']}  "
                 f"failed_predicate={r.get('failed_predicate')}\n")
        fh.write(f"latency={r['latency_s']}s  llm_calls={r['llm_calls']}  steps={r['steps']}\n")
        fh.write(f"TASK TEXT: {r.get('task_text','')}\n")
        fh.write(f"FILES: {r.get('files','')}\n")
        fh.write("=" * 78 + "\n")

        # [1] PM retrieval
        pm = r.get("pm", {})
        _blk(fh, "[1] PM RETRIEVAL  (vector search over the seeded trajectory bank)",
             f"query        : {pm.get('query','')}\n"
             f"retrieved    : {pm.get('retrieved')}\n"
             f"cosine score : {pm.get('sim_score')}\n"
             f"retrieved plan (injected into the orchestrator prompt):\n" +
             "".join(f"   - {p}\n" for p in (pm.get('retrieved_plan') or [])))

        # [2] Orchestrator
        _blk(fh, "[2] ORCHESTRATOR — SYSTEM PROMPT (given to the orchestrator)",
             r.get("orchestrator_system_prompt", ""))
        _blk(fh, "[2b] ORCHESTRATOR — DELEGATIONS (its output)",
             "".join(f"   {i+1}. [{d['app']}] ({d['category']}) {d['subtask']}\n"
                     for i, d in enumerate(r.get("orchestrator_delegations", []))))

        # [3] Wave computation
        w = r.get("waves", {})
        fo = w.get("fanout_expanded", {})
        _blk(fh, "[3] WAVE COMPUTATION (dependency: read wave  ->  act wave; fan-out expands per item)",
             "READ WAVE (parallel):\n" + "".join(f"   {x}\n" for x in w.get("read_wave", [])) +
             "ACT WAVE planned (parallel):\n" + "".join(f"   {x}\n" for x in w.get("act_wave_planned", [])) +
             ("FAN-OUT EXPANSION:\n   before: " + str(fo.get("before", [])) +
              "\n   after : " + str(fo.get("after", [])) + "\n" if fo else ""))

        # [4] Sub-agents
        fh.write(f"\n{'-'*78}\n[4] SUB-AGENTS ({len(r.get('subagents',[]))})\n{'-'*78}\n")
        for i, sa in enumerate(r.get("subagents", [])):
            fh.write(f"\n>> SUB-AGENT #{i+1}  wave={sa.get('wave')}  app={sa['app']}\n")
            fh.write(f"   SUB-TASK: {sa['subtask']}\n")
            tm = sa.get("tool_memory_retrieved", {})
            fh.write(f"   TOOL MEMORY retrieved (app='{sa['app']}'):\n")
            for act, spec in tm.items():
                fh.write(f"      {act}({', '.join(spec.get('args', []))})"
                         + (f"  [{spec.get('format') or spec.get('note')}]" if (spec.get('format') or spec.get('note')) else "") + "\n")
            _blk(fh, f"   SUB-AGENT #{i+1} — SYSTEM PROMPT (given to the sub-agent)",
                 sa.get("system_prompt", "") or "")
            fh.write("   STEPS:\n")
            for j, st in enumerate(sa.get("steps", [])):
                fh.write(f"      step {j+1}: {json.dumps(st['action'])[:140]}\n"
                         f"              -> {st['obs'][:140]}\n")

        _blk(fh, "[5] EVALUATION",
             f"success={r['success']}  failed_predicate={r.get('failed_predicate')}")


async def main_async(args):
    args.pm_bank = os.path.abspath(args.pm_bank)           # resolve BEFORE _setup chdirs into OfficeBench
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
    if args.pm_bank and os.path.exists(args.pm_bank):      # SEED PM from the bank of past successes
        bank = json.load(open(args.pm_bank))
        for r in bank:
            traj = Trajectory(task=r["task"], plan=r.get("plan", []), subagent_steps=r.get("steps", []))
            traj.embedding = _embed(r["task"])
            pm._log.append(traj)
        print(f"PM SEEDED from {os.path.basename(args.pm_bank)}: {len(pm)} past successful trajectories")
    _load_pm(pm, _embed)                                    # also warm from prior runs (persistent)
    print(f"PM has {len(pm)} trajectories (vector search retrieval)")
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
            _sim = (r.get('pm') or {}).get('sim_score')
            hit = (" <- PM(sim=%s):'%s'" % (_sim, r['pm_hit_task'][:40])) if r.get('pm_hit_task') else ""
            print(f"  [{len(results)}/{len(tasks)}] {r['task']} L{r['level']} success={r['success']} "
                  f"latency={r['latency_s']}s calls={r['llm_calls']} used_pm={r['used_pm']}{hit}", flush=True)
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
    ap.add_argument("--pm-bank", dest="pm_bank",
                    default=os.path.join(os.path.dirname(__file__), "pm_bank.json"),
                    help="seed PM from this bank of past successful trajectories (default: concurrent_mm/pm_bank.json)")
    ap.add_argument("--tag", default="", help="save traces/results to one_traces_<tag>/ one_results_<tag>/")
    args = ap.parse_args()
    if args.tag:                                            # isolate this run's traces/results
        globals()["TRACE_DIR"] = os.path.join(os.path.dirname(__file__), f"one_traces_{args.tag}")
        globals()["RESULTS_DIR"] = os.path.join(os.path.dirname(__file__), f"one_results_{args.tag}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
