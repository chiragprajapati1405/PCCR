"""M1: Multi-agent OfficeBench runner (orchestrator + per-app agents), NO memory.

Replaces OfficeBench's single-agent app-switching loop with a LegoMem-style loop:
  orchestrator decomposes the task and delegates ONE subtask at a time to an app
  agent; the app agent executes that subtask using only its app's actions and
  returns a short result; the orchestrator re-plans until it finishes.

Shared env (one Docker container); agents act via env.exec_action. State-based
evaluation, identical to runner.py. This is the memory-free spine; M2 adds the
two memory injection points, M3 adds the rho-gate.

  source cerebras.env
  python -m officebench_eval.multiagent            # validate on a sample vs single-agent
"""
from __future__ import annotations
import json, os, shutil, time

from .cerebras_llm import CerebrasLLM
from .runner import _setup, cap_for_level

MAX_ORCH = {1: 6, 2: 8, 3: 10}
MAX_AGENT_STEPS = 12

ORCH_SYS = ("You are an ORCHESTRATOR solving an office task by delegating to specialized "
            "app agents. Decompose the task and delegate ONE subtask at a time to the most "
            "appropriate app agent. After each agent result, decide the next subtask or finish. "
            "All needed files already exist in the system under /testbed/data -- operate "
            "AUTONOMOUSLY and NEVER ask the user for anything. If unsure which files exist, "
            "first delegate to the 'shell' agent to list /testbed/data. Avoid the 'llm' agent. "
            "Respond with EXACTLY ONE JSON object and nothing else.")
AGENT_SYS = ("You are an APP AGENT. You execute ONE assigned subtask using ONLY your app's "
             "actions. Emit ONE action at a time as a single JSON object. Fully COMPLETE the "
             "subtask before stopping: read ALL relevant items before concluding (for "
             "earliest/latest/highest/lowest, inspect every candidate); after editing a file, "
             "SAVE it; verify the result. When done, emit "
             "{\"action\":\"subtask_done\",\"result\":\"<the exact value(s) found or what you changed>\"}.")


def _first_json(s):
    """First balanced JSON object (handles concatenated/truncated model output)."""
    i = s.find("{")
    if i < 0:
        return None
    depth = 0; in_str = False; esc = False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:        esc = False
            elif c == "\\": esc = True
            elif c == '"':  in_str = False
        elif c == '"':      in_str = True
        elif c == "{":      depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try: return json.loads(s[i:j + 1])
                except Exception: return None
    return None


def _orch_prompt(task, app_keys, intros, history, mem=""):
    h = ""
    for k, (app, sub, res) in enumerate(history):
        h += f"  {k+1}. [{app}] {sub} -> {str(res)[:160]}\n"
    return (mem +
            f"##Task: {task}\n"
            f"##App agents available:\n{intros}\n"
            f"##Subtasks completed so far:\n{h or '  (none)'}\n"
            "##Decide the NEXT step. Reply with ONE json:\n"
            '  delegate: {"agent":"<app>","subtask":"<what the agent should do>"}\n'
            '  or finish: {"action":"finish","answer":"<answer if the task is a question, else None>"}\n'
            "##Command:")


def _agent_prompt(app, subtask, obs, menu, steps, task="", mem=""):
    h = ""
    for k, (a, o) in enumerate(steps):
        h += f"  {k+1}. {a} -> [{str(o)[:140]}]\n"
    return (mem +
            f"##You are the '{app}' agent.\n"
            f"##Overall task (context): {task}\n"
            f"##Your subtask: {subtask}\n"
            f"##{app} actions you can use:\n{menu}\n"
            f"##Your steps so far:\n{h or '  (none)'}\n"
            f"##Latest observation: {str(obs)[:300]}\n"
            "##Emit ONE json: an app action, or {\"action\":\"subtask_done\",\"result\":\"...\"}\n"
            "##Command:")


def _json_objs(s, maxn=3):
    """Extract up to maxn COMPLETE balanced JSON objects from a string (so we never
    show a truncated/half action). Used to clean curated 'steps' into whole actions."""
    out, i = [], 0
    while len(out) < maxn:
        st = s.find("{", i)
        if st < 0:
            break
        depth = 0; instr = False; esc = False; end = -1
        for j in range(st, len(s)):
            c = s[j]
            if instr:
                if esc: esc = False
                elif c == "\\": esc = True
                elif c == '"': instr = False
            elif c == '"': instr = True
            elif c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = j + 1; break
        if end < 0:
            break
        out.append(s[st:end]); i = end
    return out


def _agent_mem_block(real, app, subtask, exclude_task=None):
    """Subtask-conditioned agent memory: COMPLETE past actions of THIS app from banked
    tasks similar to the assigned subtask (LegoMem-Dynamic; query = subtask, LOO)."""
    if real is None:
        return ""
    acts = real.preload(subtask, k=3, min_score=0.5, exclude_task=exclude_task).get(app, [])
    lines = []
    for sm in acts:
        for obj in _json_objs(sm.action, maxn=2):       # whole action JSONs, never truncated
            lines.append(f" - {obj}")
    if not lines:
        return ""
    return (f"##EXAMPLE {app} actions that worked on a similar subtask "
            f"(reference for FORMAT/approach; adapt the arguments to YOUR subtask):\n"
            + "\n".join(lines[:4]) + "\n\n")


def _run_agent(env, apps, app, subtask, agent_llm, trace, task="", real=None, em=None, exclude_task=None):
    if app not in env.available_apps:
        return f"(no such app agent: {app})"
    env.exec_action(json.dumps({"app": "system", "action": "switch_app", "target_app": app}))
    menu = "".join(f" - {apps.AVAILABLE_ACTIONS[app][a].DEMO}\n"
                   for a in apps.AVAILABLE_ACTIONS.get(app, {}))
    mem = _agent_mem_block(real, app, subtask, exclude_task=exclude_task)
    if mem and em is not None:
        em["agent_consults"] += 1; em["agent_tokens"] += max(1, len(mem) // 4)
    steps, result = [], None
    for _ in range(MAX_AGENT_STEPS):
        d = _first_json(agent_llm.generate(_agent_prompt(app, subtask, env.observation, menu, steps, task, mem)))
        if not d:
            steps.append(("<malformed>", "")); continue
        if d.get("action") == "subtask_done":
            result = d.get("result", "done"); break
        d.setdefault("app", app)
        if d.get("action") in ("switch_app", "finish_task"):     # agent's lane is its app only
            result = "(agent tried to switch/finish; returning to orchestrator)"; break
        a = json.dumps(d)
        env.exec_action(a)
        steps.append((a, env.observation))
        trace.append((f"[{app}] {a}", (env.observation or "")[:300]))
    return result or (env.observation or "done")


def _orch_mem_block(real, task, exclude_task=None):
    """Orchestrator memory: clean high-level plans from banked tasks similar to this
    one. Skips the task's own entry (LOO) and any raw-JSON 'plan' (only natural-language
    steps), and explicitly tells the orchestrator to keep its delegations fully specific."""
    if real is None:
        return ""
    lines = []
    for sc, m in real.search(task, k=6):
        if exclude_task and m.description == exclude_task:
            continue
        steps = [s.strip() for s in m.plan
                 if s.strip() and not s.strip().startswith("{") and '"app"' not in s]
        if not steps:                                   # skip records whose plan is raw JSON
            continue
        lines.append(f" - ({sc:.2f}) {m.description[:70]} | {' ; '.join(steps[:6])}")
        if len(lines) >= 4:
            break
    if not lines:
        return ""
    return ("##SIMILAR PAST TASKS (high-level SHAPE only -- the current task likely differs "
            "in a key detail). Use them to plan the sequence of agents, but write each "
            "delegated subtask FULLY SPECIFIC: include exact file paths and the concrete "
            "operation. NEVER shorten a subtask to a bare action name.\n"
            + "\n".join(lines) + "\n\n")


def _synth_prompt(task, history):
    """Force the orchestrator to commit a final answer from the gathered agent results."""
    h = "".join(f"  {i+1}. [{a}] {s} -> {str(r)[:220]}\n" for i, (a, s, r) in enumerate(history))
    return (f"##Task: {task}\n"
            f"##Results gathered from the app agents:\n{h}\n"
            "##Now COMMIT the final result. If the task asks a QUESTION, answer it using the "
            "results above (copy exact values). If it only creates/modifies files, answer \"None\".\n"
            "##Reply with ONE json and nothing else: {\"answer\":\"<final answer or None>\"}\n##Answer:")


def run_ma(task_id, subtask_id, model="gpt-oss-120b", container="ob-test", real_arch=None):
    _setup()
    import apps
    from utils.env import OfficeAgentEnv
    import utils.evaluate as ev

    config = json.load(open(f"tasks/{task_id}/subtasks/{subtask_id}.json"))
    level = int(task_id.split("-")[0])
    env = OfficeAgentEnv(image_name="officebench", container_name=container,
                         task=config["task"], verbose=False)
    env.reset()
    env.prepare_docker_env(testbed_dir=f"tasks/{task_id}/testbed/", app_dir="apps/")
    env.cache_docker_status(local_cache_dir=f"tasks/{task_id}/cache/{subtask_id}/")

    orch = CerebrasLLM(model_name=model, system_message=ORCH_SYS); orch.max_tokens = 1024
    agent = CerebrasLLM(model_name=model, system_message=AGENT_SYS); agent.max_tokens = 2048
    app_keys = list(env.available_apps.keys())
    intros = "".join(f" - {env.available_apps[a].INTRO}\n" for a in app_keys)

    t0 = time.perf_counter()
    excl = config["task"]                              # leave-one-out: drop the task's own bank entry
    orch_mem = _orch_mem_block(real_arch, config["task"], exclude_task=excl)
    em = {"orch_consult": int(bool(orch_mem)), "orch_tokens": max(1, len(orch_mem) // 4) if orch_mem else 0,
          "agent_consults": 0, "agent_tokens": 0}
    history, trace, answer = [], [], "None"
    seen, dup = {}, 0                                  # loop suppression: signatures already delegated
    for _ in range(MAX_ORCH[level]):
        d = _first_json(orch.generate(_orch_prompt(config["task"], app_keys, intros, history, orch_mem)))
        if not d:
            continue
        if isinstance(d.get("delegate"), dict):       # some outputs wrap as {"delegate": {...}}
            d = d["delegate"]
        if d.get("agent"):                            # a delegation
            app, sub = str(d["agent"]), str(d.get("subtask", ""))
            sig = (app, " ".join(sub.lower().split())[:40])
            if sig in seen:                           # already attempted -> don't loop; nudge, then bail
                dup += 1
                history.append((app, sub, f"(ALREADY DONE earlier; result was: "
                                          f"{str(seen[sig])[:90]}). Delegate a DIFFERENT next step, or finish."))
                if dup >= 2:
                    break                             # stuck in a loop -> force the answer
                continue
            res = _run_agent(env, apps, app, sub, agent, trace, task=config["task"],
                             real=real_arch, em=em, exclude_task=excl)
            seen[sig] = res
            history.append((app, sub, res))
        elif d.get("action") == "finish" or "answer" in d:
            answer = d.get("answer", "None"); break
        # else: unrecognized intent -> retry next round

    # forced answer synthesis: if the loop never committed a finish, derive one from the results
    if str(answer).strip().lower() in ("none", "") and history:
        syn = _first_json(orch.generate(_synth_prompt(config["task"], history)))
        if syn and str(syn.get("answer", "")).strip():
            answer = syn["answer"]
    env.exec_action(json.dumps({"app": "system", "action": "finish_task", "answer": answer}))

    out_dir = f"tasks/{task_id}/outputs/{subtask_id}/multiagent"
    shutil.rmtree(out_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out_dir)
    testbed = os.path.join(out_dir, "testbed")
    ok, failed = True, None
    for item in config["evaluation"]:
        try:
            if not getattr(ev, item["function"])(testbed, item["args"]):
                ok, failed = False, item["function"]; break
        except Exception as e:
            ok, failed = False, f"{item['function']}:ERR:{str(e)[:40]}"; break
    env.close()
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(f"tasks/{task_id}/cache/{subtask_id}", ignore_errors=True)
    return {"task": task_id, "subtask": subtask_id, "level": level, "success": ok,
            "failed_predicate": failed, "orch_rounds": len(history), "em": em,
            "agent_calls": orch.calls + agent.calls, "wall_s": round(time.perf_counter() - t0, 1),
            "delegations": [(a, s) for a, s, _ in history], "trace": trace, "answer": answer,
            "task_text": config["task"], "orch_mem": orch_mem}


def main(per_level=5):
    """M1 validation: multi-agent (no memory) vs single-agent baseline (t0_progress)
    on the SAME stratified TRAIN sample."""
    _P = os.path.dirname(os.path.abspath(__file__))
    t0 = json.load(open(f"{_P}/t0_progress.json"))               # single-agent no-memory base
    train = json.load(open(f"{_P}/split.json"))["train"]
    by_lvl = {1: [], 2: [], 3: []}
    for it in train:
        k = f"{it['task']}/{it['subtask']}"
        if k in t0:                                              # only tasks with a single-agent result
            by_lvl.setdefault(it["level"], []).append(it)
    sample = [it for lvl in (1, 2, 3) for it in by_lvl.get(lvl, [])[:per_level]]
    out = f"{_P}/calibration/ma_m1.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    print(f"M1: multi-agent (no memory) vs single-agent, {len(sample)} train tasks")
    for i, it in enumerate(sample):
        key = f"{it['task']}/{it['subtask']}"
        if key not in res:
            try:
                res[key] = run_ma(it["task"], it["subtask"])
            except Exception as e:
                res[key] = {"success": False, "error": str(e)[:120], "level": it["level"]}
            json.dump(res, open(out, "w"), indent=2)
        r = res[key]; sa = bool(t0.get(key))
        print(f"  [{i+1}/{len(sample)}] {key} L{it['level']}  MA={'OK ' if r.get('success') else 'fail'}"
              f"  SA={'OK' if sa else 'fail'}  rounds={r.get('orch_rounds','-')} calls={r.get('agent_calls','-')}",
              flush=True)
    # head-to-head on the sampled tasks
    keys = [f"{it['task']}/{it['subtask']}" for it in sample if f"{it['task']}/{it['subtask']}" in res]
    ma = sum(bool(res[k].get("success")) for k in keys)
    sa = sum(bool(t0.get(k)) for k in keys)
    print(f"\n=== M1 head-to-head on {len(keys)} tasks ===")
    print(f"  single-agent (t0): {sa}/{len(keys)}")
    print(f"  multi-agent (M1) : {ma}/{len(keys)}")
    print(f"  MA-only wins: {[k for k in keys if res[k].get('success') and not t0.get(k)]}")
    print(f"  SA-only wins: {[k for k in keys if t0.get(k) and not res[k].get('success')]}")


def rerun(keys):
    """Re-run specific task/subtask keys (tuned) -> ma_m1_tuned.json, compare to M1 + single-agent."""
    _P = os.path.dirname(os.path.abspath(__file__))
    t0 = json.load(open(f"{_P}/t0_progress.json"))
    old = json.load(open(f"{_P}/calibration/ma_m1.json")) if os.path.exists(f"{_P}/calibration/ma_m1.json") else {}
    out = f"{_P}/calibration/ma_m1_tuned.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    print(f"TUNED re-run (steps={MAX_AGENT_STEPS}, +full-task context): {len(keys)} tasks")
    for i, key in enumerate(keys):
        t, s = key.split("/")
        if key not in res:
            try: res[key] = run_ma(t, s)
            except Exception as e: res[key] = {"success": False, "error": str(e)[:120]}
            json.dump(res, open(out, "w"), indent=2)
        r = res[key]
        print(f"  [{i+1}/{len(keys)}] {key}  TUNED={'OK ' if r.get('success') else 'fail'}"
              f"  (M1={'OK' if old.get(key,{}).get('success') else 'fail'}, "
              f"SA={'OK' if t0.get(key) else 'fail'})  rounds={r.get('orch_rounds','-')}", flush=True)
    rec = sum(bool(res[k].get("success")) for k in keys)
    print(f"\nTUNED recovered {rec}/{len(keys)} of the L1 failures (were 0/{len(keys)} in M1)")


def m2(per_level=5):
    """M2: same train sample as M1, but memory ON (orch + subtask-conditioned agent).
    Head-to-head vs M1 (no memory) on identical tasks."""
    from .gate import load_calibration
    from .real_arch import RealArch
    _P = os.path.dirname(os.path.abspath(__file__))
    cost, util = load_calibration(f"{_P}/calibration")
    real = RealArch(f"{_P}/em_bank.json", cost, util, theta=0.0)
    t0 = json.load(open(f"{_P}/t0_progress.json"))
    m1 = json.load(open(f"{_P}/calibration/ma_m1.json"))
    train = json.load(open(f"{_P}/split.json"))["train"]
    by_lvl = {1: [], 2: [], 3: []}
    for it in train:
        if f"{it['task']}/{it['subtask']}" in t0:
            by_lvl.setdefault(it["level"], []).append(it)
    sample = [it for lvl in (1, 2, 3) for it in by_lvl.get(lvl, [])[:per_level]]
    out = f"{_P}/calibration/ma_m2.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    print(f"M2: multi-agent + MEMORY vs M1 (no memory), {len(sample)} train tasks")
    for i, it in enumerate(sample):
        key = f"{it['task']}/{it['subtask']}"
        if key not in res:
            try: res[key] = run_ma(it["task"], it["subtask"], real_arch=real)
            except Exception as e: res[key] = {"success": False, "error": str(e)[:120], "level": it["level"]}
            json.dump(res, open(out, "w"), indent=2)
        r = res[key]
        print(f"  [{i+1}/{len(sample)}] {key} L{it['level']}  M2={'OK ' if r.get('success') else 'fail'}"
              f"  M1={'OK' if m1.get(key,{}).get('success') else 'fail'}  "
              f"consults(o/a)={r.get('em',{}).get('orch_consult','-')}/{r.get('em',{}).get('agent_consults','-')}",
              flush=True)
    keys = [f"{it['task']}/{it['subtask']}" for it in sample if f"{it['task']}/{it['subtask']}" in res]
    M2 = sum(bool(res[k].get("success")) for k in keys)
    M1 = sum(bool(m1.get(k, {}).get("success")) for k in keys)
    print(f"\n=== M2 head-to-head on {len(keys)} tasks ===")
    print(f"  multi-agent NO memory (M1): {M1}/{len(keys)}")
    print(f"  multi-agent + memory  (M2): {M2}/{len(keys)}   (memory lift = {M2-M1})")
    print(f"  memory-only wins: {[k for k in keys if res[k].get('success') and not m1.get(k,{}).get('success')]}")
    print(f"  memory-only losses: {[k for k in keys if m1.get(k,{}).get('success') and not res[k].get('success')]}")


def m1b(per_level=5):
    """Clean no-memory baseline with CURRENT code (12 steps) -> ma_m1b.json, so M2's
    memory effect is isolated from the agent-step bump made during tuning."""
    _P = os.path.dirname(os.path.abspath(__file__))
    t0 = json.load(open(f"{_P}/t0_progress.json"))
    train = json.load(open(f"{_P}/split.json"))["train"]
    by_lvl = {1: [], 2: [], 3: []}
    for it in train:
        if f"{it['task']}/{it['subtask']}" in t0:
            by_lvl.setdefault(it["level"], []).append(it)
    sample = [it for lvl in (1, 2, 3) for it in by_lvl.get(lvl, [])[:per_level]]
    out = f"{_P}/calibration/ma_m1b.json"
    res = json.load(open(out)) if os.path.exists(out) else {}
    m2res = json.load(open(f"{_P}/calibration/ma_m2.json")) if os.path.exists(f"{_P}/calibration/ma_m2.json") else {}
    print(f"M1b: clean NO-memory baseline (steps={MAX_AGENT_STEPS}), {len(sample)} tasks")
    for i, it in enumerate(sample):
        key = f"{it['task']}/{it['subtask']}"
        if key not in res:
            try: res[key] = run_ma(it["task"], it["subtask"], real_arch=None)
            except Exception as e: res[key] = {"success": False, "error": str(e)[:120], "level": it["level"]}
            json.dump(res, open(out, "w"), indent=2)
        r = res[key]
        print(f"  [{i+1}/{len(sample)}] {key} L{it['level']}  M1b={'OK ' if r.get('success') else 'fail'}"
              f"  M2={'OK' if m2res.get(key,{}).get('success') else 'fail'}", flush=True)
    keys = [f"{it['task']}/{it['subtask']}" for it in sample if f"{it['task']}/{it['subtask']}" in res and f"{it['task']}/{it['subtask']}" in m2res]
    b = sum(bool(res[k].get("success")) for k in keys)
    m = sum(bool(m2res[k].get("success")) for k in keys)
    print(f"\n=== CLEAN memory effect on {len(keys)} tasks (same 12-step code) ===")
    print(f"  no memory (M1b): {b}/{len(keys)}")
    print(f"  + memory  (M2) : {m}/{len(keys)}   (clean memory lift = {m-b})")
    print(f"  memory-only wins: {[k for k in keys if m2res[k].get('success') and not res[k].get('success')]}")
    print(f"  memory-only losses: {[k for k in keys if res[k].get('success') and not m2res[k].get('success')]}")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["m2"]:
        m2()
    elif sys.argv[1:2] == ["m1b"]:
        m1b()
    elif len(sys.argv) > 1:
        rerun(sys.argv[1:])
    else:
        main()
