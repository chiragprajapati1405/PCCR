"""Live terminal view of ONE task: shows rho-gated memory retrieval, then each
step framed as orchestrator delegation -> sub-agent action -> observation.

  source cerebras.env
  python -m officebench_eval.watch 3-1 0 pccr      # task, subtask, method
  python -m officebench_eval.watch 1-1 0 no_memory

(switch_app = orchestrator delegating to an app; an app action = that sub-agent
performing. Uses its own container 'ob-watch', so it won't disturb the chain.)
"""
from __future__ import annotations

import json
import os
import sys

from .gate import load_calibration, stores_for
from .memory import ProcedureMemory
from .runner import OB, REPO, _setup


def watch(task_id, subtask_id, method="pccr", theta=0.5, model="gpt-oss-120b", max_iter=20):
    _setup()
    from utils.env import OfficeAgentEnv
    from utils.policies import LLMPolicy
    from officebench_eval.pccr_policy import make_pccr_policy

    config = json.load(open(f"tasks/{task_id}/subtasks/{subtask_id}.json"))
    mem, stores, pat = None, frozenset(), "-"
    if method != "no_memory":
        mem = ProcedureMemory()
        bp = os.path.join(REPO, "officebench_eval/em_bank.json")
        if os.path.exists(bp):
            mem.load(bp)
        pats = {(p["task"], p["subtask"]): p["pattern"]
                for p in json.load(open(os.path.join(REPO, "officebench_eval/patterns.json")))}
        pat = pats.get((task_id, subtask_id), "multi_app")
        cost, util = load_calibration(os.path.join(REPO, "officebench_eval/calibration"))
        stores = stores_for(method, pat, theta, cost, util)

    bar = "=" * 80
    print(f"\n{bar}\nTASK {task_id}/{subtask_id}  pattern=[{pat}]  method={method}  "
          f"gated_stores={sorted(stores)}\n  {config['task']}\n{bar}")

    env = OfficeAgentEnv(image_name="officebench", container_name="ob-watch",
                         task=config["task"], verbose=False)
    env.reset()
    env.prepare_docker_env(testbed_dir=f"tasks/{task_id}/testbed/", app_dir="apps/")
    policy = make_pccr_policy(LLMPolicy, model, env, config, mem, method, stores)

    if mem and "orchestrator" in stores:
        hits = mem.retrieve_orchestrator(config["task"], 5, exclude_task=config["task"])
        print("\n[MEMORY → orchestrator] retrieved past task plans:")
        for sc, r in hits:
            print(f"   ({sc:.2f}) {r['task'][:62]} | plan: {' ; '.join(r['plan'][:5])[:110]}")

    done, n = False, 0
    while not done and n < max_iter:
        n += 1
        action = policy.forward(env)
        try:
            ad = json.loads(policy.proc_action(action))
        except Exception:
            ad = {}
        app, act = ad.get("app", ""), ad.get("action", "")
        args = {k: v for k, v in ad.items() if k not in ("app", "action")}
        if app == "system" and act == "switch_app":
            label = f"ORCHESTRATOR  ──delegate──>  [{ad.get('target_app')}] agent"
        elif app == "system" and act == "finish_task":
            label = f"ORCHESTRATOR  ──finish──  answer={ad.get('answer')!r}"
        elif app:
            label = f"  SUBAGENT[{app}]  ──do──>  {act}  {json.dumps(args)[:90]}"
        else:
            label = f"  (raw) {str(action)[:90]}"

        obs, reward, done, info = env.step(action)
        mem_note = ""
        if mem and "agent" in stores and app and app != "system":
            ah = mem.retrieve_agent(config["task"], app, 3, exclude_task=config["task"])
            if ah:
                mem_note = f"   [MEMORY → {app} agent: {ah[0][1]['text'][:60]}]"
        print(f"\nstep {n:>2}: {label}{mem_note}")
        print(f"         OBSERVATION: {str(obs)[:170]}")

    env.close()
    print(f"\n{bar}\n(finished in {n} steps)\n")


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    sid = sys.argv[2] if len(sys.argv) > 2 else "0"
    method = sys.argv[3] if len(sys.argv) > 3 else "pccr"
    watch(tid, sid, method)
