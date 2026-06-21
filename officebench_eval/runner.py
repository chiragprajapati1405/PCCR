"""Per-task runner: drive ONE OfficeBench subtask with the PCCR policy in the
real Docker env, evaluate natively, return a full trace.

Reusable by T0 (collect training trajectories, no_memory) and T1 (compare
methods). Self-contained: sets DOCKER_HOST + OfficeBench sys.path.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OB = os.path.join(REPO, "OfficeBench")

if not os.environ.get("DOCKER_HOST"):
    _sock = os.path.expanduser("~/.colima/default/docker.sock")
    if os.path.exists(_sock):
        os.environ["DOCKER_HOST"] = f"unix://{_sock}"

_READY = False
def _setup():
    global _READY
    if _READY:
        return
    sys.path.insert(0, OB)
    sys.path.insert(0, REPO)
    os.chdir(OB)
    _READY = True


def run_task(task_id, subtask_id, model="gpt-oss-120b", memory=None, method="no_memory",
             stores=frozenset(), pattern=None, max_iter=20, container="ob-run",
             exclude_task=None):
    _setup()
    from utils.env import OfficeAgentEnv
    from utils.policies import LLMPolicy
    import utils.evaluate as ev
    from officebench_eval.pccr_policy import make_pccr_policy

    config = json.load(open(f"tasks/{task_id}/subtasks/{subtask_id}.json"))
    env = OfficeAgentEnv(image_name="officebench", container_name=container,
                         task=config["task"], verbose=False)
    env.reset()
    env.prepare_docker_env(testbed_dir=f"tasks/{task_id}/testbed/", app_dir="apps/")
    policy = make_pccr_policy(LLMPolicy, model, env, config, memory, method, stores,
                              exclude_task=exclude_task)

    t0 = time.perf_counter()
    done, n, steps = False, 0, []
    while not done and n < max_iter:
        n += 1
        action = policy.forward(env)
        obs, reward, done, info = env.step(action)
        steps.append((action, obs))

    out = f"/tmp/ob_{container}_{task_id}_{subtask_id}"
    shutil.rmtree(out, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out)
    testbed = os.path.join(out, "testbed")
    failed = None
    ok = True
    for item in config["evaluation"]:
        if not getattr(ev, item["function"])(testbed, item["args"]):
            ok, failed = False, item["function"]
            break
    env.close()

    return {
        "task": task_id, "subtask": subtask_id, "level": int(task_id.split("-")[0]),
        "pattern": pattern, "method": method, "success": ok, "failed_predicate": failed,
        "steps": n, "llm_calls": policy.llm.calls, "wall_s": round(time.perf_counter() - t0, 1),
        "em": policy.em_trace,
        # FULL action (untruncated) so the EM bank's plans + agent-index parse are
        # lossless on long actions (email bodies, multi-field creates); observations
        # truncated to keep traces/bank a sane size.
        "trajectory": [(str(a), (o or "")[:400]) for a, o in steps],
        "task_text": config["task"],
    }


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    sid = sys.argv[2] if len(sys.argv) > 2 else "0"
    r = run_task(tid, sid, method="no_memory")
    print(f"\n{tid}/{sid}: success={r['success']} steps={r['steps']} "
          f"calls={r['llm_calls']} wall={r['wall_s']}s")
