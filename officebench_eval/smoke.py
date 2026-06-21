"""Substrate smoke test: run ONE OfficeBench subtask end-to-end in the real
Docker env with the Cerebras backbone (no memory), then evaluate natively.

Validates: container boot, app availability, the agent loop, and state-based
evaluation — before we build memory / parallelism on top.

  source cerebras.env
  python -m officebench_eval.smoke 1-1 0
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OB = os.path.join(REPO, "OfficeBench")

# point the docker SDK at colima's socket if DOCKER_HOST isn't already set
if not os.environ.get("DOCKER_HOST"):
    sock = os.path.expanduser("~/.colima/default/docker.sock")
    if os.path.exists(sock):
        os.environ["DOCKER_HOST"] = f"unix://{sock}"


def run_one(task_id="1-1", subtask_id="0", model="gpt-oss-120b", max_iter=15):
    sys.path.insert(0, OB)
    sys.path.insert(0, REPO)
    os.chdir(OB)                                   # OfficeBench uses relative paths
    from utils.env import OfficeAgentEnv
    from utils.policies import LLMPolicy
    import utils.evaluate as ev                     # evaluation.py main() is py3.12-only; use fns directly
    from officebench_eval.cerebras_llm import CerebrasLLM

    def evaluate_output(testbed_dir, cfg):
        for item in cfg["evaluation"]:
            fn = getattr(ev, item["function"])
            if not fn(testbed_dir, item["args"]):
                print(f"  eval FAILED: {item['function']} {item['args']}")
                return False
        return True

    config = json.load(open(f"tasks/{task_id}/subtasks/{subtask_id}.json"))
    print(f"TASK {task_id}/{subtask_id}: {config['task'][:100]}")

    env = OfficeAgentEnv(image_name="officebench", container_name="ob-smoke",
                         task=config["task"], verbose=False)
    env.reset()
    env.prepare_docker_env(testbed_dir=f"tasks/{task_id}/testbed/", app_dir="apps/")
    # placeholder branch-name avoids LLMPolicy's 'gpt'/'gemini' OpenAI/Gemini paths
    # (which need their own keys); we then swap in the real Cerebras client.
    policy = LLMPolicy(model_name="local-oss", key="", env=env, config=config)
    policy.llm = CerebrasLLM(model_name=model, system_message=policy.system_message)  # real model id

    t0 = time.perf_counter()
    done, n = False, 0
    while not done and n < max_iter:
        n += 1
        action = policy.forward(env)
        obs, reward, done, info = env.step(action)
        print(f"  step {n}: {str(action)[:80]} -> {str(obs)[:80]}")

    out = f"/tmp/ob_smoke_{task_id}_{subtask_id}"
    shutil.rmtree(out, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out)               # creates out/testbed
    testbed = os.path.join(out, "testbed")
    ok = evaluate_output(testbed, config)
    env.close()
    print(f"\nRESULT {task_id}/{subtask_id}: success={ok}  steps={n}  "
          f"llm_calls={policy.llm.calls}  wall={time.perf_counter()-t0:.1f}s")
    return ok


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    sid = sys.argv[2] if len(sys.argv) > 2 else "0"
    run_one(tid, sid)
