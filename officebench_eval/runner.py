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

def cap_for_level(level, l3_cap=30):
    """Step budget scaled to difficulty: L3 pipelines (explore -> 3-4 app
    switches -> multi-file ops) legitimately need more steps than L1 edits.
    Measured: L3 successes use up to 16 steps and ~half its failures were
    budget-bound at a flat 20 cap. Default L3=30 (the paper baseline); NEXT_STEPS
    B4 raises it to 45 via l3_cap (opt-in) for multi-output L3 tasks that need
    budget for the SECOND app after a long cell-by-cell Excel build."""
    return {1: 15, 2: 22, 3: int(l3_cap)}.get(int(level), 22)


def _parse_action(a):
    import ast
    a = (a or "").strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            d = parser(a)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return {}


def format_steps(trajectory):
    """Readable orchestrator-delegation / sub-agent step sequence."""
    lines = []
    for n, (a, o) in enumerate(trajectory, 1):
        d = _parse_action(a)
        app, act = d.get("app", ""), d.get("action", "")
        args = {k: v for k, v in d.items() if k not in ("app", "action")}
        if app == "system" and act == "switch_app":
            lbl = f"ORCHESTRATOR  --delegate-->  [{d.get('target_app')}] agent"
        elif app == "system" and act == "finish_task":
            lbl = f"ORCHESTRATOR  --finish--  answer={d.get('answer')!r}"
        elif app:
            lbl = f"  SUBAGENT[{app}]  --do-->  {act}  {json.dumps(args)[:120]}"
        else:
            lbl = f"  (unparsed) {str(a)[:100]}"
        lines.append(f"step {n:>2}: {lbl}")
        lines.append(f"         OBSERVATION: {str(o)[:200]}")
    return lines


_READY = False
def _setup():
    global _READY
    if _READY:
        return
    sys.path.insert(0, OB)
    sys.path.insert(0, REPO)
    os.chdir(OB)
    _patch_intercode_timeout()
    _READY = True


def _patch_intercode_timeout():
    """Make intercode's signal-based `timeout` thread-safe (version-controlled here
    because OfficeBench is a gitignored external clone). env.step wraps every action
    in `with timeout()`, which uses signal.alarm -- main-thread only. Under the parallel
    driver's asyncio.to_thread (worker thread) it raises 'signal only works in main
    thread', which env.step catches and reports as 'Malformed action!' on EVERY action
    (89% malformed -> cap-hit failures). We replace __enter__/__exit__ with main-thread
    guards: main thread keeps the exact timeout (sequential/baseline unchanged); worker
    threads skip it (Docker exec + the step cap are the backstop)."""
    import threading
    try:
        from intercode.utils.utils import timeout as _t
    except Exception:
        return
    if getattr(_t, "_thread_safe", False):
        return
    import signal as _sig

    def _enter(self):
        if threading.current_thread() is threading.main_thread():
            _sig.signal(_sig.SIGALRM, self.handle_timeout)
            _sig.alarm(self.seconds)

    def _exit(self, *a):
        if threading.current_thread() is threading.main_thread():
            _sig.alarm(0)

    _t.__enter__, _t.__exit__, _t._thread_safe = _enter, _exit, True


def run_task(task_id, subtask_id, model="gpt-oss-120b", real_arch=None, method="no_memory",
             pattern=None, max_iter=20, container="ob-run", exclude_task=None, real_mem=None,
             replay=False, replay_threshold=0.75, plan_then_execute=False, batch_size=4,
             output_convention=False, completion_gate=False, self_verify=False, inject_once=False):
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
    # native eval layout: cache the INITIAL state (some evaluators diff against
    # tasks/<id>/cache/<sub>/) before the agent acts
    env.cache_docker_status(local_cache_dir=f"tasks/{task_id}/cache/{subtask_id}/")
    use_pm = method.endswith("_pm")               # legacy approximation flag
    gate_method = method[:-3] if use_pm else method
    policy = make_pccr_policy(LLMPolicy, model, env, config, real_arch=real_arch,
                              method=gate_method, pattern=pattern, exclude_task=exclude_task,
                              use_pm=use_pm, real_mem=real_mem,   # real_mem -> real MemoryManager gate arm
                              replay=replay, replay_threshold=replay_threshold,
                              plan_then_execute=plan_then_execute, batch_size=batch_size,
                              output_convention=output_convention,
                              completion_gate=completion_gate, self_verify=self_verify,
                              inject_once=inject_once)

    t0 = time.perf_counter()
    done, n, steps = False, 0, []
    while not done and n < max_iter:
        n += 1
        action = policy.forward(env)
        obs, reward, done, info = env.step(action)
        steps.append((action, obs))

    # cache FINAL state to the native path so eval args like
    # "../../../../reference/x" and "../cache/<sub>/" resolve correctly
    out_dir = f"tasks/{task_id}/outputs/{subtask_id}/{method}"
    shutil.rmtree(out_dir, ignore_errors=True)
    env.cache_docker_status(local_cache_dir=out_dir)          # -> out_dir/testbed
    testbed = os.path.join(out_dir, "testbed")
    failed, ok = None, True
    for item in config["evaluation"]:
        try:
            if not getattr(ev, item["function"])(testbed, item["args"]):
                ok, failed = False, item["function"]
                break
        except Exception as e:                                # never crash the task -> always trace
            ok, failed = False, f"{item['function']}:ERR:{str(e)[:50]}"
            break
    env.close()
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(f"tasks/{task_id}/cache/{subtask_id}", ignore_errors=True)

    return {
        "task": task_id, "subtask": subtask_id, "level": int(task_id.split("-")[0]),
        "pattern": pattern, "method": method, "success": ok, "failed_predicate": failed,
        "steps": n, "llm_calls": policy.llm.calls, "wall_s": round(time.perf_counter() - t0, 1),
        # REAL billed tokens (whole prompt + completion, from the API usage field) -- the true
        # API cost, vs em.injected_tokens which counts only the injected-memory slice.
        "prompt_tokens": getattr(policy.llm, "prompt_tokens", 0),
        "completion_tokens": getattr(policy.llm, "completion_tokens", 0),
        # 429 accounting: compute_s = wall minus time lost to rate-limit stalls (user wants
        # latency with 429s neglected). rate_limit_wait_s = failed round-trips + pacing sleeps.
        "rate_limit_wait_s": round(getattr(policy.llm, "rate_limit_wait_s", 0.0), 1),
        "rate_limit_hits": getattr(policy.llm, "rate_limit_hits", 0),
        "em": policy.em_trace,
        # FULL action (untruncated) so the EM bank's plans + agent-index parse are
        # lossless on long actions (email bodies, multi-field creates); observations
        # truncated to keep traces/bank a sane size.
        "trajectory": [(str(a), (o or "")[:400]) for a, o in steps],
        "sequence": format_steps([(str(a), o) for a, o in steps]),  # readable step-by-step
        "task_text": config["task"],
    }


if __name__ == "__main__":
    tid = sys.argv[1] if len(sys.argv) > 1 else "1-1"
    sid = sys.argv[2] if len(sys.argv) > 2 else "0"
    r = run_task(tid, sid, method="no_memory")
    print(f"\n{tid}/{sid}: success={r['success']} steps={r['steps']} "
          f"calls={r['llm_calls']} wall={r['wall_s']}s")
