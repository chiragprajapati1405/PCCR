"""AppWorld Phase-1 adapter + bare code-action agent.

Bridges AppWorld's interface (agent action = a Python code string calling apis.<app>.<method>) to a
simple step loop, the AppWorld analogue of officebench_eval/runner.py. This is the substrate the
PCCR + 2D + DAG layers plug into next (Phase 2-3). Run:
    set -a; source cerebras.env; set +a
    .venv311/bin/python -m appworld_eval.aw_runner <task_id|index>
"""
import re
import sys
import time

from appworld import AppWorld, load_task_ids
from officebench_eval.cerebras_llm import CerebrasLLM

SYSTEM = (
    "You complete tasks by writing Python code in a REPL, one focused step per turn.\n"
    "- Call app APIs as: apis.<app>.<method>(arg=..., ...).\n"
    "- Discover APIs: apis.api_docs.show_app_descriptions(); "
    "apis.api_docs.show_api_descriptions(app_name='X'); "
    "apis.api_docs.show_api_doc(app_name='X', api_name='Y').\n"
    "- Your (supervisor) info + credentials: apis.supervisor.show_profile(), show_addresses(), "
    "show_payment_cards(), show_account_passwords(). Log in to apps with those passwords.\n"
    "- Apps include: supervisor, api_docs, amazon, phone, file_system, spotify, venmo, gmail, "
    "todoist, simple_note.\n"
    "- print(...) anything you need to observe; the printed output is returned to you next turn.\n"
    "- FINISH with apis.supervisor.complete_task(answer=<value>) for a question, or "
    "apis.supervisor.complete_task() for an action task.\n"
    "Reply with ONE python code block (```python ... ```) for the next step only."
)


def _code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()


def run_task(task_id, llm, max_steps=25, verbose=False, em=None):
    """Drive one AppWorld task with the code-action agent. If `em` (AppWorldEM) is given, retrieve
    the memory block ONCE per task (PCCR inject-once) and include it in every step's prompt."""
    t0 = time.perf_counter()
    result = {"task_id": task_id}
    # NOTE: AppWorld patches the `time` module inside its context (freezegun), so perf_counter is
    # unreliable in there -- measure wall AFTER the `with` exits (time is restored on __exit__).
    with AppWorld(task_id=task_id, experiment_name="pccr_aw_p1") as w:
        instr = w.task.instruction
        mem = ""
        if em is not None:                       # inject-once: retrieve relevant solved procedures
            mem = em.inject_block(instr, k=2, exclude_task=task_id, max_chars=1600)
        mem_ctx = ("\n\n" + mem + "\n") if mem else ""
        history, steps = [], 0
        for steps in range(1, max_steps + 1):
            prompt = (SYSTEM + mem_ctx + "\n\nTASK: " + instr + "\n\n"
                      + "\n\n".join(history[-8:]) + "\n\nNext python code:")
            code = _code(llm.generate(prompt))
            try:
                out = w.execute(code)
            except Exception as e:
                out = "EXECUTE ERROR: %s" % str(e)[:300]
            history.append("CODE:\n%s\nOUTPUT:\n%s" % (code, str(out)[:900]))
            if verbose:
                print("  step %d:\n    %s\n    -> %s" % (steps, code[:160].replace("\n", " "),
                                                        str(out)[:160].replace("\n", " ")), flush=True)
            if w.task_completed():
                break
        report = w.evaluate()
        result.update({"instruction": instr, "completed": w.task_completed(),
                       "injected_tokens": max(0, len(mem) // 4),
                       "success": bool(getattr(report, "success", False)),
                       "pass_count": getattr(report, "pass_count", None),
                       "num_tests": getattr(report, "num_tests", None),
                       "steps": steps, "report": report, "history": history})
    result["wall_s"] = round(time.perf_counter() - t0, 1)     # outside context -> time restored
    result.update({"prompt_tokens": getattr(llm, "prompt_tokens", 0),
                   "completion_tokens": getattr(llm, "completion_tokens", 0),
                   "calls": getattr(llm, "calls", 0)})
    return result


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "0"
    tids = load_task_ids("train")
    tid = arg if arg in tids else tids[int(arg)]
    llm = CerebrasLLM(model_name="gpt-oss-120b", system_message=SYSTEM)
    llm.max_tokens = 4096                        # gpt-oss reasons before answering; give it room
    r = run_task(tid, llm, verbose=True)
    print("\n=== RESULT ===")
    print("task=%s  success=%s  (%s/%s tests)  completed=%s  steps=%d  calls=%d  wall=%.1fs"
          % (r["task_id"], r["success"], r["pass_count"], r["num_tests"], r["completed"],
             r["steps"], r["calls"], r["wall_s"]))
    print("instruction:", r["instruction"][:200])
