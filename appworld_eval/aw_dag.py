"""AppWorld Phase 3.2: the DAG re-plan ORCHESTRATION dimension on AppWorld, memory-backed.

AppWorld is a single stateful REPL, so the 2D *parallel sub-agent* part doesn't transfer (no shared-
state fork/merge). What transfers is the orchestrator RE-PLAN loop: a planner emits the next sub-goal
given progress + memory; a memory-backed executor accomplishes it via w.execute; repeat until FINISH.
This is the AppWorld analogue of run_task_2d(replan=True) + PCCR leaves, over one REPL.

Run:  set -a; source cerebras.env; set +a ; .venv311/bin/python -m appworld_eval.aw_dag [N]
Compares DAG-orchestrator+memory vs the flat memory agent (from dev_nomem_vs_mem.json).
"""
import json
import os
import re
import sys
import time

from appworld import AppWorld, load_task_ids
from officebench_eval.cerebras_llm import CerebrasLLM
from appworld_eval.aw_runner import SYSTEM
from appworld_eval.aw_memory import AppWorldEM

OUT = "appworld_eval/dev_dag_orch.json"
MAX_ROUNDS = 6
MICRO_STEPS = 4          # code-action steps the executor gets per sub-goal


def _code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text or "", re.S)
    return (m.group(1) if m else (text or "")).strip()


def plan_next_subgoal(instr, mem, progress, llm):
    """Orchestrator round: emit the NEXT high-level sub-goal (given memory + progress) or FINISH.
    Robust parse: strip code fences / markdown / preamble and take the substantive imperative line;
    a heading like '**Next sub-goal:**' is NOT a sub-goal."""
    p = ("You are the ORCHESTRATOR. Break the AppWorld task into a FEW HIGH-LEVEL sub-goals (login+read "
         "is ONE sub-goal, not two). Look at PROGRESS and output the NEXT sub-goal that is NOT already "
         "done, as ONE plain imperative sentence -- no markdown, no numbering, no preamble, no heading. "
         "Do NOT repeat a step already in PROGRESS. If all needed work is done AND the final answer has "
         "been submitted via complete_task, output exactly: FINISH.\n\n"
         + (mem + "\n\n" if mem else "")
         + "TASK: " + instr + "\n\nPROGRESS:\n" + (progress or "(nothing yet)") + "\n\nNEXT SUB-GOAL:")
    out = re.sub(r"```.*?```", "", (llm.generate(p) or "").strip(), flags=re.S)
    skip = ("next sub-goal", "sub-goal", "here", "okay", "sure", "the next", "plan")
    lines = [re.sub(r"^[\s>*#`\-\d.)]+", "", ln).strip() for ln in out.splitlines()]
    lines = [ln for ln in lines if ln and not ln.lower().startswith(skip)]
    cand = lines[0] if lines else out.strip("* #`\n")
    return None if cand[:8].upper().startswith("FINISH") else cand[:200]


# Executor prompt = the API-usage guidance WITHOUT the flat agent's "finish with complete_task" line,
# so a sub-goal executor never prematurely completes the whole task (the bug that ended tasks after
# the login sub-goal). complete_task is called ONLY by the finalize pass (allow_complete=True).
EXEC_SYSTEM = (
    "You accomplish ONE sub-goal by writing Python code in a REPL, one step per turn.\n"
    "- Call app APIs as apis.<app>.<method>(...); discover via apis.api_docs.show_api_descriptions("
    "app_name='X') / show_api_doc(app_name='X', api_name='Y').\n"
    "- Credentials: apis.supervisor.show_profile()/show_account_passwords(); log in to apps as needed.\n"
    "- print(...) what you find so it carries forward.\n"
)


def run_subgoal(w, subgoal, llm, mem, log, allow_complete=False):
    """Memory-backed executor for ONE sub-goal on the shared REPL. Does NOT complete the whole task
    unless allow_complete (the finalize pass). Returns the observations appended."""
    rule = ("Submit the final answer with apis.supervisor.complete_task(answer=...)."
            if allow_complete else
            "Do NOT call apis.supervisor.complete_task -- only accomplish THIS sub-goal, then "
            "print('SUBGOAL DONE').")
    obs = []
    for _ in range(MICRO_STEPS):
        prompt = (EXEC_SYSTEM + ("\n" + mem if mem else "")
                  + "\n\nOVERALL TASK: " + w.task.instruction
                  + "\n\nCURRENT SUB-GOAL: " + subgoal + "\n" + rule
                  + "\n\nRECENT:\n" + "\n".join(log[-4:]) + "\n\nNext python code:")
        code = _code(llm.generate(prompt))
        try:
            out = w.execute(code)
        except Exception as e:
            out = "EXECUTE ERROR: %s" % str(e)[:200]
        line = "CODE: %s\nOUT: %s" % (code[:200], str(out)[:400])
        log.append(line); obs.append(line)
        if "SUBGOAL DONE" in str(out) or (allow_complete and w.task_completed()):
            break
    return "\n".join(obs)[:600]


def run_task_dag(task_id, llm, em, max_rounds=MAX_ROUNDS):
    t0 = time.perf_counter()
    result = {"task_id": task_id}
    with AppWorld(task_id=task_id, experiment_name="pccr_aw_dag") as w:
        instr = w.task.instruction
        mem = em.inject_block(instr, k=2, exclude_task=task_id, max_chars=1600) if em else ""
        progress, log, rounds = "", [], 0
        for rounds in range(1, max_rounds + 1):
            sub = plan_next_subgoal(instr, mem, progress, llm)
            if sub is None:
                break
            obs = run_subgoal(w, sub, llm, mem, log)
            progress += "\n[round %d] %s\n  -> %s" % (rounds, sub, obs[:300])
            if w.task_completed():
                break
        # FINALIZE: if the orchestrator stopped but nobody submitted the answer, run one executor pass
        # that must compute + submit via apis.supervisor.complete_task (question tasks need this).
        if not w.task_completed():
            run_subgoal(w, "Using everything gathered above, compute the FINAL answer for the overall "
                           "task and submit it with apis.supervisor.complete_task(answer=...).",
                        llm, mem, log, allow_complete=True)
        report = w.evaluate()
        result.update({"instruction": instr, "completed": w.task_completed(), "rounds": rounds,
                       "success": bool(getattr(report, "success", False)),
                       "pass_count": getattr(report, "pass_count", None),
                       "num_tests": getattr(report, "num_tests", None)})
    result["wall_s"] = round(time.perf_counter() - t0, 1)
    result["calls"] = getattr(llm, "calls", 0)
    return result


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dev = load_task_ids("dev")[:n]
    em = AppWorldEM()
    prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
    flat = json.load(open("appworld_eval/dev_nomem_vs_mem.json"))   # flat mem agent baseline
    print("DAG-orchestrator+memory on %d dev tasks\n" % len(dev), flush=True)
    for i, tid in enumerate(dev, 1):
        if tid in prog:
            continue
        llm = CerebrasLLM(model_name="gpt-oss-120b", system_message=SYSTEM); llm.max_tokens = 4096
        r = run_task_dag(tid, llm, em)
        prog[tid] = {"success": r["success"], "rounds": r["rounds"], "calls": r["calls"],
                     "flat_mem": flat.get(tid, {}).get("mem")}
        json.dump(prog, open(OUT, "w"), indent=1)
        dm = sum(1 for v in prog.values() if v["success"])
        print("  [%d/%d] %s  dag=%s (flat_mem=%s)  rounds=%d calls=%d  (running dag %d)"
              % (i, len(dev), tid, r["success"], prog[tid]["flat_mem"], r["rounds"], r["calls"], dm), flush=True)
    d = [v for v in prog.values()]
    dm = sum(v["success"] for v in d)
    fm = sum(1 for v in d if v["flat_mem"])
    print("\n=== DEV %d tasks (gpt-oss-120b) ===" % len(d))
    print("  flat memory agent : %d/%d (%.0f%%)" % (fm, len(d), 100 * fm / max(1, len(d))))
    print("  DAG-orch + memory : %d/%d (%.0f%%)   [%+d vs flat]" % (dm, len(d), 100 * dm / max(1, len(d)), dm - fm))
    print("DONE")


if __name__ == "__main__":
    main()
