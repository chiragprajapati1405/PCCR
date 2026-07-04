"""AppWorld Phase 3 step 1: no-memory vs PCCR-memory on dev.

Runs each dev task twice -- bare code-action agent vs the same agent with EM procedures injected
(inject-once) -- and reports the accuracy lift. The first direct test of PCCR memory on AppWorld.
Run:  set -a; source cerebras.env; set +a ; .venv311/bin/python -m appworld_eval.aw_compare [N]
"""
import json
import os
import sys

from appworld import load_task_ids
from officebench_eval.cerebras_llm import CerebrasLLM
from appworld_eval.aw_runner import run_task, SYSTEM
from appworld_eval.aw_memory import AppWorldEM

OUT = "appworld_eval/dev_nomem_vs_mem.json"


def _fresh_llm():
    llm = CerebrasLLM(model_name="gpt-oss-120b", system_message=SYSTEM)
    llm.max_tokens = 4096
    return llm


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    dev = load_task_ids("dev")[:n]
    em = AppWorldEM()
    prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
    print("no-mem vs mem on %d dev tasks\n" % len(dev), flush=True)
    for i, tid in enumerate(dev, 1):
        if tid in prog:
            continue
        rn = run_task(tid, _fresh_llm(), em=None)          # no memory
        rm = run_task(tid, _fresh_llm(), em=em)            # PCCR memory (inject-once)
        prog[tid] = {"nomem": rn["success"], "mem": rm["success"],
                     "nomem_pass": rn["pass_count"], "mem_pass": rm["pass_count"], "num_tests": rn["num_tests"],
                     "nomem_calls": rn["calls"], "mem_calls": rm["calls"], "inj": rm["injected_tokens"],
                     "instruction": rn["instruction"][:80]}
        json.dump(prog, open(OUT, "w"), indent=1)
        nm = sum(1 for v in prog.values() if v["nomem"]); mm = sum(1 for v in prog.values() if v["mem"])
        print("  [%d/%d] %s  nomem=%s mem=%s  (running: nomem %d, mem %d)"
              % (i, len(dev), tid, rn["success"], rm["success"], nm, mm), flush=True)
    d = list(prog.values())
    nm, mm = sum(v["nomem"] for v in d), sum(v["mem"] for v in d)
    print("\n=== DEV %d tasks ===" % len(d))
    print("  NO-MEMORY : %d/%d (%.0f%%)" % (nm, len(d), 100 * nm / max(1, len(d))))
    print("  PCCR-MEM  : %d/%d (%.0f%%)   [lift %+d]" % (mm, len(d), 100 * mm / max(1, len(d)), mm - nm))
    print("  mean injected tokens: %d" % (sum(v["inj"] for v in d) // max(1, len(d))))
    print("DONE")


if __name__ == "__main__":
    main()
