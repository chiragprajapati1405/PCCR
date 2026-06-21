"""Re-run the cap-bound budget-failure tasks (cap_bound_tasks.json) at a higher
step cap (default 50); add new successes to the EM bank. Run after the break.
  source cerebras.env
  python -m officebench_eval.rerun_capbound --max-iter 50
"""
import argparse, json, os
from .memory import ProcedureMemory
from .runner import run_task
_PKG = os.path.dirname(os.path.abspath(__file__))
LIST, BANK = os.path.join(_PKG, "cap_bound_tasks.json"), os.path.join(_PKG, "em_bank.json")
PROGRESS, TRACE = os.path.join(_PKG, "t0_progress.json"), os.path.join(_PKG, "traces/t0")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iter", type=int, default=50)
    ap.add_argument("--model", default="gpt-oss-120b")
    a = ap.parse_args()
    tasks = json.load(open(LIST))
    mem = ProcedureMemory()
    if os.path.exists(BANK): mem.load(BANK)
    prog = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}
    for i, it in enumerate(tasks):
        key = f"{it['task']}/{it['subtask']}"
        r = run_task(it["task"], it["subtask"], model=a.model, method="no_memory",
                     pattern=it["pattern"], max_iter=a.max_iter, container="ob-rerun")
        base = f"{TRACE}/{it['task']}_{it['subtask']}"
        json.dump(r, open(base + ".json", "w"), indent=2)
        open(base + ".txt", "w").write("\n".join(r["sequence"]))
        if r["success"]:
            mem.add(r["task_text"], it["pattern"], it["level"], r["trajectory"])
            seen = {x["task"]: x for x in mem.records}; mem.records = list(seen.values())
            json.dump(mem.records, open(BANK, "w"), indent=2)
        prog[key] = r["success"]; json.dump(prog, open(PROGRESS, "w"))
        print(f"[{i+1}/{len(tasks)}] {key} success={r['success']} steps={r['steps']}", flush=True)
    print(f"rerun done; bank now {len(mem.records)} procedures")

if __name__ == "__main__":
    main()
