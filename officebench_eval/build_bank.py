"""T0a: build the EM procedure bank from the 148 train subtasks.

Runs each train subtask with no_memory; SUCCESSFUL trajectories become EM
procedures. Checkpointed (skips finished tasks on restart) + saves a trace per
task. Output: officebench_eval/em_bank.json.

  source cerebras.env
  python -m officebench_eval.build_bank            # full 148 (resumable)
  python -m officebench_eval.build_bank --limit 3  # quick script check
"""
from __future__ import annotations

import argparse
import json
import os
import time

from .memory import ProcedureMemory
from .runner import run_task, cap_for_level

# absolute paths (runner.run_task chdir's into OfficeBench/, so relative paths break)
_PKG = os.path.dirname(os.path.abspath(__file__))
SPLIT = os.path.join(_PKG, "split.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
BANK = os.path.join(_PKG, "em_bank.json")
PROGRESS = os.path.join(_PKG, "t0_progress.json")
TRACE_DIR = os.path.join(_PKG, "traces/t0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)      # 0 = all
    args = ap.parse_args()

    train = json.load(open(SPLIT))["train"]
    if args.limit:
        train = train[:args.limit]
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    os.makedirs(TRACE_DIR, exist_ok=True)

    done = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}
    mem = ProcedureMemory()
    if os.path.exists(BANK):
        mem.load(BANK)  # dedupe by task

    t0, ok = time.perf_counter(), sum(1 for v in done.values() if v)
    for i, it in enumerate(train):
        key = f"{it['task']}/{it['subtask']}"
        if key in done:
            continue
        pat = pats.get((it["task"], it["subtask"]), "multi_app")
        try:
            r = run_task(it["task"], it["subtask"], model=args.model, method="no_memory",
                         pattern=pat, max_iter=cap_for_level(it['level']), container="ob-train")
        except Exception as e:
            print(f"  [{i+1}/{len(train)}] {key} ERROR {str(e)[:90]}", flush=True)
            done[key] = False
            json.dump(done, open(PROGRESS, "w"))
            continue
        base = f"{TRACE_DIR}/{it['task']}_{it['subtask']}"
        json.dump(r, open(f"{base}.json", "w"), indent=2)
        with open(f"{base}.txt", "w") as fh:                  # readable step-by-step
            fh.write(f"TASK {it['task']}/{it['subtask']} [{pat}] no_memory  success={r['success']}\n"
                     f"  {r['task_text']}\n" + "=" * 70 + "\n" + "\n".join(r["sequence"]) + "\n")
        if r["success"]:
            ok += 1
            mem.add(r["task_text"], pat, it["level"], r["trajectory"])
            json.dump(mem.records, open(BANK, "w"), indent=2)
        done[key] = r["success"]
        json.dump(done, open(PROGRESS, "w"))
        print(f"  [{i+1}/{len(train)}] L{it['level']} {key} [{pat}] "
              f"success={r['success']} steps={r['steps']} | {ok} ok, "
              f"{len(mem.records)} procedures", flush=True)

    print(f"\nT0a done: {ok} train successes -> {len(mem.records)} EM procedures | "
          f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
