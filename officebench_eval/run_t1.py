"""T1: main comparison on the 152 test subtasks.

Methods: no_memory / retrieve_all / pccr (rho-gate). Uses the EM bank (T0a) and
the calibrated cost+utility (T0c, with prior fallback). Checkpointed + saves a
trace per (subtask, method). Reports success / consults / injected tokens by
LEVEL (the LegoMem-style breakdown).

  source cerebras.env
  python -m officebench_eval.run_t1 --theta 0.5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict

from .gate import load_calibration, stores_for
from .memory import ProcedureMemory
from .runner import run_task, cap_for_level

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
RESULTS = os.path.join(_PKG, "results")
TRACE_DIR = os.path.join(_PKG, "traces/t1")
PROGRESS = os.path.join(_PKG, "t1_progress.json")
METHODS = ("no_memory", "retrieve_all", "pccr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mem = ProcedureMemory()
    mem.load(BANK)
    print(f"loaded EM bank: {len(mem)} procedures")
    cost, util = load_calibration(CALIB := os.path.join(_PKG, "calibration"))
    print(f"cost={cost}")

    test = json.load(open(SPLIT))["test"]
    if args.limit:
        test = test[:args.limit]
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    os.makedirs(TRACE_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    prog = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}

    t0 = time.perf_counter()
    for i, it in enumerate(test):
        pat = pats.get((it["task"], it["subtask"]), "multi_app")
        for m in METHODS:
            key = f"{it['task']}/{it['subtask']}|{m}"
            if key in prog:
                continue
            stores = stores_for(m, pat, args.theta, cost, util)
            try:
                r = run_task(it["task"], it["subtask"], model=args.model, memory=mem,
                             method=m, stores=stores, pattern=pat, max_iter=cap_for_level(it['level']),
                             container="ob-test")
            except Exception as e:
                print(f"  {key} ERR {str(e)[:80]}", flush=True)
                prog[key] = {"success": False, "error": str(e)[:120]}
                json.dump(prog, open(PROGRESS, "w"))
                continue
            base = f"{TRACE_DIR}/{it['task']}_{it['subtask']}_{m}"
            json.dump(r, open(f"{base}.json", "w"), indent=2)
            with open(f"{base}.txt", "w") as fh:              # readable step-by-step
                fh.write(f"TASK {it['task']}/{it['subtask']} [{pat}] {m}  "
                         f"success={r['success']}  stores={sorted(stores)}\n"
                         f"  {r['task_text']}\n" + "=" * 70 + "\n"
                         + "\n".join(r["sequence"]) + "\n")
            prog[key] = {"success": r["success"], "level": it["level"], "pattern": pat,
                         "consults": len(stores), "tokens": r["em"]["injected_tokens"],
                         "steps": r["steps"]}
            json.dump(prog, open(PROGRESS, "w"))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test)}] {(time.perf_counter()-t0)/60:.0f} min", flush=True)

    _report(prog)


def _report(prog):
    # aggregate by method, overall + by level
    agg = {m: defaultdict(lambda: [0, 0, 0, 0]) for m in METHODS}  # level -> [succ,n,consults,tokens]
    for key, r in prog.items():
        if "level" not in r:
            continue
        m = key.split("|")[1]
        for lvl in (r["level"], 0):                       # 0 = overall
            a = agg[m][lvl]
            a[0] += int(r["success"]); a[1] += 1
            a[2] += r.get("consults", 0); a[3] += r.get("tokens", 0)
    print(f"\n{'='*68}\nT1 RESULTS (success | consults | inj.tokens)\n{'='*68}")
    for lvl in (0, 1, 2, 3):
        tag = "ALL" if lvl == 0 else f"L{lvl}"
        print(f"\n[{tag}]")
        for m in METHODS:
            a = agg[m][lvl]
            if a[1]:
                print(f"  {m:<14} {a[0]:>3}/{a[1]:<3} acc={a[0]/a[1]:.3f}  "
                      f"consults={a[2]:<4} tokens={a[3]}")
    json.dump({m: {str(l): agg[m][l] for l in agg[m]} for m in METHODS},
              open(f"{RESULTS}/t1_summary.json", "w"), indent=2)
    print(f"\nsaved {RESULTS}/t1_summary.json")


if __name__ == "__main__":
    main()
