"""
Per-task comparison on the SAME 30 held-out tasks: no_memory vs pccr vs
retrieve_all. Answers "is memory load-bearing?" by showing exactly which tasks
flip when memory is consulted.

Reads results_archive/a5_gptoss/{no_memory,pccr,retrieve_all}.json (cost_log).
"""
import json
import os
from collections import defaultdict

A = "results_archive/a5_gptoss"
METHODS = ["no_memory", "pccr", "retrieve_all"]


def load(method):
    p = f"{A}/{method}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    # map task-text -> (success, pattern)
    return {r["task"]: (r["success"], r["pattern"]) for r in d["cost_log"]}


def main():
    data = {m: load(m) for m in METHODS}
    if data["no_memory"] is None:
        print("no_memory.json not ready yet — run still in progress.")
        return
    if data["pccr"] is None:
        print("pccr.json missing.")
        return

    tasks = list(data["pccr"].keys())
    print(f"=== {len(tasks)} held-out tasks: per-task success ===")
    print(f"{'pattern':24s}{'no_mem':8s}{'pccr':6s}{'retr_all':9s}  task")
    helped = []          # memory passed where no-memory failed
    for t in tasks:
        nm = data["no_memory"].get(t, (None, "?"))
        pc = data["pccr"].get(t, (None, "?"))
        ra = data["retrieve_all"].get(t, (None, "?")) if data["retrieve_all"] else (None, "?")
        mark = lambda v: "✓" if v is True else ("✗" if v is False else "-")
        flag = ""
        if nm[0] is False and pc[0] is True:
            flag = "  ⬅ MEMORY HELPED"
            helped.append((pc[1], t))
        print(f"{pc[1]:24s}{mark(nm[0]):8s}{mark(pc[0]):6s}{mark(ra[0]):9s}  {t[:45]}{flag}")

    # totals
    def acc(m):
        v = [s for s, _ in data[m].values()] if data[m] else []
        return f"{sum(1 for s in v if s)}/{len(v)}"
    print(f"\n=== totals ===")
    for m in METHODS:
        if data[m]:
            print(f"  {m:14s}{acc(m)}")

    # per-pattern accuracy
    print(f"\n=== accuracy by task pattern ===")
    print(f"{'pattern':26s}{'no_mem':9s}{'pccr':7s}{'retr_all':9s}")
    pats = sorted({p for _, p in data['pccr'].values()})
    for pat in pats:
        row = {}
        for m in METHODS:
            if not data[m]:
                row[m] = "-"; continue
            vals = [s for s, p in data[m].values() if p == pat]
            row[m] = f"{sum(1 for s in vals if s)}/{len(vals)}"
        print(f"{pat:26s}{row['no_memory']:9s}{row['pccr']:7s}{row.get('retrieve_all','-'):9s}")

    print(f"\n=== VERDICT ===")
    if helped:
        print(f"Memory was load-bearing on {len(helped)} task(s) (no-memory failed, PCCR passed):")
        for pat, t in helped:
            print(f"  [{pat}] {t[:55]}")
        print("→ memory is NOT useless; the cost comparison is valid.")
    else:
        print("Memory changed NO task outcomes → on this set memory is not load-bearing;")
        print("→ lean on the ENT-critical tasks / a memory-heavier test set.")


if __name__ == "__main__":
    main()
