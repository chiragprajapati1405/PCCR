"""AppWorld Phase 2: harvest an EM procedure bank from the TRAIN ground truth.

Each train task ships a clean, commented reference solution (`ground_truth/compiled_solution.py`) --
the ideal episodic-memory procedure, no flaky agent run needed. We extract, per train task:
  instruction (specs.json) + required_apps + the solution body (commented apis.<app>.<method> calls).
Saved to appworld_eval/em_bank_appworld.json for the rho-gate to retrieve+inject in Phase 3.
Run: .venv311/bin/python -m appworld_eval.build_em_bank
"""
import json
import os
import re

from appworld import load_task_ids

DATA = "data/tasks"
OUT = "appworld_eval/em_bank_appworld.json"


def _solution_body(path):
    """Strip the canary + imports; keep the `def solution(...)` body (commented API-call procedure)."""
    if not os.path.exists(path):
        return ""
    src = open(path).read()
    # match the solution signature incl. an optional '-> None:' return annotation, keep the body only
    m = re.search(r"def solution\(.*?\)\s*(?:->[^\n:]*)?:\n(.*)", src, re.S)
    body = m.group(1) if m else src
    # dedent one level, drop the canary comment
    lines = [ln[4:] if ln.startswith("    ") else ln for ln in body.splitlines()
             if "Canary String" not in ln]
    return "\n".join(lines).strip()


def main():
    tids = load_task_ids("train")
    bank = []
    for tid in tids:
        d = os.path.join(DATA, tid)
        try:
            instr = json.load(open(os.path.join(d, "specs.json"))).get("instruction", "")
        except Exception:
            instr = ""
        try:
            apps = json.load(open(os.path.join(d, "ground_truth", "required_apps.json")))
        except Exception:
            apps = []
        proc = _solution_body(os.path.join(d, "ground_truth", "compiled_solution.py"))
        if instr and proc:
            bank.append({"task_id": tid, "instruction": instr, "apps": apps, "procedure": proc})
    json.dump(bank, open(OUT, "w"), indent=1)
    print("EM bank: %d procedures -> %s" % (len(bank), OUT))
    # quick profile
    from collections import Counter
    apps = Counter(a for e in bank for a in e["apps"])
    print("apps covered:", dict(apps.most_common()))
    print("mean procedure length: %.0f chars | mean instruction: %.0f chars"
          % (sum(len(e["procedure"]) for e in bank) / max(1, len(bank)),
             sum(len(e["instruction"]) for e in bank) / max(1, len(bank))))


if __name__ == "__main__":
    main()
