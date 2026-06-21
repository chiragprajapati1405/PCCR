"""T0c: calibrate the rho-gate's cost and utility for the 9-app setting.

COST (local, no API): average injected-procedure tokens per granularity
  (orchestrator = top-5 plans, agent = top-3 steps), anchored cheapest = 0.15
  (same convention as the earlier A1 calibration). -> calibration/store_cost.json

UTILITY (API, counterfactual + leave-one-out): on a balanced slice of TRAIN
  tasks, run each with store OFF / orchestrator-only / agent-only (LOO retrieval
  so a task never reuses its own banked solution); U(pattern,s) =
  P(success|s on) - P(success|off). -> calibration/pattern_utility.json

  source cerebras.env
  python -m officebench_eval.calibrate cost
  python -m officebench_eval.calibrate utility --per-pattern 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
CALIB = os.path.join(_PKG, "calibration")
TOK = lambda s: max(1, len(s) // 4)


def _load_memory():
    from .memory import ProcedureMemory
    mem = ProcedureMemory()
    mem.load(BANK)
    return mem


def measure_cost(sample=60):
    """Average injected tokens for each granularity over banked tasks."""
    mem = _load_memory()
    tasks = [r["task"] for r in mem.records][:sample]
    orch_tok, agent_tok = [], []
    for t in tasks:
        oh = mem.retrieve_orchestrator(t, k=5, exclude_task=t)
        ob = "\n".join(f"task: {r['task'][:80]} | plan: {' ; '.join(r['plan'][:8])}" for _s, r in oh)
        orch_tok.append(TOK(ob))
        ah = mem.retrieve_agent(t, "calendar", k=3, exclude_task=t) or \
             mem.retrieve_agent(t, "email", k=3, exclude_task=t)
        ab = "\n".join(s["text"][:120] for _s, s in ah)
        agent_tok.append(TOK(ab))
    avg_o = sum(orch_tok) / max(len(orch_tok), 1)
    avg_a = sum(agent_tok) / max(len(agent_tok), 1)
    base = max(min(avg_o, avg_a), 1)
    cost = {"orchestrator": round(0.15 * avg_o / base, 3),
            "agent": round(0.15 * avg_a / base, 3)}
    out = {**cost, "_avg_tokens": {"orchestrator": round(avg_o, 1), "agent": round(avg_a, 1)},
           "_method": "avg injected procedure tokens, anchored cheapest=0.15", "_n": len(tasks)}
    os.makedirs(CALIB, exist_ok=True)
    json.dump(out, open(f"{CALIB}/store_cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nwrote {CALIB}/store_cost.json")


def measure_utility(per_pattern=6, model="gpt-oss-120b"):
    from .runner import run_task
    mem = _load_memory()
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    train = json.load(open(SPLIT))["train"]

    # balanced calibration slice: up to `per_pattern` train tasks per pattern
    by = defaultdict(list)
    for it in train:
        by[pats.get((it["task"], it["subtask"]), "multi_app")].append(it)
    calib = []
    for pat, items in by.items():
        calib += [(it, pat) for it in items[:per_pattern]]

    prog_path = f"{CALIB}/utility_progress.json"
    os.makedirs(CALIB, exist_ok=True)
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}

    # succ[(pattern, store)] = [passed, total] for on; off tracked separately per pattern
    succ = defaultdict(lambda: [0, 0])
    print(f"counterfactual utility over {len(calib)} calib tasks x 3 conditions (LOO)...")
    for i, (it, pat) in enumerate(calib):
        for cond, stores in [("off", frozenset()), ("orchestrator", {"orchestrator"}),
                             ("agent", {"agent"})]:
            key = f"{it['task']}/{it['subtask']}|{cond}"
            if key in prog:
                ok = prog[key]
            else:
                try:
                    r = run_task(it["task"], it["subtask"], model=model, memory=mem,
                                 method=("no_memory" if cond == "off" else "retrieve_all"),
                                 stores=stores, pattern=pat, container="ob-calib",
                                 exclude_task=it["task"])     # LOO
                    ok = int(r["success"])
                except Exception as e:
                    print(f"   {key} ERR {str(e)[:70]}"); ok = 0
                prog[key] = ok
                json.dump(prog, open(prog_path, "w"))
            succ[(pat, cond)][0] += ok
            succ[(pat, cond)][1] += 1
        print(f"  [{i+1}/{len(calib)}] {it['task']}/{it['subtask']} [{pat}]", flush=True)

    # U(pattern, store) = clamp(P(success|store) - P(success|off), 0, 1)
    util = {}
    for pat in by:
        off = succ[(pat, "off")]
        p_off = off[0] / max(off[1], 1)
        util[pat] = {}
        for store in ("orchestrator", "agent"):
            on = succ[(pat, store)]
            p_on = on[0] / max(on[1], 1)
            util[pat][store] = round(max(0.0, min(1.0, p_on - p_off)), 3)
    out = {**util, "_method": "U = clamp(P(success|store on, LOO) - P(success|off), 0, 1)",
           "_raw": {f"{p}|{c}": v for (p, c), v in succ.items()}}
    json.dump(out, open(f"{CALIB}/pattern_utility.json", "w"), indent=2)
    print(json.dumps(util, indent=2))
    print(f"\nwrote {CALIB}/pattern_utility.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cost", "utility"])
    ap.add_argument("--per-pattern", type=int, default=6)
    ap.add_argument("--model", default="gpt-oss-120b")
    a = ap.parse_args()
    if a.cmd == "cost":
        measure_cost()
    else:
        measure_utility(a.per_pattern, a.model)
