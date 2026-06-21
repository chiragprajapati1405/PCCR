"""T0c: calibrate the REAL rho-gate (router.py) for OfficeBench.

With a single gated store (EM), rho = U/C reduces to a per-pattern utility
threshold, so the meaningful calibration is UTILITY. We also report EM injection
cost for the writeup.

  COST (local): avg injected EM tokens (em.search + preload); C(EM) anchored to 1.0
                so the gate thresholds on measured utility. -> calibration/store_cost.json
  UTILITY (API): counterfactual, leave-one-out: U(pattern) =
                 clamp(P(success | EM on, LOO) - P(success | off), 0, 1).
                 -> calibration/pattern_utility.json   (RealArch reads the 'orchestrator' key)

  source cerebras.env
  python -m officebench_eval.calibrate cost
  python -m officebench_eval.calibrate utility --per-pattern 6
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

from .gate import load_calibration
from .real_arch import RealArch
from .runner import cap_for_level, run_task

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
CALIB = os.path.join(_PKG, "calibration")
TOK = lambda s: max(1, len(s) // 4)


def measure_cost(sample=40):
    cost, util = load_calibration(CALIB)
    real = RealArch(BANK, cost, util, theta=0.0)
    toks = []
    for rec in json.load(open(BANK))[:sample]:
        block = "\n".join(f"{m.description[:80]} | {' ; '.join(m.plan[:8])}"
                          for _s, m in real.search(rec["task"], 5))
        block += "\n".join(sm.action[:110] for ah in real.preload(rec["task"], 3).values() for sm in ah)
        toks.append(TOK(block))
    avg = sum(toks) / max(len(toks), 1)
    out = {"orchestrator": 1.0, "agent": 1.0,            # C=1 -> gate thresholds on utility
           "_avg_em_injection_tokens": round(avg, 1), "_n": len(toks)}
    os.makedirs(CALIB, exist_ok=True)
    json.dump(out, open(f"{CALIB}/store_cost.json", "w"), indent=2)
    print(json.dumps(out, indent=2)); print(f"wrote {CALIB}/store_cost.json")


def measure_utility(per_pattern=6, model="gpt-oss-120b"):
    cost, util = load_calibration(CALIB)
    real = RealArch(BANK, cost, util, theta=0.0)
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    by = defaultdict(list)
    for it in json.load(open(SPLIT))["train"]:
        by[pats.get((it["task"], it["subtask"]), "multi_app")].append(it)
    calib = [(it, p) for p, items in by.items() for it in items[:per_pattern]]

    os.makedirs(CALIB, exist_ok=True)
    prog_path = f"{CALIB}/utility_progress.json"
    prog = json.load(open(prog_path)) if os.path.exists(prog_path) else {}
    succ = defaultdict(lambda: [0, 0])  # (pattern,cond) -> [passed,total]

    print(f"counterfactual utility over {len(calib)} calib tasks x 2 (EM off/on, LOO)...")
    for i, (it, pat) in enumerate(calib):
        for cond, method in [("off", "no_memory"), ("on", "retrieve_all")]:
            key = f"{it['task']}/{it['subtask']}|{cond}"
            if key in prog:
                ok = prog[key]
            else:
                try:
                    r = run_task(it["task"], it["subtask"], model=model, real_arch=real,
                                 method=method, pattern=pat, max_iter=cap_for_level(it["level"]),
                                 container="ob-calib", exclude_task=it["task"])  # LOO
                    ok = int(r["success"])
                except Exception as e:
                    print(f"   {key} ERR {str(e)[:60]}"); ok = 0
                prog[key] = ok; json.dump(prog, open(prog_path, "w"))
            succ[(pat, cond)][0] += ok; succ[(pat, cond)][1] += 1
        print(f"  [{i+1}/{len(calib)}] {it['task']}/{it['subtask']} [{pat}]", flush=True)

    util_out = {}
    for pat in by:
        on, off = succ[(pat, "on")], succ[(pat, "off")]
        u = max(0.0, min(1.0, on[0]/max(on[1], 1) - off[0]/max(off[1], 1)))
        util_out[pat] = {"orchestrator": round(u, 3), "agent": round(u, 3)}
    out = {**util_out, "_method": "U = clamp(P(success|EM on,LOO) - P(success|off), 0, 1)",
           "_raw": {f"{p}|{c}": v for (p, c), v in succ.items()}}
    json.dump(out, open(f"{CALIB}/pattern_utility.json", "w"), indent=2)
    print(json.dumps(util_out, indent=2)); print(f"wrote {CALIB}/pattern_utility.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cost", "utility"])
    ap.add_argument("--per-pattern", type=int, default=6)
    ap.add_argument("--model", default="gpt-oss-120b")
    a = ap.parse_args()
    measure_cost() if a.cmd == "cost" else measure_utility(a.per_pattern, a.model)
