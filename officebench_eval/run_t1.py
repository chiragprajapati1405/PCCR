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

from .gate import load_calibration
from .real_arch import RealArch
from .real_mem import RealMem
from .runner import run_task, cap_for_level

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
RESULTS = os.path.join(_PKG, "results")
TRACE_DIR = os.path.join(_PKG, "traces/t1")
PROGRESS = os.path.join(_PKG, "t1_progress.json")
METHODS = ("no_memory", "retrieve_all", "pccr")
GATE = "pccr"   # real MemoryManager: STM short-circuit -> rho-gated EM/SM/ENT, PM always-read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--max-iter", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--priors", action="store_true",
                    help="use the router's frozen per-pattern utility priors (ignore the "
                         "confounded calibrated utility) -- the clean headline run")
    args = ap.parse_args()

    cost, util = load_calibration(os.path.join(_PKG, "calibration"))
    if args.priors:
        util, cost = {}, {}        # keep the router's PATTERN_UTILITY priors AND prior cost (EM=0.55)
    real = RealArch(BANK, cost, util, theta=args.theta)   # for no_memory/retrieve_all EM
    realmem = RealMem(BANK, theta=args.theta, stm_threshold=0.85)   # gate arm: the REAL MemoryManager
    print(f"loaded real FAISS EM: {len(real)} procedures | theta={args.theta} | "
          f"utility={'frozen priors' if args.priors else 'calibrated'}")

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
            try:
                r = run_task(it["task"], it["subtask"], model=args.model, real_arch=real,
                             real_mem=realmem, method=m, pattern=pat,
                             max_iter=cap_for_level(it['level']), container="ob-test")
            except Exception as e:
                print(f"  {key} ERR {str(e)[:80]}", flush=True)
                prog[key] = {"success": False, "error": str(e)[:120]}
                json.dump(prog, open(PROGRESS, "w"))
                continue
            base = f"{TRACE_DIR}/{it['task']}_{it['subtask']}_{m}"
            json.dump(r, open(f"{base}.json", "w"), indent=2, default=str)   # default=str: tolerate stray Ellipsis/etc
            with open(f"{base}.txt", "w") as fh:              # readable step-by-step
                fh.write(f"TASK {it['task']}/{it['subtask']} [{pat}] {m}  "
                         f"success={r['success']}  consult_em={r['em']['consult_em']}\n"
                         f"  {r['task_text']}\n" + "=" * 70 + "\n"
                         + "\n".join(r["sequence"]) + "\n")
            em = r["em"]
            prog[key] = {"success": r["success"], "level": it["level"], "pattern": pat,
                         "consults": int(em["consult_em"]), "tokens": em["injected_tokens"],
                         "steps": r["steps"], "llm_calls": r["llm_calls"], "wall_s": r["wall_s"],
                         "stm_hit": int(em.get("stm_hit", False)), "top_em_sim": em.get("top_em_sim", 0.0),
                         "stores": em.get("consulted_stores", []), "pm_used": bool(em.get("pm_used", False)),
                         "failed_predicate": r.get("failed_predicate")}
            json.dump(prog, open(PROGRESS, "w"))
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{len(test)}] {(time.perf_counter()-t0)/60:.0f} min", flush=True)

    _report(prog)


def _report(prog):
    # per (method, level) accumulators
    def blank():
        return {"succ": 0, "n": 0, "consults": 0, "tokens": 0, "wall": 0.0, "calls": 0,
                "steps": 0, "stm": 0, "stm_elig": 0, "succ_em": 0, "n_em": 0, "succ_noem": 0, "n_noem": 0}
    agg = {m: defaultdict(blank) for m in METHODS}
    for key, r in prog.items():
        if "level" not in r:
            continue
        m = key.split("|")[1]
        for lvl in (r["level"], 0):                       # 0 = overall
            a = agg[m][lvl]
            ok = int(r["success"]); a["succ"] += ok; a["n"] += 1
            a["consults"] += r.get("consults", 0); a["tokens"] += r.get("tokens", 0)
            a["wall"] += r.get("wall_s", 0.0); a["calls"] += r.get("llm_calls", 0)
            a["steps"] += r.get("steps", 0); a["stm"] += r.get("stm_hit", 0)
            a["stm_elig"] += int(r.get("top_em_sim", 0.0) >= 0.9)
            if r.get("consults", 0):                       # how EM helps: success WHEN consulted vs not
                a["succ_em"] += ok; a["n_em"] += 1
            else:
                a["succ_noem"] += ok; a["n_noem"] += 1

    def acc(m, lvl):
        a = agg[m][lvl]; return a["succ"] / a["n"] if a["n"] else 0.0

    print(f"\n{'='*92}\nT1 FULL RESULTS (152 test tasks x 3 methods)\n{'='*92}")
    for lvl in (0, 1, 2, 3):
        tag = "ALL" if lvl == 0 else f"L{lvl}"
        print(f"\n[{tag}]  {'method':<13}{'acc':>10}{'consults':>10}{'inj.tok':>10}"
              f"{'avg_lat':>9}{'avg_calls':>10}{'STM_hit':>9}")
        for m in METHODS:
            a = agg[m][lvl]
            if not a["n"]:
                continue
            print(f"      {m:<13}{a['succ']:>3}/{a['n']:<3} {acc(m,lvl):>5.3f}"
                  f"{a['consults']:>10}{a['tokens']:>10}{a['wall']/a['n']:>8.1f}s"
                  f"{a['calls']/a['n']:>10.1f}{a['stm']:>9}")
        # EM contribution at this level
        em_lift = acc("retrieve_all", lvl) - acc("no_memory", lvl)
        gate_lift = acc(GATE, lvl) - acc("no_memory", lvl)
        pg = agg[GATE][lvl]
        consult_rate = pg["consults"] / pg["n"] if pg["n"] else 0
        tok_save = 1 - (pg["tokens"] / agg["retrieve_all"][lvl]["tokens"]) if agg["retrieve_all"][lvl]["tokens"] else 0
        print(f"      -> EM lift (retrieve_all - no_mem): {em_lift:+.3f} | gate lift: {gate_lift:+.3f} | "
              f"gate consult-rate: {consult_rate:.0%} | gate token-save vs retrieve-all: {tok_save:.0%} | "
              f"near-dup(>=0.9): {agg['no_memory'][lvl]['stm_elig']}")
        # within pccr: success WHEN EM consulted vs not
        if pg["n_em"] or pg["n_noem"]:
            r_em = pg["succ_em"]/pg["n_em"] if pg["n_em"] else 0
            r_no = pg["succ_noem"]/pg["n_noem"] if pg["n_noem"] else 0
            print(f"      -> within-PCCR success: EM-consulted {pg['succ_em']}/{pg['n_em']} ({r_em:.2f}) "
                  f"vs EM-skipped {pg['succ_noem']}/{pg['n_noem']} ({r_no:.2f})")

    out = {m: {str(l): dict(agg[m][l]) for l in agg[m]} for m in METHODS}
    json.dump(out, open(f"{RESULTS}/t1_summary.json", "w"), indent=2)
    print(f"\nsaved {RESULTS}/t1_summary.json  (full per-(method,level) metrics)")


if __name__ == "__main__":
    main()
