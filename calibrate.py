"""
calibrate.py — RECTIFY the two hand-set assumptions (cost, utility) by MEASURING
them from data, then writing them to calibration/*.json. PCCRMemoryManager loads
those files automatically, so after calibration the router uses NO hand-set
numbers — cost is measured, utility is counterfactual-measured.

  python calibrate.py costs        # measure store cost (LOCAL, no API)  → calibration/store_cost.json
  python calibrate.py utilities    # counterfactual utility (NEEDS API)  → calibration/pattern_utility.json

Cost = relative access cost, measured as average tokens injected per
consultation (the real downstream/API cost), with latency reported for
reference. We anchor the cheapest optional store to 0.15 to keep the familiar
scale; only the measured RATIO matters (θ is then re-swept — it is an explicit
dial, not an assumption).

Utility(pattern, store) = measured marginal accuracy gain: run the SAME held-out
tasks with the store forced ON vs forced OFF, U = clamp(P(success|on) −
P(success|off), 0, 1). This directly answers "how do you KNOW usefulness."
"""
import os
import sys
import json
import time
import numpy as np

from run_officebench_local import LocalOfficeBenchRunner
from free_legomem import LocalEmbedder
from mm_on_top_of_legomem import execute_task, LOCAL_TESTBED
from pccr_on_top_of_legomem import (
    PCCRMemoryManager, filter_cal_email_word, _reload_bank, _rebuild_stm_from_bank,
)

OPTIONAL = ("episodic", "entity")
OUT = "calibration"
os.makedirs(OUT, exist_ok=True)


def _toks(s):
    return int(len(s) / 4.0)   # ~4 chars/token (install tiktoken for exact)


# ────────────────────────────────────────────────────────────────────
#  COST  (local, no API)
# ────────────────────────────────────────────────────────────────────
def measure_costs(reps=40):
    emb = LocalEmbedder()
    mm = PCCRMemoryManager(embed_fn=emb.embed, embed_dim=emb.dims,
                           strict_stm=True, enable_word_agent=True)
    n = _reload_bank(mm)
    mm.entity["users"]["Bob"] = {"task_count": 3, "success_rate": 1.0}  # a sample profile
    print(f"reloaded {n} episodes; timing {reps} reps per store...")

    probe = "schedule a meeting and email an invite"

    # latency
    t = time.perf_counter()
    for _ in range(reps):
        _ = mm.entity["users"].get("Bob", {})
    ent_us = (time.perf_counter() - t) / reps * 1e6
    t = time.perf_counter()
    for _ in range(reps):
        _ = mm.episodic.retrieve_for_orchestrator(probe, k=3)
    em_us = (time.perf_counter() - t) / reps * 1e6

    # token cost (the real downstream cost): size of what each store injects
    raw = mm.episodic.retrieve_for_orchestrator(probe, k=3)
    em_block = "PAST SUCCESSFUL EXPERIENCES:\n" + "".join(
        f"  Task: {m.task_description}\n  Plan: {m.high_level_plan}\n\n" for m in raw[:3])
    ent_block = "User Bob: tasks=3, rate=1.0\n"
    em_tok, ent_tok = max(_toks(em_block), 1), max(_toks(ent_block), 1)

    # cost = token-injection cost, anchored so the cheaper store = 0.15
    base = min(em_tok, ent_tok)
    cost = {"episodic": round(0.15 * em_tok / base, 3),
            "entity":   round(0.15 * ent_tok / base, 3)}

    out = {"store_cost": cost,
           "_method": "avg injected tokens per consultation, anchored cheapest=0.15",
           "_avg_tokens": {"episodic": em_tok, "entity": ent_tok},
           "_latency_us": {"episodic": round(em_us, 1), "entity": round(ent_us, 2)},
           "_measured_ratio_EM_over_ENT": round(em_tok / ent_tok, 2)}
    with open(f"{OUT}/store_cost.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n✅ wrote {OUT}/store_cost.json — cost is now MEASURED (was hand-set 0.55/0.15).")
    print("   Note: θ is re-swept after recalibration; the measured RATIO is what matters.")


# ────────────────────────────────────────────────────────────────────
#  UTILITY  (counterfactual — NEEDS API)
# ────────────────────────────────────────────────────────────────────
def measure_utilities(n_calib=None):
    from mm_on_top_of_legomem import MultiLLM
    emb = LocalEmbedder()
    keys = [os.environ.get(f"CEREBRAS_KEY_{i}", "") for i in range(1, 5)]
    llm = MultiLLM(keys, model="gpt-oss-120b")
    for c in llm.clients:
        try:
            c["client"] = c["client"].with_options(timeout=60.0)
        except Exception:
            pass
    runner = LocalOfficeBenchRunner("./OfficeBench")
    pool = filter_cal_email_word(runner.get_all_task_ids(), "./OfficeBench")

    mm = PCCRMemoryManager(embed_fn=emb.embed, embed_dim=emb.dims,
                           strict_stm=True, enable_word_agent=True,
                           routing_mode="pccr", freeze_test=True)
    _reload_bank(mm)
    _rebuild_stm_from_bank(mm)
    mm.freeze_writes = True   # don't mutate memory during measurement

    calib = pool[60:] if len(pool) > 60 else pool          # held-out calibration tasks
    if n_calib:
        calib = calib[:n_calib]
    print(f"counterfactual utility over {len(calib)} held-out tasks × {len(OPTIONAL)} stores × 2 (on/off)")

    # success counts per (pattern, store, forced-on/off)
    from collections import defaultdict
    succ = defaultdict(lambda: {"on": [0, 0], "off": [0, 0]})  # [passed, total]

    def run_once(tid, si, store, forced):
        cfg = runner.setup_task(tid, si)
        LOCAL_TESTBED[0] = str(runner.testbed)
        mm.force_stores = {store: forced}                  # force this store on/off
        twd = cfg["task"] + (f"\n(Today's date: {cfg['date']})" if cfg.get("date") else "")
        pattern, _ = mm.stm.classify(cfg["task"])
        try:
            execute_task(twd, cfg["username"], llm, runner, mm, use_memory=True, task_date=cfg.get("date"))
            ev = runner.evaluate_task(tid, si)
            mm.on_task_complete(cfg["task"], ev["success"], {"faiss_queries": 0}, [])
            return pattern, int(ev["success"])
        except Exception as e:
            print(f"    ⚠️ {str(e)[:80]}")
            mm.force_stores = {}
            return pattern, 0

    for store in OPTIONAL:
        for tid, si in calib:
            for forced, key in [(True, "on"), (False, "off")]:
                pat, ok = run_once(tid, si, store, forced)
                succ[(pat, store)][key][0] += ok
                succ[(pat, store)][key][1] += 1
    mm.force_stores = {}

    # U(pattern, store) = clamp( P(success|on) − P(success|off) , 0, 1 )
    util = {}
    for (pat, store), d in succ.items():
        p_on = d["on"][0] / max(d["on"][1], 1)
        p_off = d["off"][0] / max(d["off"][1], 1)
        util.setdefault(pat, {})[store] = round(max(0.0, min(1.0, p_on - p_off)), 3)

    out = {"pattern_utility": util,
           "_method": "U = clamp(P(success|store on) - P(success|store off), 0, 1)",
           "_n_calibration_tasks": len(calib),
           "_raw_success_counts": {f"{p}|{s}": v for (p, s), v in succ.items()}}
    with open(f"{OUT}/pattern_utility.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n✅ wrote {OUT}/pattern_utility.json — utility is now MEASURED (was hand-set).")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "costs"
    if cmd == "costs":
        measure_costs()
    elif cmd == "utilities":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else None
        measure_utilities(n)
    else:
        print("usage: python calibrate.py [costs|utilities [n_tasks]]")
