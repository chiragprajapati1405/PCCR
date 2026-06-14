"""
analyze_results.py — turn the PCCR run JSONs into the paper's tables.

Reads the latest logs_memory_manager/pccr_<mode>_thr<theta>_*.json for each
configuration produced by run_full.sh and prints:
  Table 1  held-out comparison at theta=1.0 (pccr vs boolean vs retrieve_all)
  Table 2  cost/quality frontier (pccr at theta in {1.0, 1.2, 1.4})
  Fig 1    per-pattern EM/ENT consult rates (from decision_log)
  Drift    closed-loop utility changes (final_utility_table vs base)

Note on train/test separation: route_query() returns BEFORE logging a
RoutingDecision when memory is disabled, and the training stage runs with memory
off. Therefore every decision_log entry, and every nonzero cost_log row, is from
the held-out TEST stage — so summing cost_log gives test-only totals.
"""
import glob
import json
import os
from collections import defaultdict

LOGDIR = "logs_memory_manager"

BASE_UTILITY = {
    "single_cal_create":         {"episodic": 0.60, "entity": 0.10},
    "multi_cal_find_and_create": {"episodic": 0.78, "entity": 0.72},
    "remind_notify":             {"episodic": 0.66, "entity": 0.55},
    "email_query":               {"episodic": 0.48, "entity": 0.66},
    "email_send":                {"episodic": 0.62, "entity": 0.10},
    "cal_query":                 {"episodic": 0.58, "entity": 0.10},
    "unknown":                   {"episodic": 0.70, "entity": 0.60},
}


def latest_json(mode, thr):
    pat = os.path.join(LOGDIR, f"pccr_{mode}_thr{thr}_*.json")
    files = sorted(glob.glob(pat), key=os.path.getmtime)
    return files[-1] if files else None


def load(mode, thr):
    f = latest_json(mode, thr)
    if not f:
        return None
    with open(f) as fh:
        d = json.load(fh)
    d["_file"] = os.path.basename(f)
    return d


def test_totals(d):
    """Test-stage cost totals (train rows contribute 0)."""
    faiss = sum(r["faiss_queries"] for r in d["cost_log"])
    consults = sum(r["optional_stores_consulted"] for r in d["cost_log"])
    stm_hits = sum(1 for r in d["cost_log"] if str(r["stm_layer"]).startswith("STM"))
    n_test = d["results"]["testing"]["total"]
    acc = d["results"]["testing"]["passed"]
    # mean routing latency from decision_log (test-only)
    lats = [dec.get("latency_us", 0.0) for dec in d.get("decision_log", [])]
    avg_lat = sum(lats) / len(lats) if lats else 0.0
    return {"acc": acc, "n_test": n_test, "faiss": faiss, "consults": consults,
            "stm_hits": stm_hits, "avg_consults": consults / max(n_test, 1),
            "avg_lat_us": avg_lat}


def per_pattern_consults(d):
    """EM/ENT consult rates per task pattern from the test decision_log."""
    seen = defaultdict(lambda: {"n": 0, "EM": 0, "ENT": 0})
    for dec in d.get("decision_log", []):
        p = dec["pattern"]
        seen[p]["n"] += 1
        con = dec.get("consulted", [])
        seen[p]["EM"] += int("episodic" in con)
        seen[p]["ENT"] += int("entity" in con)
    return seen


def fmt(x, w=10):
    return str(x).ljust(w)


def main():
    print("=" * 78)
    print("  PCCR RESULTS  (held-out test, strict semantic STM)")
    print("=" * 78)

    pccr10 = load("pccr", "1.0")
    boolean = load("boolean", "1.0")
    rall = load("retrieve_all", "1.0")

    # ---- Table 1: comparison at theta=1.0 ----
    print("\nTABLE 1 — Held-out comparison (theta = 1.0)")
    print(f"  {'Method':14s}{'TestAcc':10s}{'FAISS':8s}{'Consults':10s}"
          f"{'Avg/task':10s}{'STMhits':9s}{'Lat(us)':8s}")
    for name, d in [("Retrieve-All", rall), ("Boolean", boolean), ("PCCR (ours)", pccr10)]:
        if not d:
            print(f"  {name:14s}(no JSON yet)")
            continue
        t = test_totals(d)
        print(f"  {name:14s}{fmt(str(t['acc'])+'/'+str(t['n_test']))}"
              f"{fmt(t['faiss'],8)}{fmt(t['consults'],10)}"
              f"{fmt(round(t['avg_consults'],2),10)}{fmt(t['stm_hits'],9)}"
              f"{fmt(round(t['avg_lat_us'],1),8)}")

    # ---- Table 2: frontier ----
    print("\nTABLE 2 — Cost/quality frontier (PCCR, varying theta)")
    print(f"  {'theta':8s}{'TestAcc':10s}{'FAISS':8s}{'Consults':10s}{'Avg/task':10s}")
    for thr in ["1.0", "1.2", "1.4"]:
        d = load("pccr", thr)
        if not d:
            print(f"  {thr:8s}(no JSON yet)")
            continue
        t = test_totals(d)
        print(f"  {thr:8s}{fmt(str(t['acc'])+'/'+str(t['n_test']))}"
              f"{fmt(t['faiss'],8)}{fmt(t['consults'],10)}{fmt(round(t['avg_consults'],2),10)}")

    # ---- Fig 1: per-pattern routing (PCCR theta=1.0) ----
    if pccr10:
        print("\nFIGURE 1 — Per-pattern EM/ENT consult rate (PCCR theta=1.0, test)")
        pp = per_pattern_consults(pccr10)
        print(f"  {'pattern':28s}{'n':4s}{'EM':8s}{'ENT':8s}")
        for p, c in sorted(pp.items()):
            em = f"{c['EM']}/{c['n']}"
            ent = f"{c['ENT']}/{c['n']}"
            print(f"  {p:28s}{fmt(c['n'],4)}{fmt(em,8)}{fmt(ent,8)}")

    # ---- Closed-loop drift ----
    if pccr10 and pccr10.get("final_utility_table"):
        print("\nCLOSED-LOOP DRIFT (PCCR theta=1.0): base -> final (changed cells)")
        fut = pccr10["final_utility_table"]
        any_change = False
        for p, stores in fut.items():
            for s, v in stores.items():
                base = BASE_UTILITY.get(p, {}).get(s)
                if base is not None and abs(float(v) - base) > 1e-9:
                    print(f"  {p}.{s}: {base} -> {round(float(v),3)}")
                    any_change = True
        if not any_change:
            print("  (no utility cells changed — no EM/ENT consults reached the loop)")

    print("\n" + "=" * 78)
    print("Source JSONs:")
    for name, thr in [("pccr", "1.0"), ("boolean", "1.0"), ("retrieve_all", "1.0"),
                      ("pccr", "1.2"), ("pccr", "1.4")]:
        f = latest_json(name, thr)
        print(f"  {name:14s}thr{thr}: {os.path.basename(f) if f else '— missing —'}")


if __name__ == "__main__":
    main()
