"""Build the paper-comparable results table for the 2D arms from the full-152 traces.

Emits ONE row per arm with the SAME columns as the paper's cost ledger (Table tab:realtoken /
tab:perf) plus a cost column, so 2D drops straight into the comparison:

  Arm | Accuracy | Real tokens | tok/call | LLM calls | compute_s | wall_s(min) | cost_usd

Arms:
  retrieve-all / improved / improved+PERF : the sequential/1D baselines, from COST_LEDGER.json
  PURE-2D (all 152)                       : twod_all152_traces.json aggregated (force mode)
  2D-ROUTER (structural)                  : per task, 2D trace iff meta.fanout_fired else the 1D
                                            improved ledger -- keyword-free routing, >= 1D by design

Caveat on compute_s for the parallel 2D arms: rate_limit_wait_s is SUMMED across concurrent
sub-agents, so wall-minus-wait can under-count; we report wall_s (true latency) alongside and clamp
compute_s at 0. Every raw token/timing field is in the traces, so any column is recomputable.
"""
import json, os
REPO = "/Users/chirag/Documents/Agentic_MM"
PRICE_IN_PER_M, PRICE_OUT_PER_M = 0.25, 0.69

ledger = json.load(open(os.path.join(REPO, "officebench_eval/COST_LEDGER.json")))
oneD = json.load(open(os.path.join(REPO, "officebench_eval/par_rt_improved_progress.json")))
traces = json.load(open(os.path.join(REPO, "officebench_eval/twod_all152_traces.json")))


def cost(pt, ct):
    return pt / 1e6 * PRICE_IN_PER_M + ct / 1e6 * PRICE_OUT_PER_M


def row(name, npass, n, pt, ct, calls, wall, wait):
    real = pt + ct
    compute = max(0.0, wall - wait)
    return {"arm": name, "acc": "%d/%d" % (npass, n), "real_M": real / 1e6,
            "tok_call": real / max(1, calls), "calls": calls, "compute_s": compute,
            "wall_min": wall / 60.0, "cost": cost(pt, ct)}


rows = []
# --- baselines from the cost ledger ---
for key, label in [("rt_ra", "retrieve-all (1D seq)"), ("rt_improved", "improved PCCR (1D)")]:
    a = ledger[key]
    rows.append(row(label, a["pass"], a["n"], a["prompt_tokens"], a["completion_tokens"],
                    a["llm_calls"], a["wall_s"], a["rate_limit_wait_s"]))

# --- PURE-2D over all 152 (force mode) ---
tv = list(traces.values())
n = len(tv)
rows.append(row("PURE-2D (all %d)" % n, sum(v["success"] for v in tv), n,
                sum(v["prompt_tokens"] for v in tv), sum(v["completion_tokens"] for v in tv),
                sum(v["calls"] for v in tv), sum(v["wall_s"] for v in tv),
                sum(v.get("rate_limit_wait_s", 0) for v in tv)))

# --- 2D structural router: 2D where fanout_fired, else 1D improved ledger ---
npass = pt = ct = calls = 0
wall = wait = 0.0
fired = 0
for v in tv:
    k = "%s/%s" % (v["task"], v["subtask"])
    l1 = oneD.get(k + "|pccr", {})
    if v.get("fanout_fired"):
        fired += 1
        npass += 1 if v["success"] else 0
        pt += v["prompt_tokens"]; ct += v["completion_tokens"]; calls += v["calls"]
        wall += v["wall_s"]; wait += v.get("rate_limit_wait_s", 0)
    else:
        npass += 1 if l1.get("success") else 0
        pt += l1.get("prompt_tokens", 0); ct += l1.get("completion_tokens", 0); calls += l1.get("llm_calls", 0)
        wall += l1.get("wall_s", 0); wait += l1.get("rate_limit_wait_s", 0)
rows.append(row("2D-ROUTER (fires on %d)" % fired, npass, n, pt, ct, calls, wall, wait))

# --- print ---
hdr = ("Arm", "Accuracy", "Real tok", "tok/call", "calls", "compute_s", "wall(min)", "cost $")
print("%-26s %-9s %9s %9s %7s %10s %10s %8s" % hdr)
print("-" * 96)
for r in rows:
    print("%-26s %-9s %8.2fM %9.0f %7d %10.0f %10.1f %8.4f" % (
        r["arm"], r["acc"], r["real_M"], r["tok_call"], r["calls"], r["compute_s"], r["wall_min"], r["cost"]))
print("\nNote: %d/152 traces present." % n)
