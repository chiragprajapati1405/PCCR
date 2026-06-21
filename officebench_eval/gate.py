"""The rho-gate for OfficeBench procedure memory.

The two routable "stores" are the two EM granularities (LegoMem): orchestrator
(full task plans) and agent (per-app subtask steps). The gate decides which to
consult per task: consult s iff rho = U(pattern,s)/C(s) >= theta.

Cost/utility default to priors and are OVERRIDDEN by calibration files once T0c
runs (cost = measured injected-procedure tokens; utility = counterfactual gain).
"""
from __future__ import annotations

import json

STORES = ("orchestrator", "agent")

# measured-cost placeholder (orchestrator injects top-5 plans -> larger than agent top-3)
DEFAULT_COST = {"orchestrator": 1.0, "agent": 0.6}

# utility priors per pattern (placeholder until T0c counterfactual calibration);
# higher where procedure reuse helps (complex pipelines), per LegoMem's L3 gains
DEFAULT_UTIL = {
    "lookup":        {"orchestrator": 0.20, "agent": 0.20},
    "single_action": {"orchestrator": 0.30, "agent": 0.30},
    "data_compute":  {"orchestrator": 0.55, "agent": 0.50},
    "doc_process":   {"orchestrator": 0.75, "agent": 0.70},
    "multi_app":     {"orchestrator": 0.90, "agent": 0.80},
}
_FALLBACK_U = {"orchestrator": 0.6, "agent": 0.5}


def load_calibration(path="officebench_eval/calibration"):
    cost = dict(DEFAULT_COST)
    util = {k: dict(v) for k, v in DEFAULT_UTIL.items()}
    try:
        cost.update(json.load(open(f"{path}/store_cost.json")))
    except Exception:
        pass
    try:
        util.update(json.load(open(f"{path}/pattern_utility.json")))
    except Exception:
        pass
    return cost, util


def stores_for(method, pattern, theta, cost, util):
    if method == "no_memory":
        return frozenset()
    if method == "retrieve_all":
        return frozenset(STORES)
    if method == "pccr":
        u = util.get(pattern, _FALLBACK_U)
        return frozenset(s for s in STORES if u[s] / cost[s] >= theta)
    raise ValueError(method)
