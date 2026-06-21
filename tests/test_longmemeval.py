"""Integration test on a small slice of the REAL LongMemEval data.

Skips automatically if the dataset hasn't been downloaded (see longmemeval/
README.md), so CI without the 290 MB file still passes. When present, it checks
the pipeline runs end-to-end on real questions, the rho-gate consults fewer
stores than retrieve-all, and the PARALLEL path matches the sequential one.
"""
from __future__ import annotations

import asyncio
import os

import pytest

DATA = "longmemeval_data/longmemeval_s.json"
pytestmark = pytest.mark.skipif(not os.path.exists(DATA),
                                reason="LongMemEval data not downloaded (see longmemeval/README.md)")


def test_real_longmemeval_pipeline_routes_cheaper_and_parallel_matches_sequential():
    from longmemeval.lme import load_cost, load_dataset, get_model
    from longmemeval.run import METHODS, _aggregate, _balanced_subset, run_parallel, run_sequential

    data = _balanced_subset(load_dataset(DATA), 6, seed=0)
    model, cost = get_model(), load_cost()

    seq = run_sequential(data, model, theta=0.10, cost=cost, k=5, floor=0.30)
    ra, pc = _aggregate(seq["retrieve_all"]), _aggregate(seq["pccr"])
    assert pc["consults"] < ra["consults"]                 # rho-gate is cheaper
    assert pc["sm"] <= ra["sm"]                             # prunes the expensive SM search
    assert pc["recall"] >= ra["recall"] - 0.25             # comparable recall on a tiny slice

    par = asyncio.run(run_parallel(data, model, theta=0.10, cost=cost, k=5, floor=0.30,
                                   concurrency=3))
    for m in METHODS:                                       # parallel == sequential
        for s, p in zip(seq[m], par[m]):
            assert s.correct == p.correct
            assert s.consulted == p.consulted
            assert set(s.retrieved) == set(p.retrieved)
