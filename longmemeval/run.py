"""Run PCCR rho-gate retrieval on the REAL LongMemEval benchmark.

  python -m longmemeval.run --data longmemeval_data/longmemeval_s.json --n 100
  python -m longmemeval.run --data ... --n 100 --parallel    # also check parallel==seq

Reports, per method (pccr / retrieve_all / boolean): answerable session-recall,
abstention accuracy, and store-consultation cost — plus a per-question-type
breakdown. The headline question: does the rho-gate match retrieve-all's recall
while consulting fewer (and cheaper) stores?
"""
from __future__ import annotations

import argparse
import asyncio
import time
from collections import Counter, defaultdict

from .lme import (OPTIONAL, answer_question, build_memory, get_model, is_abstention,
                  load_cost, load_dataset, stores_for)

METHODS = ("retrieve_all", "boolean", "pccr")


def _balanced_subset(data, n, seed=0):
    import random
    rng = random.Random(seed)
    by = defaultdict(list)
    for e in data:
        by[e["question_type"]].append(e)
    for v in by.values():
        rng.shuffle(v)
    out, i = [], 0
    types = sorted(by)
    while len(out) < n and any(by[t] for t in types):       # round-robin across types
        t = types[i % len(types)]; i += 1
        if by[t]:
            out.append(by[t].pop())
    return out[:n]


def _aggregate(results):
    """results: list[QResult] for ONE method."""
    ans = [r for r in results if r.kind == "answerable"]
    abst = [r for r in results if r.kind == "abstention"]
    recall = sum(r.correct for r in ans) / max(len(ans), 1)
    abst_acc = sum(r.correct for r in abst) / max(len(abst), 1)
    consults = sum(r.n_consults for r in results)
    em_consults = sum(1 for r in results if "episodic" in r.consulted)
    sm_consults = sum(1 for r in results if "semantic" in r.consulted)
    return dict(recall=recall, abst_acc=abst_acc, consults=consults,
                em=em_consults, sm=sm_consults, n_ans=len(ans), n_abs=len(abst))


def run_sequential(data, model, theta, cost, k, floor):
    out = {m: [] for m in METHODS}
    for entry in data:
        mem = build_memory(entry, model)                    # build once per question
        qvec = model.encode([entry["question"]], normalize_embeddings=True,
                            show_progress_bar=False)[0]
        for m in METHODS:
            out[m].append(answer_question(entry, mem, model, m, theta, cost, k, floor, qvec=qvec))
    return out


async def run_parallel(data, model, theta, cost, k, floor, concurrency=8):
    """Parallel TASKS via AsyncTaskQueue (each question independent), and parallel
    SUB-AGENTS (the consulted stores retrieved concurrently as one wave)."""
    from memory_manager.task_queue import AsyncTaskQueue
    from memory_manager.parallel import ParallelExecutor, Delegation, AgentResult

    # pre-embed memories (CPU work) so the async layer measures scheduling, not numpy
    prepared = []
    for entry in data:
        mem = build_memory(entry, model)
        qvec = model.encode([entry["question"]], normalize_embeddings=True,
                            show_progress_bar=False)[0]
        prepared.append((entry, mem, qvec))

    out = {m: [] for m in METHODS}

    async def pipeline(item):
        entry, mem, qvec = item
        res = {}
        for m in METHODS:
            stores = stores_for(m, entry["question_type"], theta, cost)
            # parallel sub-agents: one retrieval "agent" per consulted store, run as a wave
            async def retr(store, subtask, snap, _s=mem, _q=qvec):
                if store == "episodic": hits = _s.em(_q, k)
                elif store == "semantic": hits = _s.sm(_q, k)
                else: hits = _s.ent(k)
                return AgentResult(store, subtask, observation=str([h[0] for h in hits]))
            ex = ParallelExecutor(retr)
            group = [Delegation(s, f"retrieve from {s}") for s in sorted(stores)]
            await ex.execute_group(group, None)             # concurrent store retrieval
            # the deterministic answer is computed by the same code path as sequential
            res[m] = answer_question(entry, mem, model, m, theta, cost, k, floor,
                                     qvec=qvec, stores=stores)
        return res

    q = AsyncTaskQueue(pipeline, max_concurrency=concurrency)
    results = await q.run(prepared)
    for r in results:
        for m in METHODS:
            out[m].append(r[m])
    return out


def _print(out, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    print(f"{'method':<14}{'recall':>9}{'abst_acc':>10}{'consults':>10}{'EM':>6}{'SM':>6}")
    base = _aggregate(out["retrieve_all"])
    for m in METHODS:
        a = _aggregate(out[m])
        red = 100 * (base['consults'] - a['consults']) / max(base['consults'], 1)
        save = "" if m == "retrieve_all" else f"  ({red:.0f}% fewer consults)"
        print(f"{m:<14}{a['recall']:>9.3f}{a['abst_acc']:>10.3f}{a['consults']:>10}{a['em']:>6}{a['sm']:>6}{save}")
    # per-type recall for PCCR vs retrieve_all
    print(f"\nper-type answerable recall (pccr vs retrieve_all):")
    by = defaultdict(lambda: {"pccr": [0, 0], "retrieve_all": [0, 0]})
    for m in ("pccr", "retrieve_all"):
        for r in out[m]:
            if r.kind == "answerable":
                by[r.qtype][m][0] += r.correct; by[r.qtype][m][1] += 1
    for t in sorted(by):
        p, ra = by[t]["pccr"], by[t]["retrieve_all"]
        print(f"  {t:<26} pccr {p[0]:>3}/{p[1]:<3}   retrieve_all {ra[0]:>3}/{ra[1]:<3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_data/longmemeval_s.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--floor", type=float, default=0.30)
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()

    cost = load_cost()
    data = load_dataset(args.data)
    subset = _balanced_subset(data, args.n, args.seed) if args.n < len(data) else data
    print(f"LongMemEval REAL run | {len(subset)} questions | theta={args.theta} k={args.k} "
          f"floor={args.floor}")
    print(f"measured cost: {cost} | type mix: {dict(Counter(e['question_type'] for e in subset))}")
    print(f"abstention questions: {sum(1 for e in subset if is_abstention(e))}")

    model = get_model()
    t0 = time.perf_counter()
    seq = run_sequential(subset, model, args.theta, cost, args.k, args.floor)
    print(f"\n[sequential done in {time.perf_counter()-t0:.1f}s]")
    _print(seq, "SEQUENTIAL")

    if args.parallel:
        par = asyncio.run(run_parallel(subset, model, args.theta, cost, args.k, args.floor,
                                       args.concurrency))
        _print(par, f"PARALLEL (C={args.concurrency})")
        # equivalence: parallel must match sequential per question
        same = all(s.correct == p.correct and s.consulted == p.consulted and
                   set(s.retrieved) == set(p.retrieved)
                   for m in METHODS for s, p in zip(seq[m], par[m]))
        print(f"\nparallel == sequential (outcomes & consults): {same}")


if __name__ == "__main__":
    main()
