"""Real-LLM LongMemEval run: accuracy + tokens + store-usage + latency, with traces.

  source cerebras.env
  python -m longmemeval.run_qa --n 24 --concurrency 4

Produces:
  - ACCURACY/TOKENS/STORE-USAGE: pccr vs retrieve_all (real gpt-oss-120b answers, judged)
  - LATENCY: sequential vs parallel answer generation (real overlapping API calls)
  - TRACES: longmemeval_data/traces_<ts>.json  (per-question, inspectable)
  - SUMMARY: longmemeval_data/summary_<ts>.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import Counter, defaultdict

from .lme import (build_memory, get_model, is_abstention, load_cost, load_dataset,
                  stores_for)
from .qa import LMEClient, answer_batch_parallel, assemble_context, gen_answer, grade
from .run import _balanced_subset

METHODS = ("retrieve_all", "pccr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="longmemeval_data/longmemeval_s.json")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--theta", type=float, default=0.10)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--turns-per-store", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="gpt-oss-120b")
    args = ap.parse_args()

    cost = load_cost()
    data = _balanced_subset(load_dataset(args.data), args.n, args.seed)
    print(f"LongMemEval REAL-LLM run | {len(data)} Q | model={args.model} theta={args.theta} "
          f"k={args.k} | mix={dict(Counter(e['question_type'] for e in data))}")
    model = get_model()
    client = LMEClient(model=args.model)
    print(f"keys={client.n_keys}; embedding memories...")

    # pre-embed each question's stores + question vector (CPU)
    prepared = []
    for e in data:
        mem = build_memory(e, model)
        qvec = model.encode([e["question"]], normalize_embeddings=True, show_progress_bar=False)[0]
        prepared.append((e, mem, qvec))

    traces = []
    agg = {m: {"correct": 0, "n": 0, "tokens": 0, "consults": 0,
               "store": Counter(), "by_type": defaultdict(lambda: [0, 0])} for m in METHODS}

    # ── ACCURACY + TOKENS + STORE USAGE (real answers, judged) ─────────────────
    print("\n[1/2] accuracy + tokens + store-usage (real gpt-oss answers + judge)...")
    t0 = time.perf_counter()
    for e, mem, qvec in prepared:
        for m in METHODS:
            stores = stores_for(m, e["question_type"], args.theta, cost)
            ctx, toks, used, per_store = assemble_context(e, mem, qvec, stores, args.k, args.turns_per_store)
            pred, lat = gen_answer(client, e, ctx)
            ok = grade(e, pred)
            a = agg[m]
            a["correct"] += ok; a["n"] += 1; a["tokens"] += toks; a["consults"] += len(stores)
            for s in stores:
                if per_store[s] > 0: a["store"][s] += 1
            a["by_type"][e["question_type"]][0] += ok
            a["by_type"][e["question_type"]][1] += 1
            traces.append({
                "qid": e["question_id"], "qtype": e["question_type"], "method": m,
                "question": e["question"], "gold": e.get("answer"),
                "abstention": is_abstention(e),
                "stores_consulted": sorted(stores), "per_store_hits": dict(per_store),
                "retrieved_sessions": used, "answer_sessions": e.get("answer_session_ids"),
                "injected_tokens": toks, "llm_answer": pred, "judge_correct": ok,
                "answer_latency_s": round(lat, 3),
            })
    print(f"    done in {time.perf_counter()-t0:.1f}s; {client.calls} API calls so far")

    # ── LATENCY: sequential vs parallel (pccr context, real API overlap) ───────
    print(f"\n[2/2] latency: sequential vs parallel (C={args.concurrency}) answer generation...")
    items = []
    for e, mem, qvec in prepared:
        stores = stores_for("pccr", e["question_type"], args.theta, cost)
        ctx, _, _, _ = assemble_context(e, mem, qvec, stores, args.k, args.turns_per_store)
        items.append((e, ctx))

    t = time.perf_counter()
    for e, ctx in items:
        gen_answer(client, e, ctx)
    seq_s = time.perf_counter() - t

    t = time.perf_counter()
    asyncio.run(answer_batch_parallel(client, items, args.concurrency))
    par_s = time.perf_counter() - t

    # ── report ─────────────────────────────────────────────────────────────────
    ts = int(time.time())
    print(f"\n{'='*72}\nRESULTS ({len(data)} questions)\n{'='*72}")
    print(f"{'method':<14}{'QA acc':>9}{'inj.tokens':>12}{'consults':>10}  store-usage")
    base = agg["retrieve_all"]
    for m in METHODS:
        a = agg[m]
        acc = a["correct"] / max(a["n"], 1)
        su = " ".join(f"{s}:{a['store'][s]}" for s in ("episodic", "semantic", "entity"))
        tok_red = 100 * (base["tokens"] - a["tokens"]) / max(base["tokens"], 1)
        extra = "" if m == "retrieve_all" else f"   ({tok_red:.0f}% fewer tokens, {100*(base['consults']-a['consults'])/max(base['consults'],1):.0f}% fewer consults)"
        print(f"{m:<14}{acc:>9.3f}{a['tokens']:>12}{a['consults']:>10}  {su}{extra}")

    print(f"\nper-type QA accuracy (pccr vs retrieve_all):")
    for t_ in sorted(agg['pccr']['by_type']):
        p = agg['pccr']['by_type'][t_]; r = agg['retrieve_all']['by_type'][t_]
        print(f"  {t_:<26} pccr {p[0]:>2}/{p[1]:<2}   retrieve_all {r[0]:>2}/{r[1]:<2}")

    spd = seq_s / par_s if par_s else 0
    print(f"\nLATENCY (real gpt-oss-120b answer generation, {len(items)} calls):")
    print(f"  sequential : {seq_s:7.1f}s")
    print(f"  parallel   : {par_s:7.1f}s   (C={args.concurrency})")
    print(f"  speedup    : {spd:6.2f}x   ({100*(seq_s-par_s)/seq_s:.0f}% less wall-clock)")
    print(f"  NOTE: real speedup is capped by Cerebras rate limits across {client.n_keys} keys.")

    summary = {
        "n": len(data), "model": args.model, "theta": args.theta, "k": args.k,
        "methods": {m: {"qa_accuracy": agg[m]["correct"]/max(agg[m]["n"],1),
                        "injected_tokens": agg[m]["tokens"], "consults": agg[m]["consults"],
                        "store_usage": dict(agg[m]["store"]),
                        "by_type": {k: v for k, v in agg[m]["by_type"].items()}} for m in METHODS},
        "latency": {"sequential_s": seq_s, "parallel_s": par_s, "speedup": spd,
                    "concurrency": args.concurrency, "n_keys": client.n_keys},
        "total_api_calls": client.calls,
    }
    tr_path = f"longmemeval_data/traces_{ts}.json"
    sm_path = f"longmemeval_data/summary_{ts}.json"
    json.dump(traces, open(tr_path, "w"), indent=2)
    json.dump(summary, open(sm_path, "w"), indent=2)
    print(f"\nsaved {len(traces)} traces -> {tr_path}")
    print(f"saved summary       -> {sm_path}")


if __name__ == "__main__":
    main()
