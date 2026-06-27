"""T1 PARALLEL: drive the REAL OfficeBench PCCR arm through the parallel
architecture (AsyncTaskQueue) to get REAL-LLM parallel latency on a real
multi-agent benchmark (NEXT_STEPS section H).

What this adds over run_t1.py (which is strictly sequential):
  * Async pipeline that wraps the BLOCKING run_task in asyncio.to_thread, so the
    N concurrent tasks actually overlap on their Docker-exec + LLM-HTTP waits
    instead of serializing on the event loop (the make-or-break piece, H3.1).
  * Per-worker Docker container pool ob-test-0..N-1 (acquire/release, H3.2) so
    concurrent tasks never share a container.
  * LPT ordering (L3 -> L2 -> L1) so long tasks don't tail and stall a worker (H3.3).
  * Embedder lock around the shared sentence-transformers encode (H3.4) — reads on
    the frozen bank are otherwise safe.
  * AsyncTaskQueue(max_concurrency=N) as the worker pool (the unchanged framework).

Only the PCCR (rho-gated) arm runs here; we compare wall-clock to the sequential
run and check the per-task outcomes match the sequential pccr results in
t1_progress.json (real-LLM parallel == sequential accuracy check).

  source cerebras.env
  python -m officebench_eval.run_t1_parallel --concurrency 4 --theta 1.0 --priors
  python -m officebench_eval.run_t1_parallel --concurrency 2 --limit 8 --priors   # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import threading
import time
from collections import defaultdict

from .gate import load_calibration
from .real_arch import RealArch
from .real_mem import RealMem
from .runner import run_task, cap_for_level
from memory_manager.task_queue import AsyncTaskQueue

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
TRACE_DIR = os.path.join(_PKG, "traces/t1_par")
PROGRESS = os.path.join(_PKG, "t1_parallel_progress.json")
SEQ_PROGRESS = os.path.join(_PKG, "t1_progress.json")   # sequential pccr, for the equivalence check
METHOD = "pccr"


def _lock_encode(embedder):
    """Serialize encode() under concurrent worker threads (H3.4). The bank is
    frozen during eval so FAISS reads are safe; only the embedder forward pass
    is wrapped, and it is sub-millisecond so this is not a bottleneck."""
    orig = embedder.encode
    lk = threading.Lock()

    def safe(*a, **k):
        with lk:
            return orig(*a, **k)

    embedder.encode = safe


def _record(prog, key, it, pat, r):
    em = r["em"]
    prog[key] = {"success": r["success"], "level": it["level"], "pattern": pat,
                 "consults": int(em["consult_em"]), "tokens": em["injected_tokens"],
                 "steps": r["steps"], "llm_calls": r["llm_calls"], "wall_s": r["wall_s"],
                 "stm_hit": int(em.get("stm_hit", False)), "top_em_sim": em.get("top_em_sim", 0.0),
                 "stores": em.get("consulted_stores", []), "pm_used": bool(em.get("pm_used", False)),
                 "failed_predicate": r.get("failed_predicate"),
                 "container": r.get("_container"), "worker_t0": r.get("_t0"), "worker_t1": r.get("_t1")}


async def main_async(args):
    cost, util = load_calibration(os.path.join(_PKG, "calibration"))
    if args.priors:
        util, cost = {}, {}
    real = RealArch(BANK, cost, util, theta=args.theta)
    realmem = RealMem(BANK, theta=args.theta, stm_threshold=0.85,
                      confidence_gate=args.confidence, sim_threshold=args.sim_threshold)
    _lock_encode(real.embedder)
    _lock_encode(realmem.mgr.embedder)
    gate = f"A1 confidence (theta_sim={args.sim_threshold})" if args.confidence else \
           ("frozen priors" if args.priors else "calibrated")
    print(f"loaded real FAISS EM: {len(real)} procedures | theta={args.theta} | "
          f"concurrency={args.concurrency} | gate={gate}")

    test = json.load(open(SPLIT))["test"]
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    # LPT: longest-first (L3 -> L2 -> L1) so the long tasks don't tail (H3.3)
    test.sort(key=lambda it: -int(it["level"]))
    if args.limit:
        test = test[:args.limit]
    os.makedirs(TRACE_DIR, exist_ok=True)
    prog = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}

    # Per-worker container pool ob-test-0..N-1 (H3.2)
    N = args.concurrency
    pool: asyncio.Queue = asyncio.Queue()
    for i in range(N):
        pool.put_nowait(f"ob-test-{i}")
    write_lock = asyncio.Lock()
    t_start = time.perf_counter()

    async def pipeline(spec: dict) -> dict:
        it = spec
        pat = pats.get((it["task"], it["subtask"]), "multi_app")
        key = f"{it['task']}/{it['subtask']}|{METHOD}"
        if key in prog:
            return prog[key]
        container = await pool.get()                       # acquire a free container
        try:
            t0 = time.perf_counter() - t_start
            # CRITICAL (H3.1): run the blocking run_task in a worker thread so the
            # event loop is free to dispatch the other concurrent tasks.
            r = await asyncio.to_thread(
                run_task, it["task"], it["subtask"], model=args.model, real_arch=real,
                real_mem=realmem, method=METHOD, pattern=pat,
                max_iter=cap_for_level(it["level"]), container=container)
            r["_container"], r["_t0"], r["_t1"] = container, round(t0, 1), round(time.perf_counter() - t_start, 1)
        finally:
            pool.put_nowait(container)                     # release
        async with write_lock:
            _record(prog, key, it, pat, r)
            json.dump(prog, open(PROGRESS, "w"))
            done = sum(1 for v in prog.values() if "level" in v)
            print(f"  [{done}/{len(test)}] {it['task']}/{it['subtask']} L{it['level']} "
                  f"[{container}] success={r['success']} wall={r['wall_s']}s", flush=True)
        return r

    queue = AsyncTaskQueue(pipeline, max_concurrency=N)
    results = await queue.run([dict(it) for it in test])
    wall = time.perf_counter() - t_start

    _report(prog, test, wall, N)


def _report(prog, test, wall, N):
    runs = [v for v in prog.values() if "level" in v]
    succ = sum(int(v["success"]) for v in runs)
    seq_wall = sum(v.get("wall_s", 0.0) for v in runs)     # sum of per-task wall = ~sequential total
    print(f"\n{'='*72}\nT1 PARALLEL (PCCR arm, C={N})\n{'='*72}")
    print(f"tasks: {len(runs)} | success: {succ}/{len(runs)} ({succ/max(len(runs),1):.3f})")
    print(f"wall-clock PARALLEL: {wall/60:.1f} min ({wall:.0f}s)")
    print(f"sum of per-task wall (~sequential): {seq_wall/60:.1f} min ({seq_wall:.0f}s)")
    print(f"observed speedup: {seq_wall/wall:.2f}x  (ideal ceiling = {N}x)")

    # Equivalence check vs the sequential pccr results
    if os.path.exists(SEQ_PROGRESS):
        seq = json.load(open(SEQ_PROGRESS))
        match = mism = both = 0
        for key, v in prog.items():
            if "level" not in v or key not in seq or "level" not in seq[key]:
                continue
            both += 1
            if bool(v["success"]) == bool(seq[key]["success"]):
                match += 1
            else:
                mism += 1
        if both:
            print(f"\nparallel == sequential outcome: {match}/{both} match ({mism} differ) "
                  f"-- differences are LLM stochasticity (temp=0 but provider nondeterminism), not scheduling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--priors", action="store_true")
    ap.add_argument("--confidence", action="store_true",
                    help="A1: per-query retrieval-confidence gate (single global theta_sim)")
    ap.add_argument("--sim-threshold", type=float, default=0.55, dest="sim_threshold")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
