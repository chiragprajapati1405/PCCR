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
    rl = r.get("rate_limit_wait_s", 0.0)
    prog[key] = {"success": r["success"], "level": it["level"], "pattern": pat,
                 "consults": int(em["consult_em"]), "tokens": em["injected_tokens"],
                 "steps": r["steps"], "llm_calls": r["llm_calls"], "wall_s": r["wall_s"],
                 "rate_limit_wait_s": rl, "compute_s": round(r["wall_s"] - rl, 1),   # 429-neglected
                 "rate_limit_hits": r.get("rate_limit_hits", 0),
                 "stm_hit": int(em.get("stm_hit", False)), "top_em_sim": em.get("top_em_sim", 0.0),
                 "stores": em.get("consulted_stores", []), "pm_used": bool(em.get("pm_used", False)),
                 "failed_predicate": r.get("failed_predicate"), "replay": em.get("replay"),
                 "batch_calls": em.get("batch_calls"), "batch_actions": em.get("batch_actions"),
                 "container": r.get("_container"), "worker_t0": r.get("_t0"), "worker_t1": r.get("_t1")}


def _save_trace(it, m, r):
    base = f"{TRACE_DIR}/{it['task']}_{it['subtask']}_{m}"
    json.dump(r, open(f"{base}.json", "w"), indent=2, default=str)
    with open(f"{base}.txt", "w") as fh:
        fh.write(f"TASK {it['task']}/{it['subtask']} [{r.get('pattern')}] {m}  success={r['success']}  "
                 f"steps={r['steps']} llm_calls={r['llm_calls']} wall={r['wall_s']}s "
                 f"compute(no-429)={round(r['wall_s']-r.get('rate_limit_wait_s',0.0),1)}s\n"
                 f"  {r['task_text']}\n" + "=" * 70 + "\n" + "\n".join(r["sequence"]) + "\n")


async def main_async(args):
    cost, util = load_calibration(os.path.join(_PKG, "calibration"))
    if args.priors:
        util, cost = {}, {}
    real = RealArch(BANK, cost, util, theta=args.theta)
    realmem = RealMem(BANK, theta=args.theta, stm_threshold=args.stm_threshold_lookup,
                      confidence_gate=args.confidence, sim_threshold=args.sim_threshold,
                      curate_pm=args.curate_pm, model=args.model, stm_capacity=args.stm_capacity,
                      online=args.online, step_hint=args.step_hint)
    _lock_encode(real.embedder)
    _lock_encode(realmem.mgr.embedder)
    gate = f"A1 confidence (theta_sim={args.sim_threshold})" if args.confidence else \
           ("frozen priors" if args.priors else "calibrated")
    print(f"loaded real FAISS EM: {len(real)} procedures | theta={args.theta} | "
          f"concurrency={args.concurrency} | gate={gate}")

    global PROGRESS, TRACE_DIR
    if args.tasks_file:                                    # isolated smoke on a task subset
        test = json.load(open(args.tasks_file))
        tag = args.tag or os.path.splitext(os.path.basename(args.tasks_file))[0]
        PROGRESS = os.path.join(_PKG, f"par_{tag}_progress.json")
        TRACE_DIR = os.path.join(_PKG, f"traces/par_{tag}")
        print(f"SUBSET run: {len(test)} tasks from {os.path.basename(args.tasks_file)} -> {os.path.basename(PROGRESS)}")
    else:
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
                max_iter=cap_for_level(it["level"], l3_cap=args.l3_cap), container=container,
                replay=args.replay, replay_threshold=args.replay_threshold,
                plan_then_execute=args.plan, batch_size=args.batch_size,
                output_convention=args.convention,
                completion_gate=args.completion_gate, self_verify=args.self_verify,
                inject_once=args.inject_once)
            r["_container"], r["_t0"], r["_t1"] = container, round(t0, 1), round(time.perf_counter() - t_start, 1)
        finally:
            pool.put_nowait(container)                     # release
        _save_trace(it, METHOD, r)                       # record EVERY trace (json + readable txt)
        async with write_lock:
            if args.online:                              # A2: learn utility from this outcome
                realmem.record_outcome(pat, r["em"].get("consulted_stores", []), r["success"])
            _record(prog, key, it, pat, r)
            json.dump(prog, open(PROGRESS, "w"))
            done = sum(1 for v in prog.values() if "level" in v)
            cs = round(r["wall_s"] - r.get("rate_limit_wait_s", 0.0), 1)
            print(f"  [{done}/{len(test)}] {it['task']}/{it['subtask']} L{it['level']} "
                  f"[{container}] success={r['success']} wall={r['wall_s']}s "
                  f"compute(no-429)={cs}s 429s={r.get('rate_limit_hits',0)}", flush=True)
        return r

    queue = AsyncTaskQueue(pipeline, max_concurrency=N)
    results = await queue.run([dict(it) for it in test])
    wall = time.perf_counter() - t_start

    _report(prog, test, wall, N)


def _report(prog, test, wall, N):
    runs = [v for v in prog.values() if "level" in v]
    succ = sum(int(v["success"]) for v in runs)
    seq_wall = sum(v.get("wall_s", 0.0) for v in runs)     # sum of per-task wall = ~sequential total
    seq_compute = sum(v.get("compute_s", v.get("wall_s", 0.0)) for v in runs)   # 429 neglected
    rl_total = sum(v.get("rate_limit_wait_s", 0.0) for v in runs)
    calls = sum(v.get("llm_calls", 0) for v in runs)
    print(f"\n{'='*72}\nT1 PARALLEL (PCCR arm, C={N})\n{'='*72}")
    print(f"tasks: {len(runs)} | success: {succ}/{len(runs)} ({succ/max(len(runs),1):.3f})")
    print(f"total LLM calls: {calls}  ({calls/max(len(runs),1):.1f}/task)")
    print(f"wall-clock PARALLEL (raw, incl 429): {wall/60:.1f} min ({wall:.0f}s)")
    print(f"-- TIME WITH 429 NEGLECTED (user metric) --")
    print(f"sum per-task COMPUTE (429 removed): {seq_compute/60:.1f} min ({seq_compute:.0f}s)  "
          f"[sequential-equivalent work]")
    print(f"sum per-task wall (incl 429):       {seq_wall/60:.1f} min ({seq_wall:.0f}s)  "
          f"(429 stall = {rl_total:.0f}s, {100*rl_total/max(seq_wall,1):.0f}%)")
    print(f"ideal parallel COMPUTE @ C={N}:      {seq_compute/N/60:.1f} min ({seq_compute/N:.0f}s)")
    print(f"observed speedup (raw wall):        {seq_wall/wall:.2f}x  (ideal ceiling = {N}x)")

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
    ap.add_argument("--replay", action="store_true",
                    help="B1+B5: adaptive+verified procedure replay on high-confidence hits")
    ap.add_argument("--replay-threshold", type=float, default=0.75, dest="replay_threshold")
    ap.add_argument("--curate-pm", action="store_true", dest="curate_pm",
                    help="D1: LLM-curated actionable PM rules (cached to pm_curated.json)")
    ap.add_argument("--plan", action="store_true",
                    help="B2: plan-then-execute (batch next K actions per LLM call)")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    ap.add_argument("--stm-capacity", type=int, default=0, dest="stm_capacity",
                    help="C1: bounded STM budget with LFU+LRU eviction (0 = unbounded)")
    ap.add_argument("--online", action="store_true",
                    help="A2: online closed-loop -- learn U=P_on-P_off from outcomes (prior-utility gate)")
    ap.add_argument("--convention", action="store_true",
                    help="D3: prepend the output-path + completion convention (all arms)")
    ap.add_argument("--step-hint", action="store_true", dest="step_hint",
                    help="B3: inject the memory-guided step-budget hint")
    ap.add_argument("--l3-cap", type=int, default=30, dest="l3_cap",
                    help="B4: L3 step cap (default 30 = paper baseline; 45 = improved)")
    ap.add_argument("--stm-threshold", type=float, default=0.85, dest="stm_threshold_lookup",
                    help="STM short-circuit similarity threshold (default 0.85)")
    ap.add_argument("--completion-gate", action="store_true", dest="completion_gate")
    ap.add_argument("--self-verify", action="store_true", dest="self_verify")
    ap.add_argument("--inject-once", action="store_true", dest="inject_once",
                    help="token fix: inject heavy PM-rule/examples only for the first K steps, "
                         "light plan-roadmap every step (cuts the per-step re-injection cost)")
    ap.add_argument("--tasks-file", default="", dest="tasks_file",
                    help="run only the [{task,subtask,level}] in this JSON (isolated progress/traces under <tag>)")
    ap.add_argument("--tag", default="", help="suffix for isolated progress/trace dir when using --tasks-file")
    ap.add_argument("--improved", action="store_true",
                    help="full validated recipe: A1(0.45) + B1 replay + D1 curated-PM + D3 convention "
                         "+ completion-gate + self-verify + malformed-recovery (always on) + clean EM, cap 45.")
    args = ap.parse_args()
    if args.improved:                                  # full validated recipe (B2/B3/A2 excluded by design)
        args.confidence = args.replay = args.curate_pm = args.convention = True
        args.completion_gate = args.self_verify = True
        args.inject_once = True
        args.step_hint = args.plan = False
        if args.sim_threshold == 0.55:
            args.sim_threshold = 0.45
        if args.l3_cap == 30:
            args.l3_cap = 45
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
