"""T1 PROCESS-PARALLEL: the robust parallel OfficeBench driver.

Why this exists: the thread-based driver (run_t1_parallel, asyncio.to_thread) shares
ONE Python process across the N concurrent tasks -- shared signal machinery (so
OfficeBench's signal-based action timeout can't run in worker threads), shared os.chdir,
shared `apps` module, shared embedder/HTTP clients. That residual sharing leaves a
malformed-action penalty under concurrency (~21% vs ~11% sequential).

This driver gives each worker its OWN PROCESS via ProcessPoolExecutor:
  * own Python interpreter -> own MAIN thread -> the real signal-timeout works (not a no-op);
  * zero shared state (chdir/apps/embedder/clients all isolated);
  * own Docker container ob-test-<worker> for the process's lifetime.
Each worker builds RealMem/RealArch ONCE at process start (cached for its task stream).

  source cerebras.env
  python -m officebench_eval.run_t1_procs --procs 4 --confidence --replay --sim-threshold 0.45
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
PATTERNS = os.path.join(_PKG, "patterns.json")
SPLIT = os.path.join(_PKG, "split.json")
TRACE_DIR = os.path.join(_PKG, "traces/t1_procs")
PROGRESS = os.path.join(_PKG, "t1_procs_progress.json")
SEQ_PROGRESS = os.path.join(_PKG, "t1_progress.json")
METHOD = "pccr"

# ---- per-process global state (built once per worker via the initializer) ----
_W = {}


def _init_worker(cfg, container_q):
    """Runs once in each worker PROCESS: claim a container, build RealMem/RealArch."""
    from .real_arch import RealArch
    from .real_mem import RealMem
    _W["cfg"] = cfg
    _W["container"] = container_q.get()                    # this process owns this container
    _W["real"] = RealArch(BANK, {}, {}, theta=cfg["theta"])
    _W["rm"] = RealMem(BANK, theta=cfg["theta"], stm_threshold=0.85,
                       confidence_gate=cfg["confidence"], sim_threshold=cfg["sim_threshold"],
                       curate_pm=cfg["curate_pm"], step_hint=cfg["step_hint"])


def _run_one(spec):
    """Runs in the worker process's MAIN thread -> full signal-timeout, no shared state."""
    from .runner import run_task, cap_for_level
    cfg, container = _W["cfg"], _W["container"]
    t0 = time.perf_counter()
    r = run_task(spec["task"], spec["subtask"], model=cfg["model"], real_arch=_W["real"],
                 real_mem=_W["rm"], method=METHOD, pattern=spec["pattern"],
                 max_iter=cap_for_level(spec["level"], l3_cap=cfg["l3_cap"]), container=container,
                 replay=cfg["replay"], replay_threshold=cfg["replay_threshold"],
                 plan_then_execute=cfg["plan"], batch_size=cfg["batch_size"],
                 output_convention=cfg["convention"])
    r["_container"], r["_dur"] = container, round(time.perf_counter() - t0, 1)
    return spec, r


def _save_trace(spec, r):
    base = f"{TRACE_DIR}/{spec['task']}_{spec['subtask']}_{METHOD}"
    json.dump(r, open(f"{base}.json", "w"), indent=2, default=str)
    with open(f"{base}.txt", "w") as fh:
        fh.write(f"TASK {spec['task']}/{spec['subtask']} [{spec['pattern']}] success={r['success']} "
                 f"steps={r['steps']} calls={r['llm_calls']} wall={r['wall_s']}s "
                 f"compute(no-429)={round(r['wall_s']-r.get('rate_limit_wait_s',0.0),1)}s\n"
                 f"  {r['task_text']}\n" + "=" * 70 + "\n" + "\n".join(r["sequence"]) + "\n")


def _record(prog, spec, r):
    em = r["em"]; rl = r.get("rate_limit_wait_s", 0.0)
    prog[f"{spec['task']}/{spec['subtask']}|{METHOD}"] = {
        "success": r["success"], "level": spec["level"], "pattern": spec["pattern"],
        "consults": int(em["consult_em"]), "tokens": em["injected_tokens"],
        "steps": r["steps"], "llm_calls": r["llm_calls"], "wall_s": r["wall_s"],
        "rate_limit_wait_s": rl, "compute_s": round(r["wall_s"] - rl, 1),
        "rate_limit_hits": r.get("rate_limit_hits", 0), "stm_hit": int(em.get("stm_hit", False)),
        "top_em_sim": em.get("top_em_sim", 0.0), "stores": em.get("consulted_stores", []),
        "pm_used": bool(em.get("pm_used", False)), "failed_predicate": r.get("failed_predicate"),
        "replay": em.get("replay"), "container": r.get("_container")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sim-threshold", type=float, default=0.55, dest="sim_threshold")
    ap.add_argument("--replay-threshold", type=float, default=0.75, dest="replay_threshold")
    ap.add_argument("--batch-size", type=int, default=4, dest="batch_size")
    ap.add_argument("--confidence", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--curate-pm", action="store_true", dest="curate_pm")
    ap.add_argument("--convention", action="store_true")
    ap.add_argument("--step-hint", action="store_true", dest="step_hint")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--l3-cap", type=int, default=45, dest="l3_cap")
    ap.add_argument("--lean", action="store_true",
                    help="cost-optimal arm: A1 confidence + B1 replay ONLY (drop D1/D3/B3 token overhead)")
    args = ap.parse_args()
    if args.lean:
        args.confidence = args.replay = True
        args.curate_pm = args.convention = args.step_hint = args.plan = False

    cfg = {k: getattr(args, k) for k in ("theta", "model", "sim_threshold", "replay_threshold",
            "batch_size", "confidence", "replay", "curate_pm", "convention", "step_hint", "plan", "l3_cap")}

    test = json.load(open(SPLIT))["test"]
    pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(PATTERNS))}
    test.sort(key=lambda it: -int(it["level"]))            # LPT: longest first
    if args.limit:
        test = test[:args.limit]
    os.makedirs(TRACE_DIR, exist_ok=True)
    prog = json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}
    specs = [{"task": it["task"], "subtask": it["subtask"], "level": it["level"],
              "pattern": pats.get((it["task"], it["subtask"]), "multi_app")}
             for it in test if f"{it['task']}/{it['subtask']}|{METHOD}" not in prog]

    N = args.procs
    gate = f"A1 conf(theta_sim={args.sim_threshold})" if args.confidence else "priors"
    extras = "+".join([x for x, on in [("replay", args.replay), ("PM", args.curate_pm),
              ("conv", args.convention), ("hint", args.step_hint), ("plan", args.plan)] if on])
    print(f"PROCESS-parallel: {N} procs | {len(specs)} tasks to run ({len(prog)} done) | "
          f"gate={gate} {extras} | l3_cap={args.l3_cap}", flush=True)

    container_q = mp.Manager().Queue()
    for i in range(N):
        container_q.put(f"ob-test-{i}")
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=N, initializer=_init_worker, initargs=(cfg, container_q)) as ex:
        futs = [ex.submit(_run_one, s) for s in specs]
        for fut in cf.as_completed(futs):
            try:
                spec, r = fut.result()
            except Exception as e:
                print(f"  task ERR {str(e)[:100]}", flush=True)
                continue
            _save_trace(spec, r)
            _record(prog, spec, r)
            json.dump(prog, open(PROGRESS, "w"))
            done = sum(1 for v in prog.values() if "level" in v)
            cs = round(r["wall_s"] - r.get("rate_limit_wait_s", 0.0), 1)
            print(f"  [{done}/{len(test)}] {spec['task']}/{spec['subtask']} L{spec['level']} "
                  f"[{r['_container']}] success={r['success']} compute(no-429)={cs}s "
                  f"429s={r.get('rate_limit_hits',0)}", flush=True)
    _report(prog, time.perf_counter() - t0, N)


def _report(prog, wall, N):
    runs = [v for v in prog.values() if "level" in v]
    succ = sum(int(v["success"]) for v in runs)
    seq_compute = sum(v.get("compute_s", v.get("wall_s", 0.0)) for v in runs)
    seq_wall = sum(v.get("wall_s", 0.0) for v in runs)
    print(f"\n{'='*72}\nT1 PROCESS-PARALLEL ({N} procs)\n{'='*72}")
    print(f"tasks {len(runs)} | success {succ}/{len(runs)} ({succ/max(len(runs),1):.3f})")
    print(f"wall PARALLEL {wall/60:.1f} min | sum compute(no-429) {seq_compute/60:.1f} min | "
          f"speedup(raw) {seq_wall/wall:.2f}x")
    if os.path.exists(SEQ_PROGRESS):
        seq = json.load(open(SEQ_PROGRESS))
        m = sum(1 for k, v in prog.items() if "level" in v and k in seq and "level" in seq[k]
                and bool(v["success"]) == bool(seq[k]["success"]))
        b = sum(1 for k, v in prog.items() if "level" in v and k in seq and "level" in seq[k])
        if b:
            print(f"vs sequential pccr baseline: {m}/{b} outcome match")


if __name__ == "__main__":
    main()
