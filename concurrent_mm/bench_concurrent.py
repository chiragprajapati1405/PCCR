"""STEP 7 — the benchmark: fire N tasks concurrently, sweep concurrency and PM lock-strategy,
report latency / throughput / memory-access efficiency.

The deliverable graph, in table form. Answers two questions:
  * LATENCY: how fast do N parallel tasks complete, and how does it scale with N?
  * MEMORY EFFICIENCY: does PM (read-write) hamper concurrency (lock-wait rising with N) while
    Tool Memory (read-only) stays flat? And how much does each lock strategy hamper it?

Modeled executor -> runs in seconds, ZERO API cost, no rate limits. Isolates the memory manager's
concurrency from provider latency.

Run:
    python -m concurrent_mm.bench_concurrent
    python -m concurrent_mm.bench_concurrent --subagents 4 --pm-latency 0.02 --call-latency 0.5
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import time

from concurrent_mm.executors import ModeledExecutor
from concurrent_mm.manager import MemoryManager
from concurrent_mm.procedural_memory import ProceduralMemory
from concurrent_mm.task import SubtaskSpec, TaskSpec, run_task
from concurrent_mm.tool_memory import default_tool_memory

APPS = ["calendar", "email", "excel", "word", "pdf", "shell"]


def make_tasks(n: int, subagents: int, success_rate: float, seed: int = 0) -> list[TaskSpec]:
    """Synthetic but structured tasks: a read wave + an act wave of `subagents` parallel sub-agents."""
    rng = random.Random(seed)
    tasks = []
    for i in range(n):
        apps = rng.sample(APPS, k=min(subagents, len(APPS)))
        read_wave = [SubtaskSpec(app="shell", text=f"read data for task {i}")]
        act_wave = [SubtaskSpec(app=a, text=f"task {i} do {a} step") for a in apps]
        tasks.append(TaskSpec(text=f"task {i} across {' '.join(apps)}",
                              waves=[read_wave, act_wave],
                              will_succeed=(rng.random() < success_rate),
                              task_id=i))
    return tasks


async def run_batch(tasks, strategy, pm_latency, call_latency):
    """Run `tasks` concurrently through one MemoryManager; return the batch metrics."""
    tool = default_tool_memory()
    pm = ProceduralMemory(strategy=strategy, access_latency=pm_latency)
    mm = MemoryManager(tool, pm)
    ex = ModeledExecutor(call_latency=call_latency)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[run_task(t, mm, ex) for t in tasks])
    wall = time.perf_counter() - t0

    read_waits = [r.pm_read_wait_s for r in results]
    write_waits = [r.pm_write_wait_s for r in results if r.committed]
    return {
        "wall_s": wall,
        "throughput": len(tasks) / wall,
        "avg_pm_read_wait_ms": 1000 * statistics.mean(read_waits) if read_waits else 0.0,
        "max_pm_read_wait_ms": 1000 * max(read_waits) if read_waits else 0.0,
        "avg_pm_write_wait_ms": 1000 * statistics.mean(write_waits) if write_waits else 0.0,
        "tool_read_total_ms": 1000 * mm.tool_read_time_s,
        "tool_reads": tool.reads,
        "committed": sum(1 for r in results if r.committed),
        "subagents_total": sum(r.n_subagents for r in results),
    }


async def run_docker_bench(args):
    """STEP 8 — real one-container Docker latency. Fixed actions (no LLM), lock-free PM."""
    import concurrent.futures
    from concurrent_mm.executors import DockerExecutor
    Ns = [int(x) for x in args.N.split(",")]
    loop = asyncio.get_running_loop()
    loop.set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=args.threads))
    print(f"\nConcurrent Memory Manager — REAL Docker benchmark (Step 8)")
    print(f"  ONE container '{args.container}' · per-task workdirs · {args.subagents} sub-agents/task · no LLM\n")
    ex = DockerExecutor(container=args.container)
    tool = default_tool_memory()
    print(f"{'N':>5} {'wall_s':>8} {'thruput':>9} {'speedup':>8} {'sec/task':>9} {'exec_calls':>11} {'PMwait':>8}")
    per_task_base = None
    for N in Ns:
        pm = ProceduralMemory(strategy="lockfree", access_latency=0.0)   # real in-process PM
        mm = MemoryManager(tool, pm)
        ex.cleanup_workdirs()
        tasks = make_tasks(N, args.subagents, args.success_rate, seed=args.seed)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[run_task(t, mm, ex) for t in tasks])
        wall = time.perf_counter() - t0
        per_task = wall / N
        if per_task_base is None:
            per_task_base = per_task
        speedup = per_task_base / per_task if per_task else 0.0
        pm_wait = 1000 * (pm.read_wait_s + pm.write_wait_s)
        print(f"{N:>5} {wall:>8.2f} {N/wall:>9.1f} {speedup:>7.2f}x {per_task:>8.3f}s "
              f"{ex.actions:>11} {pm_wait:>6.1f}ms")
    if args.save_outputs:
        ex.save_outputs(args.save_outputs)
        print(f"\nper-task outputs copied to {args.save_outputs} (deferred eval)")


async def main_async(args):
    if args.docker:
        return await run_docker_bench(args)
    Ns = [int(x) for x in args.N.split(",")]
    strategies = args.strategies.split(",")
    print(f"\nConcurrent Memory Manager — modeled benchmark")
    print(f"  sub-agents/task={args.subagents}  PM access latency={args.pm_latency*1000:.0f}ms  "
          f"call latency={args.call_latency*1000:.0f}ms  success_rate={args.success_rate}")
    print(f"  (Tool Memory = read-only/lock-free; PM = read-write under each strategy)\n")

    for strategy in strategies:
        print(f"=== PM strategy: {strategy} ===")
        print(f"{'N':>5} {'wall_s':>8} {'thruput':>9} {'speedup':>8} "
              f"{'PMread_wait':>12} {'PMread_max':>11} {'PMwrite_wait':>13} {'toolRead':>9} {'NxM_reads':>10}")
        per_task_base = None
        for N in Ns:
            tasks = make_tasks(N, args.subagents, args.success_rate, seed=args.seed)
            m = await run_batch(tasks, strategy, args.pm_latency, args.call_latency)
            per_task = m["wall_s"] / N
            if per_task_base is None:
                per_task_base = per_task                       # anchor speedup to the first N
            speedup = per_task_base / per_task if per_task else 0.0
            print(f"{N:>5} {m['wall_s']:>8.2f} {m['throughput']:>9.1f} {speedup:>7.2f}x "
                  f"{m['avg_pm_read_wait_ms']:>10.2f}ms {m['max_pm_read_wait_ms']:>9.1f}ms "
                  f"{m['avg_pm_write_wait_ms']:>11.2f}ms {m['tool_read_total_ms']:>7.1f}ms "
                  f"{m['subagents_total']:>10}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", default="1,10,50,100", help="comma list of concurrency levels")
    ap.add_argument("--strategies", default="lockfree,global,rwlock")
    ap.add_argument("--subagents", type=int, default=4, help="M parallel sub-agents per task")
    ap.add_argument("--pm-latency", type=float, default=0.02, dest="pm_latency",
                    help="modeled PM access latency (s) — the cost a lock may serialize")
    ap.add_argument("--call-latency", type=float, default=0.516, dest="call_latency",
                    help="modeled sub-agent step latency (s) — the LLM/tool call")
    ap.add_argument("--success-rate", type=float, default=0.7, dest="success_rate")
    ap.add_argument("--seed", type=int, default=0)
    # STEP 8 — real Docker mode
    ap.add_argument("--docker", action="store_true", help="Step 8: real one-container Docker execution")
    ap.add_argument("--container", default="cmm-bench", help="the single shared container name")
    ap.add_argument("--threads", type=int, default=64, help="thread pool for concurrent docker-exec")
    ap.add_argument("--save-outputs", default="", dest="save_outputs",
                    help="copy per-task workdirs here for deferred eval")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
