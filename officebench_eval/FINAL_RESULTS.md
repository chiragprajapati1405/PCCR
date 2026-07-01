# Final results — sequential baselines vs parallel PCCR (with wall time)

Baselines run SEQUENTIALLY (N=1) — they have no parallel architecture; parallelism is PCCR's
contribution. PCCR runs on its parallel architecture (N=4). Real tokens = prompt+completion.

| Arm | Exec | Accuracy | Real tokens | tok/call | calls | injected | STM | Wall time |
|---|---|---|---|---|---|---|---|---|
| no_memory | seq N=1 | 66/152 | 3.38M | 1628 | 2075 | 0 | 0 | 58.4 min |
| retrieve_all | seq N=1 | 76/152 | 5.02M | 2161 | 2323 | 778K | 0 | 70.6 min |
| **PCCR improved+PERF** | **par N=4** | 73/152 | 6.87M | **1810** | 3796 | **641K** | **10** | **26.6 min** |

## Reading
- **Wall time (the parallel-architecture win):** PCCR finishes in **26.6 min vs retrieve_all's 70.6 min
  (2.7x faster) and no_memory's 58.4 min (2.2x)** — despite doing MORE work (3796 calls). The 2nd of
  our two contributions (parallel TASKS) turns a heavier workload into a much shorter wall-clock.
  (Not the 4x ideal because PCCR does ~1.6x the work; per equal work the parallel speedup is ~4x.)
- **Accuracy:** parity within backbone run-to-run noise — retrieve_all 71-76 across runs, PCCR 72-74.
  This sequential retrieve_all happened to land high (76); PCCR is 73.
- **Per-call cost:** PCCR beats retrieve_all on tok/call (1810 vs 2161) and injected (641K vs 778K),
  and adds STM short-circuits (10 vs 0).
- **Total cost:** PCCR is highest on real tokens/calls — the completion gate's retries do more work.
  no_memory is the token floor (3.38M) but makes almost as many calls (2075) because without memory
  guidance the agent explores more.

Data: par_seq_nomem_progress.json, par_seq_ra_progress.json, perf152_final.json, final_results.json.
