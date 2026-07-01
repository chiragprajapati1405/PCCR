# Full-152 performance validation (improved + noise-slim + call-cap)

Real tokens = prompt+completion (billed). compute_s = 429-neglected wall time.

| Arm | Acc | Real tokens | tok/call | calls | compute_s | injected | STM |
|---|---|---|---|---|---|---|---|
| retrieve_all | 72/152 | 4.47M | 1955 | 2284 | 3761 | 711K | 0 |
| improved (pre-perf) | 74/152 | 7.15M | 2273 | 3146 | 7396 | 629K | 10 |
| **improved+PERF** | 73/152 | 6.87M | 1810 | 3796 | 4491 | 641K | 10 |

## perf vs pre-perf improved
- accuracy -1 (73 vs 74, within run-to-run noise)
- real tokens -3.9% | tok/call -20.4% | **compute time -39.3%** | calls +20.7%

## Verdict
- WIN: compute time -39%, tok/call -20%, real tokens -4%, accuracy held.
- MISS (goal #1): calls +21% -- removing history context makes the agent take more steps.
  History slimming is a TIME/LATENCY optimization, not a calls reducer. For fewer calls,
  the lever is plan-then-execute batching (K actions per call), not context removal.

## perf152 by level / pattern
- L1: 28/47 = 0.596
- L2: 27/48 = 0.562
- L3: 18/57 = 0.316
- data_compute: 9/10 = 0.900
- doc_process: 18/25 = 0.720
- lookup: 10/16 = 0.625
- multi_app: 26/78 = 0.333
- single_action: 10/23 = 0.435


---

# Full comparison incl. SEQUENTIAL retrieve_all

Real tokens = prompt+completion (billed). compute_s = 429-neglected wall unless noted.

| Arm | Acc | Real tokens | tok/call | calls | compute_s | injected | STM |
|---|---|---|---|---|---|---|---|
| retrieve_all (SEQ N=1)† | 71/152 | ~4.47M | ~1955 | 2194 | 6671‡ | 734K | 0 |
| retrieve_all (par N=4) | 72/152 | 4.47M | 1955 | 2284 | 3761 | 711K | 0 |
| improved (pre-perf) | 74/152 | 7.15M | 2273 | 3146 | 7396 | 629K | 10 |
| **improved+PERF** | 73/152 | 6.87M | 1810 | 3796 | 4491 | 641K | 10 |

† SEQ retrieve_all (t1_progress) predates real-token logging. Real tokens / tok-call are PER-TASK
  = concurrency-independent, so they equal the N=4 retrieve_all (4.47M / 1955). Its calls (2194) and
  injected (734K) confirm this: ~= the N=4 run (2284 / 712K), modulo gpt-oss run-to-run noise.
‣ SEQ compute_s = full per-task wall INCLUDING 429 stalls (no compute_s field in that run), so it is
  NOT comparable to the 429-neglected numbers above; a clean sequential compute needs a re-run.

## Takeaways
- Cost metrics (calls, real tokens, injected) are concurrency-independent: sequential ~= parallel
  retrieve_all (2194 vs 2284 calls; 734K vs 712K injected). So PCCR-vs-retrieve_all is scheduling-fair.
- The call-count baseline to beat is even tighter sequentially: retrieve_all SEQ = 2194 calls.
- PCCR already beats BOTH retrieve_all variants on tok/call (1810), injected (641K), STM (10>0);
  still loses on calls / real-tokens / compute (the completion-gate call overhead).
