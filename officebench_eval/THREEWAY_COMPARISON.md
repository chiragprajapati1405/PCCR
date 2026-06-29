# OfficeBench 152-task: no_memory vs retrieve_all vs PCCR (seq152 & parallel-N4)

Backbone: gpt-oss (Cerebras). PCCR recipe: clean isolated-EM + confidence gate (θ_sim=0.45)
+ verified replay + curated PM (fixed schema-valid rules) + output/completion convention
+ malformed-recovery. seq152 = sequential main-thread; parallel-N4 = 4 Docker workers.

## By Level

| Level | no_memory | retrieve_all | PCCR seq152 | PCCR parallel-N4 |
|---|---|---|---|---|
| L1 | 26/47 = 0.553 | **34/47 = 0.723** | 24/47 = 0.511 | 29/47 = 0.617 |
| L2 | 24/48 = 0.500 | **28/48 = 0.583** | 26/48 = 0.542 | 24/48 = 0.500 |
| L3 | 13/57 = 0.228 | 9/57 = 0.158 | **14/57 = 0.246** | 11/57 = 0.193 |
| **Total** | 63/152 = 0.414 | **71/152 = 0.467** | 64/152 = 0.421 | 64/152 = 0.421 |

## By Pattern

| Pattern | no_memory | retrieve_all | PCCR seq152 | PCCR parallel-N4 |
|---|---|---|---|---|
| data_compute | 7/10 = 0.700 | **8/10 = 0.800** | 5/10 = 0.500 | 7/10 = 0.700 |
| doc_process | 12/25 = 0.480 | 15/25 = 0.600 | **18/25 = 0.720** | **18/25 = 0.720** |
| lookup | 10/16 = 0.625 | **13/16 = 0.812** | 10/16 = 0.625 | 12/16 = 0.750 |
| multi_app | **23/78 = 0.295** | 21/78 = 0.269 | 19/78 = 0.244 | 14/78 = 0.179 |
| single_action | 11/23 = 0.478 | **14/23 = 0.609** | 12/23 = 0.522 | 13/23 = 0.565 |

## Tokens (total injected)

| no_memory | retrieve_all | seq152 | parallel-N4 |
|---|---|---|---|
| 0 | 734,295 | 1,565,099 | 1,508,634 |

## Reading

- **seq152 == parallel-N4 on the total (64/152)** — the sequential run reproduces the
  parallel pass count exactly, so the multi_app shortfall is *systematic*, not a
  concurrency/threading artifact. Internal spread (seq stronger on L3+multi_app, parallel
  stronger on L1+lookup) is gpt-oss run-to-run variance on identical code + bank.
- **retrieve_all leads (0.467)**, driven by easy L1 (0.723) and lookup (0.812) where
  dumping full context happens to help.
- **PCCR's reproducible win is doc_process (0.720)** — +12pp over retrieve_all in BOTH runs
  (curated-PM + verified replay).
- **L3: every PCCR run beats retrieve_all** (0.246/0.193 vs 0.158) — large RA context
  actively hurts long-horizon tasks; PCCR gating is net-positive there.
- **multi_app is the anchor** — even no_memory (0.295) beats all memory methods; fixed PM
  rules lifted it from the poisoned 0.179 → 0.244 (seq) but it is still below no-memory.

## Failure decomposition (seq152, 88 failures)

- 25 malformed-dominated (298 malformed actions across 69 tasks; 12 of the 25 are multi_app)
- 63 genuine logic failures

Traces: `traces_seq152.tar.gz` (all 152 .json + .txt). Per-task data: `seq152_progress.json`.
