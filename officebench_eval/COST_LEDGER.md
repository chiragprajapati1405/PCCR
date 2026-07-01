# Real-token cost ledger (3 arms, N=4, OfficeBench 152 tasks)

Real tokens = prompt+completion from the API usage field (true billed cost).
injected = the old injected-memory-only metric (misleading).

| Arm | Accuracy | Real tokens (in/out) | tok/call | calls | injected | STM |
|---|---|---|---|---|---|---|
| retrieve_all | 72/152 | 4.47M (3.68M/0.79M) | 1955 | 2284 | 711K | 0 |
| lean PCCR | 72/152 | 5.82M (4.42M/1.39M) | 2342 | 2484 | 552K | 13 |
| improved PCCR | 74/152 | 7.15M (5.50M/1.65M) | 2273 | 3146 | 629K | 10 |

## Verdict
- Accuracy: lean ties retrieve_all (72/152); improved +2 (74/152). Parity-to-slight-edge.
- Real cost: lean +30%, improved +60% vs retrieve_all -- PCCR is MORE expensive.
- improved injects 12% FEWER injected tokens but costs 60% MORE real tokens: the driver is
  CALLS (+38%, completion-gate got_stuck retries), each re-sending system+history. The
  inject-once injected-token saving is swamped. The '628K<734K injected = cost win' claim is dead.
- Note: improved ran at N=4 on a 10GB Colima VM (6GB OOM-thrashed); real-tokens/accuracy are
  per-task = concurrency-independent.
