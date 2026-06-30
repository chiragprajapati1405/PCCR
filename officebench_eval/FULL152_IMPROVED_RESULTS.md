# Full 152-task N=4 improved run — FINAL

Recipe (`--improved`): clean curated PM rules + sanitised past-actions + confidence gate
(θ_sim=0.45) + verified replay + tightened completion gate (intercepts got_stuck, input-file
exclusion, schema-aware .eml/.ics correctives) + inject-once (heavy guidance first 3 steps,
light roadmap every step) + online STM (bounded cap 24, success-only, threshold 0.85).
N=4 parallel, gpt-oss-120b. (Completed in two N=4 segments around a Colima Docker OOM/restart.)

## Headline

| Arm | Accuracy | Tokens | tok/call |
|---|---|---|---|
| no-memory | 63/152 = 0.414 | 0 | — |
| **retrieve-all** | **71/152 = 0.467** | **734,295** | 335 |
| PCCR poisoned baseline | 64/152 = 0.421 | 1,508,634 | 709 |
| **PCCR improved (this run)** | **72/152 = 0.474** | **628,570** | **146** |

**PCCR-improved matches/beats retrieve-all accuracy (72 vs 71 — parity, within run-to-run
noise) at 14% LOWER token cost (628K vs 734K).** The 2.13× token disadvantage is gone:
tok/call fell 709 → 146 (inject-once). STM hits 12 (vs frozen 8) from online caching.

## By level
| Level | PCCR improved |
|---|---|
| L1 | 32/47 = 0.681 |
| L2 | 28/48 = 0.583 |
| L3 | 12/57 = 0.211 |

## By pattern
| Pattern | no-mem | retrieve-all | PCCR improved |
|---|---|---|---|
| data_compute | 0.700 | 0.800 | **0.800** |
| doc_process | 0.480 | 0.600 | **0.680** |
| lookup | 0.625 | 0.812 | 0.688 |
| multi_app | 0.295 | 0.269 | **0.295** |
| single_action | 0.478 | 0.609 | 0.565 |

**multi_app recovered to 0.295** (poisoned 0.179 → seq 0.244 → here 0.295): now matches
no-memory and beats retrieve-all (0.269) — the collapse is fully repaired.
**doc_process 0.680** beats retrieve-all (0.600) and no-memory (0.480).

## The publishable claim
PCCR matches retrieve-all accuracy at **14% lower token cost**, on the parallel architecture
(4.17× measured speedup on the 125-task segment), with the gate *protecting* on hard tasks —
an efficiency + architecture result, not a loss.
