# Raw artifacts — where every number comes from

Pointers to the raw data behind each result (so anything in this folder is
traceable to source).

## OfficeBench ρ-gate (area A) & rectification (area B)
| Artifact | Contains |
|---|---|
| `results_archive/comparison_2agent_20260611/` | A1 — 2-agent comparison JSONs (pccr/boolean/retrieve_all) |
| `results_archive/comparison_3agent_20260611/` | A2 — 3-agent comparison JSONs |
| `results_archive/frontier_3agent_20260613/`   | A3 — θ-frontier (pccr_thr1.0/1.2/1.4) |
| `results_archive/a5_gptoss/`                   | B5 — scale run (30 held-out, measured cost) |
| `results_archive/memcrit/`                     | B4 — memory-critical (pccr / retrieve_all) |
| `results_archive/a8_llama/`, `a8_glm/`         | A8 cross-model attempts (documented limitation) |
| `calibration/store_cost.json`                  | B1 — measured cost (EM 4.543 / ENT 0.15, ratio 30×) |
| `calibration/pattern_utility.json`             | B2 — counterfactual utility (relation_email ENT 0.875) |
| `logs_memory_manager/`                         | per-run decision/cost logs (latency_us, decision_us, retrieval_us) |

## Parallel architecture (area C)
| Artifact | Contains |
|---|---|
| `explanation/parallel_latency.md`              | C1–C3 writeup (latency, scaling, equivalence) |
| `tests/test_parallel.py`                       | 15 parallelism unit/integration tests |
| `tests/test_parallel_stress.py`               | C4 — 7 concurrency stress tests |
| `bench_parallel.py`                            | C1/C2 latency benchmark (re-runnable) |
| `verify_equivalence.py`                        | C3 — 60/60 equivalence harness |

## LongMemEval (area D)
| Artifact | Contains |
|---|---|
| `longmemeval_data/longmemeval_s.json`          | the benchmark (gitignored; download per longmemeval/README.md) |
| `longmemeval_data/traces_<ts>.json`            | D2 — 60 per-question traces (question, gold, stores, retrieved sessions, tokens, LLM answer, grade, latency) |
| `longmemeval_data/summary_<ts>.json`           | D2/D3 — aggregate metrics + latency |
| `longmemeval_data/qa_run30.log`                | D2/D3 — full run console log |
| `explanation/longmemeval_qa_results.md`        | D writeup |

> `results_archive/` and `calibration/` are committed; `longmemeval_data/`,
> `logs_memory_manager/`, and generated traces are **gitignored** (large /
> machine-specific) — regenerate with the reproduce commands in each area file.

## How to inspect LongMemEval traces
```python
import json, glob
t = json.load(open(sorted(glob.glob("longmemeval_data/traces_*.json"))[-1]))
for r in t:
    print(r["qtype"], r["method"], r["stores_consulted"],
          "tok=", r["injected_tokens"], "ok=", r["judge_correct"])
    print("  Q:", r["question"][:80]); print("  A:", r["llm_answer"][:80])
```
