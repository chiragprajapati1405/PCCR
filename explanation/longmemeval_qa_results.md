# LongMemEval real-LLM run — results

Real gpt-oss-120b answers on the real LongMemEval `_s` benchmark, 30 balanced
questions (5 per type), θ=0.10, k=5, turns_per_store=6.
Traces: `longmemeval_data/traces_*.json` · summary: `longmemeval_data/summary_*.json`.

## 1. How the ρ-gate worked  (accuracy + cost)

| method | QA acc | inj. tokens | consults | store usage (EM / SM / ENT) |
|---|---|---|---|---|
| retrieve_all | **0.733** | 181,896 | 90 | 30 / 30 / 30 |
| **pccr** | **0.733** | 172,558 | **75** | 30 / **15** / 30 |
| | *same accuracy* | **−5% tokens** | **−17% consults** | **half the SM searches** |

Per-type QA accuracy (pccr vs retrieve_all) — **identical everywhere**:

| type | pccr | retrieve_all |
|---|---|---|
| knowledge-update | 5/5 | 5/5 |
| single-session-user | 5/5 | 5/5 |
| single-session-assistant | 5/5 | 5/5 |
| multi-session | 4/5 | 4/5 |
| temporal-reasoning | 3/5 | 3/5 |
| single-session-preference | 0/5 | 0/5 |

**Reading it:** the ρ-gate keeps EM (the workhorse) and the cheap ENT on for
every question, and **prunes the expensive turn-level SM search on the three
single-session types** (where one session already holds the answer). Result:
**identical accuracy at 17% fewer store consultations, half the SM searches, and
5% fewer injected tokens.** Token savings are modest because the stores overlap
(SM's best turns often sit inside EM's top sessions) — so the clear win is in the
*number of (expensive) searches*, not raw context size.

## 2. How the stores were used
- **EM (episodic, session-level):** consulted on all 30 — the primary recall path.
- **SM (semantic, turn-level):** retrieve_all 30, **PCCR 15** — used only for
  multi-session / temporal / knowledge-update, pruned on single-session.
- **ENT (entity, recency):** consulted on all 30 (it is ~30× cheaper than a FAISS
  search, so the gate almost always keeps it).

## 3. How parallelism worked  (latency)
Real overlapping gpt-oss-120b answer generation, 30 calls:

| | wall-clock |
|---|---|
| sequential | 165.6 s |
| parallel (C=4) | 138.6 s |
| **speedup** | **1.19× (−16%)** |

**Honest caveat — the real speedup is rate-limit-bound.** With 4 Cerebras keys
the provider throttles bursts, so the live speedup is far below the architecture's
ceiling. Three measurements bracket it:
- **3.73×** — controlled benchmark, no rate limit (`bench_parallel.py`, modeled latency).
- **~3.0×** — a small live burst before throttling kicked in.
- **1.19×** — this 30-question live run, rate-limited.

Correctness under parallelism was separately verified: **parallel == sequential**
outcomes (LongMemEval retrieval run + the equivalence/stress tests).

## 4. Honest limitations
- **Grader:** deterministic lexical match (gold key answer present in the answer),
  a proxy for LongMemEval's GPT-4o judge — robust on factual answers but it scores
  **single-session-preference 0/5 for both methods** (the golds are long subjective
  sentences; the model paraphrases). This is a *grader* limitation, equal across
  methods, not a routing difference.
- **Latency** is rate-limited (see above); the architecture's ceiling is the
  controlled 3.73×.
- **θ / utility table** are priors; A2-style counterfactual calibration would set
  them from data (an earlier mis-set prior routed knowledge-update away from EM
  and was caught + fixed).

## Reproduce
```bash
source cerebras.env
python -m longmemeval.run_qa --n 30 --seed 0 --turns-per-store 6 --concurrency 4
# writes longmemeval_data/traces_<ts>.json + summary_<ts>.json
```
