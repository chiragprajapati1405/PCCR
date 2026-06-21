# D. LongMemEval — real long-term-memory benchmark

LongMemEval (Wu et al., ICLR 2025): 500 questions over multi-session chat
histories, 6 question types + abstention. We map each question's history onto
three stores — **EM** (session-level embeddings), **SM** (turn-level), **ENT**
(k most-recent sessions) — and route with the ρ-gate (ρ=U/C ≥ θ) using the
**measured** store cost. Embeddings: `all-MiniLM-L6-v2`. Variant: `_s` (full
haystack, ~50 distractor sessions/question). Integration: `longmemeval/`.

## D1 — Retrieval-routing (90 balanced Q, deterministic, no API)
Metric: answerable session-recall@5 + store-consultation cost.

| method | recall | consults | EM | SM |
|---|---|---|---|---|
| retrieve_all | 0.759 | 270 | 90 | 90 |
| boolean | 0.735 | 180 | 90 | 0 |
| **pccr** | **0.735** | **225** | 90 | **45** |

PCCR matches retrieve-all recall within ~2% while **halving the expensive SM
searches** (45 vs 90). **parallel == sequential: True** on real data.
On pure recall@5, EM dominates (boolean ties) — the ρ-gate's larger value shows
in QA (D2).

## D2 — Real-LLM QA (30 balanced Q, gpt-oss-120b answers) ★
Real answers generated from ρ-gated, turn-level retrieved context; graded by a
deterministic lexical grader (proxy for LongMemEval's GPT-4o judge).

| method | QA acc | inj. tokens | consults | store usage (EM/SM/ENT) |
|---|---|---|---|---|
| retrieve_all | **0.733** | 181,896 | 90 | 30 / 30 / 30 |
| **pccr** | **0.733** | 172,558 | **75** | 30 / **15** / 30 |
| | *same acc* | **−5% tokens** | **−17% consults** | **½ SM searches** |

Per-type QA accuracy — **identical** for pccr vs retrieve_all:
| type | pccr | retrieve_all |
|---|---|---|
| knowledge-update | 5/5 | 5/5 |
| single-session-user | 5/5 | 5/5 |
| single-session-assistant | 5/5 | 5/5 |
| multi-session | 4/5 | 4/5 |
| temporal-reasoning | 3/5 | 3/5 |
| single-session-preference | 0/5 | 0/5 (grader limit) |

**Store usage:** EM on all 30 (workhorse); ENT on all 30 (~30× cheaper than a
FAISS search); SM 30→15 (PCCR prunes the turn-level search on single-session).

## D3 — Real parallel latency (30 Q answer generation)
| | wall-clock |
|---|---|
| sequential | 165.6 s |
| parallel (C=4) | 138.6 s |
| **speedup** | **1.19× (−16%)** |

Honest: the live speedup is **rate-limit-bound** (4 Cerebras keys). Brackets:
3.73× (controlled, `bench_parallel`) · ~3.0× (unthrottled burst) · 1.19× (this run).

## Limitations
- **Grader** is lexical (not the GPT-4o judge) → `single-session-preference` 0/5
  for *both* methods (long subjective golds, paraphrased answers); equal across
  methods, not a routing effect.
- **Token saving modest (5%)**: the stores overlap (SM's best turns often sit in
  EM's top sessions), so the clean win is in *number of searches*, not raw tokens.
- **θ / utility = priors**: a mis-set prior (knowledge-update→ENT) was caught
  (4/12) and fixed (→11/12) — motivates A2 counterfactual calibration here too.

## Reproduce
```bash
# data (~290 MB, gitignored) — see longmemeval/README.md
python -m longmemeval.run --n 90 --seed 0 --parallel        # D1 (no API)
source cerebras.env
python -m longmemeval.run_qa --n 30 --seed 0 --turns-per-store 6 --concurrency 4   # D2 + D3
```
Traces (per-question, inspectable): `longmemeval_data/traces_<ts>.json`;
summary: `longmemeval_data/summary_<ts>.json`. Detail:
`explanation/longmemeval_qa_results.md`.
