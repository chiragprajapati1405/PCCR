# LongMemEval integration for PCCR

Runs the PCCR **ρ-gate** (and the **parallel** execution path) on the **real**
[LongMemEval](https://github.com/xiaowu0162/LongMemEval) benchmark
(Wu et al., ICLR 2025) — 500 long-term-memory questions over multi-session chat
histories.

## 1. Download the data (not committed; ~290 MB)

```bash
mkdir -p longmemeval_data && cd longmemeval_data
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
wget https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json -O longmemeval_s.json
cd ..
```

`longmemeval_data/` is gitignored. `longmemeval_s.json` is the full-haystack
variant (~50 distractor sessions/question) — the one that actually exercises
retrieval/routing. `oracle` keeps only evidence sessions (recall ≈ trivial).

## 2. How it maps onto our memory architecture

Each question's chat history is ingested into three stores (per-question,
isolated — the external analogue of per-task WM isolation):

| Store | Built from | Role |
|---|---|---|
| **EM** (episodic) | SESSION-level embeddings | "which past conversation" — the workhorse |
| **SM** (semantic) | TURN-level embeddings | fine-grained fact units |
| **ENT** (entity) | k most-RECENT sessions | current / updated facts |

The **ρ-gate** (`ρ = U(question_type, store) / C(store) ≥ θ`) decides which
stores to consult per question. `C(store)` is the **measured** cost from
`calibration/store_cost.json` (EM/SM are FAISS searches; ENT is a cheap recency
lookup). Embeddings: `all-MiniLM-L6-v2`.

## 3. Run

```bash
# retrieval-routing eval (deterministic, no API): PCCR vs retrieve-all vs boolean
python -m longmemeval.run --n 90 --seed 0

# also verify the parallel path matches sequential
python -m longmemeval.run --n 90 --seed 0 --parallel

# full set
python -m longmemeval.run --n 500
```

Reports per-method **answerable session-recall**, **abstention accuracy**, and
**store-consultation cost**, plus a per-question-type breakdown.

## 4. What this measures (and what it doesn't yet)

This is the **retrieval-routing** evaluation: does the ρ-gate retrieve the
evidence sessions while consulting fewer/cheaper stores? It is deterministic and
needs no API or LLM judge.

It does **not** yet measure end-to-end **QA-answer accuracy**, which requires
generating an answer from the retrieved context with an LLM and judging it
(LongMemEval uses a GPT-4o judge via `src/evaluation/evaluate_qa.py`). The
ρ-gate's differentiation is expected to be larger there (fine-grained SM context
helps the answer, and routing changes injected-token cost), since on pure
session-recall@5 the episodic store already dominates. Adding the QA layer
(generation with gpt-oss-120b + a judge) is the natural next step.

## Honest finding from the first real run (90 balanced questions)

- PCCR matches retrieve-all recall within ~2% (0.735 vs 0.759) while **halving
  the expensive SM searches** (45 vs 90) and cutting total consults 17%.
- **Parallel == sequential**: identical outcomes and consults — equivalence
  holds on real data, not just synthetic tests.
- A mis-set prior was caught and fixed: routing `knowledge-update` to ENT-recency
  *instead of* EM tanked its recall (4/12); the evidence lives in past sessions,
  so EM must stay on (→ 11/12). This is exactly the case for **A2-style
  counterfactual calibration** of the utility table instead of hand-setting it.
