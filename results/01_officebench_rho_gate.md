# A. ρ-gate memory routing — OfficeBench

**Setup.** OfficeBench office-automation tasks; backbone **gpt-oss-120b** (Cerebras,
temp 0); orchestrator + specialist sub-agents (calendar, email, +document in
3-agent). EM/SM are FAISS `IndexFlatIP` over 384-d `all-MiniLM-L6-v2`. A task is
correct only if all its evaluation predicates pass. Held-out train/test split;
the 3-agent comparison uses **frozen test memory** so the only variable is the
routing policy.

**Methods.** `retrieve_all` (consult every optional store) · `boolean` (hand-set
per-pattern on/off table) · **PCCR** (the ρ=U/C ≥ θ gate).

## A1 — 2-agent held-out comparison (θ=1.0, 11 tasks)
| Method | Acc | Consults | FAISS |
|---|---|---|---|
| Retrieve-All | 5/11 | 20 | 27 |
| Boolean | 5/11 | 14 | 27 |
| **PCCR** | **5/11** | **12** | **25** |

Same accuracy; PCCR at lowest cost (**−40% consults** vs retrieve-all).

## A2 — 3-agent held-out comparison (θ=1.0, 32 tasks) ★
| Method | Acc | Consults | FAISS |
|---|---|---|---|
| Retrieve-All | 10/32 | 46 | 72 |
| Boolean | 10/32 | 39 | 72 |
| **PCCR** | **11/32** | **27** | **62** |

PCCR slightly exceeds the baselines' accuracy while cutting consults **~41%**.

## A3 — 3-agent cost/quality frontier (θ sweep; identical consolidated memory) ★
| θ | Acc | Consults | FAISS | Injected tokens |
|---|---|---|---|---|
| 1.0 | 11/32 | 27 | 60 | 837 |
| **1.2** | **11/32** | **6** | 38 | **96 (−89%)** |
| 1.4 | 8/32 | 6 | 37 | 96 |

Sharp knee: θ=1.0→1.2 cuts consults 27→6 (**−78%**) and injected tokens 837→96
(**−89%**) with **no accuracy loss**; accuracy only degrades at θ=1.4.
**Headline:** PCCR @ θ=1.2 matches every baseline's accuracy while issuing
**~87% fewer consults than retrieve-all** (6 vs 46).

## A4 — Per-pattern routing (mechanism, 3-agent, θ=1.0)
| Pattern | n | EM consulted | ENT consulted |
|---|---|---|---|
| email_query (lookup) | 7 | 0/7 | 7/7 |
| email_send | 10 | 10/10 | 0/10 |
| single_cal_create | 8 | 8/8 | 0/8 |
| multi_cal_find_and_create | 1 | 1/1 | 1/1 |
| word_create | 3 | 3/3 | 0/3 |
| word_query | 3 | 0/3 | 0/3 |

The aggregate savings are the sum of interpretable per-store decisions: episodic
pruned on pure lookups (ρ_EM=0.87<1), entity pruned where a profile is
irrelevant, both consulted for coordination, neither for a pure document read.

## Reproduce
```bash
./run_3agent.sh            # comparison (train once, test all modes)
./run_sweep_3agent.sh      # θ frontier
python analyze_results.py  # build the tables
python measure_tokens.py   # injected-token savings
```
Raw JSON: `results_archive/comparison_2agent_20260611/`,
`results_archive/comparison_3agent_20260611/`,
`results_archive/frontier_3agent_20260613/`.
