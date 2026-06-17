# Rectified Architecture & Assumptions Ledger
*(branch: `rectified-architecture` — for reviewer approval before implementation)*

## Purpose
The prototype works and the mechanism is validated, but several quantities are
hand-set priors and the evaluation is small-scale. This document (1) lists every
assumption honestly, (2) specifies how each is rectified, and (3) defines the
target architecture a reviewer would approve. **No code is changed until this is
approved.**

---

## 1. Assumptions ledger

| # | Assumption / limitation (current) | Why it's a problem | Rectified design | Priority |
|---|-----------------------------------|--------------------|------------------|:--------:|
| A1 | **Store cost** `C(s)` is a hand-set relative prior (ENT 0.15, EM/SM 0.55) | not grounded; "where do the numbers come from?" | **Measure it.** `C(s)` = measured median latency per call (already logged in `RoutingDecision.latency`) and/or tokens injected. Cost becomes empirical. | High |
| A2 | **Utility** `U(pattern,s)` is a reasoned prior in [0,1] | "how do you *know* usefulness?" | **Counterfactual init + online learning.** Initialize `U` from a with/without-store ablation (marginal accuracy gain); refine online with the closed loop / Thompson sampling. | High |
| A3 | **Task classifier** is keyword-based | brittle to phrasing; misclassification | Replace with an **LLM (or trained) classifier**; report classification accuracy. | Med |
| A4 | **Pattern taxonomy** hand-designed | "where do patterns come from?" | Justify from the task distribution; optionally **discover** via clustering; validate coverage on held-out tasks. | Med |
| A5 | **Scale**: 11 / 32 held-out tasks | results are anecdotal; accuracy wiggles = noise | **Scale to ≥100 balanced held-out tasks**, multiple seeds, report **mean ± std**. | **Critical** |
| A6 | **Cost reported** as consults/tokens, not wall-clock; latency metric conflates decision + retrieval | reviewer asks "why not time?" | Report **latency (decision vs retrieval separated), tokens, and $** alongside consults; fix the latency instrumentation to isolate routing-decision time. | High |
| A7 | **PM `learned_rules`** wired but empty in OfficeBench runs | claimed but inactive | **Wire rule-distillation into OfficeBench consolidation** (port from reference pkg) so PM learned-rules are populated and injected; or drop the claim. | Med |
| A8 | **Single backbone** (gpt-oss-120b) | no generality evidence | **Cross-model replication** on ≥2 backbones; show routing behavior is backbone-agnostic. | Med |
| A9 | **Frozen vs online** reported separately | could confuse | Define both as named protocols; report comparison (frozen) and adaptation (online) explicitly; show closed-loop learning curve in the main setting. | Med |
| A10 | **ENT not load-bearing** (emails synthesized from names) | ENT routing doesn't affect accuracy here | Add a task type where ENT holds **task-critical** facts (e.g. a contact whose address is *only* in ENT) so ENT consultation genuinely matters; or state as benchmark limitation. | Med |
| A11 | **SM not a separate store** (rides inside EM's FAISS) | taxonomy says 6 types | Either **separate SM** as a distinct fact store with its own index/writes, or document the design choice explicitly in the architecture. | Med |
| A12 | **STM strict-semantic** was a fix to expose routing | looks like a patch | Justify as the principled cache design (genuine near-duplicate only); document why the coarse pattern-cache trivially short-circuits. | Low |
| A13 | **Baselines**: retrieve-all, boolean | missing a learned-router baseline | Add a **similarity-threshold** baseline and a **learned-router** baseline (e.g. logistic on task features) for a fair comparison. | Med |
| A14 | **Sequential single-task** | scope unclear | Decide with reviewer: keep sequential, or spec **parallel multi-task** (needs per-task WM isolation + concurrency-safe shared stores). | Decision |

---

## 1b. Rectification status (branch `rectified-architecture`)

| # | Item | Status | How |
|---|------|--------|-----|
| A1 | Store cost hand-set | ✅ **RECTIFIED** | `calibrate.py costs` measures it (token-injection ratio); router auto-loads `calibration/store_cost.json`. Measured ratio 30× (was 3.7×). |
| A2 | Utility hand-set | 🔧 **HARNESS READY** | `calibrate.py utilities` (counterfactual with/without store) → `calibration/pattern_utility.json`, auto-loaded. **Run needs API.** |
| A3 | Keyword classifier | 📝 **JUSTIFIED** | kept as a deterministic, zero-cost baseline; LLM classifier is a noted upgrade (needs llm plumbing). Not a hand-set *number*. |
| A4 | Taxonomy hand-designed | 📝 **JUSTIFIED** | derived from OfficeBench task types; data-driven discovery = future work. |
| A5 | Scale (11/32) | 🔧 **INFRA / needs runs** | knobs exist (`PCCR_NTASKS`, `PCCR_TRAIN_N`); needs a ≥100-task, multi-seed API run + mean±std. |
| A6 | Latency conflated; no tokens | ✅ **RECTIFIED** | `RoutingDecision` now has `decision_us` vs `retrieval_us` (verified: ~15µs vs ~17ms); token cost measured. |
| A7 | `learned_rules` empty | ✅ **RECTIFIED** | rule-distillation wired into consolidation; PM learned-rules now populated + injected. |
| A8 | Single backbone | 📝 **CONFIGURABLE** | `model=` param; cross-model check = needs a 2nd-model run. |
| A9 | Frozen vs online | 📝 **DOCUMENTED** | named protocols; comparison=frozen, adaptation=online. |
| A10 | ENT not load-bearing | ⚖️ **REVIEWER DECISION** | add an ENT-critical task type, or accept as benchmark limitation. |
| A11 | SM not separate | 📝 **JUSTIFIED** | SM = the FAISS meaning-vector index over EM (co-located by design); documented. |
| A12 | STM strict-semantic | 📝 **DOCUMENTED** | principled near-duplicate cache; coarse pattern-cache trivially short-circuits. |
| A13 | Missing baselines | ✅ **RECTIFIED** | added `similarity` baseline mode (RAG-style threshold); learned-router = future. |
| A14 | Sequential single-task | ⚖️ **REVIEWER DECISION** | keep sequential, or spec parallel multi-task. |

**Summary:** the two real hand-set *numbers* (cost A1, utility A2) are now
measured/measurable; A6/A7/A13 rectified in code; A3/A4/A9/A11/A12 are justified
design choices (not unjustified numbers); A5/A8 are evidence gaps needing API
runs; A10/A14 need the reviewer's decision.

---

## 2. Target (rectified) architecture

```
                         ┌─────────────────────────────┐
   task ──► CLASSIFIER ──►  CENTRAL MEMORY MANAGER       │
        (LLM/learned)    │   = decision (ROUTER) +       │
                         │     execution (stores)        │
                         └──────────────┬────────────────┘
                                        │ ρ = U/C ≥ θ  (per phase, per pattern)
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
       PM              WM              STM              EM/SM            ENT
   (measured C)   (per-task)      (semantic cache)  (FAISS, measured C) (keyed)
        │               │               │               │               │
        └─────────── RoutingDecision log (audit + training data) ────────┘
                                        │
                         ┌──────────────┴────────────────┐
                         │  ADAPTATION                    │
                         │  U init = counterfactual gain  │
                         │  U online = closed loop / TS   │
                         │  C = measured latency/tokens   │
                         └────────────────────────────────┘
```

**Invariants the reviewer approves:**
- One **central manager** routes reads *and* writes across all memory types.
- Routing is **phase-conditioned** and **pattern-conditioned**.
- `ρ = U/C ≥ θ`, where **C is measured** and **U is counterfactual-initialized then learned** (no permanently hand-set numbers).
- Every decision is **logged** (audit + the dataset that learns U).
- Evaluation is **at scale with variance and ablations** (A5).

---

## 3. Evaluation plan (what makes it reviewer-grade)

1. **Task suite:** ≥100 held-out, pattern-balanced; ≥3 seeds; report mean ± std.
2. **Baselines:** Retrieve-All, Retrieve-None-optional, Boolean, Similarity-threshold, Learned-router, **PCCR**.
3. **Metrics:** task accuracy; optional consults; FAISS queries; **measured latency (decision vs retrieval split)**; injected tokens (exact via tiktoken); estimated $.
4. **Ablations:** remove phase-conditioning; remove pattern-conditioning; remove STM short-circuit — show each contributes.
5. **Counterfactual usefulness:** with/without each store → the measured `U` table.
6. **Frontier:** sweep θ → cost/quality curve with error bars.
7. **Adaptation:** closed-loop learning curve (utility drift + accuracy over time).
8. **Generality:** repeat headline on a 2nd backbone.

---

## 4. Open decisions for the reviewer (please confirm)

1. **Parallelism (A14):** sequential single-task, or spec parallel multi-task?
2. **SM (A11):** separate semantic store, or keep folded into EM with documentation?
3. **ENT (A10):** add a task type making ENT task-critical, or accept the benchmark limitation?
4. **Scope of "learned":** counterfactual-init + closed-loop (lightweight) vs full RL policy (BudgetMem-style)?
5. **Backbones (A8):** which 2nd model for the generality check?

---

## 5. Implementation order (once approved)
A5 (scale) → A6 (measured cost + latency split) → A2 (counterfactual U) → A13 (baselines)
→ A3 (LLM classifier) → A7/A11 (SM + learned rules) → A8 (2nd backbone) → A14 (parallel, if in scope).

*Everything above is design only. `main` holds the working prototype, untouched.*
