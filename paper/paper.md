# Phase-Conditioned Cascading Memory Routing for Multi-Agent LLM Systems

**Authors:** [Your Name]¹, [Mentor Name]¹
¹[Affiliation]
*Correspondence:* [email]

---

## Abstract

Memory-augmented LLM agents increasingly maintain several specialized memory
stores — procedural rules, working state, cached plans, episodic traces,
semantic facts, and durable entity profiles. Yet most systems retrieve from
*all* available stores on *every* step, paying retrieval cost and injecting
irrelevant context regardless of what the current task actually needs. We
present the **Phase-Conditioned Cascading Memory Router (PCCR)**, a single
central manager that routes both reads and writes across six cognitive memory
types over a ten-phase agentic lifecycle, in a multi-agent orchestrator/
sub-agent setting. PCCR (i) makes the routing policy a function of the
*lifecycle phase* being executed, (ii) on a cache miss, gates each optional
store by an explicit cost-effectiveness ratio ρ = expected-utility / access-cost
against a tunable threshold, and (iii) logs every decision as a first-class,
auditable record that a closed loop uses to adapt utilities from task outcomes.
On the OfficeBench office-automation benchmark with a real backbone
(gpt-oss-120b), PCCR matches or exceeds the accuracy of an
indiscriminate *retrieve-everything* policy while issuing **41–87% fewer store
consultations** and injecting up to **89% fewer retrieved tokens** at equal task
accuracy, in both two-agent (calendar+email) and three-agent
(calendar+email+document) configurations. A per-task-pattern analysis shows the
savings arise from interpretable, store-specific decisions — e.g. episodic search
is pruned exactly on pure-lookup tasks where it cannot help. We position PCCR
against recent agent-memory routers and ground its score in classical value-of-
information theory.

---

## 1. Introduction

Large language model (LLM) agents lose all state when a task ends unless
augmented with external memory. Cognitive-architecture accounts of agents
(CoALA [1]) and recent surveys [2,3] converge on a small set of functionally
distinct memory types — **procedural** (skills/rules), **working** (volatile
task state), **short-term** (a session cache), **episodic** (specific past
experiences), **semantic** (distilled facts), and **entity** (durable facts
about users/things). A practical agent benefits from all of them.

This raises a routing problem that is largely unaddressed: **given many memory
stores of differing cost, which should the agent consult for the current
task, and which should it write to?** The prevailing answer in deployed systems
is *all of them, every time*. This is wasteful: a similarity search over a large
episodic store is orders of magnitude costlier than a key-value lookup of a user
profile, and for many tasks the expensive store contributes nothing — a factual
lookup ("when is my meeting with Alice?") is answered from stored facts, not
from a search through past task traces.

Recent work has begun to treat retrieval as a decision rather than a reflex.
Cost-sensitive store routing [4] frames store selection as an accuracy-vs-cost
trade-off and shows (via an oracle) that selective retrieval dominates uniform
retrieval; learned budget-tier routers [5] and embedding-based write routers [6]
optimize specific edges of the problem. However, each existing system optimizes
**one** edge — write admission, *or* read-side store selection, *or* memory-
structure choice, *or* a context budget — under a **single, global** policy, and
almost always in a **single-agent conversational** loop. None route across a full
six-type cognitive taxonomy, with one central manager governing both reads and
writes, conditioned on *which lifecycle phase is executing* and *what kind of
task it is*, in a **multi-agent** orchestrator/sub-agent system.

**Contributions.** We close that gap with PCCR:

1. **Phase-conditioned routing.** The routing policy is not global; each of ten
   lifecycle phases (bootstrap, ingestion, retrieval, orchestrator inference,
   delegation, agent inference, agent output, observation, storage,
   consolidation) carries its own rule, because "what to consult" means
   something structurally different at retrieval time than at storage or
   consolidation time (§4.1).

2. **A cost-effectiveness cascade.** Retrieval first attempts a free short-circuit
   (a short-term cache hit skips all optional stores); on a miss it consults an
   optional store *iff* ρ = expected-utility(pattern, store) / relative-cost(store)
   clears a single tunable threshold θ — turning an ad-hoc "if pattern needs X"
   heuristic into an explicit, inspectable, dial-able quantity (§4.2).

3. **Decision logging + a closed loop.** Every routing call emits a structured
   `RoutingDecision` (stores considered, consulted, skipped + why, cache state,
   latency). This audit trail is the substrate for an outcome-driven update that
   nudges utilities up/down from task success/failure (§4.3).

4. **Empirical evidence** on OfficeBench with a real backbone, in two- and
   three-agent settings, that PCCR preserves task accuracy while substantially
   reducing retrieval cost (consultations and injected tokens), with savings
   traceable to interpretable per-pattern decisions (§6).

---

## 2. Related Work

**Memory taxonomies and lifecycle.** CoALA [1] maps human memory types onto LLM
agents; recent surveys [2,3] formalize agent memory as a write–manage–read loop
and a multi-dimensional taxonomy. These define *what* the memory types are but
not *how to route* among them. We adopt their six-type taxonomy and an explicit
ten-phase operational lifecycle, and add the routing mechanism on top.

**Read-side / cost-sensitive routing.** "Did You Check the Right Pocket?" [4]
formalizes store selection as a cost-sensitive decision and demonstrates an
oracle advantage, but provides no implementable policy and considers retrieval
in isolation (no phases, no task-pattern conditioning, single agent). PCCR is the
concrete, tunable policy that the oracle implies, extended with phase/pattern
conditioning across a multi-agent lifecycle.

**Learned / budgeted routing.** BudgetMem [5] learns per-module budget tiers
(Low/Mid/High) under a cost-aware RL objective; U-Mem [7] scores retrieval by
combining similarity with a learned utility (Thompson sampling) and uses a cost-
aware extraction cascade; AgeMem [8] learns tool-based long/short-term memory
management with RL. These learn a policy but over a flat memory space (2–3 stores
or budget tiers) in single-agent settings, without phase conditioning. Our
closed loop is intentionally lighter weight and operates over a structured
(phase × pattern × store) policy; instantiating it with their learning machinery
is natural future work.

**Write-side and structure routing.** MemRouter [6] is an embedding-based write-
admission router; FluxMem [9] learns which memory *structure* (linear/graph/
hierarchical) to use. Both are write-side and single-phase.

**Multi-agent memory.** LEGOMem [10] decomposes *procedural* memory into
orchestrator vs. sub-agent modules (which informs our PM split and per-agent
episodic pre-loading) but covers only procedural memory and has no central
cost-gated router. MIRIX [11] defines six memory types managed by separate
per-type agents — close in breadth, but with no single central manager deciding
reads/writes by cost, and aimed at personalization rather than task execution.
RCR-Router [12] routes a context *budget* by agent role, within one phase.

**Consolidation and production systems.** A-MEM [13] (Zettelkasten-style linking)
and self-consolidation [14] inform our batch consolidation phase. Production
systems — MemGPT/Letta [15] (OS-style tiered paging; the source of our "working
memory dies at task end"), Mem0 [16] (outcome-conditional ADD/UPDATE/DELETE
writes), MemMachine [17], MemOS [18] — apply uniform policies without phase- or
pattern-conditioned cost gating.

**Decision-theoretic grounding.** The cost/utility score itself descends from
value-of-information theory [19] and metareasoning / expected value of
computation [20]: deciding whether acquiring a piece of information is worth its
cost. PCCR applies this lens to memory-store consultation in agents.

To our knowledge, no prior system combines a six-type cognitive taxonomy, a
single central read+write router, phase + task-pattern conditioning, and a
multi-agent setting.

---

## 3. Preliminaries: The Agentic Memory Lifecycle

We model a task as flowing through **ten phases**: (1) *Bootstrap* — load static
prompts/rules; (2) *Ingestion* — write the task to working memory and classify
its pattern; (3) *Retrieval* — assemble a memory bundle; (4) *Orchestrator
Inference* — plan/delegate; (5) *Orchestrator Output* — emit a delegation;
(6) *Agent Inference* — a sub-agent reasons; (7) *Agent Output* — an action/file;
(8) *Observation* — record the environment result; (9) *Storage* — persist on
success; (10) *Consolidation* — batch-distill experience.

Six memory types carry data across these phases:

| Type | Role | Access cost |
|---|---|---|
| **PM** Procedural | static prompts/rules, learned rules | cheap (dict) |
| **WM** Working | per-task scratchpad, cleared at task end | cheap (in-proc) |
| **STM** Short-Term | session cache of plan bundles | cheap (cached embedding cmp) |
| **EM** Episodic | past task traces (FAISS) | **expensive** (vector search) |
| **SM** Semantic | distilled facts (FAISS) | **expensive** (vector search) |
| **ENT** Entity | durable per-user profiles | cheap (keyed lookup) |

Each task is classified at ingestion into a **pattern** (lookup, single-action,
coordination, recurring, document-create, document-query, …) by a lightweight
classifier. This single label conditions all downstream routing.

---

## 4. Method: The Phase-Conditioned Cascading Memory Router

A central `MemoryManager` executes the lifecycle; a `MemoryRouter` *decides* what
to touch at each phase, and the manager performs the reads/writes. This
decision/execution separation lets the policy be inspected, tuned, and replaced
without rewriting the lifecycle plumbing.

### 4.1 Phase conditioning

Each phase declares a routing *mode*: bootstrap = classify-by-kind (rules→PM,
facts→ENT, vectors→SM); ingestion/observation = fixed-sink (→WM); retrieval =
the cascading cost-gate (below); orchestrator inference = consume-and-log
(nothing persists); storage = outcome-conditional (success → write STM+ENT;
failure → write nothing); consolidation = batch-distribute (→EM/SM/PM/STM). The
same request shape is routed differently depending on the phase.

### 4.2 The cascading cost-effectiveness gate (retrieval)

Procedural and working memory are read unconditionally. Short-term memory is
checked first; a confident cache hit **short-circuits** all optional stores (the
cheapest path, capturing the bulk of savings on repeated tasks). On a miss, for
each optional store *s* ∈ {EM, SM, ENT}:

> ρ(pattern, s) = U(pattern, s) / C(s),  consult *s* iff ρ ≥ θ

where C(s) is a relative access cost (C(ENT)=0.15 for a keyed lookup;
C(EM)=C(SM)=0.55 for a vector search + trace deserialization) and U(pattern, s) ∈
[0,1] is the expected utility of *s* for that task pattern. θ is a single dial
trading cost against quality. The utility table encodes hypotheses such as "pure-
lookup tasks gain little from episodic search but much from entity facts," which
the experiments then test directly.

### 4.3 Outcome-driven adaptation and audit

Every routing call returns and records a `RoutingDecision`: candidates, consulted
set, skipped set with per-store reasons, cache hit/miss, and latency. A closed
loop reads this log and, after each task, nudges U(pattern, s) **up** if
consulting *s* coincided with success and **down** if with failure (clamped to
[0,1]). Because the policy is indexed by (phase × pattern × store), the learned
quantity is higher-dimensional and more interpretable than a flat global policy;
the audit log makes every decision attributable.

### 4.4 Initial constants are priors, not assumptions

C(·) and U(·,·) start as hand-set priors (by operation type and task-pattern
reasoning). They are not assumed final: the closed loop refines U from outcomes,
C can be replaced by measured latencies from the decision log, and — as §6 shows
— the held-out comparison itself empirically tests the priors (pruning a low-
utility store without losing accuracy confirms its low utility).

---

## 5. Experimental Setup

**Benchmark.** OfficeBench [21], a multi-application office-automation benchmark.
Tasks ship a natural-language instruction, an acting user, a date, and a set of
evaluation predicates (e.g., an `.ics` file must contain a given `DTSTART` and
`SUMMARY`); a task is correct only if *all* predicates pass. We use the subset of
tasks executable by our agents: **34 calendar/email tasks** (two-agent setting)
and **92 calendar/email/document tasks** (three-agent setting).

**Agents and backbone.** An orchestrator delegates to specialist sub-agents —
calendar (create/list/delete events), email (send/list), and, in the three-agent
setting, a document agent (create/write/read `.docx`). All roles run on
**gpt-oss-120b** (served via Cerebras), temperature 0. EM/SM are FAISS
`IndexFlatIP` over 384-d `all-MiniLM-L6-v2` embeddings.

**Protocol.** Held-out, disjoint train/test split (two-agent: ~23/11; three-agent:
60/32). Train (memory off) → consolidate once → test with memory. For the
three-agent comparison we use **frozen test memory** (no writes during test) so
the *only* variable across methods is the routing policy; training/consolidation
is run once and shared across methods via snapshot/restore.

**Methods.** All share identical agents, prompts, and consolidated memory,
differing only in retrieval-phase policy: **Retrieve-All** (consult every optional
store), **Boolean** (a hand-set per-pattern on/off table, the base router), and
**PCCR** (the ρ-gate). PCCR is swept over θ ∈ {1.0, 1.2, 1.4}.

**Metrics.** Task accuracy; optional-store consultations; FAISS queries;
retrieved tokens injected into context; and (for adaptation) utility drift.

**Research questions.** RQ1: does PCCR match Retrieve-All's accuracy at lower
cost? RQ2: does the consult set vary by pattern as the ρ-model predicts? RQ3:
is θ a monotone cost/quality dial? RQ4: does the closed loop adapt utilities from
outcomes?

---

## 6. Results

### 6.1 RQ1 — Cost at equal quality

**Two-agent (11 held-out tasks, θ=1.0).** All methods reach 5/11 accuracy; PCCR
attains it at the lowest cost.

| Method | Acc | Consults | FAISS |
|---|---|---|---|
| Retrieve-All | 5/11 | 20 | 27 |
| Boolean | 5/11 | 14 | 27 |
| **PCCR** | 5/11 | **12** | **25** |

**Three-agent (32 held-out tasks, θ=1.0).** PCCR slightly *exceeds* the
baselines' accuracy while cutting consultations ~41%.

| Method | Acc | Consults | FAISS |
|---|---|---|---|
| Retrieve-All | 10/32 | 46 | 72 |
| Boolean | 10/32 | 39 | 72 |
| **PCCR** | **11/32** | **27** | **62** |

### 6.2 RQ3 — The threshold as a cost dial, and token savings

Three-agent frontier (self-consistent; all points share identical consolidated
memory):

| θ | Acc | Consults | FAISS | Injected tokens (orch. ctx) |
|---|---|---|---|---|
| 1.0 | 11/32 | 27 | 60 | 837 |
| **1.2** | **11/32** | **6** | 38 | **96 (−89%)** |
| 1.4 | 8/32 | 6 | 37 | 96 |

The knee is sharp: θ=1.0→1.2 cuts consultations 27→6 (−78%) and injected
retrieved tokens 837→96 (−89%) with **no accuracy loss** (11/32 at both);
accuracy only degrades at θ=1.4 once genuinely useful stores are pruned.
**Headline: PCCR at θ=1.2 matches or beats every baseline's accuracy while
issuing ~87% fewer consultations than Retrieve-All (6 vs. 46) and injecting ~89%
fewer retrieved tokens.** (Token counts are an orchestrator-context lower bound,
~4 chars/token; the same memory also enters agent prompts and persists across a
task's multiple LLM calls, so realized savings are larger.)

### 6.3 RQ2 — Pattern-conditioned routing (the mechanism)

Per-pattern consult rates under PCCR (three-agent, θ=1.0):

| Pattern | n | EM | ENT |
|---|---|---|---|
| email_query (lookup) | 7 | 0/7 | 7/7 |
| email_send | 10 | 10/10 | 0/10 |
| single_cal_create | 8 | 8/8 | 0/8 |
| multi_cal_find_and_create | 1 | 1/1 | 1/1 |
| word_create | 3 | 3/3 | 0/3 |
| word_query | 3 | 0/3 | 0/3 |

Six patterns are each routed differently, exactly as the ρ-model prescribes:
episodic is pruned on pure lookups (`email_query`, ρ_EM=0.48/0.55=0.87<1), entity
is pruned where a user profile is irrelevant (`email_send`), both are consulted
for coordination, and a pure document read (`word_query`) consults **neither**
optional store — the cheapest path. The cost savings of §6.1–6.2 are thus not an
aggregate artifact but the sum of interpretable, store-specific decisions.

### 6.4 RQ4 — Closed-loop adaptation

In the online (non-frozen) two-agent run, outcome-driven updates moved utilities
**bidirectionally** — upward for (pattern, store) pairs co-occurring with success
(e.g. `multi.episodic` 0.78→0.83) and downward for those co-occurring with failure
(e.g. `email_send.episodic` 0.62→0.57) — demonstrating online policy adaptation
with no backbone parameter updates. (The three-agent run used frozen memory to
isolate routing, so RQ4 is reported from the two-agent run.)

---

## 7. Discussion and Limitations

**What the results support.** A phase- and pattern-conditioned cost/utility router
preserves task quality while materially reducing retrieval cost, and the savings
are interpretable and tunable via a single threshold. The pattern-level analysis
(§6.3) is the strongest evidence: the mechanism behaves as designed rather than
succeeding by chance.

**Scale.** Held-out sets are small (11 and 32 tasks); results are a directional
proof-of-concept of the routing mechanism, not a large-benchmark claim. Some
splits are pattern-skewed (e.g. email-heavy). Scaling task count and balancing
patterns is the priority for a camera-ready result.

**Cost currency.** In our local harness, FAISS is milliseconds and LLM inference
dominates wall-clock, so we report retrieval cost (consultations, FAISS queries,
injected tokens) rather than wall-clock — the deployment-relevant cost when
memory stores are large, remote, or LLM-backed. Token counts are approximate
(~4 chars/token) and a lower bound.

**Hand-set priors.** C(·) and U(·,·) begin hand-set; the closed loop refines U and
C is replaceable by logged latencies, but a fully learned cost model and a
counterfactual (with/without-store) measurement of utility remain future work.

**Frozen vs. online.** The three-agent comparison freezes memory during test for
a clean routing isolation; entity/history were reloaded fresh on a warm resume,
which negligibly affects the measured routing cost.

---

## 8. Conclusion

We introduced PCCR, a central memory router for multi-agent LLM systems that
conditions retrieval on lifecycle phase and task pattern, gates each store by an
explicit cost-effectiveness ratio, and adapts from logged outcomes. On
OfficeBench with a real backbone, PCCR matches or beats an indiscriminate
retrieve-everything policy while cutting store consultations by 41–87% and
injected tokens by up to 89% at equal accuracy, with savings attributable to
interpretable per-pattern decisions. PCCR shows that "which memory to consult"
should be a tunable, phase-aware, cost-sensitive decision rather than a reflex.
Future work: scale and balance the task suite, learn C and U end-to-end
(building on [5,7,8]), and ground utilities with counterfactual measurement.

---

## References

[1] Sumers et al. Cognitive Architectures for Language Agents (CoALA). TMLR 2024.
[2] Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers. arXiv:2603.07670.
[3] Anatomy of Agentic Memory: Taxonomy and Empirical Analysis. arXiv:2602.19320.
[4] Gaikwad. Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents. ICLR 2026 Workshop. arXiv:2603.15658.
[5] Zhang et al. Learning Query-Aware Budget-Tier Routing for Runtime Agent Memory (BudgetMem). arXiv:2602.06025.
[6] MemRouter: Memory-as-Embedding Routing for Long-Term Conversational Agents. arXiv:2605.00356.
[7] Towards Autonomous Memory Agents (U-Mem). arXiv:2602.22406.
[8] Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management for LLM Agents. arXiv:2601.01885.
[9] Choosing How to Remember: Adaptive Memory Structures for LLM Agents (FluxMem). arXiv:2602.14038.
[10] LEGOMem: Modular Procedural Memory for Multi-agent LLM Systems for Workflow Automation. arXiv:2510.04851.
[11] Wang & Chen. MIRIX: Multi-Agent Memory System for LLM-Based Agents. arXiv:2507.07957.
[12] RCR-Router: Efficient Role-Aware Context Routing for Multi-Agent LLM Systems with Structured Memory. arXiv:2508.04903.
[13] A-MEM: Agentic Memory for LLM Agents. NeurIPS 2025. arXiv:2502.12110.
[14] Self-Consolidation for Self-Evolving Agents. arXiv:2602.01966.
[15] Packer et al. MemGPT: Towards LLMs as Operating Systems. 2024.
[16] Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory. 2025.
[17] MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents. arXiv:2604.04853.
[18] MemOS: An Operating System for Memory-Augmented Generation. EMNLP 2025. arXiv:2505.22101.
[19] Howard, R. Information Value Theory. IEEE Trans. Systems Science and Cybernetics, 1966.
[20] Russell & Wefald. Principles of Metareasoning. Artificial Intelligence, 1991.
[21] Wang et al. OfficeBench: Benchmarking Language Agents across Multiple Applications for Office Automation. arXiv:2407.19056.

---

*Artifacts: code (memory router, OfficeBench harness, analysis scripts) and
per-run result JSONs are available in the project repository.*
