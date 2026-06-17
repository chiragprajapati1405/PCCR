# Meeting Prep — PCCR Memory Router

## The 30-second opener
"I built a memory system for multi-agent LLM agents. The problem: agents have
many kinds of memory, and current systems search all of them on every task,
which is wasteful. My contribution is a central *router* that decides, per task,
which memories are worth consulting — using a cost-vs-usefulness score — and a
threshold that dials cost against accuracy. On the OfficeBench benchmark with a
real LLM, it matches the accuracy of 'search everything' while doing 41–87% fewer
memory lookups and injecting up to 89% fewer tokens."

## 1. The problem
- LLM agents are amnesiac; they need external memory.
- Memory isn't one thing — there are functionally distinct types (like human
  memory: skills, facts, past events, working thoughts).
- Existing systems retrieve from ALL stores on EVERY step. Wasteful: a vector
  search over episodic memory is far costlier than a key-value profile lookup,
  and for many tasks the expensive store adds nothing.
- Open question nobody fully addresses: *which* stores to read/write, per task,
  across all memory types, in a multi-agent system.

## 2. The architecture (6 memory types)
| Type | Human analogy | Cost |
|---|---|---|
| PM Procedural | how to ride a bike (rules) | cheap (dict) |
| WM Working | thought in your head now (cleared after task) | cheap |
| STM Short-Term | "I just did this" (plan cache) | cheap |
| EM Episodic | what I did last Tuesday (FAISS) | EXPENSIVE |
| SM Semantic | general facts learned (FAISS) | EXPENSIVE |
| ENT Entity | facts about a specific person | cheap |

## 3. The 10-phase lifecycle
Bootstrap → Ingestion → Retrieval → Orchestrator-Inference → Orchestrator-Output
→ Agent-Inference → Agent-Output → Observation → Storage → Consolidation.
Memory is read/written differently at each phase.

## 4. The contribution: PCCR (3 things)
1. **Phase-conditioned**: routing policy differs per phase (consulting at
   retrieval ≠ writing at storage ≠ distilling at consolidation).
2. **Cost-effectiveness cascade**: STM cache hit short-circuits everything; on a
   miss, consult store s iff ρ = utility(pattern,s)/cost(s) ≥ θ. θ is one tunable
   dial. cost(ENT)=0.15, cost(EM)=cost(SM)=0.55.
3. **Closed loop + audit**: every decision logged; utilities nudged up on
   success / down on failure.

## 5. Walk one task (use this live)
Task: "Find a common time for Bob and Tom and book a Zoom meeting."
1. Classify → pattern = multi-agent coordination.
2. STM check → miss (unseen task).
3. ρ-gate: EM 0.78/0.55=1.42 ≥1 → consult; ENT 0.72/0.15=4.8 → consult.
4. Calendar agent reads Bob's events → reads Tom's events (shared via working mem)
   → finds free hour → creates event on both calendars.
5. Success → cache plan in STM, update profiles in ENT, clear working memory.
6. Closed loop nudges multi-agent EM/ENT utility up.
Contrast: "What's in my latest email?" → lookup → EM 0.48/0.55=0.87 <1 → SKIP
episodic (it can't help a lookup). That single skip = the whole point.

## 6. Experiments
- Benchmark: OfficeBench (office automation; pass only if ALL checks pass).
- Backbone: gpt-oss-120b via Cerebras, temp 0. Embeddings: all-MiniLM-L6-v2
  (384-d), FAISS IndexFlatIP.
- Orchestrator + specialist agents (calendar, email, +document in 3-agent).
- Held-out, disjoint train/test. Baselines: Retrieve-All, Boolean (per-pattern
  on/off), PCCR (ρ-gate, swept θ=1.0/1.2/1.4).
- 3-agent uses FROZEN test memory (no writes during test) so the ONLY variable is
  routing policy.

## 7. Results (memorize these)
**3-agent comparison (32 held-out, θ=1.0):**
- Retrieve-All: 10/32, 46 consults
- Boolean: 10/32, 39 consults
- PCCR: 11/32, 27 consults  ← same/better accuracy, 41% fewer consults

**3-agent frontier:**
- θ=1.0: 11/32, 27 consults, 837 injected tokens
- θ=1.2: 11/32, 6 consults, 96 tokens (−89%)  ← sweet spot
- θ=1.4: 8/32, 6 consults  ← overshoot, accuracy drops

**Headline:** PCCR @ θ=1.2 = same accuracy as everything (11/32), 6 consults vs
retrieve-all's 46 (−87%), 89% fewer retrieved tokens.

**Per-pattern routing (proves the mechanism):**
- email_query (lookup): EM 0/7, ENT 7/7 → episodic pruned for lookups
- email_send: EM 10/10, ENT 0/10 → profile pruned
- multi-coordination: EM+ENT both
- word_query (doc read): EM 0/3, ENT 0/3 → consult NOTHING (cheapest)

**Closed loop (2-agent online run):** utilities drifted up for stores that helped
(multi.episodic 0.78→0.83), down for those that didn't (email_send.episodic
0.62→0.57).

## 8. Honest limitations (say these BEFORE she asks)
- Small scale (11/32 held-out) → proof-of-concept, not a benchmark claim;
  accuracy wiggle is single-task noise.
- We report retrieval cost (consults/tokens), not wall-clock — in our local
  setup FAISS is ms and the LLM dominates; consults/tokens are the
  deployment-relevant cost (large/remote/LLM-backed stores).
- Cost & utility values start as hand-set priors (refined by the closed loop;
  cost replaceable by logged latency). Counterfactual measurement = future work.
- Email-heavy test split; classifier is keyword-based.

## 9. Positioning vs prior work
- "Did You Check the Right Pocket?" (cost-sensitive routing) — frames the
  problem, gives only an oracle, no policy, single-agent, no phases. We give the
  implementable policy + phases + multi-agent.
- BudgetMem / U-Mem / AgeMem — learned routing but flat memory space,
  single-agent, no phase conditioning.
- MemRouter — write-side only. FluxMem — picks structure not store.
- LEGOMem — procedural memory only, no cost gate. MIRIX — 6 types but separate
  per-type agents, no central cost router.
- Grounding: value-of-information (Howard 1966), metareasoning (Russell & Wefald
  1991) — "is this lookup worth its cost" is classic decision theory.
- One-liner: "No prior system combines a six-type taxonomy + one central
  read/write router + phase & pattern conditioning + multi-agent."

## 10. What's next
Scale + balance the task set (biggest gap); report mean±std over seeds;
counterfactual usefulness measurement; learn cost/utility end-to-end; figures.
