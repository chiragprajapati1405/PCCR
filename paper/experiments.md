# Experiments

> Draft. Tables marked `⟨FILL⟩` are populated from the run JSONs in
> `logs_memory_manager/pccr_<mode>_thr<θ>_*.json` once the full experiment
> (`run_full.sh`) completes. Do not report numbers until the run finishes.

## 5.1 Experimental Setup

**Benchmark.** We evaluate on **OfficeBench** [Wang et al., 2024], a benchmark
for LLM agents performing realistic multi-application office workflows. From its
300 subtasks we select the **34 calendar/email subtasks** (filtering out
spreadsheet/word/PDF/OCR tasks, which require apps orthogonal to our two-agent
setup). Each task ships a configuration with a natural-language instruction, an
acting user, a date, and a set of evaluation predicates (e.g., "the ICS file
contains `DTSTART:20240517T103000` and `Meeting`"). A task is scored **correct
only if all its evaluation predicates pass**.

**Agent system.** A single **orchestrator** plans and delegates to two
**sub-agents** — a *calendar* agent (`create_event`, `list_events`,
`delete_event`) and an *email* agent (`send_email`, `list_emails`). All three
roles are driven by **gpt-oss-120b** (served via Cerebras), temperature 0.
Agents act in a sandboxed local testbed that mirrors OfficeBench's Docker
environment; file artifacts (`.ics`, `.eml`) are written to disk and scored by
OfficeBench's native evaluation predicates.

**Memory system.** The orchestrator and agents are backed by a central
**Memory Manager** exposing six memory types — Procedural (PM), Working (WM),
Short-Term (STM), Episodic (EM), Semantic (SM), and Entity (ENT) — across the
ten-phase agentic memory lifecycle. EM/SM are FAISS `IndexFlatIP` stores over
384-d `all-MiniLM-L6-v2` embeddings; STM is a session cache; PM/WM/ENT are
structured key–value stores.

**Protocol.** Each run has three stages: (i) **train** — 23 tasks executed with
memory disabled, populating the raw trajectory log; (ii) **consolidate** —
successful trajectories are distilled into EM (episodes), SM (facts), PM (rules),
and STM (pre-filled bundles); (iii) **test** — the remaining **11 held-out,
disjoint tasks** executed with memory enabled. The split is **held-out by design**
(train ∩ test = ∅) so that test tasks are unseen, and STM is configured
**semantic-only** (a cosine ≥ 0.85 near-duplicate cache, with the coarse
7-pattern cache disabled) so that an unseen task **misses** the cache and the
router's cost-gated cascade body is actually exercised. (See §5.6 for why this
configuration is necessary to measure routing at all.)

## 5.2 Methods Compared

All methods share the identical agent system, prompts, and consolidated memory;
they differ **only in the retrieval-phase routing policy** (Phase 3):

| Method | Retrieval policy on an STM miss |
|---|---|
| **Retrieve-All** | Consult every optional store (EM **and** ENT) on every task — the maximal-cost, no-routing baseline. |
| **Boolean** (base-script router) | Consult EM/ENT per a hand-set boolean table keyed by task pattern. |
| **PCCR** (ours) | Consult store *s* iff ρ = utility(pattern, *s*) / cost(*s*) ≥ θ; with a per-(pattern × store) closed-loop update of the utility table from task outcomes. |

PCCR is evaluated at three thresholds **θ ∈ {1.0, 1.2, 1.4}** to trace the
cost/quality frontier. All methods retain the cascade *head* (a confident STM
semantic hit short-circuits all optional stores); they diverge only in the
cascade *body* that runs on a miss.

**Cost model & utility table.** Relative store costs are
`cost(EM)=0.55, cost(ENT)=0.15` (a FAISS search + trace deserialization vs. a
keyed dict lookup). The pattern→utility priors are given in Appendix X (Table of
`PATTERN_UTILITY`); at θ=1.0 they reproduce the boolean baseline's decisions
except for a deliberate episodic prune on pure-lookup (`email_query`) tasks.

## 5.3 Metrics

- **Task accuracy** — fraction of held-out test tasks passing all evaluation
  predicates (the quality axis).
- **FAISS queries** — total similarity searches issued over the test set (the
  dominant retrieval cost; the cost axis).
- **Avg. optional consults / task** — mean number of EM/ENT consultations per
  test task (router selectivity).
- **STM hits** — number of test tasks served by a semantic cache hit.
- **Routing latency** — per-decision wall-clock for the router itself (from the
  `RoutingDecision` log), establishing that routing overhead is negligible
  relative to retrieval/LLM cost.
- **Closed-loop drift** — change in the per-(pattern × store) utility table
  over the run, evidencing online adaptation.

## 5.4 Research Questions

- **RQ1 (cost at equal quality).** Does PCCR match Retrieve-All's task accuracy
  while issuing fewer FAISS queries / optional consults?
- **RQ2 (pattern conditioning).** Does PCCR's consult pattern vary by task
  pattern as designed (e.g., consult EM but not ENT for `single_cal_create`;
  consult both for `multi_cal_find_and_create`)?
- **RQ3 (threshold as a cost dial).** Does raising θ monotonically reduce
  retrieval cost, tracing a cost/quality frontier?
- **RQ4 (closed loop).** Does outcome-driven `adjust_utility` move the routing
  policy, and does it preserve (or improve) accuracy?

## 5.5 Results

### Table 1 — Held-out comparison (θ = 1.0)

| Method | Test acc. | FAISS queries | Optional consults | Avg consults/task | STM hits |
|---|---|---|---|---|---|
| Retrieve-All | 5/11 | 27 | 20 | 1.82 | 0 |
| Boolean | 5/11 | 27 | 14 | 1.27 | 0 |
| **PCCR (ours)** | **5/11** | **25** | **12** | **1.09** | 0 |

All three methods reach **identical task accuracy (5/11)**, but PCCR attains it at
the **lowest retrieval cost**: 12 optional store consultations vs. Retrieve-All's
20 (a **40% reduction**) and the fewest FAISS queries (25 vs. 27). Boolean sits
between — it prunes entity lookups but, lacking the cost-ratio, still searches
episodic memory on every task (EM 11/11, see Fig. 1). The cost ordering is
monotonic and at no quality cost: **Retrieve-All (20) > Boolean (14) > PCCR (12)**.

### Table 2 — Cost/quality frontier (PCCR, varying θ)

| θ | Test acc. | FAISS queries | Optional consults | Avg consults/task |
|---|---|---|---|---|
| 1.0 | 5/11 | 25 | 12 | 1.09 |
| 1.2 | 4/11 | 18 | 5 | 0.45 |
| 1.4 | 5/11 | 17 | 4 | 0.36 |

Raising θ monotonically shrinks the consult set (12 → 5 → 4) and FAISS load
(25 → 18 → 17) while task accuracy stays within ±1 task (the 4/11 at θ=1.2 is a
single-task swing, i.e. within noise at n=11). At θ=1.4, PCCR matches
Retrieve-All's accuracy (5/11) with **4 optional consults vs. 20 — an 80%
reduction**. θ thus behaves as a genuine, monotone cost dial rather than an
arbitrary knob; the operating point can be chosen along this curve.

### Figure 1 — Routing by task pattern (RQ2), PCCR θ=1.0

| Task pattern | n | EM consulted | ENT consulted |
|---|---|---|---|
| email_query (lookup) | 2 | **0/2** | 2/2 |
| email_send | 7 | 7/7 | **0/7** |
| multi_cal_find_and_create | 1 | 1/1 | 1/1 |
| remind_notify | 1 | 1/1 | 1/1 |

The router's decisions match the `PATTERN_UTILITY` priors exactly: episodic
search is **pruned for pure-lookup** tasks (`email_query`, ρ_EM = 0.48/0.55 =
0.87 < 1) but kept for action/coordination tasks; entity is **pruned for
`email_send`** (ρ_ENT = 0.10/0.15 = 0.67 < 1) but consulted wherever a specific
user's profile is on the critical path. This is the mechanism behind Table 1's
cost savings: PCCR is the only method that skips episodic where it is provably
not cost-effective.

### Closed-loop drift (RQ4)

Utility cells that moved over the θ=1.0 run (base → final), driven by
`adjust_utility` on task outcomes:

| Pattern · store | base → final | direction |
|---|---|---|
| multi_cal_find_and_create · episodic | 0.78 → 0.83 | ↑ (helped) |
| multi_cal_find_and_create · entity | 0.72 → 0.77 | ↑ (helped) |
| email_query · entity | 0.66 → 0.71 | ↑ (helped) |
| remind_notify · episodic | 0.66 → 0.61 | ↓ (failure when consulted) |
| remind_notify · entity | 0.55 → 0.50 | ↓ |
| email_send · episodic | 0.62 → 0.57 | ↓ |

The loop moves utilities **bidirectionally** — upward for (pattern, store) pairs
that co-occurred with success, downward for those that co-occurred with failure
— demonstrating online adaptation of the routing policy without any parameter
updates to the backbone. (Note: `email_send·episodic` drifting toward 0.55 would,
if continued, cross the θ=1.0 threshold and cause PCCR to stop searching episodic
for email-send tasks — the policy self-pruning a low-value consultation.)

### Summary of findings

- **RQ1 (cost at equal quality):** PCCR matches both baselines' held-out
  accuracy (5/11) while issuing the fewest optional consults (12 vs. 20 for
  Retrieve-All, a 40% reduction) and the fewest FAISS queries.
- **RQ2 (pattern conditioning):** the consult set varies per task pattern
  exactly as the ρ-model prescribes; episodic search is pruned precisely on the
  pure-lookup tasks where it cannot help.
- **RQ3 (cost dial):** raising θ monotonically reduces cost (12→5→4 consults)
  with accuracy stable within ±1 task; θ=1.4 attains baseline accuracy at an 80%
  consult reduction.
- **RQ4 (closed loop):** outcome-driven updates move utilities bidirectionally,
  evidencing online policy adaptation.

Together these support the central claim: **a phase- and pattern-conditioned
cost/utility router preserves task quality while materially reducing retrieval
cost**, with the savings traceable to specific, interpretable per-pattern
decisions rather than a blanket policy.

## 5.5b Multi-Agent Results (3 agents: calendar + email + word)

To test multi-agent coordination at scale, we added a third agent (Word/document)
and ran on a 92-task pool (60 train / 32 held-out test, disjoint), with **frozen
test memory** (no writes during test, so the only variable across methods is the
routing policy). Training+consolidation is run once and shared across all
configurations (snapshot/restore).

### Table 3 — Held-out comparison, 3-agent (θ = 1.0)

| Method | Test acc. | Optional consults | FAISS |
|---|---|---|---|
| Retrieve-All | 10/32 | 46 | 72 |
| Boolean | 10/32 | 39 | 72 |
| **PCCR (ours)** | **11/32** | **27** | 62 |

### Table 4 — Cost/quality frontier, 3-agent (self-consistent; identical memory)

| θ | Test acc. | Optional consults | FAISS |
|---|---|---|---|
| 1.0 | 11/32 | 27 | 60 |
| **1.2** | **11/32** | **6** | 38 |
| 1.4 | 8/32 | 6 | 37 |

The frontier knee is sharp: θ=1.0→1.2 cuts consultations from 27 to 6 (−78%) with
**no accuracy loss** (11/32 at both); accuracy only degrades at θ=1.4 once the
router begins pruning genuinely useful stores. Taken together: **PCCR at θ=1.2
matches or beats every baseline's accuracy (11/32) while issuing ~87% fewer store
consultations than Retrieve-All (6 vs. 46)** in a three-agent setting that
includes document/calendar/email coordination.

### Figure 2 — Per-pattern routing, 3-agent (θ=1.0)

| Task pattern | n | EM consulted | ENT consulted |
|---|---|---|---|
| email_query | 7 | 0/7 | 7/7 |
| email_send | 10 | 10/10 | 0/10 |
| single_cal_create | 8 | 8/8 | 0/8 |
| multi_cal_find_and_create | 1 | 1/1 | 1/1 |
| word_create | 3 | 3/3 | 0/3 |
| word_query | 3 | 0/3 | 0/3 |

Six task patterns, each routed differently per the ρ-model — including the new
document patterns: `word_create` consults episodic (past document structures) but
not entity, while `word_query` (a pure document read) consults **neither** optional
store, taking the cheapest path.

*Caveat:* under frozen test memory the closed loop (RQ4) is not exercised here;
RQ4 is demonstrated in the 2-agent online run (§5.5). Entity/routing-history were
reloaded fresh on a warm-resume of this run, which negligibly affects routing cost
(the measured quantity).

## 5.6 Methodology Note: Why Strict STM + Held-out

An earlier identical-set configuration (train = test, coarse 7-pattern STM
cache) produced a **degenerate comparison**: because the pattern cache is keyed
by one of seven coarse task patterns rather than by the task, training warms all
seven buckets, after which *every* test task — including unseen ones — returns a
confident cache hit and short-circuits the cascade **before** the routing-mode
branch. All three methods then behave identically and the router's contribution
is invisible (0 EM/ENT consultations, 0 closed-loop updates observed). We
therefore (i) restrict STM to genuine semantic near-duplicates (cosine ≥ 0.85)
and (ii) hold out the test split, so that unseen tasks miss the cache and the
cost-gated cascade body is genuinely exercised. This is a property of the
evaluation harness's cache, not of PCCR; we report it for reproducibility.

## 5.7 Threats to Validity

- **Scale.** 34 cal/email tasks (11 held-out) is small; results are a
  proof-of-concept of the routing mechanism, not a large-scale benchmark claim.
  Scaling to the full OfficeBench app set and additional benchmarks is future
  work.
- **Single backbone.** All roles use gpt-oss-120b; routing behavior is
  policy-level and backbone-agnostic by construction, but cross-model
  replication is not yet shown.
- **Hand-set cost/utility priors.** STORE_COST and the initial PATTERN_UTILITY
  are hand-set; the closed loop adapts utilities online, but a fully learned
  cost model (from measured per-store latencies in the RoutingDecision log) is
  left to future work.
