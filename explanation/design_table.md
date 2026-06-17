# PCCR — Design Specification Table (for approval before implementation)

## Table 1 — Memory access specification

| # | Memory type | Access | When READ | When WRITTEN | Accessed by | Scope / lifetime | Shared across parallel tasks? |
|---|-------------|:------:|-----------|--------------|-------------|------------------|:-----------------------------:|
| 1 | **PM** Procedural | Both (mostly read) | every orch + agent inference | bootstrap (init) + consolidation (learned rules) | orchestrator + all sub-agents | permanent / global | **Yes** (global) |
| 2 | **WM** Working | Both | retrieval + every inference | ingestion + every observation | orchestrator (owns) + sub-agents (mediated `shared_context`) | per-task (cleared at end) | **No** (isolated per task) |
| 3 | **STM** Short-Term | Both | retrieval (cache check, 1st) | storage (on success) + consolidation (prefill) | orchestrator | session | **Yes** (session cache) |
| 4 | **EM** Episodic | Both | retrieval + plan-driven agent pre-load | consolidation only | orchestrator (full-task) + sub-agents (subtask-level, mediated) | permanent / global | **Yes** (global) |
| 5 | **SM** Semantic | Both | retrieval (inside EM's FAISS) | consolidation only | orchestrator | permanent / global | **Yes** (global) |
| 6 | **ENT** Entity | Both | retrieval (gated by ρ) | bootstrap (seed) + storage (merge, on success) | orchestrator | per-user, session-persistent | **Yes** (per user) |
| 7 | **EXT** External | Both | agent tool reads (list/read files) | agent tool writes (.ics/.eml/.docx) | sub-agents (via tools) | per-task artifacts in environment | shared environment |

Notes:
- "Both" = the type is read in some phases and written in others (not necessarily in the same phase).
- EM/SM are **read-only during a task**; they are written only in the batch consolidation phase.
- WM is the only **per-task, non-shared** store — it dies when the task ends; everything else persists across tasks.

## Table 2 — System / concurrency dimensions

| Dimension | Current design | Notes / option to approve |
|-----------|----------------|---------------------------|
| Orchestrators | 1 | central planner/router |
| Sub-agents | 2 (calendar, email) or 3 (+ document) | configurable; could add more app-agents |
| Routing | 1 **central** memory manager (decides reads + writes for all 6 types) | the contribution |
| Tasks in parallel | **1 (sequential)** | each task runs start-to-finish before the next; **parallel multi-task is a future option** (would require per-task WM isolation + concurrency-safe shared stores) |
| Agents in parallel within a task | 1 at a time (orchestrator delegates sequentially) | sub-agents hand off via WM `shared_context`; parallel sub-agent execution is a possible extension |

## Table 3 — Routing policy per phase (what the manager decides)

| Phase | Routing decision | Stores touched |
|-------|------------------|----------------|
| Bootstrap | classify-by-kind | PM, ENT (write) |
| Ingestion | fixed-sink | WM (write) |
| Retrieval | **ρ = utility/cost cascade** (the contribution) | PM+WM always; STM short-circuit; gated EM/SM/ENT |
| Orch inference | consume-and-log (nothing persists) | — |
| Orch output | delegate + plan-driven EM pre-load | WM, EM (read) |
| Agent inference | fixed read-set | PM, EM (read) |
| Agent output | classify (action→WM, file→EXT) | WM, EXT |
| Observation | fixed-sink | WM (write) |
| Storage | outcome-conditional (success→STM+ENT) | STM, ENT (write), WM (clear) |
| Consolidation | batch-distribute | EM, SM, PM, STM (write) |

---

### Proposed scope to approve
- **6 cognitive memory types + EXT**, with the access pattern in Table 1.
- **1 orchestrator + 2–3 sub-agents**, **sequential** single-task execution.
- **1 central memory manager** that routes reads *and* writes across all types,
  with the ρ = utility/cost retrieval gate as the core mechanism.

*(Columns are based on the example dimensions — happy to add/remove columns per
your feedback before we proceed to implementation.)*
