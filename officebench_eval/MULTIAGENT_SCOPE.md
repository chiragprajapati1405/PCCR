# Multi-Agent OfficeBench Runner — Scope

## Goal
Replace OfficeBench's single-agent loop with a LegoMem-style **orchestrator + per-app
agents** loop, so curated EM actually pays off, and the **ρ-gate** has heterogeneous
utility to gate. Headline result: **PCCR ρ-gate matches LegoMem (always-retrieve)
accuracy at lower consult cost** (cost–accuracy frontier).

## Why (established)
- Single-agent harness → curated EM utility ≈ 0 → gate degenerates (suppresses all).
- LegoMem gains come from (a) **orchestrator memory** (+7.5pp alone) on task
  decomposition/delegation — a step our single-agent loop *doesn't have* — and
  (b) **subtask-conditioned** agent memory. Even an all-`gpt-4o-mini` team gains +13pp,
  so the gap is **architecture, not backbone**.

## Execution loop (LegoMem-Dynamic style)
```
plan = orchestrator.plan(task, orch_memory?)            # orch memory ρ-gated, top-5 full-task plans
for round in range(max_rounds):
    deleg = orchestrator.next(task, history, plan)      # -> {agent: <app>, subtask: <text>} | FINISH
    if deleg == FINISH: break
    a_mem = retrieve_agent(deleg.subtask) if gate(deleg) else []   # ϕ(subtask), top-3, ρ-gated
    result = agent[deleg.agent].run(deleg.subtask, env, a_mem)     # emits app actions in shared env
    history.append((deleg, result))                     # orchestrator re-plans on result
evaluate config["evaluation"] predicates against /testbed   # unchanged, state-based
```

## Components: reuse vs new
REUSE (unchanged):
- `OfficeBench/utils/env.py` — Docker env, `exec_action`, state-based evaluation. One
  shared container; agents act on it via `{app, action, ...}` (switch_app + app actions).
- `memory_manager/stores/episodic.py` (EM), `router.py` (ρ-gate), `RealArch`.
- `cerebras_llm.py`; the `pccr_policy.py` fixes (max_tokens=2048, first-balanced-JSON
  `proc_action`) — lift both into the agent policy.
- `em_bank.json` (62 curated procedures: orchestrator `plan` + `curated_subtasks`).
- Optionally `memory_manager` `run_planned_task`/`Delegation` as the loop spine.

NEW:
1. **Orchestrator policy** (~150 LOC): decompose-and-delegate prompt; output one
   delegation `{agent, subtask}` or FINISH; parse robustly. Inject orch memory
   (`em.search(task, 5)`) when ρ-gate says so.
2. **Agent policy** (~120 LOC): given a subtask + the app's available actions + current
   obs, emit app actions until the subtask is done, then return a short result summary.
   Reuses OfficeBench action execution + our proc_action/max_tokens fixes.
3. **Multi-agent runner** (~150 LOC): the loop above tying orchestrator ↔ agents ↔ env;
   per-task trajectory + eval; logs consults/tokens/latency. (sibling of `runner.py`)
4. **Subtask-conditioned retrieval**: call agent retrieval with `deleg.subtask`
   (not the overall task) — small change at the call site; `RealArch.preload` already
   takes arbitrary query text.
5. **Per-store / per-(agent,pattern) utility** measurement to set θ for the gate.

## ρ-gate integration (the novelty under test)
- **EM-orchestrator**: gate per task by pattern utility (consult full-task plans or not).
- **EM-agent**: gate per (agent, subtask) — start with per-pattern utility, refine to
  per-agent. This is where selective consultation saves the most (skip agent memory on
  trivial subtasks: lookups, single set_cell, etc.).
- Gate WINS iff utility is **heterogeneous** — verify in M2 before claiming the frontier.

## Experiment plan (the 3 arms)
| arm | orch mem | agent mem | expected |
|---|---|---|---|
| no-memory | – | – | base (sanity ≥ single-agent base) |
| LegoMem (always) | always top-5 | always top-3 | accuracy ↑ (replicate +pp), cost high |
| **PCCR ρ-gate** | gated | gated | accuracy ≈ LegoMem, **consult tokens ↓** |
Measure per arm: success by level (L1/L2/L3), #consults + injected tokens, latency.
Run on the existing 152-task test split; reuse Docker `ob-test`, 10-key round-robin.

## Milestones
- **M1** — multi-agent loop, NO memory. Sanity: base success in multi-agent mode
  (should be ≥ single-agent base on the fixed pipeline). De-risks the loop/protocol.
- **M2** — add orch + agent memory (always-on). Replicate a LegoMem-style gain; this is
  the proof memory now works. Also yields per-store utility for θ.
- **M3** — add the ρ-gate; run 3 arms; show the cost–accuracy frontier vs LegoMem.

## Risks / open questions
- **Delegation protocol**: agent must reliably signal "subtask done"; orchestrator must
  re-plan. Needs careful prompting (LegoMem solves this; budget iteration).
- **gpt-oss as orchestrator**: decomposition quality unknown for this backbone (LegoMem
  used GPT-4o for orch). SLM team worked, so plausible, but M1/M2 will tell.
- **API cost / 429s**: multi-agent = more LLM calls/task (orch round + agent steps).
  3 arms × 152 tasks is heavy and rate-limited; throttle, checkpoint, run overnight.
- **Shared env**: agents interleave on one container (no per-agent isolation) — fine for
  the accuracy/gate test. The 2-D *parallel* sub-agents claim is a separate concern that
  needs per-task env isolation; keep this runner sequential first.
- **Scope creep**: this is essentially LegoMem's framework + our gate. Keep agents/orch
  minimal; don't reimplement LegoMem's QueryRewrite variant initially.

## Effort
~600–800 LOC new + prompt iteration; M1 is the bulk of the risk. Calibration + 3-arm eval
are API-bound (days of throttled runs), not code-bound.
