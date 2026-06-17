# Changelog — `rectified-architecture` branch

Branched from `main` (the working prototype). Goal: replace hand-set assumptions
with data-derived quantities and address the assumptions ledger
(`explanation/rectified_architecture.md`), **without touching the base harness**.

`git diff main...rectified-architecture`: **1 code file modified, 10 files added
(+1117 / −2 lines).**

## Untouched (intentionally)
- `mm_on_top_of_legomem.py` — base LEGOMem + MemoryManager harness
- `run_officebench_local.py`, `free_legomem.py`
- `main` branch (clean baseline preserved)

Only `pccr_on_top_of_legomem.py` (the PCCR layer) was modified.

---

## Code changes — `pccr_on_top_of_legomem.py`

### A1/A2 — cost & utility are now data-derived (were hand-set)
- **Before:** `store_cost` / `pattern_utility` were always the hand-set priors
  (cost 0.55/0.15; utility table).
- **New:** `_load_calibration()` overrides them from `calibration/store_cost.json`
  and `calibration/pattern_utility.json` when present; `cost_source` /
  `utility_source` tags record provenance ("measured" vs "prior").
- **New:** `force_stores` hook + override in `route_query` to force a store
  on/off — enables the with/without-store counterfactual used by
  `calibrate.measure_utilities`.

### A6 — routing-decision latency separated from retrieval latency
- **Before:** `RoutingDecision.latency_us` mixed the routing decision and the
  store reads into one number.
- **New:** `decision_us` (ρ-gate only, ~15µs) and `retrieval_us` (FAISS/ENT
  reads, ~17ms) are timed and logged separately → "routing overhead is
  negligible" is now a measured claim.

### A7 — PM `learned_rules` are now populated (were always empty)
- **Before:** `learned_rules` was read by `route_query` but never written →
  always empty in OfficeBench runs.
- **New:** consolidation distills a rule per task pattern (most-common plan
  signature among successes, support ≥ 2) and appends it; the orchestrator
  prompt injects them under "LEARNED FROM EXPERIENCE".

### A13 — added a `similarity` baseline routing mode
- **Before:** baselines were `pccr`, `boolean`, `retrieve_all`.
- **New:** `similarity` — RAG-style "consult a store iff its top retrieved item
  clears a similarity threshold" (no cost, no pattern), for a fairer comparison.

---

## New files

| File | Purpose |
|------|---------|
| `calibrate.py` | `measure_costs` (local, no API) + `measure_utilities` (counterfactual, API) — turns cost/utility from hand-set into measured |
| `calibration/store_cost.json` | measured store cost (EM 4.543 / ENT 0.15; token ratio 30×) auto-loaded by the router |
| `explanation/rectified_architecture.md` | assumptions ledger + per-row rectification status (reviewer doc) |
| `explanation/results.md` | all experiment results with explanations + tables |
| `explanation/cheatsheet.md` | one-page formula/tables/results cheat-sheet |
| `explanation/design_table.md` | memory-access + concurrency design tables |
| `explanation/flow.txt`, `formula_p.txt`, `memory_schema.txt` | working notes |
| `paper/meeting_prep.md` | meeting walkthrough + Q&A bank |

---

## Ledger status after these changes
- ✅ Rectified in code: **A1** (cost), **A6** (latency split), **A7** (learned
  rules), **A13** (similarity baseline).
- 🔧 Harness ready, needs an API run: **A2** (counterfactual utility), **A5**
  (scale + multi-seed), **A8** (2nd backbone).
- 📝 Justified design choices (not hand-set numbers): **A3** (keyword
  classifier), **A4** (taxonomy), **A9** (frozen/online), **A11** (SM folded in
  EM), **A12** (strict STM).
- ⚖️ Reviewer decisions: **A10** (ENT task-critical?), **A14** (parallel
  multi-task?).

## Consequence to note
Measuring cost revealed the prototype under-penalized episodic search (real token
ratio 30× vs the hand-set 3.7×), so **θ must be re-swept on the measured cost
scale** — the old θ=1.0 frontier was coupled to the compressed hand-set costs.

## Remaining commands (your API)
```bash
set -a; source cerebras.env; set +a
python calibrate.py utilities 10                                  # A2: measure utility
PCCR_AGENTS=3 PCCR_RESUME=1 PCCR_THRESHOLDS="0.1,0.15,0.2,0.3" \
  python pccr_on_top_of_legomem.py                               # re-sweep θ on measured cost
```
