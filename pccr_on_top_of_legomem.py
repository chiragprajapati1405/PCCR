"""
PCCR Router on top of the LEGOMem + MemoryManager + OfficeBench harness
════════════════════════════════════════════════════════════════════════
This file does NOT modify mm_on_top_of_legomem.py. It imports that script's
engine (LLM client, embedder, OfficeBench runner, execution loop, stores)
unchanged and layers the **Phase-Conditioned Cascading Memory Router (PCCR)**
on top by subclassing its MemoryManager.

What PCCR adds over the base script's 3-level router
----------------------------------------------------
The base script's Level-2 routing is a *boolean* table:
    routing_strategy[pattern] = {"episodic": True/False, "entity": True/False}
PCCR upgrades that single edge into a principled, tunable, inspectable policy:

  1. STORE_COST          - a relative cost per optional store (FAISS search vs
                           a dict lookup are not equally expensive).
  2. PATTERN_UTILITY     - a *graded* (0..1) expected-utility per (pattern x
                           store), replacing the boolean.
  3. rho = utility/cost  - on an STM miss, each optional store is consulted iff
                           rho >= consult_threshold. The threshold is a single
                           dial that trades cost against quality -> this is what
                           produces the paper's cost/quality frontier.
  4. RoutingDecision     - every routing call emits a structured, auditable
                           record (consulted / skipped + WHY / rho / cache-hit /
                           latency). This is the substrate the closed loop uses.
  5. adjust_utility      - a per-(pattern x store) closed loop: after each task,
                           utilities of consulted stores are nudged up on
                           success / down on failure, so rho re-gates over time.
                           (This is the higher-dimensional learned policy that
                           distinguishes PCCR from flat-pool learners like
                           U-Mem / BudgetMem, which learn one global policy.)

Tuning note: at consult_threshold = 1.0 the PATTERN_UTILITY table below
reproduces the base script's hand-set boolean decisions almost exactly (two
deliberate episodic prunes for pure-lookup patterns). Raising the threshold
prunes the least cost-effective consultations first; lowering it approaches
"retrieve everything". routing_mode also lets you run the same OfficeBench
tasks under the naive baselines for a head-to-head comparison.

Run:
    python pccr_on_top_of_legomem.py                # PCCR, threshold 1.0
    PCCR_MODE=boolean python pccr_on_top_of_legomem.py        # base-script baseline
    PCCR_MODE=retrieve_all python pccr_on_top_of_legomem.py   # query-everything baseline
    PCCR_THRESHOLD=1.4 python pccr_on_top_of_legomem.py        # tighter cost budget
"""
import os
import sys
import copy
import time
import json
from dataclasses import dataclass, field

import numpy as np

# Importing the base script runs its module-level setup (stdout tee, os/open
# monkey-patching that the engine relies on, logging config, and the
# `from run_officebench_local import ...` / `from free_legomem import ...`
# imports). Its main() is guarded by __name__ == "__main__", so it does NOT
# run on import. We reuse everything below.
from mm_on_top_of_legomem import (
    MemoryManager,
    MultiLLM,
    LocalEmbedder,
    LocalOfficeBenchRunner,
    PatternSTM,
    execute_task,
    filter_cal_email,
    SubtaskMemory,
    LOCAL_TESTBED,
    LOG_DIR,
    REPO_PATH,
)


# ══════════════════════════════════════════════════════════════════
#  STRICT (semantic-only) STM
# ══════════════════════════════════════════════════════════════════
class StrictSemanticSTM(PatternSTM):
    """STM that drops the coarse Level-1 *pattern* cache and keeps only the
    Level-2 *semantic* cache (cosine >= l2_threshold over the task embedding).

    Why: the base PatternSTM.lookup() returns an L1 hit whenever a task merely
    matches one of 7 coarse patterns (single_cal_create, email_query, ...). Once
    training warms those 7 buckets, essentially EVERY task -- even an unseen,
    held-out one -- short-circuits the cascade, so the rho=utility/cost gate
    never runs and the routing modes become indistinguishable. Restricting STM
    to genuine near-duplicate (semantic) hits makes a held-out test task MISS,
    so the cascade body actually executes and the modes diverge as designed.
    Bundles are still *written* to both caches (store_bundle unchanged); we only
    refuse to *read* the coarse pattern bucket here.
    """

    def classify(self, task_desc):
        """Extend the base 7-pattern classifier with two Word patterns. Only
        applies when the base classifier (cal/email keywords) returns 'unknown',
        so cal/email tasks keep their original labels and pure-document tasks get
        word-specific routing."""
        pat, users = super().classify(task_desc)
        t0 = task_desc.lower()
        # EM-critical: "as usual / same as / like last" needs a PAST EPISODE to
        # recover the omitted detail -> its own pattern.
        if "as usual" in t0 or "same as" in t0 or "like last" in t0 or "like before" in t0:
            return "recurring_em", users
        # A10: a task referencing a RELATION ("X's manager/assistant") needs ENT
        # to resolve the recipient -> its own pattern (ENT-critical).
        if "manager" in t0 or "assistant" in t0:
            return "relation_email", users
        if pat != "unknown":
            return pat, users
        t = task_desc.lower()
        doc_words = ("document", "docx", "word file", " word ", "report", "letter",
                     "memo", "essay", "draft", ".doc")
        if any(w in t for w in doc_words) or "write" in t or "create" in t:
            if any(k in t for k in ("read", "what", "summar", "extract", "how many", "list")):
                return "word_query", users
            return "word_create", users
        return "unknown", users

    def lookup(self, task_desc):
        t_start = time.perf_counter_ns()
        if self.l2_cache:
            q = self.embed_fn(task_desc).reshape(-1).astype("float32")
            q = q / (np.linalg.norm(q) + 1e-9)
            best_s, best_i = 0.0, -1
            for i, (_d, e, _b) in enumerate(self.l2_cache):
                s = float(np.dot(q, e))
                if s > best_s:
                    best_s, best_i = s, i
            if best_s >= self.l2_threshold and best_i >= 0:
                hit = self.l2_cache.pop(best_i)
                self.l2_cache.append(hit)
                self.stats["l2_hits"] += 1
                lat = (time.perf_counter_ns() - t_start) / 1000
                print(f"        [⚡ STM-L2 SEMANTIC HIT] score={best_s:.3f} ({lat:.1f}μs)")
                return hit[2], "STM-L2"
        self.stats["misses"] += 1
        lat = (time.perf_counter_ns() - t_start) / 1000
        print(f"        [⏳ STM MISS] semantic-only ({lat:.1f}μs) → cascade body")
        return None, "MISS"


# ══════════════════════════════════════════════════════════════════
#  PCCR COST + UTILITY MODEL
# ══════════════════════════════════════════════════════════════════
# Only EM (episodic) and ENT (entity) are *optional* in this harness: PM/WM
# are always read and STM is always checked first (the cascade head). SM rides
# inside EM's FAISS index, so it is gated together with EM (matching the base
# script's logging). These are relative costs, not measured latencies -- a
# FAISS similarity search + trace deserialization (EM) is far costlier than a
# username dict lookup (ENT). A closed-loop deployment would replace these with
# measured p50 latencies from the RoutingDecision log.
STORE_COST = {
    "episodic": 0.55,
    "entity": 0.15,
}

# Graded expected utility of consulting each optional store, per task pattern
# (the 7 patterns the base script's PatternSTM.classify produces). Calibrated
# so that at consult_threshold = 1.0 (rho_EM = u/0.55 >= 1 => u >= 0.55;
# rho_ENT = u/0.15 >= 1 => u >= 0.15) the consult set matches the base script's
# boolean routing_strategy, except EM is pruned for pure-lookup patterns
# (email_query) where a past episode rarely changes the answer -- a deliberate,
# defensible cost saving that the threshold sweep then generalizes.
PATTERN_UTILITY = {
    "single_cal_create":         {"episodic": 0.60, "entity": 0.10},
    "multi_cal_find_and_create": {"episodic": 0.78, "entity": 0.72},
    "remind_notify":             {"episodic": 0.66, "entity": 0.55},
    "email_query":               {"episodic": 0.48, "entity": 0.66},
    "email_send":                {"episodic": 0.62, "entity": 0.10},
    "cal_query":                 {"episodic": 0.58, "entity": 0.10},
    # Word (3rd-agent) patterns: doc-create benefits from past doc structures
    # (EM) but not user profiles (ENT); doc-query is a pure lookup (consult
    # neither -- cheapest path).
    "word_create":               {"episodic": 0.62, "entity": 0.10},
    "word_query":                {"episodic": 0.45, "entity": 0.10},
    # A10: relation-email is ENT-critical (recipient resolvable only via entity
    # memory). Prior reflects that; the A2 counterfactual measures the true value.
    "relation_email":            {"episodic": 0.10, "entity": 0.90},
    # EM-critical: "as usual" tasks need a past EPISODE to recover the detail.
    "recurring_em":              {"episodic": 0.90, "entity": 0.10},
    "unknown":                   {"episodic": 0.70, "entity": 0.60},
}

OPTIONAL_STORES = ("episodic", "entity")

# Apps we deliberately cannot serve (no agent for them) -> exclude such tasks.
_UNSERVED = ("excel", "xlsx", "spreadsheet", "cell", ".pdf", " pdf", "ocr",
             "image", ".png", ".jpg", "jpeg", "shell", "directory", "rename",
             "folder", "pptx")


def filter_cal_email_word(all_ids, repo="./OfficeBench"):
    """3-agent task pool: tasks runnable with exactly {calendar, email, word}
    (and not requiring excel/pdf/ocr/shell). Returns [(task_id, subtask_id)]."""
    import json as _json
    pool = []
    for tid, si in all_ids:
        try:
            d = _json.load(open(f"{repo}/tasks/{tid}/subtasks/{si}.json"))
        except Exception:
            continue
        txt = d.get("task", "").lower()
        ev = _json.dumps(d.get("evaluation", "")).lower()
        blob = txt + " " + ev
        if any(k in blob for k in _UNSERVED):
            continue
        cal = ".ics" in ev or any(k in txt for k in
              ("calendar", "meeting", "schedule", "event", "dinner", "workout", "appointment", "travelling"))
        email = ".eml" in ev or any(k in txt for k in
                ("email", "mail", "send", "notify", "remind", "compose"))
        word = ".docx" in blob or "word" in blob or "document" in blob or "letter" in blob or "memo" in blob
        if cal or email or word:
            pool.append((tid, si))
    return pool


@dataclass
class RoutingDecision:
    """An auditable record of one PCCR retrieval decision (the closed-loop
    substrate). Mirrors memory_manager/types.RoutingDecision but specialized to
    this harness's two optional stores."""

    task: str
    pattern: str
    mode: str
    consulted: list = field(default_factory=list)
    skipped: dict = field(default_factory=dict)        # store -> reason
    rho: dict = field(default_factory=dict)            # store -> rho value
    cache_hit: bool = False
    cache_layer: str = "MISS"
    latency_us: float = 0.0           # total retrieval-phase time
    decision_us: float = 0.0          # A6: routing DECISION only (the rho gate)
    retrieval_us: float = 0.0         # A6: actual store reads (FAISS/ENT)

    def explain(self) -> str:
        head = f"[PCCR/{self.mode}] pattern='{self.pattern}' cache={self.cache_layer}"
        con = "consult: " + (", ".join(self.consulted) if self.consulted else "(none optional)")
        sk = "".join(f"\n        skip {s}: {r}" for s, r in self.skipped.items())
        return f"{head}\n        {con}{sk}"


# ══════════════════════════════════════════════════════════════════
#  PCCR MEMORY MANAGER  (subclasses the base script's MemoryManager)
# ══════════════════════════════════════════════════════════════════
class PCCRMemoryManager(MemoryManager):
    """Drop-in MemoryManager whose Level-2 routing is the cost-effectiveness
    cascade. execute_task() in the base script calls route_query / ingest_task /
    read_procedural / plan_driven_load / write_working_step / on_task_complete;
    we override only route_query and on_task_complete and inherit the rest, so
    the entire engine runs unchanged."""

    def __init__(self, embed_fn, embed_dim=384, *,
                 routing_mode="pccr", consult_threshold=1.0, utility_lr=0.05,
                 strict_stm=False, enable_word_agent=False, freeze_test=False):
        super().__init__(embed_fn=embed_fn, embed_dim=embed_dim)
        self.freeze_test = freeze_test   # if True, no memory writes during test
        self.freeze_writes = False       # live flag: toggled per phase by the runner
        if strict_stm:
            # Replace the coarse pattern-cache STM with the semantic-only one so
            # held-out tasks miss and the cascade body actually runs.
            self.stm = StrictSemanticSTM(embed_fn=embed_fn)
            print(f"  📋 [STM] REPLACED with StrictSemanticSTM (L2 semantic-only)")
        if enable_word_agent:
            self._register_word_agent()
        self.routing_mode = routing_mode            # "pccr" | "boolean" | "retrieve_all"
        self.consult_threshold = consult_threshold  # the cost/quality dial
        self.utility_lr = utility_lr                # closed-loop nudge size
        # Mutable per-(pattern x store) utility table the closed loop tunes.
        self.pattern_utility = {p: dict(v) for p, v in PATTERN_UTILITY.items()}
        self.store_cost = dict(STORE_COST)
        # ASSUMPTION RECTIFICATION: if calibrated (measured) tables exist on disk,
        # load them so cost/utility are DATA-DERIVED, not hand-set priors. The
        # priors above are only a fallback for an uncalibrated system.
        self.cost_source = "prior (hand-set)"
        self.utility_source = "prior (hand-set)"
        self._load_calibration()
        # Counterfactual override hook: {store: True/False} forces consult/skip
        # (used by calibrate.measure_utilities to run with vs without a store).
        self.force_stores = {}
        self.pccr_decision_log = []
        # Per-task cost accounting for the cost/quality frontier.
        self.cost_log = []
        # Set by route_query each task; read by on_task_complete (the base
        # execute_task result dict does not propagate this field).
        self.last_optional_consults = 0
        print(f"  📋 [PCCR] router mode='{routing_mode}' "
              f"threshold={consult_threshold} (cost dial), lr={utility_lr}")

    # -- load measured cost/utility (rectified, data-derived) --------------
    def _load_calibration(self, path="calibration"):
        cpath, upath = f"{path}/store_cost.json", f"{path}/pattern_utility.json"
        if os.path.exists(cpath):
            with open(cpath) as f:
                self.store_cost.update(json.load(f).get("store_cost", {}))
            self.cost_source = "measured (tokens)"
        if os.path.exists(upath):
            measured = json.load(open(upath)).get("pattern_utility", {})
            for p, stores in measured.items():
                self.pattern_utility.setdefault(p, {}).update(stores)
            self.utility_source = "measured (counterfactual)"
        if self.cost_source.startswith("measured") or self.utility_source.startswith("measured"):
            print(f"  📐 [CALIBRATION] cost={self.cost_source}  utility={self.utility_source}")

    # -- 3rd agent: WORD ---------------------------------------------------
    def _register_word_agent(self):
        """Add the Word specialist as a third sub-agent: give it a PM prompt and
        make the orchestrator aware it can delegate document work. Execution is
        handled by LocalOfficeBenchRunner._run_word (base script untouched)."""
        self.procedural["agent_prompts"]["word"] = (
            "You are a WORD specialist. Output ONLY a Python dict. No markdown. No backticks.\n\n"
            "ACTIONS (use EXACTLY these formats):\n"
            '{"app": "word", "action": "create_new_file", "file_path": "data/report.docx"}\n'
            '{"app": "word", "action": "write_to_file", "file_path": "data/report.docx", "contents": "full document text"}\n'
            '{"app": "word", "action": "read_file", "file_path": "data/report.docx"}\n\n'
            "RULES:\n"
            "- To create a document: FIRST create_new_file, THEN write_to_file with the content.\n"
            "- file_path is relative (usually under data/); keep the filename given in the subtask.\n"
            "- Put the ENTIRE document text in 'contents'.\n"
            "- Output ONE action only.")
        self.procedural["orchestrator_prompt"] += (
            "\n\nADDITIONAL AGENT:\n"
            "- word: create_new_file, write_to_file, read_file (Word .docx documents)\n"
            "When a task involves writing, drafting, or reading a document/report/letter, "
            "delegate to 'word'. For 'write a doc and email it' style tasks, use 'word' "
            "to create the document, then 'email' to send it.")
        print(f"  📋 [PM] registered WORD agent (3-agent mode)")

    # -- the rho = utility/cost gate ---------------------------------------
    def _pccr_gate(self, pattern):
        """Return {store: (consult_bool, rho, utility, cost)} for each optional
        store under the current (possibly closed-loop-tuned) utility table."""
        util = self.pattern_utility.get(pattern, self.pattern_utility["unknown"])
        gate = {}
        for store in OPTIONAL_STORES:
            u = util.get(store, 0.5)
            c = self.store_cost.get(store, 1.0)
            rho = (u / c) if c > 0 else float("inf")
            gate[store] = (rho >= self.consult_threshold, rho, u, c)
        return gate

    def adjust_utility(self, pattern, store, delta, lo=0.0, hi=1.0):
        """Closed-loop hook: nudge one (pattern x store) utility toward what
        outcomes suggest, clamped to [lo, hi]. Called from on_task_complete."""
        tbl = self.pattern_utility.setdefault(
            pattern, dict(PATTERN_UTILITY.get(pattern, PATTERN_UTILITY["unknown"])))
        cur = tbl.get(store, 0.5)
        tbl[store] = min(hi, max(lo, cur + delta))

    # ═══ PHASE 3: MEMORY RETRIEVAL  (PCCR cascade) ═══
    def route_query(self, task_desc, use_memory=True):
        """Cascade:
          L1  always read PM + WM.
          head  check STM first; a confident hit short-circuits EM/ENT.
          body  on a miss, consult an optional store iff rho >= threshold
                (mode='pccr'); 'boolean' uses the base table; 'retrieve_all'
                consults everything.
        Returns the same dict shape the base execute_task() expects."""
        t0 = time.perf_counter_ns()
        self.last_optional_consults = 0
        result = {"procedural": None, "working": None, "stm_hit": None, "stm_layer": "MISS",
                  "episodic_orch": [], "agent_memories": {}, "entities": {},
                  "faiss_queries": 0, "routing_log": [], "stores_consulted": 0}

        # ── L1: PM (always) ──
        self._log("PM", "READ", "orchestrator prompt (L1: always)")
        result["procedural"] = self.procedural["orchestrator_prompt"]
        if self.procedural.get("learned_rules"):
            result["procedural"] += "\n\nLEARNED FROM EXPERIENCE:\n"
            for r in self.procedural["learned_rules"]:
                result["procedural"] += f"  - {r['pattern']}: {r.get('lesson','')}\n"
            self._log("PM", "READ", f"{len(self.procedural['learned_rules'])} learned rules appended")
        result["routing_log"].append("L1: PM READ")

        # ── L1: WM (always) ──
        self._log("WM", "READ", "current task context (L1: always)")
        result["working"] = self.working_memory.copy()
        result["routing_log"].append("L1: WM READ")

        if not use_memory:
            result["routing_log"].append("L1: Memory OFF")
            return result

        pattern = self.working_memory.get("pattern", "unknown")

        # ── Cascade head: STM first ──
        self._log("STM", "READ", "pattern classifier + semantic cache (L1: always)")
        cached, layer = self.stm.lookup(task_desc)
        self.working_memory["memories_read_this_task"].add("stm")
        result["routing_log"].append(f"L1: STM CHECK → {layer}")

        if cached is not None:
            result["stm_hit"] = cached
            result["stm_layer"] = layer
            result["agent_memories"] = cached.get("agent_memories", {})
            confidence = min(cached.get("success_count", 1) / 5.0, 1.0)
            if confidence >= 0.6:
                # Cascade short-circuit: skip ALL optional stores.
                dec = RoutingDecision(
                    task=task_desc[:40], pattern=pattern, mode=self.routing_mode,
                    consulted=["PM", "WM", "STM"], cache_hit=True, cache_layer=layer,
                    skipped={s: "stm_cache_hit: cascade short-circuited" for s in OPTIONAL_STORES},
                    latency_us=(time.perf_counter_ns() - t0) / 1000)
                self.pccr_decision_log.append(dec)
                print(f"      🧭 {dec.explain()}")
                result["routing_log"].append(
                    f"L3: STM confidence {confidence:.0%} → TRUST, cascade short-circuit")
                self.last_optional_consults = 0   # short-circuit: nothing optional consulted
                return result
            result["routing_log"].append(
                f"L3: STM confidence {confidence:.0%} → HINT only, run cost-gate")

        # ── Cascade body: decide consult set by mode ──
        t_decide = time.perf_counter_ns()                  # A6: time the DECISION only
        gate = self._pccr_gate(pattern)
        consult = {}
        skipped = {}
        rho_map = {}
        if self.routing_mode == "retrieve_all":
            consult = {s: True for s in OPTIONAL_STORES}
            result["routing_log"].append("L2: mode=retrieve_all → consult EM+ENT")
        elif self.routing_mode == "boolean":
            strat = self.procedural["routing_strategy"].get(
                pattern, self.procedural["routing_strategy"]["unknown"])
            consult = {"episodic": strat["episodic"], "entity": strat["entity"]}
            for s in OPTIONAL_STORES:
                if not consult[s]:
                    skipped[s] = f"boolean policy: pattern '{pattern}' → {s}=False"
            result["routing_log"].append(
                f"L2: mode=boolean pattern='{pattern}' → EM={consult['episodic']} ENT={consult['entity']}")
        elif self.routing_mode == "similarity":
            # A13 baseline: consult a store iff its top retrieved item clears a
            # similarity threshold (no cost, no pattern) — the common RAG-style policy.
            sim_thr = getattr(self, "similarity_threshold_baseline", 0.5)
            for s in OPTIONAL_STORES:
                if s == "episodic":
                    raw = self.episodic.retrieve_for_orchestrator(self.working_memory["current_task"], k=1)
                    best = 0.0
                    if raw and raw[0].embedding:
                        qe = self.episodic.embed_fn(self.working_memory["current_task"]).reshape(-1).astype("float32")
                        qe = qe / (np.linalg.norm(qe) + 1e-9)
                        me = np.array(raw[0].embedding).reshape(-1).astype("float32")
                        me = me / (np.linalg.norm(me) + 1e-9)
                        best = float(np.dot(qe, me))
                    consult[s] = best >= sim_thr
                    if not consult[s]:
                        skipped[s] = f"similarity {best:.2f} < {sim_thr}"
                else:  # entity: consult iff the task names a known user
                    consult[s] = bool(self._extract_entities(self.working_memory["current_task"]))
                    if not consult[s]:
                        skipped[s] = "similarity baseline: no known user"
            result["routing_log"].append(f"L2: mode=similarity thr={sim_thr}")
        else:  # pccr
            for s in OPTIONAL_STORES:
                ok, rho, u, c = gate[s]
                rho_map[s] = round(rho, 2)
                consult[s] = ok
                if not ok:
                    skipped[s] = (f"rho {rho:.2f} < threshold {self.consult_threshold:.2f} "
                                  f"(u={u:.2f}, cost={c:.2f})")
            result["routing_log"].append(
                f"L2: mode=pccr pattern='{pattern}' rho={rho_map} thr={self.consult_threshold}")
        decision_us = (time.perf_counter_ns() - t_decide) / 1000   # A6: decision-only latency
        t_retrieve = time.perf_counter_ns()

        # ── Counterfactual override (for utility calibration) ──
        # Force a store on/off regardless of the gate, so measure_utilities can
        # run the SAME task with vs without a store and measure the accuracy gain.
        if self.force_stores:
            for s, v in self.force_stores.items():
                if v is not None:
                    consult[s] = v
                    skipped.pop(s, None) if v else skipped.setdefault(s, "counterfactual: forced off")

        # ── EM read (gated) ── (SM rides inside EM's FAISS)
        if consult.get("episodic"):
            self._log("EM", "READ", "past experiences (FAISS)")
            self._log("SM", "READ", "cosine search (internal)")
            raw = self.episodic.retrieve_for_orchestrator(task_desc, k=3)
            result["faiss_queries"] += 1
            result["stores_consulted"] += 1
            self.working_memory["memories_read_this_task"].add("episodic")
            relevant = []
            for mem in raw:
                if mem.embedding:
                    qe = self.episodic.embed_fn(task_desc).reshape(-1).astype('float32')
                    qe = qe / (np.linalg.norm(qe) + 1e-9)
                    me = np.array(mem.embedding).reshape(-1).astype('float32')
                    sim = float(np.dot(qe, me))
                    if sim >= 0.5:
                        relevant.append(mem)
                    else:
                        self._log("EM", "FILTER", f"'{mem.task_description[:30]}' sim={sim:.2f}<0.5")
                else:
                    relevant.append(mem)
            result["episodic_orch"] = relevant
            for fm in relevant:
                for st in fm.subtask_memories:
                    if st.agent_type in ["calendar", "email", "word"]:
                        result["agent_memories"].setdefault(st.agent_type, []).append(st)
            for a in result["agent_memories"]:
                result["agent_memories"][a] = result["agent_memories"][a][:3]
            result["routing_log"].append(f"EM READ → {len(relevant)}/{len(raw)} passed filter")
        else:
            self._log("EM", "SKIP", skipped.get("episodic", "not consulted"))
            self._log("SM", "SKIP", "no FAISS needed")
            result["routing_log"].append("EM SKIP")

        # ── ENT read (gated) ──
        users_found = self._extract_entities(task_desc)
        if consult.get("entity") and users_found:
            for user in users_found:
                self._log("ENT", "READ", f"profile for '{user}'")
                result["entities"][user] = self.entity["users"].get(user, {})
            result["stores_consulted"] += 1
            self.working_memory["memories_read_this_task"].add("entity")
            result["routing_log"].append(f"ENT READ for {users_found}")
        else:
            reason = skipped.get("entity", "no known users in task" if not users_found else "not consulted")
            self._log("ENT", "SKIP", reason)
            result["routing_log"].append("ENT SKIP")

        # ── Emit the structured decision record ──
        dec = RoutingDecision(
            task=task_desc[:40], pattern=pattern, mode=self.routing_mode,
            consulted=["PM", "WM", "STM"] + [s for s in OPTIONAL_STORES if consult.get(s)],
            skipped=skipped, rho=rho_map, cache_hit=result["stm_hit"] is not None,
            cache_layer=result["stm_layer"], latency_us=(time.perf_counter_ns() - t0) / 1000,
            decision_us=decision_us,                                  # A6: pure routing-decision time
            retrieval_us=(time.perf_counter_ns() - t_retrieve) / 1000)  # A6: store-read time
        self.pccr_decision_log.append(dec)
        print(f"      🧭 {dec.explain()}")
        self.last_optional_consults = result["stores_consulted"]
        return result

    # ═══ PHASE 9: MEMORY STORAGE + closed-loop utility update ═══
    def on_task_complete(self, task_desc, success, exec_result, steps):
        # Capture what the closed loop needs BEFORE super() clears working memory.
        pattern, _ = self.stm.classify(task_desc)
        mems_read = set(self.working_memory.get("memories_read_this_task", set()))
        faiss_q = exec_result.get("faiss_queries", 0)
        stores_consulted = self.last_optional_consults   # set by route_query this task
        stm_layer = exec_result.get("stm_layer_hit", "MISS")

        cost_row = {
            "task": task_desc[:60], "pattern": pattern, "success": success,
            "mode": self.routing_mode, "threshold": self.consult_threshold,
            "faiss_queries": faiss_q, "optional_stores_consulted": stores_consulted,
            "stm_layer": stm_layer, "steps": len(steps),
        }

        if self.freeze_writes:
            # FROZEN test memory: record cost only, write NOTHING (no STM/ENT,
            # no routing_history, no utility update), then clear working memory
            # so every test task faces the identical consolidated memory.
            self.cost_log.append(cost_row)
            self.working_memory = {"current_task": None, "current_user": None, "current_date": None,
                                   "pattern": None, "step_history": [], "agent_scratchpad": [],
                                   "shared_context": {}, "memories_read_this_task": set()}
            return

        super().on_task_complete(task_desc, success, exec_result, steps)

        # Closed loop (pccr mode only): nudge consulted stores' utility by outcome.
        if self.routing_mode == "pccr":
            for store in OPTIONAL_STORES:
                if store in mems_read:
                    delta = self.utility_lr if success else -self.utility_lr
                    before = self.pattern_utility.get(pattern, {}).get(store)
                    self.adjust_utility(pattern, store, delta)
                    after = self.pattern_utility.get(pattern, {}).get(store)
                    if before is not None and before != after:
                        sign = "↑" if delta > 0 else "↓"
                        print(f"      🔁 [PCCR adjust] '{pattern}'.{store} "
                              f"{before:.2f}{sign}{after:.2f} (success={success})")

        # Record per-task cost row for the frontier.
        self.cost_log.append(cost_row)

    # -- reporting ---------------------------------------------------------
    def pccr_summary(self):
        total = len(self.cost_log)
        if not total:
            return "no tasks run"
        faiss = sum(r["faiss_queries"] for r in self.cost_log)
        consults = sum(r["optional_stores_consulted"] for r in self.cost_log)
        hits = sum(1 for r in self.cost_log if str(r["stm_layer"]).startswith("STM"))
        succ = sum(1 for r in self.cost_log if r["success"])
        return (f"mode={self.routing_mode} thr={self.consult_threshold} | "
                f"tasks={total} success={succ}/{total} | "
                f"FAISS={faiss} optional-consults={consults} | "
                f"STM-hits={hits} | avg-consults/task={consults/total:.2f}")


# ══════════════════════════════════════════════════════════════════
#  MAIN  (mirrors the base script's train→consolidate→test loop)
# ══════════════════════════════════════════════════════════════════
def _train_and_consolidate(mem_mgr, llm, runner, train_tasks, results):
    """Phase 2-10: run training (memory off) then consolidate once. Returns the
    successful trajectories. Memory built here (EM/SM/PM/STM) is read-only during
    the test phase, so it can be reused across routing modes."""
    print(f"\n{'━'*70}\n  TRAINING: {len(train_tasks)} tasks (no memory)\n{'━'*70}")
    mem_mgr.freeze_writes = False   # training MUST write to build memory
    successful = []
    for tid, si in train_tasks:
        try:
            cfg = runner.setup_task(tid, si)
            level = runner.get_level(tid)
            LOCAL_TESTBED[0] = str(runner.testbed)
            print(f"\n  Train {tid}_{si} (L{level}): {cfg['task'][:55]}...")
            twd = cfg["task"] + (f"\n(Today's date: {cfg['date']})" if cfg.get("date") else "")
            t0 = time.time()
            ex = execute_task(twd, cfg["username"], llm, runner, mem_mgr,
                              use_memory=False, task_date=cfg.get("date"))
            ev = runner.evaluate_task(tid, si)
            print(f"    Eval: {'✅' if ev['success'] else '❌'} ({ev['passed']}/{ev['total']}) in {time.time()-t0:.1f}s")
            results["training"]["total"] += 1
            results["training"]["passed"] += int(ev["success"])
            if ev["success"]:
                successful.append({"task_description": cfg["task"], "success": True,
                                   "steps": ex["steps"], "agents_used": ex["agents_used"], "level": level})
            mem_mgr.on_task_complete(cfg["task"], ev["success"], ex, ex["steps"])
        except Exception as e:
            print(f"    ⚠️ {str(e)[:120]}")
        time.sleep(1)

    print(f"\n  Training: {results['training']['passed']}/{results['training']['total']}")
    if successful:
        print(f"  Building LTM from {len(successful)} trajectories...")
        mem_mgr.write_episodic(successful, llm)
        mem_mgr.save()
        for traj in successful:
            td = traj.get("task_description", "")
            bundle = {"plan": " | ".join(f"[{s.get('agent','?')}] {s.get('subtask','')[:50]}"
                                          for s in traj.get("steps", [])),
                      "agent_memories": {}, "agents_used": traj.get("agents_used", []), "final_answer": ""}
            mem_mgr.stm.store_bundle(td, bundle, steps=traj.get("steps", []))
        # A7: distill learned RULES into PM (was wired but empty). Per task
        # pattern, take the most common plan signature among successes → a rule
        # the orchestrator prompt injects ("LEARNED FROM EXPERIENCE").
        from collections import Counter, defaultdict
        by_pat = defaultdict(list)
        for traj in successful:
            pat, _ = mem_mgr.stm.classify(traj.get("task_description", ""))
            sig = " → ".join(s.get("agent", "?") for s in traj.get("steps", []))
            if sig:
                by_pat[pat].append(sig)
        for pat, sigs in by_pat.items():
            if len(sigs) >= 2:                      # only rules with support
                common = Counter(sigs).most_common(1)[0][0]
                mem_mgr.procedural["learned_rules"].append(
                    {"pattern": pat, "lesson": f"prefer plan: {common}", "support": len(sigs)})
        if mem_mgr.procedural["learned_rules"]:
            print(f"  📘 [PM] distilled {len(mem_mgr.procedural['learned_rules'])} learned rules")
        print(f"  Memory: {mem_mgr.summary()}")
    return successful


def _test_loop(mem_mgr, llm, runner, test_tasks, results):
    """Phase 2-9 under the manager's current routing mode. The 'no_memory' mode
    runs with memory OFF (consult nothing) — the baseline that tests whether
    memory is load-bearing at all on these tasks."""
    use_mem = (mem_mgr.routing_mode != "no_memory")
    for tid, si in test_tasks:
        try:
            cfg = runner.setup_task(tid, si)
            level = runner.get_level(tid)
            LOCAL_TESTBED[0] = str(runner.testbed)
            print(f"\n  Test {tid}_{si} (L{level}): {cfg['task'][:55]}...")
            twd = cfg["task"] + (f"\n(Today's date: {cfg['date']})" if cfg.get("date") else "")
            t0 = time.time()
            ex = execute_task(twd, cfg["username"], llm, runner, mem_mgr,
                              use_memory=use_mem, task_date=cfg.get("date"))
            ev = runner.evaluate_task(tid, si)
            print(f"    Eval: {'✅' if ev['success'] else '❌'} ({ev['passed']}/{ev['total']}) "
                  f"STM={ex.get('stm_layer_hit','?')} FAISS={ex.get('faiss_queries',0)} in {time.time()-t0:.1f}s")
            results["testing"]["total"] += 1
            results["testing"]["passed"] += int(ev["success"])
            mem_mgr.on_task_complete(cfg["task"], ev["success"], ex, ex["steps"])
        except Exception as e:
            print(f"    ⚠️ {str(e)[:120]}")
        time.sleep(1)
    return results


def run_suite(mem_mgr, llm, runner, train_tasks, test_tasks, results):
    _train_and_consolidate(mem_mgr, llm, runner, train_tasks, results)
    mem_mgr.freeze_writes = mem_mgr.freeze_test
    frozen = " [FROZEN: no writes]" if mem_mgr.freeze_writes else ""
    print(f"\n{'━'*70}\n  TESTING: {len(test_tasks)} tasks (WITH PCCR memory){frozen}\n{'━'*70}")
    _test_loop(mem_mgr, llm, runner, test_tasks, results)
    return results


# -- warm resume: reload the consolidated bank from disk (skip training) -----
def _reload_bank(mem_mgr, path="./legomem_bank_mm"):
    """Reload episodic/semantic memory (the product of training+consolidation)
    from disk so we can skip re-running the 60 training tasks after a crash."""
    import faiss as _faiss
    from mm_on_top_of_legomem import FullTaskMemory, SubtaskMemory
    bank = mem_mgr.episodic
    with open(f"{path}/bank.json") as f:
        data = json.load(f)
    bank.full_task_memories = []
    for m in data.get("full_task", []):
        subs = [SubtaskMemory(**s) for s in m.get("subtask_memories", [])]
        md = {k: v for k, v in m.items() if k != "subtask_memories"}
        fm = FullTaskMemory(**md)
        fm.subtask_memories = subs
        bank.full_task_memories.append(fm)
    bank.full_task_index = _faiss.read_index(f"{path}/full_task.faiss")
    bank.agent_memories, bank.agent_indexes = {}, {}
    for a, ms in data.get("agent_memories", {}).items():
        bank.agent_memories[a] = [SubtaskMemory(**s) for s in ms]
        bank.agent_indexes[a] = _faiss.read_index(f"{path}/subtask_{a}.faiss")
    return len(bank.full_task_memories)


def _rebuild_stm_from_bank(mem_mgr):
    """Re-derive the STM prefill bundles from the reloaded episodes (no LLM)."""
    for fm in mem_mgr.episodic.full_task_memories:
        steps = [{"agent": s.agent_type, "subtask": s.subtask_description}
                 for s in fm.subtask_memories]
        bundle = {"plan": fm.high_level_plan or " | ".join(
                      f"[{s['agent']}] {s['subtask'][:50]}" for s in steps),
                  "agent_memories": {}, "agents_used": sorted({s["agent"] for s in steps}),
                  "final_answer": fm.final_answer}
        mem_mgr.stm.store_bundle(fm.task_description, bundle, steps=steps)


# -- train-once / test-per-mode optimization --------------------------------
def _snapshot_memory(mem_mgr):
    """Deep-copy exactly the state that the TEST phase mutates (STM cache, ENT
    profiles, routing history, utility table). EM/SM/PM are read-only during
    test, so they are shared across modes without copying."""
    stm = mem_mgr.stm
    return {
        "pattern_cache": copy.deepcopy(stm.pattern_cache),
        "step_cache": copy.deepcopy(stm.step_cache),
        "step_pattern_map": copy.deepcopy(stm.step_pattern_map),
        "l2_cache": copy.deepcopy(stm.l2_cache),
        "stm_stats": copy.deepcopy(stm.stats),
        "entity": copy.deepcopy(mem_mgr.entity),
        "routing_history": copy.deepcopy(mem_mgr.routing_history),
        "pattern_utility": copy.deepcopy(mem_mgr.pattern_utility),
    }


def _restore_memory(mem_mgr, snap):
    stm = mem_mgr.stm
    stm.pattern_cache = copy.deepcopy(snap["pattern_cache"])
    stm.step_cache = copy.deepcopy(snap["step_cache"])
    stm.step_pattern_map = copy.deepcopy(snap["step_pattern_map"])
    stm.l2_cache = copy.deepcopy(snap["l2_cache"])
    stm.stats = copy.deepcopy(snap["stm_stats"])
    mem_mgr.entity = copy.deepcopy(snap["entity"])
    mem_mgr.routing_history = copy.deepcopy(snap["routing_history"])
    mem_mgr.pattern_utility = copy.deepcopy(snap["pattern_utility"])
    mem_mgr.last_optional_consults = 0
    mem_mgr.memory_ops = {"reads": 0, "writes": 0}
    mem_mgr.current_task_accesses = []
    mem_mgr.cost_log = []
    mem_mgr.pccr_decision_log = []


def _save_mode_json(mem_mgr, mode, threshold, n_agents, results):
    out = {
        "config": {"mode": mode, "threshold": threshold, "n_agents": n_agents,
                   "strict_stm": True, "heldout": True, "multimode": True},
        "results": results, "pccr_summary": mem_mgr.pccr_summary(),
        "cost_log": mem_mgr.cost_log,
        "decision_log": [vars(d) for d in mem_mgr.pccr_decision_log],
        "final_utility_table": mem_mgr.pattern_utility,
    }
    path = f"{LOG_DIR}/pccr_{mode}_thr{threshold}_{int(time.time())}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved → {path}")


def run_multimode(mem_mgr, llm, runner, train_tasks, test_tasks, configs, n_agents,
                  pretrained=False):
    """Train + consolidate ONCE (unless pretrained=True, i.e. bank reloaded from
    disk), then run each (mode, threshold) config's test phase from the same
    consolidated memory (snapshot/restore). Per-config resume markers let a
    crashed run skip configs that already completed."""
    base = {"training": {"passed": 0, "total": 0}}
    if not pretrained:
        _train_and_consolidate(mem_mgr, llm, runner, train_tasks, base)
    snap = _snapshot_memory(mem_mgr)
    mem_mgr.freeze_writes = mem_mgr.freeze_test   # freeze for all test phases
    frozen = " [FROZEN: no writes during test]" if mem_mgr.freeze_writes else ""
    print(f"\n  📸 Snapshotted consolidated memory; running {len(configs)} configs on it.{frozen}")
    for mode, thr in configs:
        marker = f"{LOG_DIR}/.done_{mode}_thr{thr}_a{n_agents}"
        if os.path.exists(marker):
            print(f"  ⏭️  skip mode='{mode}' thr={thr} (already done — resume marker found)")
            continue
        _restore_memory(mem_mgr, snap)
        mem_mgr.routing_mode = mode
        mem_mgr.consult_threshold = thr
        print(f"\n{'━'*70}\n  TESTING mode='{mode}' thr={thr}: {len(test_tasks)} tasks\n{'━'*70}")
        results = {"mode": mode, "threshold": thr, "n_agents": n_agents,
                   "strict_stm": True, "heldout": True,
                   "training": dict(base["training"]), "testing": {"passed": 0, "total": 0}}
        _test_loop(mem_mgr, llm, runner, test_tasks, results)
        print(f"  {mem_mgr.pccr_summary()}")
        _save_mode_json(mem_mgr, mode, thr, n_agents, results)
        with open(marker, "w") as f:
            f.write(mem_mgr.pccr_summary())   # mark this config complete for resume


def main():
    mode = os.environ.get("PCCR_MODE", "pccr")            # pccr | boolean | retrieve_all
    threshold = float(os.environ.get("PCCR_THRESHOLD", "1.0"))
    pool_n = int(os.environ.get("PCCR_NTASKS", "34"))     # cap on pool size
    strict_stm = os.environ.get("PCCR_STRICT_STM", "1") == "1"
    heldout = os.environ.get("PCCR_HELDOUT", "1") == "1"
    split_frac = float(os.environ.get("PCCR_SPLIT", "0.7"))
    n_agents = int(os.environ.get("PCCR_AGENTS", "2"))    # 2 = cal+email, 3 = +word
    train_n = int(os.environ.get("PCCR_TRAIN_N", "0"))    # absolute train count (0 = use split_frac)
    freeze_test = os.environ.get("PCCR_FREEZE_TEST", "0") == "1"  # frozen test memory

    print("=" * 70)
    print("  PCCR Router  on top of  LEGOMem + MemoryManager + OfficeBench")
    print(f"  routing_mode={mode}  threshold={threshold}  agents={n_agents}")
    print(f"  strict_stm={strict_stm}  heldout={heldout}  split={split_frac} train_n={train_n}")
    print("=" * 70)

    embedder = LocalEmbedder()
    keys = [os.environ.get(f"CEREBRAS_KEY_{i}", "") for i in range(1, 11)]
    llm = MultiLLM(keys, model=os.environ.get("PCCR_MODEL", "gpt-oss-120b"))
    # Add a socket timeout so a dead connection (e.g. after the Mac sleeps)
    # fails fast and the key round-robin recovers, instead of hanging forever.
    for c in llm.clients:
        try:
            c["client"] = c["client"].with_options(timeout=60.0)
        except Exception:
            pass
    runner = LocalOfficeBenchRunner(REPO_PATH)
    resume = os.environ.get("PCCR_RESUME", "0") == "1"

    ent_critical = os.environ.get("PCCR_ENT_CRITICAL", "0") == "1"   # A10
    em_critical = os.environ.get("PCCR_EM_CRITICAL", "0") == "1"     # EM analog
    enable_word = (n_agents >= 3)
    if ent_critical or em_critical:
        extra = []
        if ent_critical:
            import synthetic_ent_tasks as set_mod
            set_mod.generate(); extra += set_mod.filter_ent_tasks(runner.get_all_task_ids())
        if em_critical:
            import synthetic_em_tasks as sem_mod
            sem_mod.generate(); extra += sem_mod.filter_em_tasks(runner.get_all_task_ids())
        if os.environ.get("PCCR_MEMCRIT_ONLY", "0") == "1":
            pool = list(dict.fromkeys(extra))      # ONLY the memory-critical tasks
        else:
            base = (filter_cal_email_word(runner.get_all_task_ids(), REPO_PATH) if enable_word
                    else filter_cal_email(runner.get_all_task_ids(), REPO_PATH))
            pool = list(dict.fromkeys(base + extra))   # dedupe (synthetic also match filters)
    elif enable_word:
        pool = filter_cal_email_word(runner.get_all_task_ids(), REPO_PATH)
    else:
        pool = filter_cal_email(runner.get_all_task_ids(), REPO_PATH)
    pool = pool[:pool_n] if pool_n and pool_n < len(pool) else pool

    mem_mgr = PCCRMemoryManager(embed_fn=embedder.embed, embed_dim=embedder.dims,
                                routing_mode=mode, consult_threshold=threshold,
                                strict_stm=strict_stm, enable_word_agent=enable_word,
                                freeze_test=freeze_test)
    if ent_critical:
        import synthetic_ent_tasks as set_mod
        set_mod.seed_entity(mem_mgr)          # A10: relations live ONLY in ENT
    # A5: a seed shuffles the pool before the split -> different train/test
    # partitions across seeds, so we can report mean ± std (variance), not a
    # single anecdotal run.
    seed = os.environ.get("PCCR_SEED", "")
    if seed:
        import random as _random
        _random.Random(int(seed)).shuffle(pool)
        print(f"  A5: shuffled pool with seed={seed}")
    print(f"  task pool: {len(pool)} ({'cal+email+word' if enable_word else 'cal+email'})  "
          f"freeze_test={freeze_test}")

    if heldout:
        split = train_n if train_n > 0 else max(1, int(len(pool) * split_frac))
        split = min(split, len(pool) - 1)
        train_tasks = pool[:split]
        test_tasks = pool[split:]                    # DISJOINT, unseen at test
        print(f"  held-out split: {len(train_tasks)} train / {len(test_tasks)} test (disjoint)")
    else:
        train_tasks = test_tasks = pool
        print(f"  identical set: {len(train_tasks)} train == test")

    # Train once, test many configs. Either a multi-mode comparison (PCCR_MODES,
    # fixed threshold) or a pccr threshold sweep (PCCR_THRESHOLDS).
    modes_env = os.environ.get("PCCR_MODES", "").strip()
    sweep_env = os.environ.get("PCCR_THRESHOLDS", "").strip()
    configs = None
    if sweep_env:
        configs = [("pccr", float(t)) for t in sweep_env.split(",") if t.strip()]
        print(f"  THRESHOLD SWEEP: train once → test pccr at {[c[1] for c in configs]}")
    elif modes_env:
        modes = [m.strip() for m in modes_env.split(",") if m.strip()]
        configs = [(m, threshold) for m in modes]
        print(f"  MULTIMODE: train once → test modes {modes}")
    if configs:
        pretrained = False
        if resume and os.path.exists("legomem_bank_mm/bank.json"):
            n = _reload_bank(mem_mgr)
            _rebuild_stm_from_bank(mem_mgr)
            pretrained = True
            print(f"  ♻️  RESUMED: reloaded {n} episodes from disk + rebuilt STM "
                  f"(training skipped). ENT/history start fresh.")
        # Seed EM episodes AFTER any reload (reload wipes the bank).
        if em_critical:
            import synthetic_em_tasks as sem_mod
            sem_mod.seed_episodes(mem_mgr)
        run_multimode(mem_mgr, llm, runner, train_tasks, test_tasks, configs, n_agents,
                      pretrained=pretrained)
        print(f"\n{'━'*70}\n  Complete — per-config JSONs saved.\n{'━'*70}")
        return

    results = {"mode": mode, "threshold": threshold, "n_agents": n_agents,
               "strict_stm": strict_stm, "heldout": heldout,
               "training": {"passed": 0, "total": 0},
               "testing": {"passed": 0, "total": 0}}
    run_suite(mem_mgr, llm, runner, train_tasks, test_tasks, results)

    # ── Report (the rows you graph for the cost/quality frontier) ──
    print(f"\n{'━'*70}\n  PCCR RESULTS\n{'━'*70}")
    tr, te = results["training"], results["testing"]
    print(f"  Train (no memory):  {tr['passed']}/{tr['total']}")
    print(f"  Test  (PCCR mem):   {te['passed']}/{te['total']}")
    print(f"  {mem_mgr.pccr_summary()}")
    print(f"  API calls: {llm.total_calls}")

    out = {
        "config": {"mode": mode, "threshold": threshold, "pool_n": pool_n,
                   "strict_stm": strict_stm, "heldout": heldout, "split_frac": split_frac},
        "results": results,
        "pccr_summary": mem_mgr.pccr_summary(),
        "cost_log": mem_mgr.cost_log,
        "decision_log": [vars(d) for d in mem_mgr.pccr_decision_log],
        "final_utility_table": mem_mgr.pattern_utility,   # shows closed-loop drift
    }
    out_path = f"{LOG_DIR}/pccr_{mode}_thr{threshold}_{int(time.time())}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    # NB: we do NOT close sys.stdout here. The base script wraps stdout in a
    # line-buffered TerminalTee (so logs are already flushed); closing it would
    # trigger a harmless "I/O operation on closed file" during interpreter
    # shutdown when the final flush hits the closed handle.
    main()
