"""Wire the REAL memory_manager architecture into OfficeBench (swap A).

Replaces officebench_eval's numpy ProcedureMemory + custom gate.py with the
ACTUAL components:
  - memory_manager.stores.episodic.EpisodicStore  (FAISS IndexFlatIP, our real EM)
  - memory_manager.router.MemoryRouter             (router.py, the real rho-gate; UNCHANGED)

EM is one FAISS store with two retrieval paths (as in the real architecture):
  em.search()             -> orchestrator-level (full past task plans)
  em.preload_for_agents() -> agent-level (per-app subtask memories)
The rho-gate gates EM once per task; on a HIT both paths run.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from memory_manager.embeddings import build_embedder
from memory_manager.manager import MemoryManager
from memory_manager.router import MemoryRouter
from memory_manager.stores.episodic import EpisodicStore
from memory_manager.stores.procedural import ProceduralStore
from memory_manager.types import FullTaskMemory, MemoryType, Pattern, SubtaskMemory

# 1:1 map OfficeBench's 5 patterns -> the router's Pattern enum (enum names are
# just dict keys to the router; we repurpose RECURRING/EXPLORATORY as keys).
PAT_MAP = {
    "lookup":        Pattern.LOOKUP,
    "single_action": Pattern.SINGLE_ACTION,
    "data_compute":  Pattern.RECURRING,
    "doc_process":   Pattern.EXPLORATORY,
    "multi_app":     Pattern.COORDINATION,
}
APPS = ["calendar", "email", "word", "excel", "pdf", "ocr", "shell", "search", "system"]


def _app_of(a):
    try:
        return (json.loads(a).get("app") or "").lower()
    except Exception:
        return ""


def _to_full(rec, i):
    if rec.get("curated_subtasks"):                       # LegoMem-curated agent memory
        sm = [SubtaskMemory(agent=str(st.get("agent", "agent")).replace("_agent", "").lower(),
                            subtask=str(st.get("description", ""))[:120],
                            action=str(st.get("steps", ""))[:400],
                            observation_summary=str(st.get("observations", ""))[:200],
                            outcome="success")
              for st in rec["curated_subtasks"]]
        sig = " -> ".join(str(st.get("agent", "")).replace("_agent", "") for st in rec["curated_subtasks"])
    else:                                                  # raw fallback
        sm = [SubtaskMemory(agent=_app_of(s["action"]) or "agent", subtask=rec["task"][:80],
                            action=s["action"], observation_summary=(s.get("obs") or "")[:200],
                            outcome="success")
              for s in rec["steps"] if _app_of(s["action"]) not in ("system", "")]
        sig = " -> ".join(_app_of(s["action"]) for s in rec["steps"]
                          if _app_of(s["action"]) not in ("system", ""))
    return FullTaskMemory(task_id=f"t{i}", description=rec["task"],
                          pattern=PAT_MAP.get(rec.get("pattern"), Pattern.COORDINATION),
                          plan=rec["plan"], step_signature=sig, subtask_memories=sm,
                          outcome="success")


class RealArch:
    def __init__(self, bank_path, cost, util, theta=1.0):
        self.embedder = build_embedder("sentence-transformers")
        self.em = EpisodicStore(self.embedder, Path(tempfile.mkdtemp(prefix="ob_em_")))
        recs = json.load(open(bank_path))
        fts = []
        for i, rec in enumerate(recs):
            ft = _to_full(rec, i); self.em.add(ft); fts.append(ft)   # -> FAISS index
        # PM: distil generalised procedural RULES from the bank (Phase-10 consolidation
        # logic, frozen) -> ProceduralStore.learned_rules. PM is always-read in the gate.
        self.pm = ProceduralStore()
        self._pm_rules = self._distil_pm(fts, recs)
        # configure the REAL router for OfficeBench (EM gate only; SM/ENT off here).
        # If a calibrated per-pattern utility is supplied we use it; otherwise we KEEP
        # the router's hand-set PATTERN_UTILITY prior for that pattern (frozen-prior run).
        self.router = MemoryRouter(consult_threshold=theta)
        for ob, P in PAT_MAP.items():
            self.router.pattern_utility.setdefault(P, {})
            if ob in util:                                   # calibrated value -> override prior
                self.router.pattern_utility[P][MemoryType.EM] = float(util[ob].get("orchestrator", 0.6))
            # else: keep the router's PATTERN_UTILITY[P][EM] prior
            self.router.pattern_utility[P][MemoryType.SM] = 0.0
            self.router.pattern_utility[P][MemoryType.ENT] = 0.0
        self.router.store_cost[MemoryType.EM] = float(cost.get("orchestrator", 0.55))

    def _distil_pm(self, fts, recs):
        """Cluster the bank by similarity and distil one generalised rule per cluster
        (common plan signature + a learned caution from the cluster's reflections).
        Populates the real ProceduralStore; returns rules tagged with a centroid
        embedding so the right rule can be retrieved per task."""
        idx = {id(ft): i for i, ft in enumerate(fts)}
        rules = []
        for cluster in MemoryManager._cluster_by_similarity(fts, 0.7):
            if len(cluster) < 2:                          # need support to generalise into a rule
                continue
            base = MemoryManager._distill_rule(cluster)   # {pattern, rule, support}
            members = [recs[idx[id(ft)]] for ft in cluster]
            cautions = [m.get("reflections", "") for m in members if m.get("reflections")]
            text = base["rule"] + (f"  Lesson: {cautions[0][:170]}" if cautions else "")
            cen = np.mean([np.asarray(ft.embedding, dtype=np.float32) for ft in cluster], axis=0)
            cen = cen / (np.linalg.norm(cen) + 1e-9)
            self.pm.consolidate_rule({"pattern": base["pattern"], "rule": text, "support": base["support"]})
            rules.append({"pattern": base["pattern"], "text": text, "support": base["support"], "emb": cen})
        return rules

    def pm_rule(self, task_text, min_sim=0.45):
        """Nearest distilled procedural rule for this task (PM is always-read; cheap)."""
        if not self._pm_rules:
            return None
        q = self.embedder.encode([task_text])[0]
        q = q / (np.linalg.norm(q) + 1e-9)
        best = max(self._pm_rules, key=lambda r: float(r["emb"] @ q))
        return best if float(best["emb"] @ q) >= min_sim else None

    def gate(self, ob_pattern):
        """Real rho-gate decision: does EM clear rho = U/C >= theta?"""
        P = PAT_MAP.get(ob_pattern, Pattern.COORDINATION)
        d = self.router.plan_retrieval("t", P, stm_hit=False)
        return MemoryType.EM in d.consulted_stores, d

    def search(self, task_text, k=5):
        return self.em.search(task_text, k)               # FAISS -> [(score, FullTaskMemory)]

    def preload(self, task_text, k=3, min_score=0.5, exclude_task=None):
        """Relevance-gated agent memory: pull per-app subtasks ONLY from banked
        tasks whose task-level similarity clears min_score, so we inject the
        near-duplicate's actions rather than the globally-nearest distractor.
        exclude_task drops the task's own banked entry (leave-one-out)."""
        by_app = {a: [] for a in APPS}
        for sc, m in self.em.search(task_text, k=8):
            if sc < min_score or (exclude_task and m.description == exclude_task):
                continue
            for sm in m.subtask_memories:
                if sm.agent in by_app and len(by_app[sm.agent]) < k:
                    by_app[sm.agent].append(sm)
        return by_app

    def __len__(self):
        return len(self.em)
