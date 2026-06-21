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

from memory_manager.embeddings import build_embedder
from memory_manager.router import MemoryRouter
from memory_manager.stores.episodic import EpisodicStore
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
        for i, rec in enumerate(json.load(open(bank_path))):
            self.em.add(_to_full(rec, i))                 # -> FAISS index
        # configure the REAL router for OfficeBench (EM gate only; SM/ENT off here)
        self.router = MemoryRouter(consult_threshold=theta)
        for ob, P in PAT_MAP.items():
            u = util.get(ob, {})
            self.router.pattern_utility.setdefault(P, {})
            self.router.pattern_utility[P][MemoryType.EM] = float(u.get("orchestrator", 0.6))
            self.router.pattern_utility[P][MemoryType.SM] = 0.0
            self.router.pattern_utility[P][MemoryType.ENT] = 0.0
        self.router.store_cost[MemoryType.EM] = float(cost.get("orchestrator", 0.55))

    def gate(self, ob_pattern):
        """Real rho-gate decision: does EM clear rho = U/C >= theta?"""
        P = PAT_MAP.get(ob_pattern, Pattern.COORDINATION)
        d = self.router.plan_retrieval("t", P, stm_hit=False)
        return MemoryType.EM in d.consulted_stores, d

    def search(self, task_text, k=5):
        return self.em.search(task_text, k)               # FAISS -> [(score, FullTaskMemory)]

    def preload(self, task_text, k=3):
        return self.em.preload_for_agents(APPS, task_text, k)

    def __len__(self):
        return len(self.em)
