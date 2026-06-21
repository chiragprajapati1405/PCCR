"""Procedure memory (EM) for OfficeBench — our LegoMem-PM equivalent.

Stores successful TRAINING trajectories and serves them at two granularities
(LegoMem): orchestrator (the full task plan) and agent (per-app subtask steps).
Retrieval is FAISS-style cosine over all-MiniLM-L6-v2 embeddings of the task text.
"""
from __future__ import annotations

import json

import numpy as np

_MODEL = None
def _embed(texts):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(_MODEL.encode(list(texts), normalize_embeddings=True, show_progress_bar=False))


def _app_of(action_str: str) -> str:
    try:
        d = json.loads(action_str) if isinstance(action_str, str) else action_str
        return (d.get("app") or "").lower()
    except Exception:
        return ""


def _fmt(action) -> str:
    return action if isinstance(action, str) else json.dumps(action)


class ProcedureMemory:
    def __init__(self):
        self.records = []          # {task, pattern, level, plan:[str], steps:[{action,obs}]}
        self._task_emb = None      # (N,d) orchestrator index
        self._sub = []             # [{app, text, task_idx}] agent units
        self._sub_emb = None

    def add(self, task, pattern, level, steps):
        """steps: list of (action, observation) from a SUCCESSFUL run."""
        self.records.append({
            "task": task, "pattern": pattern, "level": level,
            "plan": [_fmt(a) for a, _o in steps],
            "steps": [{"action": _fmt(a), "obs": (o or "")[:300]} for a, o in steps],
        })

    def build_index(self):
        if not self.records:
            return
        self._task_emb = _embed([r["task"] for r in self.records])
        self._sub = []
        for i, r in enumerate(self.records):
            for s in r["steps"]:
                app = _app_of(s["action"])
                if app and app not in ("system",):
                    self._sub.append({"app": app, "text": f"{s['action']} -> {s['obs'][:120]}",
                                      "task_idx": i})
        self._sub_emb = _embed([s["text"] for s in self._sub]) if self._sub else None

    def retrieve_orchestrator(self, task, k=5):
        if self._task_emb is None:
            return []
        q = _embed([task])[0]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._task_emb @ q
        return [(float(sims[i]), self.records[i]) for i in np.argsort(-sims)[:k]]

    def retrieve_agent(self, task, app, k=3):
        if self._sub_emb is None:
            return []
        cand = [j for j, s in enumerate(self._sub) if s["app"] == (app or "").lower()]
        if not cand:
            return []
        q = _embed([task])[0]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self._sub_emb[cand] @ q
        order = np.argsort(-sims)[:k]
        return [(float(sims[o]), self._sub[cand[o]]) for o in order]

    def top_similarity(self, task):
        hits = self.retrieve_orchestrator(task, k=1)
        return hits[0][0] if hits else 0.0

    def save(self, path):
        json.dump(self.records, open(path, "w"), indent=2)

    def load(self, path):
        self.records = json.load(open(path))
        self.build_index()
        return len(self.records)

    def __len__(self):
        return len(self.records)
