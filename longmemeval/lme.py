"""LongMemEval integration for PCCR — REAL data, real retrieval.

Maps each LongMemEval question's chat history onto three of our memory stores
and routes retrieval with the PCCR rho-gate:

  EM  (episodic) : SESSION-level embeddings  -> "which past conversation"
  SM  (semantic) : TURN-level embeddings     -> fine-grained fact units
  ENT (entity)   : the k most-RECENT sessions -> current/updated facts

The rho-gate (rho = U(question_type, store) / C(store) >= theta) decides which
stores to consult per question. C(store) is the MEASURED cost from
calibration/store_cost.json (episodic/semantic are FAISS searches; entity is a
cheap recency lookup). This is the same gate used on OfficeBench, applied to a
real long-term-memory benchmark.

Dataset: longmemeval_s.json (500 instances, ~50 distractor sessions each).
Download: see longmemeval/README.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

OPTIONAL = ("episodic", "semantic", "entity")

# ---- measured store cost (anchored cheapest = 0.15); see calibrate.py costs ----
def load_cost(path: str = "calibration/store_cost.json") -> dict:
    em = ent = None
    try:
        c = json.load(open(path))["store_cost"]
        em, ent = c.get("episodic"), c.get("entity")
    except Exception:
        pass
    em = em if em is not None else 4.543
    ent = ent if ent is not None else 0.15
    return {"episodic": em, "semantic": em, "entity": ent}   # SM is a FAISS search like EM


# ---- rho-gate utility table: U(question_type, store) in [0,1] -------------------
# The novelty: route to the store that actually helps THIS memory ability.
#   In LongMemEval ALL evidence lives in past sessions, so EM (session search)
#   is the workhorse and stays high for every type; the gate's job is to prune
#   the EXPENSIVE, redundant SM (turn-level) search where EM alone suffices
#   (single-session recall) and to add the cheap ENT (recency) where the answer
#   is the latest fact. (These utilities are priors; A2-style counterfactual
#   calibration from data is the principled next step — see the 4/12 knowledge-
#   update miss when ENT was wrongly treated as a substitute for EM.)
QA_UTILITY = {
    "single-session-user":       {"episodic": 0.90, "semantic": 0.30, "entity": 0.20},
    "single-session-assistant":  {"episodic": 0.90, "semantic": 0.30, "entity": 0.20},
    "single-session-preference": {"episodic": 0.90, "semantic": 0.30, "entity": 0.60},
    "multi-session":             {"episodic": 0.90, "semantic": 0.85, "entity": 0.20},
    "temporal-reasoning":        {"episodic": 0.90, "semantic": 0.85, "entity": 0.40},
    "knowledge-update":          {"episodic": 0.90, "semantic": 0.85, "entity": 0.95},
}
DEFAULT_UTIL = {"episodic": 0.90, "semantic": 0.50, "entity": 0.40}


def rho_gate(question_type: str, theta: float, cost: dict) -> set:
    """PCCR: consult store s iff U/C >= theta. STM-style short-circuit is N/A
    here (each question has its own fresh haystack)."""
    u = QA_UTILITY.get(question_type, DEFAULT_UTIL)
    return {s for s in OPTIONAL if u[s] / cost[s] >= theta}


def stores_for(method: str, question_type: str, theta: float, cost: dict) -> set:
    if method == "pccr":
        return rho_gate(question_type, theta, cost)
    if method == "retrieve_all":
        return set(OPTIONAL)
    if method == "boolean":              # fixed hand-set baseline: always EM + ENT
        return {"episodic", "entity"}
    raise ValueError(method)


# ---- data loading --------------------------------------------------------------
def load_dataset(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def is_abstention(entry: dict) -> bool:
    return str(entry["question_id"]).endswith("_abs") or not entry.get("answer_session_ids")


def _session_text(session: list) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


# ---- embedding (cached singleton) ----------------------------------------------
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


@dataclass
class QuestionMemory:
    """The three stores built for ONE question's haystack (per-question, isolated
    — the external analogue of per-task WM isolation)."""
    sids: list                      # session ids (parallel arrays below)
    sess_emb: np.ndarray            # (n_sessions, d)  -> EM
    turn_emb: np.ndarray            # (n_turns, d)     -> SM
    turn_to_sid: list               # session id for each turn row
    recency_sids: list              # session ids newest-first  -> ENT

    def em(self, qvec, k):                              # session-level
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.sess_emb @ qvec                 # benign macOS Accelerate BLAS quirk
        idx = np.argsort(-sims)[:k]
        return [(self.sids[i], float(sims[i])) for i in idx]

    def sm(self, qvec, k):                              # turn-level -> sessions
        if len(self.turn_emb) == 0:
            return []
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = self.turn_emb @ qvec
        idx = np.argsort(-sims)[:k]
        out, seen = [], set()
        for i in idx:
            sid = self.turn_to_sid[i]
            if sid not in seen:
                seen.add(sid); out.append((sid, float(sims[i])))
        return out

    def ent(self, k):                                   # recency (current facts)
        return [(sid, None) for sid in self.recency_sids[:k]]

    def sim_to(self, sid, qvec):                        # sim of a session to the query
        i = self.sids.index(sid)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return float(self.sess_emb[i] @ qvec)


def build_memory(entry: dict, model) -> QuestionMemory:
    sessions = entry["haystack_sessions"]
    sids     = list(entry["haystack_session_ids"])
    dates    = list(entry.get("haystack_dates", [""] * len(sids)))

    sess_text = [_session_text(s) for s in sessions]
    sess_emb  = model.encode(sess_text, normalize_embeddings=True, show_progress_bar=False)

    turn_text, turn_to_sid = [], []
    for sid, sess in zip(sids, sessions):
        for t in sess:
            turn_text.append(f"{t['role']}: {t['content']}")
            turn_to_sid.append(sid)
    turn_emb = (model.encode(turn_text, normalize_embeddings=True, show_progress_bar=False)
                if turn_text else np.zeros((0, sess_emb.shape[1]), dtype=sess_emb.dtype))

    # newest first by date string (LongMemEval dates sort lexicographically by Y/M/D)
    recency = [sid for _, sid in sorted(zip(dates, sids), reverse=True)]
    return QuestionMemory(sids, np.asarray(sess_emb), np.asarray(turn_emb), turn_to_sid, recency)


# ---- per-question evaluation ---------------------------------------------------
@dataclass
class QResult:
    qid: str
    qtype: str
    method: str
    consulted: tuple
    n_consults: int
    retrieved: list                 # session ids
    abstained: bool
    correct: int                    # 1/0 (recall hit for answerable; abstain for _abs)
    kind: str                       # "answerable" | "abstention"


def answer_question(entry, mem, model, method, theta, cost, k=5, floor=0.30,
                    qvec=None, stores=None) -> QResult:
    if qvec is None:
        qvec = model.encode([entry["question"]], normalize_embeddings=True,
                            show_progress_bar=False)[0]
    qtype = entry["question_type"]
    if stores is None:
        stores = stores_for(method, qtype, theta, cost)

    retrieved = []
    if "episodic" in stores: retrieved += [s for s, _ in mem.em(qvec, k)]
    if "semantic" in stores: retrieved += [s for s, _ in mem.sm(qvec, k)]
    if "entity"   in stores: retrieved += [s for s, _ in mem.ent(k)]
    retrieved = list(dict.fromkeys(retrieved))                      # dedup, keep order

    # abstention = nothing retrieved is actually relevant (max session sim < floor)
    maxsim = max((mem.sim_to(s, qvec) for s in retrieved), default=0.0)
    abstained = maxsim < floor

    gold = set(entry.get("answer_session_ids") or [])
    if is_abstention(entry):
        kind, correct = "abstention", int(abstained)
    else:
        kind, correct = "answerable", int(bool(set(retrieved) & gold) and not abstained)

    return QResult(entry["question_id"], qtype, method, tuple(sorted(stores)),
                   len(stores), retrieved, abstained, correct, kind)
