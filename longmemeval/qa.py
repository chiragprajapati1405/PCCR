"""Real-LLM QA layer for the LongMemEval integration.

Per question: the rho-gate picks stores -> retrieve evidence sessions -> assemble
context -> gpt-oss-120b ANSWERS -> gpt-oss-120b JUDGES vs the gold answer.
This is where (a) parallelism gives REAL latency wins (overlapping live API
calls) and (b) the rho-gate's token savings are measurable (fewer stores -> less
injected context).

Needs Cerebras keys in the environment (source cerebras.env).
"""
from __future__ import annotations

import asyncio
import os
import threading
import time

import numpy as np

from .lme import OPTIONAL

_TOK = lambda s: max(1, int(len(s) / 4))      # ~4 chars/token (same proxy as cost calibration)


# ── threaded multi-key Cerebras client (real concurrency for the sync SDK) ──────
class LMEClient:
    def __init__(self, model: str = "gpt-oss-120b"):
        from openai import OpenAI
        keys = [os.environ.get(f"CEREBRAS_KEY_{i}", "") for i in range(1, 5)]
        keys = [k for k in keys if k.strip()]
        if not keys:
            raise RuntimeError("no CEREBRAS_KEY_* in env — run `source cerebras.env`")
        self.clients = [OpenAI(api_key=k, base_url="https://api.cerebras.ai/v1", max_retries=2)
                        for k in keys]
        self.model = model
        self._i = 0
        self._lock = threading.Lock()
        self.n_keys = len(keys)
        self.calls = 0

    def _next(self):
        with self._lock:
            c = self.clients[self._i % len(self.clients)]
            self._i += 1
            self.calls += 1
            return c

    def call(self, system: str, user: str, max_tokens: int = 400):
        """Blocking call; returns (text, api_latency_s). Times ONLY the API call
        (no artificial pacing), so latency numbers are real. Retries on rate
        limits and on empty completions (which Cerebras returns under load)."""
        for attempt in range(12):
            c = self._next()
            try:
                t = time.perf_counter()
                r = c.chat.completions.create(
                    model=self.model, temperature=0, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt, time.perf_counter() - t
                time.sleep(1.0)                      # empty completion -> retry
            except Exception as e:
                s = str(e).lower()
                time.sleep(min(2.0 + attempt, 6.0) if ("429" in s or "rate" in s) else 1.0)
        return "", 0.0


# ── context assembly from the rho-gated stores ─────────────────────────────────
def _session_text_map(entry):
    out = {}
    for sid, sess in zip(entry["haystack_session_ids"], entry["haystack_sessions"]):
        out[sid] = "\n".join(f"{t['role']}: {t['content']}" for t in sess)
    return out


def _top_turns(mem, sims, sid_set, n):
    idxs = [i for i, sid in enumerate(mem.turn_to_sid) if sid in sid_set]
    idxs.sort(key=lambda i: -sims[i])
    return idxs[:n]


def assemble_context(entry, mem, qvec, stores, k=5, turns_per_store=4, window=1):
    """Inject the most-relevant TURNS (not whole sessions, which run ~15k chars
    and bury the evidence). Each consulted store contributes candidate turns:
      EM -> top turns within its top-k sessions
      SM -> top turns globally (turn-level)
      ENT-> top turns within the most-recent sessions
    Turns are deduped, a +/-window neighbour is added for context, grouped by
    session in order. More stores -> more candidate turns -> more injected
    tokens, so the rho-gate's store pruning shows up as a real token saving.
    Returns (context, injected_tokens, used_session_ids, per_store_flags)."""
    nturns = len(mem.turn_text)
    per_store = {s: 0 for s in OPTIONAL}
    if nturns == 0:
        return "", 0, [], per_store
    sims = mem.turn_sims(qvec)
    chosen = set()
    if "semantic" in stores:
        per_store["semantic"] = 1
        chosen.update(sorted(range(nturns), key=lambda i: -sims[i])[:turns_per_store])
    if "episodic" in stores:
        per_store["episodic"] = 1
        em_sids = {sid for sid, _ in mem.em(qvec, k)}
        chosen.update(_top_turns(mem, sims, em_sids, turns_per_store))
    if "entity" in stores:
        per_store["entity"] = 1
        chosen.update(_top_turns(mem, sims, set(mem.recency_sids[:k]), turns_per_store))

    final = set()
    for i in chosen:                                   # add neighbours within same session
        for j in range(max(0, i - window), min(nturns, i + window + 1)):
            if mem.turn_to_sid[j] == mem.turn_to_sid[i]:
                final.add(j)

    dates = dict(zip(entry["haystack_session_ids"], entry.get("haystack_dates", [])))
    blocks, used, cur = [], [], None
    for j in sorted(final, key=lambda j: (mem.turn_to_sid[j], j)):
        sid = mem.turn_to_sid[j]
        if sid != cur:
            blocks.append(f"\n[session {sid} | {dates.get(sid,'')}]"); cur = sid; used.append(sid)
        blocks.append(mem.turn_text[j])
    ctx = "\n".join(blocks).strip()
    return ctx, _TOK(ctx), used, per_store


_ANS_SYS = ("You are an assistant with access to excerpts of the user's past chat history. "
            "Answer the question using ONLY the provided memory. Be concise. "
            "If the memory does not contain the answer, reply exactly: I don't have that information.")
_JUDGE_SYS = ("You are a strict grader. Reply with ONLY one word: CORRECT or WRONG. "
              "For abstention cases where the reference says no information is available, the "
              "candidate is CORRECT only if it declines / says it doesn't know.")


def gen_answer(client, entry, ctx):
    user = f"MEMORY:\n{ctx}\n\nQUESTION (asked {entry.get('question_date','')}): {entry['question']}\n\nAnswer:"
    return client.call(_ANS_SYS, user, max_tokens=300)


import re as _re
_NUMWORD = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12"}


def _norm(s: str) -> str:
    s = (s or "").lower()
    for w, d in _NUMWORD.items():
        s = _re.sub(rf"\b{w}\b", d, s)
    return _re.sub(r"\s+", " ", _re.sub(r"[^a-z0-9 ]", " ", s)).strip()


def _content(g: str):
    return [t for t in g.split() if len(t) > 2 or t.isdigit()] or g.split()


def grade(entry, pred):
    """Deterministic grader (no LLM judge -> no extra API calls / rate-limit
    failures). Abstention: did it decline? Else: is the gold's key answer present
    in the candidate? (Lexical proxy for LongMemEval's GPT-4o judge — strong on
    factual answers, weaker on long subjective 'preference' golds.)"""
    from .lme import is_abstention
    if is_abstention(entry):
        return int(any(d in (pred or "").lower() for d in _DECLINE))
    p = _norm(pred)
    if not p:
        return 0
    gold = str(entry["answer"])
    # (a) first answer-sentence tokens mostly present (handles "30 days. 31 also ok")
    first = _content(_norm(_re.split(r"[.\n;]", gold)[0]))
    if first and sum(1 for t in set(first) if t in p) / len(set(first)) >= 0.6:
        return 1
    # (b) short gold: all key tokens present, or substring
    g = _norm(gold); gt = _content(g)
    if len(gt) <= 5:
        return int(all(t in p for t in gt) or g in p)
    # (c) long gold: at least half of content tokens present
    return int(sum(1 for t in set(gt) if t in p) / max(len(set(gt)), 1) >= 0.5)


_DECLINE = ("don't have", "do not have", "n't have that", "no information", "not enough",
            "don't know", "do not know", "couldn't find", "could not find", "cannot find",
            "can't find", "not mentioned", "no record", "didn't mention", "did not mention",
            "unable to", "not provided", "no mention", "isn't any", "no details")


def judge(client, entry, pred):
    """Abstention is graded programmatically (robust); answerable via LLM judge."""
    from .lme import is_abstention
    p = (pred or "").lower()
    if is_abstention(entry):
        return int(any(d in p for d in _DECLINE))            # correct iff it declines
    if not p.strip():
        return 0
    gold = str(entry["answer"])
    user = (f"Question: {entry['question']}\nReference answer: {gold}\nCandidate answer: {pred}\n"
            "Does the candidate convey the same key information as the reference? "
            "Reply CORRECT or WRONG.")
    out, _ = client.call(_JUDGE_SYS, user, max_tokens=8)
    o = out.strip().lower()
    neg = any(w in o for w in ("wrong", "incorrect", "false", "no,", "does not", "doesn't"))
    pos = any(w in o for w in ("correct", "yes", "true", "match"))
    return 1 if (pos and not neg) else 0


# ── parallel answer generation (real concurrent API calls via threads) ─────────
async def answer_batch_parallel(client, items, concurrency):
    """items: list of (entry, ctx). Returns list of (text, latency) in order.
    Real overlap: each blocking call runs in a thread, so network waits overlap."""
    sem = asyncio.Semaphore(concurrency)
    loop = asyncio.get_event_loop()

    async def one(entry, ctx):
        async with sem:
            return await loop.run_in_executor(None, lambda: gen_answer(client, entry, ctx))

    return await asyncio.gather(*[one(e, c) for e, c in items])
