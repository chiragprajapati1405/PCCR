"""Drive OfficeBench through the REAL MemoryManager -- no approximations.

Replaces the bespoke RealArch gate/STM/PM logic with the actual architecture:
  - consolidate() fills EM + SM + PM(rules) + STM(prefill) from the bank, exactly
    as designed (clusters successes, distils PM rules, extracts SM facts, prefills
    STM exemplars).
  - retrieve() runs the real cascade: STM lookup FIRST (>= stm_cache_threshold) ->
    HIT short-circuits (cached_bundle, skip EM/SM/ENT); MISS -> rho-gate picks
    EM/SM/ENT by utility/cost. PM rule + orchestrator prompt are always-read.

So STM, EM, SM, ENT, PM behave precisely as the architecture specifies.
"""
from __future__ import annotations

import json
from collections import defaultdict

from memory_manager import config
from memory_manager.manager import MemoryManager
from memory_manager.types import MemoryType, Pattern
from .real_arch import APPS, PAT_MAP, _to_full

ORCH_PROMPT = "You are an orchestrator solving an office-automation task across apps."
AGENT_PROMPTS = {a: f"You are the {a} agent." for a in APPS}
TASK_RULES: dict[str, str] = {}
AGENT_PROFILES = {a: {} for a in APPS}


class RealMem:
    """The real MemoryManager wired for OfficeBench (EM/SM/PM/STM all live)."""

    def __init__(self, bank_path, theta=1.0, stm_threshold=0.92, username="user", date="2026-06-08"):
        s = config.Settings(embedding_backend="sentence-transformers", llm_backend="stub",
                            stm_cache_threshold=stm_threshold)
        self.mgr = MemoryManager(s)
        self.mgr.bootstrap(ORCH_PROMPT, AGENT_PROMPTS, TASK_RULES, AGENT_PROFILES)
        self.mgr.router.consult_threshold = theta
        self.username, self.date = username, date
        emb = self.mgr.embedder
        for i, rec in enumerate(json.load(open(bank_path))):
            ft = _to_full(rec, i)                                   # outcome="success"
            ft.embedding = emb.encode([ft.description])[0]          # needed for clustering
            self.mgr._task_log.append(ft)
        self.stats = self.mgr.consolidate()                        # fills EM/SM/PM/STM (as designed)

    def retrieve(self, task_desc, ob_pattern, exclude_task=None):
        """Real retrieve() cascade -> (injected memory block, trace)."""
        self.mgr.ingest_task(task_desc, self.username, self.date)
        self.mgr.wm.set_pattern(PAT_MAP.get(ob_pattern, Pattern.COORDINATION))  # use OB classification
        b = self.mgr.retrieve()
        consulted = [s.value for s in b.decision.consulted_stores]
        stm_hit = b.cached_bundle is not None
        blocks = []
        # PM: the learned procedural rule for this pattern (always-read)
        pm_rule = self.mgr.pm.get_rule_for_pattern(self.mgr.wm.current.pattern.value)
        if pm_rule:
            blocks.append(f"##PROCEDURAL RULE: {pm_rule}")
        if stm_hit:                                                # STM HIT -> cached plan, EM skipped
            cb = b.cached_bundle
            blocks.append("##CACHED PLAN (near-identical past task): " + " ; ".join(cb.plan[:8]))
        else:
            if b.episodes:                                          # EM (rho-gated)
                blocks.append("##RELEVANT PAST TASK PLANS:")
                for sc, m in b.episodes:
                    if exclude_task and m.description == exclude_task:
                        continue
                    blocks.append(f" - ({sc:.2f}) {m.description[:70]} | {' ; '.join(m.plan[:6])}")
                # agent-level actions grouped by app
                by_app = defaultdict(list)
                for _sc, m in b.episodes:
                    if exclude_task and m.description == exclude_task:
                        continue
                    for sm in m.subtask_memories:
                        if sm.agent in APPS and len(by_app[sm.agent]) < 3:
                            by_app[sm.agent].append(sm.action[:140])
                for app, acts in by_app.items():
                    blocks.append(f"##PAST {app} ACTIONS: " + " | ".join(acts))
            if b.facts:                                            # SM (rho-gated)
                blocks.append("##RELEVANT FACTS: " + " ; ".join(f for _s, f in b.facts[:5]))
            if b.profile:                                          # ENT (rho-gated)
                blocks.append("##USER PROFILE: " + ", ".join(f"{k}={v}" for k, v in b.profile.items()))
        text = ("\n".join(blocks) + "\n\n") if blocks else ""
        trace = {"stm_hit": stm_hit, "consulted_stores": consulted,
                 "consult_em": MemoryType.EM in b.decision.consulted_stores,
                 "consult_sm": MemoryType.SM in b.decision.consulted_stores,
                 "consult_ent": MemoryType.ENT in b.decision.consulted_stores,
                 "pm_used": bool(pm_rule), "injected_tokens": max(1, len(text) // 4)}
        return text, trace
