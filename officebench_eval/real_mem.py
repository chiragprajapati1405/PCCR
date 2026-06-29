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

# OfficeBench tool failure signals (the env's own per-action success/error strings).
_FAIL_SIGNALS = ("Malformed action", "does not exist", "Failed", "Error", "No such file")


def is_action_failure(obs: str) -> bool:
    o = str(obs or "")
    return any(sig in o for sig in _FAIL_SIGNALS)


def _ok_actions(rec) -> list[str]:
    """The ordered action strings from a banked trajectory whose tool result did
    NOT signal failure -- the clean, executable sequence to replay/adapt."""
    out = []
    for st in rec.get("steps", []):
        if not is_action_failure(st.get("obs", "")):
            a = st.get("action", "")
            if a:
                out.append(a)
    return out


# Valid OfficeBench app/action names. Past-action *examples* are sanitised to these so the
# bank's STALE malformed steps (pre-fix traces emitted shell action "run"/"list_directory",
# which do not exist) are never injected as "successful past actions" for the agent to imitate.
_VALID_ACTIONS = {
    "shell": {"command"},
    "excel": {"convert_to_pdf", "create_new_file", "delete_cell", "read_file", "set_cell"},
    "word": {"convert_to_pdf", "create_new_file", "read_file", "write_to_file"},
    "pdf": {"image_convert_to_pdf", "convert_to_image", "convert_to_word", "read_file"},
    "ocr": {"recognize_file"},
    "calendar": {"create_event", "delete_event", "list_events"},
    "email": {"list_emails", "read_email", "send_email"},
}
_ACTION_ALIAS = {  # stale/hallucinated name -> correct name (shell only ever needs `command`)
    "run": "command", "list_directory": "command", "list_dir": "command",
    "list_files": "command", "list": "command", "execute": "command", "ls": "command",
}


def _clean_past_action(action_str: str, agent: str):
    """Extract the executable action JSON from a banked step blob (which may be wrapped in
    <think>...</think><action>{...}</action>) and keep it ONLY if its action name is valid
    for the agent's app, remapping a few known stale aliases. Returns a compact one-line
    string, or None to DROP a poisoned/unparseable example (so it is never injected)."""
    import re
    m = re.search(r"\{.*\}", str(action_str or ""), re.S)   # the action JSON (largest brace span)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    act = str(obj.get("action", ""))
    valid = _VALID_ACTIONS.get(agent, set())
    if act not in valid:
        act = _ACTION_ALIAS.get(act, "")
        if act not in valid:
            return None                                     # genuinely invalid -> drop
        obj["action"] = act
    obj.pop("app", None)                                    # app is already the block header
    return json.dumps(obj, separators=(",", ":"))[:160]


class RealMem:
    """The real MemoryManager wired for OfficeBench (EM/SM/PM/STM all live)."""

    def __init__(self, bank_path, theta=1.0, stm_threshold=0.92, username="user", date="2026-06-08",
                 confidence_gate=False, sim_threshold=0.55, curate_pm=False, model="gpt-oss-120b",
                 stm_capacity=0, online=False, step_hint=False):
        s = config.Settings(embedding_backend="sentence-transformers", llm_backend="stub",
                            stm_cache_threshold=stm_threshold, confidence_gate=confidence_gate,
                            stm_capacity=stm_capacity)
        # CRITICAL: MemoryManager's EM/SM use config.FAISS_DIR, which is a SHARED PERSISTENT
        # directory -- every RealMem build LOADED the existing index and ADDED 62 more vectors
        # (62 -> 124 -> 186 ...), polluting EM with duplicate/stale vectors AND corrupting the
        # file under concurrent (process) builds. Point each RealMem at a FRESH, ISOLATED temp
        # dir so it holds EXACTLY this bank's 62 procedures, clean (mirrors RealArch's tempdir).
        import tempfile
        config.FAISS_DIR = tempfile.mkdtemp(prefix="rm_faiss_")
        self.mgr = MemoryManager(s)
        self.mgr.bootstrap(ORCH_PROMPT, AGENT_PROMPTS, TASK_RULES, AGENT_PROFILES)
        self.mgr.router.consult_threshold = theta
        # A1: per-query retrieval-confidence gate (single global similarity threshold)
        self.confidence_gate = confidence_gate
        if confidence_gate:
            self.mgr.router.confidence_only = True
            self.mgr.router.confidence_sim_threshold = sim_threshold
        # A2: online closed-loop -- learn U=P_on-P_off per (pattern,store) from outcomes.
        # Uses the prior-utility gate (not confidence_only, which it would bypass).
        self.online = online
        if online:
            from memory_manager.online_utility import OnlineUtility
            self.mgr.router.online = OnlineUtility()
        self.step_hint = step_hint            # B3: opt-in step-budget hint (default off = baseline)
        self.username, self.date = username, date
        emb = self.mgr.embedder
        self._clean_actions: dict[str, list[str]] = {}             # desc -> ordered SUCCESSFUL actions (for replay, B1)
        bank = json.load(open(bank_path))
        for i, rec in enumerate(bank):
            ft = _to_full(rec, i)                                   # outcome="success"
            ft.embedding = emb.encode([ft.description])[0]          # needed for clustering
            self.mgr._task_log.append(ft)
            self._clean_actions[rec["task"]] = _ok_actions(rec)
        self.stats = self.mgr.consolidate()                        # fills EM/SM/PM/STM (as designed)
        if curate_pm:                                              # D1: LLM-curated actionable PM rules
            self._curate_pm_rules(bank, model)

    # -- D1: LLM-curated, actionable PM rules (replaces rule-based step-signatures) --
    def _curate_pm_rules(self, bank, model, cache_path=None):
        """Distil each task pattern's successful episodes + reflections into ONE crisp,
        actionable PM rule (canonical step order, key params, cautions, and -- for
        multi-output patterns -- a completion checklist). One Cerebras call per pattern,
        cached to disk so the parallel run / re-runs don't repeat it."""
        import os
        cache_path = cache_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "pm_curated.json")
        curated = {}
        if os.path.exists(cache_path):
            try:
                curated = json.load(open(cache_path))
            except Exception:
                curated = {}
        if not curated:
            from .cerebras_llm import CerebrasLLM
            llm = CerebrasLLM(model_name=model, system_message="You distil reusable office-automation procedures.")
            llm.max_tokens = 600
            by_pat = defaultdict(list)
            for rec in bank:
                by_pat[rec.get("pattern", "multi_app")].append(rec)
            for pat, recs in by_pat.items():
                ex = recs[:6]
                body = "\n\n".join(
                    f"TASK: {r['task']}\nPLAN: {' ; '.join(r.get('plan', [])[:8])}\n"
                    f"LESSON: {str(r.get('reflections',''))[:280]}" for r in ex)
                prompt = (
                    "The agent may ONLY use these EXACT app/action names (anything else is rejected as a "
                    "'Malformed action'):\n"
                    "  shell: command   (to list files, action='command' with command='ls /testbed/data' -- "
                    "there is NO 'list_directory', 'list', or 'run' action)\n"
                    "  excel: read_file, set_cell, delete_cell, create_new_file, convert_to_pdf\n"
                    "  word: read_file, write_to_file, create_new_file, convert_to_pdf\n"
                    "  pdf: read_file, convert_to_word, convert_to_image\n"
                    "  ocr: recognize_file        calendar: create_event, delete_event, list_events\n"
                    "  email: list_emails, read_email, send_email        llm: complete_text\n"
                    "When the rule mentions an action, use ONLY a name from this list. NEVER invent names "
                    "like list_directory, run, list, write_text, extract_text, recognize.\n\n"
                    f"Below are {len(ex)} SUCCESSFUL examples of the same kind of task (pattern: {pat}).\n\n"
                    f"{body}\n\n"
                    "Write ONE reusable PROCEDURAL RULE an agent should follow for this kind of task. Cover: "
                    "the canonical step order across apps; the key parameters to get exactly right (file paths "
                    "under /testbed/data, exact filenames, cell refs); the common failure modes to AVOID "
                    "(cautions -- ALWAYS re-read the current data to find the actual row/cell/record positions "
                    "rather than assuming fixed indices; include ALL matching items; save the file after "
                    "editing; verify the output before finishing); and IF these tasks produce multiple output "
                    "files across apps, an explicit checklist of ALL outputs to produce before finishing. "
                    "2-4 imperative sentences, no preamble.")
                curated[pat] = (llm.generate(prompt) or "").strip()
            try:
                json.dump(curated, open(cache_path, "w"), indent=2)
            except Exception:
                pass
        # install: replace the rule-based learned_rules with the curated ones (keyed by enum value)
        from officebench_eval.real_arch import PAT_MAP
        self.mgr.pm.learned_rules = []
        for pat, rule in curated.items():
            if rule:
                self.mgr.pm.consolidate_rule(
                    {"pattern": PAT_MAP.get(pat, Pattern.COORDINATION).value, "rule": rule, "support": 0})
        self.pm_curated = curated

    def replay_candidate(self, task_desc, exclude_task=None, min_sim=0.75):
        """B1: the best EM episode's ORDERED SUCCESSFUL action sequence, IF the top
        match clears min_sim (a small, safe delta to adapt-and-replay). Returns
        (sim, matched_description, [action_json_str, ...]) or None."""
        for sc, m in self.mgr.em.search(task_desc, k=3):
            if exclude_task and m.description == exclude_task:
                continue
            if sc < min_sim:
                return None
            acts = self._clean_actions.get(m.description, [])
            return (float(sc), m.description, acts) if acts else None
        return None

    def record_outcome(self, ob_pattern, consulted_stores, success):
        """A2: feed one completed task's outcome into the online estimator so the gate
        adapts (U=P_on-P_off). Maps the OB pattern + consulted store names to the
        router's types. No-op unless online learning is enabled."""
        if not self.online:
            return
        consulted = {MemoryType(s) for s in consulted_stores if s in (
            MemoryType.EM.value, MemoryType.SM.value, MemoryType.ENT.value)}
        self.mgr.router.record_outcome(PAT_MAP.get(ob_pattern, Pattern.COORDINATION), consulted, bool(success))

    def retrieve(self, task_desc, ob_pattern, exclude_task=None):
        """Real retrieve() cascade -> (injected memory block, trace)."""
        self.mgr.ingest_task(task_desc, self.username, self.date)
        self.mgr.wm.set_pattern(PAT_MAP.get(ob_pattern, Pattern.COORDINATION))  # use OB classification
        b = self.mgr.retrieve()
        consulted = [s.value for s in b.decision.consulted_stores]
        stm_hit = b.cached_bundle is not None
        # Split the cascade into a HEAVY procedural part (PM rule + concrete past-action
        # examples + facts/profile) and a LIGHT roadmap part (the plan steps). The heavy part
        # is STATIC guidance, most useful while planning -- re-sending it on every step is the
        # root cause of the 2x token cost vs retrieve-all. The policy injects `heavy` only for
        # the first few steps (inject-once) and `light` every step (the episode roadmap, at
        # parity with retrieve-all). `blocks` (heavy+light combined) is kept for compatibility.
        heavy, light = [], []
        # PM: the learned procedural rule for this pattern (always-read)
        pm_rule = self.mgr.pm.get_rule_for_pattern(self.mgr.wm.current.pattern.value)
        if pm_rule:
            heavy.append(f"##PROCEDURAL RULE: {pm_rule}")
        if stm_hit:                                                # STM HIT -> cached plan, EM skipped
            cb = b.cached_bundle
            light.append("##CACHED PLAN (near-identical past task): " + " ; ".join(cb.plan[:8]))
        else:
            if b.episodes:                                          # EM (rho-gated)
                # B3: memory-guided step budget -- the closest past episode's length is a
                # soft expectation that curbs exploratory flailing (a hint, not a hard cap).
                top_sc, top_m = b.episodes[0]
                n_exp = len(self._clean_actions.get(top_m.description, [])) or len(top_m.plan)
                if self.step_hint and top_sc >= 0.6 and n_exp:
                    light.append(f"##STEP BUDGET: a near-identical past task finished in ~{n_exp} "
                                 "actions; plan efficiently and avoid unnecessary exploration.")
                light.append("##RELEVANT PAST TASK PLANS:")               # the roadmap (cheap, every step)
                for sc, m in b.episodes:
                    if exclude_task and m.description == exclude_task:
                        continue
                    light.append(f" - ({sc:.2f}) {m.description[:70]} | {' ; '.join(m.plan[:6])}")
                # agent-level concrete action examples grouped by app (heavy: planning aid)
                by_app = defaultdict(list)
                for _sc, m in b.episodes:
                    if exclude_task and m.description == exclude_task:
                        continue
                    for sm in m.subtask_memories:
                        if sm.agent in APPS and len(by_app[sm.agent]) < 3:
                            clean = _clean_past_action(sm.action, sm.agent)
                            if clean:                          # drop poisoned/invalid examples
                                by_app[sm.agent].append(clean)
                for app, acts in by_app.items():
                    if acts:
                        heavy.append(f"##PAST {app} ACTIONS: " + " | ".join(acts))
            if b.facts:                                            # SM (rho-gated)
                heavy.append("##RELEVANT FACTS: " + " ; ".join(f for _s, f in b.facts[:5]))
            if b.profile:                                          # ENT (rho-gated)
                heavy.append("##USER PROFILE: " + ", ".join(f"{k}={v}" for k, v in b.profile.items()))
        heavy_text = ("\n".join(heavy) + "\n") if heavy else ""
        light_text = ("\n".join(light) + "\n") if light else ""
        text = (heavy_text + light_text + "\n") if (heavy_text or light_text) else ""
        trace = {"stm_hit": stm_hit, "consulted_stores": consulted,
                 "consult_em": MemoryType.EM in b.decision.consulted_stores,
                 "consult_sm": MemoryType.SM in b.decision.consulted_stores,
                 "consult_ent": MemoryType.ENT in b.decision.consulted_stores,
                 "pm_used": bool(pm_rule), "injected_tokens": max(1, len(text) // 4),
                 "heavy_block": heavy_text, "light_block": light_text}
        return text, trace
