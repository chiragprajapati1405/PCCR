"""PCCR policy for OfficeBench — driven by the REAL architecture (swap A).

Same app-switching loop, but memory is the real FAISS EpisodicStore gated by the
real router.py rho-gate (via RealArch). On an EM hit, em.search() supplies
orchestrator-level past plans and em.preload_for_agents() supplies per-app
subtask memories. no_memory injects nothing; retrieve_all always injects; pccr
injects only when the real rho-gate consults EM.
"""
from __future__ import annotations

from .cerebras_llm import CerebrasLLM

# Output-path + completion convention (NEXT_STEPS D3). The agent prompt never told
# the model WHERE to write outputs or that L3 tasks need MULTIPLE files; tracing the
# 67 L3 file_exist failures showed 17 wrong-dir/renamed + 36 "second output never made".
# Applied uniformly to ALL arms (a harness-fairness fix, not a PCCR-only advantage).
OUTPUT_CONVENTION = (
    "##OUTPUT RULES (follow exactly):\n"
    "1. Write EVERY output file under /testbed/data/ using the EXACT filename named in "
    "the task. Do not add prefixes, do not rename (e.g. if the task says report.pdf, "
    "write /testbed/data/report.pdf, not /testbed/report.pdf or a renamed file).\n"
    "2. Many tasks require MULTIPLE outputs across different apps (e.g. an Excel file AND "
    "calendar .ics events AND a converted PDF AND a sent email). Produce ALL required "
    "outputs before you finish — do NOT stop after the first one. Re-read the task and "
    "confirm every requested file/action is done before calling finish_task.\n"
)


def make_pccr_policy(LLMPolicyCls, model, env, config, real_arch=None, method="no_memory",
                     pattern=None, exclude_task=None, use_pm=False, real_mem=None):

    class PCCRPolicy(LLMPolicyCls):
        def __init__(self):
            super().__init__(model_name="local-oss", key="", env=env, config=config)
            self.llm = CerebrasLLM(model_name=model, system_message=self.system_message)
            self.llm.max_tokens = 2048   # 512 truncated gpt-oss actions to a lone '{' (~35% malformed)
            self.real, self.method, self.pattern = real_arch, method, pattern
            self.exclude_task = exclude_task
            self.task = config["task"]
            self._realmem_block = ""     # set on the gate arm when driven by the real MemoryManager
            self._pm_text = ""
            self._consult_em = False
            self._stm_hit = False
            self._rho_decision = None
            self._top_em_sim = 0.0
            if real_arch is not None:
                hits = real_arch.search(self.task, k=1)
                self._top_em_sim = float(hits[0][0]) if hits else 0.0

            gate_arm = method not in ("no_memory", "retrieve_all")
            if gate_arm and real_mem is not None:
                # === REAL ARCHITECTURE: MemoryManager.retrieve() full cascade ===
                # STM-first short-circuit, then rho-gated EM/SM/ENT, PM always-read.
                text, tr = real_mem.retrieve(self.task, pattern, exclude_task=exclude_task)
                self._realmem_block = text
                self.em_trace = {"method": method, "consult_em": tr["consult_em"],
                                 "stm_hit": tr["stm_hit"], "consult_sm": tr.get("consult_sm"),
                                 "consult_ent": tr.get("consult_ent"), "pm_used": tr["pm_used"],
                                 "consulted_stores": tr["consulted_stores"], "injected_tokens": 0,
                                 "orchestrator_hits": [], "agent_hits": [],
                                 "top_em_sim": round(self._top_em_sim, 3), "faiss_backed": True}
            else:
                # no_memory (nothing) / retrieve_all (always inject EM via _memory_block)
                if use_pm and real_arch is not None and method != "no_memory":
                    r = real_arch.pm_rule(self.task)
                    if r:
                        self._pm_text = f"##PROCEDURAL RULE (learned from {r['support']} past tasks): {r['text']}\n\n"
                if real_arch is not None and method == "retrieve_all":
                    self._consult_em = True
                self.em_trace = {"method": method, "consult_em": self._consult_em,
                                 "orchestrator_hits": [], "agent_hits": [], "injected_tokens": 0,
                                 "faiss_backed": True, "top_em_sim": round(self._top_em_sim, 3),
                                 "stm_hit": self._stm_hit,
                                 "consulted_stores": (["episodic"] if self._consult_em else []),
                                 "pm_used": bool(self._pm_text)}

        def proc_action(self, action):
            """Return the FIRST balanced JSON object. The base class takes
            first-'{' to last-'}', which merges concatenated actions
            ({...}{...}) into one malformed blob; gpt-oss frequently emits
            several at once. We brace-match and stop at the first complete one."""
            i = action.find("{")
            if i < 0:
                return action
            depth, in_str, esc = 0, False, False
            for j in range(i, len(action)):
                c = action[j]
                if in_str:
                    if esc:        esc = False
                    elif c == "\\": esc = True
                    elif c == '"':  in_str = False
                elif c == '"':      in_str = True
                elif c == "{":      depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return action[i:j + 1]            # first complete object
            return action[i:]                              # truncated: hand back partial

        def build_prompt(self, env):
            base = super().build_prompt(env)
            # gate arm: the real MemoryManager bundle; else PM(always) + EM(gated)
            prefix = self._realmem_block if self._realmem_block else (self._pm_text + self._memory_block(env))
            # D3: output-path + completion convention on EVERY arm (not counted as
            # injected memory tokens — it is a static harness instruction, like the system prompt)
            base = OUTPUT_CONVENTION + "\n" + base
            if prefix:
                self.em_trace["injected_tokens"] += max(1, len(prefix) // 4)
                return prefix + "\n\n" + base
            return base

        def _memory_block(self, env):
            if not self._consult_em or self.real is None:
                return ""
            blocks = []
            hits = self.real.search(self.task, k=5)                 # FAISS orchestrator-level
            if hits:
                blocks.append("RELEVANT PAST TASK PLANS (reuse the steps if similar):")
                for sc, m in hits:
                    if self.exclude_task and m.description == self.exclude_task:
                        continue
                    blocks.append(f" - ({sc:.2f}) {m.description[:80]} | "
                                  f"plan: {' ; '.join(m.plan[:8])}")
                self.em_trace["orchestrator_hits"].append([m.description[:60] for _s, m in hits[:5]])
            app = getattr(env, "current_app", None)
            if app:
                pre = self.real.preload(self.task, k=3)             # FAISS agent-level
                ah = pre.get(app, [])
                if ah:
                    blocks.append(f"RELEVANT PAST {app} ACTIONS:")
                    for sm in ah:
                        blocks.append(f" - {sm.action[:110]}")
                    self.em_trace["agent_hits"].append([sm.action[:50] for sm in ah])
            return "\n".join(blocks)

    return PCCRPolicy()
