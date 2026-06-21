"""PCCR policy for OfficeBench — driven by the REAL architecture (swap A).

Same app-switching loop, but memory is the real FAISS EpisodicStore gated by the
real router.py rho-gate (via RealArch). On an EM hit, em.search() supplies
orchestrator-level past plans and em.preload_for_agents() supplies per-app
subtask memories. no_memory injects nothing; retrieve_all always injects; pccr
injects only when the real rho-gate consults EM.
"""
from __future__ import annotations

from .cerebras_llm import CerebrasLLM


def make_pccr_policy(LLMPolicyCls, model, env, config, real_arch=None, method="no_memory",
                     pattern=None, exclude_task=None):

    class PCCRPolicy(LLMPolicyCls):
        def __init__(self):
            super().__init__(model_name="local-oss", key="", env=env, config=config)
            self.llm = CerebrasLLM(model_name=model, system_message=self.system_message)
            self.real, self.method, self.pattern = real_arch, method, pattern
            self.exclude_task = exclude_task
            self.task = config["task"]
            # decide EM consult ONCE per task (Phase-3 rho-gate)
            self._consult_em = False
            self._rho_decision = None
            if real_arch is not None and method != "no_memory":
                if method == "retrieve_all":
                    self._consult_em = True
                else:  # pccr -> real router.py rho-gate
                    self._consult_em, self._rho_decision = real_arch.gate(pattern)
            self.em_trace = {"method": method, "consult_em": self._consult_em,
                             "orchestrator_hits": [], "agent_hits": [], "injected_tokens": 0,
                             "faiss_backed": True}

        def build_prompt(self, env):
            base = super().build_prompt(env)
            mem = self._memory_block(env)
            if mem:
                self.em_trace["injected_tokens"] += max(1, len(mem) // 4)
                return mem + "\n\n" + base
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
