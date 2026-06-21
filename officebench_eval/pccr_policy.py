"""PCCR policy for OfficeBench — our architecture driving the env.

Replaces OfficeBench's single LLMPolicy: same app-switching loop, but the
prompt is augmented with rho-gated EM (procedure memory) at two granularities:
  orchestrator -> retrieved past task PLANS (guide the overall plan)
  agent        -> retrieved past per-app ACTIONS (guide the current app step)
no_memory injects nothing (the floor); retrieve_all always injects; pccr injects
only the stores the rho-gate opened. The Cerebras backbone is swapped in.

Built as a factory because the LLMPolicy base class is only importable once
OfficeBench is on sys.path (done by the runner).
"""
from __future__ import annotations

from .cerebras_llm import CerebrasLLM


def make_pccr_policy(LLMPolicyCls, model, env, config, memory=None, method="no_memory",
                     stores=frozenset(), k_orch=5, k_agent=3, exclude_task=None):

    class PCCRPolicy(LLMPolicyCls):
        def __init__(self):
            super().__init__(model_name="local-oss", key="", env=env, config=config)
            self.llm = CerebrasLLM(model_name=model, system_message=self.system_message)
            self.memory, self.method, self.stores = memory, method, stores
            self.k_orch, self.k_agent = k_orch, k_agent
            self.exclude_task = exclude_task           # leave-one-out for calibration
            self.task = config["task"]
            self.em_trace = {"method": method, "stores": sorted(stores),
                             "orchestrator_hits": [], "agent_hits": [], "injected_tokens": 0}

        def proc_action(self, action):
            """Robust action extraction: take the FIRST balanced {...} object.
            gpt-oss often emits two JSON blocks at once; OfficeBench's default
            (first '{' .. last '}') captures BOTH -> invalid JSON -> 'Malformed
            action' and ~half the steps wasted. This fixes that for all methods."""
            s = action or ""
            i = s.find("{")
            if i < 0:
                return action
            depth = 0
            for j in range(i, len(s)):
                if s[j] == "{":
                    depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return s[i:j + 1]
            return s[i:]

        def build_prompt(self, env):
            base = super().build_prompt(env)
            mem = self._memory_block(env)
            if mem:
                self.em_trace["injected_tokens"] += max(1, len(mem) // 4)
                return mem + "\n\n" + base
            return base

        def _memory_block(self, env):
            if not self.memory or self.method == "no_memory" or not self.stores:
                return ""
            blocks = []
            if "orchestrator" in self.stores:
                hits = self.memory.retrieve_orchestrator(self.task, self.k_orch,
                                                         exclude_task=self.exclude_task)
                if hits:
                    blocks.append("RELEVANT PAST TASK PLANS (reuse the steps if the task is similar):")
                    for sc, r in hits:
                        blocks.append(f" - ({sc:.2f}) task: {r['task'][:80]} | "
                                      f"plan: {' ; '.join(r['plan'][:8])}")
                    self.em_trace["orchestrator_hits"].append([r["task"][:60] for _, r in hits])
            if "agent" in self.stores and getattr(env, "current_app", None):
                hits = self.memory.retrieve_agent(self.task, env.current_app, self.k_agent,
                                                  exclude_task=self.exclude_task)
                if hits:
                    blocks.append(f"RELEVANT PAST {env.current_app} ACTIONS:")
                    for sc, s in hits:
                        blocks.append(f" - ({sc:.2f}) {s['text'][:120]}")
                    self.em_trace["agent_hits"].append([s["text"][:50] for _, s in hits])
            return "\n".join(blocks)

    return PCCRPolicy()
