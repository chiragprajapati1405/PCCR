"""Procedural Memory (PM): static rules and prompts.

Per the lifecycle table this is written ONCE at bootstrap ("Developer -> PM
store", "Once at startup (never again)") and then only ever READ. It also
absorbs the consolidation phase's "PM: consolidate -> learned_rules", which
is the one exception to "never write again": batches of similar episodes can
append durable rules here after training.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProceduralStore:
    orchestrator_prompt: str = ""
    agent_prompts: dict[str, str] = field(default_factory=dict)
    task_type_rules: dict[str, str] = field(default_factory=dict)   # pattern code -> rule text
    agent_capability_profiles: dict[str, dict] = field(default_factory=dict)
    learned_rules: list[dict] = field(default_factory=list)         # filled by consolidation

    # -- Bootstrap: WRITE (init) --------------------------------------------

    def bootstrap(
        self,
        orchestrator_prompt: str,
        agent_prompts: dict[str, str],
        task_type_rules: dict[str, str],
        agent_capability_profiles: dict[str, dict],
    ) -> None:
        self.orchestrator_prompt = orchestrator_prompt
        self.agent_prompts = dict(agent_prompts)
        self.task_type_rules = dict(task_type_rules)
        self.agent_capability_profiles = dict(agent_capability_profiles)

    # -- Reads (every orchestrator / agent call) ----------------------------

    def get_orchestrator_prompt(self) -> str:
        return self.orchestrator_prompt

    def get_agent_prompt(self, agent_name: str) -> str:
        return self.agent_prompts.get(agent_name, "")

    def get_rule_for_pattern(self, pattern_code: str) -> str:
        """Static bootstrap rule for this pattern PLUS any consolidated/learned
        rules for it (the consolidation -> retrieval path; previously learned_rules
        were write-only because this only read task_type_rules)."""
        static = self.task_type_rules.get(pattern_code, "")
        learned = " ".join(r["rule"] for r in self.learned_rules
                           if r.get("pattern") == pattern_code and r.get("rule"))
        return (static + " " + learned).strip()

    def get_capability_profile(self, agent_name: str) -> dict:
        return self.agent_capability_profiles.get(agent_name, {})

    # -- Consolidation: the one allowed post-bootstrap write ----------------

    def consolidate_rule(self, rule: dict) -> None:
        """Append a learned rule distilled from 10+ similar episodes."""
        self.learned_rules.append(rule)
