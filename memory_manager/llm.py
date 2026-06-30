"""LLM client abstraction used by the orchestrator and sub-agents.

Two backends:
  - AnthropicLLM : real calls via the `anthropic` SDK (needs ANTHROPIC_API_KEY)
  - StubLLM      : deterministic, rule-based "model" that inspects the prompt
                   and returns canned JSON/text. This keeps the lifecycle
                   (and, critically, the *router*) fully testable and runnable
                   without API keys or network access -- the router's job is
                   to decide *what context to assemble*, which we can verify
                   independent of which model consumes that context.

Both implement `complete(system, user) -> str`.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

DEFAULT_MODEL = "claude-haiku-4-5-20251001"


class LLM:
    def complete(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = 1024):
        import anthropic  # local import: optional dependency

        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class StubLLM(LLM):
    """A small rule-based stand-in for an LLM, specialized for this use case.

    It looks for marker strings the orchestrator/agent prompts embed (e.g.
    "ROLE: orchestrator", "AVAILABLE_AGENTS: ...", "SUBTASK: ...") and returns
    the JSON shape that role expects. This is intentionally narrow: it exists
    to make the *memory and routing* logic testable, not to demonstrate
    reasoning quality.
    """

    def __init__(self, seed: int = 0):
        self._seed = seed

    def complete(self, system: str, user: str) -> str:
        if "ROLE: orchestrator" in system:
            return self._orchestrate(system, user)
        if "ROLE: agent" in system:
            return self._act(system, user)
        if "ROLE: classifier" in system:
            return self._classify(user)
        return json.dumps({"thought": "stub default response", "text": "ok"})

    # -- role handlers ----------------------------------------------------

    def _classify(self, user: str) -> str:
        text = user.lower()
        # Order matters: question-style lookups are checked first because a
        # phrase like "...my next MEETING with Dana..." would otherwise also
        # trip the coordination check below.
        if any(k in text for k in ("when is", "what time", "do i have", "look up", "who is")):
            pattern = "A"
        elif any(k in text for k in ("schedule", "and then", "then send", "and send")):
            pattern = "C"
        elif any(k in text for k in ("weekly", "every", "again", "as usual", "same as")):
            pattern = "D"
        elif any(k in text for k in ("mess", "deal with", "sort out", "figure out")):
            pattern = "E"
        else:
            pattern = "B"
        return json.dumps({"pattern": pattern})

    def _orchestrate(self, system: str, user: str) -> str:
        agents = self._extract_list(system, "AVAILABLE_AGENTS")
        # Count completed steps via the step-history rendering's unique
        # "] ACTION:" marker (memory_manager.stores.working.history_text).
        # Counting "OBSERVATION:" directly would double-count: retrieved
        # Semantic-memory facts embed truncated observation text and contain
        # the same substring, which would inflate this and cause premature
        # FINISH whenever SM happens to be consulted.
        history_len = user.count("] ACTION:")
        task = self._extract_field(user, "TASK")

        plan = self._infer_plan(task, agents)
        step = min(history_len, len(plan) - 1)
        agent, instruction = plan[step]

        finished = history_len >= len(plan)
        return json.dumps(
            {
                "thought": f"step {history_len}: delegate to {agent}",
                "agent": "FINISH" if finished else agent,
                "subtask": "done" if finished else instruction,
                "plan_driven": True,
                "loop_detected": False,
            }
        )

    def _act(self, system: str, user: str) -> str:
        agent = self._extract_field(system, "AGENT_NAME") or "agent"
        subtask = self._extract_field(user, "SUBTASK") or ""
        action = {
            "app": agent,
            "action": self._infer_action(agent, subtask),
            "user": self._extract_field(user, "USERNAME") or "user",
            "summary": subtask[:120],
            "time": self._extract_field(user, "DATE") or "",
        }
        return json.dumps(action)

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _extract_field(text: str, name: str) -> Optional[str]:
        match = re.search(rf"{name}:\s*(.+)", text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_list(text: str, name: str) -> list[str]:
        match = re.search(rf"{name}:\s*\[(.*?)\]", text)
        if not match:
            return []
        return [item.strip().strip("'\"") for item in match.group(1).split(",") if item.strip()]

    @staticmethod
    def _infer_plan(task: Optional[str], agents: list[str]) -> list[tuple[str, str]]:
        task = (task or "").lower()
        has = lambda name: any(name in a.lower() for a in agents)
        plan: list[tuple[str, str]] = []
        if "schedule" in task or "meeting" in task:
            if has("search"):
                plan.append(("search_agent", "Find relevant thread/details for: " + task))
            if has("calendar"):
                plan.append(("calendar_agent", "Create the calendar event for: " + task))
            if has("email"):
                plan.append(("email_agent", "Send confirmation email about: " + task))
        elif "send" in task or "email" in task or "reply" in task:
            if has("search"):
                plan.append(("search_agent", "Find context needed to write the email for: " + task))
            if has("email"):
                plan.append(("email_agent", "Compose and send email for: " + task))
        elif "when is" in task or "do i have" in task or "find" in task:
            if has("search"):
                plan.append(("search_agent", "Look up the answer for: " + task))
        else:
            if has("search"):
                plan.append(("search_agent", "Investigate: " + task))
            if has("email"):
                plan.append(("email_agent", "Take appropriate email action for: " + task))
        if not plan:
            plan = [(agents[0] if agents else "search_agent", "Handle: " + task)]
        return plan

    @staticmethod
    def _infer_action(agent: str, subtask: str) -> str:
        subtask = subtask.lower()
        if "calendar" in agent:
            return "create_event"
        if "email" in agent:
            return "send_email" if "send" in subtask or "compose" in subtask or "confirmation" in subtask else "read_email"
        if "search" in agent:
            return "search"
        return "noop"


def build_llm(backend: str = "stub", **kwargs) -> LLM:
    if backend == "stub":
        return StubLLM()
    if backend == "anthropic":
        return AnthropicLLM(**kwargs)
    raise ValueError(f"Unknown LLM backend: {backend!r}")
