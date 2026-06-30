"""Core enums and data structures shared across the memory lifecycle.

These mirror the columns of the lifecycle table the system is built from:
phases, memory types, task patterns, and the data objects that flow between
stores. Keeping them centralized lets the router (memory_manager/router.py)
make decisions purely in terms of these symbols, independent of any single
store's implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class Phase(Enum):
    """The 10 lifecycle phases from the agentic memory lifecycle table.

    Order matters: it is the sequence a single task moves through, and the
    router keeps a distinct rule table per phase (see router.PHASE_POLICY).
    """

    BOOTSTRAP = "system_bootstrap"
    INGESTION = "task_ingestion"
    RETRIEVAL = "memory_retrieval"
    ORCH_INFERENCE = "orchestrator_inference"
    ORCH_OUTPUT = "orchestrator_output"
    AGENT_INFERENCE = "agent_inference"
    AGENT_OUTPUT = "agent_output"
    OBSERVATION = "environment_observation"
    STORAGE = "memory_storage"
    CONSOLIDATION = "memory_consolidation"


class MemoryType(Enum):
    """The seven memory types the lifecycle table routes data through.

    EXT and NONE are not "memory" in the cognitive sense (the table calls EXT
    "a carpenter's table, not the carpenter's memory"), but the router has to
    know about them to correctly route data *out* of the memory system.
    """

    PM = "procedural"     # static rules / prompts -- permanent
    WM = "working"        # per-task volatile scratchpad -- dies at task end
    STM = "short_term"    # session cache of distilled bundles -- fast path
    EM = "episodic"       # specific past experiences -- FAISS + disk
    SM = "semantic"       # meaning vectors -- FAISS + disk
    ENT = "entity"        # facts about users/things -- session-persistent
    EXT = "external"      # files written to the environment, not memory
    NONE = "none"         # ephemeral, consumed once, never stored


class Operation(Enum):
    WRITE = "write"
    READ = "read"
    LOAD = "load"
    CONSUME = "consume"
    LOG = "log"
    COMPARE = "compare"
    SKIP = "skip"
    CLEAR = "clear"


class Pattern(Enum):
    """Task-pattern classification (A-E), assigned at ingestion time.

    This single upfront classification is what the cascading router uses to
    decide, before any expensive retrieval happens, which memory types are
    even worth consulting for a given task. Definitions chosen to map onto
    the personal-assistant use case (environment/tasks.py):

      A - LOOKUP        : read-only question answerable from known facts/state
                          (e.g. "when is my next meeting with Dana?")
      B - SINGLE_ACTION : one agent, one side-effect
                          (e.g. "send Marco the Q3 deck")
      C - COORDINATION  : multiple agents must hand off shared context
                          (e.g. "read the thread and schedule the meeting,
                          then email the confirmation")
      D - RECURRING     : matches a previously-seen template/pattern closely
                          (e.g. "send the weekly status update")
      E - EXPLORATORY   : novel/ambiguous, best served by similarity search
                          over past episodes rather than rules
                          (e.g. "deal with the mess in my inbox from Acme")
    """

    LOOKUP = "A"
    SINGLE_ACTION = "B"
    COORDINATION = "C"
    RECURRING = "D"
    EXPLORATORY = "E"

    @classmethod
    def from_code(cls, code: str) -> "Pattern":
        for member in cls:
            if member.value == code:
                return member
        raise ValueError(f"Unknown pattern code: {code!r}")


# ---------------------------------------------------------------------------
# Data objects that flow between phases / stores
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One row of the per-task step history kept in Working Memory."""

    step: int
    agent: str
    action: str
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class TaskContext:
    """Working Memory (WM): the volatile per-task scratchpad.

    Everything here is created at Task Ingestion and is wiped (all fields set
    to None / containers cleared) at Memory Storage, regardless of whether the
    task succeeded -- exactly as the lifecycle table specifies ("WM: DIES NOW").
    """

    task_id: str
    description: str
    username: str
    date: str
    pattern: Optional[Pattern] = None
    step_history: list[StepRecord] = field(default_factory=list)
    shared_context: dict[str, Any] = field(default_factory=dict)
    last_action: Optional[dict[str, Any]] = None
    forced_finish: Optional[dict[str, Any]] = None
    success: Optional[bool] = None
    # Orchestrator-Output phase fields (table: "WM for ALL items"):
    pending_delegation: Optional[dict[str, Any]] = None   # {thought, agent, subtask}
    detected_agents: set[str] = field(default_factory=set)
    loop_detected: bool = False

    def clear(self) -> None:
        """Implements 'WM: CLEAR -> all fields -> None' from the storage phase."""
        self.pattern = None
        self.step_history = []
        self.shared_context = {}
        self.last_action = None
        self.forced_finish = None
        self.success = None
        self.pending_delegation = None
        self.detected_agents = set()
        self.loop_detected = False


@dataclass
class SubtaskMemory:
    """A single agent's distilled experience within one task (lives in EM)."""

    agent: str
    subtask: str
    action: str
    observation_summary: str
    outcome: str  # "success" | "failure" | "partial"


@dataclass
class FullTaskMemory:
    """An episodic record of one complete task (EM, FAISS-indexed via SM).

    Produced at Memory Storage on success and at Memory Consolidation (where
    many of these get distilled further into Procedural rules).
    """

    task_id: str
    description: str
    pattern: Pattern
    plan: list[str]
    step_signature: str             # e.g. "email:read -> calendar:create -> email:send"
    subtask_memories: list[SubtaskMemory] = field(default_factory=list)
    embedding: Optional[Any] = None  # float32[384], filled in by the embedder
    outcome: str = "success"
    timestamp: float = field(default_factory=time.time)


@dataclass
class MemoryBundle:
    """Short-Term Memory (STM) cache entry: a shortcut for a future similar task."""

    signature: str
    plan: list[str]
    subtask_memories: list[SubtaskMemory]
    embedding: Optional[Any] = None
    pattern: Optional[Pattern] = None
    hits: int = 0


@dataclass
class EntityProfile:
    """Entity Memory (ENT): durable facts about a user (merged across tasks)."""

    username: str
    facts: dict[str, Any] = field(default_factory=dict)

    def merge(self, new_facts: dict[str, Any]) -> None:
        self.facts.update(new_facts)


@dataclass
class RoutingDecision:
    """An auditable record of one router decision.

    This is the substrate the *closed-loop* extension (phase 2 of the project)
    will eventually learn from: every decision is logged with enough context
    to later ask "did consulting this store actually help the task succeed?".
    Kept here (not in router.py) so stores/manager can also construct/log them
    without importing router internals.
    """

    phase: Phase
    task_id: str
    pattern: Optional[Pattern]
    candidate_stores: list[MemoryType]
    consulted_stores: list[MemoryType]
    skipped_stores: dict[MemoryType, str]   # store -> reason for skipping
    cache_hit: bool = False
    latency_s: float = 0.0
    timestamp: float = field(default_factory=time.time)
    # Filled in later, at task-completion time, by the closed-loop hook:
    contributed_to_success: Optional[dict[str, bool]] = None
