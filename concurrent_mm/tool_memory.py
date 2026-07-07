"""STEP 1 — Tool Memory (READ-ONLY, shared).

The sub-agent's store: *how to call each tool* (the schema/format/gotchas), NOT past episodes.
Keyed by app -> {action -> usage}. Static after build, so it is genuinely immutable and needs
NO locks: N x M concurrent sub-agent reads scale perfectly (zero contention). This is the
lock-free half of the read/write split.

Accessed ONLY on the sub-agent side (`MemoryManager.subagent_read_tool_memory`); the orchestrator
never touches it.
"""
from __future__ import annotations

import json
import os


class ToolMemory:
    """Immutable read-only tool manual. `get(app)` is an O(1) dict lookup, no lock."""

    def __init__(self, spec: dict):
        # spec = {app: {action: {"args": [...], "format": "...", "note": "..."}}}
        self._spec = spec
        self.reads = 0                      # instrumentation: total reads served

    @classmethod
    def load(cls, path: str) -> "ToolMemory":
        with open(path) as fh:
            return cls(json.load(fh))

    def get(self, app: str) -> dict:
        """Read the app's tool spec (menu + manual). Lock-free, immutable."""
        self.reads += 1
        return self._spec.get(app, {})

    def apps(self) -> list[str]:
        return list(self._spec.keys())

    def __len__(self) -> int:
        return sum(len(v) for v in self._spec.values())


# --- default OfficeBench tool manual (from subagent_parallel.VALID_ACTIONS/ARG_SCHEMA/ACTION_HINTS) ---
DEFAULT_SPEC = {
    "shell":    {"command": {"args": ["command"], "note": "only 'command' exists; e.g. ls /testbed/data"}},
    "excel":    {"read_file": {"args": ["file_path"]},
                 "set_cell": {"args": ["file_path", "text", "row_idx", "column_idx"], "note": "row/col 1-based ints"},
                 "delete_cell": {"args": ["file_path", "row_idx", "column_idx"]},
                 "create_new_file": {"args": ["file_path"]},
                 "convert_to_pdf": {"args": ["file_path"]}},
    "word":     {"read_file": {"args": ["file_path"]},
                 "write_to_file": {"args": ["file_path", "contents"]},
                 "create_new_file": {"args": ["file_path"]},
                 "convert_to_pdf": {"args": ["file_path"]}},
    "pdf":      {"read_file": {"args": ["pdf_file_path"]},
                 "convert_to_word": {"args": ["pdf_file_path"]},
                 "convert_to_image": {"args": ["pdf_file_path"]}},
    "ocr":      {"recognize_file": {"args": ["image_path"]}},
    "calendar": {"create_event": {"args": ["user", "summary", "time_start", "time_end"],
                                  "format": "times = 'YYYY-MM-DD HH:MM:SS' (with seconds)"},
                 "delete_event": {"args": ["user", "summary"]},
                 "list_events": {"args": ["user"]}},
    "email":    {"list_emails": {"args": ["user"]},
                 "read_email": {"args": ["user"]},
                 "send_email": {"args": ["sender", "recipient", "subject", "content"],
                                "note": "sender/recipient are person names, not addresses"}},
}


def default_tool_memory() -> ToolMemory:
    return ToolMemory(DEFAULT_SPEC)


def write_default(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(DEFAULT_SPEC, fh, indent=2)
