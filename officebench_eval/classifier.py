"""LLM task classifier for 9-app OfficeBench.

The rho-gate conditions utility on a task PATTERN. With 9 apps the old
keyword patterns don't transfer, so we classify each task into a stable enum via
the LLM. The enum is fixed (so U(pattern, store) is measurable counterfactually);
the LLM does the mapping. Adds 1 cheap call per task; report classifier accuracy
as a caveat.
"""
from __future__ import annotations

from .cerebras_llm import CerebrasLLM

# 5-pattern enum (merged from an initial 7 after a 300-task distribution analysis:
# file_ops folded into single_action; doc_create+doc_extract -> doc_process; the
# sparse buckets were too thin to calibrate utility on). multi_app is kept coarse
# on purpose -- split only if utility calibration (T0c) shows it is non-uniform.
PATTERNS = [
    "lookup",         # read/find/answer info, no side effects        (low memory utility)
    "single_action",  # one app: a single create/edit or file op
    "data_compute",   # spreadsheet calculation / aggregation
    "doc_process",    # produce OR extract/convert a document/file (word/excel/pdf/image/ocr)
    "multi_app",      # multi-step pipeline across apps                (high memory utility)
]

_SYS = (
    "You classify an office-automation task into exactly ONE category. "
    "Reply with ONLY the category word, nothing else.\n"
    "Categories:\n"
    "- lookup: read/find/answer information, no changes made\n"
    "- single_action: one app, a single create/edit or file operation (add a calendar event, "
    "send an email, set a cell, move/delete a file)\n"
    "- data_compute: spreadsheet calculation or aggregation\n"
    "- doc_process: produce OR extract/convert a document or file (write a Word/PDF/Excel/image; "
    "OCR an image; convert PDF to text)\n"
    "- multi_app: a multi-step pipeline across several apps (e.g. extract from a file, produce a "
    "document, then email or schedule)"
)

# remap from the original 7-label run -> the 5-pattern enum (free, deterministic)
MERGE = {"file_ops": "single_action", "doc_create": "doc_process", "doc_extract": "doc_process"}


class TaskClassifier:
    def __init__(self, model: str = "gpt-oss-120b"):
        self.llm = CerebrasLLM(model_name=model, system_message=_SYS)

    def classify(self, task: str) -> str:
        out = (self.llm.generate(f"Task: {task}\nCategory:") or "").strip().lower()
        for p in PATTERNS:                 # tolerant parse: first enum word present
            if p in out:
                return p
        return "multi_app"                 # safe default (assume it may need memory)
