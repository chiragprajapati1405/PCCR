"""LLM task classifier for 9-app OfficeBench.

The rho-gate conditions utility on a task PATTERN. With 9 apps the old
keyword patterns don't transfer, so we classify each task into a stable enum via
the LLM. The enum is fixed (so U(pattern, store) is measurable counterfactually);
the LLM does the mapping. Adds 1 cheap call per task; report classifier accuracy
as a caveat.
"""
from __future__ import annotations

from .cerebras_llm import CerebrasLLM

PATTERNS = [
    "lookup",         # read/find/answer info, no side effects
    "single_action",  # one app, one create/edit (add event, send email, set a cell)
    "multi_app",      # multiple apps / multi-step coordination (read -> compute -> notify)
    "doc_create",     # produce a document or file (word/excel/pdf/image)
    "doc_extract",    # extract/convert content (pdf->text, OCR an image)
    "data_compute",   # spreadsheet calculation / aggregation
    "file_ops",       # shell/system file manipulation
]

_SYS = (
    "You classify an office-automation task into exactly ONE category. "
    "Reply with ONLY the category word, nothing else.\n"
    "Categories:\n"
    "- lookup: read/find/answer information, no changes made\n"
    "- single_action: one app, one create/edit (add a calendar event, send an email, set a cell)\n"
    "- multi_app: several apps or multi-step coordination (read then compute then notify)\n"
    "- doc_create: produce a document or file (Word/Excel/PDF/image)\n"
    "- doc_extract: extract or convert content (PDF to text, OCR an image)\n"
    "- data_compute: spreadsheet calculation or aggregation\n"
    "- file_ops: shell/system file manipulation"
)


class TaskClassifier:
    def __init__(self, model: str = "gpt-oss-120b"):
        self.llm = CerebrasLLM(model_name=model, system_message=_SYS)

    def classify(self, task: str) -> str:
        out = (self.llm.generate(f"Task: {task}\nCategory:") or "").strip().lower()
        for p in PATTERNS:                 # tolerant parse: first enum word present
            if p in out:
                return p
        return "multi_app"                 # safe default (assume it may need memory)
