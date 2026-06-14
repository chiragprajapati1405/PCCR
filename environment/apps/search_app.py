"""Simulated cross-app search/QA app.

Used by the search agent to gather context from email + calendar, and by
LOOKUP-pattern tasks to produce a final written answer (-> answer.txt, the
EXTERNAL "Answer file" artifact named in the lifecycle table)."""
from __future__ import annotations

from pathlib import Path

from .calendar_app import CalendarApp
from .email_app import EmailApp


class SearchApp:
    def __init__(self, email_app: EmailApp, calendar_app: CalendarApp, answers_dir: Path):
        self._email = email_app
        self._calendar = calendar_app
        self._dir = Path(answers_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._answer_count = 0

    def search(self, query: str) -> dict:
        return {
            "emails": self._email.search(query)[:3],
            "events": self._calendar.search(query)[:3],
        }

    def write_answer(self, question: str, answer: str) -> tuple[Path, dict]:
        self._answer_count += 1
        path = self._dir / f"answer_{self._answer_count:03d}.txt"
        path.write_text(f"Q: {question}\nA: {answer}\n")
        return path, {"question": question, "answer": answer, "file": str(path)}
