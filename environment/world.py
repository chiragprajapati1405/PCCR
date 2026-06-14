"""The simulated environment: bundles the apps, executes agent actions, and
produces the observation strings / structured data / file artifacts that
flow into Phase 8 (Environment Observation).

This is intentionally a thin, deterministic dispatcher -- its job is to give
the memory lifecycle (and especially the router) realistic, reproducible
inputs to route, not to demonstrate sophisticated NLP. Action parameters are
derived from the subtask text with simple heuristics so the same task always
produces the same trace (important for STM-cache-hit and EM-similarity
experiments to be reproducible).
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from .apps import CalendarApp, EmailApp, SearchApp

_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,})\b")
_STOPWORDS = {"Send", "Schedule", "Create", "Find", "Look", "Read", "Reply", "Email", "Compose",
              "Investigate", "Take", "Handle", "The", "Confirmation", "About", "For"}


def _guess_recipient(text: str, fallback: str = "team@company.com") -> str:
    for name in _NAME_RE.findall(text):
        if name not in _STOPWORDS:
            return f"{name.lower()}@company.com"
    return fallback


def _guess_attendees(text: str, organizer: str) -> list[str]:
    found = [n for n in _NAME_RE.findall(text) if n not in _STOPWORDS]
    return [organizer] + [f"{n.lower()}@company.com" for n in found[:2]] or [organizer]


class World:
    def __init__(self, eml_dir: Path, ics_dir: Path, answers_dir: Path,
                 seed_inbox: Optional[list[dict]] = None, seed_events: Optional[list[dict]] = None):
        self.email = EmailApp(eml_dir, seed_inbox)
        self.calendar = CalendarApp(ics_dir, seed_events)
        self.search = SearchApp(self.email, self.calendar, answers_dir)

    # -- the single entry point agents/manager call --------------------------

    def execute(self, action: dict, *, username: str, date: str, subtask: str) -> tuple[str, dict, Optional[Path]]:
        """Dispatch an action JSON to the right app.

        Returns (observation_string, structured_observation, artifact_path).
        `observation_string` matches the table's spec: "OBSERVATION:..." in
        the 100-1500 char range; `structured_observation` is the dict that
        gets fanned out via shared_context for other agents to consume.
        """
        app = (action.get("app") or "").lower()
        kind = (action.get("action") or "noop").lower()

        if "email" in app:
            return self._run_email(kind, username, date, subtask)
        if "calendar" in app:
            return self._run_calendar(kind, username, date, subtask)
        if "search" in app:
            return self._run_search(subtask)
        return f"OBSERVATION: no-op for app={app!r} action={kind!r}", {}, None

    # -- per-app handlers -----------------------------------------------------

    def _run_email(self, kind: str, username: str, date: str, subtask: str) -> tuple[str, dict, Optional[Path]]:
        if kind == "send_email":
            to = _guess_recipient(subtask)
            subject = " ".join(subtask.split()[:8]).rstrip(":") or "Update"
            body = f"Hi,\n\n{subtask}\n\nBest,\n{username}"
            path, record = self.email.send(to=to, subject=subject, body=body, sender=f"{username}@company.com", date=date)
            obs = f"OBSERVATION: sent email to {to} with subject '{subject}'. Saved to {path.name}."
            return obs, {"sent": record}, path

        hits = self.email.search(subtask)
        if hits:
            preview = "; ".join(f"[{m['sender']}] {m['subject']}" for m in hits[:3])
            obs = f"OBSERVATION: found {len(hits)} relevant email(s): {preview}"
        else:
            obs = "OBSERVATION: no relevant emails found in inbox."
        return obs, {"emails": hits[:3]}, None

    def _run_calendar(self, kind: str, username: str, date: str, subtask: str) -> tuple[str, dict, Optional[Path]]:
        if kind == "create_event":
            title = " ".join(subtask.split()[:6]).rstrip(":") or "Meeting"
            start_dt = datetime.strptime(date, "%Y-%m-%d") + timedelta(days=1, hours=14)
            end_dt = start_dt + timedelta(minutes=30)
            attendees = _guess_attendees(subtask, organizer=f"{username}@company.com")
            path, record = self.calendar.create_event(
                title=title,
                start=start_dt.strftime("%Y%m%dT%H%M%SZ"),
                end=end_dt.strftime("%Y%m%dT%H%M%SZ"),
                attendees=attendees,
                organizer=f"{username}@company.com",
            )
            obs = f"OBSERVATION: created calendar event '{title}' with {attendees}. Saved to {path.name}."
            return obs, {"event": record}, path

        hits = self.calendar.search(subtask)
        if hits:
            preview = "; ".join(f"{e['title']} @ {e['start']}" for e in hits[:3])
            obs = f"OBSERVATION: found {len(hits)} matching event(s): {preview}"
        else:
            obs = "OBSERVATION: no matching events found on calendar."
        return obs, {"events": hits[:3]}, None

    def _run_search(self, subtask: str) -> tuple[str, dict, Optional[Path]]:
        results = self.search.search(subtask)
        n_emails, n_events = len(results["emails"]), len(results["events"])
        obs = f"OBSERVATION: search surfaced {n_emails} email(s) and {n_events} calendar event(s) relevant to: {subtask[:80]}"
        return obs, results, None
