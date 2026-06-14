"""Simulated email app: an inbox the agent can search/read, plus the ability
to send mail, which writes a real .eml (RFC 2822) file -- the lifecycle
table's "EXTERNAL (environment, NOT memory)" artifact."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from email.message import EmailMessage
from email.utils import format_datetime
from pathlib import Path
from typing import Optional


def _rfc2822(date_str: str) -> str:
    """Accepts a plain 'YYYY-MM-DD' and renders a valid RFC 2822 Date header."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        dt = datetime.now(timezone.utc)
    return format_datetime(dt)


class EmailApp:
    def __init__(self, eml_dir: Path, seed_inbox: Optional[list[dict]] = None):
        self._dir = Path(eml_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Each message: {id, thread_id, sender, subject, body, date}
        self.inbox: list[dict] = list(seed_inbox or [])
        self._sent_count = 0

    # -- read-side: structured observations (Phase 8 input) ------------------

    def search(self, query: str) -> list[dict]:
        terms = [t for t in query.lower().split() if len(t) > 2]
        hits = []
        for msg in self.inbox:
            haystack = f"{msg['subject']} {msg['body']} {msg['sender']}".lower()
            if any(term in haystack for term in terms):
                hits.append(msg)
        return hits or self.inbox[:3]

    def read_thread(self, thread_id: str) -> list[dict]:
        return [m for m in self.inbox if m.get("thread_id") == thread_id]

    # -- write-side: produces a real .eml artifact (EXT, not memory) ---------

    def send(self, to: str, subject: str, body: str, sender: str, date: str) -> tuple[Path, dict]:
        msg = EmailMessage()
        msg["From"] = sender
        msg["To"] = to
        msg["Subject"] = subject
        msg["Date"] = _rfc2822(date)
        msg.set_content(body)

        self._sent_count += 1
        path = self._dir / f"sent_{self._sent_count:03d}_{int(time.time())}.eml"
        path.write_bytes(bytes(msg))

        record = {"to": to, "subject": subject, "body_preview": body[:140], "file": str(path)}
        return path, record
