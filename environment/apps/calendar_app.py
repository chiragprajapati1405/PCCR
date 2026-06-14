"""Simulated calendar app: a list of existing events the agent can query,
plus the ability to create an event, which writes a real .ics (VEVENT) file."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional

_ICS_TEMPLATE = """BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//AgenticMM//MemoryManagerDemo//EN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{stamp}
DTSTART:{start}
DTEND:{end}
SUMMARY:{summary}
ATTENDEE:{attendee}
ORGANIZER:{organizer}
END:VEVENT
END:VCALENDAR
"""


class CalendarApp:
    def __init__(self, ics_dir: Path, seed_events: Optional[list[dict]] = None):
        self._dir = Path(ics_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        # Each event: {id, title, start, end, attendees}
        self.events: list[dict] = list(seed_events or [])

    # -- read-side ------------------------------------------------------------

    def search(self, query: str) -> list[dict]:
        terms = [t for t in query.lower().split() if len(t) > 2]
        hits = []
        for ev in self.events:
            haystack = f"{ev['title']} {' '.join(ev.get('attendees', []))}".lower()
            if any(term in haystack for term in terms):
                hits.append(ev)
        return hits or self.events[:3]

    # -- write-side: produces a real .ics artifact (EXT, not memory) ---------

    def create_event(self, title: str, start: str, end: str, attendees: list[str], organizer: str) -> tuple[Path, dict]:
        uid = uuid.uuid4().hex
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        ics = _ICS_TEMPLATE.format(
            uid=uid, stamp=stamp, start=start, end=end, summary=title,
            attendee=", ".join(attendees), organizer=organizer,
        )
        path = self._dir / f"event_{uid[:8]}.ics"
        path.write_text(ics)

        event = {"id": uid, "title": title, "start": start, "end": end, "attendees": list(attendees)}
        self.events.append(event)
        record = {"title": title, "start": start, "end": end, "attendees": attendees, "file": str(path)}
        return path, record
