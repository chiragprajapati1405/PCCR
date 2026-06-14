"""Seed data + a task suite spanning all five patterns (A-E).

The suite is deliberately constructed so that a batch run exercises every
lifecycle phase and every routing branch at least once:
  - near-duplicate tasks (e.g. the two "weekly status update" runs) should
    produce an STM cache hit on the second occurrence;
  - LOOKUP/COORDINATION/EXPLORATORY tasks pull on different optional stores
    (ENT vs. EM vs. SM) so the cost-gated cascade visibly branches per pattern;
  - running `consolidate()` after the batch gives EM/SM/PM something to learn
    from, which a second batch can then draw on.
"""
from __future__ import annotations

USERNAME = "chirag"
TODAY = "2026-06-08"


def seed_inbox() -> list[dict]:
    return [
        {
            "id": "m1", "thread_id": "t-acme", "sender": "dana@acme.com",
            "subject": "Re: Acme Q3 sync - proposed times",
            "body": "Hi, following up on our Q3 sync -- could we lock in a 30 minute call next week? "
                    "I'm free most afternoons.",
            "date": "2026-06-05",
        },
        {
            "id": "m2", "thread_id": "t-acme", "sender": "dana@acme.com",
            "subject": "Re: Acme Q3 sync - proposed times",
            "body": "Quick nudge on this -- want to make sure we get it on the calendar before the end of the month.",
            "date": "2026-06-07",
        },
        {
            "id": "m3", "thread_id": "t-conf", "sender": "events@conference.io",
            "subject": "Your badge for DevSummit is ready",
            "body": "Thanks for registering. Your badge and schedule are attached.",
            "date": "2026-06-04",
        },
        {
            "id": "m4", "thread_id": "t-conf", "sender": "marco@partner.com",
            "subject": "Great meeting you at DevSummit",
            "body": "Loved our chat about agent memory systems -- let's continue over a call sometime.",
            "date": "2026-06-06",
        },
        {
            "id": "m5", "thread_id": "t-status", "sender": "lead@company.com",
            "subject": "Reminder: weekly status update due Friday",
            "body": "Just a reminder to send your weekly status update to the team by end of day Friday.",
            "date": "2026-06-03",
        },
    ]


def seed_events() -> list[dict]:
    return [
        {"id": "e1", "title": "1:1 with manager", "start": "20260609T150000Z", "end": "20260609T153000Z",
         "attendees": ["chirag@company.com", "lead@company.com"]},
        {"id": "e2", "title": "DevSummit talk: Agent Memory", "start": "20260604T090000Z", "end": "20260604T100000Z",
         "attendees": ["chirag@company.com"]},
    ]


def seed_entities() -> dict[str, dict]:
    return {
        USERNAME: {
            "team": "platform",
            "manager": "lead@company.com",
            "weekly_report_recipient": "team@company.com",
            "timezone": "America/Los_Angeles",
        }
    }


# Each task: (description, pattern_hint) -- pattern_hint is for humans reading
# this file / for assertions in tests; the system always classifies for itself.
DEMO_TASKS: list[dict] = [
    {  # A - LOOKUP: should lean on ENT (+ SM), rarely EM
        "description": "When is my next meeting with Dana, and do I have anything else on my calendar this week?",
        "pattern_hint": "A",
    },
    {  # B - SINGLE_ACTION: one agent, one side effect
        "description": "Send Marco a short email proposing a follow-up call about agent memory systems.",
        "pattern_hint": "B",
    },
    {  # C - COORDINATION: multi-agent handoff (search -> calendar -> email)
        "description": "Read the Acme thread with Dana, schedule the Q3 sync she proposed, and then send her a confirmation email.",
        "pattern_hint": "C",
    },
    {  # D - RECURRING: first occurrence (no cache yet)
        "description": "Send the weekly status update to the team as usual.",
        "pattern_hint": "D",
    },
    {  # D - RECURRING: near-duplicate of the previous -- should hit STM cache
        "description": "Send this week's status update to the team, same as usual.",
        "pattern_hint": "D",
    },
    {  # E - EXPLORATORY: ambiguous, leans on SM/EM similarity search
        "description": "Deal with the mess of conference follow-up emails sitting in my inbox from DevSummit.",
        "pattern_hint": "E",
    },
]


def iter_tasks():
    for spec in DEMO_TASKS:
        yield spec["description"], USERNAME, TODAY, spec["pattern_hint"]
