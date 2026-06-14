"""Entity Memory (ENT): durable facts about users/things.

Per the lifecycle table: written once at bootstrap (developer-seeded profiles),
then read on retrieval ("on miss + if pattern needs"), and updated at storage
time only "success + new info only" via a merge ("Dict (merged)"). Lifetime is
"session" -- it survives across tasks within a run but is not the permanent,
never-changing kind of memory PM is.
"""
from __future__ import annotations

from typing import Optional

from ..types import EntityProfile


class EntityStore:
    def __init__(self):
        self._profiles: dict[str, EntityProfile] = {}

    # -- Bootstrap: WRITE (init) ("Developer -> Entity store") --------------

    def seed(self, username: str, facts: dict) -> None:
        self._profiles[username] = EntityProfile(username=username, facts=dict(facts))

    # -- Retrieval: READ ("on miss + if pattern needs") ---------------------

    def get(self, username: str) -> Optional[EntityProfile]:
        return self._profiles.get(username)

    def has_fact(self, username: str, key: str) -> bool:
        profile = self._profiles.get(username)
        return bool(profile and key in profile.facts)

    # -- Storage: WRITE ("success + new info only") -------------------------

    def merge(self, username: str, new_facts: dict) -> bool:
        """Merge new facts into the profile. Returns True iff anything changed
        (the lifecycle table gates this write on "new info only")."""
        if not new_facts:
            return False
        profile = self._profiles.setdefault(username, EntityProfile(username=username))
        changed = any(profile.facts.get(k) != v for k, v in new_facts.items())
        if changed:
            profile.merge(new_facts)
        return changed
