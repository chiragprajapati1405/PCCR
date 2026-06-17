"""
A10 — make ENTITY memory task-critical.

In OfficeBench, recipients are named in the task and addresses are synthesized
from names, so entity memory is never load-bearing. These synthetic tasks fix
that: the recipient is referenced by a *relation* ("Bob's manager") whose
resolution lives ONLY in entity memory. If the router skips ENT, the orchestrator
cannot resolve the relation → the task fails. If ENT is consulted, the relation
is in the prompt → success. This makes ENT's measured utility genuinely high for
this pattern (and lets the A2 counterfactual prove it).

  generate()         writes task configs into OfficeBench/tasks/9-* (gitignored)
  filter_ent_tasks() returns their (tid, si)
  seed_entity(mm)    seeds mm.entity["users"] with the relations (the ONLY place
                     the relation is stored — not in the task text)
"""
import os
import json

# The relations live ONLY here (and get seeded into ENT) — never in the task text.
ENT_RELATIONS = {
    "Bob":   {"manager": "Alice", "assistant": "David"},
    "Tom":   {"manager": "Mary",  "assistant": "John"},
    "Alice": {"manager": "John",  "assistant": "Bob"},
    "Mary":  {"manager": "Alice", "assistant": "Tom"},
}
TASKS_ROOT = "OfficeBench/tasks"
LEVEL = "9"   # synthetic ENT-critical tasks use the "9-*" id space


def _tasks():
    """One task per (user, relation): email a token to that person's relation."""
    out = []
    i = 0
    for user, rels in ENT_RELATIONS.items():
        for rel, person in rels.items():
            i += 1
            token = f"ENTCRIT{i:03d}"
            out.append({
                "tid": f"{LEVEL}-{i}", "person": person, "user": user, "rel": rel, "token": token,
                "config": {
                    "username": user, "date": "2020-05-01", "weekday": "Friday", "time": "10:00 AM",
                    # NOTE: the task names the RELATION, never the person's name.
                    "task": f"Send an email to {user}'s {rel} with the subject '{token}' "
                            f"and body 'Please review {token}.'",
                    "evaluation": [
                        {"function": "evaluate_contain",
                         "args": {"doc_type": "email", "username": person, "keywords": [token]}}
                    ],
                },
            })
    return out


def generate(root=TASKS_ROOT):
    """Write the synthetic task configs to disk (idempotent)."""
    tasks = _tasks()
    for t in tasks:
        d = f"{root}/{t['tid']}/subtasks"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/0.json", "w") as f:
            json.dump(t["config"], f, indent=2)
    print(f"  [A10] generated {len(tasks)} ENT-critical tasks under {root}/{LEVEL}-*")
    return [(t["tid"], "0") for t in tasks]


def filter_ent_tasks(all_ids):
    return [(tid, si) for tid, si in all_ids if tid.startswith(f"{LEVEL}-")]


def seed_entity(mem_mgr):
    """Seed the relations into ENT — the ONLY place they exist."""
    for user, rels in ENT_RELATIONS.items():
        prof = mem_mgr.entity["users"].setdefault(user, {})
        prof.update(rels)               # e.g. {"manager": "Alice", "assistant": "David"}
    print(f"  [A10] seeded ENT relations for {list(ENT_RELATIONS)} (manager/assistant)")


if __name__ == "__main__":
    ids = generate()
    print("task ids:", ids)
