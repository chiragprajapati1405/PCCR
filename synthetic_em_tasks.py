"""
Make EPISODIC memory task-critical (the EM analog of synthetic_ent_tasks / A10).

Each pair = (a seed EPISODE stored in EM) + (an under-specified "as usual" TASK).
The task omits the specific detail (a unique event-name token); that detail lives
ONLY in the past episode. Without consulting EM the agent can't produce the token
→ eval fails. With EM, the episode is retrieved and the token reused → pass.

  generate()          writes the "as usual" task configs to OfficeBench/tasks/8-*
  filter_em_tasks()   returns their (tid, si)
  seed_episodes(mm)   injects the detail-bearing episodes into mm.episodic (EM)
"""
import os
import json

# Each episode carries a UNIQUE token in the event name — the reusable detail
# that exists ONLY in episodic memory.
EM_EPISODES = [
    {"user": "Bob",   "topic": "project review",    "token": "EMTOKA01", "start": "2024-05-20 09:00:00", "end": "2024-05-20 10:00:00"},
    {"user": "Tom",   "topic": "team standup",      "token": "EMTOKA02", "start": "2024-05-20 10:00:00", "end": "2024-05-20 10:30:00"},
    {"user": "Alice", "topic": "design sync",       "token": "EMTOKA03", "start": "2024-05-21 14:00:00", "end": "2024-05-21 15:00:00"},
    {"user": "Mary",  "topic": "budget review",     "token": "EMTOKA04", "start": "2024-05-21 11:00:00", "end": "2024-05-21 12:00:00"},
    {"user": "Bob",   "topic": "1:1 with mentor",   "token": "EMTOKA05", "start": "2024-05-22 16:00:00", "end": "2024-05-22 16:30:00"},
    {"user": "Tom",   "topic": "sprint planning",   "token": "EMTOKA06", "start": "2024-05-22 13:00:00", "end": "2024-05-22 14:00:00"},
    {"user": "Alice", "topic": "client check-in",   "token": "EMTOKA07", "start": "2024-05-23 15:00:00", "end": "2024-05-23 15:30:00"},
    {"user": "Mary",  "topic": "retro meeting",     "token": "EMTOKA08", "start": "2024-05-23 09:30:00", "end": "2024-05-23 10:00:00"},
]
TASKS_ROOT = "OfficeBench/tasks"
LEVEL = "8"   # EM-critical tasks use the "8-*" id space


def _event_name(e):
    return f"{e['topic']} {e['token']}"


def seed_episodes(mm):
    """Inject the detail-bearing episodes into EM so they're retrievable."""
    from mm_on_top_of_legomem import FullTaskMemory, SubtaskMemory
    for i, e in enumerate(EM_EPISODES):
        name = _event_name(e)
        plan = (f"calendar: create event '{name}' for {e['user']} "
                f"from {e['start']} to {e['end']}")
        sub = SubtaskMemory(subtask_id=f"emseed_{i}",
                            subtask_description=f"create event '{name}' for {e['user']} from {e['start']} to {e['end']}",
                            agent_type="calendar", tool_calls=["create_event"], outcome="success")
        fm = FullTaskMemory(memory_id=f"emseed_{i}",
                            task_description=f"Schedule {e['user']}'s {e['topic']} meeting",
                            high_level_plan=plan, subtask_memories=[sub], final_answer="done")
        mm.episodic.add_full_task(fm)
    print(f"  [EM] seeded {len(EM_EPISODES)} detail-bearing episodes into EM")


def generate(root=TASKS_ROOT):
    tasks = []
    for i, e in enumerate(EM_EPISODES, 1):
        tid = f"{LEVEL}-{i}"
        d = f"{root}/{tid}/subtasks"
        os.makedirs(d, exist_ok=True)
        cfg = {
            "username": e["user"], "date": "2020-05-01", "weekday": "Friday", "time": "10:00 AM",
            # under-specified: "as usual" — the event name/token is NOT in the task
            "task": f"Schedule {e['user']}'s {e['topic']} meeting, same as usual.",
            "evaluation": [
                {"function": "evaluate_contain",
                 "args": {"doc_type": "ics", "file": f"./calendar/{e['user']}.ics",
                          "keywords": [e["token"]]}}
            ],
        }
        with open(f"{d}/0.json", "w") as f:
            json.dump(cfg, f, indent=2)
        tasks.append((tid, "0"))
    print(f"  [EM] generated {len(tasks)} EM-critical tasks under {root}/{LEVEL}-*")
    return tasks


def filter_em_tasks(all_ids):
    return [(tid, si) for tid, si in all_ids if tid.startswith(f"{LEVEL}-")]


if __name__ == "__main__":
    print(generate())
