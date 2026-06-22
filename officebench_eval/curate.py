"""LegoMem-style memory curation (option D) using LegoMem's EXACT prompt.

The prompt below is reproduced verbatim from the LEGOMem paper (Appendix A,
"Memory Curation Prompt"); the rendered guillemets are restored to <think>/
<action>/<...> angle-bracket tags, and {start_tag}/{end_tag} -> <START>/<END>.
It distills a successful trajectory into structured procedural memory
(high_level_plan + subtasks + reflections), which we bank for reuse.
(Backbone here is gpt-oss-120b; LegoMem used GPT-4o.)

  source cerebras.env
  python -m officebench_eval.curate
"""
from __future__ import annotations

import argparse
import json
import os
import re

from .cerebras_llm import CerebrasLLM

_PKG = os.path.dirname(os.path.abspath(__file__))
BANK = os.path.join(_PKG, "em_bank.json")
RAW_BACKUP = os.path.join(_PKG, "em_bank_raw.json")

# ---- LEGOMem Memory Curation Prompt (verbatim, Appendix A) ---------------------
LEGOMEM_CURATION_PROMPT = r"""From the following agent trajectory, generate memory that can be useful for future LLM agents' reference.
# Trajectory:
{full_trajectory}
# Example:
<START>
{
  "high_level_plan": "1. Check Bob's calendar availability for the specified time slot. 2. Add the meeting to Bob's calendar for 5172024 from 10:30 a.m. to 11:00 a.m.",
  "subtasks": [
    {
      "agent": "calendar_agent",
      "description": "Check Bob's schedule on 5/17/2024 from 10:30 a.m. to 11:00 a.m to ensure there are no conflicts",
      "steps": "<think>I need to check Bob's existing calendar events to ensure no scheduling conflicts</think><action>{\"app\":\"calendar\",\"action\":\"list_events\",\"username\":\"Bob\"}</action>",
      "observations": "No events found for Bob - calendar is available for the requested time slot"
    },
    {
      "agent": "calendar_agent",
      "description": "Add a meeting to Bob's calendar on 5/17/2024 from 10:30 a.m. to 11:00 a.m",
      "steps": "<think>Since no conflicts were found, I can now create the new calendar event for Bob</think><action>{\"app\":\"calendar\",\"action\":\"create_event\",\"user\":\"Bob\",\"summary\":\"Meeting\",\"time_start\":\"2024-05-17 10:30:00\",\"time_end\":\"2024-05-17 11:00:00\"}</action>",
      "observations": "Successfully created a new event in Bob's calendar for the specified date and time"
    }
  ],
  "final_answer": "The meeting has been successfully added to Bob's calendar on 5172024 from 10:30 a.m. to 11:00 a.m.",
  "reflections": "Task completed successfully without any conflicts or errors. The calendar check confirmed availability, and the meeting was created with proper date/time formatting."
}
<END>
# Instructions:
Please analyze the trajectory and extract structured memory with clear thinking and well-formed actions. Use the following format for each subtask step:
<think>reasoning about what needs to be done and why this action is appropriate</think>
<action>precise tool call command in structured format</action>
The memory object should be formatted as follows:
{
  "high_level_plan": "<a string that lists the high-level steps taken and which agent performs each subtask>",
  "subtasks": [
    {
      "agent": "<copy the exact name of agent that performed the subtask>",
      "description": "<description of the subtask given by the orchestrator>",
      "steps": "<Copy the precise actions taken with think-action structure: <think>reasoning</think><action>tool_call</action>, repeat for each action. Omit some actions if there are too many similar commands (>10). Remove actions that yielded errors or were malformed.>",
      "observations": "<a very brief summary of the key observations from the function execution results>"
    },
    ...
  ],
  "final_answer": "<The final answer given by the orchestrator or answer agent>",
  "reflections": "<a concise summary that lists what was successful, what were specific failures, root cause of which action and how to avoid, if any>"
}
# Rules to follow:
1. Group together actions into subtasks if they are related and can be done together.
2. For each action in the steps field, use the think-action format with clear reasoning followed by structured tool calls.
3. When copying actions, remove function call IDs but keep the essential tool call structure.
4. Only include successful actions; omit actions that resulted in errors. If there are too many repeated similar actions, truncate and omit some, and if the action parameters (such as contents to write to a word document) are too long, you can summarize it.
5. Keep observations very concise but informative.
6. Do not include orchestrator coordination steps in the subtasks.
7. For the subtask steps field, use a string format with think-action pairs, not a list. Follow the JSON format exactly to ensure it can be parsed automatically, and put the json object between the tags <START> # your json here <END> and do not use markdown.
"""


def _trajectory_text(rec):
    return "\n".join(f"{s['action']} -> {(s.get('obs') or '')[:100]}" for s in rec.get("steps", []))


def _extract(out):
    m = re.search(r"<START>(.*?)<END>", out, re.S)
    blob = m.group(1) if m else out
    return json.loads(blob[blob.find("{"): blob.rfind("}") + 1])


def curate_one(llm, rec):
    prompt = LEGOMEM_CURATION_PROMPT.replace("{full_trajectory}", _trajectory_text(rec))
    out = llm.generate(prompt)
    try:
        d = _extract(out)
        if d.get("high_level_plan"):
            return d
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-120b")
    args = ap.parse_args()
    llm = CerebrasLLM(model_name=args.model); llm.max_tokens = 2048  # curation output is long
    bank = json.load(open(BANK))
    if not os.path.exists(RAW_BACKUP):
        json.dump(bank, open(RAW_BACKUP, "w"), indent=2)

    n = 0
    for i, rec in enumerate(bank):
        if rec.get("curated"):
            continue
        d = curate_one(llm, rec)
        if d:
            hlp = str(d["high_level_plan"])
            rec["raw_plan"] = rec.get("raw_plan", rec["plan"])
            rec["plan"] = [s.strip() for s in re.split(r"(?:^|\s)\d+\.\s*", hlp) if s.strip()] or [hlp]
            rec["curated_subtasks"] = d.get("subtasks", [])
            rec["reflections"] = d.get("reflections", "")
            rec["curated"] = True
            n += 1
        json.dump(bank, open(BANK, "w"), indent=2)
        print(f"  [{i+1}/{len(bank)}] {rec['task'][:50]} -> "
              f"{(str(len(rec['plan']))+' steps') if rec.get('curated') else 'FAIL(keep raw)'}",
              flush=True)
    print(f"\ncurated {n}/{len(bank)} (LegoMem verbatim prompt) | raw backup -> {RAW_BACKUP}")


if __name__ == "__main__":
    main()
