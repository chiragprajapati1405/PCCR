"""2nd dimension of concurrency: parallel SUB-AGENTS within a single OfficeBench task.

The 1st dimension (run_t1_parallel.py) runs parallel TASKS. This runs parallel SUB-AGENTS
*inside one task*: an LLM planner decomposes the task into independent sub-delegations
(app, subtask) -- fan-outs like "create an event for EACH member" become one delegation per
member -- the DependencyAnalyzer groups them into precedence waves (read -> create -> send),
and each wave's sub-agents run CONCURRENTLY against the SHARED Docker container.

Why this is safe / correct:
  * OfficeBench's tool execution (container.exec_run of construct_action(...)) is STATELESS per
    action -- each call is an independent process in the container. The only per-agent state
    (current_app, last observation) is Python-side, so each sub-agent keeps its own; the shared
    /testbed/data is the point (multi-app tasks need it).
  * The dependency waves guarantee only INDEPENDENT subtasks run together (same category), so
    concurrent exec_run calls never touch the same output file within a wave.
  * Reuses memory_manager.parallel: DependencyAnalyzer (L3) + ParallelExecutor (L4, snapshot->
    gather->merge). The rho-gate router and lifecycle are untouched.
"""
from __future__ import annotations

import asyncio
import json
import re

from memory_manager.parallel import AgentResult, Delegation, DependencyAnalyzer, ParallelExecutor

# valid (app, action) schema -- reused so the sub-agent prompt is grounded (same set as F1)
VALID_ACTIONS = {
    "shell": ["command"],
    "excel": ["read_file", "set_cell", "delete_cell", "create_new_file", "convert_to_pdf"],
    "word": ["read_file", "write_to_file", "create_new_file", "convert_to_pdf"],
    "pdf": ["read_file", "convert_to_word", "convert_to_image"],
    "ocr": ["recognize_file"],
    "calendar": ["create_event", "delete_event", "list_events"],
    "email": ["list_emails", "read_email", "send_email"],
}


def _parse_json_block(text):
    m = re.search(r"\[.*\]", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _parse_action(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def plan_delegations(task_text, llm):
    """ONE LLM call: decompose the task into a JSON list of independent {app, subtask}
    delegations, expanding fan-outs so parallelisable work is exposed. Returns
    list[Delegation] (empty -> caller falls back to the sequential loop)."""
    prompt = (
        "Decompose this office-automation task into the SMALLEST independent sub-tasks. "
        "Return ONLY a JSON array of objects {\"app\": <one of shell,excel,word,pdf,ocr,calendar,"
        "email>, \"subtask\": <one concrete instruction>}. RULES: (1) if the task repeats an "
        "operation over N items (each member/person/row/file), emit ONE delegation PER item so "
        "they can run in parallel; (2) a sub-task that only READS/lists goes first; (3) keep each "
        "sub-task to a single app.\n\nTASK: " + str(task_text) + "\n\nJSON:")
    dels = _parse_json_block(llm.generate(prompt)) or []
    out = []
    for d in dels:
        app = str(d.get("app", "")).lower().strip()
        sub = str(d.get("subtask", "")).strip()
        if app in VALID_ACTIONS and sub:
            out.append(Delegation(agent_type=app, subtask=sub))
    return out


def _exec_direct(env, app, action):
    """Execute ONE action via docker exec on the SHARED container -- no shared env state, so it
    is safe to call concurrently for different sub-agents in a wave."""
    import apps
    try:
        act = action.get("action")
        if act not in VALID_ACTIONS.get(app, []):
            return "Malformed action! Unknown action '%s' for app '%s'." % (act, app)
        payload = dict(action)
        payload["app"] = app
        mod = apps.AVAILABLE_ACTIONS[app][act]
        command = mod.construct_action(env.workdir, args=payload)
        cmd = env.clean_cmd(command)
        exit_code, output = env.container.exec_run(cmd, workdir=env.workdir)
        return output.decode("utf-8").split("OBSERVATION:")[-1].strip()
    except Exception as e:
        return "Malformed action! (%s)" % str(e)[:80]


def _subagent_prompt(app, subtask, last_obs):
    return (
        "You are the %s sub-agent. Do ONLY this sub-task, then stop.\n"
        "Sub-task: %s\n"
        "Valid actions for %s: %s\n"
        "Files live under /testbed/data. Reply with ONE JSON action "
        "{\"action\": <name>, ...args}. When the sub-task is complete, reply "
        "{\"action\": \"done\"}.\n%s\nJSON action:" % (
            app, subtask, app, ", ".join(VALID_ACTIONS[app]),
            ("Last result: " + last_obs[:200]) if last_obs else ""))


def _run_subagent_blocking(env, llm, app, subtask, max_steps):
    """A short LLM->action loop for one sub-agent against the shared container."""
    obs, last_action, steps = "", {}, 0
    for _ in range(max_steps):
        action = _parse_action(llm.generate(_subagent_prompt(app, subtask, obs)))
        if not action or action.get("action") == "done":
            break
        last_action = action
        steps += 1
        obs = _exec_direct(env, app, action)
        if "Malformed" not in obs and "shell" != app:
            break  # single productive action usually completes a leaf sub-task
    return AgentResult(agent_type=app, subtask=subtask, action=last_action, observation=obs), steps


def make_subagent_runner(env, llm, max_steps, step_counter):
    """Real AgentRunner for ParallelExecutor: runs the blocking sub-agent in a worker thread so
    the wave's agents overlap on their Docker-exec + LLM-HTTP waits (asyncio.gather)."""
    async def runner(agent_type, subtask, wm_snapshot):
        res, steps = await asyncio.to_thread(_run_subagent_blocking, env, llm, agent_type, subtask, max_steps)
        step_counter[0] += steps
        return res
    return runner


async def run_task_2d(task_text, env, llm, max_steps=6):
    """Orchestrate the 2nd dimension: plan -> waves -> per-wave concurrent execution.
    Returns (results, waves, planner_call) or (None,...) if planning yields nothing
    parallelisable (caller should fall back to the sequential loop)."""
    # gpt-oss is a REASONING model: it emits a chain-of-thought BEFORE the answer, so a small
    # max_tokens is consumed entirely by reasoning and returns EMPTY content -> generate() treats
    # that as failure and retries every key (a self-inflicted 429 storm). Give it room.
    if getattr(llm, "max_tokens", 0) < 4096:
        llm.max_tokens = 4096
    dels = plan_delegations(task_text, llm)
    if len(dels) < 2:
        return None, [], dels
    waves = DependencyAnalyzer().analyze(dels)
    step_counter = [0]
    executor = ParallelExecutor(make_subagent_runner(env, llm, max_steps, step_counter))
    results = []
    for wave in waves:
        results.extend(await executor.execute_group(wave, wm_snapshot=None))
    return results, waves, dels
