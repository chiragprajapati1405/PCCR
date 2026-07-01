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

# EXACT required args per action (from each app's construct_action) so the sub-agent emits a
# runnable call in one shot -- this is what makes the leaf sub-agent actually COMPLETE its subtask.
ARG_SCHEMA = {
    "shell.command": ["command"],
    "excel.read_file": ["file_path"],
    "excel.set_cell": ["file_path", "text", "row_idx", "column_idx"],
    "excel.delete_cell": ["file_path", "row_idx", "column_idx"],
    "excel.create_new_file": ["file_path"],
    "excel.convert_to_pdf": ["file_path"],
    "word.read_file": ["file_path"],
    "word.write_to_file": ["file_path", "contents"],
    "word.create_new_file": ["file_path"],
    "word.convert_to_pdf": ["file_path"],
    "pdf.read_file": ["pdf_file_path"],
    "pdf.convert_to_word": ["pdf_file_path"],
    "pdf.convert_to_image": ["pdf_file_path"],
    "ocr.recognize_file": ["image_path"],
    "calendar.create_event": ["user", "summary", "time_start", "time_end"],
    "calendar.delete_event": ["user", "summary"],
    "calendar.list_events": ["user"],
    "email.list_emails": ["user"],
    "email.read_email": ["user"],
    "email.send_email": ["sender", "recipient", "subject", "content"],
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


def plan_delegations(task_text, llm, files_hint=""):
    """ONE LLM call: decompose the task into a JSON list of independent {app, subtask}
    delegations, expanding fan-outs so parallelisable work is exposed. `files_hint` is the REAL
    directory listing so the planner never invents a filename. Returns list[Delegation]."""
    prompt = (
        "Decompose this office-automation task into the SMALLEST independent sub-tasks. "
        "Return ONLY a JSON array of objects {\"app\": <one of shell,excel,word,pdf,ocr,calendar,"
        "email>, \"subtask\": <one concrete instruction>}. RULES: (1) if the task repeats an "
        "operation over N items (each member/person/row/file), emit ONE delegation PER item so "
        "they run in parallel; (2) a sub-task that READS/lists source data MUST come before the "
        "sub-tasks that use it; (3) keep each sub-task to a single app; (4) use ONLY the EXACT "
        "filenames listed below --- never invent a filename.\n"
        + (("Available files: %s\n" % files_hint) if files_hint else "")
        + "\nTASK: " + str(task_text) + "\n\nJSON:")
    dels = _parse_json_block(llm.generate(prompt)) or []
    out = []
    for d in dels:
        app = str(d.get("app", "")).lower().strip()
        sub = str(d.get("subtask", "")).strip()
        if app in VALID_ACTIONS and sub:
            out.append(Delegation(agent_type=app, subtask=sub))
    return out


# common arg-name mistakes the model makes -> the real key each app expects
_ARG_ALIAS = {"username": "user", "to": "recipient", "from": "sender", "body": "content",
              "message": "content", "title": "summary", "start": "time_start", "end": "time_end",
              "start_time": "time_start", "end_time": "time_end", "path": "file_path",
              "filename": "file_path", "content": "contents", "text_content": "contents"}


def _exec_direct(env, app, action):
    """Execute ONE action via docker exec on the SHARED container -- no shared env state, so it
    is safe to call concurrently for different sub-agents in a wave."""
    import apps
    try:
        act = action.get("action")
        if act not in VALID_ACTIONS.get(app, []):
            return "Malformed action! Unknown action '%s' for app '%s'." % (act, app)
        payload = {}
        need = ARG_SCHEMA.get("%s.%s" % (app, act), [])
        for k, v in action.items():
            payload[_ARG_ALIAS.get(k, k) if _ARG_ALIAS.get(k, k) in need else k] = v
        # 'contents' vs 'content' collide across apps; keep both if the action needs one
        if "contents" in need and "content" in action and "contents" not in payload:
            payload["contents"] = action["content"]
        payload["action"] = act
        payload["app"] = app
        mod = apps.AVAILABLE_ACTIONS[app][act]
        command = mod.construct_action(env.workdir, args=payload)
        cmd = env.clean_cmd(command)
        exit_code, output = env.container.exec_run(cmd, workdir=env.workdir)
        return output.decode("utf-8").split("OBSERVATION:")[-1].strip()
    except Exception as e:
        return "Malformed action! (%s)" % str(e)[:80]


# app-specific gotchas the sub-agent must get right for the tool to actually SUCCEED
ACTION_HINTS = {
    "calendar.create_event": "time_start/time_end MUST be '%Y-%m-%d %H:%M:%S' (with seconds)",
    "excel.set_cell": "row_idx/column_idx are 1-based integers",
    "email.send_email": "sender+recipient are person names, not addresses",
}


def _arg_hint(app, actions):
    out = []
    for a in actions:
        key = "%s.%s" % (app, a)
        s = "%s(%s)" % (a, ", ".join(ARG_SCHEMA.get(key, [])))
        if key in ACTION_HINTS:
            s += " [%s]" % ACTION_HINTS[key]
        out.append(s)
    return "; ".join(out)


def _subagent_prompt(app, subtask, blackboard, last_obs, fanout=False):
    """A SMALL focused prompt -- the whole point of the 2nd dimension: the sub-agent sees ONLY its
    subtask, the exact arg schema, and a COMPACT blackboard of prior-wave results (e.g. the data a
    read-agent extracted), NOT the full growing task history. That is what cuts real tokens."""
    fan = ("This is a REPEATED sub-task: perform it for EVERY item in the shared context "
           "(each participant/person/row). Emit ONE action for the NEXT not-yet-done item; reply "
           "{\"action\": \"done\"} ONLY after ALL items are handled.\n") if fanout else ""
    return (
        "You are the %s sub-agent in a multi-agent office system. Do ONLY your sub-task.\n"
        "SUB-TASK: %s\n%s"
        "Actions you may use (with EXACT args): %s\n"
        "%s"
        "Files are under /testbed/data. Reply with ONE JSON tool call {\"action\": <name>, <args>}. "
        "If the sub-task is fully done, reply {\"action\": \"done\"}.\n%sJSON:" % (
            app, subtask, fan, _arg_hint(app, VALID_ACTIONS[app]),
            ("SHARED CONTEXT (results from earlier agents):\n%s\n" % blackboard[:600]) if blackboard else "",
            ("Result of your last action: %s\n" % last_obs[:250]) if last_obs else ""))


def _run_subagent_blocking(env, llm, app, subtask, blackboard, max_steps):
    """A short schema-guided LLM->action loop for one sub-agent against the shared container.
    Runs up to max_steps (a 'create for each X' leaf may need a few), stops on 'done' or when it
    stops producing valid actions. Returns the agent result + its observations for the blackboard."""
    obs, last_action, steps, log = "", {}, 0, []
    s0 = subtask.lower()
    is_fanout = any(k in s0 for k in ("each", "all", "every", "participant", "member")) or "{" in subtask
    for _ in range(max_steps):
        action = _parse_action(llm.generate(_subagent_prompt(app, subtask, blackboard, obs, is_fanout)))
        if not action or str(action.get("action")).lower() in ("done", "finish", ""):
            break
        last_action = action
        steps += 1
        obs = _exec_direct(env, app, action)
        log.append(obs[:160])
        ok = "Malformed" not in obs and "Fail" not in obs and "does not exist" not in obs
        s = subtask.lower()
        fanout = any(k in s for k in ("each", "all", "every", "participant", "member")) or "{" in subtask
        if ok and not fanout:
            break                              # single-item leaf completes in one good action
        if fanout:
            # multi-item leaf: keep creating (Bob, Carol, ...) until the agent says done or repeats.
            done_signal = str(action.get("action", "")).lower() == "done"
            if done_signal or (len(log) >= 2 and log[-1] == log[-2]):
                break
    return AgentResult(agent_type=app, subtask=subtask, action=last_action,
                       observation=" | ".join(log)[:400]), steps


def make_subagent_runner(env, llm, max_steps, step_counter, blackboard_ref):
    """Real AgentRunner for ParallelExecutor: runs the blocking sub-agent in a worker thread so
    the wave's agents overlap on Docker-exec + LLM waits. Reads the shared blackboard (prior-wave
    results) so create/send agents know the data a read-agent extracted."""
    async def runner(agent_type, subtask, wm_snapshot):
        res, steps = await asyncio.to_thread(
            _run_subagent_blocking, env, llm, agent_type, subtask, blackboard_ref[0], max_steps)
        step_counter[0] += steps
        return res
    return runner


async def run_task_2d(task_text, env, llm, max_steps=8):
    """Orchestrate the 2nd dimension: plan -> waves -> per-wave concurrent execution, threading a
    compact blackboard (prior-wave observations) forward so later agents have the data they need.
    Returns (results, waves, delegations) or (None,...) if not parallelisable (fall back to 1D)."""
    # gpt-oss is a REASONING model: it emits chain-of-thought BEFORE the answer, so a small
    # max_tokens is consumed by reasoning and returns EMPTY content -> generate() retries every key
    # (self-inflicted 429 storm). Give it room.
    if getattr(llm, "max_tokens", 0) < 4096:
        llm.max_tokens = 4096
    # REAL directory listing first -> feed it to BOTH the planner (no invented filenames) and the
    # blackboard (sub-agents see the true files).
    try:
        _ec, _out = env.container.exec_run("ls /testbed/data", workdir=env.workdir)
        files = _out.decode().replace("\n", ", ").strip()
    except Exception:
        files = ""
    seed = ("[files] /testbed/data contains: %s\n" % files) if files else ""
    dels = plan_delegations(task_text, llm, files_hint=files)
    if len(dels) < 2:
        return None, [], dels
    waves = DependencyAnalyzer().analyze(dels)
    step_counter, blackboard_ref = [0], [seed]
    executor = ParallelExecutor(make_subagent_runner(env, llm, max_steps, step_counter, blackboard_ref))
    results = []
    for wave in waves:
        wave_results = await executor.execute_group(wave, wm_snapshot=None)
        results.extend(wave_results)
        # blackboard: append this wave's observations so the NEXT wave sees the data (e.g. the
        # participant list a read-agent produced) -- compact, not the full history.
        blackboard_ref[0] += "\n".join("[%s] %s" % (r.agent_type, r.observation) for r in wave_results) + "\n"
    return results, waves, dels
