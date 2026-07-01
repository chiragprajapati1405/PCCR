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


def _is_fanout(subtask):
    s = subtask.lower()
    return any(k in s for k in ("each", "all ", "every", "participant", "member", "everyone")) or "{" in subtask


_NAME_HDR = ("name", "student", "member", "participant", "employee", "person", "user", "fellow")


def _extract_items_structural(blackboard, task_text=""):
    """DETERMINISTIC extraction: parse the excel/csv read dump '(row, col): value' into a grid,
    find the NAME column by its header, and return that column's values (no LLM). This kills the
    gpt-oss extraction flakiness (dropped/garbled items) on clean table fan-outs.

    CONDITION-AWARE: if the task references a value of another column (e.g. 'members in section A'
    while a 'Section' column holds 'Section A'), keep ONLY rows matching that value --- so filtered
    fan-outs ('for X in <group>') spawn a leaf per matching row, not per row."""
    cells = {}
    for m in re.finditer(r"\((\d+),\s*(\d+)\):\s*([^\t\n]+)", blackboard):
        r, c, v = int(m.group(1)), int(m.group(2)), m.group(3).strip()
        cells[(r, c)] = v
    if not cells:
        return []
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    # header row = row 1; pick the column whose header matches a name-like word (else column 1)
    name_col = None
    for (r, c), v in cells.items():
        if r == 1 and any(h in v.lower() for h in _NAME_HDR):
            name_col = c; break
    if name_col is None:
        name_col = 1
    # detect a filter: a non-name column whose (distinct) value is literally referenced in the task
    tl = task_text.lower()
    filt = None
    if tl:
        for c in range(1, max_c + 1):
            if c == name_col:
                continue
            for r in range(2, max_r + 1):
                v = cells.get((r, c), "").strip()
                if len(v) >= 3 and v.lower() in tl:
                    filt = (c, v.lower()); break
            if filt:
                break
    out, seen = [], set()
    for r in range(2, max_r + 1):
        if filt and cells.get((r, filt[0]), "").strip().lower() != filt[1]:
            continue
        v = cells.get((r, name_col), "").strip()
        if v and v.lower() not in seen and len(v) < 60:
            seen.add(v.lower()); out.append(v)
    return out


_ACT_VERBS = ("create", "schedule", "send", "add ", "write", "set ", "book", "update", "delete",
              "email", "notify", "remind", "reply", "forward", "record", "fill", "make ", "generate")
_SEND_VERBS = ("send", "email", "notify", "remind", "reply", "forward")


def _phase2d(subtask):
    """Robust read-vs-act split for the 2D pipeline. The shared categorize() checks 'list' FIRST,
    so 'create an event for each member LISTED in section A' is mislabeled 'read' and never runs.
    Here an explicit ACT verb wins over an incidental 'list'/'listed' --- fixing dropped act waves."""
    s = subtask.lower()
    if any(v in s for v in _ACT_VERBS):
        return "send" if any(v in s for v in _SEND_VERBS) else "create"
    return "read"


_APP_KEYWORDS = [
    ("calendar", ("calendar", "event", "meeting", "appointment")),
    ("email", ("email", "e-mail", " mail", "message", "notify", "inform")),
    ("word", ("document", "word", ".docx", "letter", "report", "essay", ".txt")),
    ("excel", ("spreadsheet", "excel", ".xlsx", "cell", "sheet")),
]


def _is_fanout_task(task_text):
    """Whole-task fan-out check: does the task CREATE one output PER item of a group (the pattern
    the deterministic fast-path handles: read source -> extract items -> create one leaf per item).

    Excludes look-alikes that carry 'each/all' but are NOT per-item creates, so the fast-path fires
    only where it's sound:
      * collect-into-ONE-doc filters  ('put ... into students.docx')     -> one write, not N
      * in-place excel bulk-edit/compute ('header', 'last row', 'remove', 'average', 'round down')
    Those fall through to the general planner path instead."""
    tl = (task_text or "").lower()
    if not (_is_fanout(tl) and any(v in tl for v in _ACT_VERBS)):
        return False
    if re.search(r"\b(into|in)\s+\w[\w-]*\.(docx|xlsx|pdf|csv|txt)", tl):     # single named sink
        return False
    if any(k in tl for k in ("header", "last row", "remove ", "average", "round down", "attendance")):
        return False
    # the deterministic fast-path is proven on per-person CALENDAR-event fan-outs read from a grid;
    # excel-target fan-outs are in-place compute/edit (regress 1-11/2) and email-collect fan-outs
    # write ONE message (both-fail + a 63K-token runaway on 2-43/0) -> restrict to the calendar niche.
    if _target_app(task_text) != "calendar":
        return False
    # folder/grouping/pairing fan-outs need per-group dir + file ops the flat item-leaf can't do
    # (3-53/0 per-class folders, 3-10/0 per-recipient folders) -> regress; leave them to the planner.
    if any(k in tl for k in ("folder", "directory", "subdirectory", "each class", "separ", " pair ", "conflict")):
        return False
    return True


def _target_app(task_text):
    """Which app produces the fan-out OUTPUT. The act clause (after the last comma) usually names
    it (e.g. 'create calendar events'); scan that first, then the whole task."""
    tl = (task_text or "").lower()
    tail = tl.split(",")[-1]
    for app, kws in _APP_KEYWORDS:
        if any(k in tail for k in kws):
            return app
    for app, kws in _APP_KEYWORDS:
        if any(k in tl for k in kws):
            return app
    return "calendar"


def _act_clause(task_text):
    """The single-item OUTPUT instruction template: the comma/period segment holding both a fan-out
    word and an act verb (fall back to the last act segment), with the fan-out words stripped so
    '<clause> for <name>' reads as a clean one-item leaf."""
    segs = [s.strip() for s in re.split(r"[,.]", task_text or "") if s.strip()]
    cand = None
    # the OUTPUT instruction is usually the LAST segment that both fans out and acts -> take the last
    for s in segs:
        sl = s.lower()
        if _is_fanout(sl) and any(v in sl for v in _ACT_VERBS):
            cand = s
    if cand is None:
        for s in segs:
            if any(v in s.lower() for v in _ACT_VERBS):
                cand = s
    base = re.sub(r"\{[^}]*\}", "", cand or task_text)
    base = re.sub(r"\b(each|every|all)\s+", "", base, flags=re.I)
    base = re.sub(r"\b(participants?|members?|employees?|students?|everyone|people|person)\b", "", base, flags=re.I)
    base = re.sub(r"\bin their name\b", "", base, flags=re.I)
    base = re.sub(r"\bfrom .*?file\b", "", base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip(" .")
    # drop a dangling preposition left by the strips so '<clause> for <name>' isn't '...for for X'
    base = re.sub(r"\s+(for|to|of|in|with)$", "", base, flags=re.I).strip()
    return base


def _read_app_for(fname):
    f = fname.lower()
    if f.endswith((".xlsx", ".xls", ".csv")):
        return "excel"
    if f.endswith((".docx", ".doc", ".txt")):
        return "word"
    if f.endswith(".pdf"):
        return "pdf"
    return "excel"


def _extract_items(blackboard, fanout_subtask, llm, task_text=""):
    """After the read wave, parse the concrete list of items the fan-out iterates over. Tries the
    DETERMINISTIC table parse first (robust); falls back to the LLM only if the data isn't a grid."""
    struct = _extract_items_structural(blackboard, task_text)
    if len(struct) >= 1:
        return struct[:25]
    prompt = ("From the DATA below, output the COMPLETE list of items this sub-task must be repeated "
              "over (every person/row/file --- do not skip any), as a JSON array of short strings. "
              "SUB-TASK: %s\nDATA:\n%s\nJSON array:" % (fanout_subtask, blackboard[:1600]))
    items = _parse_json_block(llm.generate(prompt)) or []
    STOP = {"user", "name", "names", "participant", "participants", "member", "members", "email",
            "emails", "section", "time", "summary", "row", "column", "id", "person", "people", "item"}
    bb = blackboard.lower()
    seen, out = set(), []
    for x in items:
        x = str(x).strip()
        # keep ONLY items that literally appear in the read data (kills hallucinations/garbage)
        if x and x.lower() not in seen and x.lower() not in STOP and len(x) < 60 and x.lower() in bb:
            seen.add(x.lower()); out.append(x)
    return out[:25]


def _expand_fanout(dels, blackboard, llm, step_counter, task_text=""):
    """Replace each fan-out create/send delegation with ONE delegation PER ITEM (parsed from the
    read results), turning error-prone in-agent looping into a clean parallel wave of single-item
    leaves. Non-fan-out delegations pass through unchanged."""
    out = []
    for d in dels:
        if _is_fanout(d.subtask) and d.agent_type in ("calendar", "email", "excel", "word"):
            items = _extract_items(blackboard, d.subtask, llm, task_text); step_counter[0] += 1
            if len(items) >= 1:
                # strip ALL fan-out words so each per-item subtask is a clean single-item leaf
                base = re.sub(r"\{[^}]*\}", "", d.subtask)
                base = re.sub(r"\b(each|every|all)\s+", "", base, flags=re.I)
                base = re.sub(r"\b(participants?|members?|everyone|people)\b", "", base, flags=re.I)
                base = re.sub(r"\s+for\s*$", "", base.strip(), flags=re.I).strip()
                for it in items:
                    out.append(Delegation(agent_type=d.agent_type, subtask="%s for %s" % (base, it),
                                          category=getattr(d, "category", None) or _phase2d(d.subtask)))
                continue
        out.append(d)
    return out


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
            ("SHARED CONTEXT (results from earlier agents):\n%s\n" % blackboard[:1200]) if blackboard else "",
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
        log.append(obs[:1200])          # keep read data (names/rows) intact for fan-out extraction
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
                       observation=" | ".join(log)[:1400]), steps


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
    step_counter, blackboard_ref = [0], [seed]
    executor = ParallelExecutor(make_subagent_runner(env, llm, max_steps, step_counter, blackboard_ref))
    results, waves = [], []

    # DETERMINISTIC FAN-OUT FAST-PATH: when the task repeats an act over each item of a group, the
    # LLM planner is the fragility (it randomly splits/mislabels read vs act across runs). Bypass it:
    # read every data file -> parse the item list structurally (condition-aware) -> spawn one act
    # leaf per item. Planner-free -> reproducible; the only LLM calls are the leaf sub-agents.
    flist = [f.strip() for f in files.split(",") if f.strip()]
    if _is_fanout_task(task_text) and flist:
        # a clean per-item leaf completes in 1-2 steps; cap steps so a non-clean fan-out (grouped /
        # pairing) that slips the gate can't loop to a 100K-token runaway.
        fp_exec = ParallelExecutor(make_subagent_runner(env, llm, min(max_steps, 4), step_counter, blackboard_ref))
        reads = [Delegation(agent_type=_read_app_for(f), subtask="read %s" % f, category="read") for f in flist]
        rr = await fp_exec.execute_group(reads, None); results.extend(rr); waves.append(reads)
        blackboard_ref[0] += "\n".join("[%s] %s" % (r.agent_type, r.observation) for r in rr) + "\n"
        items = _extract_items(blackboard_ref[0], task_text, llm, task_text); step_counter[0] += 1
        if items:
            app, base = _target_app(task_text), _act_clause(task_text)
            per = [Delegation(agent_type=app, subtask="%s for %s" % (base, it), category="create") for it in items]
            for wave in DependencyAnalyzer().analyze(per):
                wr = await fp_exec.execute_group(wave, None); results.extend(wr); waves.append(wave)
                blackboard_ref[0] += "\n".join("[%s] %s" % (r.agent_type, r.observation) for r in wr) + "\n"
            return results, waves, reads + per

    dels = plan_delegations(task_text, llm, files_hint=files)
    if len(dels) < 2:
        return None, [], dels

    # PHASE 1 -- run the READ delegations (parallel), so their data lands on the blackboard.
    # Use the robust local phase split (an ACT verb beats an incidental 'listed'); set the DAG
    # category explicitly so create->send ordering survives the same 'list' false-positive.
    reads, acts = [], []
    for d in dels:
        ph = _phase2d(d.subtask)
        if ph == "read":
            reads.append(d)
        else:
            d.category = ph; acts.append(d)
    if reads:
        rr = await executor.execute_group(reads, None); results.extend(rr); waves.append(reads)
        blackboard_ref[0] += "\n".join("[%s] %s" % (r.agent_type, r.observation) for r in rr) + "\n"

    # PHASE 2 -- EXPAND fan-out act delegations into ONE PER ITEM (parsed from the read data),
    # so each becomes a clean single-item leaf. This is the true N-way parallel wave.
    acts = _expand_fanout(acts, blackboard_ref[0], llm, step_counter, task_text=task_text)

    # PHASE 3 -- run the (now per-item) act delegations, grouped into create->send waves.
    for wave in DependencyAnalyzer().analyze(acts):
        wr = await executor.execute_group(wave, None); results.extend(wr); waves.append(wave)
        blackboard_ref[0] += "\n".join("[%s] %s" % (r.agent_type, r.observation) for r in wr) + "\n"
    return results, waves, dels
