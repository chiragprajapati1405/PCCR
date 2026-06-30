"""PCCR policy for OfficeBench — driven by the REAL architecture (swap A).

Same app-switching loop, but memory is the real FAISS EpisodicStore gated by the
real router.py rho-gate (via RealArch). On an EM hit, em.search() supplies
orchestrator-level past plans and em.preload_for_agents() supplies per-app
subtask memories. no_memory injects nothing; retrieve_all always injects; pccr
injects only when the real rho-gate consults EM.
"""
from __future__ import annotations

import json
import re

from .cerebras_llm import CerebrasLLM
from .real_mem import is_action_failure

_FNAME_RE = re.compile(r"[\w\-/]+\.(?:pdf|docx|xlsx|ics|txt|eml|csv|jpg|jpeg|png|md)", re.I)


def _is_finish(action) -> bool:
    a = str(action or "")
    return '"finish_task"' in a or "'finish_task'" in a


def _is_giveup(action) -> bool:
    """The agent's own give-up signal. By construction the completion gate (which only
    intercepts finish_task) NEVER sees these -- yet got_stuck is the DOMINANT failure
    termination for multi_app (43/64). We intercept it too: don't let the agent quit
    while a required output is still missing."""
    a = str(action or "")
    return '"got_stuck"' in a or "'got_stuck'" in a


def _filenames(text) -> set:
    """Output filenames mentioned in a task/action (basename, lowercased)."""
    return {m.split("/")[-1].lower() for m in _FNAME_RE.findall(str(text or ""))}

# Output-path + completion convention (NEXT_STEPS D3). The agent prompt never told
# the model WHERE to write outputs or that L3 tasks need MULTIPLE files; tracing the
# 67 L3 file_exist failures showed 17 wrong-dir/renamed + 36 "second output never made".
# Applied uniformly to ALL arms (a harness-fairness fix, not a PCCR-only advantage).
OUTPUT_CONVENTION = (
    "##OUTPUT RULES (follow exactly):\n"
    "1. Write EVERY output file under /testbed/data/ using the EXACT filename named in "
    "the task. Do not add prefixes, do not rename (e.g. if the task says report.pdf, "
    "write /testbed/data/report.pdf, not /testbed/report.pdf or a renamed file).\n"
    "2. Many tasks require MULTIPLE outputs across different apps (e.g. an Excel file AND "
    "calendar .ics events AND a converted PDF AND a sent email). Produce ALL required "
    "outputs before you finish — do NOT stop after the first one. Re-read the task and "
    "confirm every requested file/action is done before calling finish_task.\n"
)


def _adapt_prompt(past_desc, past_actions, cur_task):
    seq = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(past_actions))
    return (
        "You previously solved a SIMILAR task. Adapt its action sequence to the CURRENT task.\n\n"
        f"PAST TASK: {past_desc}\n"
        f"PAST SUCCESSFUL ACTION SEQUENCE (in order):\n{seq}\n\n"
        f"CURRENT TASK: {cur_task}\n\n"
        "Rewrite the sequence to solve the CURRENT task:\n"
        "- change parameters (file names, values, cell refs, recipients, dates) to match the current task;\n"
        "- ADD any steps the current task needs that the past one didn't;\n"
        "- REMOVE steps the current task doesn't need;\n"
        "- KEEP every output-producing step across ALL apps -- if the task needs an Excel file AND "
        "calendar events AND a PDF, the sequence must produce ALL of them (do not drop the 2nd/3rd app);\n"
        "- RE-VERIFY all data positions against the CURRENT task -- row/column indices, cell refs, "
        "filenames and which record to act on are almost always DIFFERENT from the past task; never "
        "reuse the past indices blindly (e.g. read the table and find the row that actually matches);\n"
        "- write outputs under /testbed/data with the EXACT filenames named in the task;\n"
        "- end with {\"app\":\"system\",\"action\":\"finish_task\",\"answer\":\"...\"}.\n\n"
        "Output ONLY a JSON array of action objects, in order, nothing else. Example:\n"
        '[{"app":"shell","action":"run","command":"ls /testbed/data"}, '
        '{"app":"system","action":"switch_app","target_app":"excel"}, ...]\n'
    )


def _parse_action_array(text):
    """Extract an ordered list of action-object JSON STRINGS from the adapt reply.
    Robust to prose around the array and to a stream of bare {..}{..} objects."""
    i = text.find("[")
    if i >= 0:
        try:
            arr = json.loads(text[i:text.rfind("]") + 1])
            if isinstance(arr, list):
                return [json.dumps(a) for a in arr if isinstance(a, dict) and a.get("app")]
        except Exception:
            pass
    # fallback: scan balanced top-level objects
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for j, c in enumerate(text):
        if in_str:
            if esc: esc = False
            elif c == "\\": esc = True
            elif c == '"': in_str = False
            continue
        if c == '"': in_str = True
        elif c == "{":
            if depth == 0: start = j
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    o = json.loads(text[start:j + 1])
                    if isinstance(o, dict) and o.get("app"):
                        out.append(json.dumps(o))
                except Exception:
                    pass
    return out


def make_pccr_policy(LLMPolicyCls, model, env, config, real_arch=None, method="no_memory",
                     pattern=None, exclude_task=None, use_pm=False, real_mem=None,
                     replay=False, replay_threshold=0.75,
                     plan_then_execute=False, batch_size=4, output_convention=False,
                     completion_gate=False, self_verify=False, inject_once=False, heavy_steps=3):

    class PCCRPolicy(LLMPolicyCls):
        def __init__(self):
            super().__init__(model_name="local-oss", key="", env=env, config=config)
            self.llm = CerebrasLLM(model_name=model, system_message=self.system_message)
            self.llm.max_tokens = 2048   # 512 truncated gpt-oss actions to a lone '{' (~35% malformed)
            self.real, self.method, self.pattern = real_arch, method, pattern
            self.exclude_task = exclude_task
            self.task = config["task"]
            self._realmem_block = ""     # set on the gate arm when driven by the real MemoryManager
            self._heavy_block = self._light_block = ""
            self._inject_once = bool(inject_once) and method not in ("no_memory", "retrieve_all")
            self._heavy_steps = max(1, int(heavy_steps))   # inject heavy guidance for the first K calls
            self._prompt_n = 0                              # build_prompt (≈ LLM-call) counter
            self._pm_text = ""
            self._consult_em = False
            self._stm_hit = False
            self._rho_decision = None
            self._top_em_sim = 0.0
            if real_arch is not None:
                hits = real_arch.search(self.task, k=1)
                self._top_em_sim = float(hits[0][0]) if hits else 0.0

            # B1+B5: adaptive+verified replay state (high-confidence hits only)
            self._replay_on = bool(replay) and method not in ("no_memory", "retrieve_all") and real_mem is not None
            self._replay_cand = None
            self._replay_q = []           # remaining adapted actions to execute
            self._replay_adapted = False
            self._replay_aborted = False
            if self._replay_on:
                self._replay_cand = real_mem.replay_candidate(
                    self.task, exclude_task=exclude_task, min_sim=replay_threshold)

            # B2: plan-then-execute -- batch the NEXT few actions per LLM call (the
            # medium-confidence tier / replay fallback). Verified the same way as B1.
            self._b2_on = bool(plan_then_execute) and method not in ("no_memory", "retrieve_all")
            self._batch_q = []
            self._batch_k = max(2, int(batch_size))
            self._use_convention = bool(output_convention)   # D3: opt-in (default off = baseline)
            # finish-gate state: completion gate (required files exist) + self-verify reflection
            self._completion_gate = bool(completion_gate) and method not in ("no_memory", "retrieve_all")
            self._self_verify = bool(self_verify) and method not in ("no_memory", "retrieve_all")
            self._req_files = _filenames(self.task)           # output files named in the task
            self._gate_tries = 0
            self._verified = False

            gate_arm = method not in ("no_memory", "retrieve_all")
            if gate_arm and real_mem is not None:
                # === REAL ARCHITECTURE: MemoryManager.retrieve() full cascade ===
                # STM-first short-circuit, then rho-gated EM/SM/ENT, PM always-read.
                text, tr = real_mem.retrieve(self.task, pattern, exclude_task=exclude_task)
                self._realmem_block = text
                # inject-once: heavy procedural guidance (PM rule + examples) only for the first
                # few steps; the light roadmap (plan steps) every step. Cuts the per-step token
                # cost (root cause of the 2x vs retrieve-all) without losing planning guidance.
                self._heavy_block = tr.get("heavy_block", "")
                self._light_block = tr.get("light_block", "")
                self.em_trace = {"method": method, "consult_em": tr["consult_em"],
                                 "stm_hit": tr["stm_hit"], "consult_sm": tr.get("consult_sm"),
                                 "consult_ent": tr.get("consult_ent"), "pm_used": tr["pm_used"],
                                 "consulted_stores": tr["consulted_stores"], "injected_tokens": 0,
                                 "orchestrator_hits": [], "agent_hits": [],
                                 "top_em_sim": round(self._top_em_sim, 3), "faiss_backed": True}
            else:
                # no_memory (nothing) / retrieve_all (always inject EM via _memory_block)
                if use_pm and real_arch is not None and method != "no_memory":
                    r = real_arch.pm_rule(self.task)
                    if r:
                        self._pm_text = f"##PROCEDURAL RULE (learned from {r['support']} past tasks): {r['text']}\n\n"
                if real_arch is not None and method == "retrieve_all":
                    self._consult_em = True
                self.em_trace = {"method": method, "consult_em": self._consult_em,
                                 "orchestrator_hits": [], "agent_hits": [], "injected_tokens": 0,
                                 "faiss_backed": True, "top_em_sim": round(self._top_em_sim, 3),
                                 "stm_hit": self._stm_hit,
                                 "consulted_stores": (["episodic"] if self._consult_em else []),
                                 "pm_used": bool(self._pm_text)}

        def forward(self, env):
            """Decide the next action, then apply the FINISH GATE before letting the
            agent stop: (1) hard completion gate -- block finish_task while a required
            output file named in the task hasn't been created; (2) one self-verify
            reflection when all files exist. Both target the 'quit early / wrong
            content' failure modes."""
            action = self._decide_action(env)
            # Intercept BOTH "I'm done" (finish_task) AND "I give up" (got_stuck): in either
            # case, if a required output is still missing, redirect instead of ending. got_stuck
            # gets a larger try budget -- quitting with work left is strictly worse than retrying.
            stopping = _is_finish(action) or _is_giveup(action)
            cap = 6 if _is_giveup(action) else 4
            if (self._completion_gate or self._self_verify) and stopping and self._gate_tries < cap:
                corrective = self._finish_gate(env, giveup=_is_giveup(action))
                if corrective is not None:
                    self._gate_tries += 1
                    return corrective
            return action

        # -- the original B1/B2 action decision (unchanged) ---------------------
        def _decide_action(self, env):
            """B1: adaptive+VERIFIED replay. On a high-confidence EM hit, adapt the
            cached action sequence in ONE LLM call, then execute it WITHOUT per-step
            LLM reasoning -- but verify each prior action by the tool's own success/
            error signal; on any failure, ABORT replay and hand back to the normal
            LLM loop from that point (never blind). Falls through to super().forward
            when not replaying."""
            if not self._replay_on or self._replay_cand is None or self._replay_aborted:
                return self._plan_or_single(env)

            # First entry: 1 adapt call -> the executable plan (handles the "11th step").
            if not self._replay_adapted:
                self._replay_adapted = True
                sim, past_desc, past_actions = self._replay_cand
                reply = self.llm.generate(_adapt_prompt(past_desc, past_actions, self.task))
                self._replay_q = _parse_action_array(reply)
                self.em_trace["replay"] = {"used": bool(self._replay_q), "sim": round(sim, 3),
                                           "planned": len(self._replay_q), "executed": 0, "aborted": False}
                if not self._replay_q:                       # adapt failed -> batch/normal loop
                    self._replay_aborted = True
                    return self._plan_or_single(env)

            # Verify the PREVIOUS replayed action by the env's own signal.
            if env.history:
                last_obs = env.history[-1][1]
                if is_action_failure(last_obs):
                    self._replay_aborted = True
                    self.em_trace["replay"]["aborted"] = True
                    return self._plan_or_single(env)         # recover from here (batch/LLM)

            if self._replay_q:
                action = self._replay_q.pop(0)
                self.em_trace["replay"]["executed"] += 1
                return action
            self._replay_aborted = True                       # sequence exhausted -> batch/LLM finish
            return self._plan_or_single(env)

        def _plan_or_single(self, env):
            """B2: plan-then-execute. Plan up to K next actions in ONE LLM call and
            execute them with the same tool-signal verification as B1 (abort the batch
            on any failure -> re-plan). Falls back to the base single-action loop when
            B2 is off or a batch can't be parsed."""
            if not self._b2_on:
                return super().forward(env)
            # If the PREVIOUS action failed, recover with a SINGLE LLM step instead of
            # re-planning a whole batch -- otherwise a malformed action -> batch abort ->
            # re-plan (also malformed) loop thrashes the step budget (observed on L3).
            if env.history and is_action_failure(env.history[-1][1]):
                self._batch_q = []
                return super().forward(env)
            if not self._batch_q:
                prompt = self.build_prompt(env) + (
                    f"\n\n##PLAN-THEN-EXECUTE: output the NEXT up to {self._batch_k} actions you will take, "
                    "in order, as a JSON array of action objects (same format). Stop the list at the first "
                    "action whose result you'd need to SEE before deciding the next (e.g. a read/list). "
                    "Include finish_task only when truly done.")
                self._batch_q = _parse_action_array(self.llm.generate(prompt))
                self.em_trace["batch_calls"] = self.em_trace.get("batch_calls", 0) + 1
                self.em_trace["batch_actions"] = self.em_trace.get("batch_actions", 0) + len(self._batch_q)
                if not self._batch_q:
                    return super().forward(env)
            return self._batch_q.pop(0)

        def _input_files(self, env):
            """Filenames that already existed before the agent created anything -- i.e. the
            task's INPUTS, read from the earliest directory-listing observations. Required
            OUTPUTS must exclude these, or the gate demands the agent 'create' a file it is
            only meant to read (observed on agenda.jpg, hw1.docx, ...)."""
            inputs = set()
            for a, o in (env.history or []):
                os_ = str(o or "")
                if "Success" in os_ and not is_action_failure(os_):
                    break                                    # stop at first creation: later listings
                                                             # may echo OUTPUTS the agent just made
                inputs |= _filenames(os_)                    # pre-creation listings = task inputs
            return inputs

        def _created_files(self, env):
            """Output filenames the agent has SUCCESSFULLY produced so far. Files come from
            success observations on create/write/convert actions; emails (.eml) and calendar
            events (.ics) leave NO filename in their success text, so we credit those by their
            extension whenever a send_email / create_event action succeeds."""
            made = set()
            req = self._req_files
            for a, o in (env.history or []):
                o_ = str(o or "")
                if is_action_failure(o_):
                    continue
                if "Success" in o_:
                    made |= _filenames(a)
                a_ = str(a)
                if ('"send_email"' in a_ or "email sent" in o_.lower()) and "fail" not in o_.lower():
                    made |= {f for f in req if f.endswith(".eml")}          # credit email outputs
                if '"create_event"' in a_ and ("event" in o_.lower() and "fail" not in o_.lower()):
                    made |= {f for f in req if f.endswith(".ics")}          # credit calendar outputs
            return made

        def _finish_gate(self, env, giveup=False):
            """Step 1 (completion gate): block stopping while a required OUTPUT file (task
            files minus inputs) has NOT been produced -- now also intercepts got_stuck, with
            schema-aware guidance for .eml/.ics outputs (which need send_email/create_event,
            not a file write). Step 2 (self-verify): one reflection when all outputs exist."""
            required = self._req_files - self._input_files(env)
            if self._completion_gate and required:
                missing = sorted(required - self._created_files(env))
                if missing:
                    hints = []
                    for f in missing:
                        if f.endswith(".eml"):
                            hints.append(f'{f}: send the email with email:send_email '
                                         '(fields: sender, recipient, subject, content)')
                        elif f.endswith(".ics"):
                            hints.append(f'{f}: create the calendar event with calendar:create_event '
                                         '(fields: user, summary, time_start, time_end)')
                        else:
                            hints.append(f'{f}: create it under /testbed/data with that EXACT name')
                    lead = ("##DO NOT GIVE UP -- you are not stuck; required work remains."
                            if giveup else "##NOT DONE -- do NOT finish yet.")
                    prompt = (self.build_prompt(env) + "\n\n" + lead +
                              " These required outputs do not exist yet -- produce each, then finish:\n  - " +
                              "\n  - ".join(hints))
                    return self.proc_action(self.llm.generate(prompt))
            if self._self_verify and not self._verified:
                self._verified = True
                prompt = (self.build_prompt(env) +
                          "\n\n##VERIFY before finishing: re-read the task and check that EVERY requested "
                          "file, value and item is present and CORRECT (all rows/items included, computations "
                          "right, exact filenames). If anything is missing or wrong, issue the action to fix "
                          "it now. Only if everything is truly complete, re-issue finish_task.")
                return self.proc_action(self.llm.generate(prompt))
            return None

        def proc_action(self, action):
            """Return the FIRST balanced JSON object. The base class takes
            first-'{' to last-'}', which merges concatenated actions
            ({...}{...}) into one malformed blob; gpt-oss frequently emits
            several at once. We brace-match and stop at the first complete one."""
            i = action.find("{")
            if i < 0:
                return action
            depth, in_str, esc = 0, False, False
            for j in range(i, len(action)):
                c = action[j]
                if in_str:
                    if esc:        esc = False
                    elif c == "\\": esc = True
                    elif c == '"':  in_str = False
                elif c == '"':      in_str = True
                elif c == "{":      depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return action[i:j + 1]            # first complete object
            return action[i:]                              # truncated: hand back partial

        def build_prompt(self, env):
            base = super().build_prompt(env)
            # MALFORMED-RECOVERY: gpt-oss frequently invents WRONG action names
            # (e.g. shell `command`/excel `append_row`) -> OfficeBench replies with a generic
            # "Malformed action!" and the model keeps guessing wrong names, looping to the cap.
            # When the last action was malformed, show the EXACT valid action names for the
            # current app so it corrects in one step. Applied on every arm (a format aid, not
            # memory) -- breaks the single biggest failure mode (malformed-loops).
            if env.history and "Malformed action" in str(env.history[-1][1]):
                app = getattr(env, "current_app", None)
                try:
                    valid = env.get_available_actions() if app else None
                except Exception:
                    valid = None
                if valid:
                    ex = {"shell": '{"app":"shell","action":"command","command":"ls /testbed/data"} '
                                   '(there is NO list_directory/list/run action -- use "command")',
                          "excel": '{"app":"excel","action":"read_file","file_path":"/testbed/data/x.xlsx"}',
                          "word": '{"app":"word","action":"write_to_file","file_path":"/testbed/data/x.docx","content":"..."}',
                          "pdf": '{"app":"pdf","action":"read_file","pdf_file_path":"/testbed/data/x.pdf"}',
                          "ocr": '{"app":"ocr","action":"recognize_file","image_path":"/testbed/data/x.png"}',
                          }.get(app, "")
                    base += (f"\n\n##IMPORTANT: your last action used an INVALID action name. "
                             f"The ONLY valid actions for app '{app}' are: {valid}. "
                             + (f"Correct example: {ex} " if ex else "")
                             + "Re-issue using one of these EXACT action names (or switch_app).")
            # gate arm: the real MemoryManager bundle; else PM(always) + EM(gated)
            if self._realmem_block:
                if self._inject_once:
                    # heavy procedural guidance only for the first K calls; light roadmap always.
                    self._prompt_n += 1
                    prefix = ((self._heavy_block if self._prompt_n <= self._heavy_steps else "")
                              + self._light_block)
                else:
                    prefix = self._realmem_block
            else:
                prefix = self._pm_text + self._memory_block(env)
            # D3: output-path + completion convention -- opt-in (default off so the baseline
            # arms reproduce the paper). When on, applied to every arm (a static harness
            # instruction like the system prompt; not counted as injected memory tokens).
            if self._use_convention:
                base = OUTPUT_CONVENTION + "\n" + base
            if prefix:
                self.em_trace["injected_tokens"] += max(1, len(prefix) // 4)
                return prefix + "\n\n" + base
            return base

        def _memory_block(self, env):
            if not self._consult_em or self.real is None:
                return ""
            blocks = []
            hits = self.real.search(self.task, k=5)                 # FAISS orchestrator-level
            if hits:
                blocks.append("RELEVANT PAST TASK PLANS (reuse the steps if similar):")
                for sc, m in hits:
                    if self.exclude_task and m.description == self.exclude_task:
                        continue
                    blocks.append(f" - ({sc:.2f}) {m.description[:80]} | "
                                  f"plan: {' ; '.join(m.plan[:8])}")
                self.em_trace["orchestrator_hits"].append([m.description[:60] for _s, m in hits[:5]])
            app = getattr(env, "current_app", None)
            if app:
                pre = self.real.preload(self.task, k=3)             # FAISS agent-level
                ah = pre.get(app, [])
                if ah:
                    blocks.append(f"RELEVANT PAST {app} ACTIONS:")
                    for sm in ah:
                        blocks.append(f" - {sm.action[:110]}")
                    self.em_trace["agent_hits"].append([sm.action[:50] for sm in ah])
            return "\n".join(blocks)

    return PCCRPolicy()
