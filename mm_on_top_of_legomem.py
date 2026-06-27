"""
LEGOMem with MemoryManager Architecture
═══════════════════════════════════════
Architecture:
  1 Orchestrator (gpt-oss-120b) + 2 Agents (calendar, email)

Memory Manager: 6 Memory Types
  PM  (Procedural):  Agent prompts, rules (defined inside MM)
  WM  (Working):     Current task, step history, shared context
  STM (Short-Term):  Pattern cache + step signatures + semantic cache
  EM  (Episodic):    Past task experiences (FAISS)
  SM  (Semantic):    Embedding vectors (inside FAISS)
  ENT (Entity):      User/agent profiles

10 Lifecycle Phases:
  1. System Bootstrap    6. Agent Inference
  2. Task Ingestion      7. Agent Output
  3. Memory Retrieval    8. Environment Observation
  4. Orch Inference      9. Memory Storage
  5. Orch Output        10. Memory Consolidation

Routing:
  READ:  Level 1 (Static) → Level 2 (Pattern-Aware) → Level 3 (Dynamic)
  WRITE: W1 (Success Gate) → W2 (STM) → W3 (ENT) → W4 (EM) → W5 (PM) → W6 (History)
"""

import os, sys, re, time, json, builtins, subprocess, shutil, logging
import glob as glob_module
import numpy as np
import faiss
from datetime import datetime
from dataclasses import dataclass, field, asdict
from collections import OrderedDict
from openai import OpenAI

# ══════════════════════════════════════════════════════════════════
#  TERMINAL TEE
# ══════════════════════════════════════════════════════════════════
class TerminalTee:
    def __init__(self, filename, original):
        self.log = open(filename, "a", encoding="utf-8", buffering=1)
        self.orig = original
    def write(self, m):
        self.orig.write(m); self.log.write(m)
    def flush(self):
        self.orig.flush(); self.log.flush()
    def close(self):
        if not self.log.closed: self.log.close()

LOG_DIR = "logs_memory_manager"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sys.stdout = TerminalTee(f"{LOG_DIR}/terminal_{timestamp}.log", sys.stdout)
sys.stderr = TerminalTee(f"{LOG_DIR}/terminal_{timestamp}.log", sys.stderr)

# ══════════════════════════════════════════════════════════════════
#  MONKEY-PATCHING
# ══════════════════════════════════════════════════════════════════
LOCAL_TESTBED = ["/tmp/officebench_testbed"]

def _resolve(p):
    if isinstance(p, str) and p.startswith("/testbed/"):
        return p.replace("/testbed/", f"{LOCAL_TESTBED[0]}/", 1)
    if isinstance(p, str) and p == "/testbed":
        return LOCAL_TESTBED[0]
    return p

_om = os.makedirs
def _pm(n, *a, **k): return _om(_resolve(str(n)), *a, **k)
os.makedirs = _pm
_oo = builtins.open
def _po(f, *a, **k): return _oo(_resolve(f) if isinstance(f, str) else f, *a, **k)
builtins.open = _po
_oe = os.path.exists
def _pe(p): return _oe(_resolve(p) if isinstance(p, str) else p)
os.path.exists = _pe
_of = os.path.isfile
def _pf(p): return _of(_resolve(p) if isinstance(p, str) else p)
os.path.isfile = _pf
_od = os.path.isdir
def _pd(p): return _od(_resolve(p) if isinstance(p, str) else p)
os.path.isdir = _pd
_og = glob_module.glob
def _pg(pat, *a, **k): return _og(_resolve(pat) if isinstance(pat, str) else pat, *a, **k)
glob_module.glob = _pg

# ══════════════════════════════════════════════════════════════════
#  MULTI-ACCOUNT LLM
# ══════════════════════════════════════════════════════════════════
class MultiLLM:
    def __init__(self, api_keys, model="gpt-oss-120b"):
        self.clients = []
        for i, k in enumerate(api_keys):
            if k.strip():
                self.clients.append({
                    "client": OpenAI(api_key=k, base_url="https://api.cerebras.ai/v1", max_retries=0),
                    "name": f"Acc-{i+1}"})
        if not self.clients: raise ValueError("No valid API keys")
        self.current = 0
        self.model = model
        self.total_calls = 0

    def call(self, system_prompt, user_prompt, temperature=0.0, caller="?"):
        for attempt in range(len(self.clients) * 3):
            idx = self.current % len(self.clients)
            acc = self.clients[idx]
            self.total_calls += 1
            print(f"    🔗 [{caller}] → {acc['name']} ({self.model}) [#{self.total_calls}]")
            try:
                r = acc["client"].chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=2048,
                    messages=[{"role": "system", "content": system_prompt},
                              {"role": "user", "content": user_prompt}])
                self.current += 1
                time.sleep(1.5)
                return r.choices[0].message.content
            except Exception as e:
                if "429" in str(e) or "rate" in str(e).lower():
                    print(f"    ⚡ {acc['name']} rate limited, next...")
                    self.current += 1; time.sleep(2)
                else:
                    print(f"    ⚠️ {acc['name']}: {str(e)[:60]}")
                    self.current += 1
        print(f"    ⏳ All accounts exhausted. Waiting 60s...")
        time.sleep(60)
        return self.call(system_prompt, user_prompt, temperature, caller)


def parse_action(response):
    text = re.sub(r'^```(?:python|json)?\s*', '', response.strip())
    text = re.sub(r'\s*```$', '', text).strip()
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: d = eval(m.group())
        except:
            try: d = json.loads(m.group())
            except: return None
        if isinstance(d, dict) and len(d) > 0 and ("app" in d or "agent" in d or "answer" in d):
            return d
    return None


# ══════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════
@dataclass
class SubtaskMemory:
    subtask_id: str = ""
    subtask_description: str = ""
    agent_type: str = ""
    tool_calls: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    outcome: str = ""
    execution_summary: str = ""

@dataclass
class FullTaskMemory:
    memory_id: str = ""
    task_description: str = ""
    high_level_plan: str = ""
    reasoning: str = ""
    subtask_memories: list = field(default_factory=list)
    final_answer: str = ""
    reflection: str = ""
    embedding: list = field(default_factory=list)


# ══════════════════════════════════════════════════════════════════
#  LTM: FAISS MEMORY BANK (Episodic + Semantic)
# ══════════════════════════════════════════════════════════════════
class LEGOMemBank:
    def __init__(self, embed_fn, embed_dim=384, path="./legomem_bank_mm"):
        self.embed_fn = embed_fn
        self.dim = embed_dim
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.full_task_memories = []
        self.full_task_index = faiss.IndexFlatIP(embed_dim)
        self.agent_memories = {}
        self.agent_indexes = {}

    def add_full_task(self, mem):
        emb = self.embed_fn(mem.task_description).reshape(1,-1).astype('float32')
        faiss.normalize_L2(emb)
        mem.embedding = emb[0].tolist()
        self.full_task_memories.append(mem)
        self.full_task_index.add(emb)

    def add_subtask(self, agent_type, sub):
        if agent_type not in self.agent_memories:
            self.agent_memories[agent_type] = []
            self.agent_indexes[agent_type] = faiss.IndexFlatIP(self.dim)
        emb = self.embed_fn(sub.subtask_description).reshape(1,-1).astype('float32')
        faiss.normalize_L2(emb)
        self.agent_memories[agent_type].append(sub)
        self.agent_indexes[agent_type].add(emb)

    def retrieve_for_orchestrator(self, desc, k=3):
        if self.full_task_index.ntotal == 0: return []
        emb = self.embed_fn(desc).reshape(1,-1).astype('float32')
        faiss.normalize_L2(emb)
        k = min(k, self.full_task_index.ntotal)
        D, I = self.full_task_index.search(emb, k)
        return [self.full_task_memories[i] for i in I[0] if i >= 0]

    def retrieve_for_agent(self, agent_type, desc, k=3):
        if agent_type not in self.agent_indexes: return []
        idx = self.agent_indexes[agent_type]
        if idx.ntotal == 0: return []
        emb = self.embed_fn(desc).reshape(1,-1).astype('float32')
        faiss.normalize_L2(emb)
        k = min(k, idx.ntotal)
        D, I = idx.search(emb, k)
        return [self.agent_memories[agent_type][i] for i in I[0] if i >= 0]

    def save(self):
        data = {"full_task": [asdict(m) for m in self.full_task_memories],
                "agent_memories": {a: [asdict(m) for m in ms] for a, ms in self.agent_memories.items()}}
        with _oo(f"{self.path}/bank.json", "w") as f: json.dump(data, f, indent=2)
        faiss.write_index(self.full_task_index, f"{self.path}/full_task.faiss")
        for a, idx in self.agent_indexes.items():
            faiss.write_index(idx, f"{self.path}/subtask_{a}.faiss")

    def stats(self):
        s = f"Full-task:{len(self.full_task_memories)}"
        for a, ms in self.agent_memories.items(): s += f", {a}:{len(ms)}"
        return s


# ══════════════════════════════════════════════════════════════════
#  PATTERN-BASED STM
# ══════════════════════════════════════════════════════════════════
def get_step_signature(steps):
    sig_parts = []
    for s in steps:
        agent = s.get("agent", "?")
        subtask = s.get("subtask", "").lower()
        if "list" in subtask or "read" in subtask: action = "read"
        elif "create" in subtask or "add" in subtask: action = "create"
        elif "send" in subtask: action = "send"
        elif "finish" in subtask: action = "finish"
        elif "delete" in subtask: action = "delete"
        else: action = "other"
        sig_parts.append(f"{agent}:{action}")
    return " → ".join(sig_parts)


class PatternSTM:
    def __init__(self, embed_fn, l2_size=20, l2_threshold=0.85):
        self.embed_fn = embed_fn
        self.pattern_cache = OrderedDict()
        self.step_cache = {}
        self.step_pattern_map = {}
        self.l2_cache = []
        self.l2_size = l2_size
        self.l2_threshold = l2_threshold
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}

    def classify(self, task_desc):
        text = task_desc.lower()
        words = set(re.findall(r'[a-z]+', text))
        known_users = {"bob","tom","alice","david","jane","john","mary"}
        users = list(words & known_users)
        user_count = len(users)
        has_q = bool(words & {"what","who","which","how","does","check","find"})
        has_create = bool(words & {"add","create","schedule","set","put"})
        has_find = bool(words & {"find","help","common","together","both"})
        has_remind = bool(words & {"remind","notify","alert"})
        has_send = bool(words & {"send","write","compose","forward"})
        has_cal = bool(words & {"calendar","event","meeting","dinner","workout","class",
                                "travelling","appointment","ics","schedule","shopping","zoom","lunch"})
        has_email = bool(words & {"email","mail","sent","receive","eml"})
        if has_find and has_cal and user_count >= 2: return "multi_cal_find_and_create", users
        if has_remind: return "remind_notify", users
        if has_create and has_cal: return "single_cal_create", users
        if has_send and has_email: return "email_send", users
        if has_q and has_email: return "email_query", users
        if has_q and has_cal: return "cal_query", users
        if has_cal and not has_q: return "single_cal_create", users
        if has_email: return "email_query", users
        return "unknown", users

    def lookup(self, task_desc):
        t_start = time.perf_counter_ns()
        pattern, users = self.classify(task_desc)
        if pattern != "unknown" and pattern in self.pattern_cache:
            self.pattern_cache.move_to_end(pattern)
            self.stats["l1_hits"] += 1
            lat = (time.perf_counter_ns() - t_start) / 1000
            print(f"        [⚡ STM-L1 PATTERN HIT] pattern='{pattern}' ({lat:.1f}μs)")
            return self.pattern_cache[pattern], "STM-L1"
        if self.l2_cache:
            q = self.embed_fn(task_desc).reshape(-1).astype('float32')
            q = q / (np.linalg.norm(q) + 1e-9)
            best_s, best_i = 0.0, -1
            for i, (d, e, b) in enumerate(self.l2_cache):
                s = float(np.dot(q, e))
                if s > best_s: best_s, best_i = s, i
            if best_s >= self.l2_threshold and best_i >= 0:
                hit = self.l2_cache.pop(best_i)
                self.l2_cache.append(hit)
                self.stats["l2_hits"] += 1
                lat = (time.perf_counter_ns() - t_start) / 1000
                print(f"        [⚡ STM-L2 SEMANTIC HIT] score={best_s:.3f} ({lat:.1f}μs)")
                return hit[2], "STM-L2"
        self.stats["misses"] += 1
        lat = (time.perf_counter_ns() - t_start) / 1000
        print(f"        [⏳ STM MISS] pattern='{pattern}' ({lat:.1f}μs) → LTM")
        return None, "MISS"

    def store_bundle(self, task_desc, bundle, steps=None):
        pattern, _ = self.classify(task_desc)
        if pattern != "unknown":
            if pattern in self.pattern_cache:
                old = self.pattern_cache[pattern]
                bundle["success_count"] = old.get("success_count", 0) + 1
            else:
                bundle.setdefault("success_count", 1)
            self.pattern_cache[pattern] = bundle
            self.pattern_cache.move_to_end(pattern)
            print(f"        [📦→STM-L1] pattern='{pattern}' (count={bundle['success_count']})")
        if steps:
            sig = get_step_signature(steps)
            self.step_cache[sig] = bundle
            if pattern != "unknown": self.step_pattern_map[pattern] = sig
            print(f"        [📦→STM-L1b] sig='{sig}'")
        emb = self.embed_fn(task_desc).reshape(-1).astype('float32')
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.l2_cache.append((task_desc, emb, bundle))
        if len(self.l2_cache) > self.l2_size: self.l2_cache.pop(0)

    def summary(self):
        t = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["misses"]
        r = ((self.stats["l1_hits"] + self.stats["l2_hits"]) / t * 100) if t > 0 else 0
        return f"Patterns:{len(self.pattern_cache)} Sigs:{len(self.step_cache)} L1:{self.stats['l1_hits']} L2:{self.stats['l2_hits']} Miss:{self.stats['misses']} Rate:{r:.0f}%"


# ══════════════════════════════════════════════════════════════════
#  PLAN-DRIVEN AGENT LOADING
# ══════════════════════════════════════════════════════════════════
def extract_agents_from_plan(delegation):
    thought = str(delegation.get("thought", "")).lower()
    subtask = str(delegation.get("subtask", "")).lower()
    full_text = thought + " " + subtask
    agents_needed = set()
    if any(w in full_text for w in ["calendar","event","schedule","meeting","dinner",
                                     "workout","class","ics","appointment","travelling","reminder"]):
        agents_needed.add("calendar")
    if any(w in full_text for w in ["email","send","mail","notify","remind","eml","recipient","message"]):
        agents_needed.add("email")
    assigned = delegation.get("agent", "")
    if assigned in ["calendar", "email"]: agents_needed.add(assigned)
    return agents_needed


# ══════════════════════════════════════════════════════════════════
#  MEMORY CONSTRUCTION (Episodic distillation)
# ══════════════════════════════════════════════════════════════════
CURATION_PROMPT = """Extract a structured memory. Output ONLY valid JSON:
{
  "task_description": "original task",
  "high_level_plan": "Step 1: calendar creates event. Step 2: email sends reminder.",
  "reasoning": "why this worked",
  "subtasks": [
    {"subtask_description": "what was done", "agent_type": "calendar|email|system",
     "tool_calls": ["create_event"], "outcome": "success"}
  ],
  "final_answer": "result",
  "reflection": "what worked"
}"""

def construct_memories(llm, embed_fn, trajectories, bank):
    stats = {"full": 0, "subtask": 0}
    for traj in trajectories:
        if not traj.get("success"): continue
        resp = llm.call(CURATION_PROMPT, json.dumps(traj, default=str)[:3000], caller="MEMORY-CURATOR")
        try: p = json.loads(resp)
        except: p = {"task_description": traj.get("task_description",""),
                      "high_level_plan":"", "subtasks":[], "reasoning":"", "reflection":""}
        mem = FullTaskMemory(
            memory_id=f"m_{stats['full']}", task_description=p.get("task_description", traj.get("task_description","")),
            high_level_plan=p.get("high_level_plan",""), reasoning=p.get("reasoning",""),
            final_answer=p.get("final_answer",""), reflection=p.get("reflection",""))
        for st in p.get("subtasks",[]):
            sub = SubtaskMemory(subtask_id=f"s_{stats['subtask']}",
                subtask_description=st.get("subtask_description",""),
                agent_type=st.get("agent_type","system"),
                tool_calls=st.get("tool_calls",[]), outcome=st.get("outcome","success"))
            mem.subtask_memories.append(sub)
            if sub.agent_type in ["calendar","email"] and sub.subtask_description:
                bank.add_subtask(sub.agent_type, sub); stats["subtask"] += 1
        bank.add_full_task(mem); stats["full"] += 1
        time.sleep(1)
    return stats


# ══════════════════════════════════════════════════════════════════
#  MEMORY MANAGER — Central Router (6 Types, 3-Level Read, 6 Write Rules)
# ══════════════════════════════════════════════════════════════════
class MemoryManager:

    def __init__(self, embed_fn, embed_dim=384):
        """PHASE 1: SYSTEM BOOTSTRAP"""
        print(f"\n{'═'*70}")
        print(f"  PHASE 1: SYSTEM BOOTSTRAP")
        print(f"{'═'*70}")

        # ── Working Memory ──
        self.working_memory = {
            "current_task": None, "current_user": None, "current_date": None,
            "pattern": None, "step_history": [], "agent_scratchpad": [],
            "shared_context": {}, "memories_read_this_task": set(),
        }
        print(f"  📋 [WM] CREATED empty")

        # ── Short-Term Memory ──
        self.stm = PatternSTM(embed_fn=embed_fn)
        print(f"  📋 [STM] CREATED empty (pattern + semantic cache)")

        # ── Procedural Memory (prompts DEFINED here, not globals) ──
        self.procedural = {
            "orchestrator_prompt": """You are the ORCHESTRATOR. You PLAN and DELEGATE to agents. You NEVER execute actions.

AGENTS:
- calendar: create_event, list_events, delete_event
- email: send_email, list_emails
- system: finish_task (with answer)

Output ONLY a JSON dict:
{"thought": "what's done, what's next — MENTION ALL AGENTS you plan to use", "agent": "calendar|email|system", "subtask": "clear instruction for the agent"}

To finish: {"thought": "done", "agent": "system", "subtask": "finish_task", "answer": "the answer"}

CRITICAL RULES:
1. In your "thought", MENTION ALL AGENTS you will need.
2. Dates: YYYY-MM-DD HH:MM:SS
3. NEVER repeat a delegation that already succeeded.
4. If observation says data was found, USE that data. Do NOT re-read.
5. Maximum 8 delegations. Be efficient.
6. When delegating list_emails/list_events, specify USERNAME clearly.

TASK TYPE RULES:
A) PURE QUESTIONS ("who/which/what/how many/does"):
   → gather data → analyze the observation → finish_task with answer
B) PURE ACTIONS ("add/create/schedule/send/write"):
   → delegate to agent → finish_task confirming done
C) FIND-AND-ACT ("find time/help find/schedule together"):
   → list events for ALL users → compute available time → create_event for EACH user → finish
D) REMIND/NOTIFY:
   → calendar creates event → email sends reminder → finish_task
E) EMAIL QUESTIONS:
   → list_emails for correct user → analyze observation → finish_task

CRITICAL FORMAT RULE:
Put ALL details inside the "subtask" string. The agent ONLY sees subtask.
WRONG: {"agent":"calendar", "subtask":"create_event", "title":"Meeting"}
RIGHT: {"agent":"calendar", "subtask":"create event 'Meeting' for Bob from 2024-05-17 10:30:00 to 2024-05-17 11:00:00"}

Output ONLY valid JSON. No markdown.""",

            "agent_prompts": {
                "calendar": """You are a CALENDAR specialist. Output ONLY a Python dict. No markdown. No backticks.

ACTIONS (use EXACTLY these formats):
{"app": "calendar", "action": "create_event", "user": "Bob", "summary": "Meeting", "time_start": "2024-05-17 10:30:00", "time_end": "2024-05-17 11:00:00"}
{"app": "calendar", "action": "list_events", "user": "Bob"}
{"app": "calendar", "action": "delete_event", "user": "Bob", "summary": "Meeting"}
{"app": "calendar", "action": "create_event", "user": "Bob", "summary": "Class", "time_start": "2024-05-15 16:00:00", "time_end": "2024-05-15 18:00:00", "location": "Room 101"}

RULES:
- Extract ALL details from the subtask: user, event name, start time, end time
- The "user" field MUST match the person mentioned in the subtask
  If subtask says "for Tom" → "user": "Tom"
  If subtask says "for Bob" → "user": "Bob"
  NEVER default to Bob if another name is specified
- The "summary" MUST be the event name from the subtask
- Dates: YYYY-MM-DD HH:MM:SS
- No end time mentioned → add 1 hour to start time
- If a location/place is mentioned in the subtask, include "location" field
- NEVER leave summary, time_start, or time_end empty
- Output ONE action only""",

                "email": """You are an EMAIL specialist. Output ONLY a Python dict. No markdown. No backticks.

ACTIONS (use EXACTLY these formats):
{"app": "email", "action": "send_email", "sender": "Bob", "recipient": "Alice", "subject": "Update", "content": "Hello"}
{"app": "email", "action": "list_emails", "user": "Bob"}

RULES:
- Output ONE action only
- For send_email: always include sender, recipient, subject, content
- For list_emails: use the correct username
- There is NO "search_emails" action. Use list_emails instead.""",

                "system": """Output ONLY: {"app": "system", "action": "finish_task", "answer": "your answer"}""",
            },

            # Level 2: Pattern-aware routing strategy
            "routing_strategy": {
                "single_cal_create":         {"episodic": True,  "entity": False, "reason": "entity won't change create plan"},
                "multi_cal_find_and_create": {"episodic": True,  "entity": True,  "reason": "need both users' profiles"},
                "remind_notify":             {"episodic": True,  "entity": True,  "reason": "need user for email targeting"},
                "email_query":               {"episodic": True,  "entity": True,  "reason": "need user email context"},
                "email_send":                {"episodic": True,  "entity": False, "reason": "just need past send examples"},
                "cal_query":                 {"episodic": True,  "entity": False, "reason": "just need past query examples"},
                "unknown":                   {"episodic": True,  "entity": True,  "reason": "unknown, read everything"},
            },

            "learned_rules": [],
        }
        print(f"  📋 [PM] DEFINED orchestrator prompt + 3 agent prompts")
        print(f"  📋 [PM] DEFINED routing strategy (7 patterns)")

        # ── Episodic + Semantic ──
        self.episodic = LEGOMemBank(embed_fn=embed_fn, embed_dim=embed_dim)
        print(f"  📋 [EM+SM] CREATED empty FAISS ({embed_dim}d)")

        # ── Entity Memory ──
        self.entity = {
            "users": {},
            "agents": {
                "calendar": {"actions": ["create_event","list_events","delete_event"], "domain": "scheduling"},
                "email": {"actions": ["send_email","list_emails"], "domain": "communication"},
            },
        }
        print(f"  📋 [ENT] DEFINED 2 agent profiles, 0 user profiles")

        # ── Level 3: Dynamic routing history ──
        self.routing_history = {}

        # ── Tracking ──
        self.memory_ops = {"reads": 0, "writes": 0}
        self.current_task_accesses = []

        print(f"  ✅ Bootstrap complete")

    # ── Logging ──
    def _log(self, mem_type, op, detail=""):
        self.current_task_accesses.append({"type": mem_type, "op": op, "detail": detail})
        icons = {"READ":"📖","WRITE":"📝","SKIP":"⏭️","CLEAR":"🧹","CONSUME":"🔥","COMPARE":"⚖️","ENRICH":"✨","FILTER":"🔍"}
        print(f"      {icons.get(op,'💾')} [{mem_type}] {op}: {detail}")
        if "READ" in op: self.memory_ops["reads"] += 1
        if op in ("WRITE","ENRICH","CLEAR"): self.memory_ops["writes"] += 1

    def _print_summary(self, title="MEMORY ACCESS SUMMARY"):
        if not self.current_task_accesses: return
        counts = {}
        for e in self.current_task_accesses:
            k = f"{e['type']}({e['op']})"
            counts[k] = counts.get(k, 0) + 1
        print(f"\n    ┌── {title} {'─'*(55-len(title))}")
        for mt in ["WM","PM","STM","EM","SM","ENT"]:
            parts = []
            for op in ["READ","WRITE","SKIP","CLEAR","ENRICH","FILTER","CONSUME","COMPARE"]:
                c = counts.get(f"{mt}({op})", 0)
                if c: parts.append(f"{c}{op[0]}")
            if parts: print(f"    │  {mt:6s} {' + '.join(parts)}")
        total = len(self.current_task_accesses)
        skips = sum(1 for e in self.current_task_accesses if e["op"] == "SKIP")
        print(f"    │  Total: {total} ops ({skips} skipped)")
        print(f"    └{'─'*60}")

    # ═══ PHASE 2: TASK INGESTION ═══
    def ingest_task(self, task_desc, username, task_date=None):
        self.current_task_accesses = []
        pattern, _ = self.stm.classify(task_desc)
        self._log("WM", "WRITE", f"task='{task_desc[:35]}...' user={username} date={task_date}")
        self._log("WM", "WRITE", f"pattern='{pattern}' step_history=[] shared_context={{}}")
        self.working_memory = {
            "current_task": task_desc, "current_user": username,
            "current_date": task_date, "pattern": pattern,
            "step_history": [], "agent_scratchpad": [],
            "shared_context": {}, "memories_read_this_task": set(),
        }

    # ═══ PHASE 3: MEMORY RETRIEVAL (3-Level Routing) ═══
    def route_query(self, task_desc, use_memory=True):
        result = {"procedural": None, "working": None, "stm_hit": None, "stm_layer": "MISS",
                  "episodic_orch": [], "agent_memories": {}, "entities": {},
                  "faiss_queries": 0, "routing_log": []}

        # Level 1: Static (always)
        self._log("PM", "READ", "orchestrator prompt (L1: always)")
        result["procedural"] = self.procedural["orchestrator_prompt"]
        if self.procedural.get("learned_rules"):
            result["procedural"] += "\n\nLEARNED FROM EXPERIENCE:\n"
            for r in self.procedural["learned_rules"]:
                result["procedural"] += f"  - {r['pattern']}: {r.get('lesson','')}\n"
            self._log("PM", "READ", f"{len(self.procedural['learned_rules'])} learned rules appended")
        result["routing_log"].append("L1: PM READ")

        self._log("WM", "READ", "current task context (L1: always)")
        result["working"] = self.working_memory.copy()
        result["routing_log"].append("L1: WM READ")

        if not use_memory:
            result["routing_log"].append("L1: Memory OFF")
            return result

        # Level 1: Always check STM
        self._log("STM", "READ", "pattern classifier + semantic cache (L1: always)")
        cached, layer = self.stm.lookup(task_desc)
        self.working_memory["memories_read_this_task"].add("stm")
        result["routing_log"].append(f"L1: STM CHECK → {layer}")

        if cached is not None:
            result["stm_hit"] = cached
            result["stm_layer"] = layer
            result["agent_memories"] = cached.get("agent_memories", {})
            # Level 3: Confidence check
            confidence = min(cached.get("success_count", 1) / 5.0, 1.0)
            if confidence >= 0.6:
                result["routing_log"].append(f"L3: STM confidence {confidence:.0%} → TRUST, skip EM/ENT")
                return result
            else:
                result["routing_log"].append(f"L3: STM confidence {confidence:.0%} → HINT only, read EM")

        # Level 2: Pattern-aware
        pattern = self.working_memory.get("pattern", "unknown")
        strategy = self.procedural["routing_strategy"].get(pattern, self.procedural["routing_strategy"]["unknown"])
        result["routing_log"].append(f"L2: pattern='{pattern}' → EM={strategy['episodic']}, ENT={strategy['entity']}")

        # Level 3: Dynamic override
        dynamic = self._get_dynamic_decisions(pattern)

        # Episodic
        should_read_em = strategy["episodic"]
        if should_read_em and dynamic.get("skip_episodic"):
            should_read_em = False
            result["routing_log"].append(f"L3: EM OVERRIDE (helped {dynamic['ep_rate']:.0%}<30%)")
        if should_read_em:
            self._log("EM", "READ", "past experiences (FAISS)")
            self._log("SM", "READ", "cosine search (internal)")
            raw = self.episodic.retrieve_for_orchestrator(task_desc, k=3)
            result["faiss_queries"] += 1
            self.working_memory["memories_read_this_task"].add("episodic")
            # Relevance filter
            relevant = []
            for mem in raw:
                if mem.embedding:
                    qe = self.episodic.embed_fn(task_desc).reshape(-1).astype('float32')
                    qe = qe / (np.linalg.norm(qe) + 1e-9)
                    me = np.array(mem.embedding).reshape(-1).astype('float32')
                    sim = float(np.dot(qe, me))
                    if sim >= 0.5: relevant.append(mem)
                    else: self._log("EM", "FILTER", f"'{mem.task_description[:30]}' sim={sim:.2f}<0.5")
                else: relevant.append(mem)
            result["episodic_orch"] = relevant
            for fm in relevant:
                for st in fm.subtask_memories:
                    if st.agent_type in ["calendar","email"]:
                        result["agent_memories"].setdefault(st.agent_type, []).append(st)
            for a in result["agent_memories"]:
                result["agent_memories"][a] = result["agent_memories"][a][:3]
            result["routing_log"].append(f"L2+L3: EM READ → {len(relevant)}/{len(raw)} passed filter")
        else:
            self._log("EM", "SKIP", f"pattern '{pattern}' doesn't need episodic")
            self._log("SM", "SKIP", "no FAISS needed")
            result["routing_log"].append(f"L2: EM SKIP")

        # Entity
        users_found = self._extract_entities(task_desc)
        should_read_ent = strategy["entity"]
        if should_read_ent and dynamic.get("skip_entity"):
            should_read_ent = False
            result["routing_log"].append(f"L3: ENT OVERRIDE (helped {dynamic['ent_rate']:.0%}<30%)")
        if should_read_ent and users_found:
            for user in users_found:
                self._log("ENT", "READ", f"profile for '{user}'")
                result["entities"][user] = self.entity["users"].get(user, {})
            self.working_memory["memories_read_this_task"].add("entity")
            result["routing_log"].append(f"L2: ENT READ for {users_found}")
        else:
            self._log("ENT", "SKIP", f"pattern '{pattern}' doesn't need entity")
            result["routing_log"].append(f"L2: ENT SKIP")

        return result

    def _get_dynamic_decisions(self, pattern):
        h = self.routing_history.get(pattern)
        if not h or h.get("total", 0) < 3: return {}
        r = {}
        et = h.get("episodic_total", 0)
        if et >= 3:
            r["ep_rate"] = h["episodic_helped"] / et
            if r["ep_rate"] < 0.3: r["skip_episodic"] = True
        nt = h.get("entity_total", 0)
        if nt >= 3:
            r["ent_rate"] = h["entity_helped"] / nt
            if r["ent_rate"] < 0.3: r["skip_entity"] = True
        return r

    # ═══ PHASE 5: Plan-driven loading ═══
    def plan_driven_load(self, delegation, task_desc):
        needed = extract_agents_from_plan(delegation)
        loaded = {}; fq = 0
        if needed:
            for agt in needed:
                self._log("EM", "READ", f"batch loading [{agt}] memories (FAISS)")
                mems = self.episodic.retrieve_for_agent(agt, task_desc, k=3)
                fq += 1; loaded[agt] = mems
        return loaded, fq

    # ═══ PHASE 8: Record observation ═══
    def write_working_step(self, agent, subtask, observation):
        self._log("WM", "WRITE", f"step [{agent}] → {observation[:50]}...")
        self.working_memory["step_history"].append({
            "agent": agent, "subtask": subtask, "observation": observation})
        self.working_memory["shared_context"][agent] = {
            "last_subtask": subtask, "last_observation": observation}
        self._log("WM", "WRITE", f"shared_context['{agent}'] updated (cross-agent)")

    # ═══ PHASE 9: MEMORY STORAGE (6 Write Rules) ═══
    def on_task_complete(self, task_desc, success, exec_result, steps):
        write_log = []
        self._log("WM", "READ", f"success={success}")

        if not success:
            write_log.append("W1: FAILED → store NOTHING")
        else:
            write_log.append("W1: SUCCESS → proceed")
            # W2: Pattern → STM
            pattern, _ = self.stm.classify(task_desc)
            existing = self.stm.pattern_cache.get(pattern)
            bundle = {
                "plan": " | ".join([f"[{s.get('agent','?')}] {s.get('subtask','')[:35]}" for s in steps]),
                "agent_memories": exec_result.get("_agent_memories", {}),
                "agents_used": exec_result.get("agents_used", []),
                "final_answer": exec_result.get("final_answer", ""),
            }
            if existing and existing.get("success_count", 0) >= 5:
                self._log("STM", "SKIP", f"'{pattern}' well-tested ({existing['success_count']}x)")
                write_log.append(f"W2: STM SKIP ('{pattern}' {existing['success_count']}x)")
            else:
                self._log("STM", "WRITE", f"pattern='{pattern}'")
                self.stm.store_bundle(task_desc, bundle, steps=steps)
                write_log.append(f"W2: STM WRITE pattern='{pattern}'")

            # W3: Person → Entity
            users = self._extract_entities(task_desc)
            for user in users:
                old = self.entity["users"].get(user, {})
                new_agents = exec_result.get("agents_used", [])
                if old.get("last_agents") == new_agents and old.get("last_success"):
                    self._log("ENT", "SKIP", f"'{user}' no new info")
                    write_log.append(f"W3: ENT SKIP {user}")
                else:
                    if user not in self.entity["users"]:
                        self.entity["users"][user] = {"task_count":0,"success_count":0,
                            "complaint_history":[],"agent_frequency":{},"success_rate":0}
                    p = self.entity["users"][user]
                    p["task_count"] = p.get("task_count",0) + 1
                    p["success_count"] = p.get("success_count",0) + 1
                    p["last_task"] = task_desc[:50]
                    p["last_agents"] = new_agents
                    p["last_success"] = True
                    for a in new_agents: p["agent_frequency"][a] = p["agent_frequency"].get(a,0)+1
                    p["complaint_history"].append(task_desc[:40])
                    if len(p["complaint_history"]) > 10: p["complaint_history"].pop(0)
                    p["success_rate"] = round(p["success_count"]/max(p["task_count"],1), 2)
                    self._log("ENT", "ENRICH", f"'{user}': tasks={p['task_count']} rate={p['success_rate']}")
                    write_log.append(f"W3: ENT ENRICH {user}")

        write_log.append("W4: EM deferred (Phase 10)")
        write_log.append("W5: PM deferred (Phase 10)")

        # W6: Update routing history
        if success:
            pattern, _ = self.stm.classify(task_desc)
            mems = self.working_memory.get("memories_read_this_task", set())
            self._update_history(pattern, mems, True)
            write_log.append(f"W6: History updated '{pattern}'")

        self._log("WM", "CLEAR", "task ended")
        write_log.append("WM CLEAR")

        print(f"\n    ┌── WRITE ROUTING LOG {'─'*40}")
        for e in write_log:
            icon = "📝" if "WRITE" in e or "ENRICH" in e else "⏭️" if "SKIP" in e or "deferred" in e else "🧹" if "CLEAR" in e else "❌" if "FAILED" in e else "✅"
            print(f"    │  {icon} {e}")
        print(f"    └{'─'*60}")
        self._print_summary()
        self.working_memory = {"current_task":None,"current_user":None,"current_date":None,
            "pattern":None,"step_history":[],"agent_scratchpad":[],"shared_context":{},
            "memories_read_this_task":set()}

    def _update_history(self, pattern, mems_read, success):
        if pattern not in self.routing_history:
            self.routing_history[pattern] = {"episodic_helped":0,"episodic_total":0,
                "entity_helped":0,"entity_total":0,"total":0}
        h = self.routing_history[pattern]; h["total"] += 1
        if "episodic" in mems_read:
            h["episodic_total"] += 1
            if success: h["episodic_helped"] += 1
        if "entity" in mems_read:
            h["entity_total"] += 1
            if success: h["entity_helped"] += 1

    # ═══ Helpers ═══
    def _extract_entities(self, task_desc):
        known = {"bob","tom","alice","david","jane","john","mary"}
        words = set(re.findall(r'[a-z]+', task_desc.lower()))
        return [w.capitalize() for w in words & known]

    def read_procedural(self, component):
        self._log("PM", "READ", f"prompt for '{component}'")
        if component == "orchestrator": return self.procedural["orchestrator_prompt"]
        return self.procedural["agent_prompts"].get(component, self.procedural["agent_prompts"].get("system",""))

    def write_episodic(self, trajectories, llm):
        self._log("EM", "WRITE", f"distilling {len(trajectories)} trajectories")
        return construct_memories(llm, self.episodic.embed_fn, trajectories, self.episodic)

    def save(self):
        self.episodic.save()

    def summary(self):
        r = len(self.procedural.get("learned_rules",[]))
        u = len(self.entity["users"])
        return f"PM:{len(self.procedural['agent_prompts'])}+{r}rules | STM:{self.stm.summary()} | EM:{self.episodic.stats()} | ENT:{u}users"


# ══════════════════════════════════════════════════════════════════
#  FILE SYSTEM ENGINE
# ══════════════════════════════════════════════════════════════════
def execute_fs_action(runner, action):
    app = action.get("app","").lower()
    act = action.get("action","").lower()
    tb = LOCAL_TESTBED[0]

    if app == "system" and act == "finish_task":
        answer = str(action.get("answer",""))
        _om(f"{tb}/data", exist_ok=True)
        with _oo(f"{tb}/data/answer.txt","w") as f: f.write(answer)
        return f"OBSERVATION: Task finished. Answer: {answer}"

    if app == "calendar" and act == "create_event":
        user = action.get("user","Bob")
        summary = action.get("summary","Meeting")
        ts = str(action.get("time_start","")).replace("-","").replace(":","").replace(" ","T")
        te = str(action.get("time_end","")).replace("-","").replace(":","").replace(" ","T")
        d = f"{tb}/calendar"; _om(d, exist_ok=True)
        ics = f"{d}/{user}.ics"
        location = action.get("location","")
        evt = f"BEGIN:VEVENT\r\nSUMMARY:{summary}\r\nDTSTART:{ts}\r\nDTEND:{te}\r\n"
        if location: evt += f"LOCATION:{location}\r\n"
        evt += "END:VEVENT\r\n"
        if _oe(ics):
            with _oo(ics) as f: old = f.read()
            content = old.replace("END:VCALENDAR", evt + "END:VCALENDAR")
        else:
            content = f"BEGIN:VCALENDAR\r\nVERSION:2.0\r\n{evt}END:VCALENDAR\r\n"
        with _oo(ics,"w") as f: f.write(content)
        return f"OBSERVATION: Successfully created event '{summary}' for {user} (start: {action.get('time_start','')}, end: {action.get('time_end','')})"

    if app == "calendar" and act == "list_events":
        user = action.get("user","Bob")
        ics_path = f"{tb}/calendar/{user}.ics"
        if _oe(ics_path):
            with _oo(ics_path,"r") as f: content = f.read()
            events = []
            for block in content.split("BEGIN:VEVENT"):
                if "SUMMARY:" in block:
                    evt = {}
                    for line in block.strip().split("\n"):
                        line = line.strip().replace("\r","")
                        if line.startswith("SUMMARY:"): evt["summary"] = line.split(":",1)[1]
                        elif line.startswith("DTSTART:"): evt["start"] = line.split(":",1)[1]
                        elif line.startswith("DTEND:"): evt["end"] = line.split(":",1)[1]
                    if evt: events.append(evt)
            if events:
                txt = f"Found {len(events)} events for {user}:\n"
                for e in events:
                    txt += f"  - {e.get('summary','?')} | Start: {e.get('start','?')} | End: {e.get('end','?')}\n"
                return f"OBSERVATION: {txt}"
        return f"OBSERVATION: No events found for {user}."

    if app == "email" and act == "send_email":
        recip = action.get("recipient","User"); sender = action.get("sender","Bob")
        subj = action.get("subject","Update"); content = action.get("content","")
        for user in [recip, sender]:
            d = f"{tb}/emails/{user}"; _om(d, exist_ok=True)
            with _oo(f"{d}/{subj}.eml","w") as f:
                f.write(f"From: {sender}@example.com\r\nTo: {recip}@example.com\r\nSubject: {subj}\r\nContent-Type: text/plain\r\n\r\n{content}\r\n")
        return f"OBSERVATION: Email sent from {sender} to {recip}, subject: '{subj}'"

    if app == "email" and act == "list_emails":
        user = action.get("user","Bob")
        email_dir = f"{tb}/emails/{user}"
        if _oe(email_dir) and _od(email_dir):
            eml_files = _og(f"{email_dir}/*.eml")
            if eml_files:
                result = f"Found {len(eml_files)} emails for {user}:\n"
                for ef in sorted(eml_files)[:10]:
                    try:
                        with _oo(ef,"r",errors="ignore") as f: content = f.read()
                        from_line = subject_line = date_line = ""; body = ""
                        for line in content.split("\n"):
                            l = line.strip().replace("\r","")
                            if l.lower().startswith("from:"): from_line = l
                            elif l.lower().startswith("subject:"): subject_line = l
                            elif l.lower().startswith("date:"): date_line = l
                        parts = content.split("\n\n",1)
                        if len(parts) > 1: body = parts[1].strip()[:300]
                        result += f"\n--- {os.path.basename(ef)} ---\n  {from_line}\n  {subject_line}\n  {date_line}\n  Body: {body}\n"
                    except: pass
                return f"OBSERVATION: {result}"
        return str(runner.execute_action_direct(action))

    if app == "email" and act not in ["send_email","list_emails","read_email"]:
        return f"OBSERVATION: Invalid action '{act}'. Use: send_email, list_emails"
    return str(runner.execute_action_direct(action))


# ══════════════════════════════════════════════════════════════════
#  EXECUTION ENGINE (Phases 2-9 per task)
# ══════════════════════════════════════════════════════════════════
def execute_task(task_desc, username, llm, runner, mem_mgr,
                 use_memory=True, max_orch_steps=8, task_date=None):

    result = {"task": task_desc, "steps": [], "final_answer": "",
              "agents_used": set(), "stm_layer_hit": "MISS", "faiss_queries": 0}

    # Phase 2: Task Ingestion
    mem_mgr.ingest_task(task_desc, username, task_date)

    # Phase 3: Memory Retrieval
    mem_context = mem_mgr.route_query(task_desc, use_memory)
    result["faiss_queries"] = mem_context["faiss_queries"]

    # Build orch context from routing result
    orch_context = f"Task: {task_desc}\nUser: {username}\n"
    if task_date: orch_context += f"(Today's date: {task_date})\n"

    agent_memories = {}

    if use_memory and mem_context["stm_hit"] is not None:
        result["stm_layer_hit"] = mem_context["stm_layer"]
        cached = mem_context["stm_hit"]
        cached_plan = cached.get("plan","")
        if cached_plan: orch_context += f"\nPROVEN PLAN FROM SIMILAR TASK:\n{cached_plan}\nFollow this pattern.\n"
        agent_memories = mem_context["agent_memories"]
    elif use_memory and mem_context["episodic_orch"]:
        orch_context += "\nPAST SUCCESSFUL EXPERIENCES:\n"
        for m in mem_context["episodic_orch"][:3]:
            orch_context += f"  Task: {m.task_description}\n  Plan: {m.high_level_plan}\n\n"
        agent_memories = mem_context["agent_memories"]

    if mem_context["entities"]:
        for name, profile in mem_context["entities"].items():
            if profile:
                # A10: surface ALL durable facts (e.g. manager/assistant), not just
                # usage stats, so entity memory can be task-critical when a task
                # references a relation ("Bob's manager") only ENT can resolve.
                _stats = ("task_count", "success_rate", "last_agents", "last_success",
                          "agent_frequency", "complaint_history", "last_task")
                facts = {k: v for k, v in profile.items() if k not in _stats}
                extra = ("; " + ", ".join(f"{k}={v}" for k, v in facts.items())) if facts else ""
                orch_context += (f"\nUser {name}: tasks={profile.get('task_count',0)}, "
                                 f"rate={profile.get('success_rate','?')}{extra}\n")

    # Phase 4+5+6+7+8 loop
    orch_prompt = mem_mgr.read_procedural("orchestrator")
    plan_driven_loaded = False

    for step in range(max_orch_steps):
        # Phase 4: Orch Inference
        recent = mem_mgr.working_memory["step_history"][-4:]
        hist = ""
        if recent:
            hist = "\nPREVIOUS DELEGATIONS AND RESULTS:\n"
            for h in recent:
                hist += f"  → [{h['agent']}] {h['subtask']}\n    Result: {h['observation'][:1500]}\n\n"

        remaining = max_orch_steps - step
        orch_resp = llm.call(orch_prompt,
            f"{orch_context}{hist}Steps remaining: {remaining}\nNext delegation:",
            caller="ORCH")
        print(f"    📋 ORCH: {orch_resp[:200]}")

        delegation = parse_action(orch_resp)
        if delegation is None:
            print(f"    ❌ ORCH parse failed"); break

        agent_type = delegation.get("agent","system")
        subtask = delegation.get("subtask","")

        # Phase 5: Orch Output — Loop detection
        current_key = f"{agent_type}:{subtask[:40]}"
        wm_steps = mem_mgr.working_memory["step_history"][-2:]
        if len(wm_steps) >= 2:
            prev = [f"{h['agent']}:{h['subtask'][:40]}" for h in wm_steps]
            if prev[0] == prev[1] == current_key:
                print(f"    🔄 ORCH loop! Forcing finish...")
                force = llm.call('Output ONLY: {"agent":"system","subtask":"finish_task","answer":"your answer"}',
                    f"Task: {task_desc}\nData:\n{hist}\nAnswer:", caller="ORCH-FORCE")
                delegation = parse_action(force) or {"agent":"system","subtask":"finish_task","answer":"Unable to determine"}
                agent_type = delegation.get("agent","system"); subtask = delegation.get("subtask","")

        # Phase 5: Plan-driven agent loading
        if use_memory and not plan_driven_loaded and result["stm_layer_hit"] == "MISS":
            loaded, fq = mem_mgr.plan_driven_load(delegation, task_desc)
            result["faiss_queries"] += fq
            for a, mems in loaded.items():
                if a not in agent_memories: agent_memories[a] = mems
            plan_driven_loaded = True

        # Check finish
        if agent_type == "system" and ("finish" in subtask.lower() or delegation.get("answer")):
            answer = delegation.get("answer", subtask)
            execute_fs_action(runner, {"app":"system","action":"finish_task","answer":answer})
            result["final_answer"] = answer
            result["steps"].append({"step":step,"agent":"system","subtask":"finish_task","result":answer})
            print(f"    ✓ FINISHED: {answer[:60]}"); break

        # Phase 6: Agent Inference
        agent_prompt = mem_mgr.read_procedural(agent_type)
        agent_mem_text = ""
        if use_memory and agent_type in agent_memories:
            mems = agent_memories[agent_type]
            if mems:
                agent_mem_text = "\nPAST EXPERIENCES:\n"
                for m in mems[:3]:
                    if isinstance(m, SubtaskMemory):
                        agent_mem_text += f"  {m.subtask_description} → tools: {m.tool_calls}\n"
                    elif isinstance(m, dict):
                        agent_mem_text += f"  {m.get('subtask_description','')} → tools: {m.get('tool_calls','')}\n"

        # Cross-agent shared context
        shared_text = ""
        if mem_mgr.working_memory["shared_context"]:
            others = {k:v for k,v in mem_mgr.working_memory["shared_context"].items() if k != agent_type}
            if others:
                shared_text = "\nWHAT OTHER AGENTS HAVE DONE:\n"
                for o, ctx in others.items():
                    shared_text += f"  [{o}] {ctx['last_subtask'][:60]}\n    Result: {ctx['last_observation'][:200]}\n"

        agent_obs = "No action taken"; agent_actions = []
        for agent_step in range(2):
            agent_hist = ""
            if agent_actions:
                agent_hist = f"\nYour previous action:\n  {agent_actions[-1]}\n  Result: {agent_obs[:150]}\n"
            agent_resp = llm.call(agent_prompt,
                f"{agent_mem_text}{shared_text}Subtask: {subtask}\nUser: {username}{agent_hist}\nAction:",
                caller=f"{agent_type.upper()}-AGT")
            print(f"      📝 {agent_type.upper()}: {agent_resp[:120]}")
            agent_action = parse_action(agent_resp)
            if agent_action is None:
                agent_obs = "Agent could not determine action"; break
            print(f"      ✅ Action: {agent_action}")
            if agent_actions and str(agent_action) == str(agent_actions[-1]):
                print(f"      🔄 Agent repeat, stopping"); break

            # Phase 7: Agent Output
            obs = execute_fs_action(runner, agent_action)
            agent_obs = str(obs); agent_actions.append(agent_action)
            if agent_action.get("action") in ["create_event","send_email","delete_event","finish_task"]: break
            if agent_action.get("action") in ["list_events","list_emails","read_file"]: continue

        # Phase 8: Environment Observation
        mem_mgr.write_working_step(agent_type, subtask, agent_obs)
        result["agents_used"].add(agent_type)
        result["steps"].append({"step":step,"agent":agent_type,"subtask":subtask,"result":agent_obs[:500]})
        print(f"    Step {step+1}: [{agent_type}] {subtask[:40]} → {agent_obs[:200]}")

    result["_agent_memories"] = agent_memories
    result["agents_used"] = list(result["agents_used"])
    return result


# ══════════════════════════════════════════════════════════════════
#  TASK FILTER
# ══════════════════════════════════════════════════════════════════
def filter_cal_email(all_tasks, repo):
    filtered = []
    for tid, si in all_tasks:
        try:
            with _oo(f"{repo}/tasks/{tid}/subtasks/{si}.json") as f: cfg = json.load(f)
            txt = cfg.get("task","").lower()
            evals = str(cfg.get("evaluation","")).lower()
            exclude = ["excel","xlsx","docx","word","pdf","ocr","image","png","jpg",
                       "spreadsheet","cell","score","midterm","salary","budget","shopping list",
                       "student","grade","file name","rename","delete file","duplicate",
                       "move all","swap","sort","split"]
            if any(k in txt for k in exclude): continue
            cal_words = ["calendar","meeting","schedule","event","dinner","travelling","workout","ics"]
            email_words = ["email","send email","mail","eml","notify","remind"]
            is_cal = any(k in txt for k in cal_words) or ".ics" in evals
            is_email = any(k in txt for k in email_words) or ".eml" in evals
            if is_cal or is_email: filtered.append((tid, si))
        except: pass
    return filtered


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s',
    handlers=[logging.FileHandler(f"{LOG_DIR}/run_{timestamp}.log"), logging.StreamHandler(sys.stdout)])
RESULTS_FILE = f"{LOG_DIR}/results_{timestamp}.json"
def save_results(d):
    with _oo(RESULTS_FILE, "w") as f: json.dump(d, f, indent=2)

from run_officebench_local import LocalOfficeBenchRunner
from free_legomem import LocalEmbedder
REPO_PATH = "./OfficeBench"


def main():
    print("=" * 70)
    print("  LEGOMem + MemoryManager Architecture")
    print("  1 Orchestrator + 2 Agents (calendar, email)")
    print("  6 Memory Types | 3-Level Read | 6 Write Rules | 10 Phases")
    print("=" * 70)

    # Phase 1: System Bootstrap
    embedder = LocalEmbedder()
    keys = [os.environ.get(f"CEREBRAS_KEY_{i}","") for i in range(1, 11)]
    llm = MultiLLM(keys, model="gpt-oss-120b")
    runner = LocalOfficeBenchRunner(REPO_PATH)
    all_tasks = runner.get_all_task_ids()
    cal_email = filter_cal_email(all_tasks, REPO_PATH)

    mem_mgr = MemoryManager(embed_fn=embedder.embed, embed_dim=embedder.dims)

    train_tasks = cal_email[:15]
    test_tasks = cal_email[:15]  # Same tasks for STM verification

    print(f"\n  Model:     {llm.model}")
    print(f"  Memory:    MemoryManager (WM+STM+PM+EM+SM+ENT)")
    print(f"  Tasks:     {len(cal_email)} cal+email ({len(train_tasks)} train, {len(test_tasks)} test)")

    results = {
        "timestamp": timestamp, "model": llm.model,
        "architecture": "MemoryManager + 3-Level Routing + 6 Write Rules",
        "stm_config": {"l1_type": "pattern_classifier", "l2_size": mem_mgr.stm.l2_size,
                        "l2_threshold": mem_mgr.stm.l2_threshold},
        "training": {"tasks":[], "passed":0, "failed":0, "total":0},
        "testing": {"tasks":[], "passed":0, "failed":0, "total":0},
    }

    # Resume
    completed_train = set(); completed_test = set(); successful = []
    prev = sorted(glob_module.glob(f"{LOG_DIR}/results_*.json"))
    if prev and prev[-1] != RESULTS_FILE:
        try:
            with _oo(prev[-1]) as f: old = json.load(f)
            results["training"] = old.get("training", results["training"])
            results["testing"] = old.get("testing", results["testing"])
            for t in results["training"]["tasks"]:
                completed_train.add(t["task_id"])
                if t.get("success"):
                    successful.append({"task_description":t.get("task",""), "success":True, "level":t.get("level",1)})
            for t in results["testing"]["tasks"]:
                completed_test.add(t["task_id"])
            print(f"  Resumed: {len(completed_train)}+{len(completed_test)} done")
        except: pass

    # ═══ PHASE 1→8: Training (no memory) ═══
    print(f"\n{'━'*70}\n  TRAINING: {len(train_tasks)} tasks (no memory)\n{'━'*70}")
    for tid, si in train_tasks:
        key = f"{tid}_{si}"
        if key in completed_train: continue
        try:
            cfg = runner.setup_task(tid, si)
            level = runner.get_level(tid)
            LOCAL_TESTBED[0] = str(runner.testbed)
            print(f"\n  Train {key} (L{level}): {cfg['task'][:55]}...")
            task_with_date = cfg["task"]
            if cfg.get("date"): task_with_date += f"\n(Today's date: {cfg['date']})"
            t_start = time.time()
            ex = execute_task(task_with_date, cfg["username"], llm, runner, mem_mgr,
                              use_memory=False, task_date=cfg.get("date"))
            t_elapsed = time.time() - t_start
            print(f"    ⏱️ Task time: {t_elapsed:.1f}s")
            ev = runner.evaluate_task(tid, si)
            s = "✅" if ev["success"] else "❌"
            print(f"    Eval: {s} ({ev['passed']}/{ev['total']}) Agents: {ex['agents_used']}")
            results["training"]["total"] += 1
            if ev["success"]:
                results["training"]["passed"] += 1
                successful.append({"task_description":cfg["task"], "success":True,
                    "steps":ex["steps"], "agents_used":ex["agents_used"], "level":level})
            else:
                results["training"]["failed"] += 1
            # Phase 9
            mem_mgr.on_task_complete(cfg["task"], ev["success"], ex, ex["steps"])
            results["training"]["tasks"].append({
                "task_id":key, "level":level, "task":cfg["task"][:100],
                "success":ev["success"], "checks_passed":ev["passed"],
                "checks_total":ev["total"], "steps":len(ex["steps"]),
                "agents_used":ex["agents_used"], "time_seconds":round(t_elapsed,2),
                "faiss_queries":ex.get("faiss_queries",0)})
            save_results(results)
        except Exception as e:
            print(f"    ⚠️ {str(e)[:100]}")
        time.sleep(1)

    # Phase 10: Memory Consolidation
    print(f"\n  Training: {results['training']['passed']}/{results['training']['total']}")
    if successful:
        print(f"  Building LTM from {len(successful)} trajectories...")
        stats = mem_mgr.write_episodic(successful, llm)
        mem_mgr.save()
        # Pre-populate STM
        print(f"  Pre-populating STM from training...")
        for traj in successful:
            td = traj.get("task_description","")
            bundle = {"plan":" | ".join([f"[{s.get('agent','?')}] {s.get('subtask','')[:50]}" for s in traj.get("steps",[])]),
                       "agent_memories":{},"agents_used":traj.get("agents_used",[]),"final_answer":""}
            mem_mgr.stm.store_bundle(td, bundle, steps=traj.get("steps",[]))
        print(f"  Memory: {mem_mgr.summary()}")

    # ═══ TESTING (with memory) ═══
    print(f"\n{'━'*70}\n  TESTING: {len(test_tasks)} tasks (WITH memory)\n  {mem_mgr.summary()}\n{'━'*70}")
    total_faiss = 0
    for tid, si in test_tasks:
        key = f"{tid}_{si}"
        if key in completed_test: continue
        try:
            cfg = runner.setup_task(tid, si)
            level = runner.get_level(tid)
            LOCAL_TESTBED[0] = str(runner.testbed)
            print(f"\n  Test {key} (L{level}): {cfg['task'][:55]}...")
            task_with_date = cfg["task"]
            if cfg.get("date"): task_with_date += f"\n(Today's date: {cfg['date']})"
            t_start = time.time()
            ex = execute_task(task_with_date, cfg["username"], llm, runner, mem_mgr,
                              use_memory=True, task_date=cfg.get("date"))
            t_elapsed = time.time() - t_start
            print(f"    ⏱️ Task time: {t_elapsed:.1f}s (STM: {ex.get('stm_layer_hit','?')})")
            ev = runner.evaluate_task(tid, si)
            s = "✅" if ev["success"] else "❌"
            print(f"    Eval: {s} ({ev['passed']}/{ev['total']}) Agents: {ex['agents_used']} "
                  f"STM: {ex.get('stm_layer_hit','?')} FAISS: {ex.get('faiss_queries',0)}")
            results["testing"]["total"] += 1
            if ev["success"]:
                results["testing"]["passed"] += 1
            else:
                results["testing"]["failed"] += 1
            total_faiss += ex.get("faiss_queries", 0)
            # Phase 9
            mem_mgr.on_task_complete(cfg["task"], ev["success"], ex, ex["steps"])
            results["testing"]["tasks"].append({
                "task_id":key, "level":level, "task":cfg["task"][:100],
                "success":ev["success"], "checks_passed":ev["passed"],
                "checks_total":ev["total"], "steps":len(ex["steps"]),
                "agents_used":ex["agents_used"], "stm_hit":ex.get("stm_layer_hit","MISS"),
                "time_seconds":round(t_elapsed,2), "faiss_queries":ex.get("faiss_queries",0)})
            save_results(results)
        except Exception as e:
            print(f"    ⚠️ {str(e)[:100]}")
        time.sleep(1)

    # ═══ SUMMARY ═══
    tr = results["training"]; te = results["testing"]
    results["memory_state"] = {
        "stm": {"l1_hits":mem_mgr.stm.stats["l1_hits"],"l2_hits":mem_mgr.stm.stats["l2_hits"],
                "misses":mem_mgr.stm.stats["misses"]},
        "episodic": mem_mgr.episodic.stats(),
        "entity_users": len(mem_mgr.entity["users"]),
        "learned_rules": len(mem_mgr.procedural.get("learned_rules",[])),
        "total_reads": mem_mgr.memory_ops["reads"],
        "total_writes": mem_mgr.memory_ops["writes"],
    }

    train_times = [t["time_seconds"] for t in tr["tasks"] if "time_seconds" in t]
    test_times = [t["time_seconds"] for t in te["tasks"] if "time_seconds" in t]
    test_hit = [t["time_seconds"] for t in te["tasks"] if t.get("stm_hit","").startswith("STM")]
    test_miss = [t["time_seconds"] for t in te["tasks"] if t.get("stm_hit") == "MISS"]
    avg_tr = sum(train_times)/len(train_times) if train_times else 0
    avg_te = sum(test_times)/len(test_times) if test_times else 0
    avg_hit = sum(test_hit)/len(test_hit) if test_hit else 0
    avg_miss = sum(test_miss)/len(test_miss) if test_miss else 0
    results["timing"] = {"avg_train":round(avg_tr,2),"avg_test":round(avg_te,2),
        "avg_hit":round(avg_hit,2),"avg_miss":round(avg_miss,2)}
    save_results(results)

    print(f"\n{'━'*70}")
    print(f"  RESULTS — MemoryManager Architecture")
    print(f"{'━'*70}")
    print(f"  Train (no memory):  {tr['passed']}/{tr['total']} ({tr['passed']/max(tr['total'],1)*100:.0f}%)")
    print(f"  Test (with memory): {te['passed']}/{te['total']} ({te['passed']/max(te['total'],1)*100:.0f}%)")
    print(f"\n  Memory: {mem_mgr.summary()}")
    print(f"  FAISS queries (testing): {total_faiss}")
    print(f"  API calls: {llm.total_calls}")
    print(f"\n  Timing:")
    print(f"    Train: avg {avg_tr:.1f}s | total {sum(train_times):.0f}s")
    print(f"    Test:  avg {avg_te:.1f}s | total {sum(test_times):.0f}s")
    if test_hit: print(f"    HIT:   avg {avg_hit:.1f}s ({len(test_hit)} tasks)")
    if test_miss: print(f"    MISS:  avg {avg_miss:.1f}s ({len(test_miss)} tasks)")
    if avg_miss > 0 and avg_hit > 0: print(f"    Speedup: {avg_miss/avg_hit:.1f}x")
    print(f"\n  Entity profiles:")
    for n, p in mem_mgr.entity["users"].items():
        print(f"    {n}: tasks={p.get('task_count',0)} rate={p.get('success_rate',0)}")
    print(f"\n  Routing history:")
    for p, h in mem_mgr.routing_history.items():
        er = h["episodic_helped"]/max(h["episodic_total"],1) if h["episodic_total"]>0 else 0
        nr = h["entity_helped"]/max(h["entity_total"],1) if h["entity_total"]>0 else 0
        print(f"    {p}: EM {h['episodic_helped']}/{h['episodic_total']} ({er:.0%}), ENT {h['entity_helped']}/{h['entity_total']} ({nr:.0%})")
    print(f"{'━'*70}")


if __name__ == "__main__":
    try:
        main()
    finally:
        if hasattr(sys.stdout, 'close'): sys.stdout.close()
        if hasattr(sys.stderr, 'close'): sys.stderr.close()