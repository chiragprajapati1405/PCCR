"""
run_officebench_local.py — a no-Docker local runner for OfficeBench, used by
the LEGOMem + MemoryManager harness (mm_on_top_of_legomem.py) and the PCCR
router on top of it.

OfficeBench natively runs agents inside a Docker container and evaluates files
left in a per-task `testbed/`. This runner replicates that locally:

  setup_task(tid, si)   copies tasks/<tid>/testbed/  ->  /tmp/officebench_testbed
                        (a clean dir) and returns the task config. The harness
                        monkey-patches "/testbed/" -> "/tmp/officebench_testbed",
                        so agent file actions land exactly where evaluation looks.
  evaluate_task(tid,si) runs the config's evaluation[] functions against that
                        testbed and returns {success, passed, total}.

Only the evaluation functions that calendar/email tasks use are implemented
here (contain / not_contain / file_exist / file_not_exist / exact_match /
calendar_no_overlap). The excel/word/pdf/ocr evaluators are intentionally
omitted because filter_cal_email() excludes those task types — this keeps the
dependency surface to icalendar + pytz (no docx/PyMuPDF/tesseract needed).
"""
from __future__ import annotations

import os
import json
import glob
import shutil
from email import policy as _email_policy
from email.parser import BytesParser

LOCAL_TESTBED_DIR = "/tmp/officebench_testbed"


def _read_text(path: str) -> str:
    with open(path, "r", errors="ignore") as f:
        return f.read()


def _read_docx(path: str) -> str:
    """Read a .docx into plain text (paragraphs joined by newlines)."""
    import docx
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def _read_doc_any(path: str, dtype: str) -> str:
    if dtype in ("docx", "doc"):
        return _read_docx(path)
    return _read_text(path)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _contains_keywords(content: str, keywords) -> bool:
    content = (content or "").lower()
    for kw in keywords:
        kw = str(kw).lower()
        hay = content.replace(",", "") if _is_number(kw) else content
        if kw not in hay:
            return False
    return True


def _list_email_text(testbed_dir: str, username: str) -> str:
    """Concatenate all .eml files for a user (mirrors OfficeBench's
    email_list_emails.list_emails output, full content)."""
    folder = None
    for cand in (username, username.lower()):
        p = os.path.join(testbed_dir, "emails", cand)
        if os.path.isdir(p):
            folder = p
            break
    if folder is None:
        return ""
    parts = []
    for ef in glob.glob(os.path.join(folder, "*.eml")):
        try:
            with open(ef, "rb") as f:
                msg = BytesParser(policy=_email_policy.default).parsebytes(f.read())
            body = msg.get_body(preferencelist=("plain", "html"))
            content = body.get_content() if body else (msg.get_payload() or "")
            parts.append(f"From: {msg['From']}\nTo: {msg['To']}\n"
                         f"Subject: {msg['Subject']}\nContent: {content}")
        except Exception:
            # fall back to raw file text so keyword checks still work
            parts.append(_read_text(ef))
    return "\n".join(parts)


# ── evaluation functions (subset used by calendar/email tasks) ──────────────
def _eval_contain(testbed_dir, args):
    dtype = args.get("doc_type")
    if dtype == "email":
        return _contains_keywords(_list_email_text(testbed_dir, args["username"]), args["keywords"])
    file_path = os.path.join(testbed_dir, args["file"])
    if not os.path.isfile(file_path):
        print(f"    !!! eval file not found: {file_path}")
        return False
    if dtype in ("txt", "ics", "docx", "doc"):
        try:
            return _contains_keywords(_read_doc_any(file_path, dtype), args["keywords"])
        except Exception as e:
            print(f"    !!! read error ({dtype}) {file_path}: {str(e)[:60]}")
            return False
    print(f"    !!! unsupported doc_type for local eval: {dtype}")
    return False


def _eval_not_contain(testbed_dir, args):
    return not _eval_contain(testbed_dir, args)


def _eval_file_exist(testbed_dir, args):
    return os.path.exists(os.path.join(testbed_dir, args["file"]))


def _eval_file_not_exist(testbed_dir, args):
    return not os.path.exists(os.path.join(testbed_dir, args["file"]))


def _eval_exact_match(testbed_dir, args):
    rp = os.path.join(testbed_dir, args["result_file"])
    ep = os.path.join(testbed_dir, args["expected_file"])
    if not (os.path.isfile(rp) and os.path.isfile(ep)):
        return False
    dtype = args.get("doc_type")
    if dtype in ("txt", "ics", "docx", "doc"):
        try:
            return _read_doc_any(rp, dtype) == _read_doc_any(ep, dtype)
        except Exception:
            return False
    print(f"    !!! unsupported exact_match doc_type: {dtype}")
    return False


def _eval_calendar_no_overlap(testbed_dir, args):
    import icalendar
    import pytz
    username = args["username"]
    cal_file = os.path.join(testbed_dir, "calendar", f"{username}.ics")
    if not os.path.isfile(cal_file):
        return False
    with open(cal_file, "rb") as f:
        cal = icalendar.Calendar.from_ical(f.read())
    utc = pytz.UTC

    def proc(dt):
        return utc.localize(dt) if getattr(dt, "tzinfo", None) is None else dt

    events = [c for c in cal.walk() if c.name == "VEVENT"]
    events.sort(key=lambda x: proc(x.get("dtstart").dt))
    for i in range(len(events) - 1):
        if proc(events[i].get("dtend").dt) > proc(events[i + 1].get("dtstart").dt):
            return False
    return True


_EVAL_DISPATCH = {
    "evaluate_contain": _eval_contain,
    "evaluate_not_contain": _eval_not_contain,
    "evaluate_file_exist": _eval_file_exist,
    "evaluate_file_not_exist": _eval_file_not_exist,
    "evaluate_exact_match": _eval_exact_match,
    "evaluate_calendar_no_overlap": _eval_calendar_no_overlap,
}


class LocalOfficeBenchRunner:
    def __init__(self, repo_path: str = "./OfficeBench", testbed_dir: str = LOCAL_TESTBED_DIR):
        self.repo = repo_path.rstrip("/")
        self.testbed = testbed_dir
        self.tasks_dir = os.path.join(self.repo, "tasks")
        if not os.path.isdir(self.tasks_dir):
            raise FileNotFoundError(f"OfficeBench tasks not found at {self.tasks_dir} "
                                    f"(clone github.com/zlwang-cs/OfficeBench into {self.repo})")

    # -- discovery --------------------------------------------------------
    def get_all_task_ids(self):
        out = []
        for cfg in glob.glob(os.path.join(self.tasks_dir, "*", "subtasks", "*.json")):
            parts = cfg.split(os.sep)
            tid = parts[-3]
            si = os.path.splitext(parts[-1])[0]
            out.append((tid, si))
        return sorted(out, key=lambda x: tuple(map(int, x[0].split("-"))) + (int(x[1]),))

    def get_level(self, tid: str) -> int:
        try:
            return int(tid.split("-")[0])
        except Exception:
            return 1

    # -- per-task lifecycle ----------------------------------------------
    def _config_path(self, tid, si):
        return os.path.join(self.tasks_dir, tid, "subtasks", f"{si}.json")

    def setup_task(self, tid, si):
        """Load the task config and lay down a clean testbed seeded from
        tasks/<tid>/testbed/ (if present)."""
        with open(self._config_path(tid, si)) as f:
            config = json.load(f)
        # fresh testbed
        if os.path.exists(self.testbed):
            shutil.rmtree(self.testbed)
        os.makedirs(self.testbed, exist_ok=True)
        seed = os.path.join(self.tasks_dir, tid, "testbed")
        if os.path.isdir(seed):
            for item in os.listdir(seed):
                src = os.path.join(seed, item)
                dst = os.path.join(self.testbed, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
        return config

    def evaluate_task(self, tid, si):
        """Run the config's evaluation[] against the current testbed."""
        with open(self._config_path(tid, si)) as f:
            config = json.load(f)
        eval_items = config.get("evaluation", []) or []
        passed = 0
        for item in eval_items:
            fn = _EVAL_DISPATCH.get(item.get("function"))
            if fn is None:
                print(f"    !!! no local evaluator for '{item.get('function')}' → fail")
                ok = False
            else:
                try:
                    ok = bool(fn(self.testbed, item.get("args", {})))
                except Exception as e:
                    print(f"    !!! eval error in {item.get('function')}: {str(e)[:80]}")
                    ok = False
            passed += int(ok)
        total = len(eval_items)
        return {"success": (total > 0 and passed == total), "passed": passed, "total": total}

    # -- fallback action executor ----------------------------------------
    def execute_action_direct(self, action):
        """Fallback for actions the harness's execute_fs_action doesn't handle.
        Calendar/email/system are handled upstream; we add the WORD app here
        (3rd agent) without touching the base script. Anything else is a no-op."""
        app = str(action.get("app", "")).lower()
        act = str(action.get("action", "")).lower()
        if app == "word":
            return self._run_word(action)
        # The base execute_fs_action delegates the empty-mailbox case here.
        if app == "email" and act == "list_emails":
            user = action.get("user", action.get("username", "Bob"))
            return f"OBSERVATION: No emails found for {user}."
        if app == "calendar" and act == "list_events":
            user = action.get("user", "Bob")
            return f"OBSERVATION: No events found for {user}."
        return (f"OBSERVATION: Unsupported action {app}/{act} "
                f"(no local executor). Available apps: calendar, email, word, system.")

    # -- WORD app (3rd agent) --------------------------------------------
    def _word_path(self, file_path: str) -> str:
        fp = str(file_path).lstrip("./")
        if not fp.endswith((".docx", ".doc")):
            fp += ".docx"
        return os.path.join(self.testbed, fp)

    def _run_word(self, action):
        import docx
        act = str(action.get("action", "")).lower().replace("word_", "")
        fp = action.get("file_path") or action.get("file") or "data/document.docx"
        abspath = self._word_path(fp)
        os.makedirs(os.path.dirname(abspath), exist_ok=True)
        if act in ("create_new_file", "create"):
            docx.Document().save(abspath)
            return f"OBSERVATION: Created Word document {fp}"
        if act in ("write_to_file", "write"):
            contents = str(action.get("contents", action.get("content", "")))
            doc = docx.Document(abspath) if os.path.isfile(abspath) else docx.Document()
            for line in contents.split("\n"):
                doc.add_paragraph(line)
            doc.save(abspath)
            return f"OBSERVATION: Wrote {len(contents)} chars to {fp}"
        if act in ("read_file", "read"):
            if not os.path.isfile(abspath):
                return f"OBSERVATION: Word document {fp} not found."
            try:
                text = "\n".join(p.text for p in docx.Document(abspath).paragraphs)
            except Exception as e:
                return f"OBSERVATION: Could not read {fp}: {str(e)[:50]}"
            return f"OBSERVATION: Contents of {fp}:\n{text[:1500]}"
        return f"OBSERVATION: Unknown word action '{act}'. Use create_new_file, write_to_file, read_file."
