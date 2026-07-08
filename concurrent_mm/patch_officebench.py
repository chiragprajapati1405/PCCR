"""REQUIRED for reproduction — apply the one-container namespacing patch to a fresh OfficeBench clone.

OfficeBench's calendar/email app scripts hardcode absolute /testbed paths, so multiple tasks can't
share one container. This script patches those 6 files to honor a TESTBED_ROOT env var (default
"/testbed", so existing single-task-per-container runs are UNAFFECTED). The one-container runner
(run_real_one.py) sets TESTBED_ROOT=/testbed/run_<id> per task, isolating every task.

Idempotent — safe to run repeatedly.

  git clone https://github.com/zlwang-cs/OfficeBench.git      # if not already cloned
  python -m concurrent_mm.patch_officebench                    # apply the patch
"""
from __future__ import annotations

import os
import py_compile

FILES = [
    "OfficeBench/apps/calendar_app/calendar_create_event.py",
    "OfficeBench/apps/calendar_app/calendar_list_events.py",
    "OfficeBench/apps/calendar_app/calendar_delete_event.py",
    "OfficeBench/apps/email_app/email_send_email.py",
    "OfficeBench/apps/email_app/email_read_email.py",
    "OfficeBench/apps/email_app/email_list_emails.py",
]
HELPER = '_TB = os.environ.get("TESTBED_ROOT", "/testbed")\n'


def patch_file(path: str) -> str:
    if not os.path.exists(path):
        return f"MISSING {path}"
    src = open(path).read()
    if "_TB = os.environ.get" in src:
        return f"already patched {os.path.basename(path)}"
    lines = src.splitlines(keepends=True)
    idx = 0
    for i, ln in enumerate(lines[:15]):
        if ln.startswith(("import ", "from ")):
            idx = i + 1
    lines.insert(idx, HELPER)
    src = "".join(lines)
    # f-strings first, then plain strings (order matters so we don't double-rewrite)
    src = src.replace("f'/testbed", "f'{_TB}").replace('f"/testbed', 'f"{_TB}')
    src = src.replace("'/testbed", "_TB+'").replace('"/testbed', '_TB+"')
    # fix the helper line itself if the plain-string rewrite touched its default
    src = src.replace('os.environ.get("TESTBED_ROOT", _TB+"")', 'os.environ.get("TESTBED_ROOT", "/testbed")')
    open(path, "w").write(src)
    py_compile.compile(path, doraise=True)
    return f"patched {os.path.basename(path)}"


def main():
    for f in FILES:
        print(" ", patch_file(f))
    print("done. run_real_one.py sets TESTBED_ROOT per task to isolate them in one container.")


if __name__ == "__main__":
    main()
