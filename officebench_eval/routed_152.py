"""Compute the 2D-ROUTED accuracy on the full 152 and compare to pure 1D (improved+PERF).

Routing policy (a-priori, NO oracle): a task is handled by the 2D deterministic fan-out fast-path
iff subagent_parallel._is_fanout_task(task) fires (per-person calendar-event fan-out read from a
grid); every other task uses the 1D improved+PERF result. So 2D-routed >= 1D by construction on the
non-fired tasks, and differs only on the fired set. Prints the per-task breakdown + the headline.
"""
import json, os, sys
REPO = "/Users/chirag/Documents/Agentic_MM"
sys.path.insert(0, REPO)
os.chdir(os.path.join(REPO, "OfficeBench"))
import officebench_eval.subagent_parallel as sp

full = json.load(open(os.path.join(REPO, "officebench_eval/full152.json")))
oneD = json.load(open(os.path.join(REPO, "officebench_eval/par_rt_improved_progress.json")))
twoD = json.load(open(os.path.join(REPO, "officebench_eval/twod_elig20_progress.json")))


def s1(k):
    v = oneD.get(k + "|pccr"); return bool(v and v.get("success"))


def rt1(k):
    v = oneD.get(k + "|pccr")
    return (v.get("prompt_tokens", 0) + v.get("completion_tokens", 0)) if v else 0


def s2(k):
    v = twoD.get(k); return bool(v and v.get("success"))


def rt2(k):
    v = twoD.get(k); return v.get("real_tokens", 0) if v else 0


fired, rows = [], []
for it in full:
    k = "%s/%s" % (it["task"], it["subtask"])
    try:
        c = json.load(open("tasks/%s/subtasks/%s.json" % (it["task"], it["subtask"])))
    except Exception:
        continue
    if sp._is_fanout_task(c["task"]):
        fired.append(k); rows.append((k, s1(k), s2(k), rt1(k), rt2(k)))

oneD_total = sum(1 for v in oneD.values() if v.get("success"))
routed = oneD_total + sum((1 if s2(k) else 0) - (1 if s1(k) else 0) for k in fired)

print("2D routing fires on %d tasks (calendar-event fan-out from a grid):\n" % len(fired))
print("  %-8s  %-8s  %-8s  %10s  %10s" % ("task", "1D", "2D", "1D_real", "2D_real"))
for k, a, b, ra, rb in rows:
    tag = "  <-- GAIN" if (b and not a) else ("  <-- REGRESS" if (a and not b) else "")
    print("  %-8s  %-8s  %-8s  %10d  %10d%s" % (k, "PASS" if a else "fail", "PASS" if b else "fail", ra, rb, tag))
fa, fb = sum(r[1] for r in rows), sum(r[2] for r in rows)
ta, tb = sum(r[3] for r in rows), sum(r[4] for r in rows)
print("\n  fired-set accuracy:  1D %d/%d  ->  2D %d/%d" % (fa, len(rows), fb, len(rows)))
print("  fired-set real tokens: 1D %d  ->  2D %d  (%.1fx)" % (ta, tb, ta / max(1, tb)))
print("\nHEADLINE  ->  1D improved+PERF = %d/152   |   2D-routed = %d/152" % (oneD_total, routed))
