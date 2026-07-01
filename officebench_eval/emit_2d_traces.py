"""Render the machine JSON traces (twod_all152_traces.json) into human-readable per-task .txt files
in officebench_eval/traces/twod_all152/<task>_<sub>_2d.txt -- same folder convention + step style as
the 1D traces in traces/par_full152/, but showing the 2D decomposition: the plan (delegations) then
each parallel WAVE of sub-agents with their action + observation. Re-runnable; overwrites."""
import json, os
REPO = "/Users/chirag/Documents/Agentic_MM"
SRC = os.path.join(REPO, "officebench_eval/twod_all152_traces.json")
DST = os.path.join(REPO, "officebench_eval/traces/twod_all152")
os.makedirs(DST, exist_ok=True)

traces = json.load(open(SRC))
n = 0
for key, t in traces.items():
    tid, sid = t["task"], t["subtask"]
    lines = []
    hdr = ("TASK %s/%s [L%s] 2d  success=%s  path=%s  fanout_fired=%s  leaves=%s  "
           "real_tokens=%d calls=%d wall=%.1fs rate_limit_wait=%.1fs cost=$%.4f" % (
               tid, sid, t["level"], t["success"], t.get("path"), t.get("fanout_fired"),
               t.get("leaves"), t["real_tokens"], t["calls"], t["wall_s"],
               t.get("rate_limit_wait_s", 0.0), t.get("cost_usd", 0.0)))
    lines.append(hdr)
    lines.append("  " + t["task_text"])
    if t.get("failed_predicate"):
        lines.append("  FAILED PREDICATE: %s   (eval spec: %s)" % (t["failed_predicate"], ", ".join(t.get("eval_spec", []))))
    if t.get("error"):
        lines.append("  ERROR: %s" % t["error"])
    if t.get("items"):
        lines.append("  EXTRACTED ITEMS (%d): %s" % (len(t["items"]), ", ".join(map(str, t["items"]))))
    lines.append("=" * 70)
    # plan
    lines.append("PLAN (%d delegations):" % len(t.get("delegations", [])))
    for d in t.get("delegations", []):
        lines.append("  [%s] %s  (%s)" % (d["app"], d["subtask"], d.get("category", "")))
    lines.append("-" * 70)
    # walk the flat agent list, chunked by wave_sizes, into parallel WAVEs
    agents, sizes = t.get("agents", []), t.get("wave_sizes", [])
    i = 0
    for wi, sz in enumerate(sizes, 1):
        lines.append("WAVE %d (%d parallel sub-agent%s):" % (wi, sz, "" if sz == 1 else "s"))
        for a in agents[i:i + sz]:
            lines.append("  SUBAGENT[%s]  %s" % (a["app"], a["subtask"]))
            lines.append("    ACTION: %s" % a.get("action", ""))
            lines.append("    OBSERVATION: %s" % a.get("observation", ""))
        i += sz
    if i < len(agents):                          # any trailing agents not covered by wave_sizes
        lines.append("(unwaved sub-agents):")
        for a in agents[i:]:
            lines.append("  SUBAGENT[%s]  %s\n    ACTION: %s\n    OBSERVATION: %s" % (
                a["app"], a["subtask"], a.get("action", ""), a.get("observation", "")))
    open(os.path.join(DST, "%s_%s_2d.txt" % (tid, sid)), "w").write("\n".join(lines) + "\n")
    n += 1

print("wrote %d readable traces to %s" % (n, DST))
