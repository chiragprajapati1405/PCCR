"""Smoke run on ONLY the tasks that emitted poisoned action names (run/list_directory/
write_cell) in seq152 -- validates the two memory-poison fixes (pm_curated[doc_process]
write_cell -> set_cell, and _clean_past_action sanitising injected past-actions).
Same recipe/driver as run_full_clean (sequential, fixed rules), separate trace dir.
Compares against the seq152 baseline for these exact tasks (prev 9/23 passed)."""
import json, os, time
REPO = "/Users/chirag/Documents/Agentic_MM"
from officebench_eval.real_mem import RealMem
from officebench_eval.real_arch import RealArch
from officebench_eval.runner import run_task, cap_for_level

BANK = os.path.join(REPO, "officebench_eval/em_bank.json")
OUT = os.path.join(REPO, "officebench_eval/smoke_affected_progress.json")
TR = os.path.join(REPO, "officebench_eval/traces/smoke_affected")
os.makedirs(TR, exist_ok=True)
SIM = float(os.environ.get("SIM_THRESHOLD", "0.45"))

tasks = json.load(open(os.path.join(REPO, "officebench_eval/smoke_affected.json")))
pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(os.path.join(REPO, "officebench_eval/patterns.json")))}
prev = json.load(open(os.path.join(REPO, "officebench_eval/seq152_progress.json")))
real = RealArch(BANK, {}, {}, theta=1.0)
rm = RealMem(BANK, theta=1.0, stm_threshold=0.85, confidence_gate=True, sim_threshold=SIM, curate_pm=True)
prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
print("SMOKE (poison-fix): clean-EM + A1 + B1 + D1(fixed) + sanitised past-actions | %d tasks" % len(tasks), flush=True)
t0 = time.perf_counter()
for it in tasks:
    key = "%s/%s" % (it["task"], it["subtask"])
    if key in prog:
        continue
    r = run_task(it["task"], it["subtask"], real_arch=real, real_mem=rm, method="pccr",
                 pattern=pats.get((it["task"], it["subtask"]), "multi_app"),
                 max_iter=cap_for_level(it["level"], l3_cap=45), container="ob-test-0",
                 replay=True, replay_threshold=0.75, output_convention=True,
                 completion_gate=True, self_verify=True)
    base = os.path.join(TR, "%s_%s" % (it["task"], it["subtask"]))
    json.dump(r, open(base + ".json", "w"), indent=2, default=str)
    with open(base + ".txt", "w") as fh:
        fh.write("TASK %s/%s [%s]  success=%s  steps=%d calls=%d tokens=%d fp=%s\n  %s\n%s\n%s\n" % (
            it["task"], it["subtask"], r.get("pattern"), r["success"], r["steps"], r["llm_calls"],
            r["em"]["injected_tokens"], r.get("failed_predicate"), r["task_text"], "=" * 72,
            "\n".join(r.get("sequence", []))))
    em = r["em"]
    prog[key] = {"success": r["success"], "level": it["level"], "pattern": pats.get((it["task"], it["subtask"])),
                 "prev_success": prev.get(key, {}).get("success"), "steps": r["steps"],
                 "tokens": em["injected_tokens"], "failed_predicate": r.get("failed_predicate")}
    json.dump(prog, open(OUT, "w"))
    p = sum(1 for v in prog.values() if v["success"])
    flip = "" if prog[key]["prev_success"] == r["success"] else (" FLIP+" if r["success"] else " FLIP-")
    print("  [%d/%d] %s: %s (prev %s)%s  running %d  (%.1f min)" % (
        len(prog), len(tasks), key, "PASS" if r["success"] else "fail",
        prog[key]["prev_success"], flip, p, (time.perf_counter() - t0) / 60), flush=True)

passed = sum(1 for v in prog.values() if v["success"])
prev_p = sum(1 for v in prog.values() if v["prev_success"])
print("\nSMOKE: %d/%d passed  (seq152 baseline on these: %d/%d)" % (passed, len(prog), prev_p, len(prog)), flush=True)
# malformed recount on smoke traces
import glob, re
mf = sum(open(f).read().count("Malformed action") for f in glob.glob(TR + "/*.txt"))
bad = sum(len(re.findall(r"--do-->\s+(run|list_directory|list_dir|write_cell)\b", open(f).read()))
          for f in glob.glob(TR + "/*.txt"))
print("malformed obs in smoke: %d | poisoned-action emissions remaining: %d" % (mf, bad), flush=True)
print("DONE", flush=True)
