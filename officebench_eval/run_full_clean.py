"""Full 152-task improved-PCCR run, SEQUENTIAL main-thread (reliable: no thread/
process/FAISS-corruption issues). Validated recipe from the deep analysis:
clean EM (RealMem now uses an isolated tempdir) + A1 confidence gate + B1 replay
+ D1 curated PM + D3 output/completion convention + malformed-recovery (in build_prompt).
Records per-task results + traces; reports pass count vs retrieve_all baseline.
"""
import json, os, time
REPO = "/Users/chirag/Documents/Agentic_MM"
from officebench_eval.real_mem import RealMem
from officebench_eval.real_arch import RealArch
from officebench_eval.runner import run_task, cap_for_level

BANK = os.path.join(REPO, "officebench_eval/em_bank.json")
OUT = os.path.join(REPO, "officebench_eval/seq152_progress.json")
TR = os.path.join(REPO, "officebench_eval/traces/seq152")
os.makedirs(TR, exist_ok=True)
SIM = float(os.environ.get("SIM_THRESHOLD", "0.45"))

test = json.load(open(os.path.join(REPO, "officebench_eval/split.json")))["test"]
pats = {(p["task"], p["subtask"]): p["pattern"] for p in json.load(open(os.path.join(REPO, "officebench_eval/patterns.json")))}
real = RealArch(BANK, {}, {}, theta=1.0)
rm = RealMem(BANK, theta=1.0, stm_threshold=0.85, confidence_gate=True, sim_threshold=SIM, curate_pm=True)
prog = json.load(open(OUT)) if os.path.exists(OUT) else {}
print("FULL CLEAN run: clean-EM + A1(theta=%.2f) + B1 + D1 + D3 + malformed-recovery | %d tasks" % (SIM, len(test)), flush=True)
t0 = time.perf_counter()
for i, it in enumerate(test):
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
    with open(base + ".txt", "w") as fh:                       # readable step-by-step alongside the json
        fh.write("TASK %s/%s [%s]  success=%s  steps=%d calls=%d tokens=%d fp=%s\n  %s\n%s\n%s\n" % (
            it["task"], it["subtask"], r.get("pattern"), r["success"], r["steps"], r["llm_calls"],
            r["em"]["injected_tokens"], r.get("failed_predicate"), r["task_text"], "=" * 72,
            "\n".join(r.get("sequence", []))))
    em = r["em"]
    prog[key] = {"success": r["success"], "level": it["level"], "pattern": pats.get((it["task"], it["subtask"])),
                 "steps": r["steps"], "llm_calls": r["llm_calls"], "wall_s": r["wall_s"],
                 "tokens": em["injected_tokens"], "consults": int(em["consult_em"]), "stm_hit": int(em.get("stm_hit", False)),
                 "pm_used": bool(em.get("pm_used", False)), "top_em_sim": em.get("top_em_sim", 0.0),
                 "replay": em.get("replay"), "failed_predicate": r.get("failed_predicate")}
    json.dump(prog, open(OUT, "w"))
    p = sum(1 for v in prog.values() if v["success"])
    print("  [%d/%d] %s: %s  (running pass %d, %.0f min)" % (
        len(prog), len(test), key, "PASS" if r["success"] else "fail", p, (time.perf_counter()-t0)/60), flush=True)

passed = sum(1 for v in prog.values() if v["success"])
print("\nIMPROVED PCCR (clean, full recipe): %d/152 = %.3f  |  retrieve_all: 71/152 = 0.467" % (passed, passed/152), flush=True)
print("tokens total: %d (retrieve_all 734295)" % sum(v["tokens"] for v in prog.values()), flush=True)
print("DONE", flush=True)
