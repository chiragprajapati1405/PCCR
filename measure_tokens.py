"""
Measure the retrieved-memory TOKENS injected into the orchestrator context per
threshold — a cost metric that isn't polluted by rate-limit/wall-clock noise.

For each held-out test task we reconstruct exactly the memory block that
route_query/execute_task would inject when a store is consulted (episodic =
"PAST SUCCESSFUL EXPERIENCES" with the top-3 relevance-filtered episodes; entity
= the "User X: tasks=.. rate=.." line), count its tokens (~4 chars/token, the
standard approximation; tiktoken not installed), and sum over the tasks where
that threshold's decision log actually consulted the store.

This is a conservative LOWER bound on token savings: the injected memory also
appears in agent prompts and persists across the task's multiple LLM calls, so
real token savings are larger. We report the orchestrator-context injection.
"""
import glob
import json
import numpy as np

from run_officebench_local import LocalOfficeBenchRunner
from pccr_on_top_of_legomem import filter_cal_email_word, _reload_bank, PCCRMemoryManager
from free_legomem import LocalEmbedder

CHARS_PER_TOKEN = 4.0
ENT_TOKENS_IF_POPULATED = 12   # "User X: tasks=N, rate=R" ~12 tokens (nominal)


def toks(s):
    return int(len(s) / CHARS_PER_TOKEN)


def main():
    emb = LocalEmbedder()
    mm = PCCRMemoryManager(embed_fn=emb.embed, embed_dim=emb.dims,
                           strict_stm=True, enable_word_agent=True)
    n = _reload_bank(mm)
    runner = LocalOfficeBenchRunner("./OfficeBench")
    pool = filter_cal_email_word(runner.get_all_task_ids(), "./OfficeBench")
    test = pool[60:]   # held-out 32 (PCCR_TRAIN_N=60)
    print(f"reloaded {n} episodes | test tasks: {len(test)}")

    def task_text(tid, si):
        return json.load(open(f"./OfficeBench/tasks/{tid}/subtasks/{si}.json"))["task"]

    def em_injection_tokens(desc):
        """Replicate route_query's EM retrieval + 0.5 relevance filter, build the
        orchestrator 'PAST SUCCESSFUL EXPERIENCES' block, return its token count."""
        raw = mm.episodic.retrieve_for_orchestrator(desc, k=3)
        qe = emb.embed(desc).reshape(-1).astype("float32")
        qe = qe / (np.linalg.norm(qe) + 1e-9)
        rel = []
        for m in raw:
            if m.embedding:
                me = np.array(m.embedding).reshape(-1).astype("float32")
                me = me / (np.linalg.norm(me) + 1e-9)
                if float(np.dot(qe, me)) >= 0.5:
                    rel.append(m)
            else:
                rel.append(m)
        if not rel:
            return 0
        block = "PAST SUCCESSFUL EXPERIENCES:\n"
        for m in rel[:3]:
            block += f"  Task: {m.task_description}\n  Plan: {m.high_level_plan}\n\n"
        return toks(block)

    # token cost of EM injection per test task (in test order)
    em_tok = [em_injection_tokens(task_text(tid, si)) for tid, si in test]
    print(f"avg EM-injection size when consulted: ~{int(np.mean([t for t in em_tok if t]))} tokens\n")

    def latest_3agent(thr):
        for f in reversed(sorted(glob.glob(f"logs_memory_manager/pccr_pccr_thr{thr}_*.json"))):
            d = json.load(open(f))
            if d["config"].get("n_agents") == 3:
                return d
        return None

    print(f"{'theta':7s}{'EMconsults':12s}{'ENTconsults':13s}{'EM tokens':11s}{'ENT tokens':12s}{'TOTAL tokens':12s}")
    base = None
    for thr in ["1.0", "1.2", "1.4"]:
        d = latest_3agent(thr)
        dec = d["decision_log"]
        assert len(dec) == len(test), f"decision_log {len(dec)} != test {len(test)}"
        em_c = ent_c = em_t = ent_t = 0
        for i, dd in enumerate(dec):
            if "episodic" in dd["consulted"]:
                em_c += 1
                em_t += em_tok[i]
            if "entity" in dd["consulted"]:
                ent_c += 1
                ent_t += ENT_TOKENS_IF_POPULATED
        total = em_t + ent_t
        if base is None:
            base = total
        save = "" if thr == "1.0" else f"  (−{100*(base-total)/base:.0f}% vs θ=1.0)"
        print(f"{thr:7s}{em_c:<12d}{ent_c:<13d}{em_t:<11d}{ent_t:<12d}{total:<12d}{save}")

    print("\nNote: ~4 chars/token approx; ENT uses a nominal 12 tok/consult (ENT was")
    print("empty on this warm-resume run, so its real contribution here is ~0). EM is")
    print("the dominant retrieved content. This counts orchestrator-context injection")
    print("only — a conservative lower bound (agent prompts + multi-step repetition add more).")


if __name__ == "__main__":
    main()
