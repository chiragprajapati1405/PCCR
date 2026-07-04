"""AppWorld Phase 2: episodic-memory retriever over the train-ground-truth EM bank.

Embeds the 90 train instructions with all-MiniLM-L6-v2 (the PCCR embedder) and, for a new task,
returns the k most-similar solved procedures. This is the memory the rho-gate injects in Phase 3 --
the AppWorld analogue of officebench_eval real_arch/real_mem, minus the OfficeBench-specific pieces.
"""
import json
import os

import numpy as np

BANK = os.path.join(os.path.dirname(__file__), "em_bank_appworld.json")


class AppWorldEM:
    def __init__(self, bank_path=BANK, model_name="all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.bank = json.load(open(bank_path))
        self._model = SentenceTransformer(model_name)
        texts = [e["instruction"] for e in self.bank]
        self._emb = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def retrieve(self, instruction, k=2, exclude_task=None):
        """Top-k solved procedures most similar to `instruction` (cosine), skipping exclude_task."""
        q = self._model.encode([instruction], normalize_embeddings=True)[0]
        sims = self._emb @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            e = self.bank[i]
            if exclude_task and e["task_id"] == exclude_task:
                continue
            out.append((float(sims[i]), e))
            if len(out) >= k:
                break
        return out

    def inject_block(self, instruction, k=2, exclude_task=None, max_chars=1400):
        """A compact '[memory] similar solved tasks + their procedures' block for the agent prompt."""
        hits = self.retrieve(instruction, k=k, exclude_task=exclude_task)
        if not hits:
            return ""
        parts = ["[memory] similar solved tasks (reference procedures):"]
        for sim, e in hits:
            parts.append("### TASK (sim=%.2f): %s\napps: %s\n%s"
                         % (sim, e["instruction"], ", ".join(e["apps"]), e["procedure"]))
        return ("\n".join(parts))[:max_chars]


if __name__ == "__main__":
    from appworld import load_task_ids
    em = AppWorldEM()
    print("EM bank loaded: %d procedures\n" % len(em.bank))
    # probe: retrieve for a few dev tasks
    dev = load_task_ids("dev")
    for tid in dev[:3]:
        specs = json.load(open("data/tasks/%s/specs.json" % tid))
        instr = specs.get("instruction", "")
        print("DEV %s: %s" % (tid, instr[:90]))
        for sim, e in em.retrieve(instr, k=2):
            print("   -> sim=%.2f  [%s]  %s" % (sim, ",".join(e["apps"]), e["instruction"][:80]))
        print()
