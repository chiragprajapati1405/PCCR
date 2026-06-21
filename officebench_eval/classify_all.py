"""Classify all 300 OfficeBench subtasks and report the pattern distribution by
level — to decide empirically whether the 7-category enum is right-sized.

  source cerebras.env
  python -m officebench_eval.classify_all
"""
from __future__ import annotations

import concurrent.futures as cf
import glob
import json
import os
from collections import Counter, defaultdict

from .classifier import PATTERNS, TaskClassifier


def all_subtasks():
    out = []
    for tid in sorted(os.listdir("OfficeBench/tasks")):
        lvl = tid.split("-")[0]
        if lvl not in ("1", "2", "3"):
            continue
        for sj in sorted(glob.glob(f"OfficeBench/tasks/{tid}/subtasks/*.json")):
            sid = os.path.splitext(os.path.basename(sj))[0]
            cfg = json.load(open(sj))
            out.append({"task": tid, "subtask": sid, "level": int(lvl), "text": cfg["task"]})
    return out


def main():
    clf = TaskClassifier()
    items = all_subtasks()
    print(f"classifying {len(items)} subtasks (9-way concurrent)...")

    def work(it):
        return it, clf.classify(it["text"])

    results = []
    with cf.ThreadPoolExecutor(max_workers=9) as ex:
        for it, pat in ex.map(work, items):
            it["pattern"] = pat
            results.append(it)

    overall = Counter(r["pattern"] for r in results)
    bylvl = defaultdict(Counter)
    for r in results:
        bylvl[r["level"]][r["pattern"]] += 1

    print(f"\ntotal {len(results)} | llm_calls {clf.llm.calls}\n")
    print(f"{'pattern':<16}{'all':>6}{'L1':>6}{'L2':>6}{'L3':>6}")
    for p in PATTERNS:
        print(f"{p:<16}{overall[p]:>6}{bylvl[1][p]:>6}{bylvl[2][p]:>6}{bylvl[3][p]:>6}")
    # any labels outside the enum?
    extra = set(overall) - set(PATTERNS)
    if extra:
        print("OUT-OF-ENUM:", {p: overall[p] for p in extra})

    json.dump(results, open("officebench_eval/patterns.json", "w"), indent=2)
    print("\nsaved labels -> officebench_eval/patterns.json")


if __name__ == "__main__":
    main()
