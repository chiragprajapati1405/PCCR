"""LegoMem-style train/test split for full OfficeBench.

LegoMem uses 148 train / 152 test over the 300 annotated subtasks. Their exact
task-ID split isn't public, so we replicate the METHODOLOGY: stratify by
difficulty level (1/2/3) and split proportionally with a fixed seed, hitting
148/152 exactly. Train subtasks populate memory (consolidation); test subtasks
are evaluated. Writes officebench_eval/split.json.

  python -m officebench_eval.split --train 148 --seed 0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter, defaultdict

OB = "OfficeBench"


def all_subtasks():
    out = []
    for tid in sorted(os.listdir(f"{OB}/tasks")):
        lvl = tid.split("-")[0]
        if lvl not in ("1", "2", "3"):
            continue                              # skip our synthetic 8-/9- tasks
        for sj in sorted(glob.glob(f"{OB}/tasks/{tid}/subtasks/*.json")):
            sid = os.path.splitext(os.path.basename(sj))[0]
            out.append({"task": tid, "subtask": sid, "level": int(lvl), "config": sj})
    return out


def make_split(train_n=148, seed=0):
    items = all_subtasks()
    total = len(items)
    frac = train_n / total
    rng = random.Random(seed)
    by = defaultdict(list)
    for it in items:
        by[it["level"]].append(it)

    train, test = [], []
    for lvl in sorted(by):
        g = by[lvl][:]
        rng.shuffle(g)
        k = round(len(g) * frac)
        train += g[:k]
        test += g[k:]

    # adjust to hit exactly train_n (move boundary items, preserving stratification)
    while len(train) > train_n:
        test.append(train.pop())
    while len(train) < train_n:
        train.append(test.pop())
    return train, test, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=148)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="officebench_eval/split.json")
    args = ap.parse_args()

    train, test, total = make_split(args.train, args.seed)
    split = {"methodology": "LegoMem-style, stratified by level, seeded",
             "seed": args.seed, "total_subtasks": total,
             "train_n": len(train), "test_n": len(test),
             "train_by_level": dict(Counter(t["level"] for t in train)),
             "test_by_level": dict(Counter(t["level"] for t in test)),
             "train": train, "test": test}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(split, open(args.out, "w"), indent=2)
    print(f"total subtasks: {total}")
    print(f"train: {len(train)}  by level {split['train_by_level']}")
    print(f"test : {len(test)}  by level {split['test_by_level']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
