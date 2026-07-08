"""Deferred eval: compare the SAVED per-task outputs (concurrent_mm/real_results/) against the
OfficeBench evaluators. This is the separate 'compare with eval' step — run_real.py stores the result
filesystem; this scores it, independently, whenever you want.

  set -a; source cerebras.env; set +a          # (not needed — eval is local/deterministic)
  python -m concurrent_mm.eval_real
"""
from __future__ import annotations

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "real_results")


def main():
    from officebench_eval.runner import _setup
    _setup()                                              # chdir into OfficeBench for the evaluators
    import utils.evaluate as ev

    dirs = sorted(d for d in os.listdir(RESULTS_DIR)
                  if os.path.isdir(os.path.join(RESULTS_DIR, d))) if os.path.isdir(RESULTS_DIR) else []
    if not dirs:
        print(f"no saved results in {RESULTS_DIR} — run `python -m concurrent_mm.run_real` first")
        return

    print(f"\nDEFERRED EVAL — comparing {len(dirs)} saved outputs against OfficeBench evaluators\n")
    print(f"{'task':>12} {'success':>8}  failed_predicate")
    succ = 0
    for d in dirs:
        base = os.path.join(RESULTS_DIR, d)
        spec = json.load(open(os.path.join(base, "_eval_spec.json")))
        testbed = os.path.join(base, "testbed")
        ok, fp = True, None
        for item in spec:
            try:
                if not getattr(ev, item["function"])(testbed, item["args"]):
                    ok, fp = False, item["function"]; break
            except Exception as e:
                ok, fp = False, f"{item['function']}:ERR:{str(e)[:40]}"; break
        succ += int(ok)
        print(f"{d:>12} {str(ok):>8}  {fp or ''}")
    print(f"\naccuracy: {succ}/{len(dirs)}  ({succ/len(dirs):.3f})")


if __name__ == "__main__":
    main()
