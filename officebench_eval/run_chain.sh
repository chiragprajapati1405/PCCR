#!/bin/bash
# Auto-proceed: T0a (build bank) -> T0c cost -> T0c utility -> T1 comparison.
# Each stage is checkpointed/resumable. Logs to officebench_eval/chain.log.
cd /Users/chirag/Documents/Agentic_MM || exit 1
source cerebras.env 2>/dev/null
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
PY=.venv/bin/python

echo "===== [1/4] T0a build EM bank  ($(date)) ====="
$PY -m officebench_eval.build_bank || { echo "T0a FAILED"; exit 1; }

echo "===== [2/4] T0c measure cost   ($(date)) ====="
$PY -m officebench_eval.calibrate cost || echo "cost step error (continuing)"

echo "===== [3/4] T0c measure utility ($(date)) ====="
$PY -m officebench_eval.calibrate utility --per-pattern 6 || echo "utility step error (continuing)"

echo "===== [4/4] T1 comparison      ($(date)) ====="
$PY -m officebench_eval.run_t1 --theta 0.5 || echo "T1 error"

echo "===== CHAIN DONE ($(date)) ====="
