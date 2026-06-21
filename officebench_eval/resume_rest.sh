#!/bin/bash
# Resume after the break: T0c cost -> T0c utility -> T1 (bank already built).
cd /Users/chirag/Documents/Agentic_MM || exit 1
source cerebras.env 2>/dev/null
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
PY=.venv/bin/python
echo "===== T0c cost ($(date)) =====";    $PY -m officebench_eval.calibrate cost
echo "===== T0c utility ($(date)) ====="; $PY -m officebench_eval.calibrate utility --per-pattern 6
echo "===== T1 ($(date)) =====";          $PY -m officebench_eval.run_t1 --theta 0.5
echo "===== RESUME DONE ($(date)) ====="
