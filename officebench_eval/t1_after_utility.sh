#!/bin/bash
cd /Users/chirag/Documents/Agentic_MM || exit 1
# wait for T0c utility to finish
until ! pgrep -f "calibrate utility" >/dev/null 2>&1; do sleep 20; done
echo "===== T0c UTILITY DONE ($(date)) ====="
cat officebench_eval/calibration/pattern_utility.json 2>/dev/null
source cerebras.env 2>/dev/null
export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
echo "===== T1 START ($(date)) ====="
.venv/bin/python -m officebench_eval.run_t1 --theta 0.5
echo "===== T1 DONE ($(date)) ====="
