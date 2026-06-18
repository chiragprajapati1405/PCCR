#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
set -a; source cerebras.env; set +a
mkdir -p logs_memory_manager
PCCR_ENT_CRITICAL=1 caffeinate -i .venv/bin/python calibrate.py utilities > logs_memory_manager/a2_utility_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "A2 done (exit $?)"
