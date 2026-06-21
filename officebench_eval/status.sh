#!/bin/bash
# Quick status of the OfficeBench eval chain. Run: bash officebench_eval/status.sh
cd /Users/chirag/Documents/Agentic_MM || exit 1
J() { .venv/bin/python -c "import json;print($1)" 2>/dev/null || echo "?"; }

pgrep -f run_chain.sh >/dev/null && echo "CHAIN: ✅ RUNNING" || echo "CHAIN: ⛔ not running"
echo "stage : $(grep -oE '\[[0-9]/4\][^(]*' officebench_eval/chain.log 2>/dev/null | tail -1)"
echo "T0a   : $(J 'len(json.load(open("officebench_eval/em_bank.json")))') procedures | train $(J 'len(json.load(open("officebench_eval/t0_progress.json")))')/148 done"
echo "T0c   : cost=$( [ -f officebench_eval/calibration/store_cost.json ] && echo done || echo pending ) utility=$( [ -f officebench_eval/calibration/pattern_utility.json ] && echo done || echo pending )"
echo "T1    : $(J 'len(json.load(open("officebench_eval/t1_progress.json")))')/456 runs done"
echo "--- last 4 log lines ---"
tail -4 officebench_eval/chain.log 2>/dev/null
