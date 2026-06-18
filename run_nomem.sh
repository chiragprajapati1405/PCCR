#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
set -a; source cerebras.env; set +a
mkdir -p logs_memory_manager results_archive/a5_gptoss
ST=$(date +%Y%m%d_%H%M%S); rm -f logs_memory_manager/.done_*
echo "=== NO-MEMORY baseline on the SAME 30 held-out tasks (seed 1) $(date) ==="
PCCR_NTASKS=0 PCCR_AGENTS=3 PCCR_ENT_CRITICAL=1 PCCR_RESUME=1 PCCR_FREEZE_TEST=1 \
  PCCR_SEED=1 PCCR_TRAIN_N=70 PCCR_MODES="no_memory" \
  caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > logs_memory_manager/nomem_${ST}.log 2>&1
echo ">>> done (exit $?) $(date)"
f=$(ls -t logs_memory_manager/pccr_no_memory_thr1.0_*.json 2>/dev/null | head -1)
[ -n "$f" ] && cp "$f" results_archive/a5_gptoss/no_memory.json
grep -h "mode=no_memory" logs_memory_manager/nomem_${ST}.log | tail -1
