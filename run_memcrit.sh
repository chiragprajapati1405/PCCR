#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
set -a; source cerebras.env; set +a
mkdir -p logs_memory_manager results_archive/memcrit
ST=$(date +%Y%m%d_%H%M%S); rm -f logs_memory_manager/.done_*
echo "=== MEMORY-CRITICAL comparison (8 ENT + 8 EM tasks), theta=0.18 $(date) ==="
PCCR_NTASKS=0 PCCR_AGENTS=3 PCCR_ENT_CRITICAL=1 PCCR_EM_CRITICAL=1 PCCR_MEMCRIT_ONLY=1 \
  PCCR_RESUME=1 PCCR_FREEZE_TEST=1 PCCR_HELDOUT=0 PCCR_THRESHOLD=0.18 \
  PCCR_MODES="no_memory,pccr,retrieve_all" \
  caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > logs_memory_manager/memcrit_${ST}.log 2>&1
echo ">>> done (exit $?) $(date)"
for m in no_memory pccr retrieve_all; do
  f=$(ls -t logs_memory_manager/pccr_${m}_thr0.18_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" results_archive/memcrit/${m}.json
done
grep -hE "mode=.* thr=0.18" logs_memory_manager/memcrit_${ST}.log
