#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
set -a; source cerebras.env; set +a
mkdir -p logs_memory_manager results_archive/a8_glm
ST=$(date +%Y%m%d_%H%M%S)
rm -f logs_memory_manager/.done_*
echo "=== A8: cross-model comparison on zai-glm-4.7 $(date) ==="
PCCR_MODEL=zai-glm-4.7 PCCR_NTASKS=0 PCCR_AGENTS=3 PCCR_ENT_CRITICAL=1 PCCR_RESUME=1 \
  PCCR_FREEZE_TEST=1 PCCR_SEED=1 PCCR_TRAIN_N=70 PCCR_MODES="pccr,boolean,retrieve_all" \
  caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > logs_memory_manager/a8_glm_${ST}.log 2>&1
echo ">>> A8 done (exit $?) $(date)"
for m in pccr boolean retrieve_all; do
  f=$(ls -t logs_memory_manager/pccr_${m}_thr1.0_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" results_archive/a8_glm/${m}.json
done
grep -h "mode=.* thr=1.0" logs_memory_manager/a8_glm_${ST}.log | tail -3
echo "=== A8 COMPLETE $(date) ==="
