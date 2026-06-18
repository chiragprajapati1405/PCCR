#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"
set -a; source cerebras.env; set +a
mkdir -p logs_memory_manager results_archive/a5_gptoss results_archive/a8_llama
ST=$(date +%Y%m%d_%H%M%S)
COMMON="PCCR_NTASKS=0 PCCR_AGENTS=3 PCCR_ENT_CRITICAL=1 PCCR_RESUME=1 PCCR_FREEZE_TEST=1 PCCR_SEED=1 PCCR_TRAIN_N=70 PCCR_MODES=pccr,boolean,retrieve_all"

echo "=== A5: full 100-task pool, 3 modes, seed 1 (gpt-oss-120b) $(date) ==="
rm -f logs_memory_manager/.done_*
env $COMMON caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > logs_memory_manager/a5_scale_${ST}.log 2>&1
echo ">>> A5 done (exit $?) $(date)"
# archive A5 JSONs so A8 (same filenames) doesn't overwrite/mix
for m in pccr boolean retrieve_all; do
  f=$(ls -t logs_memory_manager/pccr_${m}_thr1.0_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" results_archive/a5_gptoss/${m}.json
done
grep -h "mode=.* thr=1.0" logs_memory_manager/a5_scale_${ST}.log | tail -3

echo "=== A8: same comparison on llama-3.3-70b (cross-model) $(date) ==="
rm -f logs_memory_manager/.done_*
env PCCR_MODEL=llama-3.3-70b $COMMON caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > logs_memory_manager/a8_llama_${ST}.log 2>&1
echo ">>> A8 done (exit $?) $(date)"
for m in pccr boolean retrieve_all; do
  f=$(ls -t logs_memory_manager/pccr_${m}_thr1.0_*.json 2>/dev/null | head -1)
  [ -n "$f" ] && cp "$f" results_archive/a8_llama/${m}.json
done
grep -h "mode=.* thr=1.0" logs_memory_manager/a8_llama_${ST}.log | tail -3
echo "=== A5+A8 COMPLETE $(date) ==="
