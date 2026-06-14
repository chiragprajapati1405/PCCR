#!/usr/bin/env bash
# 3-mode PCCR comparison over the OfficeBench cal/email tasks.
# Runs {pccr, boolean, retrieve_all} sequentially at threshold 1.0, identical
# train/test set (34 tasks). Every run's full stdout+stderr is captured to its
# own log file under logs_memory_manager/, plus a combined driver log.
set -u

cd "$(dirname "$0")"
set -a; source cerebras.env; set +a

NTASKS="${PCCR_NTASKS:-34}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs_memory_manager"
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/comparison_driver_${STAMP}.log"

echo "=== PCCR 3-mode comparison started $(date) ===" | tee "$DRIVER_LOG"
echo "tasks=$NTASKS  modes=pccr,boolean,retrieve_all  threshold=1.0" | tee -a "$DRIVER_LOG"

for MODE in pccr boolean retrieve_all; do
  RUN_LOG="$LOGDIR/run_${MODE}_${STAMP}.log"
  echo "" | tee -a "$DRIVER_LOG"
  echo ">>> [$(date +%H:%M:%S)] MODE=$MODE  ->  $RUN_LOG" | tee -a "$DRIVER_LOG"
  PCCR_MODE="$MODE" PCCR_NTASKS="$NTASKS" PCCR_THRESHOLD=1.0 \
    .venv/bin/python pccr_on_top_of_legomem.py > "$RUN_LOG" 2>&1
  echo ">>> [$(date +%H:%M:%S)] MODE=$MODE done (exit $?)" | tee -a "$DRIVER_LOG"
  # surface the one-line summary into the driver log
  grep -h "mode=$MODE thr=" "$RUN_LOG" | tail -1 | tee -a "$DRIVER_LOG" || true
done

echo "" | tee -a "$DRIVER_LOG"
echo "=== all 3 modes complete $(date) ===" | tee -a "$DRIVER_LOG"
echo "result JSONs:" | tee -a "$DRIVER_LOG"
ls -1 "$LOGDIR"/pccr_*_thr1.0_*.json 2>/dev/null | tail -3 | tee -a "$DRIVER_LOG"
