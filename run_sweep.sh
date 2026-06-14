#!/usr/bin/env bash
# Threshold sweep for the cost/quality frontier. Waits for the 3-mode
# comparison (run_comparison.sh) to finish, then runs pccr at higher
# thresholds (1.2, 1.4) on the same 34-task identical set. Together with the
# threshold-1.0 pccr run from the comparison, this gives 3 frontier points.
set -u

cd "$(dirname "$0")"
set -a; source cerebras.env; set +a

WATCH_LOG="${1:-logs_memory_manager/comparison_driver_20260611_103407.log}"
NTASKS="${PCCR_NTASKS:-34}"
LOGDIR="logs_memory_manager"
mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_LOG="$LOGDIR/sweep_driver_${STAMP}.log"

echo "=== threshold sweep queued $(date) ===" | tee "$SWEEP_LOG"
echo "waiting for comparison to finish: $WATCH_LOG" | tee -a "$SWEEP_LOG"

# Wait (poll every 30s) until the comparison driver reports completion.
while ! grep -q "all 3 modes complete" "$WATCH_LOG" 2>/dev/null; do
  sleep 30
done
echo ">>> comparison finished, starting sweep $(date)" | tee -a "$SWEEP_LOG"

for THR in 1.2 1.4; do
  RUN_LOG="$LOGDIR/run_pccr_thr${THR}_${STAMP}.log"
  echo ">>> [$(date +%H:%M:%S)] PCCR threshold=$THR -> $RUN_LOG" | tee -a "$SWEEP_LOG"
  PCCR_MODE=pccr PCCR_NTASKS="$NTASKS" PCCR_THRESHOLD="$THR" \
    .venv/bin/python pccr_on_top_of_legomem.py > "$RUN_LOG" 2>&1
  echo ">>> [$(date +%H:%M:%S)] threshold=$THR done (exit $?)" | tee -a "$SWEEP_LOG"
  grep -h "mode=pccr thr=$THR" "$RUN_LOG" | tail -1 | tee -a "$SWEEP_LOG" || true
done

echo "=== sweep complete $(date) ===" | tee -a "$SWEEP_LOG"
ls -1 "$LOGDIR"/pccr_pccr_thr1.2_*.json "$LOGDIR"/pccr_pccr_thr1.4_*.json 2>/dev/null | tail -2 | tee -a "$SWEEP_LOG"
