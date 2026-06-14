#!/usr/bin/env bash
# 3-AGENT threshold sweep for the cost/quality frontier.
# Train + consolidate ONCE, then test pccr at theta = 1.0, 1.2, 1.4 from the
# same frozen consolidated memory -> a self-consistent frontier (all points
# share identical memory; only the threshold varies).
set -u
cd "$(dirname "$0")"
set -a; source cerebras.env; set +a

export PCCR_AGENTS=3
export PCCR_STRICT_STM=1
export PCCR_HELDOUT=1
export PCCR_FREEZE_TEST=1
export PCCR_TRAIN_N=60
export PCCR_NTASKS=0
export PCCR_THRESHOLDS="1.0,1.2,1.4"        # pccr threshold sweep
export PCCR_RESUME=1                         # reload bank from disk, skip training

LOGDIR="logs_memory_manager"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOGDIR/agent3_sweep_${STAMP}.log"
DRIVER_LOG="$LOGDIR/agent3_sweep_driver_${STAMP}.log"

echo "=== 3-AGENT threshold sweep started $(date) ===" | tee "$DRIVER_LOG"
echo "pool=92 train=60 test=32 thresholds=1.0,1.2,1.4 (train-once, frozen)" | tee -a "$DRIVER_LOG"
echo "  -> full log: $RUN_LOG" | tee -a "$DRIVER_LOG"

# caffeinate -i keeps the Mac awake (-i = prevent idle sleep) so the LLM
# connections don't die mid-run like they did overnight.
caffeinate -i .venv/bin/python pccr_on_top_of_legomem.py > "$RUN_LOG" 2>&1
echo ">>> done (exit $?) $(date)" | tee -a "$DRIVER_LOG"
grep -hE "mode=pccr thr=" "$RUN_LOG" | tee -a "$DRIVER_LOG"
