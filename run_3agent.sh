#!/usr/bin/env bash
# 3-AGENT multi-agent experiment: calendar + email + word.
# 92-task pool, held-out 60 train / 32 test, strict semantic STM.
# OPTIMIZED: train + consolidate ONCE, then test all 3 routing modes from the
# same consolidated memory (snapshot/restore) -> ~43% fewer task-runs.
set -u
cd "$(dirname "$0")"
set -a; source cerebras.env; set +a

export PCCR_AGENTS=3
export PCCR_STRICT_STM=1
export PCCR_HELDOUT=1
export PCCR_TRAIN_N=60                       # 60 train -> 32 test (pool 92)
export PCCR_NTASKS=0                         # full pool
export PCCR_THRESHOLD=1.0
export PCCR_MODES="pccr,boolean,retrieve_all"   # train once, test these modes
export PCCR_FREEZE_TEST=1                        # frozen test memory (cleanest comparison)

LOGDIR="logs_memory_manager"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOGDIR/agent3_multimode_${STAMP}.log"

echo "=== 3-AGENT multimode experiment started $(date) ===" | tee "$LOGDIR/agent3_driver_${STAMP}.log"
echo "pool=92 train=60 test=32 modes=pccr,boolean,retrieve_all (train-once)" | tee -a "$LOGDIR/agent3_driver_${STAMP}.log"
echo "  -> full log: $RUN_LOG" | tee -a "$LOGDIR/agent3_driver_${STAMP}.log"

.venv/bin/python pccr_on_top_of_legomem.py > "$RUN_LOG" 2>&1
echo ">>> done (exit $?) $(date)" | tee -a "$LOGDIR/agent3_driver_${STAMP}.log"
grep -hE "mode=(pccr|boolean|retrieve_all) thr=" "$RUN_LOG" | tee -a "$LOGDIR/agent3_driver_${STAMP}.log"
ls -1t "$LOGDIR"/pccr_*_thr1.0_*.json | head -3 | tee -a "$LOGDIR/agent3_driver_${STAMP}.log"
