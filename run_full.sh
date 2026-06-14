#!/usr/bin/env bash
# Full PCCR experiment (corrected config): strict semantic-only STM + held-out
# split, so the rho=utility/cost cascade body actually runs on unseen tasks.
#
# 5 runs over the OfficeBench cal/email pool (default 34 tasks, 70/30 split):
#   1. pccr        @ threshold 1.0   (our router)
#   2. boolean     @ threshold 1.0   (base-script routing baseline)
#   3. retrieve_all@ threshold 1.0   (query-everything baseline)
#   4. pccr        @ threshold 1.2   (frontier point)
#   5. pccr        @ threshold 1.4   (frontier point)
# Every run's full stdout+stderr -> its own log; summaries -> driver log.
set -u

cd "$(dirname "$0")"
set -a; source cerebras.env; set +a

export PCCR_STRICT_STM=1
export PCCR_HELDOUT=1
export PCCR_SPLIT="${PCCR_SPLIT:-0.7}"
export PCCR_NTASKS="${PCCR_NTASKS:-34}"

LOGDIR="logs_memory_manager"; mkdir -p "$LOGDIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
DRIVER_LOG="$LOGDIR/full_driver_${STAMP}.log"

echo "=== PCCR FULL experiment (strict STM + held-out) started $(date) ===" | tee "$DRIVER_LOG"
echo "pool=$PCCR_NTASKS split=$PCCR_SPLIT strict_stm=1 heldout=1" | tee -a "$DRIVER_LOG"

run_one () {   # $1=mode  $2=threshold  $3=tag
  local MODE="$1" THR="$2" TAG="$3"
  local RUN_LOG="$LOGDIR/full_${TAG}_${STAMP}.log"
  echo "" | tee -a "$DRIVER_LOG"
  echo ">>> [$(date +%H:%M:%S)] $TAG (mode=$MODE thr=$THR) -> $RUN_LOG" | tee -a "$DRIVER_LOG"
  PCCR_MODE="$MODE" PCCR_THRESHOLD="$THR" \
    .venv/bin/python pccr_on_top_of_legomem.py > "$RUN_LOG" 2>&1
  echo ">>> [$(date +%H:%M:%S)] $TAG done (exit $?)" | tee -a "$DRIVER_LOG"
  grep -h "mode=$MODE thr=$THR" "$RUN_LOG" | tail -1 | tee -a "$DRIVER_LOG" || true
}

run_one pccr         1.0 pccr_thr1.0
run_one boolean      1.0 boolean
run_one retrieve_all 1.0 retrieve_all
run_one pccr         1.2 pccr_thr1.2
run_one pccr         1.4 pccr_thr1.4

echo "" | tee -a "$DRIVER_LOG"
echo "=== FULL experiment complete $(date) ===" | tee -a "$DRIVER_LOG"
ls -1t "$LOGDIR"/pccr_*_thr*.json | head -5 | tee -a "$DRIVER_LOG"
