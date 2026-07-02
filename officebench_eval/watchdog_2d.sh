#!/bin/bash
# Autonomous watchdog for the pure-2D full-152 run. Keeps it alive to completion, healing the known
# failure modes, then auto-emits per-task traces + the paper table. Resume is free: run_2d_all152.py
# skips tasks already in the progress JSON.
#   * process died with <152 done            -> relaunch (resume)
#   * log stalled > STALL secs (429 spiral /  -> kill, cooldown, (colima restart if docker unhealthy),
#     hang) while process still "alive"          relaunch (resume)
set +e
cd /Users/chirag/Documents/Agentic_MM
LOG=/tmp/twod152.log
PROG=officebench_eval/twod_all152_traces.json
STATUS=/tmp/twod152_watchdog.status
STALL=480                      # 8 min with no log growth => treat as stalled
launch() {
  set -a; source cerebras.env 2>/dev/null; set +a
  nohup .venv/bin/python -m officebench_eval.run_2d_all152 >> "$LOG" 2>&1 &
  echo "$(date '+%H:%M:%S') launched pid $!" >> "$STATUS"
}
done_count() { .venv/bin/python -c "import json;print(len(json.load(open('$PROG'))))" 2>/dev/null || echo 0; }

echo "$(date '+%H:%M:%S') watchdog start (done=$(done_count))" > "$STATUS"
while true; do
  cur=$(done_count)
  if grep -q '^DONE' "$LOG" || [ "$cur" -ge 152 ]; then
    echo "$(date '+%H:%M:%S') COMPLETE done=$cur" >> "$STATUS"; break
  fi
  alive=$(pgrep -f run_2d_all152 | head -1)
  now=$(date +%s); mt=$(stat -f %m "$LOG" 2>/dev/null || echo "$now"); age=$((now-mt))
  if [ -z "$alive" ]; then
    echo "$(date '+%H:%M:%S') process DEAD at done=$cur -> resume" >> "$STATUS"
    sleep 10; launch; sleep 30
  elif [ "$age" -gt "$STALL" ]; then
    echo "$(date '+%H:%M:%S') STALL ${age}s at done=$cur -> heal" >> "$STATUS"
    pkill -9 -f run_2d_all152; sleep 25
    if ! docker ps >/dev/null 2>&1; then echo "$(date '+%H:%M:%S') docker unhealthy -> colima restart" >> "$STATUS"; colima restart >/dev/null 2>&1; sleep 20; fi
    docker rm -f ob-2dall >/dev/null 2>&1
    launch; sleep 30
  fi
  # keep the readable trace folder current as tasks land (not just at the end)
  .venv/bin/python officebench_eval/emit_2d_traces.py >/dev/null 2>&1
  sleep 45
done

# ---- finalize: materialise all 152 trace pairs + the paper-column table ----
.venv/bin/python officebench_eval/emit_2d_traces.py >> "$STATUS" 2>&1
.venv/bin/python officebench_eval/build_2d_table.py > /tmp/twod152_table.txt 2>&1
echo "$(date '+%H:%M:%S') FINALIZED: traces emitted + table at /tmp/twod152_table.txt" >> "$STATUS"
echo "WATCHDOG_DONE" >> "$STATUS"
