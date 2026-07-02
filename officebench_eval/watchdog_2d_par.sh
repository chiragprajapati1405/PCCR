#!/bin/bash
# Autonomous watchdog for the N=4 pure-2D full-152 run. Same healing as watchdog_2d.sh but for the
# task-level-parallel driver (run_2d_par152). Resume is free (skips tasks in the progress JSON).
set +e
cd /Users/chirag/Documents/Agentic_MM
LOG=/tmp/twod_par152.log
PROG=officebench_eval/twod_par152_traces.json
STATUS=/tmp/twod_par152_watchdog.status
TRACEDIR=officebench_eval/traces/twod_par152
STALL=600                       # 10 min no log growth => stalled (N=4 tasks run longer)
export TWOD_TRACES="$PROG" TWOD_TRACEDIR="$TRACEDIR"
launch() {
  set -a; source cerebras.env 2>/dev/null; set +a
  nohup .venv/bin/python -m officebench_eval.run_2d_par152 >> "$LOG" 2>&1 &
  echo "$(date '+%H:%M:%S') launched pid $!" >> "$STATUS"
}
done_count() { .venv/bin/python -c "import json;d=json.load(open('$PROG'));print(len([k for k in d if k!='_run']))" 2>/dev/null || echo 0; }

echo "$(date '+%H:%M:%S') par-watchdog start (done=$(done_count))" > "$STATUS"
while true; do
  cur=$(done_count)
  if grep -q '^DONE' "$LOG" || [ "$cur" -ge 152 ]; then echo "$(date '+%H:%M:%S') COMPLETE done=$cur" >> "$STATUS"; break; fi
  alive=$(pgrep -f run_2d_par152 | head -1)
  now=$(date +%s); mt=$(stat -f %m "$LOG" 2>/dev/null || echo "$now"); age=$((now-mt))
  if [ -z "$alive" ]; then
    echo "$(date '+%H:%M:%S') process DEAD at done=$cur -> resume" >> "$STATUS"; sleep 10; launch; sleep 40
  elif [ "$age" -gt "$STALL" ]; then
    echo "$(date '+%H:%M:%S') STALL ${age}s at done=$cur -> heal" >> "$STATUS"
    pkill -9 -f run_2d_par152; sleep 25
    if ! docker ps >/dev/null 2>&1; then echo "$(date '+%H:%M:%S') docker unhealthy -> colima restart" >> "$STATUS"; colima restart >/dev/null 2>&1; sleep 20; fi
    for i in 0 1 2 3; do docker rm -f ob-2dpar-$i >/dev/null 2>&1; done
    launch; sleep 40
  fi
  TWOD_TRACES="$PROG" TWOD_TRACEDIR="$TRACEDIR" .venv/bin/python officebench_eval/emit_2d_traces.py >/dev/null 2>&1
  sleep 60
done

TWOD_TRACES="$PROG" TWOD_TRACEDIR="$TRACEDIR" .venv/bin/python officebench_eval/emit_2d_traces.py >> "$STATUS" 2>&1
TWOD_TRACES="$PROG" .venv/bin/python officebench_eval/build_2d_table.py > /tmp/twod_par152_table.txt 2>&1
echo "$(date '+%H:%M:%S') FINALIZED: traces in $TRACEDIR + table at /tmp/twod_par152_table.txt" >> "$STATUS"
echo "WATCHDOG_DONE" >> "$STATUS"
