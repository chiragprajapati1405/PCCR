#!/bin/bash
set +e; cd /Users/chirag/Documents/Agentic_MM
LOG=/tmp/pccr2d.log; PROG=officebench_eval/par_pccr2d_progress.json; ST=/tmp/pccr2d_watchdog.status
done_ct(){ .venv/bin/python -c "import json;print(sum(1 for v in json.load(open('$PROG')).values() if 'level' in v))" 2>/dev/null||echo 0; }
launch(){ set -a; source cerebras.env 2>/dev/null; set +a; nohup .venv/bin/python -m officebench_eval.run_t1_parallel --improved --twod --concurrency 4 --tasks-file officebench_eval/full152.json --tag pccr2d >> "$LOG" 2>&1 & echo "$(date '+%H:%M:%S') launched $!" >>"$ST"; }
echo "$(date '+%H:%M:%S') watchdog start done=$(done_ct)" >"$ST"
while true; do
  c=$(done_ct); [ "$c" -ge 152 ] && { echo "$(date '+%H:%M:%S') COMPLETE $c" >>"$ST"; break; }
  a=$(pgrep -f 'run_t1_parallel.*pccr2d'|head -1); now=$(date +%s); mt=$(stat -f %m "$LOG" 2>/dev/null||echo $now); age=$((now-mt))
  if [ -z "$a" ]; then echo "$(date '+%H:%M:%S') DEAD done=$c ->resume" >>"$ST"; sleep 10; launch; sleep 40
  elif [ "$age" -gt 720 ]; then echo "$(date '+%H:%M:%S') STALL ${age}s done=$c ->heal" >>"$ST"; pkill -9 -f 'run_t1_parallel.*pccr2d'; sleep 25; docker ps>/dev/null 2>&1||{ colima restart>/dev/null 2>&1; sleep 20; }; for i in 0 1 2 3; do docker rm -f ob-test-$i>/dev/null 2>&1; done; launch; sleep 40; fi
  sleep 60
done
echo "WATCHDOG_DONE" >>"$ST"
