"""Self-healing supervisor: run the 3 real-token arms SEQUENTIALLY at N=4, surviving
Colima/Docker OOM hangs unattended. For each arm: launch run_t1_parallel; watch the
progress file; on a stall (>STALL_S with no new completion) or a no-progress exit, kill,
restart Docker, and resume (same --tag skips completed tasks). Advance to the next arm
when all 152 land. When all three finish, write the cost-ledger comparison.

Run detached:  nohup .venv/bin/python -m officebench_eval.supervise_chain > officebench_eval/supervisor.log 2>&1 &
(env must already have CEREBRAS_KEY_* -- `source cerebras.env` before launching.)
"""
import json, os, signal, subprocess, time

REPO = "/Users/chirag/Documents/Agentic_MM"
PY = os.path.join(REPO, ".venv/bin/python")
TASKS = os.path.join(REPO, "officebench_eval/full152.json")
ARMS = [("perf152", ["--improved"])]
N = 152
STALL_S = 900          # 15 min: improved's gate-retry L3 tasks are legitimately long
POLL = 30
MAX_ATTEMPTS = 80      # per-arm safety cap (avoid infinite relaunch)
CONCURRENCY = "4"      # improved at N=4 OOMs Colima repeatedly; N=2 completes (real-token/accuracy
                       # are per-task = concurrency-independent; N=4 throughput already shown by ra/lean)
STATUS = os.path.join(REPO, "officebench_eval/supervisor_status.json")


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def prog_count(tag):
    p = os.path.join(REPO, "officebench_eval/par_%s_progress.json" % tag)
    if not os.path.exists(p):
        return 0
    try:
        return len([v for v in json.load(open(p)).values() if "level" in v])
    except Exception:
        return 0


def docker_up():
    return subprocess.call("docker info", shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def restart_docker():
    log("restarting Colima/Docker ...")
    subprocess.call("colima restart", shell=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(40):
        if docker_up():
            log("Docker back up")
            time.sleep(5)
            return True
        time.sleep(10)
    log("WARNING: Docker did not come back")
    return False


def write_status(stage, extra=None):
    s = {"stage": stage, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
         "counts": {tag: prog_count(tag) for tag, _ in ARMS}}
    if extra:
        s.update(extra)
    json.dump(s, open(STATUS, "w"), indent=2)


def launch(tag, flags):
    log_path = os.path.join(REPO, "officebench_eval/%s.log" % tag)
    cmd = [PY, "-m", "officebench_eval.run_t1_parallel"] + flags + \
          ["--tasks-file", TASKS, "--tag", tag, "--concurrency", CONCURRENCY]
    fh = open(log_path, "a")
    return subprocess.Popen(cmd, stdout=fh, stderr=fh, cwd=REPO, preexec_fn=os.setsid)


def kill(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass


def run_arm(tag, flags):
    attempts = 0
    while prog_count(tag) < N and attempts < MAX_ATTEMPTS:
        attempts += 1
        if not docker_up():
            restart_docker()
        before = prog_count(tag)
        log("arm=%s attempt=%d  start at %d/%d" % (tag, attempts, before, N))
        write_status("running:%s" % tag, {"attempt": attempts})
        proc = launch(tag, flags)
        last, last_t = before, time.time()
        while proc.poll() is None:
            time.sleep(POLL)
            c = prog_count(tag)
            if c >= N:
                break
            if c > last:
                last, last_t = c, time.time()
            elif time.time() - last_t > STALL_S:
                log("arm=%s STALLED at %d (no progress %ds) -> kill+restart" % (tag, c, STALL_S))
                kill(proc)
                restart_docker()
                break
        kill(proc)
        after = prog_count(tag)
        log("arm=%s attempt=%d ended at %d/%d" % (tag, attempts, after, N))
        if after >= N:
            break
        if after <= before:                  # no progress -> Docker likely wedged
            restart_docker()
            time.sleep(15)
    return prog_count(tag)


def summarize():
    rows = {}
    for tag, _ in ARMS:
        p = os.path.join(REPO, "officebench_eval/par_%s_progress.json" % tag)
        d = [v for v in json.load(open(p)).values() if "level" in v] if os.path.exists(p) else []
        n = len(d) or 1
        rows[tag] = {
            "n": len(d),
            "pass": sum(1 for v in d if v["success"]),
            "real_tokens": sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0) for v in d),
            "prompt_tokens": sum(v.get("prompt_tokens", 0) for v in d),
            "completion_tokens": sum(v.get("completion_tokens", 0) for v in d),
            "injected_tokens": sum(v.get("tokens", 0) for v in d),
            "llm_calls": sum(v.get("llm_calls", 0) for v in d),
            "stm_hits": sum(v.get("stm_hit", 0) for v in d),
            "wall_s": round(sum(v.get("wall_s", 0) for v in d), 1),
            "compute_s_429neglected": round(sum(v.get("compute_s", v.get("wall_s", 0)) for v in d), 1),
            "rate_limit_wait_s": round(sum(v.get("rate_limit_wait_s", 0) for v in d), 1),
            "real_tok_per_call": round(sum(v.get("prompt_tokens", 0) + v.get("completion_tokens", 0) for v in d) / max(1, sum(v.get("llm_calls", 0) for v in d)), 1),
        }
    out = os.path.join(REPO, "officebench_eval/COST_LEDGER.json")
    json.dump(rows, open(out, "w"), indent=2)
    log("=== COST LEDGER ===")
    for tag, r in rows.items():
        log("%-12s pass=%d/%d  real_tok=%d (in %d/out %d)  calls=%d  inj=%d  stm=%d" % (
            tag, r["pass"], r["n"], r["real_tokens"], r["prompt_tokens"],
            r["completion_tokens"], r["llm_calls"], r["injected_tokens"]))
    write_status("DONE", {"ledger": rows})


def main():
    log("supervisor start: 3 arms x N=4, real-token accounting")
    for tag, flags in ARMS:
        if prog_count(tag) >= N:
            log("arm=%s already complete (%d) -- skip" % (tag, prog_count(tag)))
            continue
        run_arm(tag, flags)
    summarize()
    log("supervisor DONE")


if __name__ == "__main__":
    main()
