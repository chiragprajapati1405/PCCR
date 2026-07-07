"""STEP 5 + STEP 8 — Executors (pluggable).

The research contribution (the concurrent memory manager) is pure software, so we measure it with a
ModeledExecutor first: one tool/LLM step = an `await asyncio.sleep(call_latency)`. Because asyncio
overlaps a real network await exactly as it overlaps a sleep, the modeled wall-clock is what a real
I/O-bound parallel run would achieve — with ZERO API cost and no rate limits.

STEP 8: DockerExecutor runs REAL tool actions through ONE OfficeBench container, isolating each task
in its own working directory (/tmp/cmm/run_<id>) so N parallel tasks never collide. Outputs persist
per-task for OPTIONAL deferred evaluation. No LLM (fixed action per sub-agent) so we measure the real
Docker + memory-manager concurrency without provider rate limits.
"""
from __future__ import annotations

import asyncio
import os
import time


class ModeledExecutor:
    """One sub-agent action = a fixed modeled latency (measured gpt-oss-120b p50 = 0.516s/call)."""

    def __init__(self, call_latency: float = 0.516):
        self.call_latency = call_latency
        self.actions = 0

    def setup_task(self, task_id):
        return None                                      # modeled: no per-task context needed

    async def run_action(self, app: str, subtask: str, usage: dict, ctx=None) -> dict:
        await asyncio.sleep(self.call_latency)           # the I/O-bound step (overlaps across tasks)
        self.actions += 1
        return {"agent": app, "action": f"{app}.<call>", "subtask": subtask, "obs": "ok"}


def _docker_client():
    """Get a docker client, auto-detecting the colima socket (macOS) when DOCKER_HOST is unset."""
    import docker
    if os.environ.get("DOCKER_HOST"):
        return docker.from_env()
    colima = os.path.expanduser("~/.colima/default/docker.sock")
    if os.path.exists(colima):
        return docker.DockerClient(base_url=f"unix://{colima}")
    return docker.from_env()


class DockerExecutor:
    """STEP 8 — real one-container executor with per-task workdir isolation.

    All N tasks share ONE container; each task acts inside /tmp/cmm/run_<id>, so their files never
    collide. run_action executes a real command in the container (real exec latency), wrapped in
    asyncio.to_thread so N concurrent tasks overlap on the (I/O-bound) docker-exec waits.
    """

    def __init__(self, container: str = "cmm-bench", image: str = "officebench",
                 base_dir: str = "/tmp/cmm"):
        self.client = _docker_client()
        self.base_dir = base_dir
        self.actions = 0
        import docker
        try:
            self.container = self.client.containers.get(container)
            if self.container.status != "running":
                self.container.start()
        except docker.errors.NotFound:
            self.container = self.client.containers.run(
                image, name=container, command="sleep infinity", detach=True, tty=True)
        self.name = container

    def setup_task(self, task_id) -> str:
        """Create this task's isolated workdir inside the shared container; return its path."""
        workdir = f"{self.base_dir}/run_{task_id}"
        self.container.exec_run(f'/bin/bash -c "mkdir -p {workdir}"')
        return workdir

    async def run_action(self, app: str, subtask: str, usage: dict, ctx: str = None) -> dict:
        """Execute one REAL action in the task's workdir. Fixed op (no LLM): write a per-step file
        representing the sub-agent's action, so outputs exist for deferred eval and the latency is real."""
        workdir = ctx or self.base_dir
        safe = subtask.replace('"', "'")[:80]
        cmd = (f'/bin/bash -c "echo {app}: {safe} >> {workdir}/{app}_steps.txt && ls {workdir}"')
        t0 = time.perf_counter()
        exit_code, output = await asyncio.to_thread(
            self.container.exec_run, cmd, workdir=workdir)
        latency = time.perf_counter() - t0
        self.actions += 1
        return {"agent": app, "action": f"{app}.exec", "subtask": subtask,
                "obs": output.decode("utf-8", "ignore")[:120], "latency": latency}

    def save_outputs(self, local_dir: str):
        """Copy all per-task workdirs out of the container for OPTIONAL later evaluation."""
        os.makedirs(local_dir, exist_ok=True)
        os.system(f'docker cp {self.name}:{self.base_dir} "{local_dir}" >/dev/null 2>&1')

    def cleanup_workdirs(self):
        self.container.exec_run(f'/bin/bash -c "rm -rf {self.base_dir}/run_*"')
