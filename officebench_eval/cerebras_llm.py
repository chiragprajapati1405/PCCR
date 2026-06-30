"""Cerebras backbone for the OfficeBench harness.

Matches OfficeBench's `llm.generate(prompt)` interface (see utils/llm.py vLLM
class) but points the OpenAI-compatible client at Cerebras gpt-oss-120b, with
multi-key round-robin + rate-limit handling. Needs `source cerebras.env`.
"""
from __future__ import annotations

import os
import threading
import time

# GLOBAL round-robin key index shared across ALL CerebrasLLM instances. Without this,
# every instance started at key 0, so N concurrent tasks marched in LOCKSTEP and hit the
# SAME key simultaneously every step -> 429 storms. A shared counter hands consecutive
# keys to concurrent callers, spreading load across all keys.
_GLOBAL_LOCK = threading.Lock()
_GLOBAL_I = 0


class CerebrasLLM:
    def __init__(self, model_name: str = "gpt-oss-120b", system_message: str | None = None):
        from openai import OpenAI
        _start = int(os.environ.get("CEREBRAS_KEY_START", "1"))   # e.g. 16 -> use only keys 16..N (fresh quota)
        keys = [os.environ.get(f"CEREBRAS_KEY_{i}", "") for i in range(_start, 33)]
        keys = [k for k in keys if k.strip()]
        if not keys:
            raise RuntimeError("no CEREBRAS_KEY_* in env — run `source cerebras.env`")
        # max_retries=0: on a 429 the SDK would otherwise obey Retry-After (60s) on the SAME
        # key up to 2x before rotating -- blocking for ~2 min while other keys sit idle. We
        # fail FAST and let generate() rotate to the next key instantly.
        self.clients = [OpenAI(api_key=k, base_url="https://api.cerebras.ai/v1",
                               max_retries=0, timeout=90)
                        for k in keys]
        self.model_name = model_name
        self.system_message = system_message
        self._i = 0
        self._lock = threading.Lock()
        self.calls = 0
        self.max_tokens = 512                  # raise for long outputs (e.g. curation)
        self.rate_limit_wait_s = 0.0           # cumulative time lost to 429s (failed round-trip + pacing sleep)
        self.rate_limit_hits = 0
        # REAL token accounting from the API's usage field (the actual billed cost: the WHOLE
        # prompt -- system + history + memory -- plus the completion, not just injected memory).
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _next(self):
        # GLOBAL round-robin: concurrent tasks get DIFFERENT keys instead of colliding
        # on the same one. Falls back to this instance's clients (the key set is shared).
        global _GLOBAL_I
        with _GLOBAL_LOCK:
            idx = _GLOBAL_I
            _GLOBAL_I += 1
        with self._lock:
            self.calls += 1
        return self.clients[idx % len(self.clients)]

    def generate(self, prompt: str) -> str:
        n = len(self.clients)
        for attempt in range(n + 4):                 # one sweep over keys + a little slack
            c = self._next()
            t0 = time.perf_counter()
            try:
                r = c.chat.completions.create(
                    model=self.model_name, temperature=0, max_tokens=self.max_tokens,
                    messages=[{"role": "system", "content": self.system_message or ""},
                              {"role": "user", "content": prompt}])
                u = getattr(r, "usage", None)         # real billed tokens (whole prompt + completion)
                if u is not None:
                    self.prompt_tokens += int(getattr(u, "prompt_tokens", 0) or 0)
                    self.completion_tokens += int(getattr(u, "completion_tokens", 0) or 0)
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt
            except Exception as e:
                s = str(e).lower()
                if "401" in s or "organization" in s or "invalid_api_key" in s:
                    continue                         # DEAD key -> skip instantly, next key
                if "429" in s or "rate" in s:
                    # time LOST to rate limiting = the failed round-trip + the pacing sleep.
                    # Tracked so the run can report compute time with 429 stalls neglected.
                    self.rate_limit_wait_s += (time.perf_counter() - t0) + 0.4
                    self.rate_limit_hits += 1
                    time.sleep(0.4)                  # PACED rotation (no burst, no 60s block)
                    continue
                time.sleep(0.5)                      # other transient -> brief pause, next key
        return "None"
