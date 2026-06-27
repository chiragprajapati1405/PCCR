"""Cerebras backbone for the OfficeBench harness.

Matches OfficeBench's `llm.generate(prompt)` interface (see utils/llm.py vLLM
class) but points the OpenAI-compatible client at Cerebras gpt-oss-120b, with
multi-key round-robin + rate-limit handling. Needs `source cerebras.env`.
"""
from __future__ import annotations

import os
import threading
import time


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

    def _next(self):
        with self._lock:
            c = self.clients[self._i % len(self.clients)]
            self._i += 1
            self.calls += 1
            return c

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
