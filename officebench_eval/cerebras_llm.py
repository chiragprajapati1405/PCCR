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
        keys = [os.environ.get(f"CEREBRAS_KEY_{i}", "") for i in range(1, 10)]
        keys = [k for k in keys if k.strip()]
        if not keys:
            raise RuntimeError("no CEREBRAS_KEY_* in env — run `source cerebras.env`")
        self.clients = [OpenAI(api_key=k, base_url="https://api.cerebras.ai/v1", max_retries=2)
                        for k in keys]
        self.model_name = model_name
        self.system_message = system_message
        self._i = 0
        self._lock = threading.Lock()
        self.calls = 0

    def _next(self):
        with self._lock:
            c = self.clients[self._i % len(self.clients)]
            self._i += 1
            self.calls += 1
            return c

    def generate(self, prompt: str) -> str:
        for attempt in range(10):
            c = self._next()
            try:
                r = c.chat.completions.create(
                    model=self.model_name, temperature=0, max_tokens=512,
                    messages=[{"role": "system", "content": self.system_message or ""},
                              {"role": "user", "content": prompt}])
                txt = (r.choices[0].message.content or "").strip()
                if txt:
                    return txt
                time.sleep(1.0)
            except Exception as e:
                s = str(e).lower()
                time.sleep(min(2.0 + attempt, 6.0) if ("429" in s or "rate" in s) else 1.0)
        return "None"
