"""Central configuration for backends and on-disk paths.

Everything defaults to the dependency-light "stub" backends so `python
run_task.py` works immediately after `pip install -r requirements.txt`,
with no API key and no model download. Set the env vars below to switch to
the real models for paper experiments.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FAISS_DIR = DATA_DIR / "faiss"
EML_DIR = DATA_DIR / "eml"
ICS_DIR = DATA_DIR / "ics"
ANSWERS_DIR = DATA_DIR / "answers"
LOG_DIR = DATA_DIR / "logs"

for _dir in (DATA_DIR, FAISS_DIR, EML_DIR, ICS_DIR, ANSWERS_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    embedding_backend: str = os.environ.get("EMBEDDING_BACKEND", "stub")  # "stub" | "sentence-transformers"
    llm_backend: str = os.environ.get("LLM_BACKEND", "stub")              # "stub" | "anthropic"
    llm_model: str = os.environ.get("LLM_MODEL", "claude-haiku-4-5-20251001")

    # Router thresholds (see router.py) -- defaults tuned for the stub embedder.
    stm_cache_threshold: float = float(os.environ.get("STM_CACHE_THRESHOLD", "0.92"))
    em_sm_similarity_threshold: float = float(os.environ.get("EM_SM_SIMILARITY_THRESHOLD", "0.55"))
    max_episodes_per_query: int = int(os.environ.get("MAX_EPISODES_PER_QUERY", "3"))

    # A1: per-query retrieval-confidence gate. When True, retrieve() pre-searches
    # each optional store and the router consults it iff its top-k similarity
    # clears a single global threshold (label-free, no per-pattern utility table).
    confidence_gate: bool = os.environ.get("CONFIDENCE_GATE", "0") == "1"

    # C1: bounded STM hot-cache. 0 -> unbounded (legacy); >0 -> fixed budget with
    # LFU+LRU eviction so STM stays the hot set and scales to 1000+ tasks.
    stm_capacity: int = int(os.environ.get("STM_CAPACITY", "0"))

    # C3: STM pattern-match safety guard. When True, an STM short-circuit is only
    # taken if the cached bundle is the SAME task pattern (lets the threshold be
    # lowered safely). Default False -> legacy behavior (no pattern filter).
    stm_pattern_guard: bool = os.environ.get("STM_PATTERN_GUARD", "0") == "1"


SETTINGS = Settings()
