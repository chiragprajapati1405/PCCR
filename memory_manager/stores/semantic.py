"""Semantic Memory (SM): general, distilled facts ("what's true / typical").

Where Episodic Memory answers "have I done something like THIS exact task
before", Semantic Memory answers "what do I generally know that's relevant
here" -- short, reusable factual statements (e.g. "Acme stand-ups run 30
minutes", "Dana prefers afternoon meeting slots") rather than whole task
traces. Per the table it stores "meaning vectors" / float32[384], is
FAISS-backed, "persists on disk", and is populated mainly at Consolidation
("Embed model -> FAISS"); during a normal task it is read-only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ._faiss_index import PersistentVectorIndex


class SemanticStore:
    def __init__(self, embedder, directory: Path):
        self._embedder = embedder
        self._index = PersistentVectorIndex(directory, "semantic", embedder.dim)

    def __len__(self) -> int:
        return len(self._index)

    # -- Retrieval: READ ("on miss") -----------------------------------------

    def search(self, query_text: str, k: int) -> list[tuple[float, str]]:
        query_vec = self._embedder.encode([query_text])[0]
        return self._index.search(query_vec, k)

    # -- Consolidation: WRITE (embed distilled facts) ------------------------

    def add_facts(self, facts: list[str]) -> None:
        if not facts:
            return
        vectors = self._embedder.encode(facts)
        self._index.add(np.asarray(vectors, dtype=np.float32), list(facts))

    def save(self) -> None:
        self._index.save()
