"""Shared FAISS IndexFlatIP wrapper with disk persistence.

Both the Episodic (EM) and Semantic (SM) stores are, per the lifecycle table,
"FAISS IndexFlatIP" + "float32[384]" vectors that are "loaded from disk" at
Bootstrap and "persist across runs". This module is the one place that detail
lives, so EM and SM can stay focused on *what* they store (whole episodes vs.
distilled facts) rather than *how* the vector search works.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class PersistentVectorIndex:
    """Cosine-similarity search over normalized float32 vectors, with payloads.

    `payloads[i]` is arbitrary JSON-serializable metadata associated with the
    vector at row `i` (e.g. a serialized FullTaskMemory or a fact string).
    Persisted as `<name>.faiss` (the index) + `<name>.json` (the payloads).
    """

    def __init__(self, directory: Path, name: str, dim: int):
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / f"{name}.faiss"
        self._payload_path = self._dir / f"{name}.json"
        self._dim = dim
        self._payloads: list[Any] = []
        self._index = self._load_or_create()

    # -- lifecycle: Bootstrap ("Disk -> FAISS RAM") / Consolidation (write) --

    def _load_or_create(self) -> faiss.Index:
        if self._index_path.exists() and self._payload_path.exists():
            index = faiss.read_index(str(self._index_path))
            self._payloads = json.loads(self._payload_path.read_text())
            return index
        return faiss.IndexFlatIP(self._dim)

    def save(self) -> None:
        faiss.write_index(self._index, str(self._index_path))
        self._payload_path.write_text(json.dumps(self._payloads))

    # -- core operations ----------------------------------------------------

    def add(self, vectors: np.ndarray, payloads: list[Any]) -> None:
        assert vectors.shape[0] == len(payloads)
        if vectors.shape[0] == 0:
            return
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._payloads.extend(payloads)

    def search(self, query: np.ndarray, k: int) -> list[tuple[float, Any]]:
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        query = np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((float(score), self._payloads[idx]))
        return results

    def __len__(self) -> int:
        return self._index.ntotal
