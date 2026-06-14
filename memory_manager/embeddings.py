"""Embedding backend used by the Semantic (SM) and Episodic (EM) stores.

The lifecycle table specifies float32[384] vectors produced by a
SentenceTransformer and indexed with FAISS IndexFlatIP (cosine/IP similarity
on normalized vectors). We wrap that behind a small interface with a
deterministic hash-based stub so the whole pipeline (router, stores, agents)
can be developed, tested, and demoed without downloading a transformer model
or paying for API calls -- swap to the real model by setting
EMBEDDING_BACKEND=sentence-transformers (see config.py).
"""
from __future__ import annotations

import hashlib
import struct
from typing import Sequence

import numpy as np

EMBEDDING_DIM = 384


class Embedder:
    """Interface: text(s) -> float32[N, 384] L2-normalized vectors."""

    dim: int = EMBEDDING_DIM

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        raise NotImplementedError


class StubEmbedder(Embedder):
    """Deterministic, dependency-free embedder for tests and offline demos.

    Hashes overlapping word shingles into fixed vector slots. Not semantically
    meaningful in the way a trained model is, but it IS deterministic, stable
    across runs, and gives near-duplicate strings near-identical vectors and
    unrelated strings near-orthogonal ones -- enough to exercise FAISS
    similarity search and the router's "cache hit / near-miss" logic in tests.
    """

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = text.lower().split()
            shingles = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            if not shingles:
                shingles = [""]
            for shingle in shingles:
                digest = hashlib.md5(shingle.encode("utf-8")).digest()
                (idx,) = struct.unpack_from("I", digest, 0)
                (sign,) = struct.unpack_from("I", digest, 4)
                vectors[row, idx % self.dim] += 1.0 if sign % 2 == 0 else -1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class SentenceTransformerEmbedder(Embedder):
    """Wraps sentence-transformers, matching the lifecycle table's spec."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # local import: heavy

        self._model = SentenceTransformer(model_name)
        actual_dim = self._model.get_sentence_embedding_dimension()
        if actual_dim != self.dim:
            self.dim = actual_dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(backend: str = "stub") -> Embedder:
    if backend == "stub":
        return StubEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
