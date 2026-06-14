"""
free_legomem.py — local sentence-transformer embedder used by the LEGOMem +
MemoryManager harness (mm_on_top_of_legomem.py) and the PCCR router on top.

Provides LocalEmbedder with the two attributes the harness relies on:
  - .embed(text) -> np.ndarray[float32]  (single-vector, L2-normalizable)
  - .dims        -> int                  (embedding dimensionality)

Uses all-MiniLM-L6-v2 (384-d) by default: small, fast, CPU-friendly, and the
same family the FAISS IndexFlatIP in the harness assumes.
"""
from __future__ import annotations

import numpy as np

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class LocalEmbedder:
    def __init__(self, model_name: str = _DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dims = int(self.model.get_sentence_embedding_dimension())
        print(f"  📋 [LocalEmbedder] loaded {model_name} ({self.dims}d)")

    def embed(self, text) -> np.ndarray:
        """Return a 1-D float32 embedding for a string (or the first item of a
        list). Shape: (dims,). The harness reshapes/normalizes as needed."""
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        vec = self.model.encode([str(text)], convert_to_numpy=True)[0]
        return vec.astype("float32")
