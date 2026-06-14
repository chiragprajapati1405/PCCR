"""Short-Term Memory (STM): the session cache that the router checks FIRST.

Per the lifecycle table, STM holds "cached bundle"s keyed by a step signature
plus an embedding, with lifetime "session". It is the fast path: "STM: ALWAYS
check first / If STM HIT -> skip EM, SM, ENT". Writes happen only "on success"
(x3: bundle + signature + embedding) at Memory Storage, and consolidation can
also pre-populate it with successful patterns "cached for testing".

This store is therefore the crux of the cascading router's cost savings: a
hit here turns what would be three retrievals (EM similarity search, SM
similarity search, ENT profile read) into a single dict lookup plus one cheap
embedding comparison.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import MemoryBundle, Pattern


class ShortTermStore:
    def __init__(self, embedder, similarity_threshold: float):
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._bundles: dict[str, MemoryBundle] = {}   # signature -> bundle
        self._embeddings: list[np.ndarray] = []
        self._signatures: list[str] = []

    def __len__(self) -> int:
        return len(self._bundles)

    # -- Retrieval: the cache-short-circuit check ---------------------------

    def lookup(self, query_text: str) -> Optional[tuple[MemoryBundle, float]]:
        """Return (bundle, similarity) for the closest cached bundle if it
        clears the similarity threshold ('STM HIT'), else None ('STM MISS')."""
        if not self._embeddings:
            return None
        query_vec = self._embedder.encode([query_text])[0]
        # macOS Accelerate's BLAS trips numpy's FP-exception flags on plain
        # matmul (spurious divide/overflow/invalid RuntimeWarnings) even
        # though the float32 inputs and outputs are all finite -- verified by
        # inspecting arr/query/sims for nan/inf across repeated runs. The
        # warning is benign; silence it locally rather than globally.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = np.array(self._embeddings) @ query_vec
        best = int(np.argmax(sims))
        if sims[best] >= self._threshold:
            sig = self._signatures[best]
            bundle = self._bundles[sig]
            bundle.hits += 1
            return bundle, float(sims[best])
        return None

    # -- Storage: WRITE x3 (bundle + signature + embedding), success only ---

    def put(self, signature: str, plan: list[str], subtask_memories, embedding: np.ndarray, pattern: Pattern) -> None:
        """Cache this successful task as a future shortcut.

        Two different things get deduplicated independently:
          - bundle CONTENT is keyed by step signature (many worded-differently
            tasks legitimately share one plan template -- no need to store it
            twice);
          - the embedding INDEX gets one entry per task occurrence, each
            pointing at its (possibly shared) bundle. This is what makes the
            cache actually fire for a recurring task: lookup must match THIS
            task's own wording, not whichever differently-worded task happened
            to be first to produce the same plan signature.
        """
        if signature not in self._bundles:
            self._bundles[signature] = MemoryBundle(
                signature=signature, plan=list(plan), subtask_memories=list(subtask_memories),
                embedding=embedding, pattern=pattern,
            )
        self._signatures.append(signature)
        self._embeddings.append(np.asarray(embedding, dtype=np.float32))

    # -- Consolidation: pre-fill with successful patterns -------------------

    def prefill(self, signature: str, plan: list[str], subtask_memories, embedding: np.ndarray, pattern: Pattern) -> None:
        self.put(signature, plan, subtask_memories, embedding, pattern)
