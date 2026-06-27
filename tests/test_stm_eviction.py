"""C1: bounded STM hot-cache with LFU+LRU eviction."""
import numpy as np

from memory_manager.stores.short_term import ShortTermStore
from memory_manager.types import Pattern


class _Embedder:
    """Deterministic 2-D unit-vector embedder keyed by the first char."""
    def encode(self, texts):
        out = []
        for t in texts:
            a = (ord(t[0]) - 97) * 0.3
            out.append(np.array([np.cos(a), np.sin(a)], dtype=np.float32))
        return np.array(out)


def _put(stm, tag):
    v = _Embedder().encode([tag])[0]
    stm.put(tag, [f"plan-{tag}"], [], v, Pattern.LOOKUP)


def test_capacity_enforced():
    stm = ShortTermStore(_Embedder(), 0.99, capacity=3)
    for tag in "abcd":
        _put(stm, tag)
    assert stm.entries == 3
    assert "a" not in stm._signatures          # oldest cold entry evicted


def test_hot_entries_survive():
    stm = ShortTermStore(_Embedder(), 0.99, capacity=3)
    for tag in "abc":
        _put(stm, tag)
    stm.lookup("b"); stm.lookup("b"); stm.lookup("c")   # make b,c hot
    _put(stm, "e"); _put(stm, "f")                       # forces eviction of cold entries
    assert stm.entries == 3
    assert "b" in stm._signatures and "c" in stm._signatures
    assert "a" not in stm._signatures                   # cold -> evicted


def test_unbounded_is_legacy():
    stm = ShortTermStore(_Embedder(), 0.99, capacity=None)
    for tag in "abcdef":
        _put(stm, tag)
    assert stm.entries == 6
