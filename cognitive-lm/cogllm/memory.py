"""The two non-parametric pieces of the model.

Familiarity  (knowing THAT):  Bloom filters over entity keys and fact keys.
             About 10 bits per key buys a 1% false-positive rate, and there
             are no false negatives: if the model saw it, the filter says so.
Recollection (knowing WHAT):  an exact episodic store for facts that have
             not (yet) been consolidated into the weights, with exposure
             counts that drive consolidation during sleep.
"""

import math
from collections import Counter

import numpy as np

_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x):
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
    return x ^ (x >> np.uint64(31))


class Bloom:
    def __init__(self, n_items, bits_per_item=10.0, seed=0):
        self.m = max(64, int(math.ceil(n_items * bits_per_item)))
        self.k = max(1, int(round(bits_per_item * math.log(2))))
        self.bits = np.zeros(self.m, dtype=bool)
        self.salt = np.uint64(seed * 0x5851F42D4C957F2D & 0xFFFFFFFFFFFFFFFF)

    def _pos(self, keys):
        keys = np.asarray(keys, dtype=np.uint64)
        with np.errstate(over="ignore"):
            h1 = _splitmix64(keys ^ self.salt)
            h2 = _splitmix64(h1) | np.uint64(1)
            i = np.arange(self.k, dtype=np.uint64)
            return ((h1[:, None] + i[None, :] * h2[:, None]) % np.uint64(self.m)).astype(np.int64)

    def add(self, keys):
        self.bits[self._pos(keys).ravel()] = True

    def __contains__(self, key):
        return bool(self.contains(np.array([key]))[0])

    def contains(self, keys):
        return self.bits[self._pos(keys)].all(axis=1)

    def n_bits(self):
        return self.m


class EpisodicStore:
    """Exact key -> value memory with exposure counts (one-shot writes)."""

    def __init__(self):
        self.vals = {}      # key -> Counter(value -> times seen with that value)

    def write(self, key, value):
        self.vals.setdefault(key, Counter())[value] += 1

    def count(self, key):
        c = self.vals.get(key)
        return sum(c.values()) if c else 0

    def recall(self, key):
        c = self.vals.get(key)
        return c.most_common(1)[0][0] if c else None

    def evict(self, key):
        self.vals.pop(key, None)

    def __len__(self):
        return len(self.vals)

    def __contains__(self, key):
        return key in self.vals

    def n_bits(self, key_bits, value_bits, count_bits=4):
        return len(self.vals) * (key_bits + value_bits + count_bits)
