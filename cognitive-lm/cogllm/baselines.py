"""Value stores with no weights: the baselines CAR has to beat on bits.

CAR keeps ~14 checksum bits per fact and lets the weights supply the value.
That only pays off if those bits are fewer than what it costs to store the
value itself, so the fair comparison is against the cheapest exact stores:

    exact dict            key -> value, exact keys: key_bits + value_bits per fact
    key filter + static   Bloom over keys (membership, false-positive rate eps_k)
    function              plus a static function key -> value, which answers
                          every key but stores no keys. log2 V bits per fact is
                          its lower bound; ribbon retrieval (Dillinger & Walzer
                          2021) gets within a few percent of it. A key-filter
                          false positive returns an arbitrary value, which is
                          a hallucination.
"""

import math
from collections import Counter

import numpy as np

from .memory import Bloom
from .system import fact_code, key_bits


def _majority(world, mentions, values):
    table = {}
    for (e, r), v in zip(mentions, values):
        table.setdefault(fact_code(world, world.names[e], r), Counter())[int(v)] += 1
    return table


class ExactDict:
    """min_count > 1 abstains on facts read fewer times (the "don't assert
    singletons" baseline for noisy extraction)."""

    def __init__(self, world, mentions, values, min_count=1):
        self.w, self.min_count = world, min_count
        self.table = _majority(world, mentions, values)

    def answer(self, names, rels):
        pred = np.full(len(names), -1, dtype=np.int64)
        for i, (nm, r) in enumerate(zip(names, rels)):
            c = self.table.get(fact_code(self.w, nm, r))
            if c and sum(c.values()) >= self.min_count:
                pred[i] = c.most_common(1)[0][0]
        return pred

    def bits(self):
        return {"dict": len(self.table) * (key_bits(self.w) + math.ceil(math.log2(self.w.cfg.n_values)))}


class KeyFilterStatic:
    def __init__(self, world, mentions, values, key_bits_per_item=10.0, seed=0):
        self.w = world
        self.table = _majority(world, mentions, values)
        codes = np.fromiter(self.table.keys(), dtype=np.uint64, count=len(self.table))
        self.key = Bloom(len(codes), key_bits_per_item, seed=seed * 7 + 11)
        self.key.add(codes)
        self.rng = np.random.default_rng(seed + 12)

    def answer(self, names, rels):
        codes = np.array([fact_code(self.w, nm, r) for nm, r in zip(names, rels)], dtype=np.uint64)
        ok = self.key.contains(codes)
        pred = np.full(len(names), -1, dtype=np.int64)
        for i in np.where(ok)[0]:
            c = self.table.get(int(codes[i]))
            # a static function has no keys: on a key it never stored it returns some value
            pred[i] = c.most_common(1)[0][0] if c else int(self.rng.integers(self.w.cfg.n_values))
        return pred

    def bits(self):
        return {"key_filter": self.key.n_bits(), "static_function_bound": len(self.table) * math.log2(self.w.cfg.n_values)}
