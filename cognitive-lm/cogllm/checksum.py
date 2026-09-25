"""Checksum-aided recall: the weights propose, a knowledge checksum disposes.

Borrowed from CRC-aided list decoding (polar codes in 5G): a list decoder
proposes the L most likely codewords and a short CRC picks the one that
checks out, or flags a decoding failure. Here the weights are the list
decoder and the checksum is a Bloom filter over every (entity, relation,
value) triple read during training. It stores no values, only ~b bits per
fact, and it is written in one shot while reading.

It is also the generate-recognize theory of human recall (Kintsch 1970;
Anderson & Bower 1972): recall = generate candidates, then recognise the
right one. The weights only have to get the answer into the top N, not to
the top 1, and they never have to know what they don't know.

Answer levels (graded metamemory):
    UNKNOWN_ENTITY  never heard of this person          (entity filter says no)
    UNKNOWN_FACT    know the person, never read this    (fact-key filter says no)
    TIP_OF_TONGUE   read it, but can't bring it back    (key familiar, no candidate verifies)
    value           recalled and recognised
"""

import numpy as np

from .evaluate import answer_probs
from .memory import Bloom
from .system import fact_code, name_code
from .world import World

UNKNOWN_ENTITY, UNKNOWN_FACT, TIP_OF_TONGUE = -2, -1, -3


def triple_code(world: World, name, r, v):
    return fact_code(world, name, r) * world.cfg.n_values + int(v)


class CountMin:
    """Tiny count-min sketch with saturating 4-bit counters (for noisy data)."""

    def __init__(self, n_items, bits_per_item=16.0, depth=4, counter_bits=4, seed=0):
        width = max(16, int(n_items * bits_per_item / (depth * counter_bits)))
        self.width, self.depth, self.cap = width, depth, 2 ** counter_bits - 1
        self.table = np.zeros((depth, width), dtype=np.uint8)
        self.blooms = [Bloom(1, 1, seed=seed * 31 + i) for i in range(depth)]  # reuse hashing only
        for b in self.blooms:
            b.m, b.k = width, 1
        self.counter_bits = counter_bits

    def add(self, keys):
        for i, b in enumerate(self.blooms):
            pos = b._pos(keys)[:, 0]
            np.add.at(self.table[i], pos, 1)
            np.minimum(self.table[i], self.cap, out=self.table[i])

    def count(self, keys):
        return np.min(np.stack([self.table[i][b._pos(keys)[:, 0]] for i, b in enumerate(self.blooms)]), axis=0)

    def n_bits(self):
        return self.table.size * self.counter_bits


class CheckedRecall:
    def __init__(self, world: World, mentions, triple_bits=10.0, key_bits=10.0, use_key_filter=True,
                 counting=False, seed=0, values=None, prefix_bits=0.0):
        """mentions: (M, 2) entity, relation. values: value per mention (default: the true value)."""
        self.w = world
        vals = world.values[mentions[:, 0], mentions[:, 1]] if values is None else values
        names = world.names[mentions[:, 0]]
        ents = np.unique([name_code(world, n) for n in names]).astype(np.uint64)
        keys = np.unique([fact_code(world, n, r) for n, r in zip(names, mentions[:, 1])]).astype(np.uint64)
        tri_all = np.array([triple_code(world, n, r, v) for n, r, v in zip(names, mentions[:, 1], vals)], dtype=np.uint64)
        tris = np.unique(tri_all)
        self.ent = Bloom(len(ents), key_bits, seed=seed * 7 + 1)
        self.key = Bloom(len(keys), key_bits, seed=seed * 7 + 2) if use_key_filter else None
        self.ent.add(ents)
        if self.key is not None:
            self.key.add(keys)
        # membership always comes from a Bloom filter; a count-min sketch (a
        # count-min alone is a poor membership test at this budget) only breaks
        # ties between verified candidates, so a value read 5 times beats a
        # mislinked one read once
        self.counting = counting
        self.tri = Bloom(len(tris), triple_bits, seed=seed * 7 + 3)
        self.tri.add(tris)
        if counting:
            self.cms = CountMin(len(tris), 8, seed=seed * 7 + 4)
            self.cms.add(tri_all)
        self.n_facts = len(tris)
        # prefix checksum (multi-token values): has any fact for this key started
        # with this first sub-token? Prunes wrong branches before the full check.
        self.prefix = None
        if prefix_bits > 0:
            P = world.cfg.value_pool
            fk = np.array([fact_code(world, n, r) for n, r in zip(names, mentions[:, 1])], dtype=np.uint64)
            pre = np.unique(fk * np.uint64(P) + (vals // P).astype(np.uint64))
            self.prefix = Bloom(len(pre), prefix_bits, seed=seed * 7 + 5)
            self.prefix.add(pre)

    def bits(self):
        b = {"entity_filter": self.ent.n_bits(), "triple_checksum": self.tri.n_bits()}
        if self.key is not None:
            b["key_filter"] = self.key.n_bits()
        if self.counting:
            b["count_sketch"] = self.cms.n_bits()
        if self.prefix is not None:
            b["prefix_checksum"] = self.prefix.n_bits()
        return b

    def answer(self, model, names, rels, N=5, probs=None):
        w = self.w
        n = len(names)
        pred = np.full(n, UNKNOWN_FACT, dtype=np.int64)
        ecode = np.array([name_code(w, nm) for nm in names], dtype=np.uint64)
        ent_ok = self.ent.contains(ecode)
        key_ok = ent_ok.copy()
        if self.key is not None:
            key_ok &= self.key.contains(np.array([fact_code(w, nm, r) for nm, r in zip(names, rels)], dtype=np.uint64))
        pred[~ent_ok] = UNKNOWN_ENTITY
        idx = np.where(key_ok)[0]
        if len(idx) == 0:
            return pred
        if probs is None:
            probs = answer_probs(model, w, names[idx], rels[idx])[:, 1:]
        else:
            probs = probs[idx]
        fcodes = np.array([fact_code(w, names[i], rels[i]) for i in idx], dtype=np.uint64)
        V = np.uint64(w.cfg.n_values)
        chunk = max(1, 2_000_000 // max(1, N))
        for s in range(0, len(idx), chunk):
            p = probs[s:s + chunk]
            if self.prefix is not None:
                P = w.cfg.value_pool
                pc = fcodes[s:s + chunk, None] * np.uint64(P) + np.arange(P, dtype=np.uint64)[None, :]
                pok = self.prefix.contains(pc.ravel()).reshape(pc.shape)
                p = (p.reshape(len(p), P, -1) * pok[:, :, None]).reshape(len(p), -1)
                p = np.where(p > 0, p, -1.0)  # pruned branches sort last
            order = np.argsort(-p, axis=1)[:, :N] if N < p.shape[1] else np.argsort(-p, axis=1)
            tri = fcodes[s:s + chunk, None] * V + order.astype(np.uint64)
            ok = self.tri.contains(tri.ravel()).reshape(tri.shape)
            if self.counting:
                c = self.cms.count(tri.ravel()).reshape(tri.shape).astype(float)
                # most-read verified candidate wins; ties go to the weights' ranking
                sc = c - 1e-3 * np.arange(order.shape[1])[None, :]
                best = np.where(ok.any(1), np.argmax(np.where(ok, sc, -1e9), 1), -1)
            else:
                best = np.where(ok.any(1), ok.argmax(1), -1)
            pred[idx[s:s + chunk]] = np.where(best >= 0, order[np.arange(len(order)), np.maximum(best, 0)], TIP_OF_TONGUE)
        return pred
