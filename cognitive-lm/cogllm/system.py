"""The complementary-memory language model (CMLM).

    familiarity (Bloom)  ->  do I know this person?  do I know this fact?
    episodic store       ->  exact recall of facts not yet consolidated
    parametric core      ->  recall of facts consolidated during sleep

Wake:  every mention goes into the Bloom filters and the episodic store
       (one shot, no gradients).
Sleep: facts seen at least k times are replayed into the core. Facts the
       core then recalls correctly are evicted from the store; the rest stay.
Query: unfamiliar person  -> "I don't know who that is"
       familiar person, unfamiliar fact -> "I don't know that about them"
       familiar fact in store -> answer from the store
       familiar fact, consolidated -> answer from the core
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from .evaluate import answer_probs
from .memory import Bloom, EpisodicStore
from .model import GPT, train_lm
from .world import World

ABSTAIN_ENTITY = -2   # never heard of this person
ABSTAIN_FACT = -1     # know the person, not this fact


def name_code(world: World, name):
    return int(name[0]) * world.cfg.n_last + int(name[1])


def fact_code(world: World, name, r):
    return name_code(world, name) * world.cfg.n_relations + int(r)


def key_bits(world: World):
    return math.ceil(math.log2(world.cfg.n_first * world.cfg.n_last)) + math.ceil(math.log2(world.cfg.n_relations))


def value_bits(world: World):
    return math.ceil(math.log2(world.cfg.n_values))


def stream_tensors(world: World, mentions):
    seqs = torch.tensor([world.fact_seq(e, r) for e, r in mentions], dtype=torch.long)
    mask = torch.ones(len(seqs), seqs.shape[1] - 1, dtype=torch.bool)
    return seqs, mask


@dataclass
class CMLM:
    world: World
    ent_bloom: Bloom
    fact_bloom: Bloom
    store: EpisodicStore
    core: GPT = None
    k: float = 2
    evicted: int = 0
    failed_consolidation: int = 0

    # ---- wake ------------------------------------------------------------
    @classmethod
    def wake(cls, world: World, mentions, bits_per_item=10.0, k=2, seed=0):
        ents = {name_code(world, world.names[e]) for e, _ in mentions}
        facts = {fact_code(world, world.names[e], r) for e, r in mentions}
        eb = Bloom(len(ents), bits_per_item, seed=seed * 2 + 1)
        fb = Bloom(len(facts), bits_per_item, seed=seed * 2 + 2)
        eb.add(np.fromiter(ents, dtype=np.uint64))
        fb.add(np.fromiter(facts, dtype=np.uint64))
        store = EpisodicStore()
        for e, r in mentions:
            store.write((int(e), int(r)), int(world.values[e, r]))
        return cls(world, eb, fb, store, k=k)

    # ---- sleep -----------------------------------------------------------
    def sleep(self, mentions, d, n_layer, epochs, seed=0, core=None, log=None, verify_threshold=0.5):
        """Replay facts with count >= k into the core, then evict what it learned.

        Replay keeps each consolidated fact's natural frequency, so the core sees
        exactly the gradient exposures a plain LM would have seen for it.
        """
        w = self.world
        chosen = [(int(e), int(r)) for e, r in mentions if self.store.count((int(e), int(r))) >= self.k]
        if core is None and chosen:
            torch.manual_seed(seed)
            core = GPT(w.vocab_size, d=d, n_layer=n_layer)
            seqs, mask = stream_tensors(w, chosen)
            train_lm(core, seqs, mask, epochs=epochs, seed=seed, log=log)
        self.core = core
        cand = sorted({key for key in chosen})
        if not cand or core is None:
            return self
        names = np.array([w.names[e] for e, _ in cand])
        rels = np.array([r for _, r in cand])
        p = answer_probs(core, w, names, rels)[:, 1:]
        truth = np.array([w.values[e, r] for e, r in cand])
        ok = (p.argmax(1) == truth) & (p[np.arange(len(truth)), truth] >= verify_threshold)
        for key, good in zip(cand, ok):
            if good:
                self.store.evict(key)
        self.evicted = int(ok.sum())
        self.failed_consolidation = int((~ok).sum())
        return self

    # ---- query -----------------------------------------------------------
    def answer(self, names, rels):
        w = self.world
        n = len(names)
        pred = np.full(n, ABSTAIN_FACT, dtype=np.int64)
        source = np.array(["abstain"] * n, dtype=object)
        ecode = np.array([name_code(w, nm) for nm in names], dtype=np.uint64)
        fcode = np.array([fact_code(w, nm, r) for nm, r in zip(names, rels)], dtype=np.uint64)
        ent_ok = self.ent_bloom.contains(ecode)
        fact_ok = self.fact_bloom.contains(fcode) & ent_ok
        pred[~ent_ok] = ABSTAIN_ENTITY
        need_core = []
        # names -> entity index (only real entities can be in the store)
        idx_of = getattr(self, "_idx_of", None)
        if idx_of is None:
            idx_of = self._idx_of = {name_code(w, nm): i for i, nm in enumerate(w.names)}
        for i in np.where(fact_ok)[0]:
            e = idx_of.get(int(ecode[i]))
            key = (e, int(rels[i])) if e is not None else None
            if key is not None and key in self.store:
                pred[i] = self.store.recall(key)
                source[i] = "store"
            else:
                need_core.append(i)
        if need_core:
            need_core = np.array(need_core)
            if self.core is not None:
                p = answer_probs(self.core, w, names[need_core], rels[need_core])[:, 1:]
                pred[need_core] = p.argmax(1)
                source[need_core] = "core"
            # no core: familiar but nothing to recall -> "tip of the tongue" abstain
        return pred, source

    # ---- accounting ------------------------------------------------------
    def bits(self, param_bits=16):
        w = self.world
        core = self.core.n_params() * param_bits if self.core is not None else 0
        return {
            "core": core,
            "store": self.store.n_bits(key_bits(w), value_bits(w)),
            "bloom": self.ent_bloom.n_bits() + self.fact_bloom.n_bits(),
        }
