"""A synthetic world of arbitrary facts with Zipfian popularity.

Every entity (a two-token name) has one value per relation, drawn uniformly
at random, so no fact can be inferred from any other. This is the "arbitrary
facts" setting of Kalai & Vempala (2024): the only way to answer is to have
seen the fact, and the only honest answer to an unseen fact is "I don't know".

A training corpus is a stream of fact mentions sampled from the popularity
distribution, so some facts are mentioned many times, many are mentioned
once (singletons), and many are never mentioned at all. That gives us exact
control over exposure counts, which real corpora never do.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

PAD, BOS, EOS, IDK = 0, 1, 2, 3
N_SPECIAL = 4


@dataclass
class WorldConfig:
    n_first: int = 100          # first-name tokens
    n_last: int = 100           # last-name tokens
    n_entities: int = 4000      # people who exist in the world
    n_relations: int = 4        # born_in, works_at, studied_at, hobby ...
    n_values: int = 48          # distinct values per relation
    zipf_s: float = 1.05        # entity popularity exponent
    n_mentions: int = 40000     # size of the training stream
    seed: int = 0


@dataclass
class World:
    cfg: WorldConfig
    names: np.ndarray            # (n_entities, 2) first/last token ids (local)
    values: np.ndarray           # (n_entities, n_relations) value ids (local)
    fact_p: np.ndarray           # (n_entities, n_relations) mention probability
    counts: np.ndarray           # (n_entities, n_relations) exposures in stream
    stream: np.ndarray           # (n_mentions, 2) entity, relation per mention
    ghost_names: np.ndarray      # names that exist nowhere in the world
    vocab_size: int = 0
    rng: np.random.Generator = field(default=None, repr=False)

    # ---- token layout -------------------------------------------------
    def first_tok(self, i):
        return N_SPECIAL + i

    def last_tok(self, i):
        return N_SPECIAL + self.cfg.n_first + i

    def rel_tok(self, r):
        return N_SPECIAL + self.cfg.n_first + self.cfg.n_last + r

    def val_tok(self, r, v):
        base = N_SPECIAL + self.cfg.n_first + self.cfg.n_last + self.cfg.n_relations
        return base + r * self.cfg.n_values + v

    def val_toks(self, r):
        return [self.val_tok(r, v) for v in range(self.cfg.n_values)]

    # ---- sequences ----------------------------------------------------
    def prompt(self, name, r):
        """BOS first last rel  -> the model predicts the answer next."""
        return [BOS, self.first_tok(name[0]), self.last_tok(name[1]), self.rel_tok(r)]

    def fact_seq(self, e, r):
        return self.prompt(self.names[e], r) + [self.val_tok(r, self.values[e, r]), EOS]

    # ---- bookkeeping ----------------------------------------------------
    def singleton_rate(self):
        """Good-Turing missing-mass estimate N1 / M."""
        return float((self.counts == 1).sum()) / len(self.stream)

    def true_missing_mass(self):
        return float(self.fact_p[self.counts == 0].sum())


def build_world(cfg: WorldConfig) -> World:
    rng = np.random.default_rng(cfg.seed)
    all_names = np.array([(f, l) for f in range(cfg.n_first) for l in range(cfg.n_last)])
    assert cfg.n_entities * 2 <= len(all_names), "not enough names for entities + ghosts"
    perm = rng.permutation(len(all_names))
    names = all_names[perm[: cfg.n_entities]]
    ghost_names = all_names[perm[cfg.n_entities: 2 * cfg.n_entities]]

    values = rng.integers(0, cfg.n_values, size=(cfg.n_entities, cfg.n_relations))

    ranks = np.arange(1, cfg.n_entities + 1)
    ent_p = 1.0 / ranks ** cfg.zipf_s
    ent_p /= ent_p.sum()
    fact_p = np.repeat(ent_p[:, None] / cfg.n_relations, cfg.n_relations, axis=1)

    flat = rng.choice(fact_p.size, size=cfg.n_mentions, p=fact_p.ravel())
    stream = np.stack(np.unravel_index(flat, fact_p.shape), axis=1)
    counts = np.zeros_like(values)
    np.add.at(counts, (stream[:, 0], stream[:, 1]), 1)

    vocab = N_SPECIAL + cfg.n_first + cfg.n_last + cfg.n_relations + cfg.n_relations * cfg.n_values
    return World(cfg, names, values, fact_p, counts, stream, ghost_names, vocab, rng)


def sample_queries(world: World, n: int, seed: int = 1):
    """Fresh queries from the SAME popularity distribution as training.

    This is the natural test distribution: most traffic hits popular facts,
    and a fraction equal to the missing mass hits facts never seen.
    """
    rng = np.random.default_rng(seed)
    flat = rng.choice(world.fact_p.size, size=n, p=world.fact_p.ravel())
    return np.stack(np.unravel_index(flat, world.fact_p.shape), axis=1)


def count_buckets(c):
    """0, 1, 2-3, 4-7, 8-15, 16+"""
    edges = [0, 1, 2, 4, 8, 16]
    labels = ["0", "1", "2-3", "4-7", "8-15", "16+"]
    idx = np.searchsorted(edges, c, side="right") - 1
    return [labels[i] for i in idx]


def describe(world: World) -> str:
    c = world.counts
    h = Counter(count_buckets(c.ravel()))
    lines = [
        f"entities={world.cfg.n_entities} relations={world.cfg.n_relations} "
        f"facts={c.size} mentions={len(world.stream)} vocab={world.vocab_size}",
        f"distinct facts seen={int((c > 0).sum())} singletons={int((c == 1).sum())} "
        f"unseen={int((c == 0).sum())}",
        f"Good-Turing singleton rate N1/M={world.singleton_rate():.3f} "
        f"true missing mass={world.true_missing_mass():.3f}",
        "facts per exposure bucket: " + ", ".join(f"{k}:{h[k]}" for k in ["0", "1", "2-3", "4-7", "8-15", "16+"]),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe(build_world(WorldConfig())))
