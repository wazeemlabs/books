"""A harder world: answers are multi-token strings from a large value space.

Each value is L sub-tokens from a shared pool of P (default 2 x 32 = 1024
values), like a year or a city name spelled with sub-words. With a value
space this large you can't just try every possible answer against a
checksum: the weights have to narrow it down first.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .world import BOS, EOS, IDK, N_SPECIAL, World, WorldConfig, build_world


@dataclass
class MTConfig(WorldConfig):
    value_len: int = 2
    value_pool: int = 32


class MTWorld(World):
    def sub_tok(self, t):
        return N_SPECIAL + self.cfg.n_first + self.cfg.n_last + self.cfg.n_relations + t

    def digits(self, v):
        P, L = self.cfg.value_pool, self.cfg.value_len
        return [(int(v) // P ** (L - 1 - i)) % P for i in range(L)]

    def value_tokens(self, v):
        return [self.sub_tok(t) for t in self.digits(v)]

    def fact_seq(self, e, r, v=None):
        v = self.values[e, r] if v is None else v
        return self.prompt(self.names[e], r) + self.value_tokens(v) + [EOS]


def build_mt_world(cfg: MTConfig) -> MTWorld:
    base = WorldConfig(**{k: getattr(cfg, k) for k in WorldConfig.__dataclass_fields__})
    base.n_values = cfg.value_pool ** cfg.value_len
    w = build_world(base)
    mt = MTWorld(**{k: getattr(w, k) for k in World.__dataclass_fields__})
    mt.cfg = cfg
    cfg.n_values = base.n_values
    mt.vocab_size = N_SPECIAL + cfg.n_first + cfg.n_last + cfg.n_relations + cfg.value_pool
    return mt


def mt_seqs(world: MTWorld, mentions, values=None):
    vals = world.values[mentions[:, 0], mentions[:, 1]] if values is None else values
    s = torch.tensor([world.prompt(world.names[e], r) + world.value_tokens(v) + [EOS]
                      for (e, r), v in zip(mentions, vals)], dtype=torch.long)
    return s, torch.ones(len(s), s.shape[1] - 1, dtype=torch.bool)


@torch.no_grad()
def answer_probs_mt(model, world: MTWorld, names, rels, batch=2048):
    """Exact distribution over [IDK, v_0 .. v_{V-1}] for 2-token values.

    p(v) = p(d1) * p(d2 | d1), where p(d1) is normalised together with IDK.
    """
    assert world.cfg.value_len == 2
    P = world.cfg.value_pool
    subs = torch.tensor([world.sub_tok(t) for t in range(P)])
    out = np.zeros((len(names), 1 + P * P), dtype=np.float32)
    for i in range(0, len(names), batch):
        n, r = names[i:i + batch], rels[i:i + batch]
        x = torch.tensor([world.prompt(nm, rr) for nm, rr in zip(n, r)])
        B = len(x)
        first = model(x)[:, -1]
        p1 = torch.softmax(torch.cat([first[:, [IDK]], first[:, subs]], 1), 1)
        x2 = torch.cat([x.repeat_interleave(P, 0), subs.repeat(B)[:, None]], 1)
        p2 = torch.softmax(model(x2)[:, -1][:, subs], 1).view(B, P, P)
        joint = p1[:, 1:, None] * p2
        out[i:i + B, 0] = p1[:, 0].numpy()
        out[i:i + B, 1:] = joint.reshape(B, P * P).numpy()
    return out
