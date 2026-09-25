"""A world where facts are written as short sentences with paraphrases.

    BOS  first last  t1 t2  d1 d2  EOS
         (person)    (one of 3 phrasings of the relation)  (2-token answer)

There is no (entity, relation) key anywhere in the data: a fact is just a
token sequence, and the same relation is phrased three different ways. This
is what an extraction-free method has to cope with.
"""

from dataclasses import dataclass

import numpy as np
import torch

from .world import BOS, EOS, IDK, N_SPECIAL
from .world_mt import MTConfig, MTWorld, build_mt_world


@dataclass
class TextConfig(MTConfig):
    n_templates: int = 3
    template_len: int = 2


class TextWorld(MTWorld):
    def tpl_tok(self, r, k, i):
        c = self.cfg
        base = N_SPECIAL + c.n_first + c.n_last + c.n_relations + c.value_pool
        return base + (r * c.n_templates + k) * c.template_len + i

    def prompt(self, name, r, k=0):
        return [BOS, self.first_tok(name[0]), self.last_tok(name[1])] + \
               [self.tpl_tok(int(r), int(k), i) for i in range(self.cfg.template_len)]

    def sentence(self, e, r, k, v=None):
        v = self.values[e, r] if v is None else v
        return self.prompt(self.names[e], r, k) + self.value_tokens(v) + [EOS]


def build_text_world(cfg: TextConfig) -> TextWorld:
    mt = build_mt_world(cfg)
    tw = TextWorld.__new__(TextWorld)
    tw.__dict__.update(mt.__dict__)
    c = cfg
    tw.vocab_size = N_SPECIAL + c.n_first + c.n_last + c.n_relations + c.value_pool + \
        c.n_relations * c.n_templates * c.template_len
    # every mention gets a random phrasing
    rng = np.random.default_rng(c.seed + 11)
    tw.templates = rng.integers(0, c.n_templates, len(tw.stream))
    # which phrasings each fact was seen with (for the paraphrase test)
    tw.seen_tpl = np.zeros((c.n_entities, c.n_relations, c.n_templates), dtype=np.int32)
    np.add.at(tw.seen_tpl, (tw.stream[:, 0], tw.stream[:, 1], tw.templates), 1)
    return tw


def text_seqs(w: TextWorld, mentions, templates, values=None):
    vals = w.values[mentions[:, 0], mentions[:, 1]] if values is None else values
    return torch.tensor([w.sentence(e, r, k, v) for (e, r), k, v in zip(mentions, templates, vals)], dtype=torch.long)
