"""Relation-free checksum: (entity, value) pairs instead of (entity, relation, value).

    python -m experiments.assoc --seed 0

Building the triple checksum needs relation extraction. Building a pair
checksum needs only entity recognition + linking: "were these two things
ever mentioned together?" (associative recognition). Cheaper and immune to
how the relation is phrased, but the checksum can no longer tell WHICH
relation the pair came from. How much hallucination does that cost?
Uses the weights trained by experiments/run_care.py.
"""

import argparse
import json
import os

import numpy as np
import torch

from cogllm.checksum import CheckedRecall
from cogllm.evaluate import score
from cogllm.model import GPT
from cogllm.world import sample_queries
from cogllm.world_mt import MTConfig, answer_probs_mt, build_mt_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="32,64,96")
    ap.add_argument("--ckpt", default="results/ckpt")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    w = build_mt_world(MTConfig(n_first=160, n_last=160, n_entities=10000, zipf_s=1.0, n_mentions=60000, seed=a.seed))
    # same test set as run_care.py
    rng = np.random.default_rng(a.seed + 100)
    ent_seen = w.counts.sum(1) > 0
    unseen_pairs = np.argwhere((w.counts == 0) & ent_seen[:, None])
    rng.shuffle(unseen_pairs)
    idk_set = {(int(e), int(r)) for e, r in unseen_pairs[: len(unseen_pairs) // 2]}
    q = sample_queries(w, 40000, seed=a.seed + 7)
    q = np.array([x for x in q if (int(x[0]), int(x[1])) not in idk_set][:20000])
    qn, qr = w.names[q[:, 0]], q[:, 1]
    truth, counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]

    triple = CheckedRecall(w, w.stream, triple_bits=14, seed=a.seed)
    flat = w.stream.copy()
    flat[:, 1] = 0  # forget the relation
    pair = CheckedRecall(w, flat, triple_bits=14, seed=a.seed, values=w.values[w.stream[:, 0], w.stream[:, 1]])
    zeros = np.zeros_like(qr)
    out = {}
    for d in [int(x) for x in a.sizes.split(",")]:
        m = GPT(w.vocab_size, d=d, n_layer=2)
        m.load_state_dict(torch.load(f"{a.ckpt}/care_s{a.seed}_ce_d{d}.pt"))
        m.eval()
        p = answer_probs_mt(m, w, qn, qr)[:, 1:]
        for N in [1, 5, 20]:
            for tag, cr, rels in [("triple", triple, qr), ("pair", pair, zeros)]:
                pred = cr.answer(None, qn, rels, N=N, probs=p)
                r = score(np.where(pred >= 0, pred, -1), truth, counts)
                out[f"d{d}_N{N}_{tag}"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
            print(f"d={d} N={N}: triple acc {out[f'd{d}_N{N}_triple']['accuracy']:.3f} hal "
                  f"{out[f'd{d}_N{N}_triple']['hallucination']:.4f} | pair acc {out[f'd{d}_N{N}_pair']['accuracy']:.3f} "
                  f"hal {out[f'd{d}_N{N}_pair']['hallucination']:.4f}", flush=True)
    json.dump(out, open(f"results/assoc_seed{a.seed}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
