"""Where does the weights' confidence come from: the fact, or the person?

    python -m experiments.anatomy --d 64 --out results/anatomy.json

Trains a plain and an IDK-trained LM on seed 0, then looks at confidence and
answer rates for seen and unseen facts, split by how famous the ENTITY is
(total mentions of that person across all relations). If the weights knew
which facts they had seen, confidence would depend on the fact's own count.
If they only know which people are familiar, confidence follows the person.
"""

import argparse
import json
import os

import numpy as np
import torch

from cogllm.evaluate import answer_probs
from cogllm.model import GPT, train_lm
from cogllm.system import stream_tensors
from cogllm.world import EOS, IDK, WorldConfig, build_world
from experiments.run import auroc

FAME = [(0, 0, "never mentioned"), (1, 2, "1-2"), (3, 7, "3-7"), (8, 31, "8-31"), (32, 10**9, "32+")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/anatomy.json")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    w = build_world(WorldConfig(n_first=160, n_last=160, n_entities=10000, zipf_s=1.0, n_mentions=60000, seed=a.seed))
    rng = np.random.default_rng(a.seed + 100)
    fame = w.counts.sum(1)

    # same IDK recipe as experiments/run.py
    half = len(w.ghost_names) // 2
    ghost_train = w.ghost_names[:half]
    unseen_pairs = np.argwhere((w.counts == 0) & (fame > 0)[:, None])
    rng.shuffle(unseen_pairs)
    idk_pairs = unseen_pairs[: len(unseen_pairs) // 2]
    idk_set = {(int(e), int(r)) for e, r in idk_pairs}
    n_idk = int(0.25 * 60000)
    g_idx = rng.integers(0, len(ghost_train), n_idk // 2)
    g_rel = rng.integers(0, 4, n_idk // 2)
    p_idx = rng.integers(0, len(idk_pairs), n_idk - n_idk // 2)
    prompts = [w.prompt(ghost_train[i], r) for i, r in zip(g_idx, g_rel)]
    prompts += [w.prompt(w.names[idk_pairs[i][0]], idk_pairs[i][1]) for i in p_idx]
    idk_seqs = torch.tensor([p + [IDK, EOS] for p in prompts])
    idk_mask = torch.zeros(len(idk_seqs), idk_seqs.shape[1] - 1, dtype=torch.bool)
    idk_mask[:, 3] = True

    # every fact of every entity, minus the pairs used to teach IDK
    allp = np.array([(e, r) for e in range(w.cfg.n_entities) for r in range(4) if (e, r) not in idk_set])
    names, rels = w.names[allp[:, 0]], allp[:, 1]
    cnt, fm = w.counts[allp[:, 0], allp[:, 1]], fame[allp[:, 0]]
    out = {"d": a.d}
    s, m = stream_tensors(w, w.stream)
    for kind in ["plain", "idk"]:
        torch.manual_seed(a.seed)
        model = GPT(w.vocab_size, d=a.d, n_layer=2)
        if kind == "plain":
            train_lm(model, s, m, epochs=a.epochs, seed=a.seed, log=print)
        else:
            train_lm(model, torch.cat([s, idk_seqs]), torch.cat([m, idk_mask]), epochs=a.epochs, seed=a.seed, log=print)
        p = answer_probs(model, w, names, rels)
        conf = p[:, 1:].max(1)
        answers = p[:, 1:].max(1) > p[:, 0]
        res = {"by_fame": {}}
        for lo, hi, lab in FAME:
            f = (fm >= lo) & (fm <= hi)
            row = {}
            for grp, sel in [("unseen", f & (cnt == 0)), ("seen1", f & (cnt == 1)), ("seen2+", f & (cnt >= 2))]:
                if sel.sum() == 0:
                    continue
                row[grp] = {"n": int(sel.sum()), "mean_conf": float(conf[sel].mean()),
                            "answer_rate": float(answers[sel].mean())}
            if (f & (cnt == 1)).sum() and (f & (cnt == 0)).sum():
                row["auroc_seen1_vs_unseen_same_fame"] = auroc(conf[f & (cnt == 1)], conf[f & (cnt == 0)])
            res["by_fame"][lab] = row
        res["auroc_seen1_vs_unseen_overall"] = auroc(conf[cnt == 1], conf[cnt == 0])
        res["auroc_fame_predicts_conf_on_unseen"] = auroc(conf[(cnt == 0) & (fm >= 8)], conf[(cnt == 0) & (fm == 0)])
        out[kind] = res
        print(kind, json.dumps(res, indent=1))
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
