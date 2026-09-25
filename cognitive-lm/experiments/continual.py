"""Learning new facts after training, without forgetting the old ones.

    python -m experiments.continual --seed 0 --out results/continual_seed0.json

1,000 people the model never read about arrive (4,000 new facts, each read
once). Options:

  wake only (CARE)       write them to the filters + overflow store. No gradients.
  fine-tune              train the weights on the new facts only
  + self-replay          ... mixed with the model's own greedy answers for old facts
  + verified replay      ... mixed with only the old answers the CHECKSUM verifies
  + true replay          ... mixed with the original training data (needs the old data kept)

Then each fine-tuned model is put back into CARE (filters + checksum updated,
overflow re-computed), and we measure old and new facts.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.checksum import CheckedRecall
from cogllm.evaluate import score
from cogllm.model import GPT, train_lm
from cogllm.world_mt import MTConfig, answer_probs_mt, build_mt_world, mt_seqs
from experiments.pilot_mt import CARE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--ckpt", default="results/ckpt")
    ap.add_argument("--out", default="results/continual_seed0.json")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    w = build_mt_world(MTConfig(n_first=160, n_last=160, n_entities=10000, zipf_s=1.0, n_mentions=60000, seed=a.seed))
    base = GPT(w.vocab_size, d=a.d, n_layer=2)
    base.load_state_dict(torch.load(f"{a.ckpt}/care_s{a.seed}_ce_d{a.d}.pt"))
    base.eval()

    rng = np.random.default_rng(a.seed + 77)
    never = np.where(w.counts.sum(1) == 0)[0]
    new_ents = rng.choice(never, 1000, replace=False)
    new_ment = np.array([(e, r) for e in new_ents for r in range(w.cfg.n_relations)])
    old_keys = np.argwhere(w.counts > 0)
    all_ment = np.concatenate([w.stream, new_ment])

    # evaluation sets: old seen facts (weighted by popularity via the stream) and the new facts
    old_q = w.stream[rng.integers(0, len(w.stream), 10000)]
    new_q = new_ment

    def evaluate(model, tag):
        res = {}
        cr = CheckedRecall(w, all_ment, triple_bits=14, seed=a.seed)
        care = CARE(w, cr, all_ment, w.values[all_ment[:, 0], all_ment[:, 1]], 20)
        kept = care.sleep(lambda n, r: answer_probs_mt(model, w, n, r))
        for name, qs in [("old", old_q), ("new", new_q)]:
            names, rels = w.names[qs[:, 0]], qs[:, 1]
            truth = w.values[qs[:, 0], qs[:, 1]]
            cnt = np.ones(len(qs), dtype=int)  # all of these were read
            p = answer_probs_mt(model, w, names, rels)[:, 1:]
            res[f"{name}_weights_greedy_acc"] = float((p.argmax(1) == truth).mean())
            pred, _ = care.answer(names, rels, p)
            r = score(np.where(pred >= 0, pred, -1), truth, cnt)
            res[f"{name}_CARE_acc"], res[f"{name}_CARE_halluc"] = r["accuracy"], r["hallucination"]
        res["overflow_store"] = kept
        log(f"{tag:<22} old: weights {res['old_weights_greedy_acc']:.3f} CARE {res['old_CARE_acc']:.3f} | "
            f"new: weights {res['new_weights_greedy_acc']:.3f} CARE {res['new_CARE_acc']:.3f} | "
            f"overflow {kept} | halluc old {res['old_CARE_halluc']:.4f} new {res['new_CARE_halluc']:.4f}")
        return res

    out = {"wake_only": evaluate(base, "wake only (no gradients)")}
    new_s, new_m = mt_seqs(w, new_ment)

    # the model's own answers for old keys: greedy, and checksum-verified
    old_names, old_rels = w.names[old_keys[:, 0]], old_keys[:, 1]
    p_old = answer_probs_mt(base, w, old_names, old_rels)[:, 1:]
    greedy_vals = p_old.argmax(1)
    cr_old = CheckedRecall(w, w.stream, triple_bits=14, seed=a.seed)
    verified = cr_old.answer(None, old_names, old_rels, N=20, probs=p_old)
    ok = verified >= 0
    out["replay_sizes"] = {"self": int(len(old_keys)), "verified": int(ok.sum()),
                           "self_replay_wrong_share": float((greedy_vals != w.values[old_keys[:, 0], old_keys[:, 1]]).mean()),
                           "verified_wrong_share": float((verified[ok] != w.values[old_keys[ok, 0], old_keys[ok, 1]]).mean())}
    log(f"replay sets: {out['replay_sizes']}")
    self_s, self_m = mt_seqs(w, old_keys, greedy_vals)
    ver_s, ver_m = mt_seqs(w, old_keys[ok], verified[ok])
    true_s, true_m = mt_seqs(w, w.stream)

    # every variant sees each new fact R x epochs times; replay variants mix in an
    # equal number of replayed sequences (50/50), drawn fresh from their pool
    R = 4
    new_rep_s, new_rep_m = new_s.repeat(R, 1), new_m.repeat(R, 1)
    g = torch.Generator().manual_seed(a.seed)
    for tag, pool_s, pool_m in [("fine-tune", None, None), ("+ self-replay", self_s, self_m),
                                ("+ verified replay", ver_s, ver_m), ("+ true replay", true_s, true_m)]:
        model = GPT(w.vocab_size, d=a.d, n_layer=2)
        model.load_state_dict(base.state_dict())
        if pool_s is None:
            s2, m2 = new_rep_s, new_rep_m
        else:
            idx = torch.randint(0, len(pool_s), (len(new_rep_s),), generator=g)
            s2, m2 = torch.cat([new_rep_s, pool_s[idx]]), torch.cat([new_rep_m, pool_m[idx]])
        train_lm(model, s2, m2, epochs=a.epochs, lr=1e-3, seed=a.seed)
        out[tag] = evaluate(model, tag)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
