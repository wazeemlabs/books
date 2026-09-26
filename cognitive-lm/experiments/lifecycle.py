"""Lifelong CARE: several days of wake (read new facts) and sleep (consolidate).

    python -m experiments.lifecycle --seed 0 --days 5 --out results/lifecycle_seed0.json

Each day the world introduces 500 new people (Zipf-popular, so some facts
are read many times and many only once) and keeps talking about old people.

  wake   every mention is written into the filters and the episodic overflow
         store. No gradients: new facts are answerable immediately.
  sleep  the weights are trained on the overflow facts, plus a replay set to
         avoid forgetting; then every overflow fact the weights + checksum
         now recall is evicted.

Policies compared (same days, same data):
  no_sleep          never train; the store only grows
  sleep             train on overflow facts only
  sleep_self        + replay of the model's own greedy answers for old facts
  sleep_verified    + replay of only the old answers the checksum verifies
Measured every day: CARE accuracy and hallucination (old facts, new facts,
never-seen facts), store size, and what the weights alone still recall.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.checksum import CheckedRecall
from cogllm.evaluate import score
from cogllm.memory import EpisodicStore
from cogllm.system import key_bits, value_bits
from cogllm.model import GPT, train_lm
from cogllm.world_mt import MTConfig, answer_probs_mt, build_mt_world, mt_seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--new_people", type=int, default=500)
    ap.add_argument("--new_mentions", type=int, default=4000)
    ap.add_argument("--old_mentions", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--policies", default="no_sleep,sleep,sleep_self,sleep_verified,sleep_audit")
    ap.add_argument("--ckpt", default="results/ckpt")
    ap.add_argument("--out", default="results/lifecycle_seed0.json")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    w = build_mt_world(MTConfig(n_first=160, n_last=160, n_entities=10000, zipf_s=1.0, n_mentions=60000, seed=a.seed))
    rng = np.random.default_rng(a.seed + 404)
    never = np.where(w.counts.sum(1) == 0)[0]
    rng.shuffle(never)

    # the days' streams are fixed up front so every policy sees the same data
    days = []
    for day in range(a.days):
        people = never[day * a.new_people:(day + 1) * a.new_people]
        pop = 1.0 / np.arange(1, len(people) + 1) ** 1.0
        pop /= pop.sum()
        e = rng.choice(people, a.new_mentions, p=pop)
        r = rng.integers(0, w.cfg.n_relations, a.new_mentions)
        flat = rng.choice(w.fact_p.size, a.old_mentions, p=w.fact_p.ravel())
        old = np.stack(np.unravel_index(flat, w.fact_p.shape), axis=1)
        days.append({"people": people, "mentions": np.concatenate([np.stack([e, r], 1), old])})
    unseen_q = np.argwhere(w.counts == 0)
    unseen_q = unseen_q[~np.isin(unseen_q[:, 0], never[:a.days * a.new_people])]
    unseen_q = unseen_q[rng.choice(len(unseen_q), 4000, replace=False)]

    base = GPT(w.vocab_size, d=a.d, n_layer=2)
    base.load_state_dict(torch.load(f"{a.ckpt}/care_s{a.seed}_ce_d{a.d}.pt"))
    out = {"config": vars(a), "policies": {}}

    for policy in a.policies.split(","):
        model = GPT(w.vocab_size, d=a.d, n_layer=2)
        model.load_state_dict(base.state_dict())
        model.eval()
        mentions = w.stream.copy()
        store = EpisodicStore()
        for (e, r) in mentions:
            store.write((int(e), int(r)), int(w.values[e, r]))
        hist = []
        for day in range(-1, a.days):
            if day >= 0:
                # ---- wake: read today's mentions (filters + store, no gradients)
                today = days[day]["mentions"]
                mentions = np.concatenate([mentions, today])
                for (e, r) in today:
                    store.write((int(e), int(r)), int(w.values[e, r]))
            cr = CheckedRecall(w, mentions, triple_bits=14, seed=a.seed)
            keys = np.unique(mentions, axis=0)

            if day >= 0 and policy != "no_sleep":
                # ---- sleep: consolidate the overflow store into the weights
                sk = np.array(list(store.vals.keys()))
                sv = np.array([store.recall(tuple(k)) for k in sk])
                s_seq, s_mask = mt_seqs(w, sk, sv)
                seqs, mask = s_seq.repeat(2, 1), s_mask.repeat(2, 1)
                stored = {(int(k[0]), int(k[1])) for k in sk}
                old = np.array([k for k in keys if (int(k[0]), int(k[1])) not in stored]).reshape(-1, 2)
                if policy in ("sleep_self", "sleep_verified", "sleep_audit") and len(old):
                    old = old[rng.choice(len(old), min(len(old), len(seqs)), replace=False)]
                    p = answer_probs_mt(model, w, w.names[old[:, 0]], old[:, 1])[:, 1:]
                    if policy == "sleep_self":
                        rv, keep = p.argmax(1), np.ones(len(old), bool)
                    else:
                        rv = cr.answer(None, w.names[old[:, 0]], old[:, 1], N=20, probs=p)
                        keep = rv >= 0
                    r_seq, r_mask = mt_seqs(w, old[keep], rv[keep])
                    seqs, mask = torch.cat([seqs, r_seq]), torch.cat([mask, r_mask])
                audit_writebacks = 0
                cons = np.array([k for k in keys if (int(k[0]), int(k[1])) not in stored]).reshape(-1, 2)
                if policy == "sleep_audit" and len(cons):
                    # remember what the pre-sleep weights recall (verified) for every consolidated key
                    p_old = answer_probs_mt(model, w, w.names[cons[:, 0]], cons[:, 1])[:, 1:]
                    v_old = cr.answer(None, w.names[cons[:, 0]], cons[:, 1], N=20, probs=p_old)
                train_lm(model, seqs, mask, epochs=a.epochs, lr=1e-3, seed=a.seed + day)
                if policy == "sleep_audit" and len(cons):
                    # two-phase commit: anything the new weights no longer recall goes back into the store
                    p_new = answer_probs_mt(model, w, w.names[cons[:, 0]], cons[:, 1])[:, 1:]
                    v_new = cr.answer(None, w.names[cons[:, 0]], cons[:, 1], N=20, probs=p_new)
                    lost = (v_old >= 0) & (v_new != v_old)
                    for k, v in zip(cons[lost], v_old[lost]):
                        store.write((int(k[0]), int(k[1])), int(v))
                    audit_writebacks = int(lost.sum())
                # evict what the weights + checksum now recall
                p = answer_probs_mt(model, w, w.names[sk[:, 0]], sk[:, 1])[:, 1:]
                rec = cr.answer(None, w.names[sk[:, 0]], sk[:, 1], N=20, probs=p)
                for k, v, pr in zip(sk, sv, rec):
                    if pr == v:
                        store.evict((int(k[0]), int(k[1])))

            # ---- measure
            def care_answer(qk):
                names, rels = w.names[qk[:, 0]], qk[:, 1]
                p = answer_probs_mt(model, w, names, rels)[:, 1:]
                pred = cr.answer(None, names, rels, N=20, probs=p)
                for i, (e, r) in enumerate(qk):
                    if (int(e), int(r)) in store:
                        pred[i] = store.recall((int(e), int(r)))
                return pred, p.argmax(1)

            row = {"day": day + 1, "store": len(store), "familiar_keys": int(len(keys))}
            # Replay and audit enumerate every key read so far, which a Bloom filter
            # cannot do, so those policies also keep an exact key list. Count it.
            kb, vb = key_bits(w), value_bits(w)
            row["memory_bits"] = {"store": store.n_bits(kb, vb),
                                  "key_list": len(keys) * kb if policy in ("sleep_self", "sleep_verified", "sleep_audit") else 0,
                                  "filters": sum(cr.bits().values())}
            if policy == "sleep_audit" and day >= 0:
                row["audit_writebacks"] = audit_writebacks
            groups = {"day0": w.stream[rng.integers(0, len(w.stream), 3000)]}
            for dd in range(max(0, day + 1)):
                m = days[dd]["mentions"][: a.new_mentions]
                groups[f"day{dd + 1}"] = m[rng.integers(0, len(m), 1500)]
            for g, qk in groups.items():
                pred, greedy = care_answer(qk)
                truth = w.values[qk[:, 0], qk[:, 1]]
                r = score(np.where(pred >= 0, pred, -1), truth, np.ones(len(qk), int))
                row[f"{g}_care_acc"], row[f"{g}_care_hal"] = r["accuracy"], r["hallucination"]
                row[f"{g}_weights_acc"] = float((greedy == truth).mean())
            seen_now = {(int(e), int(r)) for e, r in keys}
            uq = np.array([k for k in unseen_q if (int(k[0]), int(k[1])) not in seen_now])
            pred, _ = care_answer(uq)
            row["unseen_hal"] = float((pred >= 0).mean())
            hist.append(row)
            new_str = " ".join(f"d{dd + 1} {row[f'day{dd + 1}_care_acc']:.3f}/{row[f'day{dd + 1}_weights_acc']:.2f}"
                               for dd in range(max(0, day + 1)))
            log(f"{policy:<15} day {day + 1}: store {len(store):>6} | day0 CARE {row['day0_care_acc']:.3f} "
                f"weights {row['day0_weights_acc']:.3f} | new (CARE/weights) {new_str} | unseen halluc {row['unseen_hal']:.4f}")
        out["policies"][policy] = hist
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
