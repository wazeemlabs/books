"""Pilot: FamLM (prequential familiarity sense) vs baselines, single pass.

    python -m experiments.pilot_fam --out results/pilot_fam.json

Systems (same data stream, one pass, same transformer size):
  lm            plain transformer
  lm_idk        + R-Tuning-style IDK examples
  fam           + familiarity embedding + recognition bias, prequential counts
  fam_ctx       familiarity embedding only (no recognition bias)
  fam_oracle    same as fam but the sketch is filled with the WHOLE stream
                before training (never experiences a first exposure)
  ngram         the count table alone (exact phrasing lookup), no transformer
Tests: natural queries, ghost people, paraphrases (fact read, but never in
this phrasing), and ingestion (new facts only counted, no gradient step).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.evaluate import score
from cogllm.famlm import N_BUCKETS, CountSketch, FamGPT, train_prequential
from cogllm.textworld import TextConfig, build_text_world, text_seqs
from cogllm.world import EOS, IDK, sample_queries
from experiments.run import auroc


@torch.no_grad()
def answer_probs(model, sketch, w, names, rels, tpls, batch=1024):
    """(n, 1 + V): IDK prob, then joint prob of each 2-token value."""
    P = w.cfg.value_pool
    subs = torch.tensor([w.sub_tok(t) for t in range(P)])
    out = np.zeros((len(names), 1 + P * P), dtype=np.float32)
    for i in range(0, len(names), batch):
        x = torch.tensor([w.prompt(n, r, k) for n, r, k in zip(names[i:i + batch], rels[i:i + batch], tpls[i:i + batch])])
        B = len(x)
        cset = torch.cat([torch.tensor([IDK]), subs])
        l1 = model(x, sketch, cand=cset)[:, -1]
        p1 = torch.softmax(torch.cat([l1[:, [IDK]], l1[:, subs]], 1), 1)
        x2 = torch.cat([x.repeat_interleave(P, 0), subs.repeat(B)[:, None]], 1)
        p2 = torch.softmax(model(x2, sketch, cand=subs)[:, -1][:, subs], 1).view(B, P, P)
        out[i:i + B, 0] = p1[:, 0].numpy()
        out[i:i + B, 1:] = (p1[:, 1:, None] * p2).reshape(B, P * P).numpy()
    return out


def ngram_answer(sketch, w, names, rels, tpls, min_order_idx=2):
    """Exact-phrasing lookup: follow the highest-order n-gram with next counts; abstain if unseen."""
    P = w.cfg.value_pool
    subs = torch.tensor([w.sub_tok(t) for t in range(P)])
    x = torch.tensor([w.prompt(n, r, k) for n, r, k in zip(names, rels, tpls)])
    c1, H1 = sketch.features(x)
    n1 = sketch.next_buckets(H1, subs[None, None, :].expand(len(x), x.shape[1], len(subs)))
    pred = np.full(len(x), -1)
    oi = min_order_idx  # order index whose n-gram covers the person + phrasing
    ok = (c1[:, -1, oi] > 0) & (c1[:, -1, oi] < N_BUCKETS - 1)
    d1 = n1[:, -1, oi, :].argmax(1)
    has1 = n1[:, -1, oi, :].max(1).values > 0
    x2 = torch.cat([x, subs[d1][:, None]], 1)
    c2, H2 = sketch.features(x2)
    n2 = sketch.next_buckets(H2, subs[None, None, :].expand(len(x2), x2.shape[1], len(subs)))
    d2 = n2[:, -1, oi + 1, :].argmax(1)
    has2 = n2[:, -1, oi + 1, :].max(1).values > 0
    good = (ok & has1 & has2).numpy()
    pred[good] = (d1 * P + d2).numpy()[good]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--n_entities", type=int, default=20000)
    ap.add_argument("--n_mentions", type=int, default=400000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--systems", default="lm,lm_idk,fam,fam_ctx,fam_oracle")
    ap.add_argument("--out", default="results/pilot_fam.json")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    nf = int(np.ceil(np.sqrt(2 * a.n_entities)))
    w = build_text_world(TextConfig(n_first=nf, n_last=nf, n_entities=a.n_entities, zipf_s=1.0,
                                    n_mentions=a.n_mentions, seed=a.seed))
    V = w.cfg.n_values
    seqs = text_seqs(w, w.stream, w.templates)
    log(f"world: {a.n_entities} people, {a.n_mentions} mentions, vocab {w.vocab_size}, "
        f"seen facts {(w.counts > 0).sum()}, singleton rate {w.singleton_rate():.3f}")
    rng = np.random.default_rng(a.seed + 1)

    # IDK examples for lm_idk (ghost names from the first half, unseen facts of known people)
    half = len(w.ghost_names) // 2
    ent_seen = w.counts.sum(1) > 0
    unseen_pairs = np.argwhere((w.counts == 0) & ent_seen[:, None])
    rng.shuffle(unseen_pairs)
    idk_pairs = unseen_pairs[: len(unseen_pairs) // 2]
    idk_set = {(int(e), int(r)) for e, r in idk_pairs}
    n_idk = 50000
    gi = rng.integers(0, half, n_idk // 2)
    pi = rng.integers(0, len(idk_pairs), n_idk // 2)
    prompts = [w.prompt(w.ghost_names[i], rng.integers(4), rng.integers(3)) for i in gi]
    prompts += [w.prompt(w.names[idk_pairs[i][0]], idk_pairs[i][1], rng.integers(3)) for i in pi]
    idk_seqs = torch.tensor([p + [IDK, EOS, EOS] for p in prompts])
    idk_mask = torch.zeros(len(idk_seqs), 7, dtype=torch.bool)
    idk_mask[:, 4] = True

    # test sets
    q = sample_queries(w, 20000, seed=a.seed + 7)
    q = np.array([x for x in q if (int(x[0]), int(x[1])) not in idk_set][:10000])
    qt = rng.integers(0, 3, len(q))
    qn, qr = w.names[q[:, 0]], q[:, 1]
    truth, counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]
    g_names = w.ghost_names[rng.integers(half, len(w.ghost_names), 3000)]
    g_rels, g_tpls = rng.integers(0, 4, 3000), rng.integers(0, 3, 3000)
    # paraphrase: facts read, but never in this phrasing
    cand = np.argwhere((w.counts > 0) & ((w.seen_tpl == 0).sum(-1) > 0))
    cand = cand[rng.choice(len(cand), min(4000, len(cand)), replace=False)]
    para_t = np.array([rng.choice(np.where(w.seen_tpl[e, r] == 0)[0]) for e, r in cand])
    seen_t = np.array([rng.choice(np.where(w.seen_tpl[e, r] > 0)[0]) for e, r in cand])
    # ingestion: people never mentioned; their facts are only COUNTED after training
    never = np.where(~ent_seen)[0]
    new_e = rng.choice(never, min(500, len(never)), replace=False)
    new_m = np.array([(e, r) for e in new_e for r in range(4)])
    new_t = rng.integers(0, 3, len(new_m))
    new_other_t = (new_t + 1 + rng.integers(0, 2, len(new_m))) % 3

    out = {"world": {"seen_facts": int((w.counts > 0).sum()), "singleton_rate": w.singleton_rate(),
                     "test_unseen": float((counts == 0).mean())}, "systems": {}}

    def evaluate(tag, model, sketch):
        res = {}
        p = answer_probs(model, sketch, w, qn, qr, qt)
        vals, conf = p[:, 1:], p[:, 1:].max(1)
        best = vals.argmax(1)
        idk = p[:, 0] > conf
        pts = []
        for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            r = score(np.where(idk | (conf < t), -1, best), truth, counts)
            pts.append((t, r["accuracy"], r["hallucination"]))
        res["greedy"] = {"accuracy": pts[0][1], "hallucination": pts[0][2]}
        res["acc_at_hal_1pct"] = max([acc for _, acc, h in pts if h <= 0.01], default=0.0)
        res["acc_at_hal_0.1pct"] = max([acc for _, acc, h in pts if h <= 0.001], default=0.0)
        res["frontier"] = pts
        sc = conf * (1 - p[:, 0])
        res["auroc_seen1_vs_unseen"] = auroc(sc[counts == 1], sc[counts == 0])
        res["auroc_seen_vs_unseen"] = auroc(sc[counts > 0], sc[counts == 0])
        gp = answer_probs(model, sketch, w, g_names, g_rels, g_tpls)
        res["ghost_mean_conf"] = float((gp[:, 1:].max(1) * (1 - gp[:, 0])).mean())
        # paraphrase
        for lab, tp in [("seen_phrasing", seen_t), ("new_phrasing", para_t)]:
            pp = answer_probs(model, sketch, w, w.names[cand[:, 0]], cand[:, 1], tp)
            tr = w.values[cand[:, 0], cand[:, 1]]
            res[f"para_{lab}_acc"] = float((pp[:, 1:].argmax(1) == tr).mean())
            res[f"para_{lab}_conf"] = float((pp[:, 1:].max(1) * (1 - pp[:, 0])).mean())
        # ingestion (counts only, no gradients) -- on a copy of the sketch
        if sketch is not None:
            import copy
            sk = copy.deepcopy(sketch)
            sk.add(text_seqs(w, new_m, new_t))
            for lab, tp in [("same_phrasing", new_t), ("other_phrasing", new_other_t)]:
                pn = answer_probs(model, sk, w, w.names[new_m[:, 0]], new_m[:, 1], tp)
                tr = w.values[new_m[:, 0], new_m[:, 1]]
                res[f"ingest_{lab}_acc"] = float((pn[:, 1:].argmax(1) == tr).mean())
                res[f"ingest_{lab}_conf"] = float((pn[:, 1:].max(1) * (1 - pn[:, 0])).mean())
        out["systems"][tag] = res
        log(f"{tag:<11} greedy acc {res['greedy']['accuracy']:.3f} hal {res['greedy']['hallucination']:.3f} | "
            f"acc@hal<=1% {res['acc_at_hal_1pct']:.3f} @0.1% {res['acc_at_hal_0.1pct']:.3f} | AUROC seen1/unseen "
            f"{res['auroc_seen1_vs_unseen']:.3f} | ghost conf {res['ghost_mean_conf']:.3f} | para acc "
            f"{res['para_seen_phrasing_acc']:.3f}->{res['para_new_phrasing_acc']:.3f}"
            + (f" | ingest acc same {res['ingest_same_phrasing_acc']:.3f} other {res['ingest_other_phrasing_acc']:.3f}"
               if sketch is not None else ""))

    final_sketch = None
    for tag in a.systems.split(","):
        torch.manual_seed(a.seed)
        sk = CountSketch()
        use = tag.startswith("fam")
        model = FamGPT(w.vocab_size, len(sk.orders), d=a.d, use_ctx=use, use_next=use and tag != "fam_ctx")
        log(f"train {tag} ({model.n_params()} params, sketch {sk.n_bits() / 8e6:.0f} MB at 8-bit counters)")
        if tag == "fam_oracle":
            for i in range(0, len(seqs), 4096):
                sk.add(seqs[i:i + 4096])
            train_prequential(model, sk, seqs, epochs=a.epochs, seed=a.seed, update=False, log=log)
        else:
            train_prequential(model, sk if use else None, seqs, epochs=a.epochs, seed=a.seed, log=log,
                              extra=(idk_seqs, idk_mask) if tag == "lm_idk" else None)
        if tag == "fam":
            final_sketch = sk
        evaluate(tag, model, sk if use else None)

    if final_sketch is not None:
        pred = ngram_answer(final_sketch, w, qn, qr, qt)
        r = score(pred, truth, counts)
        pp = ngram_answer(final_sketch, w, w.names[cand[:, 0]], cand[:, 1], para_t)
        out["systems"]["ngram"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"],
                                   "para_new_phrasing_acc": float((pp == w.values[cand[:, 0], cand[:, 1]]).mean())}
        log(f"ngram only  acc {r['accuracy']:.3f} hal {r['hallucination']:.4f} | para new phrasing acc "
            f"{out['systems']['ngram']['para_new_phrasing_acc']:.3f}")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
