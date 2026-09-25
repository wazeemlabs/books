"""Main POC experiment. Everything runs on CPU.

    python -m experiments.run --seed 0 --out results/seed0.json

Compares, on the same synthetic world and the same natural query stream:
  plain LM         weights only, abstains by confidence threshold
  IDK-trained LM   weights only, trained to say IDK (R-Tuning style)
  Bloom-gated LM   plain LM + familiarity filter (no episodic store)
  CMLM (k)         familiarity + episodic store + core consolidated at >= k
  store only       familiarity + episodic store, no core (LMLM-like corner)
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.evaluate import answer_probs, decide, fmt, score
from cogllm.model import GPT, train_lm
from cogllm.system import ABSTAIN_ENTITY, CMLM, stream_tensors
from cogllm.world import EOS, IDK, WorldConfig, build_world, describe, sample_queries

THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def auroc(pos, neg):
    """P(score(pos) > score(neg)), ties count half. Mann-Whitney U."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def frontier(probs, truth, counts, ghost_probs, allow_idk):
    pts = []
    for t in THRESHOLDS:
        r = score(decide(probs, t, allow_idk), truth, counts, decide(ghost_probs, t, allow_idk))
        pts.append({"threshold": t, "accuracy": r["accuracy"], "hallucination": r["hallucination"],
                    "coverage": r["coverage"], "ghost_hallucination": r["ghost_hallucination"]})
    return pts


def best_at(pts, max_hal):
    ok = [p for p in pts if p["hallucination"] <= max_hal]
    return max((p["accuracy"] for p in ok), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="32,64,128")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cmlm_sizes", default="32,64")
    ap.add_argument("--ks", default="1,2,3,5,10")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n_entities", type=int, default=10000)
    ap.add_argument("--n_mentions", type=int, default=60000)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--idk_frac", type=float, default=0.25)
    ap.add_argument("--n_test", type=int, default=20000)
    ap.add_argument("--out", default="results/seed0.json")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    cfg = WorldConfig(n_first=160, n_last=160, n_entities=a.n_entities, zipf_s=a.zipf,
                      n_mentions=a.n_mentions, seed=a.seed)
    w = build_world(cfg)
    log("world\n" + describe(w))
    rng = np.random.default_rng(a.seed + 100)

    # ---- IDK training data (R-Tuning style) ------------------------------
    half = len(w.ghost_names) // 2
    ghost_train, ghost_test = w.ghost_names[:half], w.ghost_names[half:]
    ent_seen = w.counts.sum(1) > 0
    unseen_pairs = np.argwhere((w.counts == 0) & ent_seen[:, None])
    rng.shuffle(unseen_pairs)
    idk_pairs = unseen_pairs[: len(unseen_pairs) // 2]
    idk_set = {(int(e), int(r)) for e, r in idk_pairs}
    n_idk = int(a.idk_frac * a.n_mentions)
    g_idx = rng.integers(0, len(ghost_train), n_idk // 2)
    g_rel = rng.integers(0, cfg.n_relations, n_idk // 2)
    p_idx = rng.integers(0, len(idk_pairs), n_idk - n_idk // 2)
    idk_prompts = [w.prompt(ghost_train[i], r) for i, r in zip(g_idx, g_rel)]
    idk_prompts += [w.prompt(w.names[idk_pairs[i][0]], idk_pairs[i][1]) for i in p_idx]
    idk_seqs = torch.tensor([p + [IDK, EOS] for p in idk_prompts])
    idk_mask = torch.zeros(len(idk_seqs), idk_seqs.shape[1] - 1, dtype=torch.bool)
    idk_mask[:, 3] = True  # only the answer slot: never teach the model a fake name

    # ---- evaluation queries ------------------------------------------------
    q = sample_queries(w, a.n_test * 2, seed=a.seed + 7)
    q = np.array([x for x in q if (int(x[0]), int(x[1])) not in idk_set][: a.n_test])
    q_names, q_rels = w.names[q[:, 0]], q[:, 1]
    q_truth, q_counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]
    q_ent_seen = ent_seen[q[:, 0]]
    gi = rng.integers(0, len(ghost_test), 4000)
    g_names, g_rels = ghost_test[gi], rng.integers(0, cfg.n_relations, 4000)
    mm = float((q_counts == 0).mean())
    log(f"test queries={len(q)} unseen fraction (empirical missing mass)={mm:.3f} "
        f"singleton rate N1/M={w.singleton_rate():.3f}")

    results = {"config": vars(a), "world": {
        "facts": int(w.counts.size), "seen": int((w.counts > 0).sum()), "singletons": int((w.counts == 1).sum()),
        "singleton_rate": w.singleton_rate(), "missing_mass": w.true_missing_mass(), "test_unseen": mm},
        "param": {}, "cmlm": {}}

    stream_seqs, stream_mask = stream_tensors(w, w.stream)
    plain_models = {}

    def membership_stats(probs):
        conf = probs[:, 1:].max(1)
        out = {}
        for c_lo, c_hi, lab in [(1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10**9, "8+")]:
            m = (q_counts >= c_lo) & (q_counts <= c_hi)
            out[f"auroc_seen{lab}_vs_unseen"] = auroc(conf[m], conf[q_counts == 0])
        return out

    # ---- weights-only baselines ---------------------------------------------
    for d in [int(x) for x in a.sizes.split(",")]:
        for kind in ["plain", "idk"]:
            torch.manual_seed(a.seed)
            m = GPT(w.vocab_size, d=d, n_layer=a.layers)
            if kind == "plain":
                seqs, mask = stream_seqs, stream_mask
            else:
                seqs, mask = torch.cat([stream_seqs, idk_seqs]), torch.cat([stream_mask, idk_mask])
            log(f"train {kind} d={d} params={m.n_params()}")
            train_lm(m, seqs, mask, epochs=a.epochs, seed=a.seed, log=log)
            probs = answer_probs(m, w, q_names, q_rels)
            gprobs = answer_probs(m, w, g_names, g_rels)
            allow = kind == "idk"
            base = score(decide(probs, 0.0, allow), q_truth, q_counts, decide(gprobs, 0.0, allow))
            forced = score(decide(probs, 0.0, allow_idk=False), q_truth, q_counts)
            pts = frontier(probs, q_truth, q_counts, gprobs, allow)
            seen = q_counts > 0
            res = {
                "params": m.n_params(), "bits": m.n_params() * 16,
                "default": base, "frontier": pts,
                "acc_at_hal_1pct": best_at(pts, 0.01), "acc_at_hal_2pct": best_at(pts, 0.02),
                "acc_at_hal_5pct": best_at(pts, 0.05),
                # recall on seen facts when forced to answer: the pure "knowing what" skill
                "forced_recall_seen": float((decide(probs, 0.0, False)[seen] == q_truth[seen]).mean()),
                "forced_by_count": forced["by_count"],
                "membership": membership_stats(probs),
            }
            if kind == "idk":
                ans = decide(probs, 0.0, True) >= 0
                res["answer_rate_by_count"] = {
                    lab: float(ans[(q_counts >= lo) & (q_counts <= hi)].mean())
                    for lo, hi, lab in [(0, 0, "0"), (1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 10**9, "8+")]}
                res["answer_rate_ghost"] = float((decide(gprobs, 0.0, True) >= 0).mean())
                pidk = probs[:, 0]
                res["membership"]["auroc_idk_seen1_vs_unseen"] = auroc(-pidk[q_counts == 1], -pidk[q_counts == 0])
            results["param"][f"{kind}_d{d}"] = res
            log(fmt(base, f"{kind} d={d}") + f"\n    acc@hal<=1% {res['acc_at_hal_1pct']:.3f}  "
                f"@2% {res['acc_at_hal_2pct']:.3f}  @5% {res['acc_at_hal_5pct']:.3f}  "
                f"forced recall(seen) {res['forced_recall_seen']:.3f}  membership {res['membership']}")
            if kind == "plain":
                plain_models[d] = m

    # ---- complementary memory -------------------------------------------------
    def eval_cmlm(sys, tag, d):
        pred, src = sys.answer(q_names, q_rels)
        gpred, _ = sys.answer(g_names, g_rels)
        p01 = np.where(pred >= 0, pred, -1)
        r = score(p01, q_truth, q_counts, np.where(gpred >= 0, gpred, -1))
        unseen = q_counts == 0
        graded_ok = np.where(q_ent_seen, pred == -1, pred == ABSTAIN_ENTITY)
        bits = sys.bits()
        r["extra"] = {
            "store_answered": float((src == "store").mean()),
            "core_answered": float((src == "core").mean()),
            "store_entries": len(sys.store),
            "total_bits": int(sum(bits.values())),
        }
        r["bits"] = bits
        r["graded_abstain_acc_on_unseen"] = float(graded_ok[unseen].mean())
        r["ghost_says_unknown_person"] = float((gpred == ABSTAIN_ENTITY).mean())
        r["evicted"], r["failed_consolidation"] = sys.evicted, sys.failed_consolidation
        results["cmlm"][tag] = r
        log(fmt(r, tag) + f"\n    graded-abstain acc {r['graded_abstain_acc_on_unseen']:.3f}  bits {bits}")

    base_sys = CMLM.wake(w, w.stream, k=10**9, seed=a.seed)
    eval_cmlm(base_sys, "store_only", 0)
    for d in [int(x) for x in a.sizes.split(",")]:
        sys = CMLM.wake(w, w.stream, k=10**9, seed=a.seed)
        sys.core, sys.store = plain_models[d], type(sys.store)()  # no store: familiarity + weights only
        eval_cmlm(sys, f"bloom_gated_d{d}", d)
    for d in [int(x) for x in a.cmlm_sizes.split(",")]:
        for k in [int(x) for x in a.ks.split(",")]:
            sys = CMLM.wake(w, w.stream, k=k, seed=a.seed)
            core = plain_models[d] if k == 1 else None  # k=1 means consolidate everything
            log(f"sleep d={d} k={k}")
            sys.sleep(w.stream, d=d, n_layer=a.layers, epochs=a.epochs, seed=a.seed, core=core, log=log)
            eval_cmlm(sys, f"cmlm_d{d}_k{k}", d)
            # recall of the core on the high-count facts it was given (interference test)
            hi = q_counts >= 10
            p = answer_probs(sys.core, w, q_names[hi], q_rels[hi])[:, 1:].argmax(1)
            results["cmlm"][f"cmlm_d{d}_k{k}"]["core_recall_count10plus"] = float((p == q_truth[hi]).mean())

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
