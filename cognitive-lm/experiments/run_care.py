"""Final experiment: checksum-aided recall on the hard world (1024 values).

    python -m experiments.run_care --seed 0 --out results/care_seed0.json

Per generator size:
    weights only          greedy, and best accuracy at <=1% / <=0.1% hallucination
    weights + IDK         R-Tuning-style "I don't know" training, same sweep
    CAR  (N = 1, 5, 20)   weights propose a top-N list, checksum recognises
    CAR + prefix (N=50)   prefix checksum prunes wrong branches first
    CARE (N = 20)         CAR + episodic overflow for what CAR can't recall
plus the store-only system, noisy extraction, and a check of the
list-decoding error formula in docs/theory.md against the measurements.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.baselines import ExactDict, KeyFilterStatic
from cogllm.checksum import TIP_OF_TONGUE, UNKNOWN_ENTITY, UNKNOWN_FACT, CheckedRecall
from cogllm.evaluate import BUCKETS, score, wilson_upper
from cogllm.memory import EpisodicStore
from cogllm.model import GPT, train_lm
from cogllm.system import CMLM, key_bits
from cogllm.world import EOS, IDK, count_buckets, sample_queries
from cogllm.world_mt import MTConfig, answer_probs_mt, build_mt_world, mt_seqs
from experiments.pilot_mt import CARE, READ_ONCE

THRESH = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]


def size_checksum(w, q, qn, qr, truth, counts, probs, rank, eps_k, a, N=20, targets=(0.01, 0.003, 0.001)):
    """Out-of-sample test of the checksum-size formula (docs/theory.md section 5).

    Split people in two. On the calibration half, measure the never-seen share
    MM and E[min(R - 1, N)], and solve the formula for the triple false-positive
    rate that meets each hallucination target:
        eps_t = target / (MM * eps_k * N + (1 - MM) * E[min(R - 1, N)])
    then size a fresh Bloom checksum for that rate and measure on the other half.
    """
    side = np.random.default_rng(a.seed + 77).random(w.cfg.n_entities) < 0.5
    cal, test = side[q[:, 0]], ~side[q[:, 0]]
    seen_c = (counts > 0) & cal
    mm = float((counts[cal] == 0).mean())
    e_r = float(np.minimum(rank[seen_c] - 1, N).mean())
    rows = []
    for target in targets:
        eps_t = target / (mm * eps_k * N + (1 - mm) * e_r)
        b = float(np.log2(1 / eps_t) / np.log(2))  # optimal Bloom: 1.44 log2(1/eps) bits per item
        c = CheckedRecall(w, w.stream, triple_bits=b, seed=a.seed + 1000)
        pred = c.answer(None, qn[test], qr[test], N=N, probs=probs[test])
        r = score(np.where(pred >= 0, pred, -1), truth[test], counts[test])
        halluc = int(round(r["hallucination"] * r["n"]))
        rows.append({"target": target, "bits_per_fact": b, "calib_mm": mm, "calib_E_min_R": e_r,
                     "measured": r["hallucination"], "n_test": r["n"],
                     "measured_ub95": wilson_upper(halluc, r["n"]), "accuracy": r["accuracy"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="16,32,64,96")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--triple_bits", type=float, default=14)
    ap.add_argument("--noise_size", type=int, default=32)
    ap.add_argument("--train_filter_k", type=int, default=1, help="train weights only on facts seen >= k times")
    ap.add_argument("--ckpt", default="results/ckpt")
    ap.add_argument("--out", default="results/care_seed0.json")
    ap.add_argument("--n_entities", type=int, default=10000)
    ap.add_argument("--n_mentions", type=int, default=60000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    w = build_mt_world(MTConfig(n_first=160, n_last=160, n_entities=a.n_entities, zipf_s=1.0, n_mentions=a.n_mentions, seed=a.seed))
    V = w.cfg.n_values
    seen_facts = int((w.counts > 0).sum())
    rng = np.random.default_rng(a.seed + 100)

    # IDK data exactly as in experiments/run.py, and its pairs excluded from the test
    half = len(w.ghost_names) // 2
    ghost_train, ghost_test = w.ghost_names[:half], w.ghost_names[half:]
    ent_seen = w.counts.sum(1) > 0
    unseen_pairs = np.argwhere((w.counts == 0) & ent_seen[:, None])
    rng.shuffle(unseen_pairs)
    idk_pairs = unseen_pairs[: len(unseen_pairs) // 2]
    idk_set = {(int(e), int(r)) for e, r in idk_pairs}
    n_idk = int(0.25 * len(w.stream))
    g_idx = rng.integers(0, len(ghost_train), n_idk // 2)
    g_rel = rng.integers(0, w.cfg.n_relations, n_idk // 2)
    p_idx = rng.integers(0, len(idk_pairs), n_idk - n_idk // 2)
    prompts = [w.prompt(ghost_train[i], r) for i, r in zip(g_idx, g_rel)]
    prompts += [w.prompt(w.names[idk_pairs[i][0]], idk_pairs[i][1]) for i in p_idx]
    idk_seqs = torch.tensor([p + [IDK, EOS, EOS] for p in prompts])
    idk_mask = torch.zeros(len(idk_seqs), idk_seqs.shape[1] - 1, dtype=torch.bool)
    idk_mask[:, 3] = True

    q = sample_queries(w, 40000, seed=a.seed + 7)
    q = np.array([x for x in q if (int(x[0]), int(x[1])) not in idk_set][:20000])
    qn, qr = w.names[q[:, 0]], q[:, 1]
    truth, counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]
    gi = rng.integers(0, len(ghost_test), 4000)
    gn, gr = ghost_test[gi], rng.integers(0, w.cfg.n_relations, 4000)
    mm = float((counts == 0).mean())
    log(f"test: {len(q)} queries, unseen share {mm:.3f}, singleton rate {w.singleton_rate():.3f}")
    out = {"config": vars(a), "world": {"seen_facts": seen_facts, "singleton_rate": w.singleton_rate(),
                                        "missing_mass": w.true_missing_mass(), "test_unseen": mm},
           "sizes": {}, "noise": []}

    stream = w.stream
    if a.train_filter_k > 1:
        stream = stream[w.counts[stream[:, 0], stream[:, 1]] >= a.train_filter_k]
    s_all, m_all = mt_seqs(w, stream)
    os.makedirs(a.ckpt, exist_ok=True)

    def gen(tag, d, seqs, mask):
        path = f"{a.ckpt}/care_s{a.seed}_{tag}.pt"
        torch.manual_seed(a.seed)
        m = GPT(w.vocab_size, d=d, n_layer=2)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path))
        else:
            log(f"train {tag} ({m.n_params()} params)")
            train_lm(m, seqs, mask, a.epochs, seed=a.seed)
            torch.save(m.state_dict(), path)
        m.eval()
        return m

    def summarize(pred, gpred, bits=None, extra=None):
        r = score(np.where(pred >= 0, pred, -1), truth, counts, np.where(gpred >= 0, gpred, -1))
        if bits is not None:
            r["bits"] = bits
            r["non_weight_bits_per_seen_fact"] = sum(v for k, v in bits.items() if k != "weights") / seen_facts
            r["total_bits_per_seen_fact"] = sum(bits.values()) / seen_facts
        r["levels"] = {lab: float((pred == c).mean()) for lab, c in
                       [("unknown_entity", UNKNOWN_ENTITY), ("unknown_fact", UNKNOWN_FACT), ("tip_of_tongue", TIP_OF_TONGUE),
                        ("read_once", READ_ONCE)]}
        r.update(extra or {})
        return r

    def weights_sweep(p, gp, allow_idk):
        """p: (n, 1+V) with IDK in column 0."""
        vals, gvals = p[:, 1:], gp[:, 1:]
        conf, best = vals.max(1), vals.argmax(1)
        gconf, gbest = gvals.max(1), gvals.argmax(1)
        idk = (p[:, 0] > conf) if allow_idk else np.zeros(len(p), bool)
        gidk = (gp[:, 0] > gconf) if allow_idk else np.zeros(len(gp), bool)
        pts = []
        for t in THRESH:
            pr = np.where(idk | (conf < t), -1, best)
            gpr = np.where(gidk | (gconf < t), -1, gbest)
            r = score(pr, truth, counts, gpr)
            pts.append({"t": t, "accuracy": r["accuracy"], "hallucination": r["hallucination"],
                        "ghost_hallucination": r["ghost_hallucination"]})
        op = max([x for x in pts if x["hallucination"] <= 0.01], key=lambda x: x["accuracy"], default=None)
        return {"frontier": pts, "default": pts[0],
                "acc_at_hal_1pct": op["accuracy"] if op else 0.0,
                # ghost-people rate at that same operating point, not at threshold 0
                "ghost_at_hal_1pct": op["ghost_hallucination"] if op else None,
                "acc_at_hal_0.1pct": max([x["accuracy"] for x in pts if x["hallucination"] <= 0.001], default=0.0)}

    cr = CheckedRecall(w, w.stream, triple_bits=a.triple_bits, seed=a.seed)
    crp = CheckedRecall(w, w.stream, triple_bits=a.triple_bits, prefix_bits=8, seed=a.seed)
    # measured false-positive rates of the filters, on random keys that were never written
    rk = np.random.default_rng(a.seed + 999).integers(2**40, 2**62, 200000).astype(np.uint64)
    eps_t = float(cr.tri.contains(rk).mean())
    eps_k = float(cr.key.contains(rk).mean())
    out["filters"] = {"eps_triple": eps_t, "eps_key": eps_k, "eps_entity": float(cr.ent.contains(rk).mean())}
    log(f"measured FPR: triple {eps_t:.5f} key {eps_k:.5f}")

    st = CMLM.wake(w, w.stream, k=10**9, seed=a.seed)
    sp, _ = st.answer(qn, qr)
    gsp, _ = st.answer(gn, gr)
    sb = st.bits()
    sb["store"] = st.store.n_bits(key_bits(w), int(np.ceil(np.log2(V))))
    out["store_only"] = summarize(sp, gsp, sb, {"store_entries": len(st.store)})
    log(f"store only: acc {out['store_only']['accuracy']:.3f} hal {out['store_only']['hallucination']:.4f}")

    # the cheapest exact value stores (cogllm/baselines.py): what CAR's bits must beat
    true_vals = w.values[w.stream[:, 0], w.stream[:, 1]]
    for tag, st_ in [("exact_dict", ExactDict(w, w.stream, true_vals)),
                     ("key_filter_static_function", KeyFilterStatic(w, w.stream, true_vals, seed=a.seed))]:
        out[tag] = summarize(st_.answer(qn, qr), st_.answer(gn, gr), st_.bits())
        log(f"{tag}: acc {out[tag]['accuracy']:.3f} hal {out[tag]['hallucination']:.4f} "
            f"bits/fact {out[tag]['non_weight_bits_per_seen_fact']:.1f}")

    # controls without any weights: candidates in random order (prefix-pruned or full scan)
    rnd = np.random.default_rng(a.seed + 5).random((len(q), V)).astype(np.float32)
    grnd = np.random.default_rng(a.seed + 6).random((len(gn), V)).astype(np.float32)
    out["no_weights"] = {}
    for tag, c, N in [("prefix_random_order_N50", crp, 50), ("prefix_random_order_N200", crp, 200),
                      ("random_order_N20", cr, 20), ("full_scan_N1024", cr, V)]:
        pred = c.answer(None, qn, qr, N=N, probs=rnd)
        gpred = c.answer(None, gn, gr, N=N, probs=grnd)
        out["no_weights"][tag] = summarize(pred, gpred, dict(c.bits()))
        log(f"no weights {tag}: acc {out['no_weights'][tag]['accuracy']:.3f} hal {out['no_weights'][tag]['hallucination']:.4f}")
    for b in [20, 28]:
        cfull = CheckedRecall(w, w.stream, triple_bits=b, seed=a.seed)
        pred = cfull.answer(None, qn, qr, N=V, probs=rnd)
        gpred = cfull.answer(None, gn, gr, N=V, probs=grnd)
        out["no_weights"][f"full_scan_N1024_b{b}"] = summarize(pred, gpred, dict(cfull.bits()))
        log(f"no weights full scan b={b}: acc {out['no_weights'][f'full_scan_N1024_b{b}']['accuracy']:.3f} "
            f"hal {out['no_weights'][f'full_scan_N1024_b{b}']['hallucination']:.4f}")

    bk = np.array(count_buckets(counts))
    for d in [int(x) for x in a.sizes.split(",")]:
        res = {}
        m = gen(f"ce_d{d}", d, s_all, m_all)
        mi = gen(f"idk_d{d}", d, torch.cat([mt_seqs(w, w.stream)[0], idk_seqs]),
                 torch.cat([mt_seqs(w, w.stream)[1], idk_mask]))
        res["params"] = m.n_params()
        p, gp = answer_probs_mt(m, w, qn, qr), answer_probs_mt(m, w, gn, gr)
        pi, gpi = answer_probs_mt(mi, w, qn, qr), answer_probs_mt(mi, w, gn, gr)
        res["weights"] = weights_sweep(p, gp, False)
        res["weights_idk"] = weights_sweep(pi, gpi, True)
        vals = p[:, 1:]
        rank = (vals > vals[np.arange(len(vals)), truth][:, None]).sum(1) + 1  # 1 = top
        res["rank_in_list"] = {f"top{N}": {b: float((rank[bk == b] <= N).mean()) for b in BUCKETS if b != "0"}
                               for N in [1, 5, 20, 50]}
        for N in [1, 5, 20]:
            pred = cr.answer(None, qn, qr, N=N, probs=vals)
            gpred = cr.answer(None, gn, gr, N=N, probs=gp[:, 1:])
            r = summarize(pred, gpred, dict(cr.bits(), weights=res["params"] * 16))
            # list-decoding prediction (docs/theory.md, section 5)
            seen = counts > 0
            pred_hal = (~seen).mean() * eps_k * (1 - (1 - eps_t) ** N) \
                + seen.mean() * np.mean(np.where(rank[seen] <= N, 1 - (1 - eps_t) ** (rank[seen] - 1), 1 - (1 - eps_t) ** N))
            pred_acc = seen.mean() * np.mean((rank[seen] <= N) * (1 - eps_t) ** (rank[seen] - 1))
            r["predicted"] = {"hallucination": float(pred_hal), "accuracy": float(pred_acc)}
            res[f"CAR_N{N}"] = r
        # ablation: entity + key filters with the weights' top answer, no triple checksum
        res["key_filter_greedy"] = summarize(cr.answer(None, qn, qr, N=1, probs=vals, verify=False),
                                             cr.answer(None, gn, gr, N=1, probs=gp[:, 1:], verify=False),
                                             {k: v for k, v in dict(cr.bits(), weights=res["params"] * 16).items()
                                              if k != "triple_checksum"})
        res["checksum_sizing"] = size_checksum(w, q, qn, qr, truth, counts, vals, rank, eps_k, a)
        for N in [20, 50]:
            pred = crp.answer(None, qn, qr, N=N, probs=vals)
            gpred = crp.answer(None, gn, gr, N=N, probs=gp[:, 1:])
            res[f"CAR_prefix_N{N}"] = summarize(pred, gpred, dict(crp.bits(), weights=res["params"] * 16))
        care = CARE(w, cr, w.stream, w.values[w.stream[:, 0], w.stream[:, 1]], 20)
        kept = care.sleep(lambda n, r_: answer_probs_mt(m, w, n, r_))
        pred, src = care.answer(qn, qr, vals)
        gpred, _ = care.answer(gn, gr, gp[:, 1:])
        res["CARE_N20"] = summarize(pred, gpred, dict(care.bits(), weights=res["params"] * 16),
                                    {"store_entries": kept, "store_share_of_seen_facts": kept / seen_facts,
                                     "answered_from_store": float((src == "store").mean())})
        pred, _ = care.answer(qn, qr, vals, strict=True)
        gpred, _ = care.answer(gn, gr, gp[:, 1:], strict=True)
        res["CARE_N20_strict"] = summarize(pred, gpred, dict(care.bits(), weights=res["params"] * 16))
        if d == a.noise_size:  # ablation: weights trained only on facts seen >= 2 times
            keep2 = w.counts[w.stream[:, 0], w.stream[:, 1]] >= 2
            mk = gen(f"cek2_d{d}", d, *mt_seqs(w, w.stream[keep2]))
            pk, gpk = answer_probs_mt(mk, w, qn, qr)[:, 1:], answer_probs_mt(mk, w, gn, gr)[:, 1:]
            res["k2_weights_greedy"] = score(pk.argmax(1), truth, counts)["accuracy"]
            for N in [1, 5]:
                res[f"k2_CAR_N{N}"] = summarize(cr.answer(None, qn, qr, N=N, probs=pk), cr.answer(None, gn, gr, N=N, probs=gpk))
            res["k2_CAR_prefix_N50"] = summarize(crp.answer(None, qn, qr, N=50, probs=pk), crp.answer(None, gn, gr, N=50, probs=gpk))
        out["sizes"][str(d)] = res
        wo = res["weights"]
        log(f"d={d} ({res['params']} params) weights: greedy acc {wo['default']['accuracy']:.3f} hal {wo['default']['hallucination']:.3f}"
            f" | acc@hal<=1% {wo['acc_at_hal_1pct']:.3f} | IDK acc@hal<=1% {res['weights_idk']['acc_at_hal_1pct']:.3f}")
        for k in ["CAR_N1", "CAR_N5", "CAR_N20", "CAR_prefix_N20", "CAR_prefix_N50", "CARE_N20", "CARE_N20_strict"]:
            r = res[k]
            s = f"   {k:<16} acc {r['accuracy']:.3f} hal {r['hallucination']:.4f} tot {r['levels']['tip_of_tongue']:.3f}"
            if "predicted" in r:
                s += f"  (predicted acc {r['predicted']['accuracy']:.3f} hal {r['predicted']['hallucination']:.4f})"
            if "store_entries" in r:
                s += f"  store {r['store_entries']} ({100 * r['store_share_of_seen_facts']:.0f}% of seen facts)"
            log(s)

    # noisy extraction
    for mis in [0.05, 0.10]:
        r2 = np.random.default_rng(a.seed + 55)
        ment = w.stream.copy()
        vals_m = w.values[ment[:, 0], ment[:, 1]].copy()
        bad = r2.random(len(ment)) < mis
        ment[bad, 0] = r2.integers(0, w.cfg.n_entities, bad.sum())
        sn, mn = mt_seqs(w, ment, vals_m)
        gm = gen(f"ce_d{a.noise_size}_mis{int(mis * 100)}", a.noise_size, sn, mn)
        pn = answer_probs_mt(gm, w, qn, qr)[:, 1:]
        row = {"mislink": mis, "weights_greedy": score(pn.argmax(1), truth, counts)["hallucination"]}
        crn = CheckedRecall(w, ment, triple_bits=a.triple_bits, counting=True, seed=a.seed, values=vals_m)
        for N in [1, 5, 20]:
            pr = crn.answer(None, qn, qr, N=N, probs=pn)
            rr = score(np.where(pr >= 0, pr, -1), truth, counts)
            row[f"CAR_N{N}"] = {"accuracy": rr["accuracy"], "hallucination": rr["hallucination"]}
        care = CARE(w, crn, ment, vals_m, 20)
        care.sleep(lambda n, r_: answer_probs_mt(gm, w, n, r_))
        for strict in [False, True]:
            pr, _ = care.answer(qn, qr, pn, strict=strict)
            rr = score(np.where(pr >= 0, pr, -1), truth, counts)
            row["CARE_N20" + ("_strict" if strict else "")] = {"accuracy": rr["accuracy"], "hallucination": rr["hallucination"]}
        sto = CMLM.wake(w, ment, k=10**9, seed=a.seed)
        sto.store = EpisodicStore()
        for (e, r_), v in zip(ment, vals_m):
            sto.store.write((int(e), int(r_)), int(v))
        sp, _ = sto.answer(qn, qr)
        rr = score(np.where(sp >= 0, sp, -1), truth, counts)
        row["store_only"] = {"accuracy": rr["accuracy"], "hallucination": rr["hallucination"]}
        rr = score(ExactDict(w, ment, vals_m, min_count=2).answer(qn, qr), truth, counts)
        row["store_no_singletons"] = {"accuracy": rr["accuracy"], "hallucination": rr["hallucination"]}
        out["noise"].append(row)
        log(f"mislink {mis}: " + json.dumps(row))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
