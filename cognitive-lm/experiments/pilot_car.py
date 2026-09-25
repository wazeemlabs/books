"""Pilot: checksum-aided recall (CAR) and its variations, on seed 0.

    python -m experiments.pilot_car --out results/pilot_car.json

1. How often is the right answer in the weights' top N?        (list-decoding view)
2. CAR vs weights-only vs store-only: accuracy / hallucination / bits
3. Sweep list size N and checksum bits
4. Key filter on/off (two-stage gating)
5. Generators: plain CE, CE on facts seen >= 2x, list-decoding loss
6. Noisy fact extraction (entity mislinks): CAR (Bloom / counting) vs store
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.checksum import TIP_OF_TONGUE, UNKNOWN_ENTITY, UNKNOWN_FACT, CheckedRecall
from cogllm.evaluate import BUCKETS, answer_probs, score
from cogllm.listloss import train_lm_list
from cogllm.model import GPT, train_lm
from cogllm.system import CMLM, stream_tensors
from cogllm.world import EOS, WorldConfig, build_world, count_buckets, sample_queries


def noisy_stream(w, mislink, seed):
    """Mislinked mentions keep their value but are filed under a random other person."""
    rng = np.random.default_rng(seed + 55)
    ment = w.stream.copy()
    vals = w.values[ment[:, 0], ment[:, 1]].copy()
    bad = rng.random(len(ment)) < mislink
    ment[bad, 0] = rng.integers(0, w.cfg.n_entities, bad.sum())
    return ment, vals


def seqs_from(w, ment, vals):
    s = torch.tensor([w.prompt(w.names[e], r) + [w.val_tok(r, v), EOS] for (e, r), v in zip(ment, vals)])
    return s, torch.ones(len(s), s.shape[1] - 1, dtype=torch.bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/pilot_car.json")
    ap.add_argument("--n_entities", type=int, default=10000)
    ap.add_argument("--n_mentions", type=int, default=60000)
    ap.add_argument("--ckpt", default="results/ckpt")
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    w = build_world(WorldConfig(n_first=160, n_last=160, n_entities=a.n_entities, zipf_s=1.0, n_mentions=a.n_mentions, seed=a.seed))
    q = sample_queries(w, 20000, seed=a.seed + 7)
    qn, qr = w.names[q[:, 0]], q[:, 1]
    truth, counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]
    rng = np.random.default_rng(a.seed + 3)
    gn = w.ghost_names[rng.integers(len(w.ghost_names) // 2, len(w.ghost_names), 4000)]
    gr = rng.integers(0, w.cfg.n_relations, 4000)
    os.makedirs(a.ckpt, exist_ok=True)
    out = {"generators": {}, "car": {}, "noise": []}

    def gen(tag, d, fit):
        path = f"{a.ckpt}/s{a.seed}_{tag}.pt"
        torch.manual_seed(a.seed)
        m = GPT(w.vocab_size, d=d, n_layer=2)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path))
            m.eval()
        else:
            log(f"train generator {tag}")
            fit(m)
            torch.save(m.state_dict(), path)
        return m

    s_all, m_all = stream_tensors(w, w.stream)
    keep2 = w.counts[w.stream[:, 0], w.stream[:, 1]] >= 2
    s_k2, m_k2 = stream_tensors(w, w.stream[keep2])
    gens = {
        "ce_d32": gen("ce_d32", 32, lambda m: train_lm(m, s_all, m_all, a.epochs, seed=a.seed)),
        "ce_d64": gen("ce_d64", 64, lambda m: train_lm(m, s_all, m_all, a.epochs, seed=a.seed)),
        "ce_k2_d32": gen("ce_k2_d32", 32, lambda m: train_lm(m, s_k2, m_k2, a.epochs, seed=a.seed)),
        "list5_d32": gen("list5_d32", 32, lambda m: train_lm_list(m, s_all, w, N=5, epochs=a.epochs, seed=a.seed)),
    }
    probs = {k: answer_probs(m, w, qn, qr)[:, 1:] for k, m in gens.items()}
    gprobs = {k: answer_probs(m, w, gn, gr)[:, 1:] for k, m in gens.items()}
    bk = np.array(count_buckets(counts))

    # ---- 1. top-N recall by exposure bucket --------------------------------------
    for k, p in probs.items():
        rank = (p > p[np.arange(len(p)), truth][:, None]).sum(1)  # 0 = top-1
        row = {"params": gens[k].n_params()}
        for N in [1, 2, 5, 10, 20]:
            row[f"top{N}"] = {b: float((rank[bk == b] < N).mean()) for b in BUCKETS if b != "0"}
            row[f"top{N}"]["all_seen"] = float((rank[counts > 0] < N).mean())
        # weights-only operating points (confidence threshold), for reference
        conf, best = p.max(1), p.argmax(1)
        pts = []
        for t in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
            r = score(np.where(conf >= t, best, -1), truth, counts)
            pts.append((t, r["accuracy"], r["hallucination"]))
        row["weights_only_acc_at_hal_1pct"] = max([acc for _, acc, h in pts if h <= 0.01], default=0.0)
        out["generators"][k] = row
        log(f"{k}: params {row['params']}  top1 {row['top1']['all_seen']:.3f} top5 {row['top5']['all_seen']:.3f} "
            f"top10 {row['top10']['all_seen']:.3f} | singletons top1 {row['top1']['1']:.3f} top5 {row['top5']['1']:.3f} "
            f"top20 {row['top20']['1']:.3f} | weights-only acc@hal<=1% {row['weights_only_acc_at_hal_1pct']:.3f}")

    # ---- 2-5. CAR sweeps -------------------------------------------------------------
    def run_car(tag, cr, gname, N):
        pred = cr.answer(None, qn, qr, N=N, probs=probs[gname])
        gpred = cr.answer(None, gn, gr, N=N, probs=gprobs[gname])
        r = score(np.where(pred >= 0, pred, -1), truth, counts, np.where(gpred >= 0, gpred, -1))
        bits = cr.bits()
        bits["weights"] = gens[gname].n_params() * 16
        r["bits"], r["total_bits"] = bits, int(sum(bits.values()))
        r["levels"] = {lab: float((pred == code).mean()) for lab, code in
                       [("unknown_entity", UNKNOWN_ENTITY), ("unknown_fact", UNKNOWN_FACT), ("tip_of_tongue", TIP_OF_TONGUE)]}
        r["ghost_unknown_entity"] = float((gpred == UNKNOWN_ENTITY).mean())
        r.pop("by_count", None)
        out["car"][tag] = r
        log(f"{tag:<34} acc {r['accuracy']:.3f} hal {r['hallucination']:.4f} abst {r['abstain']:.3f} "
            f"ghost-hal {r['ghost_hallucination']:.4f} tot-bits {r['total_bits'] / 1e6:.2f}M "
            f"(checksum {bits['triple_checksum'] / 1e3:.0f}k) tot {r['levels']['tip_of_tongue']:.3f}")

    cr10 = CheckedRecall(w, w.stream, triple_bits=10, seed=a.seed)
    for N in [1, 2, 3, 5, 10, 20, 48]:
        run_car(f"ce_d32 N={N} b=10", cr10, "ce_d32", N)
    for b in [4, 6, 8, 14, 20]:
        crb = CheckedRecall(w, w.stream, triple_bits=b, seed=a.seed)
        run_car(f"ce_d32 N=5 b={b}", crb, "ce_d32", 5)
        run_car(f"ce_d32 N=48 b={b} (full scan)", crb, "ce_d32", 48)
    for b in [6, 10]:
        crn = CheckedRecall(w, w.stream, triple_bits=b, use_key_filter=False, seed=a.seed)
        run_car(f"ce_d32 N=5 b={b} no-key-filter", crn, "ce_d32", 5)
        run_car(f"ce_d32 N=48 b={b} no-key-filter", crn, "ce_d32", 48)
    for gname in gens:
        for N in [1, 5, 10]:
            run_car(f"{gname} N={N} b=10", cr10, gname, N)

    store = CMLM.wake(w, w.stream, k=10**9, seed=a.seed)
    sp, _ = store.answer(qn, qr)
    rs = score(np.where(sp >= 0, sp, -1), truth, counts)
    sb = store.bits()
    out["store_only"] = {"accuracy": rs["accuracy"], "hallucination": rs["hallucination"], "bits": sb,
                         "total_bits": int(sum(sb.values()))}
    log(f"store only: acc {rs['accuracy']:.3f} hal {rs['hallucination']:.4f} bits {sum(sb.values()) / 1e6:.2f}M")

    # ---- 6. noisy extraction ---------------------------------------------------------------
    for mis in [0.05, 0.10, 0.20]:
        ment, vals = noisy_stream(w, mis, a.seed)
        s_n, m_n = seqs_from(w, ment, vals)
        gm = gen(f"ce_d32_mis{int(mis * 100)}", 32, lambda m: train_lm(m, s_n, m_n, a.epochs, seed=a.seed))
        pn = answer_probs(gm, w, qn, qr)[:, 1:]
        row = {"mislink": mis}
        conf, best = pn.max(1), pn.argmax(1)
        row["weights_only_default"] = score(best, truth, counts)["hallucination"]
        for counting in [False, True]:
            cr = CheckedRecall(w, ment, triple_bits=16 if counting else 10, counting=counting, seed=a.seed, values=vals)
            for N in [1, 5, 48]:
                pred = cr.answer(None, qn, qr, N=N, probs=pn)
                r = score(np.where(pred >= 0, pred, -1), truth, counts)
                row[f"car_{'count' if counting else 'bloom'}_N{N}"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
        st = CMLM.wake(w, ment, k=10**9, seed=a.seed)
        st.store = type(st.store)()
        for (e, r_), v in zip(ment, vals):
            st.store.write((int(e), int(r_)), int(v))
        sp, _ = st.answer(qn, qr)
        r = score(np.where(sp >= 0, sp, -1), truth, counts)
        row["store_only"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
        out["noise"].append(row)
        log(f"mislink {mis}: " + json.dumps({k: v for k, v in row.items() if k != 'mislink'}))

    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
