"""Pilot on the hard world (1024 multi-token values): CAR and CARE variations.

    python -m experiments.pilot_mt --out results/pilot_mt.json

CARE = Checksum-Aided Recall with Episodic overflow:
    familiarity filters  -> graded "I don't know"
    weights              -> propose a top-N list (list decoder)
    triple checksum      -> recognise the right candidate (Bloom, ~b bits/fact)
    episodic overflow    -> exact store ONLY for facts the weights + checksum
                            could not bring back at sleep time
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from cogllm.checksum import TIP_OF_TONGUE, UNKNOWN_ENTITY, UNKNOWN_FACT, CheckedRecall

READ_ONCE = -4  # "I read that once, but can't confirm it"
from cogllm.evaluate import BUCKETS, score
from cogllm.listloss import train_lm_gated, train_lm_list_mt
from cogllm.memory import EpisodicStore
from cogllm.model import GPT, train_lm
from cogllm.system import CMLM, key_bits
from cogllm.world import count_buckets, sample_queries
from cogllm.world_mt import MTConfig, answer_probs_mt, build_mt_world, mt_seqs


class CARE:
    def __init__(self, world, cr: CheckedRecall, mentions, values, N):
        self.w, self.cr, self.N = world, cr, N
        self.store = EpisodicStore()
        for (e, r), v in zip(mentions, values):
            self.store.write((int(e), int(r)), int(v))
        self.idx_of = {int(nm[0]) * world.cfg.n_last + int(nm[1]): i for i, nm in enumerate(world.names)}

    def sleep(self, probs_fn):
        """Evict every stored fact that weights + checksum already recall correctly."""
        keys = list(self.store.vals.keys())
        e = np.array([k[0] for k in keys])
        r = np.array([k[1] for k in keys])
        p = probs_fn(self.w.names[e], r)[:, 1:]
        pred = self.cr.answer(None, self.w.names[e], r, N=self.N, probs=p)
        kept = 0
        for key, pr in zip(keys, pred):
            if pr >= 0 and pr == self.store.recall(key):
                self.store.evict(key)
            else:
                kept += 1
        return kept

    def answer(self, names, rels, probs, strict=False):
        """strict: a fact read only once and never recalled by the weights is
        reported as 'read once, unconfirmed' (READ_ONCE, an abstention)
        instead of being asserted. Singletons are where both recall and
        extraction noise are least reliable (Good-Turing again)."""
        pred = self.cr.answer(None, names, rels, N=self.N, probs=probs)
        src = np.array(["car"] * len(names), dtype=object)
        for i, (nm, r) in enumerate(zip(names, rels)):
            e = self.idx_of.get(int(nm[0]) * self.w.cfg.n_last + int(nm[1]))
            if e is not None and (e, int(r)) in self.store:
                key = (e, int(r))
                pred[i] = READ_ONCE if strict and self.store.count(key) < 2 else self.store.recall(key)
                src[i] = "store"
        return pred, src

    def bits(self):
        b = self.cr.bits()
        b["overflow_store"] = self.store.n_bits(key_bits(self.w), int(np.ceil(np.log2(self.w.cfg.n_values))))
        return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="results/pilot_mt.json")
    ap.add_argument("--ckpt", default="results/ckpt")
    ap.add_argument("--gens", default="ce_d32,ce_d64,cek2_d32,gate_d32,list_d32")
    ap.add_argument("--n_entities", type=int, default=10000)
    ap.add_argument("--n_mentions", type=int, default=60000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("THREADS", "4")))
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    w = build_mt_world(MTConfig(n_first=160, n_last=160, n_entities=a.n_entities, zipf_s=1.0, n_mentions=a.n_mentions, seed=a.seed))
    V = w.cfg.n_values
    q = sample_queries(w, 20000, seed=a.seed + 7)
    qn, qr = w.names[q[:, 0]], q[:, 1]
    truth, counts = w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]]
    rng = np.random.default_rng(a.seed + 3)
    gn = w.ghost_names[rng.integers(len(w.ghost_names) // 2, len(w.ghost_names), 4000)]
    gr = rng.integers(0, w.cfg.n_relations, 4000)
    os.makedirs(a.ckpt, exist_ok=True)
    out = {"generators": {}, "car": {}, "care": {}, "noise": []}
    s_all, m_all = mt_seqs(w, w.stream)

    def gen(tag, fit, d):
        path = f"{a.ckpt}/mt_s{a.seed}_{tag}.pt"
        torch.manual_seed(a.seed)
        m = GPT(w.vocab_size, d=d, n_layer=2)
        if os.path.exists(path):
            m.load_state_dict(torch.load(path))
        else:
            log(f"train {tag}")
            fit(m)
            torch.save(m.state_dict(), path)
        m.eval()
        return m

    keep2 = w.counts[w.stream[:, 0], w.stream[:, 1]] >= 2
    s_k2, m_k2 = mt_seqs(w, w.stream[keep2])
    fits = {
        "ce": lambda m: train_lm(m, s_all, m_all, a.epochs, seed=a.seed),
        "cek2": lambda m: train_lm(m, s_k2, m_k2, a.epochs, seed=a.seed),
        "gate": lambda m: train_lm_gated(m, s_all, w, slots=[3, 4], Ns=(4, 2), epochs=a.epochs, seed=a.seed),
        "list": lambda m: train_lm_list_mt(m, s_all, w, Ns=(3, 2), epochs=a.epochs, seed=a.seed),
    }
    gens = {}
    for g in a.gens.split(","):
        kind, d = g.split("_d")
        gens[g] = gen(g, fits[kind], int(d))
    probs = {k: answer_probs_mt(m, w, qn, qr)[:, 1:] for k, m in gens.items()}
    gprobs = {k: answer_probs_mt(m, w, gn, gr)[:, 1:] for k, m in gens.items()}
    bk = np.array(count_buckets(counts))

    for k, p in probs.items():
        rank = (p > p[np.arange(len(p)), truth][:, None]).sum(1)
        row = {"params": gens[k].n_params()}
        for N in [1, 5, 10, 20, 50]:
            row[f"top{N}"] = {b: float((rank[bk == b] < N).mean()) for b in BUCKETS if b != "0"}
            row[f"top{N}"]["all_seen"] = float((rank[counts > 0] < N).mean())
        conf, best = p.max(1), p.argmax(1)
        pts = []
        for t in [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]:
            r = score(np.where(conf >= t, best, -1), truth, counts)
            pts.append((r["accuracy"], r["hallucination"]))
        row["weights_greedy"] = {"accuracy": pts[0][0], "hallucination": pts[0][1]}
        row["weights_only_acc_at_hal_1pct"] = max([acc for acc, h in pts if h <= 0.01], default=0.0)
        out["generators"][k] = row
        log(f"{k}: params {row['params']} greedy acc {pts[0][0]:.3f} hal {pts[0][1]:.3f} | "
            f"acc@hal<=1% {row['weights_only_acc_at_hal_1pct']:.3f} | seen top1 {row['top1']['all_seen']:.3f} "
            f"top5 {row['top5']['all_seen']:.3f} top20 {row['top20']['all_seen']:.3f} top50 {row['top50']['all_seen']:.3f}")

    def record(tag, pred, gpred, bits, extra=None):
        r = score(np.where(pred >= 0, pred, -1), truth, counts, np.where(gpred >= 0, gpred, -1))
        r["bits"], r["total_bits"] = bits, int(sum(bits.values()))
        r["marginal_bits"] = int(sum(v for k, v in bits.items() if k != "weights"))
        r["levels"] = {lab: float((pred == c).mean()) for lab, c in
                       [("unknown_entity", UNKNOWN_ENTITY), ("unknown_fact", UNKNOWN_FACT), ("tip_of_tongue", TIP_OF_TONGUE)]}
        r.update(extra or {})
        return r

    def show(tag, r):
        log(f"{tag:<30} acc {r['accuracy']:.3f} hal {r['hallucination']:.4f} tot {r['levels']['tip_of_tongue']:.3f} "
            f"ghost {r['ghost_hallucination']:.4f} bits/seen-fact (non-weight) {r['marginal_bits'] / (w.counts > 0).sum():.1f}"
            + (f" store {r['store_entries']}" if "store_entries" in r else ""))

    seen_facts = int((w.counts > 0).sum())
    crs = {b: CheckedRecall(w, w.stream, triple_bits=b, seed=a.seed) for b in [8, 10, 14, 20, 28]}
    for gname in gens:
        for b in [10, 14, 20]:
            for N in [1, 5, 20, 50]:
                cr = crs[b]
                pred = cr.answer(None, qn, qr, N=N, probs=probs[gname])
                gpred = cr.answer(None, gn, gr, N=N, probs=gprobs[gname])
                bits = dict(cr.bits(), weights=gens[gname].n_params() * 16)
                tag = f"CAR {gname} N={N} b={b}"
                out["car"][tag] = record(tag, pred, gpred, bits)
                show(tag, out["car"][tag])
    crp = CheckedRecall(w, w.stream, triple_bits=14, prefix_bits=8, seed=a.seed)
    for gname in gens:
        for N in [5, 20, 50, 200]:
            pred = crp.answer(None, qn, qr, N=N, probs=probs[gname])
            gpred = crp.answer(None, gn, gr, N=N, probs=gprobs[gname])
            tag = f"CAR+prefix {gname} N={N} b=14+8"
            out["car"][tag] = record(tag, pred, gpred, dict(crp.bits(), weights=gens[gname].n_params() * 16))
            show(tag, out["car"][tag])
    for b in [14, 20, 28]:
        g0 = list(gens)[0]
        pred = crs[b].answer(None, qn, qr, N=V, probs=probs[g0])
        gpred = crs[b].answer(None, gn, gr, N=V, probs=gprobs[g0])
        tag = f"full scan N={V} b={b}"
        out["car"][tag] = record(tag, pred, gpred, dict(crs[b].bits()))
        show(tag, out["car"][tag])

    st = CMLM.wake(w, w.stream, k=10**9, seed=a.seed)
    sp, _ = st.answer(qn, qr)
    gsp, _ = st.answer(gn, gr)
    sb = st.bits()
    sb["store"] = st.store.n_bits(key_bits(w), int(np.ceil(np.log2(V))))
    out["store_only"] = record("store", sp, gsp, sb, {"store_entries": len(st.store)})
    show("store only", out["store_only"])

    for gname in gens:
        for b, N in [(14, 20), (20, 50), (14, 5)]:
            care = CARE(w, crs[b], w.stream, w.values[w.stream[:, 0], w.stream[:, 1]], N)
            kept = care.sleep(lambda n, r: answer_probs_mt(gens[gname], w, n, r))
            pred, src = care.answer(qn, qr, probs[gname])
            gpred, _ = care.answer(gn, gr, gprobs[gname])
            bits = dict(care.bits(), weights=gens[gname].n_params() * 16)
            tag = f"CARE {gname} N={N} b={b}"
            out["care"][tag] = record(tag, pred, gpred, bits, {"store_entries": kept, "store_share_of_facts": kept / seen_facts,
                                                                "answered_from_store": float((src == 'store').mean())})
            show(tag, out["care"][tag])

    # noisy extraction: 5% / 10% of mentions filed under the wrong person
    g0 = list(gens)[0]
    for mis in [0.05, 0.10]:
        rng2 = np.random.default_rng(a.seed + 55)
        ment = w.stream.copy()
        vals = w.values[ment[:, 0], ment[:, 1]].copy()
        bad = rng2.random(len(ment)) < mis
        ment[bad, 0] = rng2.integers(0, w.cfg.n_entities, bad.sum())
        sn, mn = mt_seqs(w, ment, vals)
        gm = gen(f"ce_d32_mis{int(mis * 100)}", lambda m: train_lm(m, sn, mn, a.epochs, seed=a.seed), 32)
        pn = answer_probs_mt(gm, w, qn, qr)[:, 1:]
        row = {"mislink": mis, "weights_greedy_hal": score(pn.argmax(1), truth, counts)["hallucination"]}
        for counting in [False, True]:
            cr = CheckedRecall(w, ment, triple_bits=14, counting=counting, seed=a.seed, values=vals)
            for N in [5, 20]:
                pr = cr.answer(None, qn, qr, N=N, probs=pn)
                r = score(np.where(pr >= 0, pr, -1), truth, counts)
                row[f"car_{'count' if counting else 'bloom'}_N{N}"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
            care = CARE(w, cr, ment, vals, 20)
            care.sleep(lambda n, r: answer_probs_mt(gm, w, n, r))
            pr, _ = care.answer(qn, qr, pn)
            r = score(np.where(pr >= 0, pr, -1), truth, counts)
            row[f"care_{'count' if counting else 'bloom'}_N20"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
        sto = EpisodicStore()
        for (e, r_), v in zip(ment, vals):
            sto.write((int(e), int(r_)), int(v))
        cm = CMLM.wake(w, ment, k=10**9, seed=a.seed)
        cm.store = sto
        sp, _ = cm.answer(qn, qr)
        r = score(np.where(sp >= 0, sp, -1), truth, counts)
        row["store_only"] = {"accuracy": r["accuracy"], "hallucination": r["hallucination"]}
        out["noise"].append(row)
        log(f"mislink {mis}: " + json.dumps(row))

    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
