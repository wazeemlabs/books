"""Checksum-aided recall on a real model: OLMo-2 1B on PopQA.

    python -m experiments.real_popqa --n 300 --ghosts 100 --model data/models/OLMo-2-0425-1B

The checksum is built from the model's own pretraining corpus (OLMo-mix-1124,
through infini-gram), so "read" means read by this model:

    entity filter      the subject's name occurs in the corpus
    pair checksum      subject and candidate occur within 100 tokens of each other
                       (no fact extraction at all; relation-blind, like QuCo-RAG)
    triple checksum    the candidate matches the Wikidata answer: perfect
    (oracle)           extraction, the upper bound a real extractor approaches

Bloom false positives are emulated (realworld/bench.py): a candidate that is
not in the checksum still passes with probability eps = 0.6185^bits.

Systems, all from the same generations:
    greedy                 always answers
    greedy + threshold     abstains below a sequence-probability threshold (sweep)
    entity filter + greedy abstains on subjects the corpus never names
    CAR pair, N            first of [greedy, beam 1..N-1] that passes the pair checksum
                           (N=1 is the QuCo-RAG check, abstaining instead of retrieving)
    CAR oracle triple, N   first candidate that passes the triple checksum
Ghost questions (people who do not exist) measure answers to the unanswerable.
"""

import argparse
import json
import os
import time

import numpy as np

from realworld import popqa
from realworld.bench import N_LIST, Bench, bloom_eps, passes_fp
from realworld.corpus import pair_query

N_GOLD_ALIASES = 3  # aliases checked per gold answer, to bound corpus queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--ghosts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beams", type=int, default=10)
    ap.add_argument("--bits", type=float, default=14.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="allenai/OLMo-2-0425-1B", help="hub id or local directory")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results/real_popqa.json")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    os.makedirs(a.cache, exist_ok=True)
    B = Bench(a.n, a.ghosts, seed=a.seed, beams=a.beams, cache=a.cache, model=a.model, batch=a.batch, log=log)

    subj = B.subject
    pair_q = sorted({pair_query(subj[i], c) for i in subj if B.ent[i] > 0 for c in B.cand[i]})
    gold_q = sorted({pair_query(q.subject, al) for q in B.qs if B.ent[q.id] > 0 for al in q.answers[:N_GOLD_ALIASES]})
    log(f"corpus queries: {len(pair_q)} candidate pairs, {len(gold_q)} gold pairs")
    pc = dict(zip(pair_q, B.corpus.counts(pair_q)))
    gc = dict(zip(gold_q, B.corpus.counts(gold_q)))
    log("corpus counts done")

    eps = bloom_eps(a.bits)
    read = {q.id: B.ent[q.id] > 0 and any(gc.get(pair_query(q.subject, al), 0) > 0 for al in q.answers[:N_GOLD_ALIASES])
            for q in B.qs}

    def pair_ok(i, c):
        return pc.get(pair_query(subj[i], c), 0) > 0 or passes_fp(f"pair|{i}|{c}", eps, a.seed)

    def triple_ok(i, c):
        if i in B.gold:
            return popqa.is_correct(c, B.gold[i]) or passes_fp(f"triple|{i}|{c}", eps, a.seed)
        return passes_fp(f"triple|{i}|{c}", eps, a.seed)  # ghosts have no facts

    systems = {"greedy": lambda i: B.G[i]["greedy"],
               "entity_filter_greedy": lambda i: B.G[i]["greedy"] if B.ent[i] > 0 else None}
    for N in N_LIST:
        systems[f"CAR_pair_N{N}"] = B.car(pair_ok, N)
        systems[f"CAR_triple_oracle_N{N}"] = B.car(triple_ok, N)

    out = {"config": vars(a), "eps": eps, "exemplars": B.exemplars, "popularity_edges_log10": B.edges.tolist(),
           "read_share": float(np.mean(list(read.values()))),
           "entity_known_share": float(np.mean([B.ent[q.id] > 0 for q in B.qs])),
           "ghost_entity_known_share": float(np.mean([B.ent[g["id"]] > 0 for g in B.ghosts])), "systems": {}}
    answers = {}
    for name, fn in systems.items():
        out["systems"][name], answers[name] = B.evaluate(fn, read)
        r = out["systems"][name]
        log(f"{name:<24} acc {r['accuracy']:.3f} hal {r['hallucination']:.3f} abstain {r['abstain']:.3f} "
            f"ghosts answered {r['ghost_answered']:.3f}")
    out["greedy_threshold_frontier"] = B.threshold_frontier()
    # how often a wrong candidate still co-occurs with its subject: the pair checksum's real error rate
    out["pair_checksum"] = B.pass_rates(lambda i, c: pc.get(pair_query(subj[i], c), 0) > 0)
    out["top_n_recall"] = B.top_n_recall()
    log(f"pair checksum: {json.dumps(out['pair_checksum'])}; top-N recall {out['top_n_recall']}")
    out["answers"] = answers
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
