"""Checksum-aided recall on a real model: OLMo-2 1B on PopQA.

    python -m experiments.real_popqa --n 1000 --ghosts 300

The checksum is built from the model's own pretraining corpus (OLMo-mix-1124,
through infini-gram), so "read" means read by this model:

    entity filter      the subject's name occurs in the corpus
    pair checksum      subject and candidate occur within 100 tokens of each other
                       (no fact extraction at all; relation-blind, like QuCo-RAG)
    triple checksum    the candidate matches the Wikidata answer: perfect
    (oracle)           extraction, the upper bound a real extractor approaches

Bloom false positives are emulated: a candidate that is not in the checksum
still passes with probability eps (a keyed hash, so the result is fixed),
eps = 0.6185^bits for an optimal Bloom filter.

Systems, all from the same generations:
    greedy                 always answers
    greedy + threshold     abstains below a sequence-probability threshold (sweep)
    entity filter + greedy abstains on subjects the corpus never names
    QuCo-style check       greedy answer kept only if subject and answer co-occur
    CAR pair, N            first of [greedy, beam 1..N-1] that passes the pair checksum
    CAR oracle triple, N   first candidate that passes the triple checksum
Ghost questions (people who do not exist) measure answers to the unanswerable.
"""

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np

from realworld import ghosts as ghost_mod
from realworld import popqa
from realworld.corpus import Corpus, pair_query

N_LIST = (1, 3, 5, 10)
N_GOLD_ALIASES = 3  # aliases checked per gold answer, to bound corpus queries
EXEMPLAR_RELATIONS = ("occupation", "place of birth", "genre", "country", "director", "capital of")


def passes_fp(key, eps, salt):
    """Deterministic Bloom false positive: True with probability eps."""
    h = int.from_bytes(hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=8).digest(), "big")
    return h / 2 ** 64 < eps


def candidates(gen, n):
    out, seen = [], set()
    for text in [gen["greedy"]] + [b[0] for b in gen["beams"]]:
        norm = popqa.normalize(text)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(text)
        if len(out) == n:
            break
    return out


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
    corpus = Corpus(f"{a.cache}/infinigram.jsonl")

    allq = popqa.load()
    qs, edges = popqa.sample(allq, a.n, seed=a.seed)
    chosen = {q.id for q in qs}
    exemplars = []
    for rel in EXEMPLAR_RELATIONS:  # one worked example per relation, never a test question
        q = next(x for x in allq if x.relation == rel and x.id not in chosen and x.s_pop > 10_000)
        exemplars.append((q.question, q.answers[0]))
    prefix = "".join(f"Q: {q}\nA: {ans}\n\n" for q, ans in exemplars)
    ghosts = ghost_mod.make(allq, corpus, a.ghosts, seed=a.seed)
    log(f"{len(qs)} questions, {len(ghosts)} ghosts")

    from realworld.generate import Generator, generate_cached  # heavy import only when generating
    gen = Generator(a.model)
    items = [(q.id, q.question) for q in qs] + [(g["id"], g["question"]) for g in ghosts]
    G = generate_cached(gen, items, f"{a.cache}/generations_seed{a.seed}_b{a.beams}.jsonl", prefix,
                        batch=a.batch, n_beams=a.beams)
    log("generation done")

    subj = {q.id: q.subject for q in qs} | {g["id"]: g["subject"] for g in ghosts}
    ent = dict(zip(subj, corpus.counts(list(subj.values()))))
    cand = {i: candidates(G[i], max(N_LIST)) for i in subj}
    pair_q = sorted({pair_query(subj[i], c) for i in subj if ent[i] > 0 for c in cand[i]})
    gold_q = sorted({pair_query(q.subject, al) for q in qs if ent[q.id] > 0 for al in q.answers[:N_GOLD_ALIASES]})
    log(f"corpus queries: {len(pair_q)} candidate pairs, {len(gold_q)} gold pairs")
    pc = dict(zip(pair_q, corpus.counts(pair_q)))
    gc = dict(zip(gold_q, corpus.counts(gold_q)))
    log("corpus counts done")

    eps = 0.5 ** (a.bits * math.log(2))  # optimal Bloom: (1/2)^(bits ln 2)
    gold = {q.id: q.answers for q in qs}
    pop_bin = {q.id: int(np.clip(np.searchsorted(edges, np.log10(q.s_pop + 1), side="right") - 1, 0, 4)) for q in qs}
    read = {q.id: ent[q.id] > 0 and any(gc.get(pair_query(q.subject, al), 0) > 0 for al in q.answers[:N_GOLD_ALIASES])
            for q in qs}

    def pair_ok(i, c):
        return ent[i] > 0 and (pc.get(pair_query(subj[i], c), 0) > 0 or passes_fp(f"pair|{i}|{c}", eps, a.seed))

    def triple_ok(i, c):
        if i in gold:
            return popqa.is_correct(c, gold[i]) or passes_fp(f"triple|{i}|{c}", eps, a.seed)
        return passes_fp(f"triple|{i}|{c}", eps, a.seed)  # ghosts have no facts

    systems = {"greedy": lambda i: G[i]["greedy"],
               "entity_filter_greedy": lambda i: G[i]["greedy"] if ent[i] > 0 else None}
    for N in N_LIST:
        systems[f"CAR_pair_N{N}"] = lambda i, N=N: next((c for c in cand[i][:N] if pair_ok(i, c)), None)
        systems[f"CAR_triple_oracle_N{N}"] = lambda i, N=N: next(
            (c for c in cand[i][:N] if ent[i] > 0 and triple_ok(i, c)), None)

    def evaluate(answer):
        ans = {i: answer(i) for i in subj}
        real = [q.id for q in qs]
        correct = np.array([ans[i] is not None and popqa.is_correct(ans[i], gold[i]) for i in real])
        answered = np.array([ans[i] is not None for i in real])
        bins = np.array([pop_bin[i] for i in real])
        rd = np.array([read[i] for i in real])
        r = {"accuracy": float(correct.mean()), "hallucination": float((answered & ~correct).mean()),
             "abstain": float((~answered).mean()),
             "ghost_answered": float(np.mean([ans[g["id"]] is not None for g in ghosts])),
             "by_popularity": [{"accuracy": float(correct[bins == b].mean()),
                                "hallucination": float((answered & ~correct)[bins == b].mean())} for b in range(5)],
             "on_read": {"accuracy": float(correct[rd].mean()), "hallucination": float((answered & ~correct)[rd].mean())},
             "on_unread": {"accuracy": float(correct[~rd].mean()), "hallucination": float((answered & ~correct)[~rd].mean())}}
        return r, ans

    out = {"config": vars(a), "eps": eps, "exemplars": exemplars, "popularity_edges_log10": edges.tolist(),
           "read_share": float(np.mean(list(read.values()))), "entity_known_share": float(np.mean([ent[q.id] > 0 for q in qs])),
           "ghost_entity_known_share": float(np.mean([ent[g["id"]] > 0 for g in ghosts])), "systems": {}}
    answers = {}
    for name, fn in systems.items():
        out["systems"][name], answers[name] = evaluate(fn)
        r = out["systems"][name]
        log(f"{name:<24} acc {r['accuracy']:.3f} hal {r['hallucination']:.3f} abstain {r['abstain']:.3f} "
            f"ghosts answered {r['ghost_answered']:.3f}")

    # confidence-threshold frontier for the greedy answer
    lp = np.array([G[q.id]["greedy_logprob"] for q in qs])
    frontier = []
    for t in np.quantile(lp, np.linspace(0, 0.95, 20)):
        r, _ = evaluate(lambda i, t=t: G[i]["greedy"] if G[i]["greedy_logprob"] >= t else None)
        frontier.append({"threshold_logprob": float(t), **{k: r[k] for k in ("accuracy", "hallucination", "abstain", "ghost_answered")}})
    out["greedy_threshold_frontier"] = frontier

    # how often a wrong candidate still co-occurs with its subject: the pair checksum's real error rate
    wrong = [(i, c) for q in qs for i in [q.id] if ent[i] > 0 for c in cand[i] if not popqa.is_correct(c, gold[i])]
    right = [(i, c) for q in qs for i in [q.id] if ent[i] > 0 for c in cand[i] if popqa.is_correct(c, gold[i])]
    out["pair_checksum"] = {"wrong_candidates": len(wrong),
                            "wrong_pass_rate": float(np.mean([pc.get(pair_query(subj[i], c), 0) > 0 for i, c in wrong])),
                            "right_candidates": len(right),
                            "right_pass_rate": float(np.mean([pc.get(pair_query(subj[i], c), 0) > 0 for i, c in right]))}
    out["top_n_recall"] = {N: float(np.mean([any(popqa.is_correct(c, gold[q.id]) for c in cand[q.id][:N]) for q in qs]))
                           for N in N_LIST}
    log(f"pair checksum: {json.dumps(out['pair_checksum'])}; top-N recall {out['top_n_recall']}")
    out["answers"] = {name: {i: v for i, v in ans.items()} for name, ans in answers.items()}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
