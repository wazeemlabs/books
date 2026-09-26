"""Checksums built by reading the corpus: OLMo-2 1B on PopQA, step 2.

    python -m experiments.real_extract --n 300 --ghosts 100

For each question, up to k passages of OLMo-mix-1124 that mention the
subject (realworld/passages.py). Three checksums are built from them:

    passage, any       the candidate occurs anywhere in a passage (~200 tokens)
    passage, near      the candidate occurs within `near` words of the subject
    extracted          an instruct model reads each passage and answers the
                       question from it alone, never seeing the candidates;
                       a candidate passes if it matches an extracted value

Each is scored with CAR at N = 1, 3, 5, 10 on the same generations as
experiments/real_popqa.py, next to greedy, the entity filter and the oracle.
Control: the extractor also reads passages about a different subject. Any
answer there that matches the gold answer came from its own memory, not the
passage.
"""

import argparse
import json
import os
import time

import numpy as np

from realworld import popqa
from realworld.bench import N_LIST, Bench, bloom_eps, passes_fp
from realworld.extract import MODEL as EXTRACTOR, NONE, Extractor, extract_cached, key
from realworld.passages import fetch, near


def matches(candidate, value):
    """Same fact if either names the other (whole-word containment after normalising)."""
    return popqa.is_correct(candidate, [value]) or popqa.is_correct(value, [candidate])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--ghosts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=10, help="passages per subject")
    ap.add_argument("--near", type=int, default=10, help="words between subject and value")
    ap.add_argument("--bits", type=float, default=14.0)
    ap.add_argument("--extractor", default=EXTRACTOR, help="hub id or local directory")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--out", default="results/real_extract_pilot.json")
    a = ap.parse_args()
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)
    B = Bench(a.n, a.ghosts, seed=a.seed, cache=a.cache, log=log)

    known = [q for q in B.qs if B.ent[q.id] > 0]
    P = fetch(B.corpus, sorted({q.subject for q in known}), k=a.k)
    passages = {q.id: P[q.subject]["passages"] for q in known}
    log(f"passages: {sum(map(len, passages.values()))} for {len(passages)} questions")

    # the control pairs each question with the passages of the next question's subject
    ctrl_of = {q.id: known[(j + 1) % len(known)].id for j, q in enumerate(known)}
    question = {q.id: q.question for q in B.qs}
    jobs = [(i, question[i], p) for i in passages for p in passages[i]]
    jobs += [(f"ctrl{i}", question[i], p) for i in passages for p in passages[ctrl_of[i]]]
    X = extract_cached(lambda: Extractor(a.extractor), jobs, f"{a.cache}/extract_k{a.k}.jsonl", log=log)
    facts = {i: [X[key(i, p)] for p in passages[i] if X[key(i, p)] != NONE] for i in passages}
    ctrl = {i: [X[key(f"ctrl{i}", p)] for p in passages[ctrl_of[i]] if X[key(f"ctrl{i}", p)] != NONE] for i in passages}
    log("extraction done")

    eps = bloom_eps(a.bits)

    def with_fp(tag, test):
        return lambda i, c: (i in passages and test(i, c)) or passes_fp(f"{tag}|{i}|{c}", eps, a.seed)

    checksums = {
        "passage_any": with_fp("any", lambda i, c: any(near(p, B.subject[i], c) for p in passages[i])),
        f"passage_near{a.near}": with_fp("near", lambda i, c: any(near(p, B.subject[i], c, a.near) for p in passages[i])),
        "extracted": with_fp("ext", lambda i, c: any(matches(c, v) for v in facts[i])),
        "triple_oracle": with_fp("triple", lambda i, c: i in B.gold and popqa.is_correct(c, B.gold[i])),
    }
    systems = {"greedy": lambda i: B.G[i]["greedy"],
               "entity_filter_greedy": lambda i: B.G[i]["greedy"] if B.ent[i] > 0 else None}
    for name, ok in checksums.items():
        for N in N_LIST:
            systems[f"CAR_{name}_N{N}"] = B.car(ok, N)

    out = {"config": vars(a), "eps": eps, "systems": {}, "checksums": {}}
    answers = {}
    for name, fn in systems.items():
        out["systems"][name], answers[name] = B.evaluate(fn)
        r = out["systems"][name]
        log(f"{name:<28} acc {r['accuracy']:.3f} hal {r['hallucination']:.3f} abstain {r['abstain']:.3f} "
            f"ghosts {r['ghost_answered']:.3f}")
    for name, ok in checksums.items():
        out["checksums"][name] = B.pass_rates(ok)
        log(f"checksum {name}: {json.dumps(out['checksums'][name])}")

    gold_in_facts = [any(popqa.is_correct(v, B.gold[i]) for v in facts[i]) for i in passages]
    gold_in_ctrl = [any(popqa.is_correct(v, B.gold[i]) for v in ctrl[i]) for i in passages]
    out["extraction"] = {
        "questions_with_passages": len(passages),
        "passages": int(sum(map(len, passages.values()))),
        "questions_with_any_fact": float(np.mean([len(facts[i]) > 0 for i in passages])),
        "gold_among_extracted": float(np.mean(gold_in_facts)),
        "control_any_answer": float(np.mean([len(ctrl[i]) > 0 for i in passages])),
        # the extractor answering correctly from a passage about someone else: its own memory
        "control_gold_leak": float(np.mean(gold_in_ctrl)),
    }
    log(f"extraction: {json.dumps(out['extraction'])}")
    out["greedy_threshold_frontier"] = B.threshold_frontier()
    out["top_n_recall"] = B.top_n_recall()
    out["facts"] = facts
    out["answers"] = answers
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
