"""The PopQA benchmark every real-model checksum is scored on: one setup,
one scorer.

A Bench holds the sampled questions, the ghost questions, the model's
cached generations and candidate lists, and whether the corpus names each
subject. A checksum is a function verdict(qid, candidate) -> bool; CAR with
list size N answers the first of the first N candidates it accepts.
"""

import hashlib
import math

import numpy as np

from . import ghosts as ghost_mod
from . import popqa
from .corpus import Corpus

N_LIST = (1, 3, 5, 10)
EXEMPLAR_RELATIONS = ("occupation", "place of birth", "genre", "country", "director", "capital of")


def bloom_eps(bits):
    return 0.5 ** (bits * math.log(2))  # optimal Bloom filter: (1/2)^(bits ln 2)


def passes_fp(key, eps, salt):
    """Deterministic Bloom false positive: True with probability eps."""
    h = int.from_bytes(hashlib.blake2b(f"{salt}|{key}".encode(), digest_size=8).digest(), "big")
    return h / 2 ** 64 < eps


def candidates(gen, n):
    """The greedy answer, then the beams, deduplicated by normalised text."""
    out, seen = [], set()
    for text in [gen["greedy"]] + [b[0] for b in gen["beams"]]:
        norm = popqa.normalize(text)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(text)
        if len(out) == n:
            break
    return out


class Bench:
    def __init__(self, n, n_ghosts, seed=0, beams=10, cache="data/cache", model=None, batch=8, log=print):
        self.seed = seed
        self.corpus = Corpus(f"{cache}/infinigram.jsonl")
        allq = popqa.load()
        self.qs, self.edges = popqa.sample(allq, n, seed=seed)
        chosen = {q.id for q in self.qs}
        self.exemplars = []
        for rel in EXEMPLAR_RELATIONS:  # one worked example per relation, never a test question
            q = next(x for x in allq if x.relation == rel and x.id not in chosen and x.s_pop > 10_000)
            self.exemplars.append((q.question, q.answers[0]))
        self.ghosts = ghost_mod.make(allq, self.corpus, n_ghosts, seed=seed)
        log(f"{len(self.qs)} questions, {len(self.ghosts)} ghosts")

        from .generate import generate_cached, load_cached
        items = [(q.id, q.question) for q in self.qs] + [(g["id"], g["question"]) for g in self.ghosts]
        path = f"{cache}/generations_seed{seed}_b{beams}.jsonl"
        G = load_cached(path)
        if any(i not in G for i, _ in items):
            from .generate import Generator  # loads the model only when something is missing
            prefix = "".join(f"Q: {q}\nA: {a}\n\n" for q, a in self.exemplars)
            G = generate_cached(Generator(model), items, path, prefix, batch=batch, n_beams=beams)
        self.G = G

        self.subject = {q.id: q.subject for q in self.qs} | {g["id"]: g["subject"] for g in self.ghosts}
        self.relation = {q.id: q.relation for q in self.qs} | {g["id"]: g["relation"] for g in self.ghosts}
        self.ent = dict(zip(self.subject, self.corpus.counts(list(self.subject.values()))))
        self.cand = {i: candidates(G[i], max(N_LIST)) for i in self.subject}
        self.gold = {q.id: q.answers for q in self.qs}
        self.pop_bin = {q.id: int(np.clip(np.searchsorted(self.edges, np.log10(q.s_pop + 1), side="right") - 1, 0, 4))
                        for q in self.qs}

    def car(self, verdict, N):
        """CAR answer function: first of the first N candidates the checksum accepts,
        after the entity filter; None is an abstention."""
        return lambda i: next((c for c in self.cand[i][:N] if self.ent[i] > 0 and verdict(i, c)), None)

    def evaluate(self, answer, read=None):
        ans = {i: answer(i) for i in self.subject}
        real = [q.id for q in self.qs]
        correct = np.array([ans[i] is not None and popqa.is_correct(ans[i], self.gold[i]) for i in real])
        answered = np.array([ans[i] is not None for i in real])
        bins = np.array([self.pop_bin[i] for i in real])
        wrong = answered & ~correct
        r = {"accuracy": float(correct.mean()), "hallucination": float(wrong.mean()),
             "abstain": float((~answered).mean()),
             "ghost_answered": float(np.mean([ans[g["id"]] is not None for g in self.ghosts])),
             "by_popularity": [{"accuracy": float(correct[bins == b].mean()),
                                "hallucination": float(wrong[bins == b].mean())} for b in range(5)]}
        if read is not None:
            rd = np.array([read[i] for i in real])
            r["on_read"] = {"accuracy": float(correct[rd].mean()), "hallucination": float(wrong[rd].mean())}
            r["on_unread"] = {"accuracy": float(correct[~rd].mean()), "hallucination": float(wrong[~rd].mean())}
        return r, ans

    def pass_rates(self, verdict):
        """How often the checksum accepts wrong and right candidates of real questions."""
        wrong, right = [], []
        for q in self.qs:
            if self.ent[q.id] == 0:
                continue
            for c in self.cand[q.id]:
                (right if popqa.is_correct(c, q.answers) else wrong).append(verdict(q.id, c))
        return {"wrong_candidates": len(wrong), "wrong_pass_rate": float(np.mean(wrong)),
                "right_candidates": len(right), "right_pass_rate": float(np.mean(right))}

    def top_n_recall(self):
        return {N: float(np.mean([any(popqa.is_correct(c, q.answers) for c in self.cand[q.id][:N]) for q in self.qs]))
                for N in N_LIST}

    def threshold_frontier(self, points=20):
        lp = np.array([self.G[q.id]["greedy_logprob"] for q in self.qs])
        out = []
        for t in np.quantile(lp, np.linspace(0, 0.95, points)):
            r, _ = self.evaluate(lambda i, t=t: self.G[i]["greedy"] if self.G[i]["greedy_logprob"] >= t else None)
            out.append({"threshold_logprob": float(t), **{k: r[k] for k in ("accuracy", "hallucination", "abstain", "ghost_answered")}})
        return out
