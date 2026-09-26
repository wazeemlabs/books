"""PopQA (Mallen et al., ACL 2023): 14,267 questions about Wikidata facts,
each with its subject, relation, gold answer aliases and subject popularity
(monthly Wikipedia page views).

    data/popqa/test.tsv from https://huggingface.co/datasets/akariasai/PopQA
"""

import csv
import json
import re
import string
import unicodedata
from dataclasses import dataclass

import numpy as np

URL = "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv"


@dataclass
class Question:
    id: str
    subject: str
    relation: str
    question: str
    answers: list          # gold aliases
    s_pop: int
    subject_aliases: list


def load(path="data/popqa/test.tsv"):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            out.append(Question(row["id"], row["subj"], row["prop"], row["question"],
                                json.loads(row["possible_answers"]), int(row["s_pop"]),
                                json.loads(row["s_aliases"])))
    return out


def sample(questions, n, seed=0, bins=5):
    """n questions, equal numbers from each popularity quantile (the long tail
    is where hallucination lives, and uniform sampling would under-weight it)."""
    rng = np.random.default_rng(seed)
    # one fixed order per bin, so a smaller sample is a subset of a larger one
    # and a pilot's cached corpus counts carry over to the full run
    pop = np.log10(np.array([q.s_pop for q in questions]) + 1)
    edges = np.quantile(pop, np.linspace(0, 1, bins + 1))
    bin_of = np.clip(np.searchsorted(edges, pop, side="right") - 1, 0, bins - 1)
    picked = []
    for b in range(bins):
        idx = np.where(bin_of == b)[0]
        picked += list(rng.permutation(idx)[:n // bins])
    return [questions[i] for i in sorted(picked)], edges


def normalize(s):
    """SQuAD-style answer normalisation: lower case, no punctuation, articles or extra spaces."""
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_correct(candidate, answers):
    """PopQA's metric: correct if any gold alias appears in the normalised answer."""
    c = normalize(candidate)
    return bool(c) and any(normalize(a) and f" {normalize(a)} " in f" {c} " for a in answers)
