"""Questions nobody could have read: people who do not exist.

A ghost's name joins the first name of one PopQA person with the surname of
another, and is kept only if the full name never appears in the model's
pretraining corpus. It is then asked a PopQA question template that fits a
person. The only correct answer is to abstain.
"""

import numpy as np

# PopQA relations whose subject is a person
PERSON_RELATIONS = ("occupation", "place of birth", "father", "mother", "religion", "sport")


def templates(questions):
    out = {}
    for q in questions:
        if q.relation in PERSON_RELATIONS and q.subject in q.question and q.relation not in out:
            out[q.relation] = q.question.replace(q.subject, "{}")
    return out


def make(questions, corpus, n, seed=0):
    rng = np.random.default_rng(seed)
    people = [q.subject for q in questions if q.relation in PERSON_RELATIONS and len(q.subject.split()) == 2]
    firsts = sorted({p.split()[0] for p in people})
    lasts = sorted({p.split()[1] for p in people})
    real = set(people)
    tmpl = templates(questions)
    rels = sorted(tmpl)
    names = []
    while len(names) < n:
        want = n - len(names)
        cand = sorted({f"{rng.choice(firsts)} {rng.choice(lasts)}" for _ in range(3 * want)} - real - set(names))
        counts = corpus.counts(cand)
        names += [c for c, k in zip(cand, counts) if k == 0][:want]
    return [{"id": f"ghost{i}", "subject": nm, "relation": r, "question": tmpl[r].format(nm)}
            for i, (nm, r) in enumerate(zip(names, rng.choice(rels, n)))]
