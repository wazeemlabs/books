"""Passages of the pretraining corpus that mention a subject.

infini-gram's find query gives every occurrence of the subject's name in
OLMo-mix-1124 (exact: a single phrase is never subsampled). We read up to k
of them, spread evenly over all occurrences, each as a window of
max_disp_len tokens around the mention. A checksum built from these
passages reflects what the model read about the subject; with many
mentions it is a sample, so it can miss facts (the model then abstains)
but it does not invent them.
"""

import numpy as np

DOC_FIELDS = ("doc_ix", "spans", "needle_offset", "disp_len")


def _text(spans):
    # a matched span loses the space its first token carried; put it back
    out = ""
    for i, (txt, clause) in enumerate(spans):
        if clause is not None and out and not out[-1].isspace():
            out += " "
        out += txt
        if clause is not None and i + 1 < len(spans) and spans[i + 1][0][:1].isalnum():
            out += " "
    return out


def fetch(corpus, subjects, k=10, window=200):
    """{subject: {"mentions": n, "passages": [text, ...]}} for every subject."""
    finds = dict(zip(subjects, corpus.queries([{"query_type": "find", "query": s} for s in subjects],
                                              keep=("cnt", "segment_by_shard"))))
    plan = []
    for s, f in finds.items():
        seg = f["segment_by_shard"]
        sizes = np.array([b - a for a, b in seg])
        total = int(sizes.sum())
        if total == 0:
            continue
        # k evenly spaced occurrences over all shards, the same on every run
        for g in np.unique(np.linspace(0, total - 1, min(k, total)).astype(int)):
            shard = int(np.searchsorted(np.cumsum(sizes), g, side="right"))
            rank = int(seg[shard][0] + g - (np.cumsum(sizes)[shard - 1] if shard else 0))
            plan.append((s, {"query_type": "get_doc_by_rank", "query": s, "s": shard, "rank": rank,
                             "max_disp_len": window}))
    docs = corpus.queries([p for _, p in plan], keep=DOC_FIELDS)
    out = {s: {"mentions": int(sum(b - a for a, b in f["segment_by_shard"])), "passages": []} for s, f in finds.items()}
    seen = {s: set() for s in subjects}
    for (s, _), d in zip(plan, docs):
        t = _text(d["spans"])
        if t not in seen[s]:  # the web repeats itself; keep each passage once
            seen[s].add(t)
            out[s]["passages"].append(t)
    return out


def _positions(words, phrase):
    n = len(phrase)
    return [i for i in range(len(words) - n + 1) if words[i:i + n] == phrase] if n else []


def near(passage, subject, value, window=None):
    """Does value occur in the passage (window=None), or within `window` words
    of a mention of subject? Compared on normalised words."""
    from .popqa import normalize
    words, v = normalize(passage).split(), normalize(value).split()
    vp = _positions(words, v)
    if not vp or window is None:
        return bool(vp)
    s = normalize(subject).split()
    return any(abs(i - j) <= window + max(len(s), len(v)) for i in _positions(words, s) for j in vp)
