"""What if the (entity, relation) keys are noisy, as with a real fact extractor?

    python -m experiments.robustness

During wake, each mention is
  recorded correctly           with prob. recall * (1 - mislink)
  recorded under a WRONG person with prob. recall * mislink   (entity linking error)
  missed entirely              with prob. 1 - recall
Then the store-only system (familiarity + episodic store, no core) answers the
natural query stream. No training needed, so this runs in seconds.
"""

import json

import numpy as np

from cogllm.evaluate import score
from cogllm.system import ABSTAIN_ENTITY, CMLM
from cogllm.world import WorldConfig, build_world, sample_queries


def run(seed, recall, mislink):
    w = build_world(WorldConfig(n_first=160, n_last=160, n_entities=10000, zipf_s=1.0, n_mentions=60000, seed=seed))
    rng = np.random.default_rng(seed + 55)
    u = rng.random(len(w.stream))
    keep = u < recall
    wrong = keep & (rng.random(len(w.stream)) < mislink)
    ment = w.stream.copy()
    ment[wrong, 0] = rng.integers(0, w.cfg.n_entities, wrong.sum())
    sys = CMLM.wake(w, ment[keep], k=10**9, seed=seed)
    # the store keeps the value that was actually mentioned, even if filed under the wrong person
    sys.store = type(sys.store)()
    for (e, r), (e0, r0) in zip(ment[keep], w.stream[keep]):
        sys.store.write((int(e), int(r)), int(w.values[e0, r0]))
    q = sample_queries(w, 20000, seed=seed + 7)
    pred, _ = sys.answer(w.names[q[:, 0]], q[:, 1])
    res = score(np.where(pred >= 0, pred, -1), w.values[q[:, 0], q[:, 1]], w.counts[q[:, 0], q[:, 1]])
    return {k: res[k] for k in ["accuracy", "hallucination", "abstain"]}


if __name__ == "__main__":
    rows = []
    for recall, mislink in [(1.0, 0.0), (0.9, 0.0), (0.7, 0.0), (1.0, 0.01), (1.0, 0.05), (0.9, 0.05), (0.8, 0.1)]:
        rs = [run(s, recall, mislink) for s in range(3)]
        row = {"recall": recall, "mislink": mislink,
               **{k: float(np.mean([r[k] for r in rs])) for k in rs[0]}}
        rows.append(row)
        print(row)
    json.dump(rows, open("results/robustness.json", "w"), indent=1)
