"""Scoring: accuracy, hallucination and abstention, split by exposure count."""

from collections import defaultdict

import numpy as np
import torch

from .world import IDK, World, count_buckets

BUCKETS = ["0", "1", "2-3", "4-7", "8-15", "16+"]


@torch.no_grad()
def answer_probs(model, world: World, names: np.ndarray, rels: np.ndarray, batch=4096):
    """Probability over [IDK, v_0 .. v_{V-1}] at the answer slot, per query."""
    V = world.cfg.n_values
    out = np.zeros((len(names), V + 1), dtype=np.float32)
    for i in range(0, len(names), batch):
        n, r = names[i:i + batch], rels[i:i + batch]
        x = torch.tensor([world.prompt(nm, rr) for nm, rr in zip(n, r)])
        logits = model(x)[:, -1]
        cand = torch.tensor([[IDK] + world.val_toks(rr) for rr in r])
        out[i:i + batch] = torch.softmax(logits.gather(1, cand), dim=-1).numpy()
    return out


def decide(probs, threshold=0.0, allow_idk=True):
    """Return predicted value id, or -1 for "I don't know".

    The model abstains if it puts most mass on IDK, or if its best value is
    below `threshold` (confidence thresholding, the standard parametric way).
    """
    vals = probs[:, 1:]
    best = vals.argmax(1)
    conf = vals.max(1)
    pred = best.copy()
    if allow_idk:
        pred[probs[:, 0] > conf] = -1
    pred[conf < threshold] = -1
    return pred


def wilson_upper(k, n, z=1.96):
    """95% Wilson upper bound on a rate seen k times in n trials (n = 0 gives 1)."""
    if n == 0:
        return 1.0
    p = k / n
    centre = p + z * z / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((centre + half) / (1 + z * z / n))


def score(pred, truth, counts, ghost=None):
    """pred/truth: value ids (-1 = abstain). counts: exposure count per query.

    For facts never seen (count 0) the only correct behaviour is to abstain,
    so any answer there counts as a hallucination even if it guessed right.
    """
    pred, truth, counts = map(np.asarray, (pred, truth, counts))
    answered = pred >= 0
    seen = counts > 0
    correct = answered & seen & (pred == truth)
    halluc = answered & ~(seen & (pred == truth))
    res = {
        "n": len(pred),
        "accuracy": float(correct.mean()),
        "hallucination": float(halluc.mean()),
        "abstain": float((~answered).mean()),
        "coverage": float(answered.mean()),
        "halluc_given_answered": float(halluc.sum() / max(1, answered.sum())),
        # conditional rates, so "0.00%" carries the sample size it rests on
        "unseen_n": int((~seen).sum()),
        "halluc_on_unseen": float(halluc[~seen].sum() / max(1, (~seen).sum())),
        "halluc_on_unseen_ub95": wilson_upper(int(halluc[~seen].sum()), int((~seen).sum())),
        "halluc_on_seen": float(halluc[seen].sum() / max(1, seen.sum())),
    }
    b = np.array(count_buckets(counts))
    per = {}
    for k in BUCKETS:
        m = b == k
        if m.sum() == 0:
            continue
        per[k] = {
            "n": int(m.sum()),
            "accuracy": float(correct[m].mean()),
            "hallucination": float(halluc[m].mean()),
            "abstain": float((~answered[m]).mean()),
        }
    res["by_count"] = per
    if ghost is not None:
        ghost = np.asarray(ghost)
        res["ghost_hallucination"] = float((ghost >= 0).mean())
        res["ghost_hallucination_ub95"] = wilson_upper(int((ghost >= 0).sum()), len(ghost))
    return res


def fmt(res, name=""):
    s = (f"{name:<28} acc {res['accuracy']:.3f}  halluc {res['hallucination']:.3f}  "
         f"abstain {res['abstain']:.3f}")
    if "ghost_hallucination" in res:
        s += f"  ghost-halluc {res['ghost_hallucination']:.3f}"
    for k, v in res.get("extra", {}).items():
        s += f"  {k} {v:.3f}" if isinstance(v, float) else f"  {k} {v}"
    s += "\n" + " " * 28 + " | ".join(
        f"n={k}: acc {v['accuracy']:.2f} hal {v['hallucination']:.2f}" for k, v in res["by_count"].items())
    return s
