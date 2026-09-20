"""Chapter 29: guess several tokens, check them in one pass.

Five measurements:

1. Exactness. The tokens speculative decoding keeps are distributed
   exactly as the expensive model alone would have produced them, and
   this samples the distribution to show it.
2. Acceptance. How often a draft's guess survives, for drafts of
   several qualities -- here, quantized copies of the target from
   Chapter 24, which is a draft that is cheaper and related.
3. The formula. Expected tokens per round against the closed form,
   simulated.
4. The speedup, against how many tokens are guessed and what a draft
   step costs, with the best number of guesses for each.
5. The case study: what it is worth on the book's 8B model, and what
   it does to the gap between tokens.

    python3 -m bench.run_ch29
"""

from __future__ import annotations

import numpy as np

from tinyserve import model as m, quantize as q, speculative as sp
from tinyserve.reference import CONTEXT_TOKENS, PARAMS, WEIGHT_BYTES
from tinyserve.serving import decode_step

from .harness import write

SEED = 0
CFG = m.Config()
PROMPT = 40
DRAWS = 400_000
ALPHAS = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]
# Every k up to here, so that each curve's peak is on the chart
# rather than off the end of it.
KS = list(range(1, 25))
# A draft step as a fraction of a target step. A quantized copy of the
# same model reads fewer weight bytes and a decode step is limited by
# exactly that (Chapter 4), so its cost is its bytes per weight over
# the target's. A separately trained small model is cheaper still.
DRAFT_COSTS = {
    "int8 copy of the target": 1 / 2,
    "int4 copy, groups of 32": 0.625 / 2,
    "a model a tenth the size": 0.1,
    "a model a fortieth the size": 0.025,
}
DRAFT_SCHEMES = [
    ("int8, per tensor", q.Scheme(8)),
    ("int4, groups of 32", q.Scheme(4, group=32)),
    ("int4, per tensor", q.Scheme(4)),
    ("int2, per tensor", q.Scheme(2)),
]


def exactness() -> dict:
    """Sample the kept tokens and compare with what the target wanted."""
    rng = np.random.default_rng(SEED)
    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    q_ = np.array([0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.15, 0.15])
    counts = np.zeros(len(p))
    accepted = 0
    for _ in range(DRAWS):
        token = sp.sample(q_, rng)
        ok, kept = sp.verify(p, q_, token, rng)
        counts[kept] += 1
        accepted += ok
    empirical = counts / DRAWS
    se = np.sqrt(p * (1 - p) / DRAWS)
    return {
        "draws": DRAWS,
        "target": p.tolist(),
        "draft": q_.tolist(),
        "kept": empirical.tolist(),
        "standard_error": se.tolist(),
        "largest_deviation": float(np.abs(empirical - p).max()),
        "largest_in_standard_errors": float(
            np.max(np.abs(empirical - p) / se)),
        "total_variation": float(0.5 * np.abs(empirical - p).sum()),
        "acceptance_rate": accepted / DRAWS,
        # What the draft alone would have produced, for contrast: this
        # is the distribution the correction has to undo.
        "draft_total_variation": float(0.5 * np.abs(q_ - p).sum()),
    }


def acceptance() -> dict:
    """How often a real draft's guess survives.

    The drafts are quantized copies of the target (Chapter 24). That is
    a draft that is genuinely cheaper to run and genuinely related to
    what it is drafting for, which is the property that matters: an
    unrelated model agrees with the target about as often as chance,
    and this measures that too.
    """
    net = m.build(CFG, seed=SEED)
    prompt = (np.arange(PROMPT) * 7) % CFG.vocab_size
    target = m.forward(net, prompt)
    p = np.exp(target - target.max(-1, keepdims=True))
    p = p / p.sum(-1, keepdims=True)

    rows = []
    for name, scheme in DRAFT_SCHEMES:
        draft_logits = m.forward(q.quantize_model(net, scheme), prompt)
        d = np.exp(draft_logits - draft_logits.max(-1, keepdims=True))
        d = d / d.sum(-1, keepdims=True)
        greedy = float((draft_logits.argmax(-1) == target.argmax(-1)).mean())
        # Sampled acceptance: the chance the rule keeps a guess, which
        # is sum over tokens of min(p, q) -- the overlap of the two
        # distributions, averaged over positions.
        overlap = float(np.minimum(p, d).sum(-1).mean())
        rows.append({"draft": name, "greedy_agreement": greedy,
                     "sampled_acceptance": overlap,
                     "bytes_per_weight": scheme.bytes_per_weight(
                         per_scale=scheme.group or PARAMS)})
    # How peaked the target's own distribution is. An untrained model
    # is nearly uniform, and two nearly-uniform distributions overlap a
    # great deal whatever they are -- so the sampled acceptance rates
    # above are inflated and the chapter says so with this number
    # beside them.
    entropy = float(-np.sum(p * np.log(p + 1e-12), axis=-1).mean())
    unrelated = m.build(m.Config(n_layers=1, d_model=128), seed=1)
    ul = m.forward(unrelated, prompt)
    rows.append({
        "draft": "an unrelated small model",
        "greedy_agreement": float((ul.argmax(-1) == target.argmax(-1)).mean()),
        "sampled_acceptance": None, "bytes_per_weight": None})
    return {"rows": rows, "positions": PROMPT,
            "chance": 1.0 / CFG.vocab_size,
            "vocab": CFG.vocab_size,
            "target_entropy_nats": entropy,
            "uniform_entropy_nats": float(np.log(CFG.vocab_size)),
            "target_top1": float(np.median(p.max(-1)))}


def formula() -> dict:
    """The closed form for tokens per round, against the process."""
    rng = np.random.default_rng(SEED)
    n, trials = 4, 4000
    rows = []
    for alpha in ALPHAS:
        for k in (1, 2, 4, 8):
            p = np.full(n, (1 - alpha) / (n - 1))
            p[0] = alpha
            q_ = np.zeros(n); q_[0] = 1.0
            total = sum(len(sp.round_of(np.tile(p, (k + 1, 1)),
                                        np.tile(q_, (k, 1)), [0] * k, rng).tokens)
                        for _ in range(trials))
            rows.append({"alpha": alpha, "k": k,
                         "simulated": total / trials,
                         "formula": sp.expected_tokens(alpha, k),
                         "trials": trials})
    return {"rows": rows,
            "largest_gap": max(abs(r["simulated"] - r["formula"]) / r["formula"]
                               for r in rows)}


def speedups() -> dict:
    """How much faster, and how many guesses are worth making."""
    grid = []
    for label, cost in DRAFT_COSTS.items():
        for alpha in ALPHAS:
            k, gain = sp.best_k(alpha, cost)
            grid.append({"draft": label, "draft_cost": cost, "alpha": alpha,
                         "best_k": k, "speedup": gain,
                         "tokens_per_round": sp.expected_tokens(alpha, k),
                         "helps": gain > 1.0})
    curves = []
    for label, cost in DRAFT_COSTS.items():
        for alpha in (0.5, 0.7, 0.9):
            curves.append({
                "draft": label, "alpha": alpha, "draft_cost": cost,
                "ks": KS,
                "speedups": [sp.speedup(alpha, k, cost) for k in KS],
            })
    return {"grid": grid, "curves": curves, "ks": KS,
            "costs": DRAFT_COSTS}


def reference() -> dict:
    """What it is worth on the model the case study serves."""
    step = decode_step(1, CONTEXT_TOKENS)
    one_token_ms = step.inter_token_ms
    rows = []
    for label, cost in DRAFT_COSTS.items():
        for alpha in (0.7, 0.8, 0.9):
            k, gain = sp.best_k(alpha, cost)
            rows.append({
                "draft": label, "draft_cost": cost, "alpha": alpha,
                "best_k": k, "speedup": gain,
                "itl_ms": one_token_ms / gain,
                "round_ms": one_token_ms * (k * cost + 1),
                "tokens_per_round": sp.expected_tokens(alpha, k),
            })
    return {"rows": rows, "baseline_itl_ms": one_token_ms,
            "weight_bytes": WEIGHT_BYTES, "params": PARAMS,
            "context": CONTEXT_TOKENS}


def main() -> None:
    ex, acc, form = exactness(), acceptance(), formula()
    sped, ref = speedups(), reference()
    payload = {
        "exactness": ex, "acceptance": acc, "formula": form,
        "speedups": sped, "reference": ref,
        "assumptions": {
            "seed": SEED, "draws": DRAWS, "prompt_tokens": PROMPT,
            "alphas": ALPHAS, "ks": KS, "draft_costs": DRAFT_COSTS,
            "model": {"layers": CFG.n_layers, "d_model": CFG.d_model},
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch29.json", payload)
    print(f"wrote {path}")
    print(f"  exactness over {ex['draws']:,} draws: largest deviation "
          f"{ex['largest_deviation']:.5f} "
          f"({ex['largest_in_standard_errors']:.2f} standard errors); "
          f"total variation {ex['total_variation']:.5f} against the draft's "
          f"own {ex['draft_total_variation']:.3f}")
    print("  how often a draft's guess survives:")
    for r in acc["rows"]:
        s = ("--" if r["sampled_acceptance"] is None
             else f"{r['sampled_acceptance']*100:5.1f}%")
        print(f"    {r['draft']:28} greedy {r['greedy_agreement']*100:5.1f}%"
              f"   sampled {s}")
    print(f"  formula against simulation: worst gap "
          f"{form['largest_gap']*100:.1f}%")
    print("  best number of guesses, and what it is worth:")
    for r in sped["grid"]:
        if r["alpha"] in (0.7, 0.9):
            print(f"    {r['draft']:28} a={r['alpha']}: k={r['best_k']:2d}"
                  f"  {r['speedup']:5.2f}x"
                  f"{'' if r['helps'] else '   (does not help)'}")
    print(f"  the 8B model, {ref['baseline_itl_ms']:.2f} ms between tokens "
          f"without any of this:")
    for r in ref["rows"]:
        if r["alpha"] == 0.8:
            print(f"    {r['draft']:28} k={r['best_k']:2d}  "
                  f"{r['speedup']:5.2f}x  {r['itl_ms']:5.2f} ms between tokens")


if __name__ == "__main__":
    main()
