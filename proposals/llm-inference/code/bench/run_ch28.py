"""Chapter 28: whether the model actually got worse.

Six measurements, all of them about the experiment rather than the
model:

1. What each of the usual benchmarks can detect, and what it cannot.
2. What pairing is worth, and why it is worth most exactly where a
   compression decision lives.
3. How many items it takes to sign off a stated quality budget.
4. What running several benchmarks does to the chance of a false
   result.
5. Why perplexity can barely move while the answers change.
6. Chapter 24's own evaluation, measured against this chapter's
   standard.

    python3 -m bench.run_ch28
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tinyserve.evaluate import (ALPHA, POWER, Comparison, any_false_positive,
                                largest_difference, loss_shift, min_detectable,
                                paired_scores_power, perplexity_ratio, power,
                                replies_touched)
from tinyserve.reference import GPU_USD_PER_HOUR, OUTPUT_TOKENS

from .harness import write

SEED = 0
TRIALS = 4_000
# Test-set sizes, read from each dataset's own published splits
# (FACTS.md). "an internal eval" is the size a team actually writes.
BENCHMARKS = [
    ("MMLU (test)", 14_042, 0.65),
    ("GSM8K (test)", 1_319, 0.55),
    ("HumanEval", 164, 0.60),
    ("an internal eval", 200, 0.80),
]
# How often two models disagree on an item. A compression run changes
# few answers, which is the regime the whole chapter is about.
DISCORDANCES = [0.02, 0.05, 0.10, 0.20, 0.40]
OPERATING_DISCORDANCE = 0.05
ACCURACY = 0.60                         # the baseline Chapter 24's run is held at
BUDGETS = [0.005, 0.01, 0.02, 0.05]     # quality drops a team might allow
N_BENCHMARKS = [1, 2, 3, 5, 10, 20]
# What "a small share of tokens got much worse" does to a mean loss.
BROKEN_FRACTIONS = [0.001, 0.01, 0.05]
WORSE_BY = [0.5, 1.0, 2.0, 4.0]         # nats added to those tokens' loss
CH24 = Path("results/ch24.json")
SHIFTS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]


def mdd(n: int, accuracy: float, discordance: float, test: str,
        rng) -> float:
    return min_detectable(n, accuracy, discordance, test, rng, trials=TRIALS)


def benchmarks(rng) -> list[dict]:
    """What each benchmark can find, paired and not."""
    rows = []
    for name, n, accuracy in BENCHMARKS:
        paired = mdd(n, accuracy, OPERATING_DISCORDANCE, "paired", rng)
        unpaired = mdd(n, accuracy, OPERATING_DISCORDANCE, "unpaired", rng)
        rows.append({
            "benchmark": name, "n": n, "accuracy": accuracy,
            "discordance": OPERATING_DISCORDANCE,
            "paired_mdd": paired, "unpaired_mdd": unpaired,
            "ratio": (unpaired / paired
                      if paired and not math.isnan(unpaired) else None),
            "paired_items": paired * n if not math.isnan(paired) else None,
        })
    return rows


def pairing(rng) -> list[dict]:
    """What pairing is worth, against how much the models disagree."""
    rows = []
    for d in DISCORDANCES:
        for name, n, accuracy in BENCHMARKS[:2]:
            paired = mdd(n, accuracy, d, "paired", rng)
            unpaired = mdd(n, accuracy, d, "unpaired", rng)
            rows.append({
                "benchmark": name, "n": n, "discordance": d,
                "paired_mdd": paired, "unpaired_mdd": unpaired,
                "ratio": (unpaired / paired
                          if paired and not math.isnan(unpaired) else None),
            })
    return rows


def sign_off(rng) -> list[dict]:
    """How many items it takes to show a drop is under a stated budget.

    A quality budget is only a decision if the experiment can see it.
    This searches for the smallest benchmark whose smallest detectable
    difference is at or below the budget, which is the number a
    go/no-go meeting actually needs.
    """
    rows = []
    for budget in BUDGETS:
        for test in ("paired", "unpaired"):
            lo, hi = 32, 1 << 21
            if mdd(hi, 0.65, OPERATING_DISCORDANCE, test, rng) > budget:
                rows.append({"budget": budget, "test": test, "items": None})
                continue
            while lo < hi:
                mid = int(math.sqrt(lo * hi))
                if mid <= lo:
                    mid = lo + 1
                if mdd(mid, 0.65, OPERATING_DISCORDANCE, test, rng) <= budget:
                    hi = mid
                else:
                    lo = mid
                if hi - lo <= max(16, lo // 20):
                    break
            rows.append({"budget": budget, "test": test, "items": hi})
    return rows


def multiple() -> list[dict]:
    return [{"benchmarks": k, "any_false_positive": any_false_positive(k),
             "alpha": ALPHA} for k in N_BENCHMARKS]


def perplexity() -> dict:
    rows = []
    for frac in BROKEN_FRACTIONS:
        for worse in WORSE_BY:
            delta = loss_shift(frac, worse)
            rows.append({"fraction": frac, "worse_by": worse,
                         "mean_loss_shift": delta,
                         "perplexity_ratio": perplexity_ratio(delta),
                         "perplexity_pct": (perplexity_ratio(delta) - 1) * 100,
                         "replies_touched": replies_touched(frac,
                                                            OUTPUT_TOKENS)})
    return {"rows": rows, "fractions": BROKEN_FRACTIONS,
            "worse_by": WORSE_BY, "reply_tokens": OUTPUT_TOKENS}


def chapter_24(rng) -> dict:
    """Chapter 24's evaluation, held to this chapter's standard."""
    if not CH24.exists():
        raise SystemExit("run `make ch24` first: this chapter measures that "
                         "chapter's evaluation")
    d = json.loads(CH24.read_text())
    n = d["answers"]["positions"]
    rows = []
    for r in d["answers"]["rows"]:
        changed = max(r["positions_that_changed"], 1e-6)
        # The most favourable case there is: every changed answer goes
        # the same way, so the scheme is as bad as that much
        # disagreement allows. If the experiment cannot see even this,
        # it cannot see anything.
        best_case = largest_difference(ACCURACY, changed)
        verdict = power(Comparison(n, ACCURACY, changed, best_case),
                        TRIALS, rng)
        rows.append({
            "scheme": r["scheme"], "changed": r["positions_that_changed"],
            "changed_items": r["positions_that_changed"] * n,
            "best_case_difference": best_case,
            "paired_power": verdict["paired"],
            "unpaired_power": verdict["unpaired"],
            "shift_over_margin": r["shift_over_margin"],
        })
    continuous = [{"shift": s,
                   "power": paired_scores_power(n, s, 1.0, TRIALS, rng)}
                  for s in SHIFTS]
    return {"positions": n, "rows": rows, "continuous": continuous,
            "note": "the shift column is what that chapter measured instead"}


def frontier() -> dict:
    """What each scheme saves, against what it would take to sign it off."""
    d = json.loads(CH24.read_text())
    rows = []
    for m in d["memory"]["rows"]:
        # A decode step is memory-bound, so tokens a second scale with
        # the time to read the weights, and dollars per token with its
        # reciprocal.
        saving = 1.0 - 1.0 / m["over_bf16"] if m["over_bf16"] else 0.0
        rows.append({
            "scheme": m["scheme"], "bytes_per_weight": m["bytes_per_weight"],
            "weights_gb": m["weights_gb"], "read_ms": m["read_ms"],
            "faster_by": m["over_bf16"],
            "cost_saved": saving,
            "free_for_cache_gb": m["free_for_cache_gb"],
        })
    return {"rows": rows, "bf16_read_ms": d["memory"]["bf16_read_ms"],
            "gpu_usd_per_hour": GPU_USD_PER_HOUR}


def main() -> None:
    rng = np.random.default_rng(SEED)
    bench = benchmarks(rng)
    pair = pairing(rng)
    signoff = sign_off(rng)
    mult = multiple()
    ppl = perplexity()
    ch24 = chapter_24(rng)
    front = frontier()

    payload = {
        "benchmarks": bench, "pairing": pair, "sign_off": signoff,
        "multiple": mult, "perplexity": ppl, "chapter_24": ch24,
        "frontier": front,
        "assumptions": {
            "seed": SEED, "trials": TRIALS, "alpha": ALPHA, "power": POWER,
            "benchmarks": [{"name": n, "n": k, "accuracy": a}
                           for n, k, a in BENCHMARKS],
            "discordances": DISCORDANCES,
            "operating_discordance": OPERATING_DISCORDANCE,
            "budgets": BUDGETS, "n_benchmarks": N_BENCHMARKS,
            "broken_fractions": BROKEN_FRACTIONS, "worse_by": WORSE_BY,
            "shifts": SHIFTS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch28.json", payload)
    print(f"wrote {path}")

    pc = lambda x: "--" if x is None or math.isnan(x) else f"{x * 100:.2f}pt"
    print(f"  smallest difference each benchmark can find, at "
          f"{OPERATING_DISCORDANCE:.0%} disagreement:")
    for r in bench:
        print(f"    {r['benchmark']:<18}{r['n']:>7,} items   paired "
              f"{pc(r['paired_mdd']):>8}   unpaired {pc(r['unpaired_mdd']):>8}"
              + (f"   ({r['ratio']:.1f}x)" if r["ratio"] else ""))
    print("  what pairing is worth, against how much they disagree:")
    for r in pair:
        print(f"    {r['benchmark']:<18}disagree {r['discordance']:5.0%}   "
              f"paired {pc(r['paired_mdd']):>8}   unpaired "
              f"{pc(r['unpaired_mdd']):>8}"
              + (f"   ({r['ratio']:.1f}x)" if r["ratio"] else ""))
    print("  items needed to sign off a quality budget:")
    for r in signoff:
        got = "more than 2 million" if r["items"] is None else f"{r['items']:,}"
        print(f"    under {r['budget'] * 100:.1f} points, {r['test']:<9}"
              f"{got:>20}")
    print("  running several benchmarks and taking the best:")
    for r in mult:
        print(f"    {r['benchmarks']:>3}: {r['any_false_positive'] * 100:5.1f}% "
              f"chance at least one moves by luck")
    print("  what breaking a few tokens does to perplexity:")
    for r in ppl["rows"]:
        if r["worse_by"] in (1.0, 4.0):
            print(f"    {r['fraction'] * 100:5.1f}% of tokens, "
                  f"{r['worse_by']:.0f} nats worse: perplexity "
                  f"+{r['perplexity_pct']:.2f}%, and "
                  f"{r['replies_touched'] * 100:.0f}% of replies touched")
    print(f"  Chapter 24's {ch24['positions']} positions, as an experiment:")
    for r in ch24["rows"]:
        print(f"    {r['scheme']:<34}{r['changed_items']:4.0f} changed   "
              f"best case {r['best_case_difference'] * 100:4.1f}pt   "
              f"paired power {r['paired_power'] * 100:5.1f}%   unpaired "
              f"{r['unpaired_power'] * 100:5.1f}%")
    print("    the same 48 positions with a number per position:")
    for r in ch24["continuous"]:
        print(f"      shift {r['shift']:.1f} sd: power {r['power'] * 100:5.1f}%")


if __name__ == "__main__":
    main()
