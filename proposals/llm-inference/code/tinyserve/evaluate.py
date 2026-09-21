"""Whether a difference between two models is a difference at all.

Every chapter of Part V changes the model and then asks whether the
answers got worse. That question is a statistical one, and it is
usually asked badly: two accuracy numbers are printed, one is higher,
and a decision is made. Most of the differences reported that way are
noise, and most of the real ones are missed, and both failures come
from the same place -- running the wrong test on a paired experiment.

Four pieces:

* `mcnemar_p` -- the test for two models answering the *same* items.
  It looks only at the items they disagree on, which is the whole
  reason it can see what the other test cannot.
* `two_proportion_p` -- the test that gets used instead: two accuracy
  rates compared as if they came from different samples. They did not.
* `power` and `min_detectable` -- how small a difference each test can
  actually find on a benchmark of a given size, measured by
  simulation rather than by a formula, so the answer includes the
  discreteness that formulas smooth over.
* `paired_scores_power` -- what happens when the metric is a number
  rather than a verdict, which is usually available and almost always
  better.

Nothing here is specific to language models. It is specific to
comparing two things on the same test set, which is what a compression
decision always is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ALPHA = 0.05
POWER = 0.80
EXACT_BELOW = 25        # discordant pairs below which the exact test is used


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial test against p = 1/2.

    Used when there are few discordant pairs, where the chi-square
    approximation is not good and the answer matters most: a small
    benchmark comparing two models that mostly agree is exactly the
    case a compression decision lands in.
    """
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (1 << n)
    return min(1.0, 2.0 * tail)


def mcnemar_p(b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """McNemar's test: is the disagreement one-sided?

    `b` is how many items the first model got right and the second
    wrong; `c` the reverse. Items both got right and items both got
    wrong do not appear, which is the point -- they carry no
    information about which model is better, and including them is
    what makes the naive test blind.

    Uses the continuity-corrected chi-square where there are enough
    discordant pairs for it, and the exact binomial where there are
    not.
    """
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    total = b + c
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = np.where(total > 0,
                       (np.abs(b - c) - 1.0) ** 2 / np.maximum(total, 1.0),
                       0.0)
        chi = np.maximum(chi, 0.0)
    p = np.array([1.0 - (math.erf(math.sqrt(x / 2.0)) if x > 0 else 0.0)
                  for x in np.atleast_1d(chi)])
    small = np.atleast_1d(total) < EXACT_BELOW
    if small.any():
        bb = np.atleast_1d(b).astype(int)
        tt = np.atleast_1d(total).astype(int)
        for i in np.flatnonzero(small):
            p[i] = binom_two_sided(int(bb[i]), int(tt[i]))
    return p.reshape(np.shape(b)) if np.shape(b) else p[0]


def two_proportion_p(k1: np.ndarray, n1: int, k2: np.ndarray,
                     n2: int) -> np.ndarray:
    """The test that gets run instead: two accuracy rates, pooled.

    Correct for two independent samples. A benchmark run twice is not
    two independent samples -- it is the same questions twice -- and
    treating it as though it were throws the pairing away and with it
    most of the power.
    """
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (p1 - p2) / np.maximum(se, 1e-300), 0.0)
    return np.array([2.0 * (1.0 - normal_cdf(abs(v)))
                     for v in np.atleast_1d(z)]).reshape(np.shape(k1))


@dataclass(frozen=True)
class Comparison:
    """Two models on the same benchmark.

    `accuracy` is what the first one scores. `discordance` is the share
    of items the two answer differently, whichever way -- the number a
    compression run should report and almost never does. `difference`
    is the true gap between them, which the experiment is trying to
    detect.
    """

    n: int
    accuracy: float
    discordance: float
    difference: float

    def cells(self) -> tuple[float, float, float, float]:
        """The four cell probabilities of the paired table."""
        b = (self.discordance - self.difference) / 2      # first right only
        c = (self.discordance + self.difference) / 2      # second right only
        both = self.accuracy - b
        neither = 1.0 - both - b - c
        if min(b, c, both, neither) < -1e-12:
            raise ValueError(
                f"no table has accuracy {self.accuracy}, discordance "
                f"{self.discordance} and difference {self.difference}")
        return max(both, 0.0), max(b, 0.0), max(c, 0.0), max(neither, 0.0)


def largest_difference(accuracy: float, discordance: float) -> float:
    """The biggest true gap a table with these margins can hold.

    All the disagreement can go one way, but only until the cell it
    would have to come out of runs out: a model cannot be right on
    more items than there are, and the second model's extra wins have
    to come from items the first one got wrong.
    """
    return max(0.0, min(discordance, 2.0 * (1.0 - accuracy) - discordance))


def power(cmp: Comparison, trials: int, rng: np.random.Generator,
          alpha: float = ALPHA) -> dict[str, float]:
    """How often each test finds the difference, simulated.

    Both tests see exactly the same simulated benchmark runs, so the
    comparison between them is paired too.
    """
    draws = rng.multinomial(cmp.n, cmp.cells(), size=trials)
    both, b, c, _ = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
    paired = mcnemar_p(b, c)
    first, second = both + b, both + c
    unpaired = two_proportion_p(first, cmp.n, second, cmp.n)
    return {"paired": float((paired < alpha).mean()),
            "unpaired": float((unpaired < alpha).mean()),
            "mean_discordant": float((b + c).mean())}


def min_detectable(n: int, accuracy: float, discordance: float,
                   test: str, rng: np.random.Generator, trials: int = 2_000,
                   alpha: float = ALPHA, target: float = POWER) -> float:
    """The smallest true difference this benchmark can find, by bisection.

    Returns a share of items, so 0.01 is one accuracy point. `nan`
    means no difference small enough to be consistent with the stated
    discordance reaches the target power, which is itself the answer:
    the benchmark cannot do this.
    """
    hi = largest_difference(accuracy, discordance)
    lo = 0.0
    if hi <= 0 or power(Comparison(n, accuracy, discordance, hi), trials, rng,
                        alpha)[test] < target:
        return float("nan")
    for _ in range(18):
        mid = (lo + hi) / 2
        if power(Comparison(n, accuracy, discordance, mid), trials, rng,
                 alpha)[test] >= target:
            hi = mid
        else:
            lo = mid
    return hi


def paired_scores_power(n: int, shift: float, spread: float, trials: int,
                        rng: np.random.Generator,
                        alpha: float = ALPHA) -> float:
    """The same experiment with a number per item instead of a verdict.

    A benchmark that scores each item pass or fail throws away
    everything except the sign. A benchmark that keeps a number -- a
    log-probability, a distance, a margin -- keeps the size too, and a
    paired t-test on the per-item differences can see a shift far
    smaller than the pass rate ever will.
    """
    diffs = rng.normal(shift, spread, size=(trials, n))
    mean = diffs.mean(axis=1)
    sd = diffs.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mean / (sd / math.sqrt(n)), 0.0)
    # Normal critical value: at the sample sizes here the difference
    # from Student's t is under a per cent, and it is the same for both
    # arms of every comparison this is used in.
    crit = 1.959963984540054
    return float((np.abs(t) > crit).mean())


def any_false_positive(k: int, alpha: float = ALPHA) -> float:
    """Chance of at least one benchmark moving by luck, out of `k`.

    The same arithmetic as a cache's false-match rate: a small
    per-trial error rate, applied often enough, is a large rate.
    """
    return 1.0 - (1.0 - alpha) ** k


def loss_shift(fraction: float, worse_by: float) -> float:
    """What breaking a fraction of tokens does to a mean loss.

    Perplexity is the exponential of a mean log-loss over tokens. A
    change that makes a small share of tokens much worse moves that
    mean by the product, which is small; the answers those tokens were
    part of are not small.
    """
    return fraction * worse_by


def perplexity_ratio(loss_delta: float) -> float:
    return math.exp(loss_delta)


def replies_touched(fraction: float, tokens: int) -> float:
    """Chance a reply of `tokens` contains at least one broken token.

    The number a mean over tokens cannot show you. A fraction small
    enough to be invisible in a perplexity figure is not small at the
    scale a user experiences, because a user experiences a whole reply
    and not an average token.
    """
    return 1.0 - (1.0 - fraction) ** tokens


# --- tests ---------------------------------------------------------------


def test_a_fraction_too_small_to_see_is_not_small_per_reply() -> None:
    """One token in a thousand moves a mean loss by a tenth of a per
    cent and shows up in a quarter of three-hundred-token replies."""
    assert loss_shift(0.001, 1.0) == 0.001
    assert perplexity_ratio(0.001) - 1 < 0.002
    assert 0.2 < replies_touched(0.001, 300) < 0.3


def test_the_exact_test_agrees_with_the_obvious_cases() -> None:
    assert binom_two_sided(0, 0) == 1.0
    assert abs(binom_two_sided(5, 10) - 1.0) < 1e-12
    # Ten discordant pairs all one way: 2 x (1/2)^10.
    assert abs(binom_two_sided(0, 10) - 2 / 1024) < 1e-12


def test_the_paired_test_rejects_at_the_rate_it_claims() -> None:
    """A test whose false-positive rate is not alpha is not a test.

    The unpaired test's is far *below* alpha here, and that is not a
    point in its favour. On paired data the two accuracy rates move
    together, so treating them as independent samples inflates the
    standard error: the test under-rejects, which is the same defect
    as its lack of power seen from the other side. It is not
    conservative in a useful way -- it is wrong in a way that happens
    to be safe against one of the two errors.
    """
    rng = np.random.default_rng(0)
    got = power(Comparison(1_000, 0.60, 0.10, 0.0), 4_000, rng)
    assert 0.02 <= got["paired"] <= 0.07, got
    assert got["unpaired"] < 0.01, got


def test_pairing_finds_what_the_naive_test_cannot() -> None:
    """The chapter's whole point, as an assertion.

    Two models that agree on most items and differ consistently on the
    rest: the paired test sees it, the unpaired one does not.
    """
    rng = np.random.default_rng(1)
    got = power(Comparison(1_319, 0.60, 0.04, 0.02), 3_000, rng)
    assert got["paired"] > 0.90, got
    assert got["unpaired"] < 0.50, got


def test_the_smallest_detectable_difference_shrinks_with_the_benchmark() -> None:
    rng = np.random.default_rng(2)
    small = min_detectable(200, 0.60, 0.10, "paired", rng, trials=800)
    large = min_detectable(14_042, 0.60, 0.10, "paired", rng, trials=800)
    assert large < small, (small, large)


def test_forty_eight_items_can_only_see_a_large_difference() -> None:
    """Chapter 24 compared quantization schemes on 48 positions.

    On a pass-or-fail metric that is almost no experiment: even a ten
    point true difference is found less than half the time, and the
    unpaired test almost never. Which is why that chapter measured a
    number per position instead -- and a number per position, at the
    same 48, finds a shift of 0.4 standard deviations most of the
    time.
    """
    rng = np.random.default_rng(3)
    verdict = power(Comparison(48, 0.60, 0.10, 0.10), 8_000, rng)
    assert verdict["paired"] < 0.45, verdict
    assert verdict["unpaired"] < 0.10, verdict
    assert paired_scores_power(48, shift=0.42, spread=1.0, trials=8_000,
                               rng=rng) > 0.75


def test_a_continuous_metric_gains_power_with_the_sample() -> None:
    rng = np.random.default_rng(4)
    small = paired_scores_power(48, 0.15, 1.0, 4_000, rng)
    large = paired_scores_power(1_319, 0.15, 1.0, 4_000, rng)
    assert small < 0.35 < 0.95 < large, (small, large)


def test_the_largest_difference_is_the_largest_feasible_one() -> None:
    """Every difference up to the bound has a table, and the next one
    up does not."""
    for accuracy in (0.55, 0.65, 0.80):
        for d in (0.02, 0.10, 0.40):
            top = largest_difference(accuracy, d)
            Comparison(100, accuracy, d, top).cells()      # must not raise
            try:
                Comparison(100, accuracy, d, top + 1e-6).cells()
            except ValueError:
                continue
            if top < d:
                raise AssertionError(f"{accuracy}, {d}: bound too low")


def test_impossible_tables_are_refused() -> None:
    """A difference larger than the disagreement is not a table."""
    try:
        Comparison(100, 0.60, 0.02, 0.05).cells()
    except ValueError:
        return
    raise AssertionError("an impossible comparison was accepted")
