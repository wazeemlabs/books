"""Guess several tokens, then check them all at once.

Decoding is slow for a reason Chapter 4 made precise: producing one
token reads every weight in the model, and one token is far too little
work to justify that. A batch of sequences amortises the read across
users (Chapter 16), but a single user's reply still comes out one
token per full pass over the weights.

Speculative decoding attacks that directly. Let a cheap model guess the
next few tokens. Then run the expensive model *once* over all of them
at the same time -- which costs it barely more than producing one,
because it is reading the same weights either way -- and keep the
guesses it agrees with.

The part that makes it usable rather than a heuristic is that the
keeping can be done so that the tokens which survive are distributed
*exactly* as the expensive model alone would have produced them. Not
approximately: exactly. This module implements that rule and the tests
below check the distribution it produces.

Written for `LLM Inference from the Ground Up`, Part VI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Round:
    """What one round of guess-and-check produced."""

    proposed: int                       # how many the draft guessed
    accepted: int                       # how many survived
    tokens: list[int] = field(default_factory=list)
    resampled: bool = False             # was the last token a correction
    bonus: bool = False                 # was a free token taken at the end


def sample(probs: np.ndarray, rng: np.random.Generator) -> int:
    """One token from a distribution."""
    return int(rng.choice(len(probs), p=probs / probs.sum()))


def verify(p: np.ndarray, q: np.ndarray, token: int,
           rng: np.random.Generator) -> tuple[bool, int]:
    """Accept the draft's token, or replace it, keeping p exact.

    `p` is what the expensive model wanted, `q` what the cheap one
    proposed from, and `token` what it proposed. The rule is:

      * accept with probability min(1, p[token] / q[token]);
      * otherwise sample a replacement from the part of `p` that `q`
        over-committed to -- max(p - q, 0), renormalised.

    Why that is exact is worth following, because it looks like it
    should bias the result. A token is returned either by acceptance,
    with probability q[x] * min(1, p[x]/q[x]) = min(q[x], p[x]), or by
    the replacement draw. The replacement is only reached when the
    proposal was rejected, and the distribution it draws from is
    exactly the shortfall p - q where p exceeds q. Add the two and
    every token comes out with probability p[x].

    The shortfall is where the intuition lives: the draft over-proposes
    tokens it likes too much and under-proposes the rest, and the
    correction puts back precisely what it left out.
    """
    p = p / p.sum()
    q = q / q.sum()
    if q[token] > 0 and rng.random() < min(1.0, p[token] / q[token]):
        return True, token
    shortfall = np.maximum(p - q, 0.0)
    total = shortfall.sum()
    if total <= 0:                      # p and q agree everywhere p is left
        return False, sample(p, rng)
    return False, sample(shortfall / total, rng)


def round_of(p_rows: np.ndarray, q_rows: np.ndarray, proposed: list[int],
             rng: np.random.Generator) -> Round:
    """One guess-and-check round over `len(proposed)` guesses.

    `p_rows` has one more row than there are guesses: the expensive
    model scored every guessed position *and* the position after the
    last one, in the same pass. That extra row is what makes a fully
    accepted round produce one more token than was guessed, for free.
    """
    out = Round(proposed=len(proposed), accepted=0)
    for i, token in enumerate(proposed):
        ok, chosen = verify(p_rows[i], q_rows[i], token, rng)
        out.tokens.append(chosen)
        if not ok:
            out.resampled = True
            return out
        out.accepted += 1
    # Every guess survived, so the expensive model's own next token is
    # already computed and costs nothing more.
    out.tokens.append(sample(p_rows[len(proposed)] / p_rows[len(proposed)].sum(),
                             rng))
    out.bonus = True
    return out


def expected_tokens(alpha: float, k: int) -> float:
    """Tokens per round when each guess is accepted with probability α.

    A round yields one token for the first rejection, plus one for each
    guess accepted before it, plus a bonus if all k survive. Summing
    the geometric series gives (1 - α^(k+1)) / (1 - α), which is the
    expression in Leviathan et al. Note what it says at the limits: no
    draft at all (k = 0) yields one token, and a perfect draft yields
    k + 1 rather than k, because of the bonus.
    """
    if alpha >= 1.0:
        return float(k + 1)
    return float((1 - alpha ** (k + 1)) / (1 - alpha))


def speedup(alpha: float, k: int, draft_cost: float) -> float:
    """How much faster, given what a draft step costs.

    `draft_cost` is one draft step as a fraction of one target step. A
    round costs k draft steps and one target step, and yields
    `expected_tokens` of them, against one token per target step
    without any of this.
    """
    cost = k * draft_cost + 1.0
    return expected_tokens(alpha, k) / cost


def best_k(alpha: float, draft_cost: float, upto: int = 64) -> tuple[int, float]:
    """The number of guesses worth making, and what it is worth.

    There is always a best k and it is not large. Each extra guess is
    paid for whether or not it survives, and the chance it survives
    falls geometrically, so the gain per guess decays while the cost
    per guess does not.
    """
    scores = [(speedup(alpha, k, draft_cost), k) for k in range(0, upto + 1)]
    best, k = max(scores)
    if k == upto:
        raise ValueError(
            f"the best k is at the top of the range searched ({upto}), so it "
            f"is a truncation and not an optimum. Raise `upto`.")
    return k, best


# --- tests: the specification -------------------------------------------


def test_the_kept_tokens_have_the_expensive_models_distribution() -> None:
    """The claim the whole technique rests on, checked by sampling.

    If this were only approximately true, speculative decoding would be
    a quality trade like quantization and would need an evaluation
    before anyone could deploy it. It is exactly true, so it does not.

    Checked against a draft that is deliberately wrong -- a different
    distribution, not a slightly worse one -- because a rule that is
    exact only when the draft is already good would be worthless.
    """
    rng = np.random.default_rng(0)
    n = 8
    p = np.array([0.30, 0.25, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02])
    q = np.array([0.05, 0.05, 0.10, 0.10, 0.20, 0.20, 0.15, 0.15])

    draws = 400_000
    counts = np.zeros(n)
    for _ in range(draws):
        token = sample(q, rng)
        _, kept = verify(p, q, token, rng)
        counts[kept] += 1
    empirical = counts / draws

    # Three standard errors of a binomial, which a correct rule clears
    # and a biased one does not.
    tolerance = 3 * np.sqrt(p * (1 - p) / draws)
    worst = float(np.max(np.abs(empirical - p) - tolerance))
    assert worst < 0, (
        f"the kept tokens are not distributed as p: largest excess over "
        f"three standard errors is {worst:.3g}\\n  p         {p}\\n"
        f"  empirical {empirical.round(4)}")


def test_a_useless_draft_still_gives_the_right_answer() -> None:
    """The degenerate case: a draft that proposes only one token, badly.

    The rule must still produce p. This is the case where the
    correction does all the work, and getting it wrong would be
    invisible on a good draft.
    """
    rng = np.random.default_rng(1)
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.0, 0.0, 1.0])       # always proposes the least likely
    counts = np.zeros(3)
    draws = 200_000
    for _ in range(draws):
        _, kept = verify(p, q, 2, rng)
        counts[kept] += 1
    empirical = counts / draws
    tolerance = 3 * np.sqrt(p * (1 - p) / draws)
    assert np.all(np.abs(empirical - p) < tolerance), (p, empirical)


def test_a_perfect_draft_is_always_accepted() -> None:
    """When the two models agree, nothing is ever thrown away."""
    rng = np.random.default_rng(2)
    p = np.array([0.4, 0.35, 0.25])
    for _ in range(2000):
        token = sample(p, rng)
        ok, kept = verify(p, p, token, rng)
        assert ok and kept == token


def test_the_expected_tokens_formula_matches_the_simulation() -> None:
    """The formula the chapter plots, against the loop it describes.

    A closed form is only worth quoting if it reproduces the process,
    and the bonus token at the end is the part most easily dropped.
    """
    rng = np.random.default_rng(3)
    n = 4
    for alpha in (0.2, 0.5, 0.8, 0.95):
        for k in (1, 2, 4, 8):
            # A pair of distributions with exactly this acceptance rate:
            # the draft proposes token 0 always, and p gives it α.
            p = np.full(n, (1 - alpha) / (n - 1))
            p[0] = alpha
            q = np.zeros(n)
            q[0] = 1.0
            total = 0
            trials = 4000
            for _ in range(trials):
                r = round_of(np.tile(p, (k + 1, 1)), np.tile(q, (k, 1)),
                             [0] * k, rng)
                total += len(r.tokens)
            measured = total / trials
            want = expected_tokens(alpha, k)
            assert abs(measured - want) < 0.06 * want + 0.05, (
                f"alpha={alpha} k={k}: simulated {measured:.3f}, "
                f"formula {want:.3f}")


def test_more_guesses_stop_paying() -> None:
    """There is a best number of guesses, and it is small.

    If `best_k` ever returned the top of its range, the cost model
    would be missing the term that makes long speculations wasteful,
    and every recommendation in the chapter would be wrong.
    """
    for alpha, cost in ((0.5, 0.1), (0.7, 0.2), (0.9, 0.05)):
        k, gain = best_k(alpha, cost, upto=64)
        assert 0 < k < 64, (alpha, cost, k)
        assert gain > 1.0, (alpha, cost, gain)
        # and one more guess than the best must be worse
        assert speedup(alpha, k + 1, cost) <= gain + 1e-12
