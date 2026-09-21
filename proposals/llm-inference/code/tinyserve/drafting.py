"""Drafting without a second model: heads, n-grams, and trees.

Chapter 29 needed two models -- a small one to guess and a big one to
check -- and most of the difficulty of running speculative decoding in
production is the small one: finding it, training it, serving it,
keeping the two in step. This chapter's methods remove it.

Three pieces here, and they answer three different questions:

* `head_cost` -- what a draft head weighs. A Medusa head is an extra
  projection to the whole vocabulary, which for a large vocabulary is
  not a small matrix, and every decode step reads it. The question it
  answers is what the speedup has to beat.
* `NgramIndex` -- drafting with no parameters at all, by looking for
  what followed the same words earlier in the prompt. The question it
  answers is how often that works, and on what.
* the tree functions -- given a fixed number of tokens a verification
  pass may carry, how they should be arranged. A chain of k guesses
  and a tree of k nodes cost the same and are not worth the same.

References: Cai, Li, Geng, Peng, Lee, Chen and Dao, "Medusa", ICML
2024; Li, Wei, Zhang and Zhang, "EAGLE", ICML 2024; Saxena, "Prompt
Lookup Decoding", 2023.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import prod

from .model import Config
from .reference import ELEM_BYTES, MODEL, PARAMS, WEIGHT_BYTES, params

# --- what a draft head weighs -------------------------------------------


@dataclass(frozen=True)
class HeadCost:
    name: str
    parameters: int
    bytes: int
    share_of_model: float
    note: str


def medusa_cost(heads: int = 3, layers: int = 1,
                cfg: Config = MODEL) -> HeadCost:
    """The parameters Medusa's decoding heads add.

    One head, as the reference implementation builds it, is
    `medusa_num_layers` residual blocks -- each a square `Linear` on
    the hidden size, with a bias -- followed by its own
    `Linear(hidden_size, vocab_size, bias=False)`. The last of those is
    the whole cost: a second copy of the output projection, per head.

    Defaults are the ones in the project's own training command:
    three heads, one layer each.
    """
    block = cfg.d_model * cfg.d_model + cfg.d_model      # weight + bias
    to_vocab = cfg.d_model * cfg.vocab_size              # no bias
    per_head = layers * block + to_vocab
    total = heads * per_head
    return HeadCost(
        name=f"Medusa, {heads} heads",
        parameters=total,
        bytes=total * ELEM_BYTES,
        share_of_model=total / params(cfg),
        note=(f"{heads} x ({layers} residual block(s) of "
              f"{block:,} + an output projection of {to_vocab:,})"))


def eagle_cost(published_parameters: int, cfg: Config = MODEL) -> HeadCost:
    """EAGLE's draft, taken from the numbers its authors publish.

    Unlike Medusa's, EAGLE's head is not reconstructable from a
    sentence: it is a small autoregressive model over the target's
    second-to-top-layer features. Its authors publish a parameter count
    per target model, so that is what is used, rather than a guess at
    the architecture.
    """
    return HeadCost(
        name="EAGLE draft head",
        parameters=published_parameters,
        bytes=published_parameters * ELEM_BYTES,
        share_of_model=published_parameters / params(cfg),
        note="published by its authors for a model of this size")


def ngram_cost(cfg: Config = MODEL) -> HeadCost:
    """No parameters at all. The comparison the other two are against."""
    return HeadCost(name="n-gram from the prompt", parameters=0, bytes=0,
                    share_of_model=0.0,
                    note="a dictionary over the prompt, built per request")


def step_cost_ratio(head: HeadCost) -> float:
    """What the head does to a memory-bound decode step.

    A decode step is dominated by reading the weights once. Weights
    that are bigger take longer to read, in proportion, and every step
    pays it -- including the steps where the draft guesses wrong.
    """
    return (WEIGHT_BYTES + head.bytes) / WEIGHT_BYTES


# --- drafting from the prompt, with no model at all ---------------------

TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(text: str) -> list[str]:
    """Words and punctuation, which is a stand-in for sub-word tokens.

    A real tokenizer splits into smaller and more repetitive units, so
    a real n-gram drafter has more to match on than this one does.
    Everything measured with this is therefore a floor rather than an
    estimate, and Chapter 30 says so where it uses the numbers.
    """
    return TOKEN.findall(text)


class NgramIndex:
    """Every n-gram in a context, and what followed it each time.

    This is prompt lookup decoding: to guess what comes next, find the
    longest recent run of words that appeared earlier in the context,
    and propose whatever followed it then. No model, no parameters, no
    training -- and nothing at all to propose when the context has
    nothing to say.

    The context grows as the reply is written, and so does the index:
    `extend` adds the n-grams that end at a newly written token. Real
    implementations match against the whole context and not only the
    prompt, which matters most on a reply that repeats itself.
    """

    def __init__(self, context: list[str], min_n: int = 2,
                 max_n: int = 8) -> None:
        if min_n < 1 or max_n < min_n:
            raise ValueError(f"bad n-gram range {min_n}..{max_n}")
        self.min_n, self.max_n = min_n, max_n
        self.context = list(context)
        self.following: dict[tuple[str, ...], Counter] = defaultdict(Counter)
        for n in range(min_n, max_n + 1):
            for i in range(len(self.context) - n):
                self.following[tuple(self.context[i:i + n])][
                    self.context[i + n]] += 1

    def extend(self, token: str) -> None:
        """Append one token, and index the n-grams that now end at it."""
        ctx = self.context
        for n in range(self.min_n, self.max_n + 1):
            if len(ctx) >= n:
                self.following[tuple(ctx[-n:])][token] += 1
        ctx.append(token)

    def candidates(self, recent: list[str], breadth: int) -> list[str]:
        """The most likely next tokens after `recent`, longest match first.

        Longest match first is the whole of the ranking: a nine-word
        match is better evidence than a two-word one, however often the
        two-word one occurred. Within one length, by how often.
        """
        for n in range(min(self.max_n, len(recent)), self.min_n - 1, -1):
            counts = self.following.get(tuple(recent[-n:]))
            if counts:
                return [tok for tok, _ in counts.most_common(breadth)]
        return []


def coverage(prompt: list[str], truth: list[str], depths: int,
             breadths: list[int], min_n: int = 2, max_n: int = 8
             ) -> dict[str, object]:
    """How often the true continuation is among the draft's top `b`.

    Measured position by position over real text, with the context
    growing as it would during a reply. For each starting point the
    drafter is asked for `b` candidates at depth 1; if the true token
    is among them the walk continues to depth 2 with the true token
    appended, and so on. Conditioning on having been right so far is
    deliberate: it is the only state a tree ever reaches depth `d` in.

    Each breadth is walked on its own index, because the drafter's
    proposals do not change what actually gets written -- the target
    model decides that, and here the target's output is the text.
    """
    rates: dict[int, list[float]] = {}
    proposed_at_all = 0
    for b in breadths:
        index = NgramIndex(prompt, min_n=min_n, max_n=max_n)
        hits, seen = [0] * depths, [0] * depths
        empty = 0
        for start in range(len(truth)):
            window = index.context[-max_n:]
            for d in range(depths):
                if start + d >= len(truth):
                    break
                seen[d] += 1
                got = index.candidates(window, b)
                if d == 0 and not got:
                    empty += 1
                if truth[start + d] not in got:
                    break
                hits[d] += 1
                window = (window + [truth[start + d]])[-max_n:]
            index.extend(truth[start])
        rates[b] = [h / s if s else 0.0 for h, s in zip(hits, seen)]
        proposed_at_all = 1 - empty / max(len(truth), 1)
    return {"cover": rates, "positions": len(truth),
            "had_a_guess": proposed_at_all,
            "min_n": min_n, "max_n": max_n}


# --- chains, trees, and what a verification pass should carry -----------


def expected_accepted(shape: tuple[int, ...],
                      cover: dict[int, list[float]]) -> float:
    """Tokens a round yields, for a tree of this shape.

    `shape[d]` is how many children every node at depth `d` has, so a
    shape of (1, 1, 1) is Chapter 29's chain of three guesses and
    (4, 2, 1) is a tree that tries four tokens first and narrows.

    A round is accepted down to the first depth where the target's own
    token is not among the candidates offered there, so the chance of
    getting at least `d` tokens is the product of the coverage rates
    down to `d`. Summing those gives the expected length, and the
    bonus token the target produces itself makes it one more --
    exactly as in `speculative.expected_tokens`.
    """
    total, reach = 1.0, 1.0
    for d, b in enumerate(shape):
        if b not in cover:
            raise KeyError(f"no coverage measured at breadth {b}")
        if d >= len(cover[b]):
            raise KeyError(f"no coverage measured at depth {d + 1}")
        reach *= cover[b][d]
        total += reach
    return total


def nodes(shape: tuple[int, ...]) -> int:
    """Tokens a verification pass has to carry for this shape.

    Every node in the tree is one position in the forward pass, and
    the tree attention mask is what keeps the branches from seeing
    each other. A chain of k guesses is k nodes; widening the first
    level to b multiplies everything below it.
    """
    return sum(prod(shape[:d + 1]) for d in range(len(shape)))


def best_shape(budget: int, cover: dict[int, list[float]],
               max_depth: int) -> tuple[tuple[int, ...], float, int]:
    """The arrangement of at most `budget` nodes that yields most.

    Searched exhaustively over shapes that fit, which is small enough
    to be exact: breadths come from what was measured, depth is
    bounded, and the node count prunes hard.
    """
    best: tuple[tuple[int, ...], float, int] = ((), 1.0, 0)
    widths = sorted(cover)

    def walk(shape: tuple[int, ...]) -> None:
        nonlocal best
        n = nodes(shape)
        if shape:
            value = expected_accepted(shape, cover)
            if value > best[1] or (value == best[1] and n < best[2]):
                best = (shape, value, n)
        if len(shape) >= max_depth:
            return
        for b in widths:
            nxt = shape + (b,)
            if nodes(nxt) <= budget:
                walk(nxt)

    walk(())
    return best


def chain_shape(k: int) -> tuple[int, ...]:
    return (1,) * k


def reach(shape: tuple[int, ...], cover: dict[int, list[float]]) -> list[float]:
    """The chance of getting *at least* `d` tokens, for d = 1, 2, ...

    The rates `coverage` reports are conditional on having been right
    so far, which is the form the arithmetic needs and the wrong form
    to read. A drafter that is right 13% of the time at depth one and
    84% of the time at depth five is not a drafter that usually gets
    five tokens; it is one that almost never gets there.
    """
    out, running = [], 1.0
    for d, b in enumerate(shape):
        running *= cover[b][d]
        out.append(running)
    return out


def zipf_cover(alpha: float, breadths: list[int], depths: int,
               vocab: int = 128_256, decay: float | None = None
               ) -> dict[int, list[float]]:
    """A drafter whose ranked guesses are worth something, modelled.

    The measured n-gram drafter gets almost nothing from a second
    candidate: its alternatives are rarer matches, not better guesses.
    A trained head is different -- it produces a distribution, and its
    second-most-likely token is a real second guess. To say anything
    about trees over such a head, this models the rank of the target's
    own token in the draft's ranking as Zipf with exponent `decay`,
    and reports the chance that rank is `b` or better.

    The exponent is solved for rather than chosen: it is whatever makes
    the top-1 rate equal `alpha`. So one measured number fixes the
    whole curve, and the curve is a model, which Chapter 30 says
    wherever it uses it.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    ranks = list(range(1, vocab + 1))

    def top1(s: float) -> float:
        total = sum(r ** -s for r in ranks)
        return 1.0 / total

    lo, hi = 0.0, 40.0
    if decay is None:
        for _ in range(200):
            mid = (lo + hi) / 2
            if top1(mid) < alpha:
                lo = mid
            else:
                hi = mid
        decay = (lo + hi) / 2
    weights = [r ** -decay for r in ranks]
    total = sum(weights)
    cumulative = []
    running = 0.0
    for w in weights[:max(breadths)]:
        running += w
        cumulative.append(running / total)
    return {b: [cumulative[b - 1]] * depths for b in breadths}


# --- tests ---------------------------------------------------------------


def test_a_medusa_head_is_mostly_its_output_projection() -> None:
    """The cost is the vocabulary, not the hidden layer.

    Worth asserting because it is the thing that makes a head
    expensive for a modern tokenizer and cheap for an old one.
    """
    cost = medusa_cost(heads=1)
    to_vocab = MODEL.d_model * MODEL.vocab_size
    assert to_vocab / cost.parameters > 0.95, to_vocab / cost.parameters
    assert medusa_cost(heads=3).parameters == 3 * cost.parameters


def test_a_bigger_head_slows_every_step() -> None:
    """Including the steps where the draft is wrong, which is the point."""
    assert step_cost_ratio(ngram_cost()) == 1.0
    assert step_cost_ratio(medusa_cost(heads=3)) > 1.1


def test_a_chain_is_the_tree_chapter_29_measured() -> None:
    """With one candidate at every depth the tree formula must reduce to
    the geometric series Chapter 29 derived."""
    from .speculative import expected_tokens
    alpha = 0.7
    cover = {1: [alpha] * 8}
    for k in range(0, 8):
        mine = expected_accepted(chain_shape(k), cover)
        theirs = expected_tokens(alpha, k)
        assert abs(mine - theirs) < 1e-12, (k, mine, theirs)


def test_a_wider_tree_costs_what_it_carries() -> None:
    assert nodes((1, 1, 1)) == 3
    assert nodes((4,)) == 4
    assert nodes((4, 2)) == 4 + 8
    assert nodes((2, 2, 2)) == 2 + 4 + 8


def test_breadth_beats_depth_when_the_draft_is_poor() -> None:
    """A second guess at depth one is worth more than a first guess at
    depth four, once the first guess is usually wrong. The search has
    to find that rather than be told it."""
    cover = {1: [0.4, 0.4, 0.4, 0.4], 2: [0.6, 0.6, 0.6, 0.6]}
    shape, value, _ = best_shape(budget=6, cover=cover, max_depth=4)
    assert shape[0] == 2, shape
    assert value > expected_accepted(chain_shape(4), cover), value


def test_the_index_proposes_what_followed_before() -> None:
    ctx = tokens("the cat sat on the mat . the cat sat on the rug .")
    index = NgramIndex(ctx, min_n=2, max_n=4)
    assert index.candidates(tokens("the cat sat on"), 2)[0] == "the"
    assert index.candidates(tokens("nothing like this"), 3) == []


def test_extending_the_index_matches_rebuilding_it() -> None:
    """The reply is indexed as it is written, which is only worth doing
    if it gives the same index as starting again would."""
    text = tokens("one two three one two four one two three five six")
    grown = NgramIndex(text[:5], min_n=2, max_n=4)
    for tok in text[5:]:
        grown.extend(tok)
    whole = NgramIndex(text, min_n=2, max_n=4)
    assert grown.context == whole.context
    assert grown.following == whole.following


def test_a_drafter_with_nothing_to_copy_proposes_nothing() -> None:
    """The failure mode prompt lookup has and a model does not: on text
    that repeats nothing, it is not a bad drafter, it is no drafter."""
    fresh = tokens("alpha beta gamma delta epsilon zeta eta theta")
    out = coverage(tokens("nothing here resembles that"), fresh,
                   depths=2, breadths=[1, 2])
    assert out["cover"][1][0] == 0.0, out
    assert out["had_a_guess"] == 0.0, out
