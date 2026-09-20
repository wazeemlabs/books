"""What a server can do, from the hardware and the model alone.

A roofline model of decoding: each step fetches every weight once for
the whole batch, plus each sequence's own cached keys and values, and
performs two operations per parameter per token. Whichever of those two
costs is larger sets the step time.

This is arithmetic over published specifications, not a measurement,
and every chapter that uses it says so. Chapters 16 and 17 measure the
real curve; until then this is what predicts it.

Shared by Chapters 1 and 5 so the two cannot disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from .cost import flops_forward
from .reference import (HBM_BYTES_PER_S, KV_BYTES_PER_TOKEN, MODEL, PARAMS,
                        PEAK_BF16_FLOPS, WEIGHT_BYTES)


@dataclass(frozen=True)
class Step:
    """One decode step, for `batch` sequences each `seq` tokens long."""

    batch: int
    seq: int
    seconds: float
    bytes_read: int
    flops: int
    bound_by: str

    @property
    def tokens_per_s(self) -> float:
        return self.batch / self.seconds

    @property
    def inter_token_ms(self) -> float:
        """What one user waits between words: the whole step, every time."""
        return self.seconds * 1e3

    @property
    def flop_utilization(self) -> float:
        return self.flops / PEAK_BF16_FLOPS / self.seconds

    def usd_per_m_tokens(self, usd_per_hour: float) -> float:
        return (usd_per_hour / 3600) / self.tokens_per_s * 1e6


def mixed_step(batch: int, seq: int, chunk: int = 0, chunk_cached: int = 0,
               bytes_per_weight: int = 2) -> Step:
    """One iteration that advances `batch` sequences *and* reads a prompt chunk.

    This is what a stall-free schedule actually runs (Chapter 18). A
    single forward pass carries two kinds of work at once:

      * `batch` sequences each producing one token, over caches `seq`
        tokens long -- the decode step of Chapter 16;
      * `chunk` tokens of somebody's prompt, with `chunk_cached` tokens
        of that same prompt already processed in earlier iterations.

    The weights are fetched once for all of it, which is the entire
    reason mixing is worth doing. The arithmetic adds up: two
    operations per parameter per decoded token, plus a forward pass
    over the chunk attending back over everything before it.

    With `chunk=0` this is exactly `decode_step`; with `batch=0` it is
    exactly `prefill_step`. Both are defined in terms of it, and
    `test_the_two_steps_are_one_step` asserts that they agree.
    """
    scale = bytes_per_weight / 2
    cached = batch * seq + chunk_cached
    bytes_read = int(WEIGHT_BYTES * scale + cached * KV_BYTES_PER_TOKEN * scale)
    flops = 2 * PARAMS * batch + flops_forward(MODEL, chunk, chunk_cached + chunk)
    t_memory = bytes_read / HBM_BYTES_PER_S
    t_compute = flops / PEAK_BF16_FLOPS
    return Step(batch=batch, seq=seq, seconds=max(t_memory, t_compute),
                bytes_read=bytes_read, flops=flops,
                bound_by="memory" if t_memory >= t_compute else "compute")


def test_the_two_steps_are_one_step() -> None:
    """A mixed step with nothing mixed in must be the step it came from.

    Chapters 1, 5, 16 and 17 all quote `decode_step` and `prefill_step`.
    Chapter 18 needs a step that does both at once, and the only safe
    way to add one is to define the old two in terms of the new one --
    then check that nothing moved.
    """
    for batch, seq in ((1, 1500), (8, 2000), (64, 1500), (256, 8192)):
        a, b = decode_step(batch, seq), mixed_step(batch, seq)
        assert (a.seconds, a.bytes_read, a.flops) == (b.seconds, b.bytes_read,
                                                      b.flops)
    for tokens in (1, 128, 1200, 8192):
        a, b = prefill_step(tokens), mixed_step(0, 0, chunk=tokens)
        assert (a.seconds, a.bytes_read, a.flops) == (b.seconds, b.bytes_read,
                                                      b.flops)
    # Splitting a prompt must not change what it costs to read it.
    # It does move slightly, because `flops_forward` charges attention
    # as a full t_new x t_total rectangle rather than the causal half,
    # and the rectangle shrinks when the prompt is split. At the case
    # study's prompt length attention is a few per cent of a prefill,
    # so the drift is small -- but it is checked rather than assumed.
    import math
    whole = flops_forward(MODEL, 1200, 1200)
    for budget in (128, 256, 512, 1024):
        n = math.ceil(1200 / budget)
        split = sum(mixed_step(0, 0, min(budget, 1200 - k * budget),
                               k * budget).flops for k in range(n))
        assert abs(split - whole) / whole < 0.02, (
            f"chunking at {budget} moved the arithmetic by "
            f"{abs(split - whole) / whole:.1%}, which is too much to ignore")

    # The real cost of a small chunk: the weights are re-read for it,
    # and with nothing else in the step there is nothing to share them
    # with, so it stops being compute-bound.
    assert mixed_step(0, 0, chunk=64).bound_by == "memory"
    assert mixed_step(0, 0, chunk=4096).bound_by == "compute"
    # Unless there is decode work riding along on the same fetch.
    assert mixed_step(64, 1500, chunk=64).bytes_read < (
        mixed_step(64, 1500).bytes_read + mixed_step(0, 0, 64).bytes_read)


def decode_step(batch: int, seq: int, bytes_per_weight: int = 2) -> Step:
    """Model one decode step under the roofline.

    The weights are fetched once and shared by the whole batch; each
    sequence's cache is its own and is not shared. That asymmetry is
    why batching helps and why long contexts blunt it.

    `bytes_per_weight` is the precision the model is served at. Halving
    it halves what must be fetched, which is the whole of Part V's
    argument; the arithmetic is unchanged.
    """
    return mixed_step(batch, seq, bytes_per_weight=bytes_per_weight)


def largest_batch_within(itl_ms: float, seq: int, ceiling: int) -> int:
    """The biggest batch whose step still fits an inter-token budget.

    This is the whole of capacity planning in one function: throughput
    is bought with latency, and the budget decides how much you may buy.
    """
    best = 0
    for b in range(1, ceiling + 1):
        if decode_step(b, seq).inter_token_ms <= itl_ms:
            best = b
        else:
            break
    return best


def prefill_step(tokens: int, bytes_per_weight: int = 2) -> Step:
    """Model the prefill of a prompt `tokens` long.

    The same two costs as a decode step, with the numbers the other way
    round: the arithmetic is over every token of the prompt at once, so
    it dominates, and the weights are still fetched exactly once.

    The FLOP count is Chapter 3's, which charges attention
    `t_new * t_total` rather than the causal half. That makes this an
    upper bound on prefill, which is the conservative direction for a
    scheduler: it never under-states how long a prompt will stall
    everyone else.
    """
    step = mixed_step(0, 0, chunk=tokens, bytes_per_weight=bytes_per_weight)
    return Step(batch=1, seq=tokens, seconds=step.seconds,
                bytes_read=step.bytes_read, flops=step.flops,
                bound_by=step.bound_by)


def test_prefill_is_compute_bound_and_decode_is_not() -> None:
    """The division Chapter 3 measured, asserted where it is used."""
    assert prefill_step(1200).bound_by == "compute"
    assert decode_step(1, 1500).bound_by == "memory"
