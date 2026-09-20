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


def decode_step(batch: int, seq: int, bytes_per_weight: int = 2) -> Step:
    """Model one decode step under the roofline.

    The weights are fetched once and shared by the whole batch; each
    sequence's cache is its own and is not shared. That asymmetry is
    why batching helps and why long contexts blunt it.

    `bytes_per_weight` is the precision the model is served at. Halving
    it halves what must be fetched, which is the whole of Part V's
    argument; the arithmetic is unchanged.
    """
    scale = bytes_per_weight / 2
    bytes_read = int(WEIGHT_BYTES * scale + batch * seq * KV_BYTES_PER_TOKEN * scale)
    flops = 2 * PARAMS * batch
    t_memory = bytes_read / HBM_BYTES_PER_S
    t_compute = flops / PEAK_BF16_FLOPS
    return Step(batch=batch, seq=seq, seconds=max(t_memory, t_compute),
                bytes_read=bytes_read, flops=flops,
                bound_by="memory" if t_memory >= t_compute else "compute")


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
    scale = bytes_per_weight / 2
    bytes_read = int(WEIGHT_BYTES * scale)
    flops = flops_forward(MODEL, tokens, tokens)
    t_memory = bytes_read / HBM_BYTES_PER_S
    t_compute = flops / PEAK_BF16_FLOPS
    return Step(batch=1, seq=tokens, seconds=max(t_memory, t_compute),
                bytes_read=bytes_read, flops=flops,
                bound_by="memory" if t_memory >= t_compute else "compute")


def test_prefill_is_compute_bound_and_decode_is_not() -> None:
    """The division Chapter 3 measured, asserted where it is used."""
    assert prefill_step(1200).bound_by == "compute"
    assert decode_step(1, 1500).bound_by == "memory"
