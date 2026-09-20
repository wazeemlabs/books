"""Two ways to produce the same tokens, at very different costs.

`naive` is Chapter 11: recompute the whole sequence for every token.
`cached` is Chapter 12: compute each token's keys and values once.

Both are greedy, so for a given prompt they must return identical
tokens. `check_equivalence` asserts it: the cache is an optimization,
not an approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .model import KVCache, Model, forward


@dataclass
class Run:
    """One generation, with the timing of every step."""

    tokens: list[int]
    ttft_s: float  # time to the first generated token
    step_s: list[float] = field(default_factory=list)  # decode steps after the first
    peak_cache_bytes: int = 0

    @property
    def total_s(self) -> float:
        return self.ttft_s + sum(self.step_s)

    @property
    def decode_tok_per_s(self) -> float:
        return len(self.step_s) / sum(self.step_s) if self.step_s else float("nan")


def naive(model: Model, prompt: list[int], n_new: int) -> Run:
    """Recompute every token's keys and values at every step."""
    seq = list(prompt)
    ttft, steps = 0.0, []
    for i in range(n_new):
        t0 = perf_counter()
        logits = forward(model, np.array(seq))
        seq.append(int(logits[-1].argmax()))
        dt = perf_counter() - t0
        if i == 0:
            ttft = dt
        else:
            steps.append(dt)
    return Run(tokens=seq[len(prompt) :], ttft_s=ttft, step_s=steps)


def cached(model: Model, prompt: list[int], n_new: int) -> Run:
    """Read earlier keys and values from the cache; compute only the new token."""
    cache = KVCache(model.cfg, max_seq=len(prompt) + n_new)

    t0 = perf_counter()
    logits = forward(model, np.array(prompt), cache)  # prefill
    nxt = int(logits[-1].argmax())
    ttft = perf_counter() - t0

    out, steps = [nxt], []
    for _ in range(n_new - 1):
        t0 = perf_counter()
        logits = forward(model, np.array([nxt]), cache)  # decode: one token
        nxt = int(logits[-1].argmax())
        steps.append(perf_counter() - t0)
        out.append(nxt)

    return Run(tokens=out, ttft_s=ttft, step_s=steps,
               peak_cache_bytes=cache.bytes_per_token() * cache.length)


def check_equivalence(model: Model, prompt: list[int], n_new: int) -> bool:
    return naive(model, prompt, n_new).tokens == cached(model, prompt, n_new).tokens
