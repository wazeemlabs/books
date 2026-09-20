"""What this machine can actually do: bandwidth and arithmetic.

Two measurements, one definition each, shared by every chapter that
needs them. Chapter 4 runs them single-threaded, to compare fetching
with computing on equal terms; Chapter 16 runs them with whatever
threads the batch sweep uses, so that its roofline describes the same
machine the sweep ran on. Same code, different conditions, and each
chapter says which.
"""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np

MATMUL_N = 1024


def stream_bandwidth(kib: int, target_s: float = 0.25) -> float:
    """Bytes per second reading a working set of `kib` KiB, repeatedly.

    A dot product of an array with itself touches every byte and does
    one multiply-add per four bytes -- far below any break-even point,
    so what this measures is fetching, not arithmetic. It runs through
    the vectorized BLAS path, so the result is not limited by Python
    call overhead the way a plain sum is.
    """
    a = np.ones(kib * 1024 // 4, dtype=np.float32)
    np.dot(a, a)                             # warm this level of the hierarchy
    reps, elapsed = 0, 0.0
    t0 = perf_counter()
    while elapsed < target_s:
        np.dot(a, a)
        reps += 1
        elapsed = perf_counter() - t0
    return a.nbytes * reps / elapsed


def matmul_flops(n: int = MATMUL_N, runs: int = 7) -> tuple[float, float]:
    """Peak arithmetic rate, and how much it moves between runs.

    On a shared machine this is the noisiest number in the chapter, so
    it is reported with its spread rather than alone.
    """
    x = np.random.rand(n, n).astype(np.float32)
    y = np.random.rand(n, n).astype(np.float32)
    x @ y
    rates = []
    for _ in range(runs):
        t0 = perf_counter(); x @ y; rates.append(2 * n**3 / (perf_counter() - t0))
    mid = median(rates)
    return mid, (max(rates) - min(rates)) / 2 / mid
