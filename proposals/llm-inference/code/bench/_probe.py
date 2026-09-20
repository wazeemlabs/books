"""One measurement, in a process with a fixed thread count.

Thread limits are read by the maths library when it loads, so they can
only be set before the process starts. Chapter 7 runs this module once
per thread count and collects the JSON it prints.

    OPENBLAS_NUM_THREADS=2 python3 -m bench._probe 1024 512
"""

from __future__ import annotations

import json
import sys
from statistics import median
from time import perf_counter

import numpy as np


def timed(fn, runs: int = 5) -> float:
    fn()
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        fn()
        times.append(perf_counter() - t0)
    return median(times)


def main(matmul_n: int, stream_mib: int) -> None:
    x = np.random.rand(matmul_n, matmul_n).astype(np.float32)
    y = np.random.rand(matmul_n, matmul_n).astype(np.float32)
    gflops = 2 * matmul_n**3 / timed(lambda: x @ y) / 1e9

    a = np.ones(stream_mib * 1024 * 1024 // 4, dtype=np.float32)
    bandwidth = a.nbytes / timed(lambda: np.dot(a, a)) / 1e9

    print(json.dumps({"gflops": gflops, "bandwidth_gb_s": bandwidth}))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
