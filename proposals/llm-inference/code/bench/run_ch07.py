"""Chapter 7: why wide hardware needs wide work.

An accelerator is thousands of arithmetic units fed by one memory
system. Hand it a small job and almost all of it idles -- which is
exactly what decoding one sequence does, and why Chapter 1 found a
$30,000 card running at a fraction of a percent of its capability.

You cannot measure a GPU from a laptop, but you can measure the same
effect on whatever you have, because it is a property of wide hardware
rather than of any one chip.

1. How much of this machine a matrix multiply uses, against its size.
2. How that changes with the number of cores allowed, for an operation
   limited by arithmetic and one limited by memory.

    python3 -m bench.run_ch07
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from tinyserve import device
from tinyserve.reference import (MODEL, PEAK_BF16_FLOPS,
                                 RIDGE_FLOP_PER_BYTE,
                                 STREAMING_MULTIPROCESSORS)

from .harness import write

SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
THREAD_COUNTS = [1, 2, 4]
PROBE_N = 1024
PROBE_MIB = 512


def probe(threads: int) -> dict:
    """Run one measurement in a fresh process with a fixed thread count."""
    env = {**os.environ, **{k: str(threads) for k in
           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
    out = subprocess.run(
        [sys.executable, "-m", "bench._probe", str(PROBE_N), str(PROBE_MIB)],
        capture_output=True, text=True, env=env, timeout=300, check=True)
    return json.loads(out.stdout.strip())


def main() -> None:
    rep = device.report()

    sizes = []
    for n in SIZES:
        gf = device.achieved_gflops(n)
        sizes.append({"n": n, "gflops": gf,
                      "work_flop": 2 * n**3,
                      "bytes_touched": 3 * n * n * 4})
    peak = max(s["gflops"] for s in sizes)
    for s in sizes:
        s["share_of_peak"] = s["gflops"] / peak

    scaling = []
    for t in THREAD_COUNTS:
        r = probe(t)
        scaling.append({"threads": t, **r})
    base = scaling[0]
    for s in scaling:
        s["gflops_speedup"] = s["gflops"] / base["gflops"]
        s["bandwidth_speedup"] = s["bandwidth_gb_s"] / base["bandwidth_gb_s"]

    # What a single decode step actually looks like to the hardware.
    d = MODEL.d_model
    payload = {
        "device": rep,
        "sizes": sizes,
        "peak_gflops": peak,
        "scaling": scaling,
        "decode_shape": {
            "description": "one token through one projection of the reference model",
            "m": 1, "k": d, "n": d,
            "comparison": "the same projection during prefill of a 1,200-token "
                          "prompt is 1,200 rows instead of 1",
        },
        "accelerator": {
            "peak_bf16_flops": PEAK_BF16_FLOPS,
            "ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE,
            "streaming_multiprocessors": STREAMING_MULTIPROCESSORS,
            # A kernel occupying one multiprocessor reports full utilization.
            "one_sm_share": 1 / STREAMING_MULTIPROCESSORS,
        },
        "smallest_share": min(s["share_of_peak"] for s in sizes),
    }

    path = write("results/ch07.json", payload)
    print(f"wrote {path}")
    print(f"  {rep['name']} ({rep['cores_visible']} cores)")
    for s in sizes:
        print(f"  {s['n']:>5}x{s['n']:<5} {s['gflops']:8.1f} GFLOP/s "
              f"({s['share_of_peak'] * 100:5.1f}% of this machine's best)")
    for s in scaling:
        print(f"  {s['threads']} thread(s): arithmetic {s['gflops']:7.1f} GFLOP/s "
              f"({s['gflops_speedup']:.2f}x)  memory {s['bandwidth_gb_s']:5.1f} GB/s "
              f"({s['bandwidth_speedup']:.2f}x)")


if __name__ == "__main__":
    main()
