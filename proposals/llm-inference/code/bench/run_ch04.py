"""Chapter 4: why fetching is slower than computing, measured on this machine.

Chapter 3 showed decode sitting far below the accelerator's break-even
point without saying why memory is the slow part. This chapter measures
it, on whatever machine the reader has:

1. The cache cliff -- effective read bandwidth against working-set size.
   The memory hierarchy becomes visible as the set outgrows each level.
2. This machine's own break-even point, to compare with an accelerator's.
3. The wall itself -- decode time against model size, as the weights
   outgrow cache and the measurement converges on what bandwidth alone
   predicts. This is the claim Chapter 3 made and did not prove.

Everything is measured single-threaded, so bandwidth and arithmetic are
comparable; the script re-executes itself to guarantee it.

    python3 -m bench.run_ch04
"""

from __future__ import annotations

import os
import sys

# Single-threaded, set before NumPy loads, so the two measurements below
# describe the same machine.
if os.environ.get("_CH04_PINNED") != "1":
    os.environ.update({k: "1" for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    os.environ["_CH04_PINNED"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "bench.run_ch04", *sys.argv[1:]])

from statistics import median  # noqa: E402
from time import perf_counter  # noqa: E402

import numpy as np  # noqa: E402

from tinyserve.model import Config, KVCache, build, forward  # noqa: E402
from tinyserve.reference import (HBM_BYTES_PER_S, PEAK_BF16_FLOPS,  # noqa: E402
                                 RIDGE_FLOP_PER_BYTE, WEIGHT_BYTES)

from .harness import write  # noqa: E402
from .machine import MATMUL_N, matmul_flops, stream_bandwidth  # noqa: E402

# Working sets from inside L1 to far outside the last level of cache.
WORKING_SETS_KIB = [64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

# Model shapes whose weights cross the cache boundary.
SHAPES = [(128, 4), (256, 4), (384, 6), (512, 6), (768, 8), (1024, 8), (1280, 10)]
DECODE_STEPS, PROMPT = 8, 16


def cache_topology() -> list[dict]:
    """The cache levels this core actually sees, from the kernel.

    Sizes are per core where the level is private and shared where it is
    not, which is what matters to a single-threaded measurement.
    """
    from pathlib import Path
    levels = []
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.exists():
        return levels
    for entry in sorted(base.glob("index*")):
        try:
            kind = (entry / "type").read_text().strip()
            if kind == "Instruction":
                continue
            size = (entry / "size").read_text().strip()
            shared = (entry / "shared_cpu_list").read_text().strip()
            kib = int(size.rstrip("KM")) * (1024 if size.endswith("M") else 1)
            levels.append({"level": int((entry / "level").read_text()),
                           "type": kind, "kib": kib, "shared_with": shared})
        except (OSError, ValueError):
            continue
    return levels


def decode_step_seconds(cfg: Config, runs: int = 3) -> tuple[float, int]:
    model = build(cfg)
    weight_bytes = model.n_params * 4
    times = []
    for _ in range(runs):
        cache = KVCache(cfg, max_seq=PROMPT + DECODE_STEPS + 1)
        forward(model, np.arange(PROMPT) % cfg.vocab_size, cache)
        nxt = np.array([0])
        for _ in range(DECODE_STEPS):
            t0 = perf_counter(); forward(model, nxt, cache)
            times.append(perf_counter() - t0)
    return median(times), weight_bytes


def main() -> None:
    # 1. The hierarchy.
    cliff = [{"kib": k, "bytes_per_s": stream_bandwidth(k)}
             for k in WORKING_SETS_KIB]
    dram = cliff[-1]["bytes_per_s"]        # largest set: genuinely from memory
    fastest = max(c["bytes_per_s"] for c in cliff)

    # 2. This machine's break-even point.
    flops, flops_spread = matmul_flops()
    ridge = flops / dram

    # 3. The wall: decode time against model size.
    wall = []
    for d_model, n_layers in SHAPES:
        cfg = Config(vocab_size=512, d_model=d_model, n_layers=n_layers,
                     n_heads=max(1, d_model // 64), n_kv_heads=max(1, d_model // 64),
                     d_ff=4 * d_model, max_seq=64)
        step_s, weight_bytes = decode_step_seconds(cfg)
        predicted = weight_bytes / dram
        wall.append({
            "d_model": d_model, "n_layers": n_layers,
            "weight_bytes": weight_bytes,
            "weight_mib": weight_bytes / 1024**2,
            "decode_step_s": step_s,
            "predicted_from_bandwidth_s": predicted,
            "measured_over_predicted": step_s / predicted,
            "effective_bytes_per_s": weight_bytes / step_s,
        })

    payload = {
        "cache_topology": cache_topology(),
        "experiment": {"working_sets_kib": WORKING_SETS_KIB,
                       "matmul_n": MATMUL_N, "decode_steps": DECODE_STEPS,
                       "threads": 1},
        "cliff": cliff,
        "machine": {
            "dram_bytes_per_s": dram,
            "fastest_bytes_per_s": fastest,
            "cache_advantage": fastest / dram,
            "flops": flops,
            "flops_spread_frac": flops_spread,
            "flops_noisy": flops_spread > 0.05,
            "ridge_flop_per_byte": ridge,
        },
        "accelerator": {
            "hbm_bytes_per_s": HBM_BYTES_PER_S,
            "peak_bf16_flops": PEAK_BF16_FLOPS,
            "ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE,
            "weight_bytes_8b": WEIGHT_BYTES,
            "decode_floor_ms": WEIGHT_BYTES / HBM_BYTES_PER_S * 1e3,
            "ridge_ratio_vs_machine": RIDGE_FLOP_PER_BYTE / ridge,
        },
        "wall": wall,
    }

    path = write("results/ch04.json", payload)
    print(f"wrote {path}")
    print(f"  fastest read {fastest / 1e9:6.1f} GB/s (small working set)")
    print(f"  slowest read {dram / 1e9:6.1f} GB/s (largest working set)"
          f"  -> cache is {fastest / dram:.1f}x faster than memory")
    print(f"  arithmetic   {flops / 1e9:6.1f} GFLOP/s  -> breaks even at "
          f"{ridge:.0f} FLOP/byte (accelerator: {RIDGE_FLOP_PER_BYTE:.0f})")
    for w in wall:
        print(f"  d_model {w['d_model']:5d}  weights {w['weight_mib']:8.1f} MiB"
              f"  decode {w['decode_step_s'] * 1e3:7.2f} ms"
              f"  vs bandwidth prediction {w['measured_over_predicted']:5.2f}x")


if __name__ == "__main__":
    main()