"""Chapter 8: the roofline, built from this machine and tested against it.

A roofline says the fastest any operation can run is whichever is
smaller: the machine's peak arithmetic, or its memory bandwidth times
how much arithmetic that operation does per byte. One line, two
numbers, and it predicts whether an optimization can possibly help
before you write it.

This measures the two numbers, places several real operations on the
chart, and checks each one against the bound. Then it does the same
arithmetic for the accelerator, where it decides the design.

    python3 -m bench.run_ch08
"""

from __future__ import annotations

from statistics import median
from time import perf_counter

import numpy as np

from tinyserve import serving
from tinyserve.cost import bytes_read, flops_forward
from tinyserve.model import Config, KVCache, build, forward
from tinyserve.reference import (HBM_BYTES_PER_S, MODEL, PARAMS,
                                 PEAK_BF16_FLOPS, PROMPT_TOKENS,
                                 RIDGE_FLOP_PER_BYTE, WEIGHT_BYTES)

from .harness import write

MATMUL_SIZES = [16, 64, 256, 1024, 2048]
STREAM_MIB = 512
TINY_PROMPT = 256


def timed(fn, runs: int = 5) -> float:
    fn()
    ts = []
    for _ in range(runs):
        t0 = perf_counter(); fn(); ts.append(perf_counter() - t0)
    return median(ts)


def matmul_point(n: int) -> dict:
    """An n x n multiply: 2n^3 operations over 3n^2 values of traffic.

    The byte count is the *compulsory* traffic -- two matrices in, one
    out. Caches can serve some of it, which is exactly why small sizes
    beat the bound below.
    """
    x = np.random.rand(n, n).astype(np.float32)
    y = np.random.rand(n, n).astype(np.float32)
    t = timed(lambda: x @ y)
    flops, byts = 2 * n**3, 3 * n * n * 4
    return {"name": f"{n}x{n} multiply", "kind": "matmul",
            "flops": flops, "bytes": byts, "intensity": flops / byts,
            "seconds": t, "achieved_flops": flops / t}


def stream_point() -> dict:
    a = np.ones(STREAM_MIB * 1024 * 1024 // 4, dtype=np.float32)
    t = timed(lambda: np.dot(a, a))
    flops, byts = 2 * a.size, a.nbytes
    return {"name": "streaming dot product", "kind": "memory",
            "flops": flops, "bytes": byts, "intensity": flops / byts,
            "seconds": t, "achieved_flops": flops / t}


def tinyserve_points() -> list[dict]:
    decode_runs = 20
    headroom = decode_runs + 8          # warmup plus timed steps, plus margin
    cfg = Config(max_seq=TINY_PROMPT + headroom)
    model = build(cfg)
    tokens = np.arange(TINY_PROMPT) % cfg.vocab_size

    def prefill() -> None:
        forward(model, tokens, KVCache(cfg, max_seq=cfg.max_seq))

    t_pre = timed(prefill)

    cache = KVCache(cfg, max_seq=cfg.max_seq)
    forward(model, tokens, cache)
    nxt = np.array([0])
    start = cache.length
    t_dec = timed(lambda: forward(model, nxt, cache), runs=decode_runs)
    # The cache grew while being timed, so model the step at the mean length.
    decode_total = (start + cache.length) // 2

    out = []
    for name, t, t_new, t_total in (
            ("tinyserve prefill", t_pre, TINY_PROMPT, TINY_PROMPT),
            ("tinyserve decode", t_dec, 1, decode_total)):
        f = flops_forward(cfg, t_new, t_total)
        b = bytes_read(cfg, model.n_params, t_new, t_total)
        out.append({"name": name, "kind": "model",
                    "flops": f, "bytes": b, "intensity": f / b,
                    "seconds": t, "achieved_flops": f / t})
    return out


def main() -> None:
    points = [matmul_point(n) for n in MATMUL_SIZES]
    points.append(stream_point())
    points += tinyserve_points()

    peak = max(p["achieved_flops"] for p in points)
    bandwidth = max(p["bytes"] / p["seconds"] for p in points
                    if p["kind"] == "memory")
    ridge = peak / bandwidth

    for p in points:
        roof = min(peak, p["intensity"] * bandwidth)
        p["roofline_flops"] = roof
        p["fraction_of_roof"] = p["achieved_flops"] / roof
        p["predicted_bound_by"] = "memory" if p["intensity"] < ridge else "compute"

    # The accelerator, where the same arithmetic decides the design.
    acc_decode = flops_forward(MODEL, 1, PROMPT_TOKENS + 300) / (
        WEIGHT_BYTES + (PROMPT_TOKENS + 299) * 2 * MODEL.n_layers
        * MODEL.n_kv_heads * MODEL.head_dim * 2)
    payload = {
        "machine": {"peak_flops": peak, "bandwidth_bytes_per_s": bandwidth,
                    "ridge_flop_per_byte": ridge},
        "points": points,
        "accelerator": {
            "peak_flops": PEAK_BF16_FLOPS, "bandwidth_bytes_per_s": HBM_BYTES_PER_S,
            "ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE,
            "decode_intensity": acc_decode,
            # Two operations per parameter per token, so this many tokens in
            # flight are needed before the arithmetic becomes the limit.
            # Weights-only arithmetic: how many tokens in flight WOULD reach
            # the ridge if each sequence's cache were free. It is not.
            "tokens_in_flight_if_cache_were_free": RIDGE_FLOP_PER_BYTE / 2,
            # Every sequence adds its own cached keys and values, so the
            # denominator grows with the batch too and intensity saturates.
            "intensity_ceiling": (2 * PARAMS) / ((PROMPT_TOKENS + 300)
                                                 * 2 * MODEL.n_layers
                                                 * MODEL.n_kv_heads
                                                 * MODEL.head_dim * 2),
            # The ceiling is 2*params / (context * KV bytes per token), so the
            # two ways to raise it are a smaller cache or a shorter context.
            "ceiling_variants": [
                {"what": "as served here", "factor": 1},
                {"what": "KV cache in one byte instead of two", "factor": 2},
                {"what": "half the context length", "factor": 2},
                {"what": "both", "factor": 4},
            ],
            "params": PARAMS,
            # Batching raises intensity: the weights are fetched once
            # whatever the batch, so more tokens ride the same fetch.
            "batching": [
                {"batch": b,
                 "intensity": serving.decode_step(b, PROMPT_TOKENS + 300).flops
                              / serving.decode_step(b, PROMPT_TOKENS + 300).bytes_read,
                 "achieved_flops": serving.decode_step(b, PROMPT_TOKENS + 300).flops
                                   / serving.decode_step(b, PROMPT_TOKENS + 300).seconds,
                 "bound_by": serving.decode_step(b, PROMPT_TOKENS + 300).bound_by}
                for b in (1, 8, 32, 64, 148, 256, 325)
            ],
        },
        "note": "the byte counts are compulsory traffic; caches serve some of "
                "it, so points above the roof are cache effects, not errors",
    }

    path = write("results/ch08.json", payload)
    print(f"wrote {path}")
    print(f"  this machine: peak {peak / 1e9:,.0f} GFLOP/s, "
          f"bandwidth {bandwidth / 1e9:.1f} GB/s, ridge {ridge:.0f} FLOP/byte")
    for p in points:
        print(f"  {p['name']:24} intensity {p['intensity']:8.2f}  "
              f"achieved {p['achieved_flops'] / 1e9:8.1f} GFLOP/s  "
              f"{p['fraction_of_roof'] * 100:5.0f}% of the bound  "
              f"({p['predicted_bound_by']}-bound)")
    a = payload["accelerator"]
    print(f"  accelerator ridge {a['ridge_flop_per_byte']:.0f} FLOP/byte")
    print(f"    tokens in flight to reach it, if the cache were free: "
          f"{a['tokens_in_flight_if_cache_were_free']:.0f}")
    print(f"    but intensity saturates at {a['intensity_ceiling']:.0f} FLOP/byte, "
          f"{a['ridge_flop_per_byte'] / a['intensity_ceiling']:.1f}x short of the ridge:")
    print("    batching alone can never make this workload compute-bound.")


if __name__ == "__main__":
    main()
