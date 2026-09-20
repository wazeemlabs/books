"""Chapter 12 experiment: what the KV cache is worth, and why.

Produces results/ch12.json, from which the chapter's tables and figures
are generated. Nothing in the chapter is typed in by hand.

    python3 -m bench.run_ch12
"""

from __future__ import annotations

from typing import Callable

from tinyserve.generate import Run, cached, check_equivalence, naive
from tinyserve.model import Config, KVCache, build

from .harness import Repeated, pct, write

PROMPT_LEN = 128
N_NEW = 128
SWEEP = [16, 32, 64, 128, 256]
WARMUP, RUNS = 2, 7      # head-to-head
SWEEP_WARMUP, SWEEP_RUNS = 1, 5  # the sweep repeats five generations per point
SEED_PROMPT = list(range(PROMPT_LEN))


# --- the arithmetic, so measurement can be checked against theory -------


def flops_forward(c: Config, t_new: int, t_total: int) -> int:
    """Multiply-add FLOPs for one forward pass over `t_new` tokens when the
    sequence is `t_total` long. Every matmul, counted as 2 FLOPs per MAC."""
    d, hd = c.d_model, c.head_dim
    per_layer = (
        2 * t_new * d * (c.n_heads * hd)             # Q projection
        + 2 * 2 * t_new * d * (c.n_kv_heads * hd)    # K and V projections
        + 2 * t_new * (c.n_heads * hd) * d           # output projection
        + 2 * 2 * t_new * d * c.d_ff                 # feed-forward, both matmuls
        + 2 * 2 * c.n_heads * t_new * t_total * hd   # scores, then values
    )
    return c.n_layers * per_layer + 2 * t_new * d * c.vocab_size


def flops_naive(c: Config, prompt: int, n_new: int) -> int:
    return sum(flops_forward(c, prompt + i, prompt + i) for i in range(n_new))


def flops_cached(c: Config, prompt: int, n_new: int) -> int:
    total = flops_forward(c, prompt, prompt)  # prefill
    return total + sum(flops_forward(c, 1, prompt + i) for i in range(1, n_new))


# --- measurement --------------------------------------------------------


def measure(fn: Callable[[], Run], warmup: int = WARMUP, runs: int = RUNS) -> list[Run]:
    for _ in range(warmup):
        fn()
    return [fn() for _ in range(runs)]


def measure_paired(fns: dict[str, Callable[[], Run]], warmup: int = WARMUP,
                   runs: int = RUNS) -> dict[str, list[Run]]:
    """Measure several variants interleaved, not one after the other.

    A shared machine drifts: the load it is under changes over seconds.
    Running all of A and then all of B lets that drift land entirely on
    one variant and show up as a difference between them. Alternating
    A, B, A, B cancels it, and lets each repeat be compared as a pair.
    """
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    out: dict[str, list[Run]] = {k: [] for k in fns}
    for _ in range(runs):
        for name, fn in fns.items():
            out[name].append(fn())
    return out


def paired_ratio(a: list[Run], b: list[Run]) -> dict:
    """Ratio of totals computed per repeat, then summarized.

    A ratio of two medians hides how much the ratio itself moves; a
    median of per-repeat ratios does not.
    """
    return Repeated([x.total_s / y.total_s for x, y in zip(a, b)]).summary()


def summarize(runs: list[Run]) -> dict:
    steps = [s for r in runs for s in r.step_s]  # pooled across repeats
    return {
        "ttft_s": Repeated([r.ttft_s for r in runs]).summary(),
        "total_s": Repeated([r.total_s for r in runs]).summary(),
        "decode_step_s": {"p50": pct(steps, 50), "p99": pct(steps, 99), "n": len(steps)},
        "decode_tok_per_s": Repeated([r.decode_tok_per_s for r in runs]).summary(),
    }


def main() -> None:
    cfg = Config()
    model = build(cfg)
    payload: dict = {
        "model": {
            "params": model.n_params,
            "config": {k: getattr(cfg, k) for k in
                       ("vocab_size", "d_model", "n_layers", "n_heads",
                        "n_kv_heads", "d_ff")},
            "head_dim": cfg.head_dim,
        },
        "experiment": {"prompt_len": PROMPT_LEN, "n_new": N_NEW,
                       "warmup": WARMUP, "runs": RUNS},
    }

    # 1. The cache must not change the output.
    payload["equivalence"] = {
        "identical_tokens": check_equivalence(model, SEED_PROMPT, 16),
        "note": "greedy decoding; the cache is an optimization, not an approximation",
    }

    # 2. Head-to-head at the reference length.
    runs = measure_paired({
        "naive": lambda: naive(model, SEED_PROMPT, N_NEW),
        "cached": lambda: cached(model, SEED_PROMPT, N_NEW),
    })
    head_to_head = {name: summarize(rs) for name, rs in runs.items()}
    ratio = paired_ratio(runs["naive"], runs["cached"])
    head_to_head["speedup_total"] = ratio["median"]
    head_to_head["speedup_total_stats"] = ratio
    head_to_head["speedup_decode_p50"] = (
        head_to_head["naive"]["decode_step_s"]["p50"]
        / head_to_head["cached"]["decode_step_s"]["p50"])
    payload["head_to_head"] = head_to_head

    # 3. How each one scales with the number of tokens generated.
    sweep = []
    for n in SWEEP:
        row = {"n_new": n}
        rs = measure_paired({"naive": lambda n=n: naive(model, SEED_PROMPT, n),
                             "cached": lambda n=n: cached(model, SEED_PROMPT, n)},
                            warmup=SWEEP_WARMUP, runs=SWEEP_RUNS)
        for name in ("naive", "cached"):
            r = Repeated([x.total_s for x in rs[name]])
            row[name + "_s"] = r.median
            row[name + "_noisy"] = r.noisy
        sp = paired_ratio(rs["naive"], rs["cached"])
        row["speedup"] = sp["median"]
        row["speedup_spread"] = sp["spread_frac"]
        row["naive_flops"] = flops_naive(cfg, PROMPT_LEN, n)
        row["cached_flops"] = flops_cached(cfg, PROMPT_LEN, n)
        row["flop_ratio"] = row["naive_flops"] / row["cached_flops"]
        sweep.append(row)
    payload["sweep"] = sweep

    # 4. Cost of each step by position -- the shape of the waste.
    long = max(SWEEP)
    payload["per_step"] = {
        "n_new": long,
        "naive_s": (lambda r: [r.ttft_s] + r.step_s)(naive(model, SEED_PROMPT, long)),
        "cached_s": (lambda r: [r.ttft_s] + r.step_s)(cached(model, SEED_PROMPT, long)),
    }

    # 5. What the cache costs in memory, here and at production scale.
    cache = KVCache(cfg, max_seq=PROMPT_LEN + N_NEW)
    ref = {"name": "Llama-3-style 8B", "n_layers": 32, "n_kv_heads": 8,
           "head_dim": 128, "bytes_per_element": 2}
    ref["bytes_per_token"] = (2 * ref["n_layers"] * ref["n_kv_heads"]
                              * ref["head_dim"] * ref["bytes_per_element"])
    payload["memory"] = {
        "tinyserve": {
            "bytes_per_token": cache.bytes_per_token(),
            "formula": "2 * n_layers * n_kv_heads * head_dim * bytes_per_element",
            "allocated_bytes": cache.nbytes,
            "used_at_end_bytes": cache.bytes_per_token() * (PROMPT_LEN + N_NEW),
        },
        "reference_8b": ref | {
            "kib_per_token": ref["bytes_per_token"] / 1024,
            "gib_per_8k_sequence": ref["bytes_per_token"] * 8192 / 1024**3,
        },
    }

    path = write("results/ch12.json", payload)
    print(f"wrote {path}")
    print(f"  params            {model.n_params:,}")
    print(f"  identical tokens  {payload['equivalence']['identical_tokens']}")
    print(f"  speedup (total)   {ratio['median']:.1f}x +/-{ratio['spread_frac']*100:.0f}%"
          f" at {PROMPT_LEN}+{N_NEW} tokens (paired, n={ratio['n']})")
    print(f"  decode p50        naive {head_to_head['naive']['decode_step_s']['p50']*1e3:.2f} ms"
          f" / cached {head_to_head['cached']['decode_step_s']['p50']*1e3:.2f} ms")
    print(f"  KV per token      {cache.bytes_per_token()} B (tinyserve),"
          f" {ref['bytes_per_token']/1024:.0f} KiB (8B reference)")


if __name__ == "__main__":
    main()
