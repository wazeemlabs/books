"""Chapter 3: why prefill and decode behave like two different programs.

Two parts, kept separate because they have different standing:

1. MEASURED on tinyserve: how long it takes to read a prompt versus to
   write a token, across prompt lengths.
2. ARITHMETIC for a production 8B, labelled as such: how much work each
   phase does per byte it fetches, against the hardware's break-even
   point. This is where the measured gap comes from.

    python3 -m bench.run_ch03
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from tinyserve.cost import bytes_read, flops_forward, intensity
from tinyserve.model import Config, KVCache, build, forward

from tinyserve.reference import (CONTEXT_TOKENS as REF_CONTEXT,
                                 ELEM_BYTES as REF_ELEM, HBM_BYTES_PER_S,
                                 KV_BYTES_PER_TOKEN, MODEL as REF,
                                 PARAMS as REF_PARAMS, PEAK_BF16_FLOPS,
                                 PROMPT_TOKENS as REF_PROMPT,
                                 RIDGE_FLOP_PER_BYTE as RIDGE)

from .harness import Repeated, pct, write

PROMPTS = [64, 128, 256, 512, 1024]
DECODE_STEPS = 24
WARMUP, RUNS = 1, 5



def time_prefill(model, cfg, p: int) -> float:
    tokens = np.arange(p) % cfg.vocab_size
    cache = KVCache(cfg, max_seq=p + DECODE_STEPS + 1)
    t0 = perf_counter()
    forward(model, tokens, cache)
    return perf_counter() - t0


def time_decode(model, cfg, p: int) -> list[float]:
    tokens = np.arange(p) % cfg.vocab_size
    cache = KVCache(cfg, max_seq=p + DECODE_STEPS + 1)
    forward(model, tokens, cache)  # prefill first; not timed
    steps = []
    nxt = np.array([0])
    for _ in range(DECODE_STEPS):
        t0 = perf_counter()
        forward(model, nxt, cache)
        steps.append(perf_counter() - t0)
    return steps


def main() -> None:
    cfg = Config(max_seq=2048)
    model = build(cfg)

    rows = []
    for p in PROMPTS:
        for _ in range(WARMUP):
            time_prefill(model, cfg, p); time_decode(model, cfg, p)
        pre = Repeated([time_prefill(model, cfg, p) for _ in range(RUNS)])
        steps = [s for _ in range(RUNS) for s in time_decode(model, cfg, p)]

        per_token_prefill = pre.median / p
        decode_p50 = pct(steps, 50)
        rows.append({
            "prompt": p,
            "prefill_s": pre.median,
            "prefill_noisy": pre.noisy,
            "prefill_per_token_s": per_token_prefill,
            "prefill_tokens_per_s": p / pre.median,
            "decode_step_p50_s": decode_p50,
            "decode_step_p99_s": pct(steps, 99),
            "decode_tokens_per_s": 1 / decode_p50,
            "decode_vs_prefill_per_token": decode_p50 / per_token_prefill,
            "prefill_intensity": intensity(cfg, model.n_params, p, p),
            "decode_intensity": intensity(cfg, model.n_params, 1, p + 1),
        })

    # The same arithmetic for a production model, where it decides the design.
    def ref(t_new: int, t_total: int) -> dict:
        f = flops_forward(REF, t_new, t_total)
        b = REF_PARAMS * REF_ELEM + max(0, t_total - t_new) * KV_BYTES_PER_TOKEN
        t = max(f / PEAK_BF16_FLOPS, b / HBM_BYTES_PER_S)
        return {"flops": f, "bytes": b, "intensity": f / b, "seconds": t,
                "bound_by": "compute" if f / b > RIDGE else "memory",
                "distance_from_ridge": (f / b) / RIDGE}

    reference = {
        "config": {"params": REF_PARAMS, "elem_bytes": REF_ELEM,
                   "prompt": REF_PROMPT, "context": REF_CONTEXT},
        "hardware": {"hbm_bytes_per_s": HBM_BYTES_PER_S,
                     "peak_bf16_flops": PEAK_BF16_FLOPS, "ridge_flop_per_byte": RIDGE},
        "prefill": ref(REF_PROMPT, REF_PROMPT),
        "decode": ref(1, REF_CONTEXT),
        "note": "arithmetic over published specs, not a measurement",
    }
    reference["intensity_ratio"] = (reference["prefill"]["intensity"]
                                    / reference["decode"]["intensity"])
    # Work per byte and time per token are different ratios, and confusing
    # them overstates the gap. Record both.
    reference["prefill_per_token_s"] = reference["prefill"]["seconds"] / REF_PROMPT
    reference["time_per_token_ratio"] = (reference["decode"]["seconds"]
                                         / reference["prefill_per_token_s"])
    # One whole request, so the two phases can be compared on a clock.
    n_out = 300
    pre_s, dec_s = reference["prefill"]["seconds"], reference["decode"]["seconds"]
    reference["one_request"] = {
        "output_tokens": n_out, "prefill_s": pre_s, "decode_s": dec_s * n_out,
        "total_s": pre_s + dec_s * n_out,
        "decode_share": dec_s * n_out / (pre_s + dec_s * n_out),
    }

    payload = {
        "model": {"params": model.n_params,
                  "config": {k: getattr(cfg, k) for k in
                             ("vocab_size", "d_model", "n_layers", "n_heads",
                              "n_kv_heads", "d_ff")},
                  "head_dim": cfg.head_dim},
        "experiment": {"prompts": PROMPTS, "decode_steps": DECODE_STEPS,
                       "runs": RUNS, "warmup": WARMUP},
        "measured": rows,
        "reference_8b": reference,
    }

    path = write("results/ch03.json", payload)
    print(f"wrote {path}")
    for r in rows:
        print(f"  prompt {r['prompt']:5d}  prefill {r['prefill_tokens_per_s']:9,.0f} tok/s"
              f"  decode {r['decode_tokens_per_s']:7,.0f} tok/s"
              f"  per-token gap {r['decode_vs_prefill_per_token']:6.1f}x")
    p_, d_ = reference["prefill"], reference["decode"]
    print(f"  8B reference: prefill {p_['intensity']:8.1f} FLOP/byte ({p_['bound_by']}),"
          f" decode {d_['intensity']:.2f} ({d_['bound_by']}),"
          f" ridge {RIDGE:.0f}")


if __name__ == "__main__":
    main()
