"""Chapter 24: weights as integers.

Chapter 22 shrank numbers by narrowing their exponent and mantissa.
This shrinks them by taking the exponent away entirely: one scale for a
group of weights, and a small integer each.

Five measurements:

1. A worked example, small enough to check by hand.
2. What each scheme costs on the model's own weights: eight bits and
   four, symmetric and asymmetric, one scale per tensor, per channel
   and per group.
3. The outlier. A single large weight sets the scale for everything
   sharing it, and this is what that costs.
4. What it does to the model's answers, against the margin by which
   the model was making them. Quantization only changes a decision
   where its error exceeds the model's own confidence.
5. The memory, for the model the case study serves, counting the
   scales.

    python3 -m bench.run_ch24
"""

from __future__ import annotations

import numpy as np

from tinyserve import model as m, quantize as q
from tinyserve.precision import BF16, relative_error, round_to
from tinyserve.reference import MODEL, PARAMS, HBM_BYTES_PER_S, GPU_BYTES

from .harness import write

SEED = 0
CFG = m.Config()
PROMPT_TOKENS = 48
GROUPS = [128, 64, 32, 16]

SCHEMES = [
    q.Scheme(8), q.Scheme(8, symmetric=False), q.Scheme(8, axis=1),
    q.Scheme(4), q.Scheme(4, symmetric=False), q.Scheme(4, axis=1),
    q.Scheme(4, group=64), q.Scheme(4, group=32),
    q.Scheme(4, group=32, symmetric=False),
]


def worked_example() -> dict:
    """Eight numbers quantized by hand, so the arithmetic is visible."""
    x = np.array([[0.0, 0.10, -0.25, 0.80, -0.05, 0.40, -0.80, 0.15]],
                 dtype=np.float32)
    scheme = q.Scheme(4)
    codes, scale, zero = q.quantize(x, scheme)
    back = q.dequantize(codes, scale, zero, x.shape, scheme)
    return {
        "values": x.ravel().tolist(),
        "bits": scheme.bits,
        "largest": float(np.abs(x).max()),
        "qmax": scheme.qmax,
        "qmin": scheme.qmin,
        "scale": float(scale.ravel()[0]),
        "codes": codes.ravel().tolist(),
        "recovered": back.ravel().tolist(),
        "errors": (back - x).ravel().tolist(),
        "largest_error": float(np.abs(back - x).max()),
        "half_a_step": float(scale.ravel()[0]) / 2,
    }


def _weights(net) -> dict[str, np.ndarray]:
    """Every matrix a quantization recipe would touch."""
    out = {}
    for i, layer in enumerate(net.layers):
        for field in ("wq", "wk", "wv", "wo", "w1", "w2"):
            out[f"layer {i} {field}"] = getattr(layer, field)
    return out


def schemes() -> dict:
    """What each scheme costs, over every weight matrix in the model."""
    net = m.build(CFG, seed=SEED)
    mats = _weights(net)
    rows = []
    for scheme in SCHEMES:
        errs, worst = [], 0.0
        for w in mats.values():
            if scheme.group and w.shape[0] % scheme.group:
                errs = None
                break
            e = relative_error(w, q.round_trip(w, scheme))
            errs.append(e["rms_relative_to_largest"])
            worst = max(worst, e["max_relative_to_largest"])
        if errs is None:
            continue
        per_scale = (scheme.group if scheme.group
                     else (next(iter(mats.values())).shape[0]
                           if scheme.axis is not None
                           else next(iter(mats.values())).size))
        rows.append({
            "scheme": scheme.name, "bits": scheme.bits,
            "symmetric": scheme.symmetric,
            "granularity": ("group" if scheme.group else
                            "channel" if scheme.axis is not None else "tensor"),
            "group": scheme.group,
            "rms": float(np.mean(errs)),
            "worst": worst,
            "bytes_per_weight": scheme.bytes_per_weight(per_scale=per_scale),
            "values_per_scale": per_scale,
        })
    # bfloat16 for comparison: the format the book has been serving in.
    bf = [relative_error(w, round_to(w, BF16))["rms_relative_to_largest"]
          for w in mats.values()]
    return {"rows": rows, "matrices": len(mats),
            "bfloat16_rms": float(np.mean(bf)),
            "bfloat16_bytes_per_weight": 2.0}


def group_sweep() -> list[dict]:
    """How much a finer scale buys, and what it costs to store."""
    net = m.build(CFG, seed=SEED)
    mats = _weights(net)
    rows = []
    for g in GROUPS:
        scheme = q.Scheme(4, group=g)
        errs = [relative_error(w, q.round_trip(w, scheme))["rms_relative_to_largest"]
                for w in mats.values() if w.shape[0] % g == 0]
        rows.append({
            "group": g, "rms": float(np.mean(errs)),
            "bytes_per_weight": scheme.bytes_per_weight(per_scale=g),
            "overhead_pct": (scheme.bytes_per_weight(per_scale=g) / 0.5 - 1) * 100,
        })
    return rows


def outliers() -> dict:
    """What one large weight does to the others sharing its scale.

    Constructed, and said so. This book's model has Gaussian weights
    with no outliers to speak of; real trained models have channels
    whose values run far larger than the rest, which is the finding
    behind every per-channel and per-group recipe in the literature.
    The sweep below puts one in deliberately and measures what it
    costs, which is the honest way to show an effect a toy model does
    not exhibit.
    """
    rng = np.random.default_rng(SEED)
    base = (rng.standard_normal((256, 64)) * 0.1).astype(np.float32)
    typical = float(np.abs(base).max())
    rows = []
    for factor in (1, 3, 10, 30, 100):
        w = base.copy()
        w[0, 0] = typical * factor           # one weight, far out
        entry = {"outlier_factor": factor,
                 "outlier_value": float(w[0, 0])}
        for scheme in (q.Scheme(4), q.Scheme(4, axis=1), q.Scheme(4, group=32)):
            back = q.round_trip(w, scheme)
            # The error over everything *except* the outlier: what the
            # ordinary weights pay for its presence.
            mask = np.ones_like(w, dtype=bool)
            mask[0, 0] = False
            err = float(np.sqrt(np.mean((back[mask] - w[mask]) ** 2)))
            entry[scheme.name] = err / float(np.abs(w[mask]).max())
        rows.append(entry)
    return {"rows": rows, "typical_largest": typical,
            "shape": list(base.shape)}


def answers() -> dict:
    """What quantization does to the model's decisions, and when.

    Measured position by position on one prompt rather than by letting
    the model generate freely. Free generation compounds: one different
    token changes every token after it, so a single early flip makes
    two schemes look far apart when their arithmetic is nearly
    identical. Per position, on the same input, the comparison is
    between the schemes rather than between their histories.
    """
    net = m.build(CFG, seed=SEED)
    prompt = (np.arange(PROMPT_TOKENS) * 7) % CFG.vocab_size
    base = m.forward(net, prompt)
    top2 = np.sort(base, axis=-1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    rows = []
    for scheme in SCHEMES:
        if scheme.group and CFG.d_model % scheme.group:
            continue
        logits = m.forward(q.quantize_model(net, scheme), prompt)
        shift = np.abs(logits - base).max(axis=-1)
        flipped = logits.argmax(-1) != base.argmax(-1)
        rows.append({
            "scheme": scheme.name,
            "median_shift": float(np.median(shift)),
            "shift_over_margin": float(np.median(shift / margin)),
            "positions_where_shift_exceeds_margin": float(np.mean(shift > margin)),
            "positions_that_changed": float(np.mean(flipped)),
        })
    return {
        "rows": rows,
        "positions": int(prompt.size),
        "margin_median": float(np.median(margin)),
        "margin_p10": float(np.percentile(margin, 10)),
        "margin_max": float(margin.max()),
    }


def memory() -> dict:
    """The reference model's weights under each scheme."""
    rows = []
    for scheme in SCHEMES:
        per_scale = scheme.group or (MODEL.d_model if scheme.axis is not None
                                     else PARAMS)
        b = scheme.bytes_per_weight(per_scale=per_scale)
        total = PARAMS * b
        rows.append({
            "scheme": scheme.name,
            "bytes_per_weight": b,
            "weights_gb": total / 1e9,
            "read_ms": total / HBM_BYTES_PER_S * 1e3,
            "over_bf16": (PARAMS * 2) / total,
            "free_for_cache_gb": (GPU_BYTES - total) / 1e9,
        })
    bf16_total = PARAMS * 2
    return {"rows": rows, "params": PARAMS,
            "bf16_gb": bf16_total / 1e9,
            "bf16_read_ms": bf16_total / HBM_BYTES_PER_S * 1e3,
            "bf16_free_gb": (GPU_BYTES - bf16_total) / 1e9,
            "gpu_gb": GPU_BYTES / 1e9}


def main() -> None:
    ex, sc, gs = worked_example(), schemes(), group_sweep()
    out, ans, mem = outliers(), answers(), memory()
    payload = {
        "worked": ex, "schemes": sc, "groups": gs,
        "outliers": out, "answers": ans, "memory": mem,
        "assumptions": {
            "seed": SEED, "prompt_tokens": PROMPT_TOKENS,
            "groups": GROUPS,
            "model": {"layers": CFG.n_layers, "d_model": CFG.d_model,
                      "heads": CFG.n_heads},
            "reference_params": PARAMS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch24.json", payload)
    print(f"wrote {path}")
    print(f"  worked example: scale {ex['scale']:.5f}, codes {ex['codes']}, "
          f"largest error {ex['largest_error']:.4f} "
          f"(half a step is {ex['half_a_step']:.4f})")
    print(f"  over {sc['matrices']} weight matrices "
          f"(bfloat16 for comparison: {sc['bfloat16_rms']:.2e} at "
          f"{sc['bfloat16_bytes_per_weight']} bytes a weight):")
    for r in sc["rows"]:
        print(f"    {r['scheme']:42}{r['rms']:10.2e}  "
              f"{r['bytes_per_weight']:.3f} bytes/weight")
    print("  finer groups:")
    for r in gs:
        print(f"    group {r['group']:4d}: {r['rms']:.2e}  "
              f"{r['bytes_per_weight']:.3f} bytes/weight "
              f"({r['overhead_pct']:+.0f}% over four bits)")
    print(f"  one outlier among {out['shape'][0]}x{out['shape'][1]} weights, "
          f"error on everything else:")
    for r in out["rows"]:
        print(f"    {r['outlier_factor']:4}x typical: " + "  ".join(
            f"{k.split(', ')[-1]} {r[k]:.3e}" for k in r
            if k.startswith("int4")))
    print(f"  the model's top-2 margin: median {ans['margin_median']:.3f}, "
          f"p10 {ans['margin_p10']:.3f}")
    for r in ans["rows"]:
        print(f"    {r['scheme']:42} shift {r['median_shift']:7.4f}  "
              f"over margin {r['positions_where_shift_exceeds_margin']*100:5.1f}%"
              f"  changed {r['positions_that_changed']*100:5.1f}%")
    print(f"  the {mem['params']/1e9:.0f}B model "
          f"({mem['bf16_gb']:.0f} GB in bfloat16):")
    for r in mem["rows"]:
        print(f"    {r['scheme']:42}{r['weights_gb']:7.1f} GB  "
              f"{r['read_ms']:6.2f} ms  {r['over_bf16']:.2f}x smaller  "
              f"{r['free_for_cache_gb']:.0f} GB left for cache")


if __name__ == "__main__":
    main()
