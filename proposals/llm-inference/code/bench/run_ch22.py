"""Chapter 22: what a number format costs, and what it buys.

Every number in this book so far has been a float32 or a bf16 and no
chapter has asked what that means. Five measurements that do:

1. The formats, from their field widths. What each can represent, how
   precisely, and where it stops.
2. What rounding does to the model itself -- its weights, its
   activations, its attention scores, its logits.
3. Accumulation. A tensor core multiplies in a small format and adds
   up in a large one, and this measures what happens when it does not.
4. Range. Where float16 stops being able to hold the answer and
   bfloat16 carries on, which is the whole reason the newer format won
   despite being the coarser of the two.
5. The hardware, and what the choice is worth: peak arithmetic by
   format, bytes for the reference model, and where the roofline's
   ridge moves.

Nothing here needs a GPU. Rounding is exact arithmetic on bit fields,
and the hardware figures are published specifications recorded in
FACTS.md.

    python3 -m bench.run_ch22
"""

from __future__ import annotations

import numpy as np

from tinyserve import model as m, precision as pr
from tinyserve.reference import (HBM_BYTES_PER_S, MODEL, PARAMS,
                                 PEAK_BF16_FLOPS, PROMPT_TOKENS,
                                 CONTEXT_TOKENS)

from .harness import write

SEED = 0
# Peak arithmetic on one H100 SXM, by format (FACTS.md, 2026-09-21).
# The product page quotes every tensor-core figure "with sparsity";
# these are the dense halves, which is what dense inference gets.
# Anchored on the book's bf16 peak so every chapter agrees: the
# datasheet's "1,979 teraFLOPS * With sparsity" is 989.5 dense, which is
# the ~990 the rest of the book uses. The others keep the datasheet's
# ratios to it -- TF32 half, FP8 double -- and FP32 is a separate number
# because it does not touch a tensor core at all.
PEAK_FLOPS = {
    "float32": 67e12,            # CUDA cores
    "tensorfloat32": PEAK_BF16_FLOPS / 2,
    "float16": PEAK_BF16_FLOPS,
    "bfloat16": PEAK_BF16_FLOPS,
    "float8 e4m3": PEAK_BF16_FLOPS * 2,
    "float8 e5m2": PEAK_BF16_FLOPS * 2,
}
ACC_LENGTHS = [128, 512, 2048, 8192]
ACC_ROWS = ACC_COLS = 64
RANGE_SCALES = [1, 10, 30, 100, 300, 1000]


def formats() -> list[dict]:
    """Each format's reach and resolution, from its field widths alone."""
    rows = []
    for f in pr.FORMATS:
        rows.append({
            "name": f.name,
            "bits": f.bits,
            "exponent_bits": f.exponent_bits,
            "mantissa_bits": f.mantissa_bits,
            "max_value": f.max_value,
            "min_normal": f.min_normal,
            "min_subnormal": f.min_subnormal,
            "eps": f.eps,
            "decimal_digits": f.decimal_digits,
            "has_infinity": f.has_infinity,
            "storage_bytes": f.bytes,
            "compute_only": f.compute_only,
            "orders_of_magnitude": float(np.log10(f.max_value)
                                         - np.log10(f.min_normal)),
            "peak_tflops": PEAK_FLOPS[f.name] / 1e12,
            "over_float32_flops": PEAK_FLOPS[f.name] / PEAK_FLOPS["float32"],
        })
    return rows


def verification() -> dict:
    """Check the hand-written rounding against the one format NumPy has.

    Recorded rather than asserted: the chapter says its arithmetic is
    trustworthy because of this comparison, so the comparison is a
    result with a number attached like any other.
    """
    rng = np.random.default_rng(SEED)
    values = np.concatenate([
        rng.standard_normal(200_000).astype(np.float32),
        (rng.standard_normal(200_000) * 1e4).astype(np.float32),
        (rng.standard_normal(200_000) * 1e-5).astype(np.float32),
        # Exact halfway cases, where the rounding rule is the only thing
        # that decides the answer.
        (1.0 + np.arange(0, 1024) / 1024 + 1 / 2048).astype(np.float32),
    ])
    mine = pr.round_to(values, pr.FP16)
    with np.errstate(over="ignore"):
        theirs = values.astype(np.float16).astype(np.float32)
    agree = int(np.sum((mine == theirs)
                       | (np.isnan(mine) & np.isnan(theirs))))
    return {
        "values_compared": int(values.size),
        "values_agreeing": agree,
        "halfway_cases": 1024,
        "all_agree": agree == values.size,
    }


def model_tensors() -> dict:
    """What rounding does to the numbers the model is actually made of."""
    net = m.build(m.Config(), seed=SEED)
    tokens = np.arange(64) % net.cfg.vocab_size
    trace: dict = {}
    m.forward(net, tokens, trace=trace)
    groups = {
        "attention weights (wq)": net.layers[0].wq,
        "feed-forward weights (w1)": net.layers[0].w1,
        "token embeddings": net.tok_emb,
        "activations, after a layer": trace["layers"][0]["after_attention"],
        "attention weights, after softmax":
            trace["layers"][0]["attention_weights"],
        "logits": trace["logits"],
    }
    rows = []
    for name, tensor in groups.items():
        x = np.asarray(tensor, dtype=np.float32)
        entry = {"tensor": name, "count": int(x.size),
                 "largest": float(np.abs(x).max()),
                 "smallest_nonzero": float(np.abs(x[x != 0]).min())
                 if np.any(x != 0) else 0.0}
        for f in pr.FORMATS[1:]:
            e = pr.relative_error(x, pr.round_to(x, f))
            entry[f.name] = e["rms_relative_to_largest"]
            entry[f"{f.name} overflowed"] = e["overflowed"]
            entry[f"{f.name} to zero"] = e["flushed_to_zero"]
        rows.append(entry)
    return {"rows": rows, "config": {
        "layers": net.cfg.n_layers, "d_model": net.cfg.d_model,
        "heads": net.cfg.n_heads, "tokens": int(tokens.size)}}


def _matmul_narrow(a: np.ndarray, b: np.ndarray, fmt: pr.Format) -> np.ndarray:
    """A matrix multiply whose running total is kept in `fmt`.

    Written as a loop over the shared dimension, rounding the whole
    output after each step, because that is what accumulating in a
    format means: every partial sum has to fit in it.
    """
    acc = np.zeros((a.shape[0], b.shape[1]), dtype=np.float32)
    for i in range(a.shape[1]):
        acc = pr.round_to(acc + np.outer(a[:, i], b[i, :]).astype(np.float32),
                          fmt)
    return acc


def accumulation() -> dict:
    """What a tensor core's wide accumulator is worth.

    The inputs are rounded to a small format either way. The only
    difference is where the running total lives: in float32, as the
    hardware keeps it, or in the small format itself.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for fmt in (pr.BF16, pr.FP16, pr.FP8_E4M3):
        for k in ACC_LENGTHS:
            a = (rng.standard_normal((ACC_ROWS, k)) * 0.1).astype(np.float32)
            b = (rng.standard_normal((k, ACC_COLS)) * 0.1).astype(np.float32)
            exact = a.astype(np.float64) @ b.astype(np.float64)
            scale = float(np.sqrt(np.mean(exact ** 2)))
            ar, br = pr.round_to(a, fmt), pr.round_to(b, fmt)
            wide = (ar.astype(np.float32) @ br.astype(np.float32))
            narrow = _matmul_narrow(ar, br, fmt)

            def rms(x: np.ndarray) -> float:
                return float(np.sqrt(np.mean(
                    (x.astype(np.float64) - exact) ** 2))) / scale

            w, n = rms(wide), rms(narrow)
            rows.append({"format": fmt.name, "k": k,
                         "wide_accumulator": w, "narrow_accumulator": n,
                         "ratio": n / w if w else float("inf"),
                         "rows": ACC_ROWS, "cols": ACC_COLS})
    return {"rows": rows, "note_rows": ACC_ROWS, "note_cols": ACC_COLS}


def range_failure() -> dict:
    """Where float16 runs out of room and bfloat16 does not.

    A dot product across the reference model's width, with inputs at a
    range of scales. Real models do produce activations far larger than
    this book's small one: the point of the sweep is to find the scale
    at which each format stops working, not to claim a particular model
    reaches it.
    """
    k = MODEL.d_model
    rng = np.random.default_rng(SEED)
    rows = []
    for s in RANGE_SCALES:
        a = (rng.standard_normal(k) * s).astype(np.float32)
        b = (rng.standard_normal(k) * s).astype(np.float32)
        exact = float(np.dot(a.astype(np.float64), b.astype(np.float64)))
        biggest_term = float(np.abs(a * b).max())
        entry = {"input_scale": s, "exact": exact,
                 "largest_product": biggest_term}
        for fmt in (pr.FP16, pr.BF16):
            ar, br = pr.round_to(a, fmt), pr.round_to(b, fmt)
            total = np.float32(0.0)
            blew_up = False
            for x, y in zip(ar, br):
                total = pr.round_to(np.array([total + x * y], np.float32),
                                    fmt)[0]
                if not np.isfinite(total):
                    blew_up = True
                    break
            entry[f"{fmt.name} holds the products"] = (
                biggest_term <= fmt.max_value)
            entry[f"{fmt.name} accumulator"] = (
                None if blew_up else float(total))
            entry[f"{fmt.name} overflowed"] = blew_up
        rows.append(entry)
    first_bad = next((r for r in rows if r["float16 overflowed"]), None)
    return {
        "rows": rows, "k": k,
        "fp16_max": pr.FP16.max_value,
        "bf16_max": pr.BF16.max_value,
        "first_scale_fp16_fails": first_bad["input_scale"] if first_bad else None,
        "bf16_ever_fails": any(r["bfloat16 overflowed"] for r in rows),
    }


def hardware() -> dict:
    """What the choice is worth on the accelerator the book serves on."""
    rows = []
    for f in pr.FORMATS:
        peak = PEAK_FLOPS[f.name]
        weight_bytes = PARAMS * f.bytes
        kv_per_token = (2 * MODEL.n_layers * MODEL.n_kv_heads
                        * MODEL.head_dim * f.bytes)
        rows.append({
            "name": f.name,
            "bytes_per_number": f.bytes,
            "peak_tflops": peak / 1e12,
            "weight_gb": weight_bytes / 1e9,
            "weight_read_ms": weight_bytes / HBM_BYTES_PER_S * 1e3,
            "kv_bytes_per_token": kv_per_token,
            "kv_gb_at_context": kv_per_token * CONTEXT_TOKENS / 1e9,
            "ridge_flops_per_byte": peak / HBM_BYTES_PER_S,
            # The batch at which a decode step stops being limited by
            # memory. A step at batch B does 2*P*B arithmetic and reads
            # P*bytes of weights, so its intensity is 2B/bytes: halving
            # the format doubles it. The ridge doubles too, and the two
            # cancel, which is the point the chapter makes.
            "compute_bound_above_batch": (peak / HBM_BYTES_PER_S
                                          * f.bytes / 2),
            "prefill_ms_at_peak": (2 * PARAMS * PROMPT_TOKENS) / peak * 1e3,
        })
    return {"rows": rows, "hbm_bytes_per_s": HBM_BYTES_PER_S,
            "params": PARAMS, "prompt_tokens": PROMPT_TOKENS,
            "context_tokens": CONTEXT_TOKENS}


def main() -> None:
    fmts, tensors = formats(), model_tensors()
    acc, rng_fail, hw = accumulation(), range_failure(), hardware()
    payload = {
        "formats": fmts,
        "verification": verification(),
        "tensors": tensors,
        "accumulation": acc,
        "range": rng_fail,
        "hardware": hw,
        "assumptions": {
            "seed": SEED,
            "accumulation_lengths": ACC_LENGTHS,
            "accumulation_shape": [ACC_ROWS, ACC_COLS],
            "range_scales": RANGE_SCALES,
            "peak_flops_are_dense": True,
            "model": {"params": PARAMS, "d_model": MODEL.d_model,
                      "layers": MODEL.n_layers},
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch22.json", payload)
    print(f"wrote {path}")
    print(f"  {'format':16}{'bits':>5}{'max':>13}{'eps':>11}"
          f"{'digits':>8}{'TFLOP/s':>10}")
    for r in fmts:
        print(f"  {r['name']:16}{r['bits']:5d}{r['max_value']:13.4g}"
              f"{r['eps']:11.3g}{r['decimal_digits']:8.1f}"
              f"{r['peak_tflops']:10.0f}")
    print("  rounding the model's own numbers (rms, relative to the largest):")
    for r in tensors["rows"]:
        print(f"    {r['tensor']:34}" + "".join(
            f"{r[f.name]:11.2e}" for f in pr.FORMATS[1:]))
    print("  accumulating in the small format instead of float32:")
    for r in acc["rows"]:
        print(f"    {r['format']:12} k={r['k']:5d}: "
              f"{r['wide_accumulator']:.2e} -> {r['narrow_accumulator']:.2e}"
              f"   {r['ratio']:6.1f}x worse")
    print(f"  a dot product across {rng_fail['k']:,} terms:")
    for r in rng_fail["rows"]:
        f16 = "overflowed" if r["float16 overflowed"] else f"{r['float16 accumulator']:.3g}"
        b16 = "overflowed" if r["bfloat16 overflowed"] else f"{r['bfloat16 accumulator']:.3g}"
        print(f"    inputs at {r['input_scale']:5}: exact {r['exact']:12.4g}"
              f"   float16 {f16:>12}   bfloat16 {b16:>12}")
    print("  the reference model, by format:")
    for r in hw["rows"]:
        print(f"    {r['name']:16}{r['weight_gb']:7.1f} GB weights, "
              f"{r['weight_read_ms']:6.2f} ms to read, "
              f"{r['kv_bytes_per_token'] / 1024:6.0f} KiB/token, "
              f"ridge {r['ridge_flops_per_byte']:6.0f} FLOP/byte")


if __name__ == "__main__":
    main()
