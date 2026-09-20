"""Chapter 20: the attention kernel, and the matrix it does not build.

Five measurements:

1. Equivalence. The tiled kernel is swapped into `tinyserve` and must
   generate the same tokens from the same prompt, at every tile size.
2. Traffic. The bytes each kernel moves between the two levels of the
   memory hierarchy, counted as they move, over a range of prompt
   lengths -- and the same bytes predicted by a formula, so that the
   formula can then be applied to a model too large to run here.
3. The reference model. That formula on the case study's 8B model, at
   the prompt length the case study actually serves.
4. Tiles. What the tile size costs in traffic and what it demands of
   the fast memory, against an H100's 228 KB per streaming
   multiprocessor.
5. Phases. The same comparison at prefill and at decode, which is why
   this chapter belongs to prefill.

Nothing here runs on a GPU. The byte counts are properties of the
algorithm, not of the hardware, and the times are the roofline of
Chapter 16 applied to them.

    python3 -m bench.run_ch20
"""

from __future__ import annotations

import numpy as np

from tinyserve import model as m
from tinyserve.flash import (Meter, attention_tiled, causal_mask,
                             score_matrix_bytes, tiled_traffic_bytes,
                             whole_traffic_bytes)
from tinyserve.model import DType, attention_whole
from tinyserve.reference import (ELEM_BYTES, HBM_BYTES_PER_S, MODEL,
                                 PROMPT_TOKENS, CONTEXT_TOKENS,
                                 REQUESTS_PER_S)

from .harness import write

ELEM = np.dtype(DType).itemsize

# The shape the traffic is measured on: one layer of a model small
# enough that both kernels can actually be run and counted.
HEADS, DIM = 8, 64
LENGTHS = [128, 256, 512, 1024, 2048]
TILES = [16, 32, 64, 128, 256]
Q_TILE = KV_TILE = 64           # what the rest of the chapter uses

# An H100 streaming multiprocessor's shared memory (FACTS.md, 2026-09).
SRAM_BYTES_PER_SM = 228 * 1024
SRAM_USABLE_PER_BLOCK = 227 * 1024
SEED = 0


def inputs(heads: int, new: int, seen: int, dim: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((heads, new, dim)).astype(DType)
    k = rng.standard_normal((heads, seen, dim)).astype(DType)
    v = rng.standard_normal((heads, seen, dim)).astype(DType)
    return q, k, v, causal_mask(new, seen)


def equivalence() -> dict:
    """The kernel is a seam: same prompt, same weights, same tokens."""
    cfg = m.Config(vocab_size=256, d_model=128, n_layers=4, n_heads=8,
                   n_kv_heads=4, d_ff=512, max_seq=512)
    net = m.build(cfg, seed=SEED)
    prompt = (np.arange(200) * 13 % cfg.vocab_size)
    n_new = 32

    def run(attention, **kw) -> tuple[list[int], np.ndarray]:
        cache = m.KVCache(cfg)
        fn = (lambda q, k, v, mask: attention(q, k, v, mask, **kw)) if kw else attention
        logits = m.forward(net, prompt, cache, attention=fn)
        out, last = [], logits
        for _ in range(n_new):
            nxt = int(last[-1].argmax())
            out.append(nxt)
            last = m.forward(net, np.array([nxt]), cache, attention=fn)
        return out, last

    base, base_logits = run(attention_whole)
    rows, same_all = [], True
    for q_tile, kv_tile in ((1, 1), (16, 16), (37, 64), (64, 64), (512, 512)):
        toks, logits = run(attention_tiled, q_tile=q_tile, kv_tile=kv_tile)
        same = toks == base
        same_all &= same
        gap = float(np.abs(base_logits - logits).max())
        rows.append({"q_tile": q_tile, "kv_tile": kv_tile,
                     "same_tokens": same, "max_logit_diff": gap,
                     "relative": gap / float(np.abs(base_logits).max())})

    # One prefill, to compare the outputs of the two kernels directly.
    q, k, v, mask = inputs(HEADS, 512, 512, DIM)
    whole, weights = attention_whole(q, k, v, mask)
    tiled, none = attention_tiled(q, k, v, mask, q_tile=Q_TILE, kv_tile=KV_TILE)
    return {
        "prompt_tokens": int(prompt.size),
        "tokens_generated": n_new,
        "same_tokens": same_all,
        "tile_sizes": rows,
        "attention_max_diff": float(np.abs(whole - tiled).max()),
        "attention_scale": float(np.abs(whole).max()),
        "float32_eps": float(np.finfo(DType).eps),
        "weights_from_whole": list(weights.shape),
        "weights_from_tiled": none,
        "config": {"heads": HEADS, "dim": DIM, "prefill": 512},
    }


def traffic() -> list[dict]:
    """Bytes moved by each kernel, counted as they move."""
    rows = []
    for n in LENGTHS:
        q, k, v, mask = inputs(HEADS, n, n, DIM)
        whole, tiled = Meter(), Meter()
        attention_whole(q, k, v, mask, whole)
        attention_tiled(q, k, v, mask, tiled, q_tile=Q_TILE, kv_tile=KV_TILE)
        predicted_whole = whole_traffic_bytes(HEADS, n, n, DIM, ELEM)
        predicted_tiled = tiled_traffic_bytes(HEADS, n, n, DIM, Q_TILE,
                                              KV_TILE, ELEM)
        rows.append({
            "tokens": n,
            "whole_bytes": whole.total,
            "tiled_bytes": tiled.total,
            "whole_predicted": predicted_whole,
            "tiled_predicted": predicted_tiled,
            "formulas_agree": (whole.total == predicted_whole
                               and tiled.total == predicted_tiled),
            "saving": 1 - tiled.total / whole.total,
            "ratio": whole.total / tiled.total,
            "whole_largest_intermediate": whole.largest_intermediate,
            "tiled_largest_intermediate": tiled.largest_intermediate,
            "intermediate_ratio": (whole.largest_intermediate
                                   / tiled.largest_intermediate),
            "score_matrix_bytes": score_matrix_bytes(HEADS, n, n, ELEM),
            "tiles": tiled.tiles,
            "tiles_skipped": tiled.tiles_skipped,
            "skipped_share": tiled.tiles_skipped / tiled.tiles,
        })
    return rows


def reference() -> dict:
    """The formula on the case study's model, which cannot run here.

    Every quantity is the verified formula from `tinyserve.flash`
    applied to the 8B reference model in bf16. The times are Chapter
    16's roofline: bytes over the accelerator's memory bandwidth.
    """
    c = MODEL
    rows = []
    for n in (128, 512, PROMPT_TOKENS, 2048, 4096, 8192):
        square = score_matrix_bytes(c.n_heads, n, n, ELEM_BYTES)
        qkv = ((c.n_heads + 2 * c.n_kv_heads) * n * c.head_dim * ELEM_BYTES)
        whole = whole_traffic_bytes(c.n_heads, n, n, c.head_dim, ELEM_BYTES,
                                    kv_heads=c.n_kv_heads)
        tiled = tiled_traffic_bytes(c.n_heads, n, n, c.head_dim, Q_TILE,
                                    KV_TILE, ELEM_BYTES, kv_heads=c.n_kv_heads)
        # The same kernel with its block of scores kept in the
        # scratchpad, which is what a CUDA implementation does and this
        # NumPy one cannot. The pair brackets the real figure.
        resident = tiled_traffic_bytes(c.n_heads, n, n, c.head_dim, Q_TILE,
                                       KV_TILE, ELEM_BYTES,
                                       kv_heads=c.n_kv_heads,
                                       block_crosses=False)
        rows.append({
            "tokens": n,
            "score_matrix_bytes": square,
            "qkv_bytes": qkv,
            "kv_bytes": 2 * c.n_kv_heads * n * c.head_dim * ELEM_BYTES,
            "square_over_inputs": square / qkv,
            "whole_bytes": whole,
            "tiled_bytes": tiled,
            "tiled_bytes_block_resident": resident,
            "saving": 1 - tiled / whole,
            "ratio": whole / tiled,
            "ratio_block_resident": whole / resident,
            "block_bytes": tiled - resident,
            "whole_ms_per_layer": whole / HBM_BYTES_PER_S * 1e3,
            "tiled_ms_per_layer": tiled / HBM_BYTES_PER_S * 1e3,
            "whole_ms_all_layers": whole * c.n_layers / HBM_BYTES_PER_S * 1e3,
            "tiled_ms_all_layers": tiled * c.n_layers / HBM_BYTES_PER_S * 1e3,
            "score_matrix_gb_all_layers": square * c.n_layers / 1e9,
        })
    # Where the score matrix stops being smaller than its own inputs:
    # heads*n*n == (heads + 2*kv_heads)*n*dim, solved for n.
    crossover = ((c.n_heads + 2 * c.n_kv_heads) * c.head_dim) / c.n_heads
    return {"rows": rows, "crossover_tokens": crossover,
            "crossover_over_head_dim": crossover / c.head_dim,
            "layers": c.n_layers, "heads": c.n_heads,
            "kv_heads": c.n_kv_heads, "head_dim": c.head_dim,
            "elem_bytes": ELEM_BYTES,
            "hbm_bytes_per_s": HBM_BYTES_PER_S}


def tiles() -> dict:
    """What a tile size costs in traffic, and demands of fast memory.

    A kernel holds one query tile, one key tile, one value tile and the
    block of scores between them. That is what has to fit in the memory
    attached to a core, and it is what bounds the tile size from above;
    traffic bounds it from below.
    """
    n = 2048
    rows = []
    for t in TILES:
        q, k, v, mask = inputs(HEADS, n, n, DIM)
        meter = Meter()
        attention_tiled(q, k, v, mask, meter, q_tile=t, kv_tile=t)
        # Fast-memory footprint for the reference model's head shape, in
        # bf16: the three tiles plus the block of scores between them.
        d = MODEL.head_dim
        resident = (3 * t * d * ELEM_BYTES                  # q, k and v tiles
                    + t * t * ELEM_BYTES                    # the score block
                    + t * d * 4                             # the accumulator
                    + 2 * t * 4)                            # running max, sum
        rows.append({
            "tile": t,
            "bytes": meter.total,
            "tiles_computed": meter.tiles - meter.tiles_skipped,
            "tiles_skipped": meter.tiles_skipped,
            "sram_bytes_reference_model": resident,
            "sram_share": resident / SRAM_BYTES_PER_SM,
            "fits": resident <= SRAM_USABLE_PER_BLOCK,
        })
    best = min(rows, key=lambda r: r["bytes"])
    fitting = [r for r in rows if r["fits"]]
    return {
        "tokens": n,
        "rows": rows,
        "least_traffic": best["tile"],
        "largest_that_fits": max(r["tile"] for r in fitting) if fitting else None,
        "sram_bytes_per_sm": SRAM_BYTES_PER_SM,
        "sram_usable_per_block": SRAM_USABLE_PER_BLOCK,
        "head_dim": MODEL.head_dim,
        "elem_bytes": ELEM_BYTES,
    }


def phases() -> dict:
    """Prefill against decode: where the square term is, and is not."""
    out = {}
    for name, new, seen in (("prefill", PROMPT_TOKENS, PROMPT_TOKENS),
                            ("decode", 1, CONTEXT_TOKENS)):
        c = MODEL
        whole = whole_traffic_bytes(c.n_heads, new, seen, c.head_dim,
                                    ELEM_BYTES, kv_heads=c.n_kv_heads)
        tiled = tiled_traffic_bytes(c.n_heads, new, seen, c.head_dim, Q_TILE,
                                    KV_TILE, ELEM_BYTES, kv_heads=c.n_kv_heads)
        square = score_matrix_bytes(c.n_heads, new, seen, ELEM_BYTES)
        out[name] = {
            "new_tokens": new, "seen_tokens": seen,
            "whole_bytes": whole, "tiled_bytes": tiled,
            "saving": 1 - tiled / whole,
            "score_matrix_bytes": square,
            "square_share_of_whole": 4 * square / whole,
            "kv_bytes": 2 * c.n_kv_heads * seen * c.head_dim * ELEM_BYTES,
            "kv_share_of_tiled": (2 * c.n_kv_heads * seen * c.head_dim
                                  * ELEM_BYTES) / tiled,
        }
    # And the same two, counted rather than predicted, on the small shape.
    counted = {}
    for name, new, seen in (("prefill", 1024, 1024), ("decode", 1, 1024)):
        q, k, v, mask = inputs(HEADS, new, seen, DIM)
        whole, tiled = Meter(), Meter()
        attention_whole(q, k, v, mask, whole)
        attention_tiled(q, k, v, mask, tiled, q_tile=Q_TILE, kv_tile=KV_TILE)
        counted[name] = {"whole_bytes": whole.total, "tiled_bytes": tiled.total,
                         "saving": 1 - tiled.total / whole.total}
    out["counted"] = counted
    return out


def service() -> dict:
    """What one prefill's saving is worth at the case study's rate.

    A millisecond on one prefill is hard to judge. The same millisecond
    multiplied by the prompts arriving every second is the quantity a
    fleet is sized against.
    """
    c = MODEL
    whole = whole_traffic_bytes(c.n_heads, PROMPT_TOKENS, PROMPT_TOKENS,
                                c.head_dim, ELEM_BYTES, kv_heads=c.n_kv_heads)
    tiled = tiled_traffic_bytes(c.n_heads, PROMPT_TOKENS, PROMPT_TOKENS,
                                c.head_dim, Q_TILE, KV_TILE, ELEM_BYTES,
                                kv_heads=c.n_kv_heads)
    per_prefill_s = (whole - tiled) * c.n_layers / HBM_BYTES_PER_S
    return {
        "requests_per_s": REQUESTS_PER_S,
        "saved_ms_per_prefill": per_prefill_s * 1e3,
        "saved_s_per_s_of_traffic": per_prefill_s * REQUESTS_PER_S,
        "whole_ms_per_prefill": whole * c.n_layers / HBM_BYTES_PER_S * 1e3,
        "tiled_ms_per_prefill": tiled * c.n_layers / HBM_BYTES_PER_S * 1e3,
    }


def main() -> None:
    eq, tr, ref, ti, ph = equivalence(), traffic(), reference(), tiles(), phases()
    sv = service()
    payload = {
        "equivalence": eq,
        "traffic": tr,
        "reference": ref,
        "tiles": ti,
        "phases": ph,
        "service": sv,
        "assumptions": {
            "q_tile": Q_TILE, "kv_tile": KV_TILE,
            "measured_heads": HEADS, "measured_head_dim": DIM,
            "measured_elem_bytes": ELEM,
            "reference_elem_bytes": ELEM_BYTES,
            "lengths": LENGTHS, "tile_sizes": TILES,
            "prompt_tokens": PROMPT_TOKENS, "context_tokens": CONTEXT_TOKENS,
            "sram_bytes_per_sm": SRAM_BYTES_PER_SM,
            "hbm_bytes_per_s": HBM_BYTES_PER_S,
            "seed": SEED,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch20.json", payload)
    print(f"wrote {path}")
    print(f"  same tokens with the kernel swapped: {eq['same_tokens']}, "
          f"across {len(eq['tile_sizes'])} tile sizes; "
          f"largest difference in attention {eq['attention_max_diff']:.3g} "
          f"on values up to {eq['attention_scale']:.3g}")
    print("  traffic, counted, one layer of an 8-head model:")
    for r in tr:
        print(f"    {r['tokens']:5,} tokens: whole {r['whole_bytes'] / 1e6:8.2f} MB, "
              f"tiled {r['tiled_bytes'] / 1e6:6.2f} MB, "
              f"{r['ratio']:5.2f}x less, largest intermediate "
              f"{r['whole_largest_intermediate'] / 1e6:6.2f} -> "
              f"{r['tiled_largest_intermediate'] / 1e6:.3f} MB, "
              f"{r['skipped_share'] * 100:4.1f}% of tiles skipped"
              f"{'' if r['formulas_agree'] else '   FORMULA DISAGREES'}")
    print("  the 8B reference model, per layer, from the formula:")
    for r in ref["rows"]:
        print(f"    {r['tokens']:5,} tokens: score matrix "
              f"{r['score_matrix_bytes'] / 1e6:8.1f} MB "
              f"({r['square_over_inputs']:5.1f}x its own inputs), "
              f"traffic {r['whole_bytes'] / 1e6:8.1f} -> "
              f"{r['tiled_bytes'] / 1e6:6.1f} MB, "
              f"{r['whole_ms_all_layers']:7.2f} -> "
              f"{r['tiled_ms_all_layers']:5.2f} ms over all layers")
    print(f"  tiles at {ti['tokens']:,} tokens "
          f"(fast memory for the reference model's head, bf16):")
    for r in ti["rows"]:
        print(f"    {r['tile']:4d}: {r['bytes'] / 1e6:7.2f} MB moved, "
              f"{r['sram_bytes_reference_model'] / 1024:6.1f} KB resident "
              f"({r['sram_share'] * 100:5.1f}% of an SM's 228 KB)"
              f"{'' if r['fits'] else '   does not fit'}")
    print(f"  at {sv['requests_per_s']} requests a second, "
          f"{sv['saved_ms_per_prefill']:.2f} ms saved per prefill is "
          f"{sv['saved_s_per_s_of_traffic']:.2f} accelerator-seconds of "
          f"memory time per second of traffic")
    p, d = ph["prefill"], ph["decode"]
    print(f"  prefill ({p['new_tokens']:,} new): {p['saving'] * 100:.0f}% less "
          f"traffic; decode (1 new): {d['saving'] * 100:.1f}%")


if __name__ == "__main__":
    main()
