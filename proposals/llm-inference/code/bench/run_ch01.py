"""Chapter 1: what a token costs, and why the answer spans two orders of magnitude.

This is a cost MODEL, not a benchmark. It is arithmetic over published
hardware specifications and rental prices (see FACTS.md), and it is
labelled as such wherever the book quotes it. Part VII measures the real
thing on real hardware; this chapter's job is to show a reader why the
rest of the book exists.

    python3 -m bench.run_ch01
"""

from __future__ import annotations

from tinyserve.reference import (GPU_BYTES, GPU_USD_PER_HOUR, HBM_BYTES_PER_S,
                                 KV_BYTES_PER_TOKEN, PARAMS, PEAK_BF16_FLOPS,
                                 WEIGHT_BYTES)
from tinyserve.reference import CONTEXT_TOKENS as SEQ_LEN

from .harness import write

# Concurrency ceiling from Chapter 13's paged allocator on this card.
# Checked against that chapter's results below, so the two cannot drift.
MAX_CONCURRENT = 325
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, MAX_CONCURRENT]

# Published API price for a hosted 8B, per million tokens (FACTS.md).
API_OUTPUT_USD_PER_M = 0.05
API_INPUT_USD_PER_M = 0.02


def decode_step(batch: int, seq: int = SEQ_LEN) -> dict:
    """One decode step for `batch` sequences, under a roofline model.

    Each step reads every weight once for the whole batch -- that cost is
    shared -- plus each sequence's own KV cache, which is not shared. It
    does two floating-point operations per parameter per token.
    """
    bytes_read = WEIGHT_BYTES + batch * seq * KV_BYTES_PER_TOKEN
    flops = 2 * PARAMS * batch
    t_memory = bytes_read / HBM_BYTES_PER_S
    t_compute = flops / PEAK_BF16_FLOPS
    t = max(t_memory, t_compute)
    return {
        "batch": batch,
        "tokens_per_s": batch / t,
        "step_ms": t * 1e3,
        "bound_by": "memory" if t_memory >= t_compute else "compute",
        "usd_per_m_tokens": (GPU_USD_PER_HOUR / 3600) / (batch / t) * 1e6,
        "flop_utilization": flops / PEAK_BF16_FLOPS / t,
        "weight_share_of_bytes": WEIGHT_BYTES / bytes_read,
    }


def _check_concurrency_matches_ch13() -> None:
    """Chapter 1 quotes a ceiling Chapter 13 computes. Verify, do not assume."""
    import json
    from pathlib import Path
    path = Path("results/ch13.json")
    if not path.exists():
        return
    paged = next(r for r in json.loads(path.read_text())["policies"]
                 if r["policy"] == "paged_16")
    if paged["concurrent_seqs"] != MAX_CONCURRENT:
        raise SystemExit(f"MAX_CONCURRENT={MAX_CONCURRENT} but Chapter 13 "
                         f"computes {paged['concurrent_seqs']}; reconcile them")


def main() -> None:
    _check_concurrency_matches_ch13()
    rows = [decode_step(b) for b in BATCHES]
    cheapest, dearest = rows[-1], rows[0]

    payload = {
        "model_not_measurement": True,
        "note": "Arithmetic over published specs and prices; not a benchmark. "
                "Part VII measures the real thing.",
        "assumptions": {
            "gpu": "H100 SXM 80GB", "hbm_bytes_per_s": HBM_BYTES_PER_S,
            "peak_bf16_flops": PEAK_BF16_FLOPS,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "params": PARAMS, "weight_bytes": WEIGHT_BYTES,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN, "seq_len": SEQ_LEN,
            "max_concurrent": MAX_CONCURRENT,
            "facts_verified": "2026-09",
        },
        "api_reference": {
            "model": "hosted Llama-3.1-8B, cheapest tracked provider",
            "output_usd_per_m": API_OUTPUT_USD_PER_M,
            "input_usd_per_m": API_INPUT_USD_PER_M,
        },
        "batches": rows,
        "spread": {
            "worst_usd_per_m": dearest["usd_per_m_tokens"],
            "best_usd_per_m": cheapest["usd_per_m_tokens"],
            "ratio": dearest["usd_per_m_tokens"] / cheapest["usd_per_m_tokens"],
            "worst_vs_api": dearest["usd_per_m_tokens"] / API_OUTPUT_USD_PER_M,
            "best_vs_api": cheapest["usd_per_m_tokens"] / API_OUTPUT_USD_PER_M,
        },
    }

    path = write("results/ch01.json", payload)
    print(f"wrote {path}")
    for r in rows:
        print(f"  batch {r['batch']:4d}  {r['tokens_per_s']:9,.0f} tok/s"
              f"  ${r['usd_per_m_tokens']:8.3f}/M"
              f"  {r['bound_by']:7}  {r['flop_utilization']*100:5.1f}% of peak FLOPs")
    print(f"  spread {payload['spread']['ratio']:.0f}x on identical hardware;"
          f" published API price ${API_OUTPUT_USD_PER_M}/M")


if __name__ == "__main__":
    main()
