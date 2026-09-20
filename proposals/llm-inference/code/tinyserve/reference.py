"""The production model, the accelerator, and the case study: defined once.

Chapters 1, 2, 3 and 13 all reason about the same 8B model on the same
H100 serving the same traffic. Each used to carry its own copy of those
constants, and the copies drifted: Chapter 3's shape implied 6.15B
parameters while its byte counts assumed 8.00B, so its arithmetic per
byte was wrong, and Chapters 2 and 3 disagreed about how much
arithmetic one token costs.

One definition, imported everywhere, makes that class of mistake
impossible. Every value here is recorded in FACTS.md with its source
and verification date.
"""

from __future__ import annotations

from .model import Config

# --- the accelerator: NVIDIA H100 SXM 80GB (FACTS.md, verified 2026-09)
HBM_BYTES_PER_S = 3.35e12
PEAK_BF16_FLOPS = 990e12
GPU_BYTES = 80 * 1000**3
GPU_USD_PER_HOUR = 3.25          # on-demand median across providers
RIDGE_FLOP_PER_BYTE = PEAK_BF16_FLOPS / HBM_BYTES_PER_S

# --- the model: a Llama-3-style dense 8B served in bf16.
#
# `d_ff` is set so that this shape totals ~8B parameters under the
# two-matrix feed-forward that tinyserve's cost model assumes. Real
# Llama-3 uses three smaller matrices (SwiGLU) reaching the same total;
# the parameter count is what the arithmetic depends on, and it matches.
ELEM_BYTES = 2
MODEL = Config(vocab_size=128_256, d_model=4096, n_layers=32, n_heads=32,
               n_kv_heads=8, d_ff=21_398, max_seq=8192)


def params(c: Config = MODEL) -> int:
    """Total parameters implied by a shape: the single source of truth for
    both the byte counts and the arithmetic counts."""
    embedding = c.vocab_size * c.d_model
    per_layer = (c.d_model * c.n_heads * c.head_dim            # queries
                 + 2 * c.d_model * c.n_kv_heads * c.head_dim   # keys, values
                 + c.n_heads * c.head_dim * c.d_model          # output
                 + 2 * c.d_model * c.d_ff)                     # feed-forward
    return embedding * 2 + c.n_layers * per_layer              # + output head


PARAMS = params()
WEIGHT_BYTES = PARAMS * ELEM_BYTES
KV_BYTES_PER_TOKEN = (2 * MODEL.n_layers * MODEL.n_kv_heads
                      * MODEL.head_dim * ELEM_BYTES)

# --- the case study (STANDARDS.md section 7)
PROMPT_TOKENS, OUTPUT_TOKENS = 1200, 300
CONTEXT_TOKENS = PROMPT_TOKENS + OUTPUT_TOKENS
REQUESTS_PER_S = 200


def test_reference_is_an_8b_model() -> None:
    """The shape must actually be the model it claims to be."""
    assert abs(PARAMS - 8e9) / 8e9 < 0.01, f"{PARAMS:,} is not ~8B"
    assert WEIGHT_BYTES == PARAMS * 2
    assert KV_BYTES_PER_TOKEN == 131_072, KV_BYTES_PER_TOKEN
