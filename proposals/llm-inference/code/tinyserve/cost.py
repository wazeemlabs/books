"""What a forward pass costs, in arithmetic and in bytes.

The book measures time, but time alone does not explain itself. These
functions give the two quantities that do: how much arithmetic a pass
performs, and how many bytes it must read to perform it. Their ratio --
work done per byte fetched -- is what separates prefill from decode,
and it is the number Chapter 8's roofline is built on.

Shared by Chapters 3 and 12 so that one definition serves both.
"""

from __future__ import annotations

from .model import Config


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


def kv_bytes_per_token(c: Config, elem_bytes: int = 4) -> int:
    return 2 * c.n_layers * c.n_kv_heads * c.head_dim * elem_bytes


def bytes_read(c: Config, params: int, t_new: int, t_total: int,
               elem_bytes: int = 4) -> int:
    """Bytes the pass must fetch: every weight, once, plus the cached keys
    and values of tokens it did not just compute."""
    cached = max(0, t_total - t_new)
    return params * elem_bytes + cached * kv_bytes_per_token(c, elem_bytes)


def intensity(c: Config, params: int, t_new: int, t_total: int,
              elem_bytes: int = 4) -> float:
    """Arithmetic intensity: FLOPs performed per byte fetched.

    High means the hardware's arithmetic is the limit. Low means its
    memory is. Prefill and decode sit at opposite ends of this number,
    which is why they behave like different programs.
    """
    return flops_forward(c, t_new, t_total) / bytes_read(c, params, t_new,
                                                         t_total, elem_bytes)
