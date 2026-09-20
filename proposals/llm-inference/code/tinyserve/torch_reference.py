"""The same model again, in PyTorch, purely to check the NumPy one.

`tinyserve` is written in NumPy so that every operation is visible. That
is good for teaching and bad for trust: a reader has no reason to
believe 250 lines of hand-written arithmetic implement a transformer
correctly.

This is the same architecture built on a mature framework. Chapter 10
loads one model's weights into both and checks they agree. If they do,
every measurement in Parts III to VI is measuring a real transformer.

Deliberately plain: no cache, no cleverness, no attempt to be fast.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import Model


def rms_norm(x: torch.Tensor, gain: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / torch.sqrt((x * x).mean(-1, keepdim=True) + eps) * gain


def block(x: torch.Tensor, g1: torch.Tensor, wq: torch.Tensor,
          wk: torch.Tensor, wv: torch.Tensor, wo: torch.Tensor,
          g2: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor,
          mask: torch.Tensor, n_heads: int, n_kv_heads: int,
          head_dim: int) -> torch.Tensor:
    """One transformer layer: attention, then a feed-forward network.

    Takes tensors and integers and nothing else. That is not tidiness:
    it is what lets Chapter 21 hand this function to `torch.compile`,
    which can fuse a graph of tensor operations and cannot see through
    a dataclass of NumPy arrays. A layer is also the unit a real engine
    compiles, for the same reason.
    """
    t = x.shape[0]
    n_rep = n_heads // n_kv_heads
    h = rms_norm(x, g1)
    q = (h @ wq).view(t, n_heads, head_dim).transpose(0, 1)
    k = (h @ wk).view(t, n_kv_heads, head_dim).transpose(0, 1)
    v = (h @ wv).view(t, n_kv_heads, head_dim).transpose(0, 1)
    if n_rep > 1:
        k, v = k.repeat_interleave(n_rep, 0), v.repeat_interleave(n_rep, 0)

    scores = (q @ k.transpose(1, 2)) * head_dim**-0.5 + mask
    attn = (scores.softmax(-1) @ v).transpose(0, 1).reshape(t, -1)
    x = x + attn @ wo

    h = rms_norm(x, g2)
    return x + F.gelu(h @ w1, approximate="tanh") @ w2


def on_device(model: Model, device: torch.device | str) -> dict[int, torch.Tensor]:
    """The model's weights as tensors on a device, converted once.

    `forward` converts each array as it reaches it, which is the clearest
    thing to read and fine for Chapter 10, where the point is to check
    one implementation against another. Chapter 21 runs this model
    thousands of times to measure what a *step* costs, and converting
    weights inside the step would measure the conversion. Pass the
    result of this as `weights` and it converts nothing.

    Keyed by `id` of the NumPy array, so `forward` can look up whatever
    array it was about to convert without either of them keeping a list
    in step with the other.
    """
    out = {}
    for a in [model.tok_emb, model.pos_emb, model.g_out]:
        out[id(a)] = torch.from_numpy(a.copy()).to(device)
    for layer in model.layers:
        for a in (layer.wq, layer.wk, layer.wv, layer.wo,
                  layer.w1, layer.w2, layer.g1, layer.g2):
            out[id(a)] = torch.from_numpy(a.copy()).to(device)
    return out


@torch.no_grad()
def forward(model: Model, tokens: torch.Tensor,
            weights: dict[int, torch.Tensor] | None = None) -> torch.Tensor:
    """Logits for every position, from the same weights as the NumPy model.

    With `weights` from `on_device`, the arithmetic runs wherever those
    tensors live and nothing is converted. Without it, every array is
    converted on the spot, on the CPU, which is what Chapter 10 does.
    """
    cfg = model.cfg
    t = tokens.shape[0]
    n_rep = cfg.n_heads // cfg.n_kv_heads
    if weights is None:
        tf = lambda a: torch.from_numpy(a.copy())  # noqa: E731
    else:
        tf = lambda a: weights[id(a)]              # noqa: E731

    x = tf(model.tok_emb)[tokens] + tf(model.pos_emb)[:t]
    mask = torch.full((t, t), float("-inf"),
                      device=x.device, dtype=x.dtype).triu(1)

    for layer in model.layers:
        x = block(x, tf(layer.g1), tf(layer.wq), tf(layer.wk), tf(layer.wv),
                  tf(layer.wo), tf(layer.g2), tf(layer.w1), tf(layer.w2),
                  mask, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)

    return rms_norm(x, tf(model.g_out)) @ tf(model.tok_emb).T
