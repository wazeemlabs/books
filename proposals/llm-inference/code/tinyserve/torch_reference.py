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


@torch.no_grad()
def forward(model: Model, tokens: torch.Tensor) -> torch.Tensor:
    """Logits for every position, from the same weights as the NumPy model."""
    cfg = model.cfg
    t = tokens.shape[0]
    n_rep = cfg.n_heads // cfg.n_kv_heads
    tf = lambda a: torch.from_numpy(a.copy())  # noqa: E731

    x = tf(model.tok_emb)[tokens] + tf(model.pos_emb)[:t]
    mask = torch.full((t, t), float("-inf")).triu(1)

    for layer in model.layers:
        h = rms_norm(x, tf(layer.g1))
        q = (h @ tf(layer.wq)).view(t, cfg.n_heads, cfg.head_dim).transpose(0, 1)
        k = (h @ tf(layer.wk)).view(t, cfg.n_kv_heads, cfg.head_dim).transpose(0, 1)
        v = (h @ tf(layer.wv)).view(t, cfg.n_kv_heads, cfg.head_dim).transpose(0, 1)
        if n_rep > 1:
            k, v = k.repeat_interleave(n_rep, 0), v.repeat_interleave(n_rep, 0)

        scores = (q @ k.transpose(1, 2)) * cfg.head_dim**-0.5 + mask
        attn = (scores.softmax(-1) @ v).transpose(0, 1).reshape(t, -1)
        x = x + attn @ tf(layer.wo)

        h = rms_norm(x, tf(layer.g2))
        x = x + F.gelu(h @ tf(layer.w1), approximate="tanh") @ tf(layer.w2)

    return rms_norm(x, tf(model.g_out)) @ tf(model.tok_emb).T
