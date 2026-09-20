"""A very small GPT, written in NumPy so that every operation is visible.

The model exists to be measured, not to be good at language. It has the
same shape as a production decoder -- token embeddings, a stack of
(attention, feed-forward) blocks, a final norm, and a tied output head --
at a size that runs on a laptop CPU. Weights are random; timing does not
depend on what the model has learned.

Written for `LLM Inference from the Ground Up`, Part III.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DType = np.float32


@dataclass(frozen=True)
class Config:
    """Model shape. The defaults give a ~0.98M-parameter model."""

    vocab_size: int = 512
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int = 4  # < n_heads gives grouped-query attention
    d_ff: int = 512
    max_seq: int = 1024

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must divide evenly into n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")


@dataclass
class Layer:
    wq: np.ndarray  # (d_model, n_heads * head_dim)
    wk: np.ndarray  # (d_model, n_kv_heads * head_dim)
    wv: np.ndarray  # (d_model, n_kv_heads * head_dim)
    wo: np.ndarray  # (n_heads * head_dim, d_model)
    w1: np.ndarray  # (d_model, d_ff)
    w2: np.ndarray  # (d_ff, d_model)
    g1: np.ndarray  # (d_model,) attention norm gain
    g2: np.ndarray  # (d_model,) feed-forward norm gain


@dataclass
class Model:
    cfg: Config
    tok_emb: np.ndarray  # (vocab_size, d_model), tied to the output head
    pos_emb: np.ndarray  # (max_seq, d_model)
    layers: list[Layer] = field(default_factory=list)
    g_out: np.ndarray = None  # (d_model,) final norm gain

    @property
    def n_params(self) -> int:
        n = self.tok_emb.size + self.pos_emb.size + self.g_out.size
        for l in self.layers:
            n += sum(w.size for w in (l.wq, l.wk, l.wv, l.wo, l.w1, l.w2, l.g1, l.g2))
        return n


def build(cfg: Config = Config(), seed: int = 0) -> Model:
    """Create a model with random weights, reproducibly."""
    rng = np.random.default_rng(seed)

    def normal(*shape: int, scale: float) -> np.ndarray:
        return (rng.standard_normal(shape) * scale).astype(DType)

    s = cfg.d_model**-0.5
    layers = [
        Layer(
            wq=normal(cfg.d_model, cfg.n_heads * cfg.head_dim, scale=s),
            wk=normal(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, scale=s),
            wv=normal(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, scale=s),
            wo=normal(cfg.n_heads * cfg.head_dim, cfg.d_model, scale=s),
            w1=normal(cfg.d_model, cfg.d_ff, scale=s),
            w2=normal(cfg.d_ff, cfg.d_model, scale=cfg.d_ff**-0.5),
            g1=np.ones(cfg.d_model, dtype=DType),
            g2=np.ones(cfg.d_model, dtype=DType),
        )
        for _ in range(cfg.n_layers)
    ]
    return Model(
        cfg=cfg,
        tok_emb=normal(cfg.vocab_size, cfg.d_model, scale=s),
        pos_emb=normal(cfg.max_seq, cfg.d_model, scale=s),
        layers=layers,
        g_out=np.ones(cfg.d_model, dtype=DType),
    )


# --- the operations, each one line of arithmetic ------------------------


def rms_norm(x: np.ndarray, gain: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * gain


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def share_kv_heads(k: np.ndarray, v: np.ndarray, heads: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Give every query head the keys and values of its group.

    With grouped-query attention (Chapter 7) there are fewer key heads
    than query heads, and each key head serves several query heads. The
    expansion belongs to the kernel rather than to the model: a real one
    reads the shared keys from memory once and hands the same copy to
    every query head in the group, which is why Chapter 20 charges them
    once.
    """
    n_rep = heads // k.shape[0]
    if n_rep == 1:
        return k, v
    return np.repeat(k, n_rep, axis=0), np.repeat(v, n_rep, axis=0)


def attention_whole(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    mask: np.ndarray, meter: "Meter | None" = None
                    ) -> tuple[np.ndarray, np.ndarray | None]:
    """Attention with the score matrix built in full.

    `q` is (heads, new, dim), `k` and `v` are (heads, seen, dim), and
    `mask` is (new, seen): zero where a query may attend to a key,
    negative infinity where it may not.

    This is the arithmetic of Chapter 2, lifted out of `forward` so that
    Chapter 20 can hand `forward` a different kernel without touching
    the model -- the same seam Chapter 14 used for the cache. It returns
    the weight matrix alongside the output because it has one anyway; a
    kernel that never builds one returns None there.

    `meter` is Chapter 20's byte counter. When it is None, which is
    every use before Part IV, this costs nothing.
    """
    scale = q.shape[-1] ** -0.5
    if meter is not None:
        meter.read(q), meter.read(k), meter.read(v)
    k, v = share_kv_heads(k, v, q.shape[0])
    scores = (q @ k.transpose(0, 2, 1)) * scale + mask
    if meter is not None:
        meter.write(scores), meter.read(scores)
    weights = softmax(scores)
    if meter is not None:
        meter.write(weights), meter.read(weights)
    out = weights @ v
    if meter is not None:
        meter.write(out)
    return out, weights


class KVCache:
    """Storage for the keys and values of every token the model has seen.

    One pair of arrays per layer, preallocated to `max_seq` positions.
    Chapter 13 measures what that preallocation costs; Chapter 14
    replaces it with pages.
    """

    def __init__(self, cfg: Config, max_seq: int | None = None) -> None:
        self.cfg = cfg
        self.max_seq = max_seq or cfg.max_seq
        shape = (cfg.n_kv_heads, self.max_seq, cfg.head_dim)
        self.k = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self.v = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self.length = 0

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.k) + sum(a.nbytes for a in self.v)

    def append(self, layer: int, k: np.ndarray, v: np.ndarray,
               start: int) -> tuple[np.ndarray, np.ndarray]:
        """Store this layer's new keys and values, and return everything
        attention should read.

        `forward` calls this rather than touching the arrays, so a
        different storage scheme can be substituted without changing the
        model. Chapter 14 substitutes one.
        """
        total = start + k.shape[1]
        self.k[layer][:, start:total] = k
        self.v[layer][:, start:total] = v
        return self.k[layer][:, :total], self.v[layer][:, :total]

    def bytes_per_token(self) -> int:
        c = self.cfg
        return 2 * c.n_layers * c.n_kv_heads * c.head_dim * np.dtype(DType).itemsize


def forward(model: Model, tokens: np.ndarray, cache: KVCache | None = None,
            trace: dict | None = None, attention=attention_whole) -> np.ndarray:
    """Run the model over `tokens` and return logits, shape (len(tokens), vocab).

    With no cache, `tokens` is the whole sequence and the model recomputes
    everything -- the naive loop of Chapter 11.

    With a cache, `tokens` is only what is *new*. Attention reads the keys
    and values of earlier tokens out of the cache instead of recomputing
    them, and the new keys and values are appended. This one function
    serves both prefill (many new tokens) and decode (one new token).

    Pass a dict as `trace` to record the intermediate tensors. Chapter 2
    uses it to show what actually happens to a prompt; nothing else does,
    and it costs nothing when it is None.

    `attention` is the kernel. The default builds the whole score matrix;
    Chapter 20 passes one that never does. A kernel that does not build
    the matrix cannot report attention weights, and a trace taken with
    one records None for them.
    """
    cfg = model.cfg
    t = len(tokens)
    start = cache.length if cache is not None else 0

    x = model.tok_emb[tokens] + model.pos_emb[start : start + t]
    if trace is not None:
        trace["tokens"] = tokens.tolist()
        trace["embedding"] = x.copy()
        trace["layers"] = []

    # Query position start+i may attend to key position j <= start+i.
    total = start + t
    q_pos = np.arange(start, total)[:, None]
    k_pos = np.arange(total)[None, :]
    mask = np.where(k_pos <= q_pos, 0.0, -np.inf).astype(DType)

    for i, layer in enumerate(model.layers):
        h = rms_norm(x, layer.g1)

        # (t, heads, head_dim) -> (heads, t, head_dim)
        q = (h @ layer.wq).reshape(t, cfg.n_heads, cfg.head_dim).transpose(1, 0, 2)
        k = (h @ layer.wk).reshape(t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)
        v = (h @ layer.wv).reshape(t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 0, 2)

        if cache is not None:
            k, v = cache.append(i, k, v, start)

        # Grouped-query attention is the kernel's business now: it reads
        # each shared key head once. `share_kv_heads` does the expansion.
        heads, weights = attention(q, k, v, mask)
        attn = heads.transpose(1, 0, 2).reshape(t, -1)
        x = x + attn @ layer.wo
        if trace is not None:
            after_attention = x.copy()

        h = rms_norm(x, layer.g2)
        x = x + gelu(h @ layer.w1) @ layer.w2
        if trace is not None:
            trace["layers"].append({
                # None when the kernel never assembled them (Chapter 20).
                "attention_weights": None if weights is None else weights.copy(),
                "keys": k.copy(), "values": v.copy(),
                "after_attention": after_attention,
                "after_feed_forward": x.copy(),
            })

    if cache is not None:
        cache.length = total

    logits = rms_norm(x, model.g_out) @ model.tok_emb.T
    if trace is not None:
        trace["final"] = x.copy()
        trace["logits"] = logits.copy()
    return logits
