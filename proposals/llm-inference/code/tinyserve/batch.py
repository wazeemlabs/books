"""Many sequences through one fetch of the weights.

Decoding one sequence reads every weight in the model to produce one
token. Decoding two sequences at once reads every weight *once* and
produces two. That is the whole of batching, and it is the only
technique in this book that attacks the memory wall head-on rather than
working around it.

What can be shared and what cannot:

* The weight matmuls can. A batch of B sequences turns every
  `(1, d_model) @ (d_model, n)` into `(B, d_model) @ (d_model, n)`:
  the same weights, fetched once, used B times.
* Attention cannot. Every sequence has its own keys and values, and its
  own length. It is done one sequence at a time here, which is what a
  production variable-length attention kernel does too -- it just does
  the sequences concurrently instead of in a Python loop.

`decode_step` must return exactly what `forward` returns for each
sequence separately. `same_as_unbatched` checks it.

Written for `LLM Inference from the Ground Up`, Chapter 16.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .model import DType, KVCache, Model, forward, gelu, rms_norm, softmax


def decode_step(model: Model, caches: list, tokens: np.ndarray,
                timing: dict | None = None) -> np.ndarray:
    """One decode step for every sequence in `caches` at once.

    `tokens` holds one new token per sequence, shape (B,). Returns
    logits of shape (B, vocab_size), and advances each cache by one.

    Pass a dict as `timing` to split the step into the part the batch
    shares (the weight matmuls) and the part it cannot (attention, one
    sequence at a time). Chapter 16 needs the two separately, because
    only the first is what batching is for.
    """
    cfg = model.cfg
    b_size = len(caches)
    if len(tokens) != b_size:
        raise ValueError("one new token per sequence, no more and no fewer")
    n_rep = cfg.n_heads // cfg.n_kv_heads
    starts = [c.length for c in caches]

    clock = perf_counter if timing is not None else None
    shared_s = private_s = 0.0

    x = model.tok_emb[tokens] + model.pos_emb[starts]          # (B, d_model)

    for i, layer in enumerate(model.layers):
        if clock:
            t0 = clock()
        h = rms_norm(x, layer.g1)

        # The shared part: one fetch of each weight for the whole batch.
        q = (h @ layer.wq).reshape(b_size, cfg.n_heads, cfg.head_dim)
        k = (h @ layer.wk).reshape(b_size, cfg.n_kv_heads, cfg.head_dim)
        v = (h @ layer.wv).reshape(b_size, cfg.n_kv_heads, cfg.head_dim)
        if clock:
            shared_s += clock() - t0
            t0 = clock()

        # The private part: each sequence attends over its own cache.
        attn = np.empty((b_size, cfg.n_heads * cfg.head_dim), dtype=DType)
        for b, cache in enumerate(caches):
            kk, vv = cache.append(i, k[b][:, None, :], v[b][:, None, :], starts[b])
            if n_rep > 1:
                kk = np.repeat(kk, n_rep, axis=0)
                vv = np.repeat(vv, n_rep, axis=0)
            scores = (q[b][:, None, :] @ kk.transpose(0, 2, 1)) * cfg.head_dim**-0.5
            attn[b] = (softmax(scores) @ vv).reshape(-1)

        if clock:
            private_s += clock() - t0
            t0 = clock()
        x = x + attn @ layer.wo
        h = rms_norm(x, layer.g2)
        x = x + gelu(h @ layer.w1) @ layer.w2
        if clock:
            shared_s += clock() - t0

    for cache, start in zip(caches, starts):
        cache.length = start + 1

    if clock:
        t0 = clock()
    logits = rms_norm(x, model.g_out) @ model.tok_emb.T
    if timing is not None:
        timing["shared_s"] = shared_s + (clock() - t0)
        timing["private_s"] = private_s
    return logits


def generate_batched(model: Model, prompts: list[list[int]], n_new: int,
                     caches: list | None = None) -> list[list[int]]:
    """Prefill each prompt, then decode all of them together, greedily.

    Prefill is still one sequence at a time: the prompts have different
    lengths, and a batch of them would have to be padded to the longest.
    What that costs is Chapter 16's second measurement.
    """
    if caches is None:
        caches = [KVCache(model.cfg, max_seq=len(p) + n_new + 1) for p in prompts]

    nxt = []
    for prompt, cache in zip(prompts, caches):
        logits = forward(model, np.array(prompt), cache)
        nxt.append(int(logits[-1].argmax()))

    out = [[t] for t in nxt]
    for _ in range(n_new - 1):
        logits = decode_step(model, caches, np.array(nxt))
        nxt = [int(row.argmax()) for row in logits]
        for seq, token in zip(out, nxt):
            seq.append(token)
    return out


def same_as_unbatched(model: Model, prompts: list[list[int]],
                      n_new: int) -> tuple[bool, float]:
    """Does decoding together give what decoding separately gives?

    Returns whether the tokens match and the largest difference between
    the two sets of scores.
    """
    together = generate_batched(model, prompts, n_new)

    apart = []
    for prompt in prompts:
        cache = KVCache(model.cfg, max_seq=len(prompt) + n_new + 1)
        logits = forward(model, np.array(prompt), cache)
        seq = []
        for _ in range(n_new):
            token = int(logits[-1].argmax())
            seq.append(token)
            logits = forward(model, np.array([token]), cache)
        apart.append(seq)

    return together == apart, first_step_spread(model, prompts, [1, len(prompts)])


def first_step_scores(model: Model, prompts: list[list[int]],
                      batch: int) -> np.ndarray:
    """Sequence 0's scores after one decode step, taken in a batch of
    `batch`. Everything about sequence 0 is identical in each case; only
    how many others ride along changes."""
    chosen = prompts[:batch]
    caches = [KVCache(model.cfg, max_seq=len(p) + 2) for p in chosen]
    first = [int(forward(model, np.array(p), c)[-1].argmax())
             for p, c in zip(chosen, caches)]
    return decode_step(model, caches, np.array(first))[0]


def first_step_spread(model: Model, prompts: list[list[int]],
                      sizes: list[int]) -> float:
    """The largest disagreement in sequence 0's scores across batch sizes."""
    rows = [first_step_scores(model, prompts, b) for b in sizes]
    return max(float(np.max(np.abs(a - b))) for a in rows for b in rows)


def agreement_threshold(model: Model, prompts: list[list[int]],
                        sizes: list[int]) -> dict:
    """From which batch size on does the answer stop depending on the batch?

    Sequence 0's scores are compared across batch sizes. Its own input,
    its own cache and its own position are identical every time; the
    only thing that changes is how many other sequences ride along in
    the same matrix multiply. A library that took one path through all
    of them would give one answer.
    """
    rows = {b: first_step_scores(model, prompts, b) for b in sizes}
    largest = rows[sizes[-1]]
    diffs = {b: float(np.max(np.abs(rows[b] - largest))) for b in sizes}
    stable = [b for b in sizes if diffs[b] == 0.0]
    return {
        "sizes": sizes,
        "diff_from_largest": [diffs[b] for b in sizes],
        "agree_from": min(stable) if stable else None,
        "worst": max(diffs.values()),
    }


def test_batching_changes_no_token() -> None:
    """A batch of four must generate what four separate runs generate."""
    from .model import Config, build
    cfg = Config()
    model = build(cfg)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, cfg.vocab_size, n).tolist() for n in (12, 20, 33, 7)]
    same, _ = same_as_unbatched(model, prompts, n_new=6)
    assert same, "batched decoding produced different tokens"


def test_decode_step_advances_every_sequence() -> None:
    """One step, one token each, whatever the batch."""
    from .model import Config, build
    cfg = Config()
    model = build(cfg)
    rng = np.random.default_rng(0)
    prompts = [rng.integers(0, cfg.vocab_size, n).tolist() for n in (12, 20, 33)]
    caches = [KVCache(cfg, max_seq=64) for _ in prompts]
    for prompt, cache in zip(prompts, caches):
        forward(model, np.array(prompt), cache)
    before = [c.length for c in caches]
    decode_step(model, caches, np.array([1, 2, 3]))
    assert [c.length for c in caches] == [n + 1 for n in before]

    try:
        decode_step(model, caches, np.array([1, 2]))
    except ValueError:
        return
    raise AssertionError("a token count that does not match the batch must fail")
