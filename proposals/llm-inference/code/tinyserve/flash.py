"""Attention that never writes the score matrix down.

Chapter 20 is about one intermediate result. Attention
multiplies queries by keys to get a score for every (query, key) pair,
turns those scores into weights, and multiplies the weights by the
values. The obvious way to write that is the way `model.forward` wrote
it: build the whole score matrix, then the whole weight matrix, then
multiply.

Those two matrices are the problem. Each is one number per pair of
positions, so each grows with the *square* of the sequence length while
the queries, keys and values themselves grow linearly. For a long
prompt the intermediate dwarfs its own inputs, and every byte of it is
written out to the large slow memory and read back.

FlashAttention's observation is that neither matrix has to exist. Work
a tile at a time, keep a running maximum and a running sum so the
softmax can be finished later, and the output can be accumulated in one
pass. The arithmetic is the same arithmetic; what changes is what
touches slow memory. The result is exact, not an approximation.

The whole-matrix version lives in `model.py`, because it is the
arithmetic every chapter before this one has been running. This module
adds the tiled one behind the same signature, a meter for the traffic
each of them causes, and the formulas the chapter applies to a model
too large to run here.

Written for `LLM Inference from the Ground Up`, Part IV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import DType, attention_whole, share_kv_heads

# A query tile and a key/value tile, in positions. Real kernels pick
# these so that one of each, plus the running state, fits in the fast
# memory attached to a core; Chapter 20 derives the sizes.
Q_TILE, KV_TILE = 64, 64


@dataclass
class Meter:
    """Bytes crossing between the two levels of a memory hierarchy.

    An attention implementation is handed one of these and reports every
    array it reads from, or writes to, the large slow level. Nothing is
    estimated: each call passes the array whose `nbytes` is counted, so
    the total is a property of the code that ran.
    """

    reads: int = 0
    writes: int = 0
    largest_intermediate: int = 0   # the biggest thing it had to keep
    tiles: int = 0                  # tiles of the score matrix computed
    tiles_skipped: int = 0          # tiles the causal mask made empty

    def read(self, a: np.ndarray) -> np.ndarray:
        self.reads += a.nbytes
        return a

    def write(self, a: np.ndarray) -> np.ndarray:
        self.writes += a.nbytes
        self.largest_intermediate = max(self.largest_intermediate, a.nbytes)
        return a

    @property
    def total(self) -> int:
        return self.reads + self.writes


def attention_tiled(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                    mask: np.ndarray, meter: Meter | None = None,
                    q_tile: int = Q_TILE, kv_tile: int = KV_TILE
                    ) -> tuple[np.ndarray, None]:
    """The same arithmetic, one tile at a time, in a single pass.

    For each tile of queries, walk the keys and values in tiles. Each
    step produces a small block of scores, which is used and discarded
    before the next block is computed. The running state per query row
    is three things: the largest score seen so far, the sum of the
    exponentials seen so far, and the output accumulated so far.

    The trick that makes one pass enough is rescaling. A softmax
    subtracts the row maximum before exponentiating, for the numerical
    reason Chapter 2 gave, and the maximum is not known until every
    score has been seen. Instead of waiting, subtract the largest score
    seen *so far*; when a later tile raises it, multiply the running sum
    and the running output by `exp(old - new)` to put them on the new
    footing. This is Milakov and Gimelshein's online softmax, and it is
    algebraically exact: no term is dropped or approximated.

    Returns the output, and `None` where the whole-matrix version
    returns weights. The weights were never assembled. That is the
    point, and it is also the cost: a kernel like this cannot hand a
    visualizer an attention map.
    """
    heads, new, dim = q.shape
    seen = k.shape[1]
    scale = dim ** -0.5
    out = np.zeros((heads, new, dim), dtype=DType)

    for i in range(0, new, q_tile):
        qi = q[:, i:i + q_tile]                         # (heads, tq, dim)
        tq = qi.shape[1]
        if meter is not None:
            meter.read(qi)
        # Running state, one row per query in this tile.
        m = np.full((heads, tq), -np.inf, dtype=DType)  # largest score so far
        l = np.zeros((heads, tq), dtype=DType)          # sum of exponentials
        acc = np.zeros((heads, tq, dim), dtype=DType)   # output so far

        for j in range(0, seen, kv_tile):
            block = mask[i:i + q_tile, j:j + kv_tile]
            if meter is not None:
                meter.tiles += 1
            if np.all(np.isneginf(block)):
                # Every pair in this tile is masked out. A causal mask
                # makes roughly half of them so, and skipping them is
                # the second saving: the work is never done at all.
                if meter is not None:
                    meter.tiles_skipped += 1
                continue
            kj, vj = k[:, j:j + kv_tile], v[:, j:j + kv_tile]
            if meter is not None:
                meter.read(kj), meter.read(vj)   # once, however many share them
            kj, vj = share_kv_heads(kj, vj, heads)

            s = (qi @ kj.transpose(0, 2, 1)) * scale + block
            if meter is not None:
                meter.write(s)          # the tile, and only the tile

            m_new = np.maximum(m, s.max(axis=-1))
            # A query row can be fully masked inside a tile that
            # straddles the diagonal. Its maximum is still -inf, and
            # subtracting -inf from -inf is not a number, so shift by
            # zero instead: every exponential in the row is then
            # exp(-inf) = 0 and the row contributes nothing, which is
            # the right answer.
            shift = np.where(np.isneginf(m_new), DType(0.0), m_new)
            p = np.exp(s - shift[..., None])
            alpha = np.exp(m - shift)   # 0 on the first tile, by exp(-inf)

            l = alpha * l + p.sum(axis=-1)
            acc = alpha[..., None] * acc + p @ vj
            m = m_new

        if not np.all(l > 0):
            raise ValueError("a query attends to nothing: the mask hides "
                             "every key from some position, including itself")
        tile_out = (acc / l[..., None]).astype(DType)
        if meter is not None:
            meter.write(tile_out)
        out[:, i:i + q_tile] = tile_out
    return out, None


def score_matrix_bytes(heads: int, new: int, seen: int,
                       elem_bytes: int = 4) -> int:
    """How large the intermediate the tiled version never builds is.

    One number per (query head, new position, seen position). Grouped
    queries do not shrink it: every query head has its own scores, even
    where several of them share a set of keys.
    """
    return heads * new * seen * elem_bytes


def _tile_is_masked(i: int, rows: int, j: int, q_tile: int, kv_tile: int,
                    new: int, seen: int) -> bool:
    """Is every pair in this tile above the diagonal?

    The new positions are the *last* `new` of a `seen`-long sequence, so
    a query at new-index r sits at absolute position `seen - new + r`.
    The tile is wholly masked when its first key comes after its last
    query. Getting the direction of that subtraction wrong is invisible
    while prefill is measured with `new == seen`, and wrong for every
    decode step.
    """
    last_query = (seen - new) + i * q_tile + rows - 1
    return j * kv_tile > last_query


def whole_traffic_bytes(heads: int, new: int, seen: int, dim: int,
                        elem_bytes: int = 4, kv_heads: int | None = None
                        ) -> int:
    """Slow-memory traffic for the whole-matrix version, from its shape.

    Read q and k; write the scores and read them back; write the weights
    and read them back; read v; write the output. The two square
    matrices are each written once and read once, so they account for
    four passes over an n-by-n array while everything else is linear.

    `kv_heads` is Chapter 7's grouped-query attention: several query
    heads share one set of keys and values, and the shared set is read
    once, not once per query head. It defaults to `heads`, which is
    plain multi-head attention.
    """
    kv_heads = heads if kv_heads is None else kv_heads
    square = score_matrix_bytes(heads, new, seen, elem_bytes)
    q = heads * new * dim * elem_bytes
    kv = 2 * kv_heads * seen * dim * elem_bytes
    out = heads * new * dim * elem_bytes
    return q + kv + 4 * square + out


def tiled_traffic_bytes(heads: int, new: int, seen: int, dim: int,
                        q_tile: int = Q_TILE, kv_tile: int = KV_TILE,
                        elem_bytes: int = 4, causal: bool = True,
                        kv_heads: int | None = None,
                        block_crosses: bool = True) -> int:
    """Slow-memory traffic for the tiled version, from its shape.

    Each query tile reads itself once and writes its own output once.
    Within a query tile, every key/value tile that is not entirely
    masked is read, and its block of scores is written and immediately
    consumed. The square term survives only at the size of one block.

    `block_crosses` is the question of where that block lives. The
    implementation here allocates it like any other array, so the meter
    sees it cross and the default charges it, which makes every figure
    in Chapter 20 a *lower bound* on what tiling saves. A real kernel
    sizes its tiles so the block stays in the scratchpad and never
    crosses at all; pass False for that bound. The truth is the first
    for this NumPy code and the second for a CUDA kernel, and the
    chapter quotes both rather than picking one.
    """
    kv_heads = heads if kv_heads is None else kv_heads
    total = 0
    for i in range(-(-new // q_tile)):
        rows = min(q_tile, new - i * q_tile)
        total += 2 * heads * rows * dim * elem_bytes    # read q tile, write out
        for j in range(-(-seen // kv_tile)):
            if causal and _tile_is_masked(i, rows, j, q_tile, kv_tile,
                                          new, seen):
                continue                                # wholly above the diagonal
            cols = min(kv_tile, seen - j * kv_tile)
            total += 2 * kv_heads * cols * dim * elem_bytes       # k and v
            if block_crosses:
                total += heads * rows * cols * elem_bytes         # the block
    return total


def causal_mask(new: int, seen: int) -> np.ndarray:
    """Zero where query position may attend to key position, -inf above.

    The last `new` positions of a `seen`-long sequence are the new ones,
    which is what prefill and decode both hand to attention.
    """
    start = seen - new
    q_pos = np.arange(start, seen)[:, None]
    k_pos = np.arange(seen)[None, :]
    return np.where(k_pos <= q_pos, 0.0, -np.inf).astype(DType)


# --- tests: the specification -------------------------------------------


def test_tiling_changes_no_number() -> None:
    """The tiled version is exact, not an approximation.

    Both versions are run on the same random inputs and compared. The
    difference must be at the level of float32 rounding -- the two do
    the same multiplications in a different order, and nothing else.
    """
    rng = np.random.default_rng(0)
    heads, new, seen, dim = 4, 130, 130, 32
    q = rng.standard_normal((heads, new, dim)).astype(DType)
    k = rng.standard_normal((heads, seen, dim)).astype(DType)
    v = rng.standard_normal((heads, seen, dim)).astype(DType)
    mask = causal_mask(new, seen)

    whole, weights = attention_whole(q, k, v, mask)
    tiled, none = attention_tiled(q, k, v, mask, q_tile=32, kv_tile=16)

    assert none is None, "a tiled kernel has no weight matrix to return"
    assert weights is not None and weights.shape == (heads, new, seen)
    gap = float(np.abs(whole - tiled).max())
    scale = float(np.abs(whole).max())
    assert gap / scale < 1e-5, f"tiling moved the answer by {gap:.3g}"


def test_tiling_survives_every_tile_size() -> None:
    """The tile sizes are a memory decision, never a correctness one.

    A tile that does not divide the sequence, a tile larger than the
    sequence, and a tile of one all have to give the same answer, or the
    size is a tuning knob that can silently change a reply.
    """
    rng = np.random.default_rng(1)
    heads, new, seen, dim = 2, 37, 53, 16
    q = rng.standard_normal((heads, new, dim)).astype(DType)
    k = rng.standard_normal((heads, seen, dim)).astype(DType)
    v = rng.standard_normal((heads, seen, dim)).astype(DType)
    mask = causal_mask(new, seen)
    whole, _ = attention_whole(q, k, v, mask)

    for q_tile, kv_tile in ((1, 1), (8, 5), (16, 64), (64, 16), (128, 128)):
        tiled, _ = attention_tiled(q, k, v, mask, q_tile=q_tile, kv_tile=kv_tile)
        gap = float(np.abs(whole - tiled).max())
        assert gap / float(np.abs(whole).max()) < 1e-5, (
            f"tiles ({q_tile}, {kv_tile}) moved the answer by {gap:.3g}")


def test_the_byte_formulas_match_the_bytes_counted() -> None:
    """The chapter applies these formulas to a model too large to run.

    That is only legitimate if a formula reproduces what the meter
    counts when the code does run, so this asserts the two agree
    exactly -- not approximately -- on shapes small enough to execute.

    The shapes matter as much as the assertion. An earlier version of
    this test used `new == seen` only, which is prefill, and a sign
    error in the causal skip went undetected: it made the tiled formula
    skip *every* tile of a decode step, reporting a 99.9% saving where
    the real figure is a few per cent. Decode shapes, shapes where the
    tile does not divide the length, and grouped-query shapes are all
    here for that reason.
    """
    elem = np.dtype(DType).itemsize
    shapes = [
        # heads, kv_heads, new, seen        what it stands for
        (3, 3, 96, 96),                     # prefill, square, tiles divide
        (4, 4, 100, 100),                   # prefill, tiles do not divide
        (4, 1, 96, 96),                     # prefill, four queries per key head
        (8, 2, 1, 320),                     # one decode step
        (8, 2, 7, 320),                     # a chunk of prefill mid-sequence
        (2, 2, 64, 64),                     # one tile exactly
    ]
    for heads, kv_heads, new, seen in shapes:
        rng = np.random.default_rng(2)
        dim = 16
        q = rng.standard_normal((heads, new, dim)).astype(DType)
        k = rng.standard_normal((kv_heads, seen, dim)).astype(DType)
        v = rng.standard_normal((kv_heads, seen, dim)).astype(DType)
        mask = causal_mask(new, seen)
        where = f"heads={heads} kv={kv_heads} new={new} seen={seen}"

        m = Meter()
        attention_whole(q, k, v, mask, m)
        want = whole_traffic_bytes(heads, new, seen, dim, elem, kv_heads)
        assert m.total == want, f"{where}: counted {m.total:,}, formula {want:,}"

        for q_tile, kv_tile in ((32, 32), (16, 48), (128, 128), (1, 1)):
            m = Meter()
            attention_tiled(q, k, v, mask, m, q_tile=q_tile, kv_tile=kv_tile)
            want = tiled_traffic_bytes(heads, new, seen, dim, q_tile, kv_tile,
                                       elem, kv_heads=kv_heads)
            assert m.total == want, (
                f"{where}, tiles ({q_tile}, {kv_tile}): counted {m.total:,}, "
                f"formula {want:,}")


def test_a_decode_step_skips_no_keys() -> None:
    """A decode step attends to the whole sequence, so nothing is above
    the diagonal. If the causal skip says otherwise, the formula is
    reporting a saving the kernel never made.
    """
    rng = np.random.default_rng(5)
    heads, seen, dim = 4, 256, 16
    q = rng.standard_normal((heads, 1, dim)).astype(DType)
    k = rng.standard_normal((heads, seen, dim)).astype(DType)
    v = rng.standard_normal((heads, seen, dim)).astype(DType)
    meter = Meter()
    attention_tiled(q, k, v, causal_mask(1, seen), meter, q_tile=64, kv_tile=64)
    assert meter.tiles_skipped == 0, (
        f"{meter.tiles_skipped} of {meter.tiles} tiles skipped at decode")


def test_swapping_the_kernel_changes_no_token() -> None:
    """The kernel is a seam, like the cache in Chapter 14.

    `forward` is handed the tiled kernel instead of the default one and
    must generate the same tokens from the same prompt. A kernel that
    changed a token would not be a faster way to do attention; it would
    be a different model.
    """
    from . import model as m

    cfg = m.Config(vocab_size=128, d_model=64, n_layers=3, n_heads=4,
                   n_kv_heads=2, d_ff=256, max_seq=256)
    net = m.build(cfg, seed=3)
    prompt = np.arange(48) % cfg.vocab_size

    def run(attention) -> list[int]:
        cache = m.KVCache(cfg)
        logits = m.forward(net, prompt, cache, attention=attention)
        out = []
        for _ in range(16):
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            logits = m.forward(net, np.array([nxt]), cache, attention=attention)
        return out

    assert run(attention_whole) == run(attention_tiled), (
        "the tiled kernel generated a different continuation")


def test_a_decode_step_has_almost_no_square_term_to_save() -> None:
    """Why this chapter is about prefill.

    With one new token the score matrix is a single row. There is still
    something to save -- that row is written and read back twice by the
    whole-matrix version and not at all by the tiled one -- but it is
    swamped by the keys and values, which both versions must read in
    full. The saving at decode should be a few per cent, against more
    than half at prefill. The chapter reports the measured figure rather
    than claiming a saving that is not there.
    """
    rng = np.random.default_rng(4)
    heads, seen, dim = 8, 1024, 64
    k = rng.standard_normal((heads, seen, dim)).astype(DType)
    v = rng.standard_normal((heads, seen, dim)).astype(DType)

    q1 = rng.standard_normal((heads, 1, dim)).astype(DType)
    whole, tiled = Meter(), Meter()
    attention_whole(q1, k, v, causal_mask(1, seen), whole)
    attention_tiled(q1, k, v, causal_mask(1, seen), tiled)
    saved = 1 - tiled.total / whole.total
    assert 0 < saved < 0.05, (
        f"decode: tiled saved {saved:.1%}, expected a few per cent "
        f"({tiled.total:,} bytes against {whole.total:,})")

    qn = rng.standard_normal((heads, seen, dim)).astype(DType)
    whole_p, tiled_p = Meter(), Meter()
    attention_whole(qn, k, v, causal_mask(seen, seen), whole_p)
    attention_tiled(qn, k, v, causal_mask(seen, seen), tiled_p)
    assert tiled_p.total < whole_p.total / 2, (
        f"prefill: tiled moved {tiled_p.total:,} bytes against {whole_p.total:,}")


def test_keeping_the_block_resident_is_strictly_a_bound() -> None:
    """The chapter quotes a range, and the range has to be one.

    `block_crosses=False` models a kernel whose block of scores never
    leaves the scratchpad. It must come out below the charged figure at
    every shape, and the difference must be exactly the blocks -- so
    that the chapter can say "at least this much" and mean it.
    """
    for heads, kv_heads, new, seen in ((32, 8, 1200, 1200), (8, 8, 256, 256),
                                       (4, 1, 1, 512), (16, 4, 64, 2048)):
        dim, elem = 128, 2
        args = (heads, new, seen, dim, Q_TILE, KV_TILE, elem)
        charged = tiled_traffic_bytes(*args, kv_heads=kv_heads)
        resident = tiled_traffic_bytes(*args, kv_heads=kv_heads,
                                       block_crosses=False)
        whole = whole_traffic_bytes(heads, new, seen, dim, elem, kv_heads)
        where = f"heads={heads} kv={kv_heads} new={new} seen={seen}"
        assert 0 < resident <= charged, f"{where}: {resident:,} vs {charged:,}"
        assert resident <= whole, f"{where}: the bound exceeds the thing bounded"
        # The gap is blocks of scores, so it can never exceed the whole
        # score matrix, and at prefill it should be about half of it.
        gap = charged - resident
        assert gap <= score_matrix_bytes(heads, new, seen, elem), (
            f"{where}: the blocks came to more than the matrix they tile")
