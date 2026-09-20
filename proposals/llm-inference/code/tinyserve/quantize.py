"""Integers instead of floats, built by hand.

Chapter 22 shrank numbers by giving them fewer bits of exponent and
mantissa, and every format there was still floating point: each number
carried its own scale. Quantization is the other move. Throw the
per-number exponent away, keep a single scale for a whole group of
numbers, and store each one as a small integer counting steps of that
scale.

Eight bits per weight instead of sixteen, and then four. The arithmetic
is simple enough to do on paper:

    scale = (the largest value) / (the largest integer)
    code  = round(value / scale)
    back  = code * scale

Everything difficult is in the words "a whole group of numbers". Which
numbers share a scale decides how much a single large value costs the
ones beside it, and that is the subject of Chapter 24.

Written for `LLM Inference from the Ground Up`, Part V.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import DType


@dataclass(frozen=True)
class Scheme:
    """One way of turning an array of floats into integers.

    `bits` is how many each code gets. `symmetric` decides whether zero
    is forced to land on a code exactly. `axis` and `group` decide how
    many numbers share a scale, which is the decision that matters.
    """

    bits: int = 8
    symmetric: bool = True
    # None: one scale for the whole tensor. An integer: one scale per
    # slice along that axis -- per output channel, for a weight matrix.
    axis: int | None = None
    # A scale per `group` consecutive values along the reduction axis,
    # which is what int4 formats do in practice.
    group: int | None = None

    @property
    def levels(self) -> int:
        return 1 << self.bits

    @property
    def qmax(self) -> int:
        """The largest code. Symmetric schemes keep one code for the
        sign, so they reach one less far in each direction."""
        return (self.levels // 2 - 1) if self.symmetric else self.levels - 1

    @property
    def qmin(self) -> int:
        return -(self.levels // 2) if self.symmetric else 0

    @property
    def name(self) -> str:
        kind = "symmetric" if self.symmetric else "asymmetric"
        if self.group:
            where = f"per group of {self.group}"
        elif self.axis is not None:
            where = "per channel"
        else:
            where = "per tensor"
        return f"int{self.bits}, {kind}, {where}"

    def bytes_per_weight(self, channels: int = 1, per_scale: int = 1) -> float:
        """Bits a weight costs, including its share of the scales.

        A scale is a float32 and a zero point an integer, and with a
        small enough group they stop being a rounding error: int4 in
        groups of 32 spends an extra bit per weight on them, which is
        25% on top of the four.
        """
        overhead_bits = 0.0
        if per_scale > 1:
            per_group = 32 + (32 if not self.symmetric else 0)
            overhead_bits = per_group / per_scale
        return (self.bits + overhead_bits) / 8


def _reshape_for(x: np.ndarray, scheme: Scheme) -> tuple[np.ndarray, tuple]:
    """Lay the array out so that everything sharing a scale is one row."""
    if scheme.group:
        # Groups run along the first axis, which for a weight matrix
        # (in_features, out_features) is the direction a dot product
        # sums over -- the direction where a shared scale is harmless.
        rows, cols = x.shape
        if rows % scheme.group:
            raise ValueError(f"{rows} values do not divide into groups of "
                             f"{scheme.group}")
        g = x.reshape(rows // scheme.group, scheme.group, cols)
        return g.transpose(0, 2, 1).reshape(-1, scheme.group), x.shape
    if scheme.axis is None:
        return x.reshape(1, -1), x.shape
    moved = np.moveaxis(x, scheme.axis, 0)
    return moved.reshape(moved.shape[0], -1), x.shape


def _restore(flat: np.ndarray, original: tuple, scheme: Scheme) -> np.ndarray:
    if scheme.group:
        rows, cols = original
        g = flat.reshape(rows // scheme.group, cols, scheme.group)
        return g.transpose(0, 2, 1).reshape(original)
    if scheme.axis is None:
        return flat.reshape(original)
    moved_shape = list(original)
    moved_shape.insert(0, moved_shape.pop(scheme.axis))
    return np.moveaxis(flat.reshape(moved_shape), 0, scheme.axis)


def quantize(x: np.ndarray, scheme: Scheme
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Turn floats into integer codes, a scale and a zero point.

    Symmetric: the scale is the largest magnitude divided by the largest
    code, and zero maps to code zero. Asymmetric: the scale spans the
    full range from smallest to largest, and the zero point records
    which code means zero.

    Which to use is a question about the numbers. Weights sit either
    side of zero and symmetric costs nothing. Anything one-sided -- the
    output of a ReLU, an attention weight after softmax -- wastes half
    its codes under a symmetric scheme.
    """
    x = np.asarray(x, dtype=np.float32)
    flat, shape = _reshape_for(x, scheme)

    if scheme.symmetric:
        biggest = np.abs(flat).max(axis=1, keepdims=True)
        scale = np.where(biggest == 0, 1.0, biggest / scheme.qmax)
        zero = np.zeros_like(scale)
    else:
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        span = hi - lo
        scale = np.where(span == 0, 1.0, span / (scheme.levels - 1))
        zero = np.round(-lo / scale)

    codes = np.clip(np.round(flat / scale) + zero, scheme.qmin, scheme.qmax)
    return codes.astype(np.int32), scale.astype(np.float32), zero.astype(np.float32)


def dequantize(codes: np.ndarray, scale: np.ndarray, zero: np.ndarray,
               shape: tuple, scheme: Scheme) -> np.ndarray:
    """Turn the codes back into floats, as the kernel does before it
    multiplies."""
    flat = (codes.astype(np.float32) - zero) * scale
    return _restore(flat, shape, scheme).astype(DType)


def round_trip(x: np.ndarray, scheme: Scheme) -> np.ndarray:
    """What a weight looks like after being stored as an integer."""
    x = np.asarray(x, dtype=np.float32)
    codes, scale, zero = quantize(x, scheme)
    return dequantize(codes, scale, zero, x.shape, scheme)


def quantize_model(model, scheme: Scheme, weights_only: bool = True):
    """A copy of the model with every weight matrix quantized.

    The norm gains and the embeddings are left alone: they are a
    vanishing share of the parameters and are known to be sensitive,
    which is what every production quantization recipe does too.
    """
    import copy

    out = copy.deepcopy(model)
    for layer in out.layers:
        for field in ("wq", "wk", "wv", "wo", "w1", "w2"):
            w = getattr(layer, field)
            setattr(layer, field, round_trip(w, scheme))
    return out


# --- tests: the specification -------------------------------------------


def test_the_codes_really_fit_in_their_bits() -> None:
    """A scheme that quantizes to int4 has to produce codes an int4 can
    hold, or the memory saving it claims is imaginary."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((64, 128)) * 0.1).astype(np.float32)
    for scheme in (Scheme(8), Scheme(8, symmetric=False), Scheme(4),
                   Scheme(4, symmetric=False), Scheme(4, group=32),
                   Scheme(8, axis=1)):
        codes, _, _ = quantize(x, scheme)
        assert codes.min() >= scheme.qmin, (scheme.name, codes.min())
        assert codes.max() <= scheme.qmax, (scheme.name, codes.max())
        distinct = len(np.unique(codes))
        assert distinct <= scheme.levels, (scheme.name, distinct)


def test_a_finer_scale_is_never_worse() -> None:
    """Splitting a tensor into more groups cannot make the error larger.

    Each group's scale is fitted to that group alone, so a finer
    division is strictly more information. If this ever fails, the
    reshaping is scrambling which values share a scale -- the one bug
    in this module that would be invisible in the error figures,
    because a wrong grouping still produces plausible-looking numbers.
    """
    rng = np.random.default_rng(1)
    x = (rng.standard_normal((256, 64)) * 0.1).astype(np.float32)
    err = lambda s: float(np.abs(round_trip(x, s) - x).mean())
    per_tensor = err(Scheme(4))
    per_channel = err(Scheme(4, axis=1))
    per_group = err(Scheme(4, group=64))
    finer_group = err(Scheme(4, group=16))
    assert per_channel <= per_tensor * 1.001, (per_channel, per_tensor)
    assert finer_group <= per_group * 1.001, (finer_group, per_group)


def test_grouping_puts_the_right_values_together() -> None:
    """The reshape has to be its own inverse.

    Checked by quantizing at a precision fine enough to be lossless and
    requiring the array back unchanged: if the grouping scrambles
    positions, the values come back transposed or shuffled and this
    fails, where an error measurement would not.
    """
    rng = np.random.default_rng(2)
    for shape in ((128, 64), (64, 32), (256, 8)):
        x = np.round(rng.standard_normal(shape) * 4).astype(np.float32)
        for scheme in (Scheme(16, group=32), Scheme(16, axis=1), Scheme(16)):
            back = round_trip(x, scheme)
            assert back.shape == x.shape, (shape, scheme.name, back.shape)
            assert np.abs(back - x).max() < 1e-3, (
                f"{scheme.name} on {shape} moved values it should have kept")


def test_zero_survives_a_symmetric_scheme() -> None:
    """Symmetric quantization must map zero to zero exactly.

    A weight of zero that comes back as anything else adds a bias to
    every dot product it takes part in, and a matrix with many zeros
    would acquire a systematic offset.
    """
    x = np.array([[0.0, 0.5, -0.5, 0.0, 1.0]], dtype=np.float32)
    for bits in (4, 8):
        back = round_trip(x, Scheme(bits))
        assert back[0][0] == 0.0 and back[0][3] == 0.0, back


def test_asymmetric_wins_on_one_sided_data() -> None:
    """The reason both schemes exist.

    On data that never goes negative, a symmetric scheme spends half
    its codes on values that cannot occur. An asymmetric one spends
    them all, and should be clearly better -- which is why activations
    after a softmax or a ReLU are quantized asymmetrically.
    """
    rng = np.random.default_rng(3)
    one_sided = np.abs(rng.standard_normal((32, 256))).astype(np.float32)
    sym = float(np.abs(round_trip(one_sided, Scheme(4)) - one_sided).mean())
    asym = float(np.abs(round_trip(one_sided, Scheme(4, symmetric=False))
                        - one_sided).mean())
    assert asym < sym / 1.5, (
        f"asymmetric should be clearly better on one-sided data: "
        f"{asym:.4g} against {sym:.4g}")

    # And on data either side of zero, the two should be close, because
    # the symmetric scheme is no longer wasting anything.
    balanced = rng.standard_normal((32, 256)).astype(np.float32)
    sym_b = float(np.abs(round_trip(balanced, Scheme(4)) - balanced).mean())
    asym_b = float(np.abs(round_trip(balanced, Scheme(4, symmetric=False))
                          - balanced).mean())
    assert 0.6 < asym_b / sym_b < 1.6, (asym_b, sym_b)
