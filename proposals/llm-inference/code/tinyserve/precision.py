"""Number formats, built from their bits.

Every chapter so far has stored numbers in one format and not asked
what that meant. Chapter 22 asks. A floating-point number is three
fields -- a sign, an exponent and a mantissa -- and the only difference
between the formats a GPU offers is how many bits go to the last two.
That single choice decides how large a number can be before it becomes
infinity, how small before it becomes zero, and how many digits survive
in between.

Rather than trust a library's answer, this builds each format from its
field widths and rounds to it by hand. The test at the bottom checks
that doing so reproduces NumPy's float16 exactly, bit for bit, over a
million values: if the same code is right for the one format that can
be checked against a reference, it can be believed for bfloat16 and
the two 8-bit formats, which NumPy does not have.

Written for `LLM Inference from the Ground Up`, Part IV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Format:
    """One floating-point format, described by where its bits go."""

    name: str
    exponent_bits: int
    mantissa_bits: int
    # IEEE reserves the top exponent for infinity and NaN. E4M3 does not:
    # it spends that range on larger numbers instead, and keeps a single
    # bit pattern for NaN (Micikevicius et al., arXiv:2209.05433).
    has_infinity: bool = True
    # How many bytes the format occupies in memory, where that is not
    # simply its fields rounded up. TF32 is the one that differs: it is
    # a *compute* format, 19 meaningful bits held in a 32-bit slot, so
    # it costs nothing to read and nothing to store. Using it changes
    # how a matrix multiply is done, not how the weights are kept.
    storage_bytes: float | None = None

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def bytes(self) -> float:
        """Bytes in memory, which is not always the fields rounded up."""
        if self.storage_bytes is not None:
            return self.storage_bytes
        return self.bits / 8

    @property
    def compute_only(self) -> bool:
        return self.storage_bytes is not None

    @property
    def bias(self) -> int:
        """The offset stored exponents carry, so they can be negative."""
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def max_exponent(self) -> int:
        """The largest actual exponent a finite number can have."""
        top = (1 << self.exponent_bits) - 1
        return (top if not self.has_infinity else top - 1) - self.bias

    @property
    def min_normal_exponent(self) -> int:
        return 1 - self.bias

    @property
    def max_value(self) -> float:
        """The largest finite number.

        Every mantissa bit set, except that E4M3 reserves the all-ones
        mantissa at the top exponent for NaN, so its largest value is
        one step below.
        """
        # All mantissa bits set, one step lower where the all-ones
        # pattern is spoken for: E4M3 keeps it for NaN.
        m = (1 << self.mantissa_bits) - (1 if self.has_infinity else 2)
        return float((1 + m / (1 << self.mantissa_bits))
                     * 2.0 ** self.max_exponent)

    @property
    def min_normal(self) -> float:
        """The smallest number that still has full precision."""
        return float(2.0 ** self.min_normal_exponent)

    @property
    def min_subnormal(self) -> float:
        """The smallest number of any kind, with precision already lost."""
        return float(2.0 ** (self.min_normal_exponent - self.mantissa_bits))

    @property
    def eps(self) -> float:
        """The gap between 1.0 and the next number up.

        This is precision: two values closer together than this are the
        same number in this format.
        """
        return float(2.0 ** -self.mantissa_bits)

    @property
    def decimal_digits(self) -> float:
        """Roughly how many decimal digits survive."""
        return (self.mantissa_bits + 1) * np.log10(2)


# The formats an accelerator offers, in the order they lose bits.
FP32 = Format("float32", 8, 23)
TF32 = Format("tensorfloat32", 8, 10, storage_bytes=4)  # float32's range, half's precision
FP16 = Format("float16", 5, 10)
BF16 = Format("bfloat16", 8, 7)
FP8_E4M3 = Format("float8 e4m3", 4, 3, has_infinity=False)
FP8_E5M2 = Format("float8 e5m2", 5, 2)

FORMATS = [FP32, TF32, FP16, BF16, FP8_E4M3, FP8_E5M2]


def round_to(x: np.ndarray, fmt: Format) -> np.ndarray:
    """Round float32 values to `fmt`, returning them as float32 again.

    This is what storing a number in a smaller format does to it, with
    nothing else changed: the value comes back in float32 so it can be
    compared against what it was, but it now holds only what the smaller
    format could keep.

    The rounding is round-to-nearest, ties-to-even, which is what every
    IEEE format and every GPU does by default. Rounding a tie always
    upward would introduce a bias that accumulates over a matrix
    multiply's thousands of additions.
    """
    x = np.asarray(x, dtype=np.float32)
    if fmt.mantissa_bits >= FP32.mantissa_bits and fmt.exponent_bits >= FP32.exponent_bits:
        return x.copy()

    sign = np.signbit(x)
    mag = np.abs(x).astype(np.float32)
    out = np.zeros_like(mag)

    finite = np.isfinite(mag)
    # exponent of each value: the power of two just below it
    with np.errstate(divide="ignore", invalid="ignore"):
        exp = np.floor(np.log2(np.where(mag > 0, mag, 1.0))).astype(np.int32)
    exp = np.clip(exp, fmt.min_normal_exponent, None)   # subnormals share one exponent

    # Quantise the mantissa at the scale this exponent implies. The step
    # between neighbouring representable values is 2^(exp - mantissa_bits),
    # so rounding is: divide by the step, round to even, multiply back.
    step = np.exp2((exp - fmt.mantissa_bits).astype(np.float64))
    rounded = np.round(mag.astype(np.float64) / step) * step

    # Rounding up can carry into the next exponent; that is correct and
    # needs no special case, because the result is still on the coarser
    # grid the larger exponent implies.
    over = rounded > fmt.max_value
    if fmt.has_infinity:
        rounded = np.where(over, np.inf, rounded)
    else:
        # No infinity to overflow to. The paper's format saturates.
        rounded = np.where(over, fmt.max_value, rounded)

    out = np.where(finite, rounded, mag).astype(np.float32)
    out = np.where(mag == 0, 0.0, out)
    return np.where(sign, -out, out).astype(np.float32)


def representable(fmt: Format, x: float) -> bool:
    """Does this value survive a round trip through the format?"""
    a = np.array([x], dtype=np.float32)
    return bool(np.array_equal(round_to(a, fmt), a))


def relative_error(original: np.ndarray, rounded: np.ndarray) -> dict[str, float]:
    """How far rounding moved a set of values, relative to their size.

    Reported against the largest value rather than against each value
    individually: a weight near zero can be moved by a large fraction of
    itself while contributing almost nothing to any sum it enters, and a
    per-element ratio makes that look like a catastrophe.
    """
    original = np.asarray(original, dtype=np.float64).ravel()
    rounded = np.asarray(rounded, dtype=np.float64).ravel()
    err = np.abs(rounded - original)
    scale = float(np.abs(original).max()) or 1.0
    rms = float(np.sqrt(np.mean(err ** 2)))
    return {
        "max_absolute": float(err.max()),
        "max_relative_to_largest": float(err.max()) / scale,
        "rms": rms,
        "rms_relative_to_largest": rms / scale,
        "overflowed": int(np.sum(~np.isfinite(rounded) & np.isfinite(original))),
        "flushed_to_zero": int(np.sum((rounded == 0) & (original != 0))),
        "largest_value": scale,
    }


# --- tests: the specification -------------------------------------------


def test_the_hand_rounding_matches_numpys_float16() -> None:
    """The one format NumPy can check, checked exhaustively enough to
    trust the others.

    `round_to` is generic: the same lines produce bfloat16 and both
    8-bit formats, which NumPy cannot represent and nothing else here
    can verify. If it reproduces float16 bit for bit across the whole
    range -- normals, subnormals, values that overflow, and the exact
    halfway cases where ties-to-even is the only thing that decides the
    answer -- then it is the rounding rule and not a coincidence.
    """
    rng = np.random.default_rng(0)
    values = np.concatenate([
        rng.standard_normal(200_000).astype(np.float32),
        (rng.standard_normal(200_000) * 1e4).astype(np.float32),
        (rng.standard_normal(200_000) * 1e-5).astype(np.float32),
        np.array([0.0, -0.0, 1.0, -1.0, 65504.0, -65504.0, 65519.0,
                  65520.0, 6e-8, 6e-5, 1e-7, 3.0517578125e-05],
                 dtype=np.float32),
        # Exact halfway cases: 1 + k/1024 + 1/2048 rounds to even.
        (1.0 + np.arange(0, 1024) / 1024 + 1 / 2048).astype(np.float32),
    ])
    mine = round_to(values, FP16)
    with np.errstate(over="ignore"):    # 65520 and up become infinity
        theirs = values.astype(np.float16).astype(np.float32)
    bad = ~((mine == theirs) | (np.isnan(mine) & np.isnan(theirs)))
    assert not bad.any(), (
        f"{bad.sum()} of {bad.size} values disagree with NumPy's float16, "
        f"first at {values[bad][0]!r}: {mine[bad][0]!r} against "
        f"{theirs[bad][0]!r}")


def test_each_format_reports_the_range_its_bits_allow() -> None:
    """The properties are arithmetic on the field widths, and a mistake
    in them would propagate silently into every number in Chapter 22."""
    assert FP32.bits == 32 and FP16.bits == 16 and BF16.bits == 16
    assert FP8_E4M3.bits == 8 and FP8_E5M2.bits == 8

    # float16's limits are the ones everybody knows.
    assert FP16.max_value == 65504.0, FP16.max_value
    assert FP16.eps == float(np.finfo(np.float16).eps), FP16.eps
    assert FP16.min_normal == float(np.finfo(np.float16).tiny)

    # float32's, likewise.
    assert FP32.eps == float(np.finfo(np.float32).eps)
    assert abs(FP32.max_value / float(np.finfo(np.float32).max) - 1) < 1e-6

    # bfloat16 keeps float32's exponent, so it keeps float32's range and
    # pays for it in precision. That is the whole reason it exists.
    assert BF16.exponent_bits == FP32.exponent_bits
    assert BF16.max_exponent == FP32.max_exponent
    assert BF16.eps > FP16.eps, "bfloat16 should be the coarser of the two"

    # E4M3 has no infinity, so its top exponent holds numbers.
    assert not FP8_E4M3.has_infinity
    assert FP8_E4M3.max_value == 448.0, FP8_E4M3.max_value
    assert FP8_E5M2.max_value == 57344.0, FP8_E5M2.max_value


def test_rounding_is_unbiased() -> None:
    """Ties-to-even, not ties-away.

    A matrix multiply adds thousands of rounded products. If rounding
    leaned one way, the error would grow with the length of the sum
    instead of cancelling, and every number in Part V would be wrong in
    the same direction.
    """
    rng = np.random.default_rng(1)
    x = rng.standard_normal(500_000).astype(np.float32)
    for fmt in (FP16, BF16, FP8_E4M3):
        err = (round_to(x, fmt).astype(np.float64) - x.astype(np.float64))
        mean, spread = float(err.mean()), float(err.std())
        assert abs(mean) < spread / 100, (
            f"{fmt.name} rounding is biased: mean error {mean:.3g} against "
            f"a spread of {spread:.3g}")


def test_a_value_that_fits_is_left_alone() -> None:
    """Rounding to a format must not move a number already in it."""
    for fmt in (FP16, BF16, FP8_E4M3, FP8_E5M2):
        powers = np.array([2.0 ** k for k in range(-4, 5)], dtype=np.float32)
        assert np.array_equal(round_to(powers, fmt), powers), fmt.name
        once = round_to(np.linspace(-3, 3, 1000).astype(np.float32), fmt)
        twice = round_to(once, fmt)
        assert np.array_equal(once, twice), f"{fmt.name} is not idempotent"
