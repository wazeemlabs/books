"""The measurement harness every number in this book comes from.

Rules it enforces, from the book's standards:
  * warm up before timing, and never report a cold run;
  * repeat, and report the median, with the spread when it exceeds 5%;
  * report percentiles, not averages, for latency;
  * fix every seed;
  * record provenance -- hardware, software, model, commit -- next to
    the numbers, so a reader can tell what they are looking at.

Results are written as JSON. Figures and tables are generated from that
JSON; no number in the manuscript is transcribed by hand.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Sequence

import numpy as np

from tinyserve.device import name as cpu_model


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def provenance(gpu: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything a reader needs to know what produced a number.

    `gpu` is whatever `device.require_accelerator()` returned, for a
    chapter that measured one. It is not filled in automatically: a
    machine having a GPU is not the same as a chapter having used it,
    and a Tier 0 measurement that recorded the GPU sitting idle beside
    it would be claiming something it did not do.
    """
    import os
    return {
        "measured_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": {
            "cpu": cpu_model(),
            "cores_available": os.cpu_count(),
            "gpu": gpu,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "blas": (np.__config__.show(mode="dicts") or {})
                    .get("Build Dependencies", {}).get("blas", {}).get("name", "unknown"),
        },
        "commit": git_commit(),
    }


def pct(xs: Sequence[float], p: float) -> float:
    """The p-th percentile, linear interpolation, as numpy does it."""
    return float(np.percentile(np.asarray(xs, dtype=float), p)) if len(xs) else float("nan")


@dataclass
class Repeated:
    """A quantity measured several times."""

    values: list[float]

    @property
    def median(self) -> float:
        return median(self.values)

    @property
    def spread(self) -> float:
        """Half-range as a fraction of the median: the number §1.4 asks about."""
        if len(self.values) < 2 or self.median == 0:
            return 0.0
        return (max(self.values) - min(self.values)) / 2 / abs(self.median)

    @property
    def noisy(self) -> bool:
        return self.spread > 0.05

    def summary(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "spread_frac": self.spread,
            "noisy": self.noisy,
            "n": len(self.values),
            "values": self.values,
        }


def repeat(fn: Callable[[], float], warmup: int = 1, runs: int = 3) -> Repeated:
    """Warm up, then time `runs` times. Returns every value, not just the median."""
    for _ in range(warmup):
        fn()
    return Repeated([fn() for _ in range(runs)])


def write(path: str | Path, payload: dict[str, Any],
          gpu: dict[str, Any] | None = None) -> Path:
    """Write results with provenance attached.

    Pass `gpu` when the numbers came off an accelerator, so the file
    says which one.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    def plain(o: Any) -> Any:
        if is_dataclass(o) and not isinstance(o, type):
            return asdict(o)
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        raise TypeError(f"not JSON-serializable: {type(o)}")

    out.write_text(json.dumps({"provenance": provenance(gpu), **payload},
                              indent=2, default=plain) + "\n")
    return out
