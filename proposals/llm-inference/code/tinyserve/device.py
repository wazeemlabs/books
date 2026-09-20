"""What the machine in front of you can actually do.

Every chapter that quotes a hardware number should be able to say where
it came from. This reports what this machine is and what it achieves,
so a reader can compare their own against the book's.

    python3 -m tinyserve.device
"""

from __future__ import annotations

import os
import platform
import re
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


def caches() -> list[dict]:
    """Cache levels visible to core 0, from the kernel."""
    out, base = [], Path("/sys/devices/system/cpu/cpu0/cache")
    for entry in sorted(base.glob("index*")) if base.exists() else []:
        try:
            if (entry / "type").read_text().strip() == "Instruction":
                continue
            size = (entry / "size").read_text().strip()
            out.append({
                "level": int((entry / "level").read_text()),
                "kib": int(size.rstrip("KM")) * (1024 if size.endswith("M") else 1),
                "shared_cpu_list": (entry / "shared_cpu_list").read_text().strip(),
            })
        except (OSError, ValueError):
            continue
    return out


def name() -> str:
    try:
        m = re.search(r"^model name\s*:\s*(.+)$",
                      Path("/proc/cpuinfo").read_text(), re.M)
        if m:
            return m.group(1).strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def achieved_gflops(n: int, runs: int = 5) -> float:
    """Arithmetic rate on an n x n matrix multiply.

    Matrix multiply is the operation a model spends its time in, and how
    much of the machine it uses depends sharply on how big it is. That
    dependence is the subject of Chapter 7.
    """
    x = np.random.rand(n, n).astype(np.float32)
    y = np.random.rand(n, n).astype(np.float32)
    x @ y
    times = []
    for _ in range(runs):
        t0 = perf_counter(); x @ y; times.append(perf_counter() - t0)
    return 2 * n**3 / median(times) / 1e9


def report() -> dict:
    return {
        "name": name(),
        "cores_visible": os.cpu_count(),
        "threads_configured": int(os.environ.get("OPENBLAS_NUM_THREADS", "0")) or None,
        "caches": caches(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


if __name__ == "__main__":
    r = report()
    print(f"{r['name']}  ({r['cores_visible']} cores visible)")
    for c in r["caches"]:
        shared = "private" if "-" not in c["shared_cpu_list"] else f"shared with {c['shared_cpu_list']}"
        print(f"  L{c['level']:<2} {c['kib']:>9,} KiB  {shared}")
    for n in (64, 256, 1024):
        print(f"  {n:>5}x{n:<5} matmul: {achieved_gflops(n):8.1f} GFLOP/s")
