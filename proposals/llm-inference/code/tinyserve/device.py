"""What the machine in front of you can actually do.

Every chapter that quotes a hardware number should be able to say where
it came from. This reports what this machine is and what it achieves,
so a reader can compare their own against the book's.

From Part IV the book also measures a GPU, so this reports the
accelerator as well -- and, more importantly, refuses to let a chapter
measure one it is not actually using. `require_accelerator` is the
guard: PyTorch's Metal backend will quietly run an unsupported
operation on the CPU and hand back a tensor that still claims to live
on the GPU, if it is asked to. A number measured that way is a CPU
number wearing a GPU label, and nothing downstream would notice.

    python3 -m tinyserve.device
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np


def _sysctl(key: str) -> str | None:
    """One value from macOS's sysctl, or None anywhere else."""
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", key], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def caches() -> list[dict]:
    """Cache levels visible to core 0, from the kernel."""
    if platform.system() == "Darwin":
        out = []
        for level, key in ((1, "hw.l1dcachesize"), (2, "hw.l2cachesize"),
                           (3, "hw.l3cachesize")):
            size = _sysctl(key)
            if size and size.isdigit() and int(size) > 0:
                out.append({"level": level, "kib": int(size) // 1024,
                            "shared_cpu_list": "unknown"})
        return out
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
    """The processor, as the machine names it.

    Falling back to `platform.processor()` reports "arm" on macOS, which
    is not a provenance line anybody can check a number against.
    """
    brand = _sysctl("machdep.cpu.brand_string")
    if brand:
        return brand
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


# --- the accelerator, from Part IV ---------------------------------------

# Setting this tells PyTorch's Metal backend to run an operation it has
# no kernel for on the CPU instead of refusing. That is a reasonable
# thing to want and a fatal thing to measure through, because the
# result comes back on the GPU's device and nothing distinguishes it
# from work the GPU did.
FALLBACK_VAR = "PYTORCH_ENABLE_MPS_FALLBACK"


def accelerator() -> dict[str, Any] | None:
    """What GPU this machine has that the book can measure, if any.

    Returns None rather than raising when there is none, because most of
    this book runs without one and should keep doing so.
    """
    try:
        import torch
    except ImportError:
        return None
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        return {
            "kind": "cuda", "device": "cuda",
            "name": props.name,
            "memory_bytes": props.total_memory,
            "multiprocessors": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
            "graph_capture": hasattr(torch.cuda, "CUDAGraph"),
            "torch": torch.__version__,
        }
    if torch.backends.mps.is_available():
        return {
            "kind": "mps", "device": "mps",
            # Metal reports no model string through torch, so the chip is
            # the processor: on Apple Silicon they are the same package.
            "name": name(),
            "memory_bytes": torch.mps.recommended_max_memory(),
            "multiprocessors": None,     # not exposed by the backend
            "capability": platform.mac_ver()[0] or None,
            # There is no capture API on torch.mps at all. Chapter 21
            # depends on being able to say that, so it is measured here
            # rather than remembered.
            "graph_capture": any("graph" in a.lower() or "capture" in a.lower()
                                 for a in dir(torch.mps)),
            "torch": torch.__version__,
        }
    return None


def require_accelerator() -> dict[str, Any]:
    """The guard every GPU measurement in this book goes through.

    Three things have to be true before a number off a GPU means
    anything, and all three are cheap to check and expensive to get
    wrong:

    1. There is a GPU.
    2. The silent-fallback switch is off, so an operation with no GPU
       kernel raises instead of running on the CPU and returning a
       tensor that still says "gpu".
    3. Arithmetic actually runs there and comes back with the right
       answer, which catches a backend that is present but broken.
    """
    import torch

    acc = accelerator()
    if acc is None:
        raise RuntimeError(
            "no accelerator: this measurement needs a GPU. Every chapter "
            "before Part IV runs without one.")
    if os.environ.get(FALLBACK_VAR):
        raise RuntimeError(
            f"{FALLBACK_VAR} is set. It makes an operation with no GPU "
            "kernel run on the CPU and return a tensor that still reports "
            "the GPU as its device, so a measurement taken with it on "
            "cannot be trusted to have measured the GPU. Unset it.")

    d = torch.device(acc["device"])
    a = torch.arange(64, dtype=torch.float32, device=d).reshape(8, 8)
    out = a @ a
    synchronize(acc)
    if out.device.type != acc["device"]:
        raise RuntimeError(f"a matmul on {acc['device']} came back on "
                           f"{out.device}")
    expect = np.arange(64, dtype=np.float32).reshape(8, 8)
    expect = expect @ expect
    got = out.cpu().numpy()
    if not np.allclose(got, expect, rtol=1e-5, atol=1e-4):
        raise RuntimeError(f"the accelerator computed a matmul wrongly: "
                           f"largest difference {abs(got - expect).max():.3g}")
    return acc


# A dispatch costs single-digit microseconds. A machine with other work
# on it produces timings whose noise is larger than that, and the
# regression below will happily fit a line through them and report a
# number. This is the load above which it should not be believed: one
# busy core per core the machine has is already generous.
BUSY_LOAD_PER_CORE = 0.5


def load() -> dict[str, Any]:
    """How busy the machine is, and whether that is too busy to time on."""
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return {"available": False}
    cores = os.cpu_count() or 1
    return {
        "available": True,
        "load_1min": one, "load_5min": five, "load_15min": fifteen,
        "cores": cores,
        "per_core": one / cores,
        "busy": one / cores > BUSY_LOAD_PER_CORE,
    }


def require_quiet_machine() -> dict[str, Any]:
    """Refuse to take a microsecond measurement on a loaded machine.

    Chapter 9 asks for measurements that can be trusted, and nothing
    about a timing says on its face that something else was using the
    processor while it ran. This says it.
    """
    now = load()
    if now.get("busy"):
        raise RuntimeError(
            f"the machine is busy: load {now['load_1min']:.1f} over "
            f"{now['cores']} cores ({now['per_core']:.2f} per core, "
            f"above {BUSY_LOAD_PER_CORE}). Dispatch costs a few "
            "microseconds and contention is worth more than that, so a "
            "number taken now would be noise. Wait, or stop whatever "
            "else is running.")
    return now


def synchronize(acc: dict[str, Any] | None = None) -> None:
    """Wait for the accelerator to finish what it was given.

    GPU work is queued, not run, by the call that submits it. Timing
    without this measures how long it takes to ask.
    """
    import torch

    acc = acc or accelerator()
    if acc is None:
        return
    (torch.cuda if acc["kind"] == "cuda" else torch.mps).synchronize()


def timed(fn: Callable[[], Any], acc: dict[str, Any], warmup: int = 20,
          runs: int = 9) -> float:
    """Median seconds for `fn`, with the accelerator drained each time.

    Warm up first: the first call to a shape compiles or caches a kernel,
    and that cost belongs to neither the launch nor the arithmetic.
    """
    for _ in range(warmup):
        fn()
    synchronize(acc)
    times = []
    for _ in range(runs):
        t0 = perf_counter()
        fn()
        synchronize(acc)
        times.append(perf_counter() - t0)
    return median(times)


def marginal(fn: Callable[[int], Any], acc: dict[str, Any],
             counts: Sequence[int] = (100, 200, 400, 800),
             warmup: int = 10, runs: int = 7) -> dict[str, float]:
    """Seconds for one more of whatever `fn(n)` does n of.

    Timing one operation measures mostly the cost of asking the GPU
    whether it has finished, which is far larger than the operation
    itself. The fixed cost has to be removed, and the obvious way --
    time n and 2n, subtract, divide -- turns out to be too fragile to
    use: on a laptop the noise in two timings is comparable to the few
    microseconds being measured, and the answer comes out negative often
    enough to be useless.

    So the cost is a slope instead. Time several sizes, fit a line
    through them, and take its gradient. Every sample constrains the
    answer, the fixed cost falls out as the intercept, and the quality
    of the fit says whether the model -- a constant cost per operation
    -- describes what happened at all. `r2` below 0.99 means it does
    not, and the number should not be quoted.

    Each timing uses the minimum rather than the median, which is the
    right summary for a latency floor: a run can be interrupted and come
    out slow, but nothing makes it come out faster than the machine can
    go.
    """
    xs, ys = [], []
    for n in counts:
        for _ in range(warmup):
            fn(n)
        synchronize(acc)
        best = float("inf")
        for _ in range(runs):
            t0 = perf_counter()
            fn(n)
            synchronize(acc)
            best = min(best, perf_counter() - t0)
        xs.append(float(n)); ys.append(best)

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return {
        "seconds_per_op": slope,
        "fixed_seconds": intercept,
        "r2": 1 - ss_res / ss_tot if ss_tot else float("nan"),
        "counts": list(counts),
        "seconds": ys,
    }


def report() -> dict:
    return {
        "name": name(),
        "cores_visible": os.cpu_count(),
        "threads_configured": int(os.environ.get("OPENBLAS_NUM_THREADS", "0")) or None,
        "caches": caches(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "accelerator": accelerator(),
    }


if __name__ == "__main__":
    r = report()
    print(f"{r['name']}  ({r['cores_visible']} cores visible)")
    for c in r["caches"]:
        shared = "private" if "-" not in c["shared_cpu_list"] else f"shared with {c['shared_cpu_list']}"
        print(f"  L{c['level']:<2} {c['kib']:>9,} KiB  {shared}")
    for n in (64, 256, 1024):
        print(f"  {n:>5}x{n:<5} matmul: {achieved_gflops(n):8.1f} GFLOP/s")
    acc = r["accelerator"]
    if acc is None:
        print("  no accelerator (every chapter before Part IV runs without one)")
    else:
        print(f"  {acc['kind']}: {acc['name']}, "
              f"{acc['memory_bytes'] / 1e9:.0f} GB, torch {acc['torch']}, "
              f"graph capture: {'yes' if acc['graph_capture'] else 'no'}")


def test_the_accelerator_guard_refuses_what_it_should() -> None:
    """The guard has to behave on a machine with no GPU as well as on one
    with a GPU, because the book is built on both.

    On a machine without one, `accelerator()` reports None and
    `require_accelerator()` says so rather than pretending. On a machine
    with one, the guard passes only when the silent-fallback switch is
    off -- and this checks that by turning it on and requiring a
    refusal, which is the failure that would otherwise go unnoticed.
    """
    acc = accelerator()

    if acc is None:
        try:
            require_accelerator()
        except RuntimeError as e:
            assert "no accelerator" in str(e), f"wrong refusal: {e}"
        except ImportError:
            pass                        # torch is not installed either
        else:
            raise AssertionError("require_accelerator passed with no GPU")
        return

    for key in ("kind", "device", "name", "memory_bytes", "graph_capture"):
        assert key in acc, f"the accelerator report is missing {key}"
    assert acc["kind"] in ("cuda", "mps"), acc["kind"]
    assert acc["memory_bytes"] > 0, acc["memory_bytes"]

    require_accelerator()               # must pass with the switch off

    was = os.environ.get(FALLBACK_VAR)
    os.environ[FALLBACK_VAR] = "1"
    try:
        require_accelerator()
    except RuntimeError as e:
        assert FALLBACK_VAR in str(e), f"wrong refusal: {e}"
    else:
        raise AssertionError(
            "the guard passed with the silent-fallback switch on; a "
            "measurement taken that way could be the CPU's")
    finally:
        if was is None:
            del os.environ[FALLBACK_VAR]
        else:
            os.environ[FALLBACK_VAR] = was
