"""Chapter 21: what it costs to ask the GPU to do something.

Chapter 20 counted the bytes attention moves. This counts something
else entirely: the fixed price of each instruction handed to the
accelerator, regardless of how much work that instruction carries.

Five measurements, all on the GPU this machine has:

1. Dispatch. What one operation costs before any arithmetic happens,
   measured as the gradient of time against operation count rather
   than by timing one and hoping.
2. The flat region. Matrix multiplies from tiny to large. Below a
   certain size they all take the same time, because none of them is
   doing enough work to matter beside the cost of being asked.
3. Operations per forward pass, counted exactly, and split into those
   that reach the GPU and those that only relabel a tensor.
4. Eager against compiled. The same transformer, with the operations
   fused by `torch.compile`.
5. What is left. On CUDA a captured graph removes the remaining
   per-operation cost; this backend has no capture API, which is
   measured here rather than assumed.

    python3 -m bench.run_ch21              measure
    python3 -m bench.run_ch21 --dry-run    check the code path, write nothing

The dry run exists because dispatch costs microseconds and this
measurement refuses to run on a machine with other work on it.

    python3 -m bench.run_ch21
"""

from __future__ import annotations

import sys

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tinyserve import device, model as m, torch_reference as tr
from tinyserve.reference import MODEL, CONTEXT_TOKENS

from .harness import write

# The model the chapter runs: small enough that a forward pass is
# dominated by the number of operations rather than by their size,
# which is the regime this chapter is about.
CFG = m.Config()
SEED = 0
DTYPE = torch.float32

# Matrix sizes for the flat region. The interesting part is the bottom.
SIZES = [16, 32, 64, 128, 256, 512, 1024, 2048]
# Chain lengths the dispatch cost is regressed over.
COUNTS = (100, 200, 400, 800)

# Operations that only change how a tensor is described -- its shape,
# its strides, which way round it is read -- and produce no work for
# the GPU at all. Counted separately, because including them would
# inflate every per-operation figure in the chapter.
METADATA_OPS = {
    "aten.view.default", "aten.transpose.int", "aten.t.default",
    "aten.expand.default", "aten.permute.default", "aten.detach.default",
    "aten.lift_fresh.default", "aten.reshape.default", "aten.squeeze.dim",
    "aten.unsqueeze.default", "aten.slice.Tensor", "aten.select.int",
    "aten._unsafe_view.default", "aten.alias.default",
}


class CountOps(TorchDispatchMode):
    """Every operation PyTorch dispatches, by name.

    This counts what the framework was asked to do, which is the thing
    the chapter is about: each dispatch is a decision, a lookup and,
    for most of them, an instruction sent to the accelerator.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[str(func)] = self.counts.get(str(func), 0) + 1
        return func(*args, **(kwargs or {}))

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    @property
    def metadata(self) -> int:
        return sum(n for k, n in self.counts.items() if k in METADATA_OPS)

    @property
    def work(self) -> int:
        return self.total - self.metadata


def dispatch_cost(acc: dict, counts=COUNTS) -> dict:
    """What one operation costs, by kind, before arithmetic.

    Each operation is applied in place to the same small tensor, so no
    memory is allocated inside the loop and nothing but the dispatch
    and the work itself is being timed. The tensor is small enough that
    the work is negligible, which is the point: what is left is the
    price of asking.
    """
    d = torch.device(acc["device"])
    rows = []
    for name, size, op in (
        ("add, in place", 64, lambda t: t.add_(1.0)),
        ("multiply, in place", 64, lambda t: t.mul_(1.0)),
        ("tanh, in place", 64, lambda t: t.tanh_()),
        ("add, in place, 1024x1024", 1024, lambda t: t.add_(1.0)),
        ("transpose (metadata only)", 64, lambda t: t.t()),
        ("view (metadata only)", 64, lambda t: t.view(-1)),
    ):
        x = torch.randn(size, size, device=d, dtype=DTYPE)

        def run(n: int, x=x, op=op) -> None:
            for _ in range(n):
                op(x)

        fit = device.marginal(run, acc, counts)
        rows.append({
            "operation": name, "size": size,
            "us_per_op": fit["seconds_per_op"] * 1e6,
            "fixed_us": fit["fixed_seconds"] * 1e6,
            "r2": fit["r2"],
            "metadata_only": "metadata" in name,
            "trustworthy": fit["r2"] >= 0.99,
        })
    work = [r for r in rows if not r["metadata_only"] and r["size"] == 64]
    meta = [r for r in rows if r["metadata_only"]]
    return {
        "rows": rows,
        "dispatch_us": min(r["us_per_op"] for r in work),
        "metadata_us": max(r["us_per_op"] for r in meta),
        "counts": list(counts),
    }


def size_sweep(acc: dict) -> list[dict]:
    """Where the arithmetic finally outgrows the cost of asking for it.

    Each multiply writes into a preallocated output, so the loop
    allocates nothing and the only difference between one row and the
    next is how much arithmetic the same single instruction carries.
    """
    d = torch.device(acc["device"])
    rows = []
    for n in SIZES:
        a = torch.randn(n, n, device=d, dtype=DTYPE)
        out = torch.empty(n, n, device=d, dtype=DTYPE)

        def run(k: int, a=a, out=out) -> None:
            for _ in range(k):
                torch.matmul(a, a, out=out)

        # Large multiplies are slow enough that fewer of them suffice.
        counts = COUNTS if n <= 256 else (10, 20, 40, 80)
        fit = device.marginal(run, acc, counts)
        per = fit["seconds_per_op"]
        rows.append({
            "n": n, "us_per_op": per * 1e6, "r2": fit["r2"],
            "flops": 2 * n**3, "tflops": 2 * n**3 / per / 1e12,
            "trustworthy": fit["r2"] >= 0.99,
        })
    return rows


def ops_per_forward() -> dict:
    """How many operations one forward pass of a transformer dispatches."""
    net = m.build(CFG, seed=SEED)
    tokens = torch.from_numpy((np.arange(16) % CFG.vocab_size).astype(np.int64))
    weights = tr.on_device(net, "cpu")
    with CountOps() as c:
        tr.forward(net, tokens, weights)
    by_name = sorted(c.counts.items(), key=lambda kv: -kv[1])
    return {
        "layers": CFG.n_layers,
        "total": c.total,
        "metadata": c.metadata,
        "work": c.work,
        "work_per_layer": c.work / CFG.n_layers,
        "busiest": [{"op": k, "n": v} for k, v in by_name[:8]],
    }


def eager_vs_compiled(acc: dict, tokens: int = 16) -> dict:
    """One transformer layer, operation by operation and fused.

    A layer is the unit, not the whole model: `torch.compile` traces a
    graph of tensor operations, and the loop over layers in `forward`
    carries a dataclass of NumPy arrays that it cannot see through.
    Compiling the layer is also what a serving engine does.
    """
    d = torch.device(acc["device"])
    net = m.build(CFG, seed=SEED)
    w = tr.on_device(net, d)
    layer = net.layers[0]
    args = [w[id(a)] for a in (layer.g1, layer.wq, layer.wk, layer.wv,
                               layer.wo, layer.g2, layer.w1, layer.w2)]
    x = torch.randn(tokens, CFG.d_model, device=d, dtype=DTYPE)
    mask = torch.full((tokens, tokens), float("-inf"),
                      device=d, dtype=DTYPE).triu(1)
    shape = (CFG.n_heads, CFG.n_kv_heads, CFG.head_dim)

    def eager() -> torch.Tensor:
        return tr.block(x, *args, mask, *shape)

    compiled_block = torch.compile(tr.block)

    def compiled() -> torch.Tensor:
        return compiled_block(x, *args, mask, *shape)

    a, b = eager(), compiled()
    device.synchronize(acc)
    gap = float((a - b).abs().max())
    scale = float(a.abs().max())

    with CountOps() as c:
        eager()
    t_eager = device.timed(eager, acc, warmup=20, runs=11)
    t_compiled = device.timed(compiled, acc, warmup=20, runs=11)
    return {
        "eager_us": t_eager * 1e6,
        "compiled_us": t_compiled * 1e6,
        "speedup": t_eager / t_compiled,
        "ops_eager": c.total,
        "work_ops_eager": c.work,
        "max_diff": gap,
        "scale": scale,
        "relative_diff": gap / scale if scale else 0.0,
        "tokens": tokens,
        "saved_us": (t_eager - t_compiled) * 1e6,
        "us_per_op_removed": ((t_eager - t_compiled) * 1e6 / c.work
                              if c.work else 0.0),
    }


def reference_model() -> dict:
    """Operations per token for the model the case study serves.

    This is architecture, not a measurement: the same count of matrix
    multiplies, norms and additions per layer that the model in this
    chapter dispatches, at the reference model's depth.
    """
    per_layer = ops_per_forward()["work"] / CFG.n_layers
    return {
        "layers": MODEL.n_layers,
        "work_ops_per_layer": per_layer,
        "work_ops_per_token": per_layer * MODEL.n_layers,
        "context_tokens": CONTEXT_TOKENS,
    }


def main(argv: list[str]) -> None:
    dry = "--dry-run" in argv
    acc = device.require_accelerator()
    quiet = device.load()
    contention = None
    if not dry:
        device.require_quiet_machine()
        # And the accelerator itself. A quiet processor is not a quiet
        # GPU: another process can own the device while every core
        # sits idle, and the only sign of it in the numbers is a fit
        # that will not settle.
        contention = device.require_quiet_accelerator(acc)

    counts = (10, 20) if dry else COUNTS
    dispatch = dispatch_cost(acc, counts)
    sizes = [] if dry else size_sweep(acc)
    ops = ops_per_forward()
    fused = eager_vs_compiled(acc)
    ref = reference_model()

    payload = {
        "accelerator": acc,
        "accelerator_contention": contention,
        "machine_load": quiet,
        "dispatch": dispatch,
        "sizes": sizes,
        "ops": ops,
        "fused": fused,
        "reference": ref,
        "graph_capture": {
            "available": acc["graph_capture"],
            "backend": acc["kind"],
            # Recorded rather than remembered: the chapter says this
            # backend cannot capture a graph, so it checks.
            "torch_cuda_has_capture": hasattr(torch.cuda, "CUDAGraph"),
        },
        "assumptions": {
            "model": {"layers": CFG.n_layers, "d_model": CFG.d_model,
                      "heads": CFG.n_heads, "kv_heads": CFG.n_kv_heads},
            "dtype": str(DTYPE), "seed": SEED,
            "counts": list(counts), "sizes": SIZES,
            "reference_layers": MODEL.n_layers,
        },
    }
    if dry:
        print("dry run: nothing written")
        print(f"  accelerator {acc['kind']} ({acc['name']}), "
              f"graph capture: {acc['graph_capture']}")
        print(f"  dispatch rows: {len(dispatch['rows'])}, "
              f"ops per forward: {ops['total']} "
              f"({ops['work']} reach the GPU)")
        print(f"  one layer: eager {fused['eager_us']:.1f} us over "
              f"{fused['work_ops_eager']} GPU operations vs compiled "
              f"{fused['compiled_us']:.1f} us ({fused['speedup']:.2f}x)")
        return

    path = write("results/ch21.json", payload, gpu=acc)
    print(f"wrote {path}")
    print(f"  {acc['name']} ({acc['kind']}), load "
          f"{quiet['load_1min']:.2f} over {quiet['cores']} cores")
    print("  cost of one operation:")
    for r in dispatch["rows"]:
        flag = "" if r["trustworthy"] else "   POOR FIT"
        print(f"    {r['operation']:28} {r['us_per_op']:7.2f} us   "
              f"r2 {r['r2']:.4f}{flag}")
    print("  one matrix multiply, by size:")
    for r in sizes:
        print(f"    {r['n']:5d}: {r['us_per_op']:8.2f} us  "
              f"{r['tflops']:7.3f} TFLOP/s  r2 {r['r2']:.4f}")
    print(f"  a {ops['layers']}-layer forward dispatches {ops['total']} "
          f"operations, {ops['work']} of which reach the GPU "
          f"({ops['work_per_layer']:.0f} a layer)")
    print(f"  one layer: eager {fused['eager_us']:.1f} us over "
          f"{fused['work_ops_eager']} GPU operations, compiled "
          f"{fused['compiled_us']:.1f} us ({fused['speedup']:.2f}x, "
          f"{fused['saved_us']:.1f} us saved), largest difference "
          f"{fused['relative_diff']:.1e} relative")
    print(f"  graph capture on this backend: "
          f"{'yes' if acc['graph_capture'] else 'no'}")


if __name__ == "__main__":
    main(sys.argv[1:])
