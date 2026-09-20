"""Chapter 16: batching, built and measured.

One sequence at a time reads every weight in the model to produce one
token. A batch reads them once and produces B. This measures what that
is worth, and what it costs:

1. It must not change the tokens -- and the scores turn out to depend
   on the batch anyway, for a reason worth knowing.
2. Step time and throughput against batch size, on a model small enough
   to sit in cache and one large enough not to. The two behave
   differently, and the roofline says why.
3. What a user pays for the throughput: time between their tokens.
4. What a *static* batch wastes -- padding at the front, dead slots at
   the back -- which is what Chapter 17 exists to fix.

    python3 -m bench.run_ch16
"""

from __future__ import annotations

import math
from statistics import median
from time import perf_counter

import numpy as np

from tinyserve.batch import (agreement_threshold, decode_step,
                             same_as_unbatched)
from tinyserve.cost import kv_bytes_per_token
from tinyserve.model import Config, KVCache, build, forward
from tinyserve.reference import RIDGE_FLOP_PER_BYTE
from tinyserve.serving import decode_step as modelled_step

from .harness import Repeated, write
from .machine import matmul_flops, stream_bandwidth

BATCHES = [1, 2, 4, 8, 16, 32, 64]
PROMPT, STEPS, RUNS, WARMUP = 16, 6, 3, 1
AGREEMENT_SIZES = [1, 2, 3, 4, 8, 16, 32]

# The two ends of Chapter 4's wall: one model whose weights fit in this
# machine's cache, one whose weights do not.
SMALL = Config()
LARGE = Config(vocab_size=512, d_model=1280, n_layers=10, n_heads=20,
               n_kv_heads=20, d_ff=5120, max_seq=64)

# The case study's traffic (STANDARDS.md section 7).
PROMPT_MEAN, PROMPT_CV, OUTPUT_MEAN, OUTPUT_CV = 1200, 0.6, 300, 0.8
N_REQUESTS, SEED = 4000, 0
CASE_REQUESTS_PER_S, CASE_OUTPUT_TOKENS = 200, 300


def lognormal(mean: float, cv: float, n: int, rng) -> np.ndarray:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return np.maximum(1, rng.lognormal(mu, sigma, n).round()).astype(int)


# --- 1. does it change the answer? --------------------------------------


def equivalence(model, cfg: Config) -> dict:
    rng = np.random.default_rng(1)
    prompts = [rng.integers(0, cfg.vocab_size, n).tolist()
               for n in (12, 20, 33, 7)]
    same, _ = same_as_unbatched(model, prompts, n_new=8)

    steady = [rng.integers(0, cfg.vocab_size, 24).tolist()
              for _ in range(max(AGREEMENT_SIZES))]
    return {"same_tokens": same, "tokens_compared": 8,
            "sequences": len(prompts),
            "agreement": agreement_threshold(model, steady, AGREEMENT_SIZES)}


# --- 2. the sweep -------------------------------------------------------


def cache_bytes() -> int:
    """The largest cache level this machine has, from the kernel.

    A weight matrix small enough to sit in it is not fetched from
    memory at all, however large it looks, so a batching measurement
    that uses one is measuring the wrong thing.
    """
    from pathlib import Path
    best = 0
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    for entry in sorted(base.glob("index*")) if base.exists() else []:
        try:
            size = (entry / "size").read_text().strip()
            best = max(best, int(size.rstrip("KM"))
                       * (1024**2 if size.endswith("M") else 1024))
        except (OSError, ValueError):
            continue
    return best


def one_matmul(cfg: Config, name: str, working_set_bytes: int,
               runs: int = 7) -> dict:
    """The decode step's largest weight matmul, on its own, at every batch.

    Everything else stripped away: weights read, then multiplied by B
    rows. The matrices are cycled through a working set the size of the
    model they came from, because a single matrix multiplied over and
    over would sit in cache and never be fetched at all -- which is the
    one thing this measurement must not let happen.
    """
    rng = np.random.default_rng(7)
    one = cfg.d_model * cfg.d_ff * 4
    n_mats = max(1, round(working_set_bytes / one))
    mats = [rng.random((cfg.d_model, cfg.d_ff)).astype(np.float32)
            for _ in range(n_mats)]

    rows = []
    for batch in BATCHES:
        a = rng.random((batch, cfg.d_model)).astype(np.float32)

        def pass_over_the_weights() -> float:
            """One batch through every matrix once, as a decode step does."""
            t0 = perf_counter()
            for m in mats:
                a @ m
            return (perf_counter() - t0) / n_mats

        pass_over_the_weights()              # warm up, and evict
        r = Repeated([pass_over_the_weights() for _ in range(runs)])
        rows.append({
            "batch": batch,
            "seconds": r.median,
            "spread_frac": r.spread,
            "noisy": r.noisy,
            "bytes_per_s": one / r.median,
            "flops": 2 * batch * cfg.d_model * cfg.d_ff / r.median,
            "per_token_ms": r.median / batch * 1e3,
        })
    base = next(x for x in rows if x["batch"] == 2)
    for row in rows:
        row["over_batch_two"] = row["seconds"] / base["seconds"]
        row["per_token_speedup"] = base["per_token_ms"] / row["per_token_ms"]

    # The free stretch: batch sizes from two up, for as long as the
    # multiply costs no more than the first real matrix multiply did.
    flat_to = 2
    for row in rows:
        if row["batch"] < 2:
            continue
        if row["over_batch_two"] > 1.10:
            break
        flat_to = row["batch"]
    return {"name": name, "shape": [cfg.d_model, cfg.d_ff],
            "weight_bytes": one, "matrices": n_mats, "flat_to": flat_to,
            "working_set_bytes": n_mats * one, "rows": rows}


def why_two_is_slower(cfg: Config, runs: int = 7) -> dict:
    """Two sequences take longer than one. Where does the time go?

    A single row is a matrix-vector product; two rows is a
    matrix-matrix product, and a matrix library prepares for one of
    those by copying its operand into a packing buffer. If that is the
    explanation, two matrix-vector products should beat one
    matrix-matrix product over the same two rows, by about what the
    copy costs.
    """
    rng = np.random.default_rng(11)
    b = rng.random((cfg.d_model, cfg.d_ff)).astype(np.float32)
    a = rng.random((2, cfg.d_model)).astype(np.float32)
    one, two = a[0:1], a[1:2]

    def timed(fn) -> float:
        fn()
        return median([_time(fn) for _ in range(runs)])

    def _time(fn) -> float:
        t0 = perf_counter(); fn(); return perf_counter() - t0

    gemm = timed(lambda: a @ b)
    gemv_pair = timed(lambda: (one @ b, two @ b))
    gemv = timed(lambda: one @ b)
    copy = timed(b.copy)
    return {"matrix_matrix_s": gemm, "two_matrix_vector_s": gemv_pair,
            "one_matrix_vector_s": gemv, "copy_of_the_matrix_s": copy,
            "unexplained_s": gemm - gemv_pair,
            "half_a_copy_s": copy / 2,
            "weight_bytes": int(b.nbytes)}


def achieved_gemm_rate(cfg: Config, batch: int, runs: int = 5) -> float:
    """Arithmetic rate for the shape a decode step actually runs.

    The peak from a 1024x1024 multiply is not what a batch of 8 gets:
    the matrices a decode step multiplies are B rows tall, and a short
    matrix leaves the machine's arithmetic units half empty. This
    measures the rate at the shape that matters.
    """
    rng = np.random.default_rng(7)
    a = rng.random((batch, cfg.d_model)).astype(np.float32)
    b = rng.random((cfg.d_model, cfg.d_ff)).astype(np.float32)
    a @ b
    rates = []
    for _ in range(runs):
        t0 = perf_counter()
        a @ b
        rates.append(2 * batch * cfg.d_model * cfg.d_ff / (perf_counter() - t0))
    return median(rates)


def sweep(cfg: Config, name: str, bandwidth: float) -> dict:
    """Decode step time against batch size, for one model.

    Each step is split into the part the batch shares -- every weight
    matmul -- and the part it does not: attention, which every sequence
    does over its own cache. Only the first is what batching is for.
    """
    model = build(cfg)
    weight_bytes = model.n_params * 4
    rng = np.random.default_rng(2)
    rows = []

    for batch in BATCHES:
        medians, shared, private = [], [], []
        for run in range(WARMUP + RUNS):
            caches = [KVCache(cfg, max_seq=PROMPT + STEPS + 2)
                      for _ in range(batch)]
            nxt = []
            for cache in caches:
                prompt = rng.integers(0, cfg.vocab_size, PROMPT)
                nxt.append(int(forward(model, prompt, cache)[-1].argmax()))
            times, sh, pr = [], [], []
            for _ in range(STEPS):
                timing = {}
                t0 = perf_counter()
                logits = decode_step(model, caches, np.array(nxt), timing)
                times.append(perf_counter() - t0)
                sh.append(timing["shared_s"]); pr.append(timing["private_s"])
                nxt = [int(row.argmax()) for row in logits]
            if run >= WARMUP:
                medians.append(median(times))
                shared.append(median(sh)); private.append(median(pr))
        r, s, p = Repeated(medians), Repeated(shared), Repeated(private)

        rate = achieved_gemm_rate(cfg, batch)
        memory_s = weight_bytes / bandwidth
        compute_s = 2 * model.n_params * batch / rate
        rows.append({
            "batch": batch,
            "step_s": r.summary(),
            "shared_s": s.median,
            "private_s": p.median,
            "tokens_per_s": batch / r.median,
            "inter_token_ms": r.median * 1e3,
            "per_token_ms": r.median / batch * 1e3,
            "shared_per_token_ms": s.median / batch * 1e3,
            "gemm_rate": rate,
            "predicted_shared_s": max(memory_s, compute_s),
            "predicted_bound_by": "memory" if memory_s >= compute_s else "compute",
            "shared_over_predicted": s.median / max(memory_s, compute_s),
        })

    one = rows[0]
    for row in rows:
        row["step_over_batch_one"] = row["step_s"]["median"] / one["step_s"]["median"]
        row["throughput_gain"] = row["tokens_per_s"] / one["tokens_per_s"]
        row["shared_speedup_per_token"] = (one["shared_per_token_ms"]
                                           / row["shared_per_token_ms"])
    knee = next((r["batch"] for r in rows if r["predicted_bound_by"] == "compute"),
                None)
    return {"name": name, "params": model.n_params, "weight_bytes": weight_bytes,
            "weight_mib": weight_bytes / 1024**2,
            "bandwidth_bytes_per_s": bandwidth,
            "memory_floor_s": weight_bytes / bandwidth,
            "kv_bytes_per_token": kv_bytes_per_token(cfg),
            "knee_batch": knee, "rows": rows}


# --- 3. what a static batch wastes --------------------------------------


def static_waste(prompts: np.ndarray, outputs: np.ndarray,
                 batch: int, rng) -> dict:
    """Fill a batch, run it to the end, and count what was not work.

    Two wastes, and neither is an implementation flaw -- both follow
    from the batch being fixed for its whole life:

    * every prompt is padded to the longest in the batch, because one
      matrix has one width;
    * every sequence holds its slot until the longest one finishes.
    """
    order = rng.permutation(len(prompts))
    prompt_used = prompt_held = output_used = output_held = 0
    batches = 0
    for start in range(0, len(order) - batch + 1, batch):
        idx = order[start : start + batch]
        p, o = prompts[idx], outputs[idx]
        prompt_used += int(p.sum())
        prompt_held += int(p.max()) * batch
        output_used += int(o.sum())
        output_held += int(o.max()) * batch
        batches += 1
    return {
        "batch": batch, "batches": batches,
        "prompt_utilization": prompt_used / prompt_held,
        "output_utilization": output_used / output_held,
        "total_utilization": (prompt_used + output_used) / (prompt_held + output_held),
        "steps_per_batch": output_held / batch / batches,
        "useful_steps_per_sequence": output_used / batch / batches,
    }


def one_batch_picture(prompts: np.ndarray, outputs: np.ndarray,
                      batch: int, rng) -> list[dict]:
    """One batch's worth of sequences, for the figure."""
    idx = rng.permutation(len(prompts))[:batch]
    p, o = prompts[idx], outputs[idx]
    return [{"prompt": int(a), "output": int(b),
             "padded_prompt": int(p.max()), "batch_steps": int(o.max())}
            for a, b in zip(p, o)]


def main() -> None:
    small_model = build(SMALL)
    eq = equivalence(small_model, SMALL)

    flops, flops_spread = matmul_flops()

    # Each model gets the bandwidth this machine delivers at *its* own
    # working-set size: Chapter 4's cliff, applied where it belongs.
    models = []
    for cfg, name in ((SMALL, "cache-resident"), (LARGE, "memory-resident")):
        weight_kib = build(cfg).n_params * 4 // 1024
        models.append(sweep(cfg, name, stream_bandwidth(weight_kib)))

    matmuls = [one_matmul(LARGE, "memory-resident", m["weight_bytes"])
               for m in models if m["name"] == "memory-resident"]
    matmuls += [one_matmul(SMALL, "cache-resident", m["weight_bytes"])
                for m in models if m["name"] == "cache-resident"]
    packing = why_two_is_slower(LARGE)

    rng = np.random.default_rng(SEED)
    prompts = lognormal(PROMPT_MEAN, PROMPT_CV, N_REQUESTS, rng)
    outputs = np.minimum(lognormal(OUTPUT_MEAN, OUTPUT_CV, N_REQUESTS, rng), 1024)
    waste = [static_waste(prompts, outputs, b, np.random.default_rng(SEED))
             for b in BATCHES]
    picture = one_batch_picture(prompts, outputs, 8, np.random.default_rng(SEED))

    # What the case study actually needs, against what one accelerator
    # can do at each batch size. Arithmetic over published specs.
    needed = CASE_REQUESTS_PER_S * CASE_OUTPUT_TOKENS
    fleet = [{"batch": b,
              "tokens_per_s": modelled_step(b, 1500).tokens_per_s,
              "accelerators": needed / modelled_step(b, 1500).tokens_per_s}
             for b in BATCHES]

    # The same question at the reference model's scale, from arithmetic.
    reference = [{"batch": b,
                  "inter_token_ms": modelled_step(b, 1500).inter_token_ms,
                  "tokens_per_s": modelled_step(b, 1500).tokens_per_s,
                  "bound_by": modelled_step(b, 1500).bound_by}
                 for b in BATCHES]

    payload = {
        "equivalence": eq,
        "machine": {"dram_bytes_per_s": stream_bandwidth(262144), "flops": flops,
                    "flops_spread_frac": flops_spread,
                    "ridge_flop_per_byte": flops / stream_bandwidth(262144),
                    "threads": 4,
                    "accelerator_ridge_flop_per_byte": RIDGE_FLOP_PER_BYTE},
        "models": models,
        "matmuls": matmuls,
        "packing": packing,
        "cache_bytes": cache_bytes(),
        "waste": waste,
        "picture": picture,
        "reference_8b": reference,
        "case_study": {"requests_per_s": CASE_REQUESTS_PER_S,
                       "output_tokens": CASE_OUTPUT_TOKENS,
                       "tokens_per_s_needed": needed,
                       "context": 1500, "fleet": fleet},
        "experiment": {"batches": BATCHES, "prompt": PROMPT, "steps": STEPS,
                       "runs": RUNS, "warmup": WARMUP,
                       "n_requests": N_REQUESTS, "seed": SEED},
    }

    path = write("results/ch16.json", payload)
    print(f"wrote {path}")
    a = eq["agreement"]
    print(f"  identical tokens: {eq['same_tokens']}; scores across batch sizes "
          f"differ by at most {a['worst']:.3g}")
    print(f"  peak arithmetic {flops / 1e9:.0f} GFLOP/s")
    for m in models:
        print(f"  {m['name']} ({m['weight_mib']:.1f} MiB, "
              f"{m['bandwidth_bytes_per_s'] / 1e9:.1f} GB/s at that working set, "
              f"compute-bound from batch {m['knee_batch']}):")
        for row in m["rows"]:
            print(f"    batch {row['batch']:3d}: step "
                  f"{row['inter_token_ms']:8.2f} ms "
                  f"(shared {row['shared_s'] * 1e3:7.2f}, private "
                  f"{row['private_s'] * 1e3:7.2f}), {row['tokens_per_s']:8.1f} tok/s "
                  f"({row['throughput_gain']:5.1f}x), shared per token "
                  f"{row['shared_per_token_ms']:7.3f} ms "
                  f"({row['shared_speedup_per_token']:4.1f}x), "
                  f"{row['predicted_bound_by']}-bound, measured/predicted "
                  f"{row['shared_over_predicted']:.2f}")
    for mm in matmuls:
        print(f"  one {mm['shape'][0]}x{mm['shape'][1]} weight matrix "
              f"({mm['weight_bytes'] / 1024**2:.2f} MiB each, "
              f"{mm['matrices']} of them = "
              f"{mm['working_set_bytes'] / 1024**2:.0f} MiB working set), "
              f"{mm['name']}:")
        for row in mm["rows"]:
            print(f"    batch {row['batch']:3d}: {row['seconds'] * 1e3:7.3f} ms, "
                  f"{row['bytes_per_s'] / 1e9:6.1f} GB/s, "
                  f"{row['flops'] / 1e9:7.1f} GFLOP/s, per token "
                  f"{row['per_token_ms']:7.4f} ms ({row['per_token_speedup']:5.1f}x "
                  f"the batch of two)")
    print(f"  case study needs {needed:,} output tokens/s: "
          + ", ".join(f"batch {f['batch']} -> {f['accelerators']:.0f} accelerators"
                      for f in fleet if f["batch"] in (1, 8, 64)))
    pk = packing
    print(f"  batch of two: one matrix-matrix {pk['matrix_matrix_s'] * 1e3:.2f} ms "
          f"vs two matrix-vector {pk['two_matrix_vector_s'] * 1e3:.2f} ms; "
          f"unexplained {pk['unexplained_s'] * 1e3:.2f} ms against "
          f"{pk['half_a_copy_s'] * 1e3:.2f} ms to write a copy of the matrix")
    print(f"  largest cache level: {cache_bytes() / 1024**2:.0f} MiB")
    for w in waste:
        print(f"  batch {w['batch']:3d}: prompt {w['prompt_utilization'] * 100:5.1f}% "
              f"useful, generation {w['output_utilization']*100:5.1f}%, "
              f"together {w['total_utilization']*100:5.1f}%")


if __name__ == "__main__":
    main()
