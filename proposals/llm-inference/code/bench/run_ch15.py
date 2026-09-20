"""Chapter 15: sharing a prefix, built and measured.

Chapter 14 left every sequence with its own copy of every block, even
when two sequences began with exactly the same thousand tokens. This
shares them, and measures four things:

1. It must not change the answer.
2. What a hit saves on the wall clock, against how much of the prompt
   was already cached.
3. How much of a realistic stream of requests hits, and how that falls
   as the cache is made smaller.
4. What the eviction policy is worth -- and what happens when a cache
   evicts without knowing the shape of what it holds.

    python3 -m bench.run_ch15
"""

from __future__ import annotations

import itertools
import math
from statistics import median
from time import perf_counter

import numpy as np

from tinyserve.model import Config, DType, build, forward, softmax
from tinyserve.paged import BLOCK_SIZE
from tinyserve.prefix import PrefixTree, SharedBlockPool, SharedKVCache
from tinyserve.reference import GPU_BYTES, KV_BYTES_PER_TOKEN, WEIGHT_BYTES

from .harness import Repeated, write

# --- the measured experiment (tinyserve, a real forward pass) -----------

PROMPT_TOKENS = 512
SHARED = [0, 128, 256, 384, 448, 496]
RUNS, WARMUP, DECODE_STEPS = 15, 2, 32

# --- the simulated stream of requests -----------------------------------

N_SYSTEM_PROMPTS = 8
SYSTEM_MIN, SYSTEM_MAX = 480, 900       # tokens; deliberately not block-aligned
ZIPF_S = 1.1                            # how skewed system-prompt popularity is
N_SESSIONS = 200
TURNS_MEAN = 3.0
ONE_OFF_FRACTION = 0.25
USER_MEAN, USER_CV = 120, 0.7
REPLY_MEAN, REPLY_CV = 220, 0.8
MAX_MODEL_LEN = 8192
POOL_FRACTIONS = [0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0]
POLICIES = ["lru", "lfu", "unstructured"]
SEED = 0

# The stream is simulated: what is being measured is which blocks are
# matched, kept and thrown away, and that is settled entirely by the
# index and the allocator. Neither reads the keys and values, so the
# pool's storage is given a deliberately tiny shape and the block
# *counts* are converted to bytes at the reference model's scale.
SIM_CFG = Config(vocab_size=64, d_model=16, n_layers=1, n_heads=1,
                 n_kv_heads=1, d_ff=16, max_seq=16)


def lognormal_int(mean: float, cv: float, rng) -> int:
    sigma = math.sqrt(math.log(1 + cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return int(max(1, round(rng.lognormal(mu, sigma))))


# --- 1. sharing must not change the answer ------------------------------


def identical_output(cfg: Config, model) -> dict:
    """One sequence computes a prompt; a second adopts its blocks and
    computes only the rest. Both must generate the same tokens."""
    bs = BLOCK_SIZE
    shared = 256
    rng = np.random.default_rng(1)
    prefix = rng.integers(0, cfg.vocab_size, shared).tolist()
    tail_a = rng.integers(0, cfg.vocab_size, 64).tolist()
    tail_b = rng.integers(0, cfg.vocab_size, 96).tolist()

    def generate(cache, tokens) -> tuple[list[int], np.ndarray]:
        logits = forward(model, np.array(tokens), cache)
        out = []
        for _ in range(DECODE_STEPS):
            nxt = int(logits[-1].argmax())
            out.append(nxt)
            logits = forward(model, np.array([nxt]), cache)
        return out, logits

    # Cold: sequence B computes its whole prompt.
    pool = SharedBlockPool(cfg, n_blocks=256)
    cold = SharedKVCache(pool)
    cold_tokens, cold_logits = generate(cold, prefix + tail_b)

    # Warm: sequence A runs first, its blocks are cached, B adopts them.
    pool = SharedBlockPool(cfg, n_blocks=256)
    tree = PrefixTree(pool)
    a = SharedKVCache(pool)
    generate(a, prefix + tail_a)
    tree.insert(prefix + tail_a, a.blocks)
    a.release()

    b = SharedKVCache(pool)
    matched = tree.match(prefix + tail_b)
    b.adopt(matched, len(matched) * bs)
    warm_tokens, warm_logits = generate(b, (prefix + tail_b)[len(matched) * bs:])

    # The adopted keys were computed when sequence A was 320 tokens long;
    # the cold path computed the same keys in a 352-token pass. Are they
    # the same numbers?
    by_layer = [
        max(float(np.max(np.abs(cold.pool.k[layer][cold.blocks[i]]
                                - b.pool.k[layer][blk])))
            for i, blk in enumerate(matched))
        for layer in range(cfg.n_layers)]
    key_diff = max(by_layer)

    return {
        "shared_prompt_tokens": shared,
        "blocks_adopted": len(matched),
        "tokens_skipped": len(matched) * bs,
        "same_tokens": cold_tokens == warm_tokens,
        "tokens_compared": DECODE_STEPS,
        "max_logit_diff": float(np.max(np.abs(cold_logits - warm_logits))),
        "cached_key_diff": key_diff,
        "cached_key_diff_by_layer": by_layer,
    }


def numerics(cfg: Config) -> dict:
    """Where the last-bit difference comes from -- found, not assumed.

    Two checks, each isolating one step of the forward pass. Both ask
    the same question: does this operation give the same answer for the
    same tokens when the sequence around them is a different length?
    """
    rng = np.random.default_rng(3)
    d, keep, extra = cfg.d_model, 256, 96

    # A projection contracts over d_model, which does not change.
    w = (rng.standard_normal((d, d)) * 0.1).astype(DType)
    x = (rng.standard_normal((keep + extra, d)) * 0.1).astype(DType)
    projection_same = bool(np.array_equal(x[:keep] @ w, (x @ w)[:keep]))

    # A softmax divides by a sum along the sequence, which does. The
    # extra entries are masked to -inf and contribute exactly zero, but
    # the sum groups its terms differently when the row is longer.
    s = (rng.standard_normal(keep + 1) * 3).astype(DType)
    long = np.concatenate([s, np.full(extra, -np.inf, dtype=DType)])
    a, b = softmax(s), softmax(long)[: keep + 1]
    return {
        "projection_same": projection_same,
        "softmax_same": bool(np.array_equal(a, b)),
        "softmax_diff": float(np.max(np.abs(a - b))),
        "padding_tokens": extra,
    }


# --- 2. what a hit saves on the clock -----------------------------------


def prefill_saving(cfg: Config, model) -> list[dict]:
    """Time sequence B's prefill for several sizes of cached prefix."""
    bs = BLOCK_SIZE
    rng = np.random.default_rng(2)
    out = []
    for shared in SHARED:
        prompt_a = rng.integers(0, cfg.vocab_size, PROMPT_TOKENS).tolist()
        prompt_b = prompt_a[:shared] + rng.integers(
            0, cfg.vocab_size, PROMPT_TOKENS - shared).tolist()

        cold, warm, adopted = [], [], 0
        for run in range(WARMUP + RUNS):
            pool = SharedBlockPool(cfg, n_blocks=128)
            tree = PrefixTree(pool)
            a = SharedKVCache(pool)
            forward(model, np.array(prompt_a), a)
            tree.insert(prompt_a, a.blocks)
            a.release()

            # The lookup is timed with the prefill: it is work the server
            # does on every request. The cold path below pays a lookup
            # too -- a miss -- and is not charged for it, so the
            # comparison is if anything unkind to the cache.
            b = SharedKVCache(pool)
            t0 = perf_counter()
            matched = tree.match(prompt_b)
            adopted = len(matched)
            b.adopt(matched, adopted * bs)
            rest = np.array(prompt_b[adopted * bs:])
            forward(model, rest, b)
            dt_warm = perf_counter() - t0

            c = SharedKVCache(SharedBlockPool(cfg, n_blocks=128))
            whole = np.array(prompt_b)
            t0 = perf_counter(); forward(model, whole, c); dt_cold = perf_counter() - t0
            if run >= WARMUP:                      # never report a cold run
                warm.append(dt_warm); cold.append(dt_cold)

        # Cold and warm are measured in the same iteration, one after the
        # other, so a machine-wide slowdown moves both. Chapter 9's
        # remedy: take the ratio *within* each pair and report the median
        # of those, not the ratio of two independently noisy medians.
        cold_r, warm_r = Repeated(cold), Repeated(warm)
        ratio = Repeated([c / w for c, w in zip(cold, warm)])
        out.append({
            "shared_tokens": shared,
            "shared_frac": shared / PROMPT_TOKENS,
            "blocks_adopted": adopted,
            "tokens_computed": PROMPT_TOKENS - adopted * bs,
            "cold_s": cold_r.summary(),
            "warm_s": warm_r.summary(),
            "speedup": ratio.median,
            "speedup_spread": ratio.spread,
            "speedup_if_proportional": PROMPT_TOKENS / (PROMPT_TOKENS - adopted * bs),
            "noisy": ratio.noisy,
        })
    return out


# --- 3. a stream of requests --------------------------------------------


def make_trace(rng) -> dict:
    """Sessions that come back, system prompts that repeat, and one-off
    requests that share nothing -- the three shapes real traffic has."""
    ids = itertools.count(1)
    systems = [[next(ids) for _ in range(int(n))]
               for n in rng.integers(SYSTEM_MIN, SYSTEM_MAX, N_SYSTEM_PROMPTS)]
    weights = 1 / np.arange(1, N_SYSTEM_PROMPTS + 1) ** ZIPF_S
    weights /= weights.sum()

    requests, clock = [], 0.0
    for _ in range(N_SESSIONS):
        clock += float(rng.exponential(1.0))
        one_off = rng.random() < ONE_OFF_FRACTION
        if one_off:
            system = None
            preamble = [next(ids) for _ in
                        range(int(rng.integers(SYSTEM_MIN, SYSTEM_MAX)))]
            n_turns = 1
        else:
            system = int(rng.choice(N_SYSTEM_PROMPTS, p=weights))
            preamble = systems[system]
            n_turns = int(rng.geometric(1 / TURNS_MEAN))

        conversation, when = list(preamble), clock
        for turn in range(n_turns):
            user = [next(ids) for _ in range(lognormal_int(USER_MEAN, USER_CV, rng))]
            prompt = conversation + user
            if len(prompt) > MAX_MODEL_LEN:
                break
            reply = [next(ids) for _ in range(lognormal_int(REPLY_MEAN, REPLY_CV, rng))]
            requests.append({
                "time": when, "tokens": prompt + reply, "n_prompt": len(prompt),
                "system": system, "turn": turn,
                "preamble_len": len(preamble),
                "conversation_before": len(conversation) if turn else 0,
            })
            conversation = prompt + reply
            when += float(rng.exponential(4.0))

    requests.sort(key=lambda r: r["time"])

    # What a perfect, token-granular, never-evicting cache would match.
    # Known by construction: a follow-up turn shares the whole
    # conversation so far; a first turn shares its system prompt if that
    # prompt has been served before; a one-off shares nothing.
    seen: set[int] = set()
    for r in requests:
        if r["turn"]:
            r["ideal_hit"] = r["conversation_before"]
        elif r["system"] is not None and r["system"] in seen:
            r["ideal_hit"] = r["preamble_len"]
        else:
            r["ideal_hit"] = 0
        if r["system"] is not None:
            seen.add(r["system"])
    return {"requests": requests, "systems": [len(s) for s in systems]}


def serve(requests: list[dict], n_blocks: int, policy: str,
          cfg: Config, check: bool = False) -> dict:
    """Run the stream through a pool of `n_blocks`, caching prefixes.

    Nothing is computed: the point is which blocks are matched, kept and
    thrown away, and that is settled entirely by the index and the
    allocator, both of which are the real ones.
    """
    bs = BLOCK_SIZE
    pool = SharedBlockPool(cfg, n_blocks=n_blocks, block_size=bs)
    tree = PrefixTree(pool, policy=policy)

    prompt_tokens = hit_tokens = ideal_tokens = 0
    blocks_if_unshared = mismatches = not_served = 0
    peak_cached = 0
    index_s = evict_s = 0.0

    for r in requests:
        tokens, total = r["tokens"], len(r["tokens"])
        prompt_tokens += r["n_prompt"]
        ideal_tokens += r["ideal_hit"]
        blocks_if_unshared += math.ceil(total / bs)

        t0 = perf_counter()
        matched = tree.match(tokens[: r["n_prompt"]])
        index_s += perf_counter() - t0
        seq = SharedKVCache(pool)
        seq.adopt(matched, len(matched) * bs)
        hit = len(matched) * bs
        hit_tokens += hit
        if check and hit != (r["ideal_hit"] // bs) * bs:
            mismatches += 1

        need = math.ceil(total / bs) - len(matched)
        if pool.blocks_free < need:
            t0 = perf_counter()
            tree.evict(need)
            evict_s += perf_counter() - t0
        if pool.blocks_free < need:
            not_served += 1
            seq.release()
            continue

        seq._grow_to(total)
        t0 = perf_counter()
        tree.insert(tokens, seq.blocks)
        index_s += perf_counter() - t0
        seq.release()
        peak_cached = max(peak_cached, tree.cached_blocks)

    return {
        "policy": policy, "pool_blocks": n_blocks,
        "prompt_tokens": prompt_tokens, "hit_tokens": hit_tokens,
        "hit_rate": hit_tokens / prompt_tokens,
        "ideal_tokens": ideal_tokens,
        "ideal_rate": ideal_tokens / prompt_tokens,
        "computed_tokens": prompt_tokens - hit_tokens,
        "blocks_if_unshared": blocks_if_unshared,
        "cached_blocks": tree.cached_blocks,
        "peak_cached_blocks": peak_cached,
        "stranded_blocks": tree.stranded_blocks,
        "evictions": tree.evictions,
        "index_s": index_s, "evict_s": evict_s,
        "not_served": not_served,
        "construction_mismatches": mismatches,
    }


def main() -> None:
    cfg = Config()
    model = build(cfg)

    equivalence = identical_output(cfg, model)
    numbers = numerics(cfg)
    prefill = prefill_saving(cfg, model)

    rng = np.random.default_rng(SEED)
    trace = make_trace(rng)
    requests = trace["requests"]

    # A pool large enough that nothing is ever evicted: the ceiling.
    total_blocks = sum(math.ceil(len(r["tokens"]) / BLOCK_SIZE) for r in requests)
    runs = [serve(requests, total_blocks, "lru", SIM_CFG, check=True)
            for _ in range(5)]
    unlimited = runs[0]
    index_cost = Repeated([r["index_s"] for r in runs])
    working_set = unlimited["cached_blocks"]

    sizes = []
    for policy in POLICIES:
        for frac in POOL_FRACTIONS:
            n = max(int(working_set * frac), math.ceil(MAX_MODEL_LEN / BLOCK_SIZE))
            sizes.append(serve(requests, n, policy, SIM_CFG) | {"pool_frac": frac})

    quantization = [(r["ideal_hit"] % BLOCK_SIZE) for r in requests if r["ideal_hit"]]

    payload = {
        "equivalence": equivalence,
        "numerics": numbers,
        "prefill": {"prompt_tokens": PROMPT_TOKENS, "block": BLOCK_SIZE,
                    "runs": RUNS, "warmup": WARMUP, "points": prefill},
        "trace": {
            "requests": len(requests), "sessions": N_SESSIONS,
            "system_prompts": N_SYSTEM_PROMPTS,
            "system_lengths": trace["systems"],
            "follow_up_turns": sum(1 for r in requests if r["turn"]),
            "prompt_tokens": unlimited["prompt_tokens"],
            "prompt_p50": float(np.percentile([r["n_prompt"] for r in requests], 50)),
            "prompt_p99": float(np.percentile([r["n_prompt"] for r in requests], 99)),
        },
        "unlimited": unlimited | {
            "working_set_blocks": working_set,
            "working_set_gb": working_set * BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1000**3,
            "dedup": unlimited["blocks_if_unshared"] / working_set,
            "quantization_loss_median": float(median(quantization)) if quantization else 0.0,
            "quantization_loss_mean": float(np.mean(quantization)) if quantization else 0.0,
            "index_s": index_cost.summary(),
            "index_us_per_request": index_cost.median / len(requests) * 1e6,
            "index_us_per_block": (index_cost.median
                                   / sum(math.ceil(len(r["tokens"]) / BLOCK_SIZE)
                                         for r in requests) * 1e6),
        },
        "sizes": sizes,
        "assumptions": {
            "block": BLOCK_SIZE,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "max_model_len": MAX_MODEL_LEN,
            "seed": SEED,
            "gb_per_block": BLOCK_SIZE * KV_BYTES_PER_TOKEN / 1000**3,
            # What Chapter 13 measured as free after the weights: the
            # cache and the live sequences share exactly this.
            "pool_bytes": GPU_BYTES - WEIGHT_BYTES,
        },
    }

    path = write("results/ch15.json", payload)
    print(f"wrote {path}")
    e = equivalence
    print(f"  identical tokens: {e['same_tokens']}, "
          f"{e['tokens_skipped']} prompt tokens skipped, "
          f"max logit difference {e['max_logit_diff']:.3g}, "
          f"cached keys differ by {e['cached_key_diff']:.3g}")
    n = payload["numerics"]
    print(f"  numerics: projection unchanged by sequence length: "
          f"{n['projection_same']}; softmax unchanged: {n['softmax_same']} "
          f"(differs by {n['softmax_diff']:.3g})")
    for p in prefill:
        print(f"  shared {p['shared_tokens']:4d}/{PROMPT_TOKENS}: prefill "
              f"{p['cold_s']['median'] * 1e3:7.2f} ms -> {p['warm_s']['median'] * 1e3:7.2f} ms "
              f"({p['speedup']:.2f}x)")
    u = payload["unlimited"]
    print(f"  trace: {len(requests)} requests, {u['prompt_tokens']:,} prompt tokens")
    print(f"  unlimited cache: hit rate {u['hit_rate'] * 100:.1f}% "
          f"(ideal {u['ideal_rate'] * 100:.1f}%), working set {working_set:,} blocks "
          f"= {u['working_set_gb']:.1f} GB at 8B scale, dedup {u['dedup']:.2f}x, "
          f"construction mismatches {u['construction_mismatches']}")
    for s in sizes:
        print(f"  {s['policy']:13} pool {s['pool_frac']:.2f} "
              f"({s['pool_blocks']:6,d} blocks): hit {s['hit_rate'] * 100:5.1f}%, "
              f"evictions {s['evictions']:7,d}, stranded {s['stranded_blocks']:6,d}, "
              f"unserved {s['not_served']}")


if __name__ == "__main__":
    main()
