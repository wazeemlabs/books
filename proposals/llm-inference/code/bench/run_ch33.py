"""Chapter 33: three things that move the roofline.

Long prompts, long thinking, and a model whose weights are mostly
unused. They look like three separate topics and they are one: each
changes which term of the decode step dominates, and each does it in a
way that interacts with the batch.

    python3 -m bench.run_ch33
"""

from __future__ import annotations

import math

from tinyserve.cost import flops_forward
from tinyserve.moe import (DEEPSEEK_V3, all_to_all_bytes,
                           params_read_per_machine, step_seconds)
from tinyserve.reference import (ELEM_BYTES, GPU_BYTES, GPU_USD_PER_HOUR,
                                 HBM_BYTES_PER_S, KV_BYTES_PER_TOKEN, MODEL,
                                 OUTPUT_TOKENS, PARAMS, PEAK_BF16_FLOPS,
                                 PROMPT_TOKENS, WEIGHT_BYTES)
from tinyserve.serving import decode_step, prefill_step

from .harness import write
from .run_ch19 import LINKS

CONTEXTS = [1_024, 4_096, 8_192, 16_384, 32_768, 65_536, 131_072]
THINKING = [0, 1_024, 2_048, 4_096, 8_192, 16_384, 32_768]
BATCHES = [1, 4, 8, 16, 32, 64, 128, 256, 512]
ITL_BUDGET_MS = 50
POOL_BYTES = GPU_BYTES - WEIGHT_BYTES
# Anthropic's documented floor and its documented warning (FACTS.md).
THINKING_MINIMUM = 1_024
THINKING_BATCH_ADVICE = 32_768


def concurrency(context: int) -> int:
    """Sequences whose caches fit in what the weights leave behind."""
    return int(POOL_BYTES // (context * KV_BYTES_PER_TOKEN))


def long_context() -> dict:
    """What a long prompt does to memory, to the step, and to prefill."""
    rows = []
    for n in CONTEXTS:
        kv = n * KV_BYTES_PER_TOKEN
        fits = concurrency(n)
        batch = max(1, min(fits, 64))
        step = decode_step(batch, n)
        weights = WEIGHT_BYTES
        cache = batch * kv
        pre = prefill_step(n)
        # The quadratic part of a prefill: attention's scores and
        # values, against everything else, which is linear in tokens.
        attention = MODEL.n_layers * 2 * 2 * MODEL.n_heads * n * n * MODEL.head_dim
        rows.append({
            "context": n,
            "kv_gb": kv / 1e9,
            "sequences_that_fit": fits,
            "batch": batch,
            "weight_share": weights / (weights + cache),
            "cache_share": cache / (weights + cache),
            "itl_ms": step.inter_token_ms,
            "meets_itl": step.inter_token_ms <= ITL_BUDGET_MS,
            "prefill_s": pre.seconds,
            "prefill_flops": pre.flops,
            "attention_share_of_prefill": attention / pre.flops,
        })
    # Where one sequence's cache outweighs the entire model.
    crossover = WEIGHT_BYTES / KV_BYTES_PER_TOKEN
    # Where a prefill's quadratic term overtakes its linear one.
    linear = flops_forward(MODEL, 1, 1)
    per_token_sq = MODEL.n_layers * 2 * 2 * MODEL.n_heads * MODEL.head_dim
    return {"rows": rows, "pool_bytes": POOL_BYTES,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "weight_bytes": WEIGHT_BYTES,
            "cache_outweighs_model_at": crossover,
            "attention_overtakes_at": linear / per_token_sq,
            "itl_budget_ms": ITL_BUDGET_MS}


def thinking() -> dict:
    """What a long reply does, which is not what a long prompt does."""
    rows = []
    for extra in THINKING:
        out = OUTPUT_TOKENS + extra
        end = PROMPT_TOKENS + out
        fits = concurrency(end)
        batch = max(1, min(fits, 64))
        step = decode_step(batch, (PROMPT_TOKENS + end) // 2)
        answer_s = prefill_step(PROMPT_TOKENS).seconds + out * step.seconds
        # What the machine costs while it works on this one answer,
        # divided by the answers it is working on at once.
        machine_s = answer_s / batch
        rows.append({
            "thinking_tokens": extra,
            "output_tokens": out,
            "visible_share": OUTPUT_TOKENS / out,
            "context_at_the_end": end,
            "kv_gb_at_the_end": end * KV_BYTES_PER_TOKEN / 1e9,
            "sequences_that_fit": fits,
            "batch": batch,
            "itl_ms": step.inter_token_ms,
            "seconds_to_answer": answer_s,
            "machine_seconds_per_answer": machine_s,
            "usd_per_answer": machine_s / 3600 * GPU_USD_PER_HOUR,
            "over_the_visible_answer": out / OUTPUT_TOKENS,
        })
    base = rows[0]
    return {"rows": rows, "visible_tokens": OUTPUT_TOKENS,
            "prompt_tokens": PROMPT_TOKENS,
            "baseline_usd": base["usd_per_answer"],
            "baseline_seconds": base["seconds_to_answer"],
            "thinking_minimum": THINKING_MINIMUM,
            "batch_advice_above": THINKING_BATCH_ADVICE,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR}


def mixture() -> dict:
    """What a batch does to a model whose weights are mostly unused."""
    m = DEEPSEEK_V3
    seq = PROMPT_TOKENS + OUTPUT_TOKENS
    # This model's own cache per token is not the reference model's;
    # its attention is compressed, and the chapter says so. The KV
    # figure used here is the reference model's, to keep the two terms
    # on one scale, and the conclusion is about the weight term.
    machines = math.ceil(m.total * ELEM_BYTES / GPU_BYTES)
    rows = []
    for batch in BATCHES:
        read = m.params_read(batch)
        seconds = step_seconds(m, batch, seq, HBM_BYTES_PER_S,
                               KV_BYTES_PER_TOKEN, ELEM_BYTES)
        # And the same step with the experts spread over the machines
        # the weights need anyway. Each machine reads the replicated
        # part in full and its share of the experts that were touched,
        # then the hidden states cross the network twice a layer.
        shard = params_read_per_machine(m, batch, machines)
        shard_s = (shard * ELEM_BYTES
                   + batch * seq * KV_BYTES_PER_TOKEN / machines) / HBM_BYTES_PER_S
        link = dict(LINKS)["NVLink, same node"]
        network_s = all_to_all_bytes(m, batch, ELEM_BYTES) / link
        rows.append({
            "machines": machines,
            "params_read_per_machine": shard,
            "shard_itl_ms": shard_s * 1e3,
            "network_ms": network_s * 1e3,
            "parallel_itl_ms": (shard_s + network_s) * 1e3,
            "batch": batch,
            "experts_touched": m.touched(batch),
            "params_read": read,
            "params_per_token": m.params_per_token(batch),
            "dense_equivalent": read,
            "itl_ms": seconds * 1e3,
            "tokens_per_s": batch / seconds,
            "all_to_all_gb": all_to_all_bytes(m, batch, ELEM_BYTES) / 1e9,
            "all_to_all_seconds": {
                name: all_to_all_bytes(m, batch, ELEM_BYTES) / bw
                for name, bw in LINKS},
        })
    return {
        "machines": machines,
        "model": {"name": m.name, "total": m.total, "active": m.active,
                  "routed": m.routed, "per_token": m.per_token,
                  "layers": m.layers, "d_model": m.d_model,
                  "expert_params": m.expert_params(),
                  "always_read": m.always_read()},
        "rows": rows, "seq": seq,
        "weights_gb_bf16": m.total * ELEM_BYTES / 1e9,
        "accelerators_to_hold_it": math.ceil(m.total * ELEM_BYTES / GPU_BYTES),
        "links": [name for name, _ in LINKS],
        "gpu_gb": GPU_BYTES / 1e9,
    }


def main() -> None:
    lc, th, mo = long_context(), thinking(), mixture()
    payload = {
        "long_context": lc, "thinking": th, "mixture": mo,
        "assumptions": {
            "contexts": CONTEXTS, "thinking_tokens": THINKING,
            "batches": BATCHES, "itl_budget_ms": ITL_BUDGET_MS,
            "prompt_tokens": PROMPT_TOKENS, "output_tokens": OUTPUT_TOKENS,
            "params": PARAMS, "weight_bytes": WEIGHT_BYTES,
            "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
            "gpu_bytes": GPU_BYTES, "hbm_bytes_per_s": HBM_BYTES_PER_S,
            "peak_flops": PEAK_BF16_FLOPS,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch33.json", payload)
    print(f"wrote {path}")
    print("  a long prompt:")
    for r in lc["rows"]:
        print(f"    {r['context']:>7,} tokens: {r['kv_gb']:6.2f} GB of cache, "
              f"{r['sequences_that_fit']:>5,} fit, at batch {r['batch']:>2} "
              f"the step is {r['cache_share'] * 100:4.0f}% cache, "
              f"{r['itl_ms']:6.1f} ms between tokens, prefill "
              f"{r['prefill_s'] * 1e3:7.1f} ms "
              f"({r['attention_share_of_prefill'] * 100:4.1f}% attention)")
    print(f"    one sequence's cache outweighs the whole model at "
          f"{lc['cache_outweighs_model_at']:,.0f} tokens; attention "
          f"overtakes the rest of a prefill at "
          f"{lc['attention_overtakes_at']:,.0f}")
    print("  a long reply:")
    for r in th["rows"]:
        print(f"    {r['thinking_tokens']:>6,} thinking tokens: "
              f"{r['output_tokens']:>6,} out "
              f"({r['visible_share'] * 100:4.1f}% visible), batch "
              f"{r['batch']:>2}, {r['seconds_to_answer']:6.1f} s an answer, "
              f"${r['usd_per_answer'] * 1000:6.3f} per thousand answers")
    print(f"  {mo['model']['name']}: "
          f"{mo['model']['total'] / 1e9:.0f}B total, "
          f"{mo['model']['active'] / 1e9:.0f}B active, "
          f"{mo['weights_gb_bf16']:,.0f} GB in bf16 "
          f"({mo['accelerators_to_hold_it']} accelerators just to hold it)")
    for r in mo["rows"]:
        print(f"    batch {r['batch']:>4}: {r['experts_touched']:6.1f} of "
              f"{mo['model']['routed']} experts, step reads "
              f"{r['params_read'] / 1e9:6.1f}B "
              f"({r['params_per_token'] / 1e9:5.2f}B a token), "
              f"{r['itl_ms']:6.1f} ms on one machine, "
              f"{r['parallel_itl_ms']:6.1f} ms over {r['machines']} "
              f"({r['network_ms']:.1f} ms of it network)")


if __name__ == "__main__":
    main()
