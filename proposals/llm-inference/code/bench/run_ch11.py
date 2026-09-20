"""Chapter 11: the loop that works, and how much of its work is thrown away.

The obvious way to generate text is to run the model over everything so
far, take the last token's scores, append, and repeat. It is correct,
it is about fifteen lines, and almost all of what it computes is
discarded the instant it is produced.

This measures the waste two ways: as arithmetic that is recomputed and
then dropped, and as time that grows with every token written.

    python3 -m bench.run_ch11
"""

from __future__ import annotations

from tinyserve.cost import flops_forward
from tinyserve.generate import naive
from tinyserve.model import Config, build

from .harness import Repeated, write

PROMPT = 128
LENGTHS = [16, 32, 64, 128, 256]
LONG = 256
RUNS, WARMUP = 3, 1


def waste_at(cfg: Config, prompt: int, step: int) -> dict:
    """At this step, how much of the arithmetic earns a token?

    The loop runs the model over the whole sequence, but only the last
    position's scores are used. Everything computed for the earlier
    positions is recomputed from scratch and then dropped.
    """
    total = flops_forward(cfg, prompt + step, prompt + step)
    useful = flops_forward(cfg, 1, prompt + step)
    return {"step": step, "sequence": prompt + step,
            "flops_total": total, "flops_useful": useful,
            "wasted_fraction": 1 - useful / total}


def main() -> None:
    # The default configuration, so this is the same model Chapters 10 and
    # 12 measure. Its 1,024 positions comfortably hold prompt plus output.
    cfg = Config()
    model = build(cfg)
    prompt = list(range(PROMPT))

    # Where the time goes, step by step.
    for _ in range(WARMUP):
        naive(model, prompt, 8)
    run = naive(model, prompt, LONG)
    per_step = [run.ttft_s] + run.step_s

    # How the total grows with the number of tokens written.
    sweep = []
    for n in LENGTHS:
        for _ in range(WARMUP):
            naive(model, prompt, min(n, 8))
        r = Repeated([naive(model, prompt, n).total_s for _ in range(RUNS)])
        sweep.append({
            "generated": n, "total_s": r.median, "noisy": r.noisy,
            "seconds_per_token": r.median / n,
            "flops": sum(flops_forward(cfg, PROMPT + i, PROMPT + i)
                         for i in range(n)),
            "flops_useful": sum(flops_forward(cfg, 1, PROMPT + i)
                                for i in range(n)),
        })
    for s in sweep:
        s["wasted_fraction"] = 1 - s["flops_useful"] / s["flops"]

    waste = [waste_at(cfg, PROMPT, i) for i in (0, 1, 16, 64, 128, 255)]

    payload = {
        "experiment": {"prompt": PROMPT, "lengths": LENGTHS, "longest": LONG,
                       "runs": RUNS, "warmup": WARMUP},
        "model": {"params": model.n_params,
                  "config": {k: getattr(cfg, k) for k in
                             ("vocab_size", "d_model", "n_layers", "n_heads",
                              "n_kv_heads", "d_ff")},
                  "head_dim": cfg.head_dim},
        "per_step_s": per_step,
        "sweep": sweep,
        "waste": waste,
        "summary": {
            "first_step_ms": per_step[1] * 1e3,
            "last_step_ms": per_step[-1] * 1e3,
            "step_growth": per_step[-1] / per_step[1],
            "sequence_growth": (PROMPT + LONG - 1) / (PROMPT + 1),
            "wasted_overall": sweep[-1]["wasted_fraction"],
            "useful_share": 1 - sweep[-1]["wasted_fraction"],
        },
    }

    path = write("results/ch11.json", payload)
    print(f"wrote {path}")
    for w in waste:
        print(f"  step {w['step']:4d} (sequence {w['sequence']:4d}): "
              f"{w['wasted_fraction'] * 100:6.2f}% of the arithmetic is discarded")
    for s in sweep:
        print(f"  {s['generated']:4d} tokens: {s['total_s']:7.3f} s "
              f"({s['seconds_per_token'] * 1e3:6.2f} ms per token), "
              f"{s['wasted_fraction'] * 100:.1f}% wasted")
    su = payload["summary"]
    print(f"  step cost grew {su['step_growth']:.2f}x while the sequence grew "
          f"{su['sequence_growth']:.2f}x")


if __name__ == "__main__":
    main()
