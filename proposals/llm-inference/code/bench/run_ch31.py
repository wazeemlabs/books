"""Chapter 31: making the model produce something a parser can read.

Four measurements:

1. Validity. What a model produces with and without a mask, when the
   caller needs JSON.
2. The compiled table. How many states a JSON grammar has, how many
   distinct masks those states need, and why the second number stops
   growing while the first does not.
3. What a mask costs, per token, counted rather than timed.
4. What it costs the answer. Constraining is not free in the way
   Chapter 29's verification was free: it changes the distribution,
   and this measures by how much.

    python3 -m bench.run_ch31
"""

from __future__ import annotations

import numpy as np

from tinyserve import grammar as gr
from tinyserve.reference import MODEL, PARAMS
from tinyserve.serving import decode_step
from tinyserve.reference import CONTEXT_TOKENS

from .harness import write

SEED = 0
TRIALS = 600
MAX_TOKENS = 400
DEPTHS = [1, 2, 3, 4, 5, 6]
# How strongly a model is assumed to already prefer the right token.
# 0 is a model with no idea; larger numbers are a model that has seen
# JSON before and mostly gets it right on its own.
SKILLS = [0.0, 1.0, 2.0, 4.0, 8.0]


def _skilled_logits(skill: float, rng: np.random.Generator):
    """A stand-in model that knows JSON to a controllable degree.

    Real logits come from a trained network. What matters here is only
    how often the model's own preference is already legal, so this
    gives every token a random score and adds `skill` to the ones the
    grammar would allow. At skill 0 it knows nothing; as skill grows
    it needs the mask less and less.
    """
    def logits_for(text: str, machine: gr.JsonMachine):
        base = rng.standard_normal(len(gr.VOCAB))
        if skill:
            base = base + skill * gr.mask_for(machine)
        return base
    return logits_for


def validity() -> dict:
    """What comes out, with the mask and without it."""
    rows = []
    for skill in SKILLS:
        rng = np.random.default_rng(SEED)
        counts = {"constrained": {"finished": 0, "parses": 0, "tokens": []},
                  "unconstrained": {"finished": 0, "parses": 0, "tokens": []}}
        for _ in range(TRIALS):
            for mode, on in (("constrained", True), ("unconstrained", False)):
                r = gr.generate(_skilled_logits(skill, rng),
                                max_tokens=MAX_TOKENS, rng=rng,
                                constrained=on)
                counts[mode]["finished"] += r["finished"]
                counts[mode]["parses"] += r["parses"]
                counts[mode]["tokens"].append(r["tokens"])
        row = {"skill": skill, "trials": TRIALS}
        for mode in counts:
            c = counts[mode]
            row[f"{mode}_finished"] = c["finished"] / TRIALS
            row[f"{mode}_parses"] = c["parses"] / TRIALS
            # Validity among the documents that actually ended. A walk
            # cut off at the token limit is incomplete, which is a
            # different failure from one that ended and was wrong, and
            # mixing the two would credit the mask with a defect it
            # does not have.
            row[f"{mode}_valid_if_finished"] = (
                c["parses"] / c["finished"] if c["finished"] else None)
            row[f"{mode}_median_tokens"] = float(np.median(c["tokens"]))
        rows.append(row)
    return {"rows": rows, "trials": TRIALS, "max_tokens": MAX_TOKENS}


def compiled_table() -> dict:
    """States against masks, as the grammar is allowed to nest deeper."""
    rows = []
    for depth in DEPTHS:
        states = gr.reachable_states(depth)
        masks = gr.mask_table(depth)
        rows.append({
            "depth": depth,
            "states": len(states),
            "masks": len(masks),
            "states_per_mask": len(states) / len(masks),
        })
    table = gr.mask_table(max(DEPTHS))
    sizes = [int(m.sum()) for m in table.values()]
    return {
        "rows": rows,
        "vocab": len(gr.VOCAB),
        "masks": len(table),
        "bytes_if_packed": len(table) * len(gr.VOCAB) / 8,
        "allowed_min": min(sizes), "allowed_max": max(sizes),
        "allowed_mean": float(np.mean(sizes)),
        "allowed_mean_share": float(np.mean(sizes)) / len(gr.VOCAB),
    }


def cost_per_token() -> dict:
    """What the mask costs, counted against what a decode step costs.

    Not timed. The work a mask adds is one lookup and one addition per
    token of the vocabulary, against a decode step that reads every
    weight in the model -- so the comparison that matters is a count
    of operations against a count of bytes, both of which are exact.
    """
    vocab_real = MODEL.vocab_size
    step = decode_step(1, CONTEXT_TOKENS)
    return {
        "vocab_toy": len(gr.VOCAB),
        "vocab_reference": vocab_real,
        "adds_per_token": vocab_real,
        "mask_bytes_per_token": vocab_real / 8,
        "decode_bytes_per_token": step.bytes_read,
        "mask_share_of_decode_bytes": (vocab_real / 8) / step.bytes_read,
        "decode_flops_per_token": step.flops,
        "mask_share_of_decode_flops": vocab_real / step.flops,
        "params": PARAMS,
        "decode_ms": step.inter_token_ms,
    }


def distortion() -> dict:
    """How much the mask moves the model's own distribution.

    Chapter 29's verification was exact: the tokens it kept had the
    target's distribution precisely. This is not that. A mask deletes
    part of the distribution and renormalises what is left, which is a
    real change to what the model would have said, and the honest
    thing is to measure it rather than to leave the contrast implied.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for skill in SKILLS:
        gaps, removed = [], []
        machine = gr.JsonMachine().step("{")
        for _ in range(400):
            base = rng.standard_normal(len(gr.VOCAB))
            if skill:
                base = base + skill * gr.mask_for(machine)
            free = np.exp(base - base.max()); free /= free.sum()
            masked = gr.apply(base, machine)
            kept = np.exp(masked - masked.max()); kept /= kept.sum()
            gaps.append(0.5 * float(np.abs(kept - free).sum()))
            removed.append(float(free[~gr.mask_for(machine)].sum()))
        rows.append({
            "skill": skill,
            "total_variation": float(np.mean(gaps)),
            "probability_removed": float(np.mean(removed)),
        })
    return {"rows": rows, "state": "just inside an object"}


def main() -> None:
    val, tab = validity(), compiled_table()
    cost, dist = cost_per_token(), distortion()
    payload = {
        "validity": val, "table": tab, "cost": cost, "distortion": dist,
        "assumptions": {
            "seed": SEED, "trials": TRIALS, "max_tokens": MAX_TOKENS,
            "depths": DEPTHS, "skills": SKILLS,
            "vocab": len(gr.VOCAB),
            "reference_vocab": MODEL.vocab_size,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch31.json", payload)
    print(f"wrote {path}")
    print(f"  {val['trials']} documents at each level of the model's own "
          f"skill at JSON:")
    print(f"    {'skill':>6}{'constrained':>28}{'unconstrained':>28}")
    for r in val["rows"]:
        u = r["unconstrained_valid_if_finished"]
        print(f"    {r['skill']:>6}"
              f"{r['constrained_valid_if_finished'] * 100:>14.1f}% valid"
              f"{r['constrained_finished'] * 100:>12.0f}% finished"
              f"{(u * 100 if u is not None else float('nan')):>14.1f}% valid"
              f"{r['unconstrained_finished'] * 100:>12.0f}% finished")
    print("  the grammar, compiled:")
    for r in tab["rows"]:
        print(f"    nesting depth {r['depth']}: {r['states']:5,} states, "
              f"{r['masks']:3,} distinct masks "
              f"({r['states_per_mask']:.0f} states a mask)")
    print(f"    every mask allows {tab['allowed_min']}-{tab['allowed_max']} "
          f"of {tab['vocab']} tokens, {tab['allowed_mean_share'] * 100:.0f}% "
          f"on average; the whole table packs into "
          f"{tab['bytes_if_packed']:.0f} bytes")
    c = cost
    print(f"  per token, at the reference model's {c['vocab_reference']:,} "
          f"token vocabulary: {c['adds_per_token']:,} additions and "
          f"{c['mask_bytes_per_token']:,.0f} bytes of mask, against a decode "
          f"step's {c['decode_bytes_per_token'] / 1e9:.1f} GB "
          f"({c['mask_share_of_decode_bytes'] * 100:.5f}% of it)")
    print("  what the mask does to the distribution it is applied to:")
    for r in dist["rows"]:
        print(f"    skill {r['skill']}: removes "
              f"{r['probability_removed'] * 100:5.1f}% of the probability, "
              f"total variation {r['total_variation']:.3f}")


if __name__ == "__main__":
    main()
