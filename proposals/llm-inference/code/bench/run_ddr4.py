"""Design decision record IV: the case study's decoding accelerations.

Part VI measured five things. This decides, from those measurements and
nothing else, which of them the case study turns on.

As in Design decision record I, almost nothing new runs here: every
decision is read out of the results files Chapters 29 to 33 already
wrote, so a decision cannot drift from the evidence behind it.

The one thing measured here and nowhere else is what the accelerations
do to *each other*. Each chapter measured one of them against a bare
server. A cache in front of the model removes requests, which lowers
the batch, and a lower batch is exactly where speculative decoding
starts paying -- so the two are not independent, and the record has to
say which way to spend the saving.

    python3 -m bench.run_ddr4
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from tinyserve.reference import GPU_USD_PER_HOUR, OUTPUT_TOKENS, REQUESTS_PER_S

from .harness import write

RESULTS = Path("results")
NEEDS = ["ch29", "ch30", "ch31", "ch32", "ch33", "ch41", "ch42"]
HIT_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def read() -> dict[str, dict]:
    missing = [c for c in NEEDS if not (RESULTS / f"{c}.json").exists()]
    if missing:
        raise SystemExit(f"run `make {' '.join(missing)}` first: this record "
                         "has almost no measurements of its own")
    return {c: json.loads((RESULTS / f"{c}.json").read_text()) for c in NEEDS}


def _interpolate(points: list[tuple[float, float]], x: float) -> float:
    """Log-log interpolation between measured points, clamped at the ends.

    Chapter 30 measured speculation at four batch sizes and the fleet
    runs at none of them. Interpolating between measurements is honest
    where extrapolating past them is not, so this clamps rather than
    extends.
    """
    pts = sorted(points)
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return math.exp(math.log(y0) + t * (math.log(y1) - math.log(y0)))
    return pts[-1][1]


def interaction(r: dict[str, dict]) -> dict:
    """What a cache hit does to the batch, and the batch to speculation.

    Two ways to spend the same hit rate, measured on the same numbers:
    shrink the fleet, or keep it and let the batch fall. Only one of
    them makes speculative decoding worth turning on.
    """
    ch41, ch30, ch32 = r["ch41"], r["ch30"], r["ch32"]
    capacity = ch41["sizing"]["highest_rate_meeting_both_promises"]
    fleet = ch41["sizing"]["machines"]["measured_promise"]
    batch_at = [(row["rate"], row["mean_batch"]) for row in ch41["sweep"]
                if row["settled"]]
    speedup_at = [(row["batch"], row["best_speedup"])
                  for row in ch30["best_budget"]]
    nodes_at = [(row["batch"], float(row["best_nodes"]))
                for row in ch30["best_budget"]]

    rows = []
    for h in HIT_RATES:
        # Spend it on machines: the fleet shrinks and each machine is
        # back at the load it was sized for.
        shrunk = math.ceil(REQUESTS_PER_S * (1 - h) / capacity)
        # Spend it on headroom: the fleet stays and each machine sees
        # less.
        eased_rate = REQUESTS_PER_S * (1 - h) / fleet
        eased_batch = _interpolate(batch_at, max(eased_rate, 1e-9))
        full_batch = _interpolate(batch_at, REQUESTS_PER_S / fleet)
        rows.append({
            "hit_rate": h,
            "fleet_if_shrunk": shrunk,
            "usd_if_shrunk": shrunk * GPU_USD_PER_HOUR,
            "fleet_if_kept": fleet,
            "rate_per_machine_if_kept": eased_rate,
            "batch_if_kept": eased_batch,
            "batch_if_shrunk": full_batch,
            "speculation_if_kept": _interpolate(speedup_at, eased_batch),
            "speculation_if_shrunk": _interpolate(speedup_at, full_batch),
            "tree_if_kept": _interpolate(nodes_at, eased_batch),
        })
    base = rows[0]
    best = max(rows, key=lambda x: x["speculation_if_kept"])
    return {
        "rows": rows, "capacity_per_machine": capacity, "fleet": fleet,
        "batch_at_capacity": base["batch_if_shrunk"],
        "speculation_at_capacity": base["speculation_if_kept"],
        "best_speculation": best["speculation_if_kept"],
        "best_hit_rate": best["hit_rate"],
        "speculation_range": (best["speculation_if_kept"]
                              / base["speculation_if_kept"] - 1),
        "independent": (best["speculation_if_kept"]
                        / base["speculation_if_kept"] - 1) < 0.15,
        "response_hit_measured": ch32["worth"]["response_hit_rate"],
        "note": ("Chapter 30's speedups were measured at four batch sizes "
                 "and interpolated between them, never past them"),
    }


def decisions(r: dict[str, dict]) -> list[dict]:
    """Each decision, its alternatives, and the measurement behind it."""
    ch29, ch30, ch31 = r["ch29"], r["ch30"], r["ch31"]
    ch32, ch33, ch41 = r["ch32"], r["ch33"], r["ch41"]

    batch = _interpolate([(row["rate"], row["mean_batch"])
                          for row in ch41["sweep"] if row["settled"]],
                         REQUESTS_PER_S / ch41["sizing"]["machines"]["measured_promise"])
    spec_here = _interpolate([(x["batch"], x["best_speedup"])
                              for x in ch30["best_budget"]], batch)
    nodes = [(x["batch"], float(x["best_nodes"])) for x in ch30["best_budget"]]
    at_one = next(x for x in ch30["best_budget"] if x["batch"] == 1)
    medusa = next(x for x in ch30["heads"]["rows"]
                  if x["name"] == "Medusa, 3 heads")
    grounded = ch30["assumptions"]["grounded_task"]
    prose = next(x for x in ch30["payoff"]["prose continuing the prompt"]
                 if x["budget"] == 8)
    exact = next(x for x in ch32["exact"]
                 if x["shape"] == "FAQ"
                 and x["normalizer"] == "case and punctuation")
    semantic = next(x for x in ch32["semantic"]["rows"]
                    if x["threshold"] == ch32["semantic"]["example_threshold"])
    worth = ch32["worth"]
    tab = ch31["table"]
    cost31 = ch31["cost"]
    long_ctx = ch33["long_context"]["rows"][-1]
    mo = ch33["mixture"]

    return [
        {
            "decision": "Speculative decoding behind a measurement, with the "
                        "tree sized to the batch rather than to a paper",
            "instead_of": "on by default at a published tree size, or off "
                          "because the batch is large",
            "chapter": "ch30",
            "because": (f"at the fleet's batch of {batch:.0f} a tree of "
                        f"{_interpolate(nodes, batch):.0f} tokens is worth "
                        f"{spec_here:.2f}x -- speculation survives a large "
                        f"batch -- while the tree chosen at batch 1 is "
                        f"{ch30['best_budget'][-1]['widest_speedup']:.2f}x, "
                        "slower than not speculating at all; and whether it "
                        "pays at all depends on an acceptance rate this "
                        "traffic has never been measured for"),
            "evidence": {
                "batch at the operating rate": batch,
                "best tree there, tokens": _interpolate(nodes, batch),
                "best speedup there": spec_here,
                "best speedup at batch 1": at_one["best_speedup"],
                "batch-1 tree run at batch 128":
                    ch30["best_budget"][-1]["widest_speedup"],
                "tokens a verification pass carries free at batch 1":
                    ch30["free_nodes"][0]["free_nodes"],
                "and at batch 128": ch30["free_nodes"][-1]["free_nodes"],
                "drafting rate measured, grounded replies":
                    ch30["drafting"][0]["reach"]["1"][0],
                "drafting rate measured, ungrounded prose":
                    ch30["drafting"][2]["reach"]["1"][0],
            },
            "would_change": ("the measurement that has not been made: the "
                             "share of this service's replies that quote "
                             "their prompt. Chapter 30 measured that number "
                             "spanning "
                             f"{ch30['drafting'][2]['reach']['1'][0] * 100:.0f}% "
                             f"to {ch30['drafting'][0]['reach']['1'][0] * 100:.0f}% "
                             "across four shapes of reply, which is the "
                             "difference between worthwhile and pointless"),
        },
        {
            "decision": "If it is on, draft from the prompt",
            "instead_of": "training and serving a draft head",
            "chapter": "ch30",
            "because": (f"three Medusa heads add "
                        f"{medusa['bytes'] / 1e9:.2f} GB to a "
                        f"{ch30['heads']['weight_bytes'] / 1e9:.1f} GB model, "
                        f"{medusa['share_of_model'] * 100:.1f}% more weight "
                        f"read on every step, so the speedup starts "
                        f"{medusa['step_ratio']:.3f}x behind; drafting from "
                        f"the prompt adds nothing and returns "
                        f"{next(x['chain_speedup'] for x in ch30['payoff'][grounded] if x['budget'] == 8):.2f}x "
                        "on input-grounded replies"),
            "evidence": {
                "Medusa, 3 heads, GB added": medusa["bytes"] / 1e9,
                "share of the model": medusa["share_of_model"],
                "what it costs every step": medusa["step_ratio"],
                "prompt lookup on a grounded reply":
                    next(x["chain_speedup"] for x in ch30["payoff"][grounded]
                         if x["budget"] == 8),
                "prompt lookup on ungrounded prose": prose["chain_speedup"],
            },
            "would_change": ("traffic that is not grounded in its prompt: "
                             f"the same drafter returns "
                             f"{prose['chain_speedup']:.2f}x there, and a "
                             "trained head is the only option left"),
        },
        {
            "decision": "Constrain the structured endpoint, compiling each "
                        "schema once at startup",
            "instead_of": "prompting for JSON and parsing defensively, or "
                          "compiling per request",
            "chapter": "ch31",
            "because": (f"the mask costs "
                        f"{cost31['mask_share_of_decode_bytes'] * 100:.5f}% of "
                        f"a decode step's traffic and the whole grammar "
                        f"compiles to {tab['bytes_if_packed']:.0f} bytes, so "
                        "the only real cost is the compile, which a fixed "
                        "schema pays once"),
            "evidence": {
                "distinct masks, any nesting depth": tab["masks"],
                "the whole table, packed (bytes)": tab["bytes_if_packed"],
                "mask's share of a decode step's bytes":
                    cost31["mask_share_of_decode_bytes"],
                "valid documents, constrained":
                    ch31["validity"]["rows"][0]["constrained_valid_if_finished"],
                "valid documents, unconstrained":
                    ch31["validity"]["rows"][0]["unconstrained_valid_if_finished"],
            },
            "would_change": ("accepting arbitrary JSON Schema per request, "
                             "which moves the compile onto the request path "
                             "and into the first-token latency"),
        },
        {
            "decision": "An exact-match response cache, keyed on every field "
                        "the answer depends on",
            "instead_of": "no cache above the model, or a semantic one",
            "chapter": "ch32",
            "because": (f"an exact cache answers "
                        f"{exact['hit_rate'] * 100:.0f}% of single-turn "
                        f"traffic with no wrong answers, while a similarity "
                        f"threshold set above the worst confusable pair still "
                        f"answers {semantic['wrong_rate'] * 100:.0f}% of "
                        "requests wrongly on traffic that carries account "
                        "numbers"),
            "evidence": {
                "exact hit rate, single-turn": exact["hit_rate"],
                "exact wrong answers": exact["wrong_rate"],
                "semantic hit rate at the safe threshold": semantic["hit_rate"],
                "semantic wrong answers there": semantic["wrong_rate"],
                "wrong answers per extra hit it buys":
                    semantic["wrong_per_extra_hit"],
                "what a 30% hit rate is worth, machines":
                    worth["fleet"]["no cache"] - worth["fleet"]["response cache"],
            },
            "would_change": ("traffic with no repeated questions, where the "
                             "cache is dead weight -- and agent traffic, "
                             "where it already is"),
        },
        {
            "decision": "No thinking budget on the default endpoint",
            "instead_of": "thinking on by default, as several providers now "
                          "ship it",
            "chapter": "ch33",
            "because": ("the case study promises a first token inside "
                        f"{ch41['assumptions']['ttft_budget_ms']:,} ms and a "
                        "reply in seconds; a reply behind "
                        f"{ch33['thinking']['rows'][-1]['thinking_tokens']:,} "
                        "thinking tokens takes "
                        f"{ch33['thinking']['rows'][-1]['seconds_to_answer']:,.0f} s "
                        "and costs "
                        f"{ch33['thinking']['rows'][-1]['usd_per_answer'] / ch33['thinking']['baseline_usd']:,.0f} "
                        "times as much"),
            "evidence": {
                "seconds an answer, no thinking":
                    ch33["thinking"]["rows"][0]["seconds_to_answer"],
                "seconds an answer, 32k thinking":
                    ch33["thinking"]["rows"][-1]["seconds_to_answer"],
                "cost multiple":
                    ch33["thinking"]["rows"][-1]["usd_per_answer"]
                    / ch33["thinking"]["baseline_usd"],
                "share of the reply anybody reads":
                    ch33["thinking"]["rows"][-1]["visible_share"],
            },
            "would_change": ("a second endpoint for work that is worth "
                             "minutes and is asked for asynchronously, which "
                             "is what the provider documentation recommends "
                             "above a 32k budget"),
        },
        {
            "decision": "A dense model, and a context ceiling well short of "
                        "128K",
            "instead_of": "a mixture of experts, or an unbounded context "
                          "window",
            "chapter": "ch33",
            "because": (f"{mo['model']['name']}'s weights alone are "
                        f"{mo['weights_gb_bf16']:,.0f} GB and need "
                        f"{mo['machines']} accelerators before anything is "
                        f"served, against a fleet of "
                        f"{ch41['sizing']['machines']['measured_promise']}; "
                        f"and a {long_ctx['context'] // 1024}K context admits "
                        f"{long_ctx['sequences_that_fit']} sequences a "
                        "machine"),
            "evidence": {
                "mixture weights, GB": mo["weights_gb_bf16"],
                "accelerators to hold them": mo["machines"],
                "the case study's fleet":
                    ch41["sizing"]["machines"]["measured_promise"],
                "sequences that fit at 128K":
                    long_ctx["sequences_that_fit"],
                "one prefill there, seconds": long_ctx["prefill_s"],
            },
            "would_change": ("demand large enough to keep a mixture's fleet "
                             "busy, where its cost per token is several times "
                             "better -- this is a decision about scale, not "
                             "about architecture"),
        },
    ]


def main() -> None:
    r = read()
    decs = decisions(r)
    inter = interaction(r)
    payload = {
        "decisions": decs,
        "interaction": inter,
        "assumptions": {
            "chapters": NEEDS, "hit_rates": HIT_RATES,
            "requests_per_s": REQUESTS_PER_S,
            "output_tokens": OUTPUT_TOKENS,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ddr4.json", payload)
    print(f"wrote {path}")
    for d in decs:
        print(f"  {d['decision']}")
        print(f"    instead of {d['instead_of']} ({d['chapter']})")
    print(f"  the interaction, on a fleet of {inter['fleet']}:")
    for row in inter["rows"]:
        print(f"    cache hits {row['hit_rate']:4.0%}: shrink to "
              f"{row['fleet_if_shrunk']:>2} machines "
              f"(${row['usd_if_shrunk']:,.2f}/hr, batch "
              f"{row['batch_if_shrunk']:5.1f}, speculation "
              f"x{row['speculation_if_shrunk']:.2f})  or keep "
              f"{row['fleet_if_kept']} (batch {row['batch_if_kept']:5.1f}, "
              f"speculation x{row['speculation_if_kept']:.2f} with a "
              f"{row['tree_if_kept']:.0f}-token tree)")


if __name__ == "__main__":
    main()
