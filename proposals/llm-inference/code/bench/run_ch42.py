"""Chapter 42: what a million tokens costs, and whether to serve them.

Four pieces of arithmetic, all over the fleet Chapter 41 sized:

1. Cost per million output tokens at full utilization: the number
   every other number here is a correction to.
2. What diurnal traffic does to it. A fleet is sized for the peak and
   billed for the day, and the gap between those is most of the bill.
3. The break-even against paying someone else per token, as a
   utilization and as a volume.
4. What the pricing models do to the answer: on-demand, committed
   capacity, and the discount that turns out not to exist.

Prices are recorded in FACTS.md with the page they came from and the
date. They move; everything here is swept rather than stated, so a
reader with different prices can find their own answer.

    python3 -m bench.run_ch42
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tinyserve.reference import (GPU_USD_PER_HOUR, OUTPUT_TOKENS,
                                 PROMPT_TOKENS, REQUESTS_PER_S)

from .harness import write

HOURS_PER_MONTH = 24 * 365 / 12

# Per million tokens, from the provider's own page (FACTS.md,
# 2026-09-21). An 8B-class open model, which is what the case study
# serves, so this is the alternative to serving it yourself.
API_USD_PER_M_OUTPUT = 0.14
API_USD_PER_M_INPUT = 0.14
API_MODEL = "Llama 3 8B Instruct Lite, Together"

# Named on-demand prices for the same accelerator, to show the spread
# the book's single figure is a median of (FACTS.md).
PRICES = {
    "marketplace, low end": 1.49,
    "the book's median": GPU_USD_PER_HOUR,
    "Lambda, H100 PCIe": 3.29,
    "Lambda, H100 SXM (8-GPU)": 3.99,
    "Lambda, 1-Click Cluster (16 GPUs)": 6.16,
    "hyperscaler, high end": 6.98,
}
# How much a day's traffic varies between its busiest and quietest
# hour. The case study says the traffic is diurnal; this is the shape.
PEAK_TO_TROUGH = [1, 2, 3, 4, 6, 8]
UTILIZATIONS = [round(u, 2) for u in np.arange(0.05, 1.001, 0.05)]
# What the hardware bill is a fraction of. Engineering, on-call and
# the rest are not free, and the ratio is recorded in FACTS.md as a
# range; the sweep carries both ends.
OVERHEADS = {"hardware only": 1.0, "with 3x engineering": 3.0,
             "with 5x engineering": 5.0}


def sizing() -> dict:
    """The fleet Chapter 41 sized, read from the file that sized it."""
    path = Path("results/ch41.json")
    if not path.exists():
        raise SystemExit("run `make ch41` first: this chapter prices the "
                         "fleet that chapter sized")
    d = json.loads(path.read_text())
    plan, cap = d["sizing"], d["capacity"]
    machines = plan["machines"]["measured_promise"]
    return {
        "machines": machines,
        "demand_requests_per_s": plan["demand_requests_per_s"],
        "per_machine_requests_per_s": plan["highest_rate_meeting_both_promises"],
        "capacity_requests_per_s": cap["requests_per_s"],
        "output_tokens": OUTPUT_TOKENS,
        "prompt_tokens": PROMPT_TOKENS,
        "peak_tokens_per_s": plan["demand_requests_per_s"] * OUTPUT_TOKENS,
        # What an API would bill for. A request is charged on both its
        # prompt and its reply, and this service sends four times as
        # many tokens in as it gets out -- so a comparison on output
        # alone understates what the API costs by most of the bill.
        "peak_input_tokens_per_s":
            plan["demand_requests_per_s"] * PROMPT_TOKENS,
    }


def cost_per_million(machines: int, usd_per_hour: float,
                     tokens_per_s: float, overhead: float = 1.0) -> float:
    """Dollars per million output tokens actually produced."""
    if tokens_per_s <= 0:
        return float("inf")
    usd_per_s = machines * usd_per_hour * overhead / 3600
    return usd_per_s / tokens_per_s * 1e6


def at_full_tilt(fleet: dict) -> dict:
    """The headline number, and how much the price alone moves it."""
    rows = []
    for name, price in PRICES.items():
        rows.append({
            "price_name": name, "usd_per_gpu_hour": price,
            "usd_per_hour": fleet["machines"] * price,
            "usd_per_month": fleet["machines"] * price * HOURS_PER_MONTH,
            "usd_per_m_tokens": cost_per_million(
                fleet["machines"], price, fleet["peak_tokens_per_s"]),
        })
    lo = min(rows, key=lambda r: r["usd_per_m_tokens"])
    hi = max(rows, key=lambda r: r["usd_per_m_tokens"])
    return {"rows": rows, "spread": hi["usd_per_m_tokens"] / lo["usd_per_m_tokens"],
            "cheapest": lo["price_name"], "dearest": hi["price_name"]}


def duty_cycle() -> dict:
    """What a day of diurnal traffic does to the average utilization.

    The fleet is sized for the busiest hour and paid for all of them.
    A sinusoidal day with a given peak-to-trough ratio has a mean that
    is a fixed fraction of its peak, and that fraction is the
    utilization everything else is divided by.
    """
    hours = np.arange(24)
    rows = []
    for ratio in PEAK_TO_TROUGH:
        # A sine between trough and peak, peaking in the afternoon.
        peak, trough = 1.0, 1.0 / ratio
        mid, amp = (peak + trough) / 2, (peak - trough) / 2
        shape = mid + amp * np.cos((hours - 15) / 24 * 2 * np.pi)
        rows.append({
            "peak_to_trough": ratio,
            "mean_over_peak": float(shape.mean()),
            "hours_above_80pct": int((shape > 0.8).sum()),
            "shape": shape.tolist(),
        })
    return {"rows": rows, "hours": hours.tolist()}


def api_usd_per_hour(fleet: dict, utilization: float) -> float:
    """What the same traffic would cost billed per token.

    Both directions. A provider charges for the prompt as well as the
    reply, and this service sends four times as many tokens in as it
    takes out, so leaving the prompt out understates the alternative by
    most of its cost.
    """
    out_per_s = fleet["peak_tokens_per_s"] * utilization
    in_per_s = fleet["peak_input_tokens_per_s"] * utilization
    per_s = (out_per_s * API_USD_PER_M_OUTPUT
             + in_per_s * API_USD_PER_M_INPUT) / 1e6
    return per_s * 3600


def against_the_api(fleet: dict) -> dict:
    """Where paying per token beats owning the machines.

    Swept over utilization rather than stated at one, because
    utilization is the variable that decides it and the one a team
    controls least. Both sides are compared as dollars an hour for the
    same traffic, which is the only comparison that holds when one
    side is billed per token and the other per machine-hour.
    """
    rows = []
    for overhead_name, overhead in OVERHEADS.items():
        for u in UTILIZATIONS:
            own_hour = fleet["machines"] * GPU_USD_PER_HOUR * overhead
            api_hour = api_usd_per_hour(fleet, u)
            tokens = fleet["peak_tokens_per_s"] * u
            rows.append({
                "overhead": overhead_name, "overhead_factor": overhead,
                "utilization": float(u),
                "own_usd_per_hour": own_hour,
                "api_usd_per_hour": api_hour,
                "usd_per_m_tokens": cost_per_million(
                    fleet["machines"], GPU_USD_PER_HOUR, tokens, overhead),
                "beats_api": bool(own_hour < api_hour),
                "output_tokens_per_day": tokens * 86400,
                "billed_tokens_per_day": (tokens + fleet["peak_input_tokens_per_s"]
                                          * u) * 86400,
            })
    breakeven = {}
    for name in OVERHEADS:
        wins = [r for r in rows if r["overhead"] == name and r["beats_api"]]
        if wins:
            first = min(wins, key=lambda r: r["utilization"])
            breakeven[name] = {
                "utilization": first["utilization"],
                "output_tokens_per_day": first["output_tokens_per_day"],
                "billed_tokens_per_day": first["billed_tokens_per_day"],
                "usd_per_m_tokens": first["usd_per_m_tokens"],
            }
        else:
            breakeven[name] = None
    at_peak = [r for r in rows if r["utilization"] == 1.0
               and r["overhead"] == "hardware only"][0]
    return {
        "rows": rows, "breakeven": breakeven,
        "api_usd_per_m_output": API_USD_PER_M_OUTPUT,
        "api_usd_per_m_input": API_USD_PER_M_INPUT,
        "api_model": API_MODEL,
        "at_peak_own_usd_per_hour": at_peak["own_usd_per_hour"],
        "at_peak_api_usd_per_hour": at_peak["api_usd_per_hour"],
        "at_peak_ratio": at_peak["api_usd_per_hour"] / at_peak["own_usd_per_hour"],
        "at_full_tilt": cost_per_million(fleet["machines"], GPU_USD_PER_HOUR,
                                         fleet["peak_tokens_per_s"]),
        "input_over_output": fleet["peak_input_tokens_per_s"]
                             / fleet["peak_tokens_per_s"],
    }


def committed() -> dict:
    """What a commitment buys, at one provider that publishes both.

    Recorded because it is the opposite of what people assume: this
    provider's committed clusters cost more per GPU-hour than its
    on-demand instances, because a cluster is dedicated interconnected
    capacity and not a volume discount.
    """
    on_demand = PRICES["Lambda, H100 SXM (8-GPU)"]
    cluster = PRICES["Lambda, 1-Click Cluster (16 GPUs)"]
    return {
        "on_demand": on_demand,
        "cluster": cluster,
        "ratio": cluster / on_demand,
        "cluster_is_cheaper": cluster < on_demand,
        "spot_offered": False,
        "note": "lambda.ai/pricing, 2026-09-21",
    }


def main() -> None:
    fleet = sizing()
    full, duty = at_full_tilt(fleet), duty_cycle()
    api, comm = against_the_api(fleet), committed()
    payload = {
        "fleet": fleet, "full_tilt": full, "duty_cycle": duty,
        "api": api, "committed": comm,
        "assumptions": {
            "hours_per_month": HOURS_PER_MONTH,
            "prices": PRICES, "peak_to_trough": PEAK_TO_TROUGH,
            "utilizations": UTILIZATIONS, "overheads": OVERHEADS,
            "api_usd_per_m_output": API_USD_PER_M_OUTPUT,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "requests_per_s": REQUESTS_PER_S,
            "output_tokens": OUTPUT_TOKENS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch42.json", payload)
    print(f"wrote {path}")
    print(f"  {fleet['machines']} machines for "
          f"{fleet['demand_requests_per_s']} requests a second, "
          f"{fleet['peak_tokens_per_s']:,.0f} output tokens a second at peak")
    print("  at full tilt, by what a GPU-hour costs:")
    for r in full["rows"]:
        print(f"    {r['price_name']:34} ${r['usd_per_gpu_hour']:5.2f}/hr  "
              f"${r['usd_per_hour']:7.2f}/hr fleet  "
              f"${r['usd_per_m_tokens']:.3f}/M tokens")
    print(f"    spread from cheapest to dearest: {full['spread']:.1f}x")
    print("  a diurnal day:")
    for r in duty["rows"]:
        print(f"    peak/trough {r['peak_to_trough']}: average load is "
              f"{r['mean_over_peak'] * 100:.0f}% of peak, "
              f"{r['hours_above_80pct']} hours a day above 80%")
    print(f"  against {api['api_model']} at "
          f"${api['api_usd_per_m_output']:.2f}/M out, "
          f"${api['api_usd_per_m_input']:.2f}/M in "
          f"(this service sends {api['input_over_output']:.0f}x as many "
          f"tokens in as out):")
    print(f"    at peak: own it for ${api['at_peak_own_usd_per_hour']:,.2f}/hr, "
          f"rent it for ${api['at_peak_api_usd_per_hour']:,.2f}/hr "
          f"({api['at_peak_ratio']:.1f}x)")
    for name, b in api["breakeven"].items():
        if b is None:
            print(f"    {name:22} never beats the API at any utilization")
        else:
            print(f"    {name:22} breaks even at "
                  f"{b['utilization'] * 100:.0f}% utilization "
                  f"({b['billed_tokens_per_day'] / 1e6:,.0f}M billed tokens "
                  f"a day)")
    print(f"  a commitment at one provider: on-demand ${comm['on_demand']}, "
          f"committed cluster ${comm['cluster']} "
          f"({comm['ratio']:.2f}x -- "
          f"{'cheaper' if comm['cluster_is_cheaper'] else 'dearer'})")


if __name__ == "__main__":
    main()
