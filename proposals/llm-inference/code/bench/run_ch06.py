"""Chapter 6: build or buy, decided by utilization rather than opinion.

Self-hosting is cheaper per token than an API only if the accelerator
stays busy. Rent a card by the hour and you pay for every hour, used or
not, so the cost per token is the cost at full tilt divided by how full
you keep it. This computes where each configuration crosses a published
API price.

Arithmetic over the shared serving model and verified prices
(FACTS.md), not a measurement.

    python3 -m bench.run_ch06
"""

from __future__ import annotations

from tinyserve import serving
from tinyserve.reference import CONTEXT_TOKENS

from .harness import write

# Verified September 2026; see FACTS.md.
API_USD_PER_M_OUTPUT = 0.05
PRICES = {"on-demand median": 3.25, "cheapest marketplace": 1.49}
PRECISIONS = {"bf16 (two bytes)": 2, "fp8 (one byte)": 1}
MAX_CONCURRENT = 325
UTILIZATIONS = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
# Engineering and operations, as a multiple of the raw hardware bill.
ENGINEERING_MULTIPLE = (3, 5)


def main() -> None:
    scenarios = []
    for precision, elem in PRECISIONS.items():
        step = serving.decode_step(MAX_CONCURRENT, CONTEXT_TOKENS,
                                   bytes_per_weight=elem)
        for price_name, usd_hr in PRICES.items():
            at_full = step.usd_per_m_tokens(usd_hr)
            breakeven = at_full / API_USD_PER_M_OUTPUT
            scenarios.append({
                "precision": precision, "pricing": price_name,
                "usd_per_hour": usd_hr,
                "tokens_per_s": step.tokens_per_s,
                "inter_token_ms": step.inter_token_ms,
                "usd_per_m_at_full": at_full,
                "breakeven_utilization": breakeven,
                "achievable": breakeven <= 1.0,
                "curve": [{"utilization": u, "usd_per_m": at_full / u}
                          for u in UTILIZATIONS],
                # Tokens a day needed to hold the card at break-even.
                "tokens_per_day_at_breakeven": (step.tokens_per_s * 86400 * breakeven
                                                if breakeven <= 1 else None),
            })

    best = min(scenarios, key=lambda s: s["usd_per_m_at_full"])
    worst = max(scenarios, key=lambda s: s["usd_per_m_at_full"])

    payload = {
        "model_note": "arithmetic over the shared serving model and verified "
                      "prices; not a measurement",
        "api": {"usd_per_m_output": API_USD_PER_M_OUTPUT,
                "note": "cheapest tracked hosted 8B, September 2026"},
        "assumptions": {"batch": MAX_CONCURRENT, "context": CONTEXT_TOKENS,
                        "prices_usd_per_hour": PRICES,
                        "engineering_multiple": ENGINEERING_MULTIPLE},
        "scenarios": scenarios,
        "summary": {
            "best": {"precision": best["precision"], "pricing": best["pricing"],
                     "usd_per_m_at_full": best["usd_per_m_at_full"],
                     "breakeven_utilization": best["breakeven_utilization"]},
            "worst": {"precision": worst["precision"], "pricing": worst["pricing"],
                      "usd_per_m_at_full": worst["usd_per_m_at_full"],
                      "breakeven_utilization": worst["breakeven_utilization"]},
            "n_achievable": sum(1 for s in scenarios if s["achievable"]),
            "n_total": len(scenarios),
        },
    }

    path = write("results/ch06.json", payload)
    print(f"wrote {path}")
    for s in scenarios:
        b = s["breakeven_utilization"]
        verdict = (f"needs {b * 100:.0f}% utilization" if b <= 1
                   else f"IMPOSSIBLE (would need {b * 100:.0f}%)")
        print(f"  {s['precision']:18} {s['pricing']:22} "
              f"${s['usd_per_m_at_full']:.3f}/M at full tilt -> {verdict}")


if __name__ == "__main__":
    main()
