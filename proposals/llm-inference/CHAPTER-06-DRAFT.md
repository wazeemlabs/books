# 6. The Serving Landscape

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ch06.md` and `code/results/ch06.json`; run `make ch06` in
`code/` to recompute and re-render.*

**Depends on:** Chapter 5.
**Tier 0** — arithmetic only, free.

## Objectives

By the end of this chapter you can:

1. Decide whether a given workload should call an API or run its own
   server, and defend the answer with a number.
2. Name the three serving engines and each one's defining idea, and say
   where an orchestrator sits relative to them.
3. Pick a class of accelerator for a model, from the model's size.
4. Find your way around the rest of this book.

## Why it matters

Chapter 5 finished the picture: you know what decoding
costs, why memory is the limit, what batching buys and what it charges.
Before Part II starts measuring, this chapter answers the question a
reader with a real service is holding — **should I be doing this at
all?** — and then lays out the territory.

The honest answer for most teams is no, and it is worth knowing why
before you spend a quarter finding out.

## Buy, unless you can keep it busy

Renting an accelerator is like leasing a lorry. The lease costs the
same whether it is full or empty, so the cost per parcel is the cost
per hour divided by how many parcels you actually move. Call an API
instead and you pay per parcel, with nothing owed when you ship
nothing.

So the decision is not really about technology. **It is a bet that you
can keep an expensive machine busy.**

Here is that bet, priced. Each row runs the case study's model at its
best operating point from Chapter 5 — 325
concurrent sequences — and asks what utilization it would take to beat
a published API price of $0.05 per million output tokens:

<!-- include: tables/ch06-breakeven.md -->
| Precision | GPU pricing | Cost per 1M tokens at full tilt | Utilization needed to beat the API |
|---|---|---|---|
| bf16 (two bytes) | on-demand median | $0.066 | **never** — would need 133% |
| bf16 (two bytes) | cheapest marketplace | $0.030 | **61%** |
| fp8 (one byte) | on-demand median | $0.033 | **66%** |
| fp8 (one byte) | cheapest marketplace | $0.015 | **30%** |

Against a published API price of $0.05 per million output tokens (cheapest tracked hosted 8B, September 2026). Hardware only: engineering and operations typically add another 3-5x on top.

![Cost per million tokens against utilization](code/figures/ch06-breakeven.svg)

**Figure 6.1** — Self-hosting is a bet on utilization. Colour shows the
GPU pricing, line style the precision. *Provenance in
`code/figures/ch06-breakeven.caption.txt`.*

Read the first row twice, because it is the most useful line in this
chapter. **Serving the model the obvious way — full precision, GPU
rented at the on-demand price — cannot beat the API at any
utilization.** It would need 133% of a machine that only
has 100%. The gap is not something you can close with effort; the
configuration is simply wrong.

Now read what changes it. Serving the weights at one byte instead of
two (Chapter 25 and Chapter 26) takes the cost from
$0.066 to $0.033 and turns "impossible" into
"66% utilization". Buying the same card on a cheaper
contract instead of by the hour does about the same. Do both and you
need 30%.

**Only 3 of these 4 configurations can beat
the API, and the techniques this book teaches are what separate them.**
That is the argument for reading on, and it is also the argument for
not self-hosting until you have read on.

> **If you're new here: what utilization means here**
>
> It is the fraction of the time your accelerator is actually decoding
> at the batch size you designed for. Not "the process is running" —
> doing useful work.
>
> Real traffic is diurnal: your busy hour may be ten times your quiet
> hour, and a fleet sized for the peak sits idle at night. A service
> that averages 10% would pay **$0.66** per million tokens on
> the first row above, against $0.05 to buy it. Utilization is
> not a detail of the calculation. It usually *is* the calculation.

And one figure the table cannot show. Everything above is the hardware
bill only. Published analyses put engineering and operations at
**3 to 5 times** the raw cost of the machines —
the people who build the deployment, carry the pager, and do the work
of Part VIII. Include them and the break-even utilizations above move
sharply in the API's favour.

**So: buy, unless volume is large and steady, or your data cannot
leave, or you need a model nobody hosts.** Those three exceptions cover
a great many real services, which is why the rest of this book exists.
They are exceptions nonetheless.

## The three engines

If you do serve it yourself, you will not write the server. Three
open-source engines do this work, and they differ less than their
documentation suggests — each has the KV cache of Chapter 12,
the paging of Chapter 14 and the continuous batching of
Chapter 17. What differs is the idea each one was
built around.

- **vLLM** — built around **paging the KV cache**. It introduced the
  idea that Chapter 14 builds, and it is the default
  choice: the widest model support and the largest community.
  Chapter 34 takes it apart.
- **SGLang** — built around **reusing shared prefixes**. Its radix tree
  of cached prompts, which Chapter 15 builds, makes it
  strong where many requests begin the same way: agents, few-shot
  prompts, long system messages. Chapter 35 covers it.
- **TensorRT-LLM** — built around **compiling ahead of time**. NVIDIA's
  engine trades flexibility for a compilation step that specializes the
  model to the exact card and shapes. Fastest where the workload is
  fixed and known. Chapter 36 covers it.

**And one thing that is not an engine.** NVIDIA's **Dynamo** is an
orchestration layer that sits *above* these: it schedules whichever
engine you chose across many accelerators and handles splitting prefill
from decode at rack scale (Chapter 19). It replaces none of
them, and reading it as a fourth option is a common and costly
confusion.

Which to choose is a question worth measuring rather than debating, and
Chapter 37 runs the same traffic through all three on the
same hardware.

## Which accelerator

A model must fit, with room left for the conversations. From
Chapter 12: weights take two bytes per parameter, and each
token of every live conversation costs cache on top.

| Model | Weights (bf16) | Fits comfortably on |
|---|---|---|
| Up to 3B | under 6 GB | a consumer card, or a laptop |
| 7–9B | 14–18 GB | one 24 GB card (L4, L40S, RTX-class) |
| 30–40B | 60–80 GB | one 80 GB card (A100, H100) |
| 70B | ~140 GB | two 80 GB cards, or one 141 GB H200 |
| Frontier open models | terabytes | a node, or several (Chapter 38) |

Two warnings. First, "fits" is not "serves well": a model that just
fits leaves no room for KV cache, and Chapter 13
shows how quickly that ceiling binds. Second, the biggest card is
rarely the cheapest per token for a small model — an 8B model on an
H100 spends most of that card's capability idle, and a cheaper card
with less bandwidth may well win on cost per token. Measure, do not
assume.

## A map of what follows

You now have every concept Part I can give you. From here:

- **Part II (Chapter 7–Chapter 10)** —
  how to measure honestly. The roofline that predicts whether an idea
  can work, and the harness that every number in this book comes from.
- **Part III (Chapter 11–Chapter 19)** —
  build a serving engine, one mechanism at a time, and watch each one
  pay for itself.
- **Part IV** — read the kernels underneath, and the profiler output
  that points at them.
- **Part V** — make the model smaller, and measure what that costs in
  quality.
- **Part VI** — make decoding produce more per pass.
- **Part VII** — the real engines, and an honest comparison.
- **Part VIII** — capacity, cost, reliability, and the job itself.

If you only want the single highest-value chapter, it is
Chapter 17. If you want the one that most often saves a
service in production, it is Chapter 13.

## Where this is soft

**The break-even table is arithmetic, not a quote.** It uses one
model, one card, one operating point and prices verified in September
2026. Your model, traffic and contract will move every number. The
*method* — cost at full tilt, divided by utilization, against the price
you would otherwise pay — is what to carry away.

**It also flatters self-hosting**, by counting only the hardware. The
3–5x engineering multiple is quoted but not
modelled.

**An 8B model on an H100 is not the cheapest way to serve an 8B
model.** The book uses that pairing so that one reference runs through
every chapter, not because it is optimal. Exercise 6.4 asks you to find
something better.

**Prices move fast, and always downward so far.** Everything here is
recorded in `FACTS.md` with its date. Re-verify before quoting it to
anyone with a budget.

## In production

- **Start on an API.** Ship, learn the traffic, then price the switch
  against real numbers rather than projected ones.
- **Measure utilization before you migrate.** It is the term that
  dominates, and the one teams estimate most optimistically.
- **Two exceptions bypass the arithmetic entirely**: data that may not
  leave your infrastructure, and a model nobody offers. Both are
  decisions made above your pay grade, and both land this book on your
  desk regardless.
- **Revisit the decision yearly.** API prices have fallen faster than
  hardware prices; a migration that paid last year may not this year,
  and the reverse.

## Numbers to remember

| Quantity | Value |
|---|---|
| The self-hosting question | can you keep it busy? |
| Cost per token, self-hosted | cost at full tilt ÷ utilization |
| Naive configuration versus the API | $0.066 against $0.05 — never wins |
| Best configuration here | $0.015, needing 30% utilization |
| Hidden multiplier | engineering and ops, 3–5x the hardware |
| The three engines | vLLM (paging), SGLang (prefix reuse), TensorRT-LLM (compilation) |

## Sources

- Published GPU rental price indexes and hosted-model price sheets,
  September 2026 — the prices throughout, recorded in `FACTS.md`.
- Published build-versus-buy analyses, 2026 — the utilization threshold
  and the engineering multiple, recorded in `FACTS.md`. They put the
  practical threshold near 60%, which is where two of the four rows
  above independently land.
- vLLM, SGLang and TensorRT-LLM documentation at the versions in
  `environment.lock`; NVIDIA's Dynamo product documentation for the
  orchestration layer.

## Exercises

**★ 6.1** Your service produces 4 million output tokens a day and your
accelerator, kept busy, would produce 1.2 billion. What utilization
is that, and what does the table above say you should do?

**★ 6.2** A colleague proposes self-hosting to "save money on the API
bill", quoting the cost at full tilt. Name the two things missing from
their estimate, and say which is likely to be larger.

**★★ 6.3** Your traffic is ten times heavier at noon than at midnight.
Sketch how cost per token varies across the day for a fleet sized to
the peak, and describe two ways to raise the average utilization
without hurting the busy hour.

**★★ 6.4** The chapter admits an 8B model on an H100 is not the
cheapest pairing. Using published bandwidth figures and rental prices
for one cheaper card, estimate its cost per million output tokens with
the method of this chapter. Does it beat the H100? What did you have to
assume, and which assumption is least safe?

**★★★ 6.5** Build the decision as a tool rather than a table. Extend
`run_ch06` to take a traffic profile (requests per hour across a day),
a model size and a price list, and report cost per million tokens for
self-hosting against an API, including a fleet that scales with demand.
Use it on the case study's diurnal traffic and report what fraction of
the saving comes from scaling the fleet rather than from the hardware.
