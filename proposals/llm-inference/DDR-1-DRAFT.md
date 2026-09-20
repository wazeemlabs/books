# Design decision record I

### The case study's scheduler and cache policy

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ddr1.md` and `code/results/ddr1.json`; run `make ddr1` in
`code/` to re-derive and re-render.*

**Closes Part III.** Every number in this record was measured in
Chapters 13 to 19 and is read back out of
those chapters' results files. One thing is
measured here and nowhere else — how many machines the service needs —
and it is marked where it appears.

---

A design decision record is a short document that says what was
decided, what else was on the table, what the measurements said, and
what would change the answer. It is written once, at the moment the
decision is live, and read later by whoever inherits the system and
wants to know whether a choice was reasoned or inherited.

<!-- defines: design decision record -->

It exists because of a specific failure. A system accumulates settings.
Two years later nobody can tell which of them were chosen and which
were defaults nobody touched, so nobody dares change any of them, and
the system slowly calcifies around decisions that may never have been
made. Writing the reason down is cheap. Reconstructing it is not.

Six records close the six parts of this book. This is the first, and it
commits the case study — 200 requests a second, 1,200-token
prompts, 300-token replies, a first token inside
1,000 ms and 50 ms between the words after it — to a
scheduler and a cache policy. Parts IV, V and VI will decide kernels,
model and precision, and deployment, against the same traffic.

## The decision

There are 7 of them, taken from 7 chapters of
measurement.

<!-- include: tables/ddr1-decisions.md -->
| Decision | Instead of | Decided by | What decided it |
|---|---|---|---|
| **Page the KV cache in blocks of 16 tokens** | reserving each sequence's whole context up front, or paging in a larger block | Chapter 14 | reserving the context admits 59 sequences where paging admits 327, and at this block size paging wastes 7.7 tokens of the sequence's own memory |
| **Prefix caching on, with a prefix tree and least-recently-used eviction of leaves** | no prefix cache, an unstructured block cache, or evicting the least frequently used block | Chapter 15 | 85% of prompt tokens are already in the cache at 7 GB, against an 85.4% ceiling, and the lookup costs 135 microseconds a request |
| **Continuous batching: decide the batch every iteration** | a static batch formed on a timeout | Chapter 17 | 1.9x the throughput, and a median first token 1,568x sooner, at the same 12 requests a second on the same memory |
| **Chunked prefill, with a 512-token per-iteration budget** | giving a prefill an iteration of its own, or choosing a larger budget | Chapter 18 | of the budgets that keep both promises at 24 requests a second, this one has the highest throughput (5,586 tokens a second) and the lowest wait between tokens (8.4 ms at the 99th percentile) |
| **First come, first served, with no priority tiers** | shortest job first, or longest job first | Chapter 18 | with the memory this service has, the queue order is worth 0.2% of the median end-to-end time, and the simplest order is the one that needs no estimate of how long a reply will be |
| **Preempt by recomputing, not by swapping the cache out to host memory** | copying an evicted cache out over PCIe and back | Chapter 18 | 1.25x the throughput, although recomputing loses every individual comparison: one 1,500-token sequence costs 23.8 ms to recompute against 6.1 ms to copy |
| **One fleet where every machine does both phases** | splitting the same fleet into 4 prefill machines and 8 decode machines | Chapter 19 | 1.03x the throughput on identical hardware, a first token 3.5x faster at the 99th percentile, and a second token that does not wait for a cache to cross a network |

Every row is read out of the named chapter's results file when this table is generated. Nothing in this table was measured for this record, and nothing in it was typed in by hand: change a chapter's measurement and the row changes with it.

Read down the first column and you have the server: a paged KV cache in
16-token blocks, a prefix tree in front of it, a batch re-decided
every iteration, prompts admitted 512 tokens at a time, served in
the order they arrive, and evicted by throwing their work away rather
than by copying it out. One fleet of identical machines, each doing
both phases.

That is, to within the odd detail, what vLLM and SGLang do by default.
This record's contribution is not the configuration. It is the second
and fourth columns: what each choice was weighed against, and the
number that settled it.

## What was considered

Three of the seven decisions were close, and those are the interesting
ones. The other four were not close at all, and it is worth being
explicit about which is which, because "we measured it" reads the same
either way.

**Not close.** Paging the cache admits 327 sequences where
reserving each sequence's whole context admits 59
(Chapter 14). Continuous batching delivers 1.9x
the throughput of a static batch and a first token three orders of
magnitude sooner (Chapter 17). Prefix caching reaches
85% of prompt tokens already in memory for 7 GB of
cache, at a lookup cost three orders of magnitude below the prefill it
avoids (Chapter 15). Preempting by recomputation rather
than by copying the cache out delivers 1.25x the
throughput and a first-token p99 2.1x better
(Chapter 18) — a wide margin, arrived at counter-intuitively,
which is the subject of the next section. A service that did any of
these four differently would be worse at everything, not better at
something.

**Close, and decided on a margin.** The token budget is a genuine
trade: a smaller one gives a better gap between tokens and a worse
first token, a larger one the reverse, and the measurements only narrow
the choice to a band (Chapter 18). 512 is the best row
in that band, not a different kind of answer from the rows either side
of it.

**Close, and decided on something other than the measurement.** The
queue order is worth 0.2% of the median end-to-end time at
this service's memory. That is not a result; it is an absence of one.
First come, first served was chosen because when three orders perform
identically, the one that needs no estimate of how long a reply will be
is the one to ship. Squeeze the pool to 1.9 GB and the absence
becomes a result — shortest job first is 10.9x better on
the median — which is exactly why the condition is written down in the
next table rather than left for someone to rediscover.

**Close, and decided against the literature.** Disaggregating prefill
and decode is the most-published idea in this part of the field, and on
this traffic, on this hardware, the colocated fleet won:
1.03x the throughput and a first token several times
faster. Chapter 19 spent a section on why
that is not a contradiction of the papers. It is the decision in this
record with the shortest expected life, and the one to revisit first
when the hardware stops being homogeneous.

## What the measurements said

The full evidence for each row is in the results file
(`code/results/ddr1.json`), which carries the specific numbers each
decision turned on. Two of them are worth pulling out here because they
are the ones people get wrong.

**The cheaper operation produced the worse system.** Preempting by
copying a sequence's cache to host memory is, per event,
3.9x cheaper than throwing the work away and
recomputing it: one 1,500-token sequence costs
24 ms to recompute and 6 ms to copy over PCIe 5.0.
Recomputing still won the system — 1.25x the throughput —
because a recomputed sequence re-enters through the chunked prefill
path and takes its memory back gradually, while a swapped one needs
its whole cache restored before it can take a single step and sits in
the queue until that is possible. The arithmetic measured one event.
The system measured the shape of the re-entry, and they disagreed.

**A few per cent of throughput is a machine.** The colocated fleet
beat the best disaggregated split of the same hardware by
3%, which sounds like nothing. At this load it is
the difference between a fleet that keeps up and one that does not,
and a fleet that does not keep up is one that needs another machine.
A percentage measured near a saturation point is not a small number.

### How many machines

This is the one measurement made for this record rather than borrowed
from a chapter. Chapter 19 fixed the fleet
at 12 machines and asked which design used them better.
Nobody asked how few would do, and that is the number the bill is
written against.

<!-- include: tables/ddr1-fleet.md -->
| Machines | Tokens/s | Share of what the traffic asks for | TTFT p99 | Between tokens, p99 | $/hour | $/M output tokens |
|---|---|---|---|---|---|---|
| 4 | 37,044 | 62% (behind) | 5.2 s | 10.3 ms | $13.00 | $0.097 |
| 5 | 45,713 | 77% (behind) | 2.3 s | 10.3 ms | $16.25 | $0.099 |
| 6 | 53,613 | 90% (behind) | 601 ms | 10.3 ms | $19.50 | $0.101 |
| 7 | 56,087 | 94% (behind) | 253 ms | 10.0 ms | $22.75 | $0.113 |
| **8** | 57,143 | 96% | 105 ms | 8.8 ms | $26.00 | $0.126 |
| 9 | 57,425 | 96% | 84 ms | 8.5 ms | $29.25 | $0.141 |
| 10 | 57,648 | 97% | 82 ms | 8.4 ms | $32.50 | $0.157 |
| 11 | 57,770 | 97% | 79 ms | 8.4 ms | $35.75 | $0.172 |
| 12 | 57,861 | 97% | 77 ms | 8.4 ms | $39.00 | $0.187 |
| 14 | 58,011 | 97% | 72 ms | 8.4 ms | $45.50 | $0.218 |
| 16 | 58,177 | 98% | 70 ms | 8.3 ms | $52.00 | $0.248 |

2,000 requests at 200 a second (seed 0), every machine running Chapter 18's scheduler at a 512-token budget over 30,515 blocks. Throughput is measured over the arrival window with the first 50% discarded. A fleet is "behind" when it delivers less than 95% of the tokens the traffic asks for, or misses the 1,000 ms first-token promise. The dollar figures are $3.25 an hour a machine, on-demand (FACTS.md).

**8 machines.** At 7 the fleet delivers
94% of the tokens the traffic asks for and falls behind; at
8 it delivers 96% and keeps up, with a first
token at 105 ms for the slowest request in a hundred, well
inside the 1,000 ms promise. At $3.25 an hour a
machine that is $26.00 an hour, $18,980 a month, and
$0.126 per million output tokens.

Three things in that table are worth more than the number itself.

**Throughput stops improving almost immediately.** Going from
8 machines to 16 — doubling the fleet — moves
delivered throughput from 96% to 98% of what
the traffic asks for. It cannot do better, because the traffic is not
asking for more. Past the point where a fleet keeps up, machines buy
latency, not throughput: the first-token p99 falls from
105 ms to 70 ms across that same doubling.
Whether that is worth 2.0x the cost per token
($0.126 to $0.248 per million) is a product
question, and this record's answer is no.

**The cheapest fleet per token is the one that fails.** The smallest
fleet in the sweep, 4 machines, serves the same traffic at
$0.097 per million tokens against 8's
$0.126 — cheaper, because the machines are busier — and makes
the slowest user in a hundred wait 5.2 s for a first
word, on a fleet that is not keeping up at all. Cost per token, on its
own, is a metric that recommends an unusable service, and it gets
cheaper the further you push in that direction. It means something only
beside a promise that is being kept.

**Chapter 16's arithmetic landed on the same number.** Pricing
one accelerator at a batch of 64, at the full context, with
no prefill and no scheduler, Chapter 16 put the fleet at
8.0 machines. The measured answer is 8. A machine
in the measured fleet delivers 7,143 tokens a second
against the 7,500 that arithmetic allows —
95% of it.

Resist reading that as the roofline predicting the fleet. It left out
prefill, which this fleet spends real iterations on; a batch that moves
with the traffic rather than sitting at 64; the arrival
variance that decides the first-token percentile; and every sequence
whose context is shorter than the 1,500 tokens it charged
for. Four omissions, and here they cancel to within a machine. Nothing
measured in this book says they cancel anywhere else. The roofline is
a floor to check a measurement against — if a fleet comes in above it,
the measurement is wrong — and this record used it that way and then
measured anyway.

## What would change this

<!-- include: tables/ddr1-would-change.md -->
| Decision | What would change it |
|---|---|
| Page the KV cache in blocks of 16 tokens | a kernel that needs longer contiguous runs than a block, or replies short enough that a block's worth of waste stops being small beside them |
| Prefix caching on, with a prefix tree and least-recently-used eviction of leaves | traffic with no shared prefixes, where the index costs something and returns nothing. Nothing measured here makes the feature worth turning off when prefixes are shared at all |
| Continuous batching: decide the batch every iteration | nothing in this book. Static batching lost on every measure at every rate tried |
| Chunked prefill, with a 512-token per-iteration budget | a looser between-token promise, which would buy a larger budget and a better first token; or a tighter one, which the smaller budgets serve at a first token this service could not sell |
| First come, first served, with no priority tiers | running the pool near full. Squeezed, shortest first is 10.9x better on the median and 13.5x better on the worst slowdown, and this decision flips |
| Preempt by recomputing, not by swapping the cache out to host memory | a swap-in that restores a cache gradually rather than all at once. What lost here was the shape of the re-entry, not the cost of the copy: even over NVLink, which is fast enough on the arithmetic, swapping still lost |
| One fleet where every machine does both phases | hardware that differs by phase, or a promise tight enough that one fleet has to over-provision to keep it. Neither is true here, and this is the decision in this record most likely to be wrong for a service that is not this one |

A decision without a condition attached to it is a habit. These are the conditions -- the things that, if they became true of a service, would make the row above the wrong answer for it.

Two of those conditions deserve a sentence more.

**The pool.** Four of the seven decisions are policies about memory:
block size, prefix caching, queue order and preemption mode. How hard
this service pressed on each of them varies, and the record should be
exact about that.

Prefix caching was measured across the whole range: at
3% of the pool the hit rate is 55%,
at the whole pool 85% (Chapter 15). The queue
order was measured at two pool sizes, and the two answers disagree by
10.9x on the median (Chapter 18). The
preemption mode was measured at one — the squeezed
1.9 GB — because at a full pool it cannot be measured at all:
this service preempts 0 sequences there, and
Chapter 17's pool sweep finds the first preemption only
at 10% of the pool, 3 of
them. Block size was measured at one pool size and one alone.

So one of those four settings is doing nothing here, and would be doing
a great deal on a service that ran its memory hot. That is the general
shape of it: a policy about memory does nothing while memory is
plentiful, and this service runs with 64 GB free after the
weights and reaches nothing like the end of it. A service running a
larger model, a longer context, or a bigger batch on the same card is
making all four of these decisions in the regime where they bite, and
should re-measure rather than inherit.

**The hardware.** Every measurement behind this record assumes one kind
of accelerator, and the case for separating prefill from decode rests
almost entirely on having two. The moment a fleet is heterogeneous —
compute-dense parts for prefill, memory-dense parts for decode — the
last row of the decision table should be re-measured before it is
trusted.

## What this record does not decide

Kernels, model, precision, and everything about deployment. This record
fixes the scheduler and the cache policy; it says nothing about how the
attention itself is computed, which is Part IV, or what is being served
and at what precision, which is Part V.

That matters for reading the numbers above. They are all produced by a
cost model, not by a GPU: a step costs the larger of its bytes over
bandwidth and its arithmetic over peak, and every chapter in this part
says so in its own caveats. Part IV's decisions change the constant in
front of that model. They do not change which of these seven options
wins, because every comparison here is between two designs paying the
same constant.

## Sources

The measurements are this book's own, from Chapters
13 to 19; each chapter's Sources section
carries the papers and engine documentation behind the design it
measures. The form of this document — decision, alternatives, evidence,
and the conditions under which it should be revisited — follows the
architecture decision record as described by Michael Nygard in
"Documenting Architecture Decisions" (2011), narrowed here to decisions
that a measurement can settle.

The prices are on-demand accelerator-hours recorded in
[FACTS.md](FACTS.md) with their source and the date they were checked;
they move, and a reader re-running this in a year should expect the
dollar figures to be the first thing that is wrong.
