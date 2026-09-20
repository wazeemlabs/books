# Design decision record I

### The case study's scheduler and cache policy

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ddr1.md` and `code/results/ddr1.json`; run `make ddr1` in
`code/` to re-derive and re-render.*

**Closes Part III.** Every number in this record was measured in
Chapters {{first_chapter}} to {{last_chapter}} and is read back out of
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
commits the case study — {{rate}} requests a second, {{prompt_tokens}}-token
prompts, {{output_tokens}}-token replies, a first token inside
{{ttft_budget}} and {{itl_budget}} between the words after it — to a
scheduler and a cache policy. Parts IV, V and VI will decide kernels,
model and precision, and deployment, against the same traffic.

## The decision

There are {{decisions}} of them, taken from {{chapters}} chapters of
measurement.

<!-- include: tables/ddr1-decisions.md -->

Read down the first column and you have the server: a paged KV cache in
{{block}}-token blocks, a prefix tree in front of it, a batch re-decided
every iteration, prompts admitted {{budget}} tokens at a time, served in
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

**Not close.** Paging the cache admits {{block_paged}} sequences where
reserving each sequence's whole context admits {{block_reserved}}
({{ch:paged-attention}}). Continuous batching delivers {{static_gain}}
the throughput of a static batch and a first token three orders of
magnitude sooner ({{ch:continuous-batching}}). Prefix caching reaches
{{cache_hit}} of prompt tokens already in memory for {{cache_gb}} of
cache, at a lookup cost three orders of magnitude below the prefill it
avoids ({{ch:prefix-caching}}). Preempting by recomputation rather
than by copying the cache out delivers {{recompute_gain}} the
throughput and a first-token p99 {{recompute_ttft_gain}} better
({{ch:chunked-prefill}}) — a wide margin, arrived at counter-intuitively,
which is the subject of the next section. A service that did any of
these four differently would be worse at everything, not better at
something.

**Close, and decided on a margin.** The token budget is a genuine
trade: a smaller one gives a better gap between tokens and a worse
first token, a larger one the reverse, and the measurements only narrow
the choice to a band ({{ch:chunked-prefill}}). {{budget}} is the best row
in that band, not a different kind of answer from the rows either side
of it.

**Close, and decided on something other than the measurement.** The
queue order is worth {{order_worth}} of the median end-to-end time at
this service's memory. That is not a result; it is an absence of one.
First come, first served was chosen because when three orders perform
identically, the one that needs no estimate of how long a reply will be
is the one to ship. Squeeze the pool to {{squeezed_gb}} and the absence
becomes a result — shortest job first is {{order_squeezed}} better on
the median — which is exactly why the condition is written down in the
next table rather than left for someone to rediscover.

**Close, and decided against the literature.** Disaggregating prefill
and decode is the most-published idea in this part of the field, and on
this traffic, on this hardware, the colocated fleet won:
{{colocated_gain}} the throughput and a first token several times
faster. {{ch:disaggregated-prefill-and-decode}} spent a section on why
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
{{recompute_over_swap}} cheaper than throwing the work away and
recomputing it: one {{context_tokens}}-token sequence costs
{{recompute_ms}} to recompute and {{swap_ms}} to copy over PCIe 5.0.
Recomputing still won the system — {{recompute_gain}} the throughput —
because a recomputed sequence re-enters through the chunked prefill
path and takes its memory back gradually, while a swapped one needs
its whole cache restored before it can take a single step and sits in
the queue until that is possible. The arithmetic measured one event.
The system measured the shape of the re-entry, and they disagreed.

**A few per cent of throughput is a machine.** The colocated fleet
beat the best disaggregated split of the same hardware by
{{colocated_gain_pct}}, which sounds like nothing. At this load it is
the difference between a fleet that keeps up and one that does not,
and a fleet that does not keep up is one that needs another machine.
A percentage measured near a saturation point is not a small number.

### How many machines

This is the one measurement made for this record rather than borrowed
from a chapter. {{ch:disaggregated-prefill-and-decode}} fixed the fleet
at {{measured}} machines and asked which design used them better.
Nobody asked how few would do, and that is the number the bill is
written against.

<!-- include: tables/ddr1-fleet.md -->

**{{chosen}} machines.** At {{one_below}} the fleet delivers
{{below_share}} of the tokens the traffic asks for and falls behind; at
{{chosen}} it delivers {{chosen_share}} and keeps up, with a first
token at {{chosen_ttft99}} for the slowest request in a hundred, well
inside the {{ttft_budget}} promise. At {{usd_gpu_hour}} an hour a
machine that is {{usd_hour}} an hour, {{usd_month}} a month, and
{{usd_m_tokens}} per million output tokens.

Three things in that table are worth more than the number itself.

**Throughput stops improving almost immediately.** Going from
{{chosen}} machines to {{biggest}} — doubling the fleet — moves
delivered throughput from {{chosen_share}} to {{biggest_share}} of what
the traffic asks for. It cannot do better, because the traffic is not
asking for more. Past the point where a fleet keeps up, machines buy
latency, not throughput: the first-token p99 falls from
{{chosen_ttft99}} to {{biggest_ttft99}} across that same doubling.
Whether that is worth {{usd_biggest_over_chosen}} the cost per token
({{usd_m_tokens}} to {{biggest_usd_m}} per million) is a product
question, and this record's answer is no.

**The cheapest fleet per token is the one that fails.** The smallest
fleet in the sweep, {{cheapest}} machines, serves the same traffic at
{{cheapest_usd_m}} per million tokens against {{chosen}}'s
{{usd_m_tokens}} — cheaper, because the machines are busier — and makes
the slowest user in a hundred wait {{cheapest_ttft99}} for a first
word, on a fleet that is not keeping up at all. Cost per token, on its
own, is a metric that recommends an unusable service, and it gets
cheaper the further you push in that direction. It means something only
beside a promise that is being kept.

**{{ch:batching}}'s arithmetic landed on the same number.** Pricing
one accelerator at a batch of {{arith_batch}}, at the full context, with
no prefill and no scheduler, {{ch:batching}} put the fleet at
{{arith_fleet}} machines. The measured answer is {{chosen}}. A machine
in the measured fleet delivers {{per_machine_tps}} tokens a second
against the {{arith_tps}} that arithmetic allows —
{{per_machine_share}} of it.

Resist reading that as the roofline predicting the fleet. It left out
prefill, which this fleet spends real iterations on; a batch that moves
with the traffic rather than sitting at {{arith_batch}}; the arrival
variance that decides the first-token percentile; and every sequence
whose context is shorter than the {{context_tokens}} tokens it charged
for. Four omissions, and here they cancel to within a machine. Nothing
measured in this book says they cancel anywhere else. The roofline is
a floor to check a measurement against — if a fleet comes in above it,
the measurement is wrong — and this record used it that way and then
measured anyway.

## What would change this

<!-- include: tables/ddr1-would-change.md -->

Two of those conditions deserve a sentence more.

**The pool.** Four of the seven decisions are policies about memory:
block size, prefix caching, queue order and preemption mode. How hard
this service pressed on each of them varies, and the record should be
exact about that.

Prefix caching was measured across the whole range: at
{{cache_small_frac}} of the pool the hit rate is {{cache_small_hit}},
at the whole pool {{cache_full_hit}} ({{ch:prefix-caching}}). The queue
order was measured at two pool sizes, and the two answers disagree by
{{order_squeezed}} on the median ({{ch:chunked-prefill}}). The
preemption mode was measured at one — the squeezed
{{squeezed_gb}} — because at a full pool it cannot be measured at all:
this service preempts {{full_pool_preemptions}} sequences there, and
{{ch:continuous-batching}}'s pool sweep finds the first preemption only
at {{first_preempt_share}} of the pool, {{first_preempt_count}} of
them. Block size was measured at one pool size and one alone.

So one of those four settings is doing nothing here, and would be doing
a great deal on a service that ran its memory hot. That is the general
shape of it: a policy about memory does nothing while memory is
plentiful, and this service runs with {{pool_gb}} free after the
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
{{first_chapter}} to {{last_chapter}}; each chapter's Sources section
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
