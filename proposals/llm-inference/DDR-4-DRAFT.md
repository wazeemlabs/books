# Design decision record IV

### The case study's decoding accelerations

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ddr4.md` and `code/results/ddr4.json`; run `make ddr4` in
`code/` to re-derive and re-render.*

**Closes Part VI.** Every number in this record was measured in
Chapters 29 to 33 and is read back out of
those chapters' results files. One thing is measured here and nowhere
else — what the accelerations do to each other — and it is marked
where it appears.

---

Part VI offered five ways to make decoding faster and one way to make
it slower. Design decision record I committed the case study to a
scheduler; this commits it to which of those five it turns on.

The traffic has not changed: 200 requests a second,
Chapter 41's fleet of 8 machines at
26 requests a second each, $3.25 an accelerator-hour.

## The decision

There are 6 of them, taken from 4 chapters of
measurement.

<!-- include: tables/ddr4-decisions.md -->
| Decision | Instead of | Decided by | What decided it |
|---|---|---|---|
| **Speculative decoding behind a measurement, with the tree sized to the batch rather than to a paper** | on by default at a published tree size, or off because the batch is large | Chapter 30 | at the fleet's batch of 60 a tree of 7 tokens is worth 3.13x -- speculation survives a large batch -- while the tree chosen at batch 1 is 0.23x, slower than not speculating at all; and whether it pays at all depends on an acceptance rate this traffic has never been measured for |
| **If it is on, draft from the prompt** | training and serving a draft head | Chapter 30 | three Medusa heads add 3.25 GB to a 16.0 GB model, 20.3% more weight read on every step, so the speedup starts 1.203x behind; drafting from the prompt adds nothing and returns 7.03x on input-grounded replies |
| **Constrain the structured endpoint, compiling each schema once at startup** | prompting for JSON and parsing defensively, or compiling per request | Chapter 31 | the mask costs 0.00010% of a decode step's traffic and the whole grammar compiles to 128 bytes, so the only real cost is the compile, which a fixed schema pays once |
| **An exact-match response cache, keyed on every field the answer depends on** | no cache above the model, or a semantic one | Chapter 32 | an exact cache answers 77% of single-turn traffic with no wrong answers, while a similarity threshold set above the worst confusable pair still answers 40% of requests wrongly on traffic that carries account numbers |
| **No thinking budget on the default endpoint** | thinking on by default, as several providers now ship it | Chapter 33 | the case study promises a first token inside 1,000 ms and a reply in seconds; a reply behind 32,768 thinking tokens takes 479 s and costs 888 times as much |
| **A dense model, and a context ceiling well short of 128K** | a mixture of experts, or an unbounded context window | Chapter 33 | DeepSeek-V3's weights alone are 1,342 GB and need 17 accelerators before anything is served, against a fleet of 8; and a 128K context admits 3 sequences a machine |

Every row is read out of the named chapter's results file when this table is generated. Nothing here was measured for this record except the table below it.

## What was considered

Two of those rows go against the obvious reading of the papers they
come from, and both are worth the paragraph.

**Speculation is not a low-batch technique; wide trees are.** The
literature reports at batch 1, where a verification pass carries
302 tokens before it costs anything and a tree of a
hundred is free. At batch 60 a pass carries
7, and the tree chosen at batch 1 is
**0.23x** — four times slower than not speculating. Size
the tree to the batch instead and the same technique is worth
3.13x at the fleet's own batch, against 4.65x at
batch 1. What the batch takes away is the breadth, not the method.

**A draft head is a weight, and weights are what a decode step is made
of.** Three Medusa heads add 3.25 GB to the model,
20.3% more, read on every step including every step whose
guesses are discarded. The speedup starts 1.203x behind before
a single token is guessed. Drafting from the prompt adds nothing at
all and returns 7.03x on replies that quote their
prompt — and 1.11x on replies that do not, which is why the
decision above it is a decision to measure rather than a decision to
enable.

## What the measurements said

### What they do to each other

This is the only thing measured for this record, because no chapter
asked it. Each of Part VI's accelerations was measured against a bare
server. A service turns several on at once.

The one pair that looked as though it might interact is the cache and
the speculation. A response cache removes requests; removing requests
lowers the batch; a lower batch is where speculative decoding is
supposed to come into its own. If that were true, the two would have
to be decided together, and the cache's value would be larger than
Chapter 32 measured.

<!-- include: tables/ddr4-interaction.md -->
| Cache hit rate | Spend it on machines | Batch | Speculation there | Or keep the fleet | Batch | Speculation there |
|---|---|---|---|---|---|---|
| 0% | 8 machines, $26.00/hr | 60 | x3.13 | 8 machines, $26.00/hr | 60 | x3.13 |
| 10% | 7 machines, $22.75/hr | 60 | x3.13 | 8 machines, $26.00/hr | 52 | x3.15 |
| 20% | 7 machines, $22.75/hr | 60 | x3.13 | 8 machines, $26.00/hr | 44 | x3.17 |
| 30% | 6 machines, $19.50/hr | 60 | x3.13 | 8 machines, $26.00/hr | 37 | x3.18 |
| 40% | 5 machines, $16.25/hr | 60 | x3.13 | 8 machines, $26.00/hr | 30 | x3.23 |
| 50% | 4 machines, $13.00/hr | 60 | x3.13 | 8 machines, $26.00/hr | 24 | x3.36 |

A cache hit can be spent two ways: shrink the fleet, or keep it and let each machine run cooler. Only the second lowers the batch, and a lower batch is where speculative decoding is supposed to come into its own. It does not: across the whole range the speedup moves by 7%. The batch sizes come from Chapter 41's own sweep and the speedups from Chapter 30's, interpolated between the loads and batches each measured and never extrapolated past them.

It is not true. Spend the measured 30% hit rate on
machines and the fleet goes from 8 to 6 with
the batch unchanged; spend it on headroom and the batch falls from
60 to 37 — and speculation goes from
3.13x to 3.36x. **Across the whole range the speedup
moves by 7%.**

So the two decisions are independent and should be made separately.
Take the cache saving in machines, which is where it is worth
something, and decide speculation on its own evidence.

The reason the interaction is absent is worth understanding, because
it is the same reason the first decision came out the way it did.
Speculation's cost at a large batch is the *width* of the verification
pass, not the fact of speculating. A narrow tree is nearly free at any
batch. Lowering the batch buys room for a wider tree, and a wider tree
is worth very little once the draft's first guess is usually right.

## What would change this

<!-- include: tables/ddr4-would-change.md -->
| Decision | What would change it |
|---|---|
| Speculative decoding behind a measurement, with the tree sized to the batch rather than to a paper | the measurement that has not been made: the share of this service's replies that quote their prompt. Chapter 30 measured that number spanning 7% to 92% across four shapes of reply, which is the difference between worthwhile and pointless |
| If it is on, draft from the prompt | traffic that is not grounded in its prompt: the same drafter returns 1.11x there, and a trained head is the only option left |
| Constrain the structured endpoint, compiling each schema once at startup | accepting arbitrary JSON Schema per request, which moves the compile onto the request path and into the first-token latency |
| An exact-match response cache, keyed on every field the answer depends on | traffic with no repeated questions, where the cache is dead weight -- and agent traffic, where it already is |
| No thinking budget on the default endpoint | a second endpoint for work that is worth minutes and is asked for asynchronously, which is what the provider documentation recommends above a 32k budget |
| A dense model, and a context ceiling well short of 128K | demand large enough to keep a mixture's fleet busy, where its cost per token is several times better -- this is a decision about scale, not about architecture |

A decision without a condition attached to it is a habit. Two of these conditions are measurements nobody has taken yet, which is the honest state of a decision record written while the service is still being built.

Two of those conditions are measurements nobody has taken. The service
does not know what share of its replies quote their prompt, which is
the number that decides whether prompt lookup is worth an afternoon —
Chapter 33's four traffic shapes spanned
7% to 92% on exactly that, and the
difference between those two is the difference between
1.11x and 7.03x. And it does not know its own
question distribution, which is what decides the cache's
77%.

Both are one query over a request log. Writing that down as a
condition, rather than guessing at the answer and recording the guess
as a decision, is what a record like this is for.

### The three that were easy

Not every decision in a record is close, and it is worth saying which
were not.

**Constraining the structured endpoint** was decided by a factor of
100% against 1.3% on documents
that parse, for a mask costing 0.00010% of a decode step's
memory traffic and a table of 128 bytes. There is no
trade-off here to reason about; there is only the compile, and a fixed
schema pays it once at startup.

**Not enabling a thinking budget** was decided by 479 s against
2.5 s for one answer, at 888x the cost, of which
0.9% is the part a reader sees. That is not a slower
endpoint, it is a different product, and it belongs behind a queue and
a different price.

**The model's shape** was decided before it was discussed: a mixture
of experts needs 1,342 GB of weights and 17
accelerators before it answers anything, against a fleet of
8. The case study is not large enough to have this problem.
Neither is most of anything, and the decision should be revisited the
moment that stops being true, because a mixture's cost per token at
scale is several times better.

## What this record does not decide

Part IV's kernels and Part V's precision are not here; Design decision
records II and III own them, and neither is written, because the
chapters they read from need hardware this book has not yet measured
on. This record assumes the model is served in bf16 on a dense
architecture, which is what Part III measured and what Part VIII
prices. If Part V's record later moves the model to int8 or fp8,
three of the rows above change with it — the fleet shrinks, the batch
grows, and speculation's tree gets narrower again.

That dependency is the argument for writing these records at all. The
decisions are not independent of each other, and the only way to know
which ones move when one of them does is to have written down what
each was decided by.

## Sources

Every measurement in this record belongs to a chapter, and each
chapter carries its own sources. The four this record reads from:

- **Chapter 29**, speculative decoding — Leviathan,
  Kalman and Matias; Chen and colleagues. The exactness result the
  whole of Part VI rests on.
- **Chapter 30**, draft heads — Cai and
  colleagues on Medusa, Li and colleagues on EAGLE, Saxena on prompt
  lookup. The head sizes priced above are from the first project's own
  implementation and the second project's published parameter counts.
- **Chapter 31**, constrained decoding — Willard and Louf; Dong and
  colleagues. The mask table measured above is this book's own, built
  to ECMA-404.
- **Chapter 32**, caching above the model — vLLM's cache isolation
  documentation, Zilliz's GPTCache, and two papers on negation
  blindness in sentence embeddings, which are why the semantic cache
  is not in the table above.
- **Chapter 33**, long context, reasoning and MoE —
  DeepSeek-AI's technical report for the mixture priced above, and
  Anthropic's extended-thinking documentation for the thinking budget
  that is not enabled.
