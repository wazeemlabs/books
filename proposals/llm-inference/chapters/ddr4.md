# Design decision record IV

### The case study's decoding accelerations

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ddr4.md` and `code/results/ddr4.json`; run `make ddr4` in
`code/` to re-derive and re-render.*

**Closes Part VI.** Every number in this record was measured in
Chapters {{first_chapter}} to {{last_chapter}} and is read back out of
those chapters' results files. One thing is measured here and nowhere
else — what the accelerations do to each other — and it is marked
where it appears.

---

Part VI offered five ways to make decoding faster and one way to make
it slower. Design decision record I committed the case study to a
scheduler; this commits it to which of those five it turns on.

The traffic has not changed: {{rate}} requests a second,
{{ch:capacity-planning}}'s fleet of {{fleet}} machines at
{{capacity}} requests a second each, {{gpu_hour}} an accelerator-hour.

## The decision

There are {{decisions}} of them, taken from {{chapters}} chapters of
measurement.

<!-- include: tables/ddr4-decisions.md -->

## What was considered

Two of those rows go against the obvious reading of the papers they
come from, and both are worth the paragraph.

**Speculation is not a low-batch technique; wide trees are.** The
literature reports at batch 1, where a verification pass carries
{{free_at_one}} tokens before it costs anything and a tree of a
hundred is free. At batch {{batch_here}} a pass carries
{{free_at_many}}, and the tree chosen at batch 1 is
**{{spec_wrong_tree}}** — four times slower than not speculating. Size
the tree to the batch instead and the same technique is worth
{{spec_here}} at the fleet's own batch, against {{spec_at_one}} at
batch 1. What the batch takes away is the breadth, not the method.

**A draft head is a weight, and weights are what a decode step is made
of.** Three Medusa heads add {{head_gb}} to the model,
{{head_share}} more, read on every step including every step whose
guesses are discarded. The speedup starts {{head_step}} behind before
a single token is guessed. Drafting from the prompt adds nothing at
all and returns {{lookup_grounded}} on replies that quote their
prompt — and {{lookup_prose}} on replies that do not, which is why the
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
{{ch:caching-above-the-model}} measured.

<!-- include: tables/ddr4-interaction.md -->

It is not true. Spend the measured {{hits_measured}} hit rate on
machines and the fleet goes from {{fleet}} to {{shrunk_fleet}} with
the batch unchanged; spend it on headroom and the batch falls from
{{batch_here}} to {{kept_batch}} — and speculation goes from
{{spec_here}} to {{best_spec}}. **Across the whole range the speedup
moves by {{spec_range}}.**

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

Two of those conditions are measurements nobody has taken. The service
does not know what share of its replies quote their prompt, which is
the number that decides whether prompt lookup is worth an afternoon —
Chapter {{last_chapter}}'s four traffic shapes spanned
{{draft_prose}} to {{draft_grounded}} on exactly that, and the
difference between those two is the difference between
{{lookup_prose}} and {{lookup_grounded}}. And it does not know its own
question distribution, which is what decides the cache's
{{cache_hit}}.

Both are one query over a request log. Writing that down as a
condition, rather than guessing at the answer and recording the guess
as a decision, is what a record like this is for.

### The three that were easy

Not every decision in a record is close, and it is worth saying which
were not.

**Constraining the structured endpoint** was decided by a factor of
{{valid_constrained}} against {{valid_unconstrained}} on documents
that parse, for a mask costing {{mask_share}} of a decode step's
memory traffic and a table of {{table_bytes}} bytes. There is no
trade-off here to reason about; there is only the compile, and a fixed
schema pays it once at startup.

**Not enabling a thinking budget** was decided by {{deep_s}} against
{{answer_s}} for one answer, at {{deep_cost}} the cost, of which
{{deep_visible}} is the part a reader sees. That is not a slower
endpoint, it is a different product, and it belongs behind a queue and
a different price.

**The model's shape** was decided before it was discussed: a mixture
of experts needs {{moe_gb}} of weights and {{moe_machines}}
accelerators before it answers anything, against a fleet of
{{fleet}}. The case study is not large enough to have this problem.
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

- **{{ch:speculative-decoding}}**, speculative decoding — Leviathan,
  Kalman and Matias; Chen and colleagues. The exactness result the
  whole of Part VI rests on.
- **{{ch:self-speculation-and-draft-heads}}**, draft heads — Cai and
  colleagues on Medusa, Li and colleagues on EAGLE, Saxena on prompt
  lookup. The head sizes priced above are from the first project's own
  implementation and the second project's published parameter counts.
- **{{ch:constrained-decoding}}**, constrained decoding — Willard and Louf; Dong and
  colleagues. The mask table measured above is this book's own, built
  to ECMA-404.
- **{{ch:caching-above-the-model}}**, caching above the model — vLLM's cache isolation
  documentation, Zilliz's GPTCache, and two papers on negation
  blindness in sentence embeddings, which are why the semantic cache
  is not in the table above.
- **{{ch:long-context-reasoning-and-moe}}**, long context, reasoning and MoE —
  DeepSeek-AI's technical report for the mixture priced above, and
  Anthropic's extended-thinking documentation for the thinking budget
  that is not enabled.
