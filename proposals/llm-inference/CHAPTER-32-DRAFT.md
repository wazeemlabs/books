# 32. Caching Above the Model

*Written to [STANDARDS.md](STANDARDS.md). Generated from
`chapters/ch32.md` and `code/results/ch32.json`; run `make ch32` in
`code/` to re-derive and re-render.*

**Depends on:** Chapter 15, Chapter 1,
Chapter 5.
**Tier 0** — a few minutes on a laptop CPU, free.

## Objectives

By the end of this chapter you can:

1. Say what a cache in front of the model saves that
   Chapter 15's cache does not, and how much more a hit is
   worth.
2. Predict a cache's hit rate from two properties of your traffic,
   neither of which is a property of the cache.
3. Name every field that belongs in a cache key, and say what each one
   costs when it is left out.
4. Decide whether a similarity threshold can be made safe, and do the
   arithmetic that settles it.

## Why it matters

Every other saving in this book makes the model cheaper to run. This
one skips it.

A request whose answer is already known costs a dictionary lookup: no
prefill, no decode, no accelerator, no queue. Nothing else in this
book comes close, and nothing else is as easy to build.

<!-- defines: response cache, cache key -->

That is a **response cache**: whole answers, stored under the request
they answer. What it stores them under is its **cache key**, and the
key is where the whole chapter ends up, because this is also the only
layer in the book that can change the answer. Chapter 15's
cache re-used *arithmetic*, and Chapter 15 checked to the
last bit that the arithmetic was the same. This re-uses a conclusion.
A conclusion is only re-usable while nothing it depended on has
changed, and a cache that does not know what it depended on will
happily serve it anyway.

## What a hit is worth

Start with why anyone bothers. One accelerator running
Chapter 18's scheduler, the case study's traffic, and a
capacity read off the highest offered load that keeps both of the
service's promises. Three ways of putting a cache in front of it.

<!-- include: tables/ch32-worth.md -->
| In front of the model | Requests a second a machine | Machines for 200 a second | Dollars an hour | First token p99 at capacity |
|---|---|---|---|---|
| no cache | 26 | 8 | $26.00 | 469 ms |
| prefix cache | 40 | 5 | $16.25 | 172 ms |
| response cache | 36 | 6 | $19.50 | 302 ms |

The machine of Chapter 41, measured the same way: every offered load run at 6,000 requests and again at twice that, and the capacity is the highest load whose numbers settled and which kept both promises. The prefix cache is Chapter 15's measured hit -- 85.4% of prompt tokens already computed -- and saves only the reading of them. The response cache hits 30% of *requests* and saves everything about them. Memory is not what binds here: the bare machine peaked at 6,755 of 30,515 blocks.

The bare machine sustains **26 requests a second**, so the
case study's 200 a second needs 8 of them at
$26.00 an hour. A response cache that catches
**30% of requests** lifts that to 36 a
second a machine — 6 machines, $19.50 —
because a request that never reaches the model cannot cost anything.

The comparison worth making is the one against
Chapter 15's cache, sitting on the same machine. That one
hit **85.4% of prompt tokens**, an enormous rate, and it is
worth +54% capacity. The response cache hits
30% of *requests* and is worth +38%. Two
different units, and the exchange rate between them is the number to
carry away: a response cache would have to catch
**35% of all requests** to be worth what
that 85.4% prompt-token hit rate is worth here.

That is the real case for caching above the model. A prefix hit
removes the reading of a prompt. A response hit removes the prompt,
the reply, the slot on the server and the memory the sequence would
have held.

The second thing a cache buys does not show up as machines at all.
Keep the 8 machines and leave the traffic alone, and let
the cache absorb 30% of it: each machine now sees
about 18 requests a second instead of 26. At
the nearest load actually measured, 20 a second, the
99th-percentile wait for a first token is 141 ms against
469 ms at 26. Tail latency near capacity is
not linear in load, so a cache that removes a third of the traffic
does much more than a third of good.

## What decides the hit rate

None of that matters if the cache does not hit, and whether it hits is
not a property of the cache.

![Hit rate against how many questions there are and how concentrated the asking is](code/figures/ch32-concentration.svg)

**Figure 32.1** — The two numbers that set a cache's hit rate, neither
of them the cache's. *Provenance in
`code/figures/ch32-concentration.caption.txt`.*

Two things decide it. **How many distinct things people ask**, and
**how concentrated the asking is**. At a fixed skew, a catalogue of
100 questions is answered 96% of the time
and one of 10,000 is answered 69% of the time.
At a fixed catalogue of 2,000, flattening the skew from
1.5 to 0.7 takes it from 91% to
59%. Nothing about the cache appears on that chart.

Both numbers come out of a request log with one query each, and they
are worth more than any amount of cache tuning. At the operating point
used for the rest of this
chapter — 2,000 distinct questions, Zipf skew 1.1 — the
hundred most popular carry 72% of all asking, which is
the other half of the news: **a cache does not have to be big.**

The shape of the traffic matters as much as its content.

<!-- include: tables/ch32-shapes.md -->
| Traffic | What the key folds | Distinct keys | Hit rate | Served wrong |
|---|---|---|---|---|
| FAQ | as typed | 5,123 | 74.4% | none |
| FAQ | case and punctuation | 4,518 | 77.4% | none |
| FAQ | and stop words dropped | 4,506 | 77.5% | **3.77%** |
| chat | as typed | 10,387 | 34.0% | none |
| chat | case and punctuation | 10,191 | 35.2% | none |
| chat | and stop words dropped | 10,179 | 35.3% | **1.53%** |
| agent | as typed | 16,339 | 9.2% | none |
| agent | case and punctuation | 16,247 | 9.7% | none |
| agent | and stop words dropped | 16,236 | 9.8% | **0.69%** |

One day of traffic over a catalogue of 2,000 distinct questions asked with a Zipf skew of 1.1, seed 0: 20,000 single-turn requests, 6,000 conversations averaging 2.6 turns, and 3,000 agent tasks of 6 steps each. Only 17% of the agent requests are the first step of a task; every later one carries a transcript no other session has produced, so it cannot repeat. Folding case and punctuation is free. Dropping function words is not: it makes "is this covered" and "is this not covered" the same key.

Three shapes over the same catalogue, from 77.4% down to
9.7%:

- **Single-turn questions** hit 77.4%. This is the case caching
  was invented for and it behaves.
- **Conversations** hit 35.2%. Only the first turn of a session
  is a question anyone else could have asked; every turn after it
  carries the conversation so far, and no two conversations are the
  same.
- **Agent traffic** hits 9.7%, and cannot do much better.
  Each task is 6 steps, of which one is the task itself
  and the rest carry a transcript of tool output that no other session
  has ever produced. Only 17% of the requests are
  even *candidates* for a repeat. **A response cache has no purchase
  on an agent loop**, which is exactly the traffic
  Chapter 15's cache handles best, because a growing
  transcript is a growing shared prefix.

One of the two folding steps in that table is free. Folding case and
punctuation takes the single-turn hit rate from 74.4% to
77.4% and serves no wrong answers, because two questions that
differ only in a capital letter have the same answer.

The other step is not free, and it is worth dwelling on because it is
standard practice borrowed from text search. Dropping function words
before indexing adds 0.1% of hits and
**3.8% of all requests are then answered wrongly**. The
reason is on the list itself. *Not*, *no*, *without*, *under*,
*over*, *before* and *after* are function words, and they are also the
entire difference between two questions with opposite answers:

```
is this covered by the warranty?      ->  covered warranty
is this not covered by the warranty?  ->  covered warranty

what is the refund policy for orders over $100?   ->  refund policy orders 100
what is the refund policy for orders under $100?  ->  refund policy orders 100
```

A cache that folds those together does not return a doubtful answer.
It returns the wrong one, with total confidence, every time.

## The key is the whole problem

That is the general case, and it is the one thing to take from this
chapter. Everything a cache gets wrong, it gets wrong because its key
left out something the answer depended on.

<!-- include: tables/ch32-keys.md -->
| Traffic | What the key holds | Hit rate | Served wrong | Of the hits |
|---|---|---|---|---|
| FAQ | everything the answer depends on | 77.4% | none | -- |
| FAQ | no tenant | 85.1% | **20.4%** | 24% |
| FAQ | no conversation | 77.4% | none | -- |
| FAQ | the question alone | 85.1% | **20.4%** | 24% |
| chat | everything the answer depends on | 35.2% | none | -- |
| chat | no tenant | 42.6% | **9.6%** | 23% |
| chat | no conversation | 86.7% | **56.7%** | 65% |
| chat | the question alone | 90.2% | **61.6%** | 68% |
| agent | everything the answer depends on | 9.7% | none | -- |
| agent | no tenant | 11.3% | **2.7%** | 24% |
| agent | no conversation | 9.7% | none | -- |
| agent | the question alone | 11.3% | **2.7%** | 24% |

The same traffic, keyed four ways, with case and punctuation folded in all of them. 3 customers share the service (60%, 30%, 10% of the traffic); the answer to a policy question depends on which of them asked. Every shorter key raises the hit rate, which is what makes it tempting. "Served wrong" is the share of all requests answered with something other than what the model would have said.

![Wrong answers against hits, for four cache keys on three kinds of traffic](code/figures/ch32-keys.svg)

**Figure 32.2** — Every field left out of the key buys hits and sells
correctness. *Provenance in `code/figures/ch32-keys.caption.txt`.*

Read the table one row at a time and the pattern is the same each
time: **the shorter key always hits more.** That is what makes this
the bug it is. Nobody removes a field from a cache key and watches the
hit rate fall; they remove it, watch the hit rate rise, and ship.

Two omissions are worth naming because both are common.

**The tenant.** Two customers ask the same words and have different
answers — their leave policy, their refund threshold, their rate.
Keyed on the question alone, the single-turn traffic hits
85.1% instead of 77.4%, and
20.4% of all requests are answered with another
customer's policy. This is not a quality problem. It is one customer
reading another customer's data, and it will be found by a customer
rather than by a dashboard.

**The conversation.** A follow-up turn is a question that means
nothing on its own: *are you sure?*, *and for the other one?*, *why?*.
Keyed on the last message alone, the conversational traffic hits
86.7% instead of 35.2%, which would look
like the best change anyone made that quarter, and
56.7% of all requests are wrong,
which is 65% of everything the cache served. The
cache found thousands of people asking *are you sure?* and gave them
all the same answer.

State the rule as a rule, because it generalises past caches:

> **A cache key is an input, and everything the answer depends on
> belongs in it.** The tenant, the system prompt, the tool set, the
> model version, the decoding parameters, the conversation, the user's
> locale, the date if the answer moves. Anything the model was told
> and the key was not is a way for one caller to be served another
> caller's answer.

The production engines treat it exactly this way. vLLM's prefix cache
— which caches arithmetic, not answers, and so is much harder to get
wrong — still ships an explicit escape hatch for this: a per-request
`cache_salt` "injected into the hash of the first block", so that
"only requests with the same salt can reuse cached KV blocks", which
limits sharing "to users or requests that explicitly agree on a common
salt". Its documentation files that under *cache isolation for
security*, and notes that a weaker hash "can cause undefined behavior
or even leak private information in multi-tenant environments". If the
cache that only reuses keys and values needs a tenant in its key, the
cache that reuses whole answers certainly does.

## Matching on meaning

<!-- defines: semantic cache -->

Everything so far matched on equality. The obvious next move is to
match on similarity instead: embed the question, find the nearest
entry, and serve it if the two are close enough. That is a **semantic
cache**, and it is what the tooling in this area mostly sells.
GPTCache, the best known of them, offers to "slash your LLM API costs
by 10x" and is honest about the price in the same README: "in a
semantic cache, you may encounter false positives during cache hits
and false negatives during cache misses."

The change sounds small and is not. Equality has no false positives.
A threshold does, which turns a lookup into a classifier, and a
classifier has an error rate whether or not anybody measures it.

Whether the error rate can be made small is a question about the
questions, not about the embedder. Here is the whole of this book's
labelled set — 88 ways of asking 22 things, where
two questions share an answer exactly when they share a topic —
scored against itself.

![Every pair of questions, scored; and what a full cache does to any threshold](code/figures/ch32-similarity.svg)

**Figure 32.3** — Left: the two distributions overlap. Right: whatever
threshold you pick, the cache multiplies it. *Provenance in
`code/figures/ch32-similarity.caption.txt`.*

The most similar pair in the set that must **not** share an answer
scores 0.77:

```
what is the refund policy for orders over 100 dollars
what is the refund policy for orders under 100 dollars
```

Call that the ceiling: the threshold has to sit above it or the cache
will answer one of those with the other. And 86% of the
pairs that *must* share an answer score below the ceiling.

Look at what is left above it. 12.1% of the true pairs
score 1.00 — they differ only in capitals and punctuation, and folding
those away, which costs nothing and risks nothing, already merges
them. Which leaves **1.52%**: the share of true pairs a
similarity threshold can safely reach that a dictionary does not
already give you for free.

Two objections to that, and they deserve straight answers.

**"That is a lexical measure, not a semantic one."** It is —
character 5-gram overlap, stated as such wherever this chapter uses
it. A sentence embedder would score the rewordings far higher and
would move the upper row of that chart to the right. What it would not
move is the ceiling, and it is the ceiling that decides the question.
Two sentences differing in one word are close under any measure
continuous in its input, and the specific words involved here are the
ones embedding models are documented to be worst at. Chiang and
colleagues built a diagnostic set out of exactly these minimal pairs —
"replacing a word with a synonym, an antonym, a typo, a random word,
and converting the original sentence into its negation" — and found
"most unsupervised sentence encoders are insensitive to negation".
Cao's 2025 study of current universal embeddings reports "a
significant lack of negation awareness in these models, often
interpreting negated text pairs as semantically similar". A better
embedder raises the floor; the ceiling is a property of the questions.

**"Fine, what does it do on traffic?"** Run it, on a stream made only
of these questions. Exact matching after folding answers
98.9% of it. Set the threshold at 0.80, just
above the ceiling, and the semantic cache answers 99.0% —
**0.03% more**, for 0.00% wrong. Safe, and worth
almost nothing, which is what 1.52% looks like in
practice.

Now do what anyone tuning this would do next, and take the threshold
one step down to buy more hits. At 0.70 the
extra hits go from
0.03% to 0.09% of requests, and the wrong
answers go from 0.00% to **3.77% of all
requests**: about **63 wrong answers for every
extra hit that step buys.** That is the trade the threshold controls,
and nobody would take it if the second number were on the dashboard
beside the first.

## What it will actually get wrong

The stream above was the clean case: every confusable pair in it
differs by a decisive word, which is the failure the literature is
about. Real traffic has a worse one.

Run the same threshold of 0.80 over the full catalogue —
2,000 questions, most of them about a particular account,
order or booking — and it answers 97.8% of requests, adds
20.4% over exact matching, and gets 40.3% of all
requests wrong: **2 wrong answers for every extra
hit.** The same threshold that was safe a paragraph ago.

Not one of those false matches was a clever failure of meaning.
100% of them were pairs like

```
What is the process for seat 33?
What is the process for seat 35?
```

which score 0.93. Two questions about different things,
near-identical as strings. Any similarity measure puts a pair like
that near the top of its range, and a service that answers questions
about orders and accounts is mostly made of pairs like that. The
negation failures the literature documents are real; an account number
is what you will meet first.

## The arithmetic a threshold cannot escape

Suppose all of that is solved. Suppose an embedder better than any
that exists: one that matches every paraphrase and confuses two
unrelated questions only one in 1,000,000 pairs. Two numbers still decide
whether the cache is safe, and neither is the threshold.

**The cache multiplies the error rate.** A false-positive rate is
quoted per pair. A lookup is not one pair: it compares the query
against every entry and takes the best. With a per-pair rate of
one in 1,000,000, a cache of 1,000 entries has a 0.1% chance of a
false match on any given lookup; at 100,000 entries it is
9.5%, and at a million 63.2%. The right panel of
Figure 32.3 is that arithmetic, and its shape is the problem: **a
semantic cache gets less safe exactly as it gets more useful.**

<!-- defines: base rate -->

**The base rate decides the rest.** Of the requests arriving at this
chapter's stream, 8.3% have an equivalent already cached that
exact matching missed. That is the entire population a similarity
threshold can help. The other 91.7% are questions
with no twin in the cache at all, and every one of them is exposed to
the false-match rate above. A test with a small error rate applied to
a large population produces more errors than a test with a large error
rate applied to a small one, which is Bayes' rule and is the whole of
the argument.

Multiply the two and the conclusion is not subtle. If you keep a
semantic cache, keep it **small**, keep it **partitioned** so a query
is only ever compared against entries that could legitimately answer
it, and **measure the wrong-answer rate** rather than the hit rate,
because they move in the same direction and only one of them is on
anybody's dashboard.

## How long an answer stays true

<!-- defines: time to live -->

One more thing the key cannot hold: when the question was asked. An
answer correct on Monday can be wrong on Tuesday because the policy
behind it changed, and no amount of care in the key catches that. The
usual answer is a **time to live** — throw an entry away after a while
whether or not anything changed.

![Hits and stale answers against how long an entry lives](code/figures/ch32-ttl.svg)

**Figure 32.4** — Both curves rise, at different rates. *Provenance
in `code/figures/ch32-ttl.caption.txt`.*

<!-- defines: false hit -->

Both curves rise with the lifetime, at different rates, and the shape
of the difference is what a setting is chosen from. Up to
1 h the cache answers 48% of requests for
0.12% stale — a real saving for nothing much. Past that
the stale curve turns up: at 1 d another 30
points of hit rate cost 55 times the staleness,
6.8% of all requests, which is
8.8% of everything the cache served. A stale
answer is a **false hit** like any other: not a slower answer or a
doubtful one, a wrong one, and the caller cannot tell.

Those numbers are what they are because a topic's answer here moves
0.5 times a day. The shape holds for any rate of
change; the knee moves with it, which is why the setting has to come
from how often *your* answers move rather than from a default.

A lifetime is the crude fix. The good fix is to make the thing that
changes the answer change the key — put a version, a policy
identifier, a document revision in it — so that an update invalidates
exactly the entries it should and nothing else. That is more work than
a TTL and it is the difference between a cache you can trust and a
cache you tolerate.

## Where this chapter simplifies

**The traffic is simulated and the questions are authored.** There is
no public trace that carries the label this chapter needs — which
questions must get the same answer — so 22 topics were written
with that label attached, and the rest of the catalogue is templated
filler whose job is to give the distribution a tail of a realistic
length. Every mechanism here is a property of caches and of how people
write questions. None of the *rates* is a property of your service.
Measure your own catalogue size, skew and confusable fraction; the
chapter is a method, not a set of constants.

**The synthetic tail is templated**, so the rate at which an
identifier collides is specific to it. That identifier collisions
happen at all, and score near the top of any similarity measure, is
not.

**The cache is free here, and a real one is not.** An exact-match
cache costs a hash and a lookup, which is close enough to free. A
semantic cache costs an embedding of every arriving question,
including every miss — a model call this chapter never charged for,
and one that has to be subtracted from the saving before a semantic
cache pays for itself at all. That arithmetic is not in the numbers
above and it moves them the wrong way.

**A right answer is assumed to exist.** Every measurement here
compares the cached answer against what the model would have said,
which is well defined only at temperature zero. Above it there is no
single right answer, "wrong hit" needs a looser definition than
inequality, and measuring the rate gets considerably harder — which
is the subject of the last exercise rather than of this chapter.

**The prefix-cache row models compute and not memory.** A shared
prefix saves the reading of a prompt *and* the pool space its blocks
would have taken, and only the first of those is in the capacity
figures. The pool was measured and was nowhere near binding on this
machine, so it changes nothing here. On a machine where memory binds,
the prefix cache is worth more than the table says.

**Nearest-neighbour search is exact here.** Production vector indexes
are approximate, and the entry they return need not be the nearest
one. Every false-match number above is the optimistic case.

## In production

**Build the exact-match cache first, and possibly only that.** It is a
dictionary. It has no false positives, it is trivial to reason about,
and on single-turn traffic it does 77.4% of the work. Measure
your catalogue size and your skew before considering anything cleverer.

**Put the cache key under review the way you would a schema.** Adding
a field to what the model is told — a new system prompt, a tool, a
tenant setting, a model version — is a change to the cache key, and
the failure of forgetting it is silent, arrives as a wrong answer, and
is found by a customer. A key built by listing every input the request
carries, rather than by listing the ones that seemed to matter, is the
only version of this that survives a year of changes.

**Do not cache what the model is not asked to be deterministic
about.** A cached answer is the same every time; a model at
temperature above zero is not. Caching a sampled reply freezes one
draw from the distribution and serves it forever, which changes what
your evaluations and your A/B tests are measuring even when no answer
is wrong.

**If you use a semantic cache, partition it and keep it small.** One
tenant, one system prompt, one document set per partition, so the
nearest-neighbour search cannot reach an entry that could not
legitimately answer the question. The false-match rate scales with
how many entries a lookup can reach, and partitioning is the only
lever that reduces that without touching the threshold.

**Log the hit rate and the wrong-answer rate together, or neither.**
The hit rate is easy and flattering. The wrong-answer rate needs a
sample of hits re-run against the model and compared, which costs a
little of what the cache saved, and it is the only number that tells
you whether the cache is working. A cache reporting 90% hits and
nothing else is reporting one half of a fraction.

**Remember what a vector index is.** Production similarity search is
approximate: the entry it returns may not be the nearest one. Every
number in this chapter assumed an exact scan, which is the optimistic
case.

## Numbers to remember

- **26 → 36 requests a second a machine** —
  what a 30% response-cache hit rate is worth, against
  40 for Chapter 15's 85.4% of prompt
  tokens. A response cache has to catch
  35% of requests to match it.
- **96% against 69%** — exact-match hit rate over
  100 distinct questions and over
  10,000. The hit rate is a property of the traffic.
- **17%** — the share of agent requests that are
  even candidates for a repeat. Everything else carries a transcript
  nobody else has produced.
- **56.7% of all requests** — what leaving the
  conversation out of the key costs on conversational traffic, for a
  hit rate that rises from 35.2% to 86.7%.
  Every shorter key hits more.
- **0.77 against 0.01** — the most similar pair
  that must *not* share an answer, against the median of the pairs
  that must. 86% of the true pairs fall below the false
  one.
- **0.1% → 9.5%** — the chance of at least one false
  match per lookup, at one in 1,000,000 per pair, as a cache grows from a
  thousand entries to a hundred thousand.
- **0.12% against 6.8%** — stale answers
  at a 1 h lifetime and at a 1 d one, when a topic's
  answer moves 0.5 times a day. Staleness is cheap
  until it is not.

## Sources

- vLLM documentation, *Automatic Prefix Caching* — the production
  treatment of cache isolation: "vLLM supports isolating prefix cache
  reuse through optional per-request salting", so that "only requests
  with the same salt can reuse cached KV blocks", filed under cache
  isolation for security. It also warns that a non-cryptographic hash
  "theoretically increases the risk of hash collisions, which can
  cause undefined behavior or even leak private information in
  multi-tenant environments".
- Zilliz, *GPTCache* — the reference open-source semantic cache, and a
  usefully honest README: "in a semantic cache, you may encounter
  false positives during cache hits and false negatives during cache
  misses."
- Cheng-Han Chiang, Yung-Sung Chuang, James Glass, Hung-yi Lee,
  "Revealing the Blind Spot of Sentence Encoder Evaluation by HEROS",
  arXiv:2306.05083 — minimal pairs built by "replacing a word with a
  synonym, an antonym, a typo, a random word, and converting the
  original sentence into its negation"; finds "most unsupervised
  sentence encoders are insensitive to negation".
- Hongliu Cao, "Semantic Adapter for Universal Text Embeddings:
  Diagnosing and Mitigating Negation Blindness to Enhance
  Universality", arXiv:2504.00584 (ECAI 2025) — "a significant lack of
  negation awareness in these models, often interpreting negated text
  pairs as semantically similar".

## Exercises

★ Your service's logs contain one line per request with the prompt
hashed. Write the two queries that tell you whether a response cache
is worth building, and say what you would conclude from each answer.

★ A colleague proposes caching on the last user message only, and
points out that the hit rate triples. Write the two-sentence reply.

★★ The chapter's exchange rate says a response cache needs
35% of requests to be worth what
Chapter 15's cache is worth here. Work out how that number
moves if replies get four times longer, and say which cache you would
build for a service that answers in a sentence.

★★ Take the stop-word list in `tinyserve/cache.py` and produce three
more pairs of questions it merges that have opposite answers. Then
say what normalization you would keep and what you would drop.

★★★ Design the wrong-answer measurement for a live cache: which hits
you re-run against the model, how many, how you compare two free-text
answers, and what you do when the sample rate the budget allows is too
low to detect the failure rate you care about.
