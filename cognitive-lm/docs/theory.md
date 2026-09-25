# Theory sketch: separate knowing THAT from knowing WHAT

Sections 1 to 4 are the round-1 argument (complementary memory). Section 5
is the round-2 architecture this repo settled on: checksum-aided recall.

This is a working note, not a finished proof. It states the argument the POC
tests, so that every claim here can be checked against a number in
`RESULTS.md`.

## Setup

- A universe of arbitrary facts f = (entity, relation) → value, with V
  possible values per relation and no pattern linking facts.
- Training data: M mentions drawn i.i.d. from a popularity distribution p
  over facts. n_f is how often fact f was mentioned. N_n is the number of
  distinct facts mentioned exactly n times.
- Queries come from the same p. The *missing mass* MM = Σ_{f: n_f = 0} p_f is
  the share of queries about facts never seen. Good-Turing estimates it as
  N_1 / M.

## 1. The floor only binds systems that cannot abstain correctly

A system that always answers is wrong on unseen facts at least (1 − 1/V) of
the time, so hallucination ≥ MM (1 − 1/V) ≈ N_1 / M. This is the
Kalai-Vempala floor in its simplest QA form.

Now let a system abstain whenever a membership test says "never seen". Say
the test has false-positive rate ε on unseen facts and false-negative rate
δ on seen ones, and recall on the facts it answers has error r. Then

    hallucination = MM · ε · (1 − 1/V)  +  (1 − MM) · (1 − δ) · r
    accuracy      = (1 − MM) · (1 − δ) · (1 − r)

So **below the floor is purely a question of the membership test**. The
first term is the part the Kalai-Vempala argument talks about. The second
is ordinary recall error.

## 2. Weights are an expensive membership test

- Any structure answering "is f in S?" with false-positive rate ε needs at
  least |S| · log2(1/ε) bits (Carter et al. 1978). A Bloom filter reaches
  1.44 · |S| · log2(1/ε). For ε = 1% that is about 9.6 bits per fact, with
  δ = 0 exactly.
- arXiv 2602.00906 shows that a parametric model trained with log-loss
  under a memory budget settles into a "hallucination channel". A fraction
  q* of non-facts gets the same score as real facts, so no confidence
  threshold separates them. In our language, ε stays high whatever δ you
  are willing to pay.
- Prediction P1: the model's confidence separates facts seen once from
  unseen facts barely better than chance (AUROC near 0.5). It improves with
  exposure count and with model size. An IDK-trained model does not fix
  this, because the problem is capacity, not the objective.

## 3. Taking membership out of the weights

Split the job the way succinct data structures do. A Bloomier filter is a
membership filter plus a static function (Chazelle et al. 2004).

- **Familiarity** (knowing THAT): Bloom filters over entity keys and fact
  keys, written in one shot while reading. The cost is ~10 bits per key,
  independent of model size.
- **Recollection** (knowing WHAT): the weights, plus an exact episodic
  store. Recollection only has to be right on queries that familiarity
  admits. A static function on |S| keys needs only |S| · log2 V bits and no
  membership overhead.

Prediction P2: at matched total bits, familiarity plus recollection
dominates weights-only models on the accuracy-versus-hallucination frontier.
Hallucination drops to about MM · ε plus recall error, with no loss of
coverage.

Prediction P3: the entity filter gives *graded* abstention for free.
"Never heard of this person" (entity not familiar) differs from "I know
them, but not this" (entity familiar, fact not familiar).

## 4. Which facts should live in the weights? (consolidation at k)

Queries follow p, so consolidation should follow query traffic. By
Good-Turing, the total probability of the facts seen exactly n times is

    θ_n ≈ (n + 1) · N_{n+1} / M.

If facts seen ≥ k times are consolidated into the weights and the rest stay
in the store, the share of queries that still need the store is

    store load(k) ≈ Σ_{n=1}^{k−1} θ_n   (+ facts the core failed to absorb)

and the number of facts the core must hold is Σ_{n≥k} N_n.

Under Zipf-like popularity, singletons are most of the *distinct* facts but
a small share of *queries*. So a small k moves most traffic to the weights
while most facts stay in the store. Pick k from the count histogram alone:
the smallest k whose consolidated set fits the core's capacity
(~ log2 V / (bits per parameter) parameters per fact).

Prediction P4: the measured store load matches the Good-Turing estimate
above. Prediction P5: removing low-count facts from the core's training
(k ≥ 2) does not hurt, and may improve, its recall of frequent facts, since
it stops wasting capacity on facts it could not retain (consistent with
arXiv 2604.08519 and Physics of LMs 3.3).

## What is not claimed

- Nothing here beats the floor for a system that must answer everything. That
  is impossible for arbitrary facts.
- For arbitrary facts an exact store is cheaper per bit than weights, so the
  case for consolidating into weights is not bits. It is retrieval load,
  latency and composition (using the fact inside reasoning), which this POC
  measures only through store load.
- Keys here are exact (entity, relation) ids. Real text needs an extractor
  (LMLM's pipeline is one) or learned keys, and those bring their own
  errors. That is the first thing to test next.

## 5. Checksum-aided recall as list decoding

Round 1 stored the *values* of rare facts outside the weights. Round 2
stores no values at all. Keep a Bloom filter over every (entity, relation,
value) triple read in training (the *checksum*, false-positive rate ε_t,
about 1.44 log2(1/ε_t) bits per fact), plus the key filter over (entity,
relation) with rate ε_k. To answer a query:

1. If the key filter says the fact was never read, abstain.
2. Otherwise take the weights' top-N candidate values, in order of
   probability.
3. Answer the first candidate whose triple passes the checksum. If none
   passes, abstain ("tip of the tongue").

This is CRC-aided list decoding (CA-SCL, as used for 5G polar codes). The
weights are the list decoder, the checksum is the CRC, and "no candidate
passes" is the decoder's failure flag. In cognitive terms it is the
generate-recognize model of recall.

Let R be the rank of the true value in the weights' distribution, and MM
the never-seen share of queries. Treating Bloom false positives as
independent:

    accuracy      ≈ (1 − MM) · E[ 1{R ≤ N} · (1 − ε_t)^(R − 1) ]
    hallucination ≈ MM · ε_k · (1 − (1 − ε_t)^N)
                  + (1 − MM) · E[ 1{R ≤ N} (1 − (1 − ε_t)^(R − 1)) + 1{R > N} (1 − (1 − ε_t)^N) ]
                  ≈ ε_t · ( MM · ε_k · N  +  (1 − MM) · E[min(R − 1, N)] )

What this says:

- **Hallucination on never-seen facts is ε_k · N · ε_t**: the product of
  two small numbers. It no longer depends on the singleton rate, the
  model size or the training objective. This is the point where the
  Kalai-Vempala floor stops binding, because the system abstains using an
  exact-by-construction familiarity signal.
- **The checksum size the weights need** to keep hallucination under δ is
  about b ≈ 1.44 · log2( E[min(R − 1, N)] / δ ) bits per fact. Better
  weights (smaller R) need a smaller checksum. Without weights (checking
  all V values) it is about 1.44 · log2(V / δ). The weights "pay" for
  log2(V / E[R]) of those bits. This is the rate split between list
  decoder and CRC.
- **N trades accuracy for hallucination.** Accuracy rises with
  P(R ≤ N), and hallucination rises about linearly in N · ε_t. A prefix
  checksum (Bloom over (key, first sub-token)) prunes wrong branches
  before the full check, so a large N costs far fewer false positives.
- **The weights only need the answer somewhere in their top N**, not at
  rank 1, which Retrieval-Constrained Decoding (arXiv 2509.23417) shows is
  often already true of real LLMs.

`experiments/run_care.py` computes these predictions from the measured
ranks and the measured ε_t and ε_k, and `RESULTS.md` §3 puts them next to
the measured numbers.

## 6. Why two memories that must agree are robust to extraction noise

When a fact extractor files a mention under the wrong entity, the checksum
learns a false triple (e′, r, v). A plain store then asserts v whenever
(e′, r) is asked. CAR asserts it only if the weights *also* rank v in their
top N for (e′, r), and the weights learned v from the true entity, not
e′. So the error passes only when two independently trained memories
agree on it. The cost is that CAR recovers fewer facts than a store. CARE's
overflow store brings that accuracy back, and with it some of the store's
exposure to noise. "Strict" CARE refuses to assert overflow facts read only
once, which is where mislinks concentrate.
