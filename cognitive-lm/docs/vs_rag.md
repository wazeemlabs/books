# Is this just RAG? An honest comparison

Short answer: **partly.** Checking an answer against something outside the
weights is an old idea, and CARE's overflow store is a small fact database,
much like LMLM's. What is different is *what* gets stored, *when* it is
consulted, and *what guarantee* you get. Those differences matter for cost
and for hallucination.

## What each system keeps outside the weights

| | RAG | KG verification (KGR etc.) | LMLM | CAR (this repo) | CARE (this repo) |
|---|---|---|---|---|---|
| Stored | document chunks + vector index | full knowledge graph | (entity, relation) → value database | a Bloom checksum of (entity, relation, value), **no values** | checksum + values only for facts the weights can't recall |
| Size per fact | thousands of bits (text + embedding) | key + value + index | key + value | ~14 bits (+ ~10 per key for familiarity) | the same + ~30 bits × leftover share |
| Needs a retriever / ANN search | yes | yes | exact lookup | no, hash lookups only | exact lookup for the leftovers |
| Extra context tokens at inference | yes (often thousands) | no | small | no | no |
| Where the answer comes from | the model reading retrieved text | model, then corrected by the KG | the database | **the model's own weights**, certified | weights first, store for the leftovers |

## Behaviour on the three kinds of question

| Question type | RAG | CAR / CARE |
|---|---|---|
| Fact in the corpus, retriever finds it | usually right; can still misread or ignore context | right if the weights put it in the top N (CAR), always (CARE) |
| Fact in the corpus, retriever misses it | the model falls back to its weights and **hallucinates at the usual rate** | unaffected: there is no retriever, and familiarity is exact |
| Fact never in the corpus | the retriever still returns its top k of *something*; abstention needs a similarity threshold with no guarantee | key filter says "never read": hallucination ≈ ε_key × N × ε_checksum, a number you set in advance |
| Question about someone who doesn't exist | same as above | "never heard of this person" |

## Where RAG is better (and CAR is not a replacement)

- **Fresh or private knowledge the model was never trained on.** CAR can only
  certify what the weights can generate. CARE's overflow and RAG both handle
  new facts; RAG handles *text*, not just triples.
- **Citations.** RAG can point to a source document. CAR can say "read 7
  times", not where.
- **Non-factoid knowledge.** Procedures, explanations and long reasoning over
  documents are RAG territory. CAR covers factual slots.
- **No extraction pipeline.** RAG indexes raw text. CAR needs facts to be
  extracted and linked (see `docs/scaling.md`).

## Where CAR / CARE is better

- **Cost.** ~2 bytes per fact and hash lookups, instead of an ANN index,
  retrieval latency and thousands of extra context tokens per question.
- **A guarantee.** The false-positive budget is set by the filter sizes and
  the list size, and the formula in `docs/theory.md` §5 predicted the
  measured hallucination closely (`RESULTS.md` §3). RAG has no comparable
  bound, because retrieval always returns something.
- **It unlocks what the model already knows.** In our runs, weights that
  "knew" most facts still hallucinated 20 to 48% of the time on facts they
  had never seen. The checksum turns that into near-zero hallucination
  without retraining and without retrieving anything.
- **Graded "I don't know".** Unknown person / unknown fact / tip of the
  tongue / read once, unconfirmed. RAG gives one similarity score.

## They combine naturally

The familiarity filters make an ideal **retrieval trigger**. Retrieve only
on "tip of the tongue" (the fact was read, but the weights can't produce a
verified answer). Never retrieve for facts never read (nothing to find) or
for facts the weights already recall with a verified answer (no need). That
avoids the two costs of always-on RAG: latency on easy questions, and
misleading context on unanswerable ones.

## Bottom line for a PhD proposal

Don't pitch CAR as "better than RAG". Pitch it as the missing *error
detection layer* for parametric knowledge. Language models decode knowledge
from a lossy memory with no error-detecting code; every other system that
does that (radio, storage, 5G) adds one. CAR adds one, at ~2 bytes per
fact, with a closed-form error budget, and composes with RAG instead of
competing with it.
