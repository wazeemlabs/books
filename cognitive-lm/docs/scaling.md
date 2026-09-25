# Taking CARE from a toy world to real language models

## What has to exist at scale

| Piece | Toy version here | Real version | Cost at 10^9 facts |
|---|---|---|---|
| Fact extraction | exact (entity, relation, value) ids | relation extraction + entity linking over the pretraining corpus, e.g. the LMLM pipeline (54.6M triples from Wikipedia with a distilled extractor), aligned to Wikidata ids | one pass of a small extractor model over the corpus |
| Entity + key filters | Bloom, 10 bits/key | same | ~1.25 GB each |
| Triple checksum | Bloom, 14 bits/fact | Bloom, or a cuckoo/xor filter (deletions, better space below ~3% FPR) | ~1.75 GB |
| Prefix checksum | one prefix per fact | one entry per value-token prefix (L − 1 per value) | ~1 GB per extra prefix level at 8 bits |
| Episodic overflow | exact dict | key-value store for facts the model can't recall after training | depends on the model; ~20% of facts for our d=64 model |
| Weights | tiny GPT | any LLM; CAR needs no retraining | none |

For scale: Dolma already runs Bloom-filter deduplication over 3T tokens at
a 1% false-positive rate, so filters over a whole pretraining corpus are
routine.

## How it runs at inference

1. The model reaches a factual slot. Either it emits a structured query
   (tool-call style), or a light claim detector marks it. For long text,
   claims are split into atomic facts (FActScore-style) after generation.
2. The subject and relation are linked to ids, and the key filter is
   checked. If it misses, the model says "I never read that", graded by
   the entity filter.
3. Beam search gives the top N values. Each is canonicalised and checked
   against the checksum, with the prefix checksum pruning beams early.
4. The first verified value is answered. Otherwise the overflow store is
   tried; otherwise "tip of the tongue". Strict mode flags overflow facts
   read only once.

Bloom lookups take nanoseconds. The real cost is N beams over the answer
span, which batches well and applies only to factual slots.

## Risks and open problems

- **Linking and canonicalisation errors.** A missed mention only lowers
  recall (the model abstains more). A mislinked one can cause a
  hallucination, but only if the weights also propose the wrong value
  (RESULTS.md §5). Paraphrased relations and values need normalisation
  or learned keys.
- **Facts that are not triples.** Procedures, reasoning and opinions are
  out of scope. CAR is for factual slots, where hallucination hurts most.
- **Privacy.** A filter over training facts is a membership oracle for the
  training data. Leave out personal-data relations, build only from
  public sources, or add noise to the filter.
- **Deletion and updates.** New facts go in instantly, with no gradient
  step. Deletion needs a counting Bloom or a cuckoo filter, which also
  gives a clean "unlearn this fact" operation that weights alone don't.
- **Adversarial inputs.** False-positive rates hold for random keys. An
  adversary who knows the hash seeds could craft collisions; use keyed
  hashes.

## A concrete next step: OLMo + Dolma

OLMo models ship with their full pretraining corpus (Dolma), so the
checksum can be built from exactly the data the model read:

1. Extract triples from the Wikipedia and encyclopedic parts of Dolma, link
   them to Wikidata, and build the filters.
2. Evaluate OLMo-1B/7B on PopQA, EntityQuestions and SimpleQA-style
   questions with CAR (N = 1, 5, 20 beams) against greedy decoding, IDK
   fine-tuning (R-Tuning), semantic entropy, and RAG over the same
   Wikipedia.
3. Report accuracy against hallucination, broken down by entity
   popularity (the long tail is where it should help most), plus filter
   size and latency.

## PhD roadmap (suggested)

- **Year 1.** The synthetic results here with more seeds and larger
  worlds, then OLMo + Dolma as above. First paper: "Language models need a
  checksum", CAR + theory + OLMo results.
- **Year 2.** Learned keys: hash the model's own subject/relation
  representations instead of extracted triples, so paraphrases and
  free-form claims work. Long-form generation with claim-level
  verification. Second paper.
- **Year 3.** Train with the checksum in the loop: pretraining that
  expects a recogniser (list-decoding objectives that actually work, which
  our two attempts did not), sleep-time consolidation of the overflow, and
  count-aware verbal confidence ("I read this once"). Thesis: recall =
  generate + recognise, with a cheap exact recogniser, is a better
  cognitive architecture for knowledge than weights alone.
