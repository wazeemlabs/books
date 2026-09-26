# Research log: ideas tried, kept and dropped

Everything here was run on the synthetic worlds in `cogllm/`, on seed 0
unless noted. Numbers are from `results/pilot_*.json` and `results/seed0.json`.

## Round 1: complementary memory (CMLM)

**Idea.** Split familiarity (Bloom filters over entity and fact keys) from
recollection (weights for facts seen ≥ k times, an exact episodic store for
the rest). Consolidate in "sleep" by exposure count, justified by
Good-Turing.

**What held up**
- Weights are a poor membership test for rare facts. Confidence separates
  "seen once" from "never seen" with AUROC 0.45 to 0.56 in the small models
  (0.5 is a coin flip), even with "I don't know" training. The Bloom filter
  is at about 0.995.
- Training the weights only on facts seen ≥ 2 times improved their top-1
  recall on the query stream (0.843 to 0.891 on the V = 48 world). This is
  the interference effect that Cram-Less (arXiv 2604.08519) also reports.

**What did not**
- For arbitrary facts an exact store is cheaper per bit than weights, so the
  store-only corner (LMLM-like) wins on bits. CMLM's weights earn their
  place only through retrieval load, not accuracy or memory. That makes it
  a weak research claim, and it is too close to LMLM.
- "The weights' confidence tracks how famous the person is, not the fact"
  was my explanation for the below-chance AUROC. It was **false**. In the
  d = 64 model, confidence on never-seen facts is ~0.61 for famous and
  obscure people alike (`results/anatomy.json`).

## Round 2: checksum-aided recall (CAR / CARE), the direction we kept

**Idea.** Treat recall like CRC-aided list decoding in 5G polar codes:
- The weights are a list decoder. They propose their top N answers.
- A checksum recognises the right one, or flags a failure. The checksum is
  a Bloom filter over every (entity, relation, value) triple read in
  training, at ~10 to 20 bits per fact, with no values stored.
- This is also the generate-recognize theory of human recall (Kintsch 1970).
- Graded metamemory falls out for free: "never heard of them", "know them,
  never read that", and "tip of the tongue" (familiar, but nothing
  verifies).

**Variations tried**

| Variation | Result | Kept? |
|---|---|---|
| CAR, weights that already know the facts (d = 64, V = 48) | Weights alone: 69.1% accuracy at ≤ 1% hallucination. With the checksum: **84.7% at 0.00%**, the ceiling. The knowledge was there; only "knowing that" was missing. | yes, the headline |
| CAR, capacity-limited weights (d = 32, V = 48), list size N | N = 1: 71.5% at 0.11% hallucination; N = 5: 76.0% at 0.43%; N = 20: 81.2% at 0.95%. Weights alone: 65.0% at ≤ 1%. | yes |
| Checksum bits b (N = 5) | b = 4: 5.8% hallucination; 6: 2.5%; 8: 1.1%; 10: 0.43%; 14: 0.07%; 20: 0.01%. It tracks N·ε, as the theory says. | b = 14 default |
| Key filter before the triple checksum (two-stage gate) | Without it, hallucination at N = 48, b = 10 goes from 1.2% to 4.6%. The false-positive rates multiply. | yes |
| Count-min sketch as the checksum | Bad membership test at this budget (~16% FPR at 16 bits per fact). | dropped |
| Count-min only as a tie-breaker between verified candidates | Under 5 to 20% entity mislinks it cuts hallucination (e.g. N = 5: 1.04% to 0.59% at 5% mislinks). | yes, for noisy data |
| CAR vs a plain fact store under mislinks | Store: 1.3%, 2.7%, 5.6% hallucination at 5, 10, 20% mislinks. CAR N = 1: 0.24%, 0.51%, 1.2%. Two independent memories must agree. The cost is lower accuracy (70% vs 84%). | yes |
| List-decoding hinge loss (train for top-N, not top-1) | **Worse at every N** (top-5 recall 0.64 vs 0.90 for plain CE). | dropped |
| Rank-gated CE (skip the push once the answer is in the top N) | Worse: top-1 on seen facts 0.614 vs 0.748 for plain CE (hard world) | dropped |
| Weights trained only on facts seen ≥ 2 times, then CAR | Hard world, 3 seeds: greedy 68.4% vs 65.8%, CAR N=5 71.0% vs 67.8%. No gain once the prefix checksum is used (88.0% vs 88.2%). | optional |
| Prefix checksum (Bloom over (key, first sub-token)) for multi-token values | 12k-param weights: 50.7% (N=1) to 86.0% at 0.73% halluc. (N=50). Weights add +10 pts and 2.7x lower hallucination vs random-order search with the same filter. | yes |
| CARE: CAR plus an episodic overflow for facts CAR can't recall after sleep | Ceiling accuracy, 0.00% hallucination at every size. The store holds 94% / 80% / 23% / 1% of facts as the weights grow from 12k to 258k params. | yes |
| CARE strict (don't assert overflow facts read once) | Under 5 / 10% mislinks, hallucination 1.09 → 0.49% and 2.11 → 1.03%, costing ~8 points of accuracy | option |

## Ideas considered but not run

- **Count-conditioned pretraining.** Feed "how many times have I seen this"
  as an input so the model learns calibrated abstention without IDK
  labels. It only makes sense in single-epoch pretraining, which tiny CPU
  models can't do. Future work.
- **Counters on memory-layer slots** (a learned count-min sketch inside
  product-key memories). It would give paraphrase-robust familiarity, but
  it inherits count-min collisions. Worth testing at GPU scale.
- **Learned keys** (hash the model's own subject/relation representation
  instead of extracted triples). This is the path to paraphrases and
  free-form text. It is the most important next step, and also the
  riskiest.

## Round 3: extraction-free and lifelong variants

| Variation | Result | Verdict |
|---|---|---|
| **FamLM**: an Engram-style hashed counter table (context counts + (context, next-token) counts), fed into the transformer and trained prequentially (counts as they were before each batch), in a world with 3 phrasings per relation | One pass over 400k mentions: **88.5% accuracy at ≤ 0.1% hallucination**, vs 57.0% for IDK training and 52.8% for the plain LM. Facts added to the counters after training are recalled at 100% with no gradients. **Paraphrase: 0%** (a fact read only in other phrasings is never recalled). The non-prequential ablation scores the same 88.5%, so the "learned familiarity from first exposures" story is **not** what drives it; exact n-gram recall is. The pure n-gram lookup gets 84.0% at 2.7% hallucination, so the learned gating adds calibration. | a useful learned n-gram memory, but not the breakthrough; the paraphrase problem is unsolved |
| FamLM with the familiarity embedding only (no recognition bias) | 61.1% greedy, no better than the plain LM | the recognition bias is what matters |
| Continual learning: fine-tune on 4,000 new facts | Weights forget old facts (87.6% → 0.9%) | as expected |
| + self-replay (the model rehearses its own greedy answers) | 46% of rehearsed "memories" are wrong; old facts 55.5% | rehearsing its own hallucinations |
| **+ checksum-verified self-replay** | only 0.9% of rehearsed memories wrong; old facts 68.0%; the **smallest overflow store of all (5,926 vs 8,545 with the original data replayed)**; CARE stays at 100% / 0% | yes |

| Relation-free pair checksum (entity, value) that needs only entity recognition | Same accuracy. Hallucination at N=1: 1.0 / 0.6 / 0.3% (37k / 123k / 258k params) vs 0.02 / 0.00 / 0.00% for triples. N=20: 1.7 to 5.5%. Errors are the model offering the person's value from a *different* relation. | a cheaper production tier; use value types in practice |
| **Lifecycle, 5 days** (500 new people/day; wake = write filters + store; sleep = train on the overflow + replay, then evict what is recalled) | never sleep: store +1,280/day, 100%. Sleep without replay: store tiny, but originals fall to 69% and day-1 facts to 18%, because evicted facts get overwritten. + self-replay: 84% / 44%. + verified replay: 91% / 64%. **+ verified replay + audit (two-phase commit): 99.8% / 99.2%, store +490/day, 0.00 to 0.05% hallucination on never-seen facts.** | yes; audited sleep is the lifelong-learning mechanism |

Open problem: **keys without an extractor.** Exact n-grams don't survive
paraphrase. The two candidates are (a) LMLM-style canonical key emission
by the model itself, which is already shown to work at 382M params, and (b)
learned, phrasing-invariant keys from the model's own representation of
the question. (b) is the riskiest and most valuable next step.

## Round 4: hostile review (2026-09-26)

A methods audit and a full-text literature check changed several claims.
All numbers are 3 seeds unless noted (`RESULTS.md`, `results/lifecycle_seed*.json`).

| Check | Result | Consequence |
|---|---|---|
| Key filter + greedy, no triple checksum | 258k params: 0.54% hallucination (vs 10.7% weights alone, 0.00% CAR). 12k params: 39.2% (vs 0.06% CAR) | for strong weights the key filter does most of the work; the checksum matters for weak weights |
| Cheapest exact stores | exact dict 27 bits/fact, 0.00%; key filter + static function 20 bits/fact, 0.08%; CAR N=1 29 bits + weights | CAR loses on bits when values are small (10 bits here); it has to be tested on open-string values |
| Predicted vs measured accuracy | uses the measured rank of the true value, so it matches by construction | dropped as evidence |
| Checksum sized out of sample | formula fitted on half the people met the target on the other half in 11 of 12 cases, 1.5x over in one | kept: the formula is predictive |
| Store that refuses facts read once, under mislinks | 10%: 80.3% / 0.74%, vs strict CARE 80.9% / 1.03% | strict CARE's robustness comes from distrusting singletons, not the weights |
| IDK training at its ≤1% point | 6.0% of ghost-people questions answered (258k), not 37% | README corrected |
| Lifecycle with the key list counted | audited sleep keeps 96.5 to 100% of facts, ≤0.4% hallucination on read facts; memory grows 72 kbit/day vs 75 for never sleeping | audited sleep preserves facts but saves almost no memory in this world |
| Literature, full texts | QuCo-RAG (corpus co-occurrence check, retrieves), ReFactX (exact triple trie, prompted IDK), SPLM / Dual-Layer / LMLM (store only unrecallable facts; evict after validation) | overflow store and evict-after-validation are prior work; the value-free checksum inside top-N recall, the CRC framing, verified replay and typed filter abstention were not found |

Next: a real model (OLMo-2 1B) on PopQA, with a checksum built from OLMo's
own pretraining corpus and open-string answers, where storing values is
expensive and the checksum's fixed cost can win.
