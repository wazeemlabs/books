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
| Rank-gated CE (skip the push once the answer is in the top N) | see `results/pilot_mt.json` | pending |
| Weights trained only on facts seen ≥ 2 times, then CAR | N = 1: 75.5% vs 71.5% for all-facts training (V = 48, d = 32) | pending on the hard world |
| Prefix checksum (Bloom over (key, first sub-token)) for multi-token values | Prunes wrong branches before the full check, so much longer lists are affordable | pending |
| CARE: CAR plus an episodic overflow for facts CAR can't recall after sleep | Store accuracy and ~0 hallucination, with a store that shrinks as the weights get better | pending |

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
