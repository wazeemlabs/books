# CARE: language models that know what they know

**Checksum-Aided Recall with Episodic overflow.** This is a proof of
concept for a PhD direction. It gives a language model a tiny external
*recognition memory*, so it can tell facts it has read from facts it is
making up.

> Status: research prototype on synthetic data, CPU scale. Every number
> below comes from `RESULTS.md` (3 seeds) or `results/pilot_*.json`
> (seed 0), and the scripts in `experiments/` reproduce them.

## The problem in one paragraph

Theory already says hallucination on arbitrary facts (birthdays, cities,
employers) is not a bug in the transformer:

- A model that always answers must be wrong at least as often as facts
  appear exactly once in its training data (Kalai & Vempala 2024; Kalai
  et al. 2025).
- Under a capacity budget, the optimal weights are a *lossy membership
  test*: some things never seen look exactly as familiar as things seen
  (ICML 2026, arXiv 2602.00906).

We measured this directly. A model can hold a fact in its weights and
still have no idea whether it read it.

## The idea

Human recall is often described as **generate, then recognise** (Kintsch
1970): you come up with candidates, then you recognise the right one, or you
say "I don't know" or "it's on the tip of my tongue". Telecoms engineers
solved the same problem for noisy channels with **CRC-aided list
decoding**, which 5G uses: a decoder proposes a short list of candidates,
and a few checksum bits pick the right one or flag a failure.

CARE puts the two together:

| Part | Job | Cost |
|---|---|---|
| **Weights** (any LM) | the list decoder: propose the top N answers | none extra, no retraining needed |
| **Knowledge checksum** | recognition: a Bloom filter over every (entity, relation, value) triple read in training. It stores *no values*. | ~14 bits per fact |
| **Familiarity filters** | "have I ever heard of this person? of this fact about them?" | ~10 bits per key |
| **Prefix checksum** | prunes wrong branches of multi-token answers early | ~8 bits per fact |
| **Episodic overflow** | exact store only for facts the weights + checksum still can't recall after training ("sleep") | only the leftovers |

Answering a question:

```
never heard of the person        -> "I don't know who that is"
know them, never read this fact  -> "I know them, but I never read that"
top-N candidates, first one that passes the checksum -> answer
nothing passes                   -> overflow store, else "it's on the tip of my tongue"
(strict mode) overflow fact read only once -> "I read that once, but can't confirm it"
```

For never-seen facts, hallucination becomes ε_key × N × ε_checksum, a
product of two small false-positive rates. It no longer depends on model
size, the singleton rate or the training objective. The full derivation,
including how many checksum bits a given model needs, is in
[`docs/theory.md`](docs/theory.md) §5. It is the same rate split as between
a list decoder and its CRC.

## Key results so far

Hard synthetic world: 10,000 people, 4 attributes, answers are 2-token
strings from 1,024 values. Popularity is Zipfian: 14,444 facts seen, 61% of
them only once. Tiny GPTs trained on CPU. Pilot, seed 0 (3-seed numbers in
`RESULTS.md`):

| Weights | Weights alone, greedy | Weights alone, best at ≤1% halluc. | + checksum, N=1 | + prefix checksum, N=20 to 50 | CARE |
|---|---|---|---|---|---|
| 37k params | 63.5% acc, **36.5% halluc.** | 62.4% | 63.5% acc, **0.00%** | 83.5% acc, 0.31% | 84.8%, 0.01% |
| 123k params | 78.4% acc, **21.6% halluc.** | 73.2% | 78.4% acc, **0.01%** | 84.3% acc, 0.03% | 84.8%, 0.00% |

The ceiling (every seen fact answered correctly) is 84.8%.

1. **The knowledge is already in the weights; knowing THAT is what's
   missing.** On the easier world, a 133k-param model memorised 98.5% of
   even its seen-once facts. Alone, it reaches only 69% accuracy at ≤1%
   hallucination. With the checksum it reaches **84.7% at 0.00%**, the
   ceiling.
2. **Weights are a poor membership test for rare facts.** Confidence
   separates "seen once" from "never seen" with AUROC 0.45 to 0.81 in
   capacity-limited models (0.5 = coin flip). "I don't know" training
   barely helps, and IDK-trained models still answered about 14 to 44% of
   questions about people who don't exist.
3. **A better model needs a smaller checksum.** Checking all 1,024 values
   without a model needs ~28 bits per fact for 0.03% hallucination. With
   the 123k model, 14 bits and N=1 give 0.01%.
4. **Two memories that must agree are robust to noisy extraction.** With 5
   to 20% of mentions filed under the wrong person, a plain fact store
   hallucinates 1.3 to 5.6%. Checksum recall with N=1 hallucinates 0.24 to
   1.2% (easier world).
5. **CARE keeps the store small.** With the 123k model only 21% of facts
   need the overflow store, and CARE matches the store's 84.8% / 0.00%
   with 22% fewer bits outside the weights.

What failed is written up honestly in [`docs/ideas.md`](docs/ideas.md):
two list-decoding training losses, a count-min-only checksum, and my own
"confidence tracks fame" hypothesis.

## Is it new?

As of 2026-09-25, we found no work that:

- uses a membership checksum over training facts *inside* recall to pick
  among the model's own candidates and abstain otherwise;
- frames LM recall as CRC-aided list decoding, with a rate split between
  weights and checksum;
- produces graded metamemory from filters;
- keeps an overflow store holding only what the weights + checksum can't
  recall.

Closest neighbours:

- **Data Portraits** (Bloom filters over training data, used for auditing
  only).
- **Retrieval-Constrained Decoding** (restricts answers to valid entity
  names).
- **KG-verification pipelines** (these store full graphs).
- **LMLM** (stores fact values in a database).
- **Prefix-constrained generative retrieval** (keeps item IDs valid).

The full map is in [`docs/literature.md`](docs/literature.md). Web search is
not proof of novelty; read the listed neighbours in full first.

## Does it scale?

See [`docs/scaling.md`](docs/scaling.md):

- **Size.** The filters cost ~2 to 4 GB per billion facts. Bloom filters
  over whole pretraining corpora are already routine (Dolma dedup).
- **No retraining.** CAR needs no change to the weights.
- **Main risks.** Fact extraction and linking at corpus scale, facts that
  are not triples, and privacy (a filter over training facts is a
  membership oracle).
- **Proposed next testbed.** OLMo + Dolma, because the checksum can be
  built from exactly the data the model read.

## Repo map

```
cogllm/world.py, world_mt.py   synthetic worlds (48-value and 1,024-value answers)
cogllm/model.py                tiny GPT + training
cogllm/checksum.py             checksum-aided recall (Bloom checksum, key/entity/prefix filters, count tie-break)
cogllm/memory.py               Bloom filter, episodic store
cogllm/system.py               round-1 complementary-memory model (kept for comparison)
experiments/run_care.py        final experiment (3 seeds) -> RESULTS.md via report_care.py
experiments/pilot_car.py       pilots on the 48-value world
experiments/pilot_mt.py        pilots on the 1,024-value world, CARE
experiments/run.py, report.py  round 1 (docs/round1/)
docs/                          theory, literature, ideas log, scaling + PhD roadmap
```

## Run it

```bash
pip install -r requirements.txt
for s in 0 1 2; do python -m experiments.run_care --seed $s --out results/care_seed$s.json; done
python -m experiments.report_care       # RESULTS.md + figures/
python -m experiments.pilot_mt          # variations on the hard world
python -m experiments.pilot_car         # variations on the easy world
```

Everything runs on CPU. The 3 final seeds take about 75 minutes on 4 cores.
