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

## Key results (3 seeds, `RESULTS.md`)

Hard synthetic world: 10,000 people, 4 attributes, answers are 2-token
strings from 1,024 values. Popularity is Zipfian: 14,444 facts seen, 61% of
them only once. About 10% of test questions are about facts never seen.
Tiny GPTs trained on CPU. The ceiling (every seen fact answered correctly)
is 89.9%.

| Weights | Weights alone, greedy acc / halluc. | Weights alone, best acc at ≤1% halluc. | + IDK training, ≤1% | **+ checksum, N=1** | **+ prefix checksum, N=50** | **CARE** (store holds) |
|---|---|---|---|---|---|---|
| 12k params | 50.7% / **49.3%** | 49.7% | 54.6% | 50.7% / 0.06% | 86.0% / 0.73% | 89.9% / 0.00% (94%) |
| 37k | 65.8% / **34.2%** | 64.4% | 68.8% | 65.8% / 0.01% | 88.2% / 0.43% | 89.9% / 0.00% (80%) |
| 123k | 82.8% / **17.2%** | 78.9% | 79.3% | 82.8% / 0.00% | 89.8% / 0.04% | 89.9% / 0.00% (23%) |
| 258k | 89.3% / **10.7%** | 85.0% | 85.1% | 89.3% / 0.00% | 89.9% / 0.00% | 89.9% / 0.00% (1%) |

1. **The knowledge is already in the weights; knowing THAT is what's
   missing.** The 258k model knows almost every fact it read (89.3% vs the
   89.9% ceiling). It still makes up an answer to every never-seen
   question: 10.7% hallucination, and 100% of questions about people who
   don't exist. "I don't know" training only gets it to 85.1% at ≤1%
   hallucination, and it still answers 37% of the ghost-people questions.
   A 14-bit-per-fact checksum gives **89.3% at 0.00%**, with no retraining.
2. **The theory predicts the numbers.** The list-decoding formula
   (`docs/theory.md` §5) matched measured accuracy exactly and measured
   hallucination to within a few hundredths of a percent, across all sizes
   and list lengths (`RESULTS.md` §3).
3. **Weights and checksum split the bits like a decoder and its CRC.**
   Without weights, checking all 1,024 values needs a 28-bit checksum
   (89.8%, 0.03%). With the 123k weights, a 14-bit checksum at N=1 gives
   0.00%. In prefix search, even the 12k-param weights add 10 points of
   accuracy and cut hallucination 2.7x compared with searching in random
   order.
4. **Graded "I don't know" comes for free.** "Never heard of this person"
   (4.6% of questions), "know them, never read this" (5.5%), "tip of the
   tongue" (0.04%).
5. **Two memories that must agree are robust to noisy extraction.** With 5
   or 10% of mentions filed under the wrong person, a plain store
   hallucinates 1.1 or 2.1%. CAR with N=1 hallucinates 0.08 or 0.34%
   (lower accuracy, though). Strict CARE halves the store's error.
6. **CARE shrinks the store as the weights get better**: from 94% of facts
   (12k params) to 23% (123k) to 1% (258k), always at the ceiling with
   0.00% hallucination.
7. **Continual learning.** New facts are written to the filters and
   overflow store instantly, with no gradients. During sleep, the model
   rehearses its own memories, but only the ones the checksum verifies:
   0.9% of those are wrong, against 46% for unverified self-replay
   (`results/continual_seed0.json`).

What failed is written up honestly in [`docs/ideas.md`](docs/ideas.md):
two list-decoding losses, a count-min-only checksum, my "confidence tracks
fame" hypothesis, and (so far) the extraction-free familiarity sense.

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

"Isn't this just RAG?" is answered honestly in
[`docs/vs_rag.md`](docs/vs_rag.md): partly, and here is exactly where it
differs and where RAG is still better.

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
