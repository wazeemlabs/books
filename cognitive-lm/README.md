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
is 89.9%. Questions arrive as exact (entity, relation) ids and fact
extraction is perfect, so these results say nothing yet about real text.

| Weights | Weights alone, greedy acc / halluc. | Weights alone, best acc at ≤1% halluc. | + IDK training, ≤1% | Key filter + greedy (no checksum) | **+ checksum, N=1** | **+ prefix checksum, N=50** |
|---|---|---|---|---|---|---|
| 12k params | 50.7% / 49.3% | 49.7% | 54.6% | 50.7% / 39.2% | 50.7% / 0.06% | 86.0% / 0.73% |
| 37k | 65.8% / 34.2% | 64.4% | 68.8% | 65.8% / 24.1% | 65.8% / 0.01% | 88.2% / 0.43% |
| 123k | 82.8% / 17.2% | 78.9% | 79.3% | 82.8% / 7.1% | 82.8% / 0.00% | 89.8% / 0.04% |
| 258k | 89.3% / 10.7% | 85.0% | 85.1% | 89.3% / 0.54% | 89.3% / 0.00% | 89.9% / 0.00% |

"0.00%" means none seen in ~2,000 never-seen questions per seed; the 95%
upper bound is 0.2% of never-seen questions.

What holds up:

1. **The weights know more than they can safely say.** The 258k model
   gets 89.3% of questions right against an 89.9% ceiling, yet answers
   every never-seen question. Checking its answer against a 14-bit-per-fact
   checksum removes the hallucinations without retraining.
2. **With weak weights the checksum is what does the work.** For the 12k
   model, the (entity, relation) key filter alone still leaves 39%
   hallucination; the triple checksum brings it to 0.06%. For the 258k
   model most of the gain is the key filter (10.7% to 0.54%).
3. **The checksum size can be chosen in advance.** Solving the
   list-decoding formula (`docs/theory.md` §5) on half the people and
   testing a checksum of that size on the other half met the hallucination
   target in 11 of 12 cases, and missed by 1.5x in the last (`RESULTS.md`
   §3b). The in-sample match of predicted and measured accuracy is close to
   an identity and should not be cited as evidence.
4. **Prefix checksums make long candidate lists cheap.** In prefix search,
   even the 12k-param weights add 10 points of accuracy and cut
   hallucination 2.7x over searching in random order with the same filter.
5. **Checksum-verified self-replay avoids rehearsing hallucinations.** 0.9%
   of rehearsed memories are wrong, against 46% for plain self-replay
   (seed 0).
6. **Graded "I don't know" comes from the filters.** "Never heard of this
   person" (4.6% of questions), "know them, never read this" (5.5%), "tip of
   the tongue" (0.04%).

What does not hold up, or is weaker than first written:

- **CAR does not save memory in this world.** A value from 1,024 options
  costs 10 bits, less than the 14-bit checksum. A key filter plus a static
  function (a value table with no keys) gets 89.9% at 0.08% hallucination
  with 20 bits per fact and no weights; CAR needs 29 bits plus the weights
  (`RESULTS.md` §2). CAR can only win on bits where values are long, open
  strings. That is the regime the real-model experiment has to test.
- **CARE's ceiling accuracy at 0.00% is the store's number.** CARE starts
  with every fact stored and only evicts facts the weights already recall,
  so its accuracy equals the store's by construction. What it shows is how
  small the store can get: 94%, 80%, 23% and 1% of facts as the weights grow.
- **Strict CARE's noise robustness is not special.** Under 10% mislinks, a
  plain store that refuses facts read once gives 80.3% / 0.74%, as good as
  strict CARE (80.9% / 1.03%).
- **Audited sleep keeps facts but not memory.** Over 5 days and 3 seeds it
  keeps 96.5 to 100% of every day's facts, with at most 0.4% hallucination on
  facts read and 0.05% on facts never read. But replay and audit need a
  list of every key read so far, which a Bloom filter can't provide. Counted
  in, memory grows 72 kbit a day against 75 for never sleeping.
- **"I don't know" training is a stronger baseline than first reported.** At
  its ≤1% operating point the 258k model answers 6.0% of questions about
  people who don't exist, not 37% (that figure was at threshold 0).
- **The pair checksum** (person, value) needs only entity recognition. It
  gets 0.3 to 1.0% hallucination at N=1 (seed 0), against 10 to 34% for the
  weights alone.

Other failures are written up in [`docs/ideas.md`](docs/ideas.md): two
list-decoding losses, a count-min-only checksum, the "confidence tracks
fame" hypothesis, and FamLM (strong on exact phrasing, 0% on paraphrase).

## Is it new?

Checked on 2026-09-26 against full texts; every paper is saved in
[`sources/`](sources/INDEX.md).

Not found anywhere:

- a compact, value-free membership filter over training facts used
  *inside* recall, to pick the first of the model's own top-N candidates
  that passes, and to abstain when none does;
- the CRC-aided list-decoding view of recall, with filter bits traded
  against list size;
- rehearsal filtered by a checksum (checksum-verified self-replay);
- a cascade of entity, key and triple filters giving typed "I don't know".

Already done, so cite rather than claim:

- **QuCo-RAG** (arXiv 2512.19134) checks generated (head, tail) pairs for
  co-occurrence in the model's own pretraining corpus, using an exact
  multi-terabyte index, and retrieves (never abstains) when the count is
  zero. It is the closest work to CAR.
- **ReFactX** (arXiv 2508.16983) constrains decoding to real Wikidata
  facts through an exact 800M-triple prefix tree, and prompts the model to
  say "I don't know".
- **An overflow store holding only facts the model can't recall**: SPLM
  (Sun, Padthe, Asai, Yih, 2025), Dual-Layer Agentic Memory (arXiv
  2608.22215), LMLM (arXiv 2505.15962).
- **Evicting a fact only after the new weights recall it**: Dual-Layer
  Agentic Memory §3.4, and the Sleeping LLM reports (Zenodo). Auditing all
  earlier facts and writing back what was forgotten is still new, but see
  the memory cost above.

"Isn't this just RAG?" is answered in [`docs/vs_rag.md`](docs/vs_rag.md).

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
cogllm/baselines.py            no-weights value stores: exact dict, key filter + static function
cogllm/system.py               round-1 complementary-memory model (kept for comparison)
experiments/run_care.py        final experiment (3 seeds) -> RESULTS.md via report_care.py
experiments/pilot_car.py       pilots on the 48-value world
experiments/pilot_mt.py        pilots on the 1,024-value world, CARE
experiments/lifecycle.py       five days of wake/sleep: verified replay, audited sleep
experiments/continual.py       one-shot continual learning, replay variants
experiments/assoc.py           relation-free (entity, value) pair checksum
experiments/pilot_fam.py       FamLM, the hashed-counter familiarity sense (cogllm/famlm.py)
experiments/run.py, report.py  round 1 (docs/round1/)
docs/                          theory, literature, ideas log, scaling + PhD roadmap
sources/                       every cited paper: PDF + full text by topic, see sources/INDEX.md
tests/                         python -m unittest discover -s tests -t .
```

## Run it

```bash
pip install -r requirements.txt
for s in 0 1 2; do python -m experiments.run_care --seed $s --out results/care_seed$s.json; done
python -m experiments.report_care       # RESULTS.md + figures/
python -m experiments.pilot_mt          # variations on the hard world
python -m experiments.pilot_car         # variations on the easy world
```

Everything runs on CPU. The 3 final seeds take about 75 minutes on 4 cores when training from scratch, and about 3 minutes from the saved checkpoints.

Trained weights for every run are in `results/ckpt/` (15 MB): `care_s{seed}_*`
from `run_care.py`, `mt_s0_*` from `pilot_mt.py`, `s0_*` from
`pilot_car.py`. The scripts load them if present instead of retraining, so
`lifecycle.py`, `continual.py` and `assoc.py` run straight away.
