# Results: OLMo-2 1B on PopQA (pilot, 300 questions + 100 ghosts, seed 0)

`python -m experiments.real_popqa --n 300 --ghosts 100`, raw numbers in `results/real_popqa_pilot.json`.

Setup: 300 PopQA questions, 60 from each fifth of subject popularity (so the long tail is over-weighted compared with PopQA). 100 ghost questions ask about people whose full name never occurs in OLMo-2's pretraining corpus. Candidates are the greedy answer followed by up to 9 more from a 10-beam search. "Read" and "co-occur" are measured on OLMo-mix-1124, the corpus the model was trained on, through infini-gram. With 300 questions, a rate near 20% has a 95% interval of about ±4.5 points.

| System | Accuracy | Hallucination | Abstain | Ghost questions answered |
|---|---|---|---|---|
| Greedy | 19.3% | 80.7% | 0% | 100% |
| Greedy, confidence threshold (log p ≥ -2.5) | 17.0% | 23.0% | 60% | 46% |
| Greedy, confidence threshold (log p ≥ -1.8) | 13.7% | 11.3% | 75% | 27% |
| Entity filter (name occurs in corpus) + greedy | 19.0% | 80.3% | 0.7% | **0%** |
| CAR, pair checksum, N=1 (the QuCo-RAG check, abstaining) | 16.0% | 30.7% | 53% | 0% |
| CAR, pair checksum, N=10 | 20.7% | 55.3% | 24% | 0% |
| CAR, oracle triple checksum, N=1 | 19.0% | 0.0% | 81% | 0% |
| CAR, oracle triple checksum, N=3 | 29.0% | 0.7% | 70% | 0% |
| CAR, oracle triple checksum, N=10 | **34.7%** | 1.3% | 64% | 0% |

Top-N recall (the right answer anywhere in the first N candidates): N=1 19.3%, N=3 29.3%, N=5 32.3%, N=10 35.3%.

What this shows:

1. **The list-decoding gap is real and large in a real model.** The right answer is in the top 10 for 35% of questions but first for only 19%. With a checksum that recognises true facts, CAR nearly doubles accuracy (19.3% to 34.7%) at 1.3% hallucination, which matches the emulated Bloom rate (14 bits, N=10). This is an upper bound: the oracle checksum assumes perfect fact extraction from the corpus.
2. **The entity filter alone stops every ghost question** (100% to 0%) and costs almost nothing on real ones, because 99.3% of PopQA subjects occur in the corpus. A confidence threshold at 60% abstention still answers 46% of ghosts. The ghosts are built as names absent from the corpus, so this is the filter's intended case, not a hard test.
3. **The extraction-free pair checksum fails.** 42% of wrong candidates co-occur with their subject within 100 tokens of the corpus (common values like "politician", "American football", "London"), and 29% of right candidates do not. At N=1 it is worse than a plain confidence threshold (16.0% / 30.7% against 17.0% / 23.0%), and a longer list only adds hallucinations. The QuCo-RAG signal is too weak to certify an answer; it can only flag.

So the idea stands or falls on the checksum between those two: something close to fact-level that can be built from the corpus without a perfect extractor.
