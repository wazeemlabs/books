# Literature map and novelty check

Compiled on 2026-09-25. The container this was written in could not open
arXiv, OpenReview, Semantic Scholar or Hugging Face pages, so every paper
below was checked through search-engine abstracts and snippets, not full
text. Items marked **(unverified)** could not be confirmed at all. Read the
closest papers in full before writing any proposal.

## 1. The theory we build on

| Paper | What it says | Why it matters here |
|---|---|---|
| Kalai & Vempala, *Calibrated Language Models Must Hallucinate*, STOC 2024 (arXiv 2311.14648) | For arbitrary facts, a calibrated LM hallucinates at a rate close to the fraction of facts seen exactly once in training (Good-Turing estimate), minus miscalibration. The bound "has nothing to do with the transformer architecture". | Sets the floor. Also says there is "no statistical reason" to hallucinate on facts seen more than once. |
| Kalai, Nachum, Vempala, Zhang, *Why Language Models Hallucinate*, 2025 (arXiv 2509.04664; also Nature 2026) | err ≥ singleton rate − small terms − miscalibration. Abstention escapes the bound: "a non-hallucinating model could be easily created, using a question-answer database and a calculator... otherwise outputs IDK". Benchmarks reward guessing. | The floor is escaped only by abstaining correctly. So the real problem is knowing what you have seen. |
| *Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing*, ICML 2026 (arXiv 2602.00906) | Memorising random facts is a membership-testing problem (Bloom filters meet log-loss). Under a memory budget the optimal parametric solution is a "hallucination channel": all facts plus a fraction q* of non-facts get the same high score, so no threshold separates them. | Our central motivation. We take their conclusion literally: if weights are a bad membership structure, give the model a good one. |
| Miao & Kearns, PNAS 2026 (arXiv 2502.08666) | First empirical test of the singleton bound on synthetic Pareto data. Upweighting ~5% of examples cuts hallucination by up to 40% (at a calibration cost). | Frequency-aware training helps. No memory, no abstention mechanism. |
| Hron et al., *Training LMs on the Knowledge Graph*, 2024 (arXiv 2408.07852) | Hallucinating on ≤5% of training facts needs ~10x more compute than Chinchilla-optimal; hallucinations of bigger LMs are harder to detect. | Scaling the weights is an expensive way to know what you know. |
| Allen-Zhu & Li, *Physics of LMs* 3.1 and 3.3 (arXiv 2309.14316, 2404.05405) | About 2 bits of knowledge per parameter at ~1000 exposures, ~1 bit/param at 100 exposures; rare "junk" data cuts useful capacity sharply. | Weights are an expensive store for rare facts: ~6 bits of value costs several 16-bit parameters. |
| Kandpal et al., ICML 2023 (arXiv 2211.08411) | Accuracy on a fact tracks how many pretraining documents mention it. | Long-tail knowledge is the weak spot. |

## 2. Closest prior work, and what each one does NOT do

| Paper | What it does | Overlap with our idea | Missing piece |
|---|---|---|---|
| **LMLM**, Zhao, ..., Weinberger, ICLR 2026 (arXiv 2505.15962); **Co-LMLM** (2607.07707) | Moves (entity, relation, value) facts to a database during pretraining and masks them from the loss. Small models (176M/382M) reach the factual precision of much larger ones. Returns "unknown" on a lookup miss. | Strong: exact store, facts out of the weights. | Externalises all facts, with no exposure counter and no consolidation into weights. No familiarity structure separate from the store, and no graded abstention. No Good-Turing argument. |
| **Memory³**, Yang et al., 2024 (arXiv 2407.01178) | Cost model: knowledge with expected usage count between ~0.5 and ~13,400 is cheapest as explicit memory; more frequent knowledge belongs in the weights. | Frequency decides which memory tier a fact lives in. | Uses inference-time usage cost, not training exposure. Static. No abstention. |
| **Pretraining with Hierarchical Memories**, Pouransari et al. (Apple), ICLR 2026 (arXiv 2510.02375) | Small "anchor" LM plus a large memory bank for long-tail knowledge. 160M + 18M fetched ≈ a dense model more than 2x bigger. | Small core, long tail outside. | The split emerges from clustering. Memory is parametric and not exact. No abstention. |
| **Cram Less to Fit More** (Apple), 2026 (arXiv 2604.08519) | Pruning facts and flattening their frequency distribution lets a 110M model memorise 1.3x more facts. | Training weights on too many facts hurts. | Drops facts instead of routing them to memory. |
| **Semi-parametric LM with selective memory**, Sun, Padthe, Asai, Yih, 2025 (OpenReview 5a2H3HEN61; venue unverified) | Stores externally only the atomic facts the model does not already know. | Selective routing. | Routes by current knownness, not by counts. No familiarity sketch. |
| **EVAF** / *Memory Depth, Not Memory Access*, 2026 (arXiv 2606.29916, 2606.26806) | CLS-inspired LoRA consolidation for agents, gated by surprise and valence, with a separate retrieval path for facts. | Selective consolidation plus a retained fast store. | Gating criterion is salience, the opposite of repetition. No hallucination or abstention analysis. |
| **Dual-Layer Agentic Memory**, 2026 (arXiv 2608.22215) | CLS framing. A fast write router (non-write / write-new / write-update) and a slow write-back into weights by fine-tuning. | Routing plus periodic consolidation. | Routes by whether the model already knows, not by counts. Agent memory, not a model of knowledge and abstention. |
| **Surprise as a Signal for Plasticity and Metacognition**, Mouchon, 2026 (arXiv 2606.31495) | Surprise-gated episodic writes, offline replay into a slow readout, and a VLM that refuses to name untaught objects based on its novelty score. | Episodic plus consolidation plus novelty-based refusal. | Vision, not LM facts. No counts. One level of refusal. |
| **Language Models Need Sleep**, Behrouz et al., 2026 (arXiv 2606.03979) | Sleep-phase distillation into parameters, with "dreaming". | Sleep consolidation for LMs. | Selection criterion unverified. No abstention link. |
| **QuCo-RAG**, 2025 (arXiv 2512.19134) | Uses exact pretraining counts (Infini-gram, 4T tokens) at the entity and entity-pair level to decide when to retrieve. | Training-data counts used at inference. | Needs the whole corpus index. Triggers retrieval, not abstention. No compact sketch and no consolidation. |
| **Mallen et al.**, ACL 2023 (arXiv 2212.10511) | Retrieve only for low-popularity entities (PopQA). | Popularity-based routing. | Popularity from Wikipedia views, at inference only. |
| **Ferrando et al.**, ICLR 2025 (arXiv 2411.14257); **Brzezinka**, 2026 (arXiv 2607.07670, 2607.13568) | Known-entity directions inside LLMs causally drive refusal. Entity familiarity is separable from fact reliability. | The entity-level versus fact-level split exists inside LLMs. | Post-hoc probes, not an architecture. Single-level refusal. |
| **RF-Mem**, ICLR 2026 (arXiv 2603.09250) | Explicit recollection-versus-familiarity dual process for retrieval in personalisation. | Same cognitive vocabulary. | Chooses a retrieval path; does not abstain. |
| **GRACE** (arXiv 2211.11031), **WISE** (arXiv 2405.14768), **Larimar** (arXiv 2403.11901) | Knowledge editing with codebooks or side memories plus routers. Larimar does one-shot episodic writes. | Routing between memory and weights. | Not count-based. No abstention by design. |
| **HOLA**, *A Hippocampus for Linear Attention*, 2026 (arXiv 2607.02303) | Exact KV "hippocampus" added to a linear-attention model. | CLS inspired, cheap core plus exact memory. | In-context memory, not a store of training facts. |
| **Titans** (2501.00663), **ATLAS**, **Nested Learning / HOPE** | Test-time memory with surprise-gated writes. | Selective writes. | Surprise, not counts. No abstention. |

## 3. Neuroscience behind the design

- McClelland, McNaughton & O'Reilly 1995; Kumaran, Hassabis & McClelland
  2016: complementary learning systems. The hippocampus learns fast, the
  neocortex learns slowly, and replay moves knowledge between them.
- **Go-CLS**, Sun, Advani, Spruston, Saxe & Fitzgerald, *Nat Neurosci* 2023:
  consolidating everything overfits. Unpredictable memories should stay in
  the hippocampus. Arbitrary one-off facts are exactly that kind.
- Yang et al. (Buzsáki lab), *Science* 2024: the experiences reactivated most
  often while awake are the ones replayed in sleep. This is a count-like
  selection rule.
- Norman & O'Reilly 2003; Yonelinas: familiarity and recollection are
  separate processes. You can know you have met someone without recalling
  their name.
- Evidence against a pure count rule: Schapiro et al. 2018 (replay
  prioritises weakly learned items); Tse et al. 2007 (schema-consistent
  information consolidates after one trial). So we treat the count threshold
  as a statistical proxy, not a biological claim.

## 4. What we found nobody doing (as of 2026-09-25)

1. Keeping a compact **familiarity structure (Bloom or count-min sketch) that
   is separate from both the weights and the value store**, and using it as
   the abstention signal. QuCo-RAG uses corpus counts, but to trigger
   retrieval and with a full-corpus index.
2. Measuring how badly weights do membership testing at low exposure, and
   showing that an explicit ~10-bit-per-fact structure removes the
   "hallucination channel" of arXiv 2602.00906.
3. **Consolidation gated on exposure count** (≥ k), with k chosen from the
   Good-Turing count histogram. Existing gates use surprise, valence,
   knownness, usage cost or clustering.
4. **Graded abstention** as a designed output: "never heard of this person"
   versus "I know them, but not this fact". So far it has only been probed
   after the fact inside LLMs.
5. Evaluating any memory-augmented LM against the singleton-rate floor, and on
   the full accuracy-versus-hallucination frontier against weights-only
   abstention.

Caveat: the search budget ran out before a dedicated search for
"count-min sketch / Bloom filter + LM abstention" could be run. Point 1 is
the most important claim to re-verify with full access.

---

# Round 2 novelty check: checksum-aided recall (CAR / CARE)

Searched on 2026-09-25 (web search; full texts again not reachable from the
container). Queries covered Bloom filters and count-min sketches for LLM
abstention, factuality and decoding; CRC and list decoding for knowledge
recall; generate-recognize recall; KG-verified top-k answers; prefix and
Bloom tries in constrained decoding; and exposure-count conditioning.

## Closest work and how CAR / CARE differs

| Work | What it does | Difference |
|---|---|---|
| **Data Portraits**, Marone & Van Durme, NeurIPS D&B 2023 (arXiv 2303.03919) | Strided Bloom filters over n-grams of a training corpus (~3% of corpus size), used to check test-set leakage and overlap of generations with the training data. | Closest data structure, used for auditing and never inside the answer loop. Surface n-grams, not (entity, relation, value) facts. No candidate lists, no abstention, no graded answers. |
| **Retrieval-Constrained Decoding**, 2025 (arXiv 2509.23417) | Restricts answers to entity surface forms from a KB. Llama-3.1-70B goes from 32.3% to 46.0% F1: models know more than greedy decoding shows. | Supports our premise that the answer is often in the top N. The constraint is type-level (valid entity names), not fact-level (this value for this subject was read), so it does not stop wrong-but-valid answers and has no abstention. |
| **KG-based verification / retrofitting** (KGR, arXiv 2311.13314; TATK, arXiv 2609.14565; LLM re-ranking of KG completion) | Extract claims or top-M candidates, verify against a stored knowledge graph, re-rank or fix. | Needs the full KG with values and an external retriever. Our checksum stores no values (~14 bits per fact), is built from the model's own training data, and comes with a closed-form false-positive / hallucination budget. |
| **Vectorizing the Trie**, 2026 (arXiv 2602.22647) and generative-retrieval constrained decoding (tries, FM-index) | Keep generated item IDs valid. It mentions a Bloom-style prefix approach with a ~2.1% false-positive rate. | Same mechanism family as our prefix checksum. It is used for catalogue validity, not per-subject factual membership, and not for abstention. We cite it for the prefix idea. |
| **The Anxiety of Influence: Bloom Filters in Transformer Attention Heads**, 2026 (arXiv 2602.17526) | Some attention heads behave like Bloom filters for "has this token appeared in the context?" | In-context membership inside the network. Ours is an external filter over training facts. It supports the view that membership testing is a natural primitive. |
| **Space-optimality rate-distortion**, ICML 2026 (arXiv 2602.00906) | Under capacity limits, weights optimally hallucinate (Bloom-filter view of memorisation). | Our motivation. We add the explicit filter they imply, and the list-decoding use of it. |
| **Memory capacity**, Morris et al. 2025 (arXiv 2505.24832) | GPT-style models store ~3.6 bits per parameter. | Puts our "14 bits per fact outside the weights" in scale: a few parameters' worth, per fact. |
| **Predictable Confabulations**, 2026 (arXiv 2605.18732) | Factual recall is a sigmoid in log(params) and topic frequency. | Explains why a frequency-agnostic familiarity signal is needed for the long tail. |
| **Dolma**, Soldaini et al. 2024 (arXiv 2402.00159) | Bloom-filter deduplication over a 3T-token pretraining corpus at 1% FPR. | Scalability evidence: Bloom filters over whole pretraining corpora are routine infrastructure. |
| **LMLM** (arXiv 2505.15962), **Memory³** (arXiv 2407.01178) | Store fact values outside the weights. | These store values. CAR stores only a membership checksum and lets the weights generate. CARE keeps values only for the facts the weights can't recall. |

## What we could not find anywhere

1. A membership checksum over training-set facts used **inside** recall to
   choose among the model's own top-N candidates, and to abstain when none
   verifies.
2. The **CRC-aided list decoding** view of LM factual recall, with its rate
   split between weights and checksum (docs/theory.md §5).
3. **Graded metamemory from filters**: unknown person / unknown fact / tip
   of the tongue / read once, unconfirmed.
4. **Sleep-time eviction**: the episodic overflow keeps only facts the
   weights plus checksum fail to recall (CARE).
5. The **agreement argument** for noise robustness: a mislinked fact must be
   both in the checksum and proposed by the weights.

Absence from web search is not proof of novelty. The first thing to do with
full library access is a careful read of Data Portraits, RCD, KGR and the
generative-retrieval constrained decoding papers.
