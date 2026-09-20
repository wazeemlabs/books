# LLM Inference from the Ground Up

**Subtitle:** Serve Models Fast and Cheap — The Path to Inference Engineering

**Series:** the third *from the Ground Up* book. Book 1 built the model;
Book 2 wielded the agent; this book serves the model at production
speed and cost. It begins where *Large Language Models from the Ground
Up* ends (Chapter 30, "Sampling & beyond").

This outline is written against [STANDARDS.md](STANDARDS.md), and every
time-sensitive value in it is tracked in [FACTS.md](FACTS.md). Every
chapter follows the nine-section skeleton defined there; this document
gives each chapter's objectives, what is built and measured, and its
primary sources.

## Audience and prerequisites

- Readers of Book 1 with a working GPT who want the next job.
- Backend and ML engineers who call an API today and want to own the
  serving layer.
- Engineers targeting inference, GPU performance, or ML platform roles.

Prerequisites: Python and curiosity. Chapter 2 supplies everything
about the model the book needs; Book 1 is a deeper path, not a
requirement. PyTorch is introduced where it is first used. No CUDA is
required; Part IV teaches the reader to read kernels and profiler
output, not to write kernels. The pedagogy rules in STANDARDS.md §10
govern how every concept is introduced: intuition, then a worked
example with real numbers, then the mechanism in the reader's own
code, then the name.

## Thesis

Inference is a memory-bandwidth problem wearing a compute problem's
clothes. Every technique in this book is one of four moves: read fewer
bytes per token (caching, quantization), amortize the bytes you read
across more tokens (batching, speculation), stop doing work you do not
need (paging, prefix reuse, chunking), or put the work where the
hardware is idle (disaggregation, parallelism). The roofline model in
Chapter 8 makes this visible; the rest of the book is its consequences.

## The spine: build an engine, then read the real ones

Following Book 1's micrograd → GPT pattern, Parts II–VI grow one small
Python engine, **`tinyserve`**, from a recompute-everything generate loop
to a paged KV cache, continuous batching with chunked prefill,
weight-only quantization, and speculative decoding. Each mechanism is
built, measured against the harness, and only then given its name and
its paper. In Part VII the reader opens vLLM and SGLang and recognizes
every component.

Two running models: Book 1's own 825K-parameter GPT (every mechanism
visible on a CPU) and an 8B-class open model chosen by criteria at
writing time (dense, permissive license, current generation, a ~1B
sibling for Tier 1; see FACTS.md — the current Qwen generation ships
no 8B, so the choice is open). Real numbers on real hardware.

## The running case study

Defined in STANDARDS.md §7: a customer-support assistant on an 8B
model, 200 requests/s at peak, 1,200 input / 300 output tokens, p99
TTFT ≤ 1,000 ms, p99 ITL ≤ 50 ms, 99.9% availability, minimum cost per
million output tokens. Chapter 5 states it, each Part's design decision
record advances it, Chapter 41 sizes it, Chapter 42 prices it, and the
capstone asks the reader to beat the book's number.

## Numbers every inference engineer should know

The front matter carries this table, in the spirit of "numbers every
programmer should know". Values are for the H100 SXM (80 GB HBM3)
unless stated and are re-verified at each revision.

| Quantity | Value | Consequence |
|---|---|---|
| HBM bandwidth | 3.35 TB/s | The decode speed limit |
| BF16 dense peak | ~990 TFLOP/s | The prefill speed limit |
| FP8 dense peak | ~1,980 TFLOP/s | Why FP8 is the default |
| Ridge point (BF16) | ~300 FLOP/byte | Below this you are bandwidth-bound |
| NVLink per GPU | 900 GB/s | Tensor parallel is feasible in-node |
| PCIe 5.0 x16 | ~64 GB/s per direction | Tensor parallel across PCIe is not |
| 8B model, BF16 weights | 16 GB | Fits one GPU with room for cache |
| Decode floor, 8B, batch 1 | 16 GB / 3.35 TB/s ≈ 4.8 ms/token | ~210 tok/s, and the GPU is 99% idle |
| KV cache, Llama-3-style 8B (32 layers, 8 KV heads, d=128, BF16) | 128 KiB per token | 1 GiB per 8K-token sequence |
| Tokens in flight to reach the ridge *if the cache were free* | ~148 | The batch that would pay for the GPU |
| Where decode's work-per-byte actually saturates (1,500-token context) | ~81 FLOP/byte | Batching alone never reaches the ridge |
| Prefill, 8B, 1,000 tokens, ideal | 16 TFLOP / 990 TFLOP/s ≈ 16 ms | Real: 2–3× this |
| Human reading speed | ~5 tokens/s | ITL below ~50 ms reads as instant |
| H100 on-demand rental (Sep 2026) | median $3.25/hr; $1.5–7 by provider | The denominator of every cost figure |

Every row is derived in the chapter that owns it, checked by the
harness, and carried in FACTS.md with its verification date.

## The compute ladder

| Tier | Runs on | Chapters | Cost to reproduce |
|---|---|---|---|
| 0 | CPU / free Colab | 1–18 (825K GPT), 24 | $0 |
| 1 | Colab T4 or one consumer GPU | 10–18 (1B model), 25, 27–32 | < $10 |
| 2 | One rented H100/H200 | 20–23, 26, 33–37, 40–45 | ~$3.25/hr median (Sep 2026), < $60 total |
| 3 | 2–8 GPU node | 19, 38, 39 | ~$25/hr, < $100 total |

Roughly two thirds of the book runs at Tier 0–1. Writing budget for
the book's own measurements: a few thousand dollars.

---

# Part I — Why Inference

No code. The reader leaves able to explain why serving is expensive,
why the two phases of a request behave differently, and what the job
is.

## 1. The Cost of a Token

A model is trained once and served for its lifetime; inference is
where the money goes. One request is followed from HTTP to invoice.
Cost per million tokens is introduced as the number the book
optimizes. What an inference engineer does in a week, and what the
role pays.

- Objectives: decompose a request's cost into GPU-seconds; state the
  three quantities that determine cost per token; describe the role.
- Sources: public price sheets and rental rates (Appendix C); Jouppi
  et al., "TPU v4" (ISCA 2023) for the inference-dominates-cost claim.

## 2. What a Model Does When It Answers

Everything about the model that this book needs, from the ground up
and in one chapter: text becomes tokens; tokens become vectors; each
layer lets every token look at the tokens before it (attention) and
then think on its own (the feed-forward block); the last layer scores
every possible next token; one is chosen; it is appended; the loop
repeats. Each step is drawn for a five-token prompt. The chapter ends
with the two facts the rest of the book turns on: producing a token
means reading every weight once, and attention at step *t* needs the
keys and values of every earlier step.

- Objectives: trace one token through the model by hand; state what
  attention needs from earlier tokens; explain why "generate" is a
  loop and what each iteration costs.
- If you're new here: tensors, matrices, and what a "parameter" is.
- Lab: the five-token walkthrough, step by step.
- Sources: Vaswani et al. (2017); *Large Language Models from the
  Ground Up*, Parts I and III, for the full derivation.

## 3. Prefill and Decode

The prompt is read in one parallel pass; the answer is written one
token at a time. Why these are effectively two different programs with
different bottlenecks, and why every optimization in the book targets
one or the other.

- Objectives: identify which phase a given optimization targets;
  explain why TTFT and ITL are governed by different limits.
- Lab: one request moving through both phases.
- Sources: Vaswani et al., "Attention Is All You Need" (2017); Shazeer,
  "Fast Transformer Decoding: One Write-Head is All You Need" (2019).

## 4. The Memory Wall

Weights live in HBM; every decode step re-reads all of them to produce
one token. Bandwidth, not FLOPs, is the limit. Why a $30,000 GPU is
99% idle serving one user, and why batching exists.

- Objectives: compute the decode floor of a model from its size and
  the GPU's bandwidth; explain why more FLOPs do not help decode.
- Sources: Wulf & McKee, "Hitting the Memory Wall" (1995); Pope et
  al., "Efficiently Scaling Transformer Inference" (MLSys 2023).

## 5. Latency, Throughput, and the SLO

TTFT, inter-token latency, throughput, goodput. Why they cannot all be
maximized; p50 versus p99; why the tail is the product. A service
level objective turns engineering into constrained optimization. The
running case study is stated here.

- Objectives: write an SLO for a chat service; explain goodput; read a
  latency-throughput curve and locate the operating point.
- Sources: Dean & Barroso, "The Tail at Scale" (CACM 2013); Beyer et
  al., *Site Reliability Engineering* (2016), ch. 4.

## 6. The Serving Landscape

API versus self-hosting, the three serving engines (vLLM, SGLang,
TensorRT-LLM) and the orchestration layer that runs them at rack scale
(Dynamo, which is not a fourth engine but a tier above them), the
hardware tiers from consumer cards to racks, who runs what and why. A
map of the rest of the book.

- Objectives: place a workload on the API/self-host decision tree;
  name each engine's defining idea; say where an orchestrator sits.

# Part II — Foundations

## 7. Reading the GPU

Streaming multiprocessors, HBM, the memory hierarchy, FLOPs versus
bytes, `nvidia-smi` and what its numbers mean. Renting an H100 for an
hour without surprises.

- Build: a script that reports a GPU's measured (not nominal)
  bandwidth and matmul throughput.
- Sources: NVIDIA H100 whitepaper; CUDA C++ Programming Guide.

## 8. Arithmetic Intensity and the Roofline

Compute the arithmetic intensity of prefill and decode by hand for a
real model; place both on the roofline. The one graph that explains
the book: decode sits far left, bandwidth-bound; batching moves it
right; prefill sits near the ridge.

- Objectives: compute intensity for any layer; predict from the
  roofline whether a change will help; derive the ~150-token ridge
  figure.
- Lab: the interactive roofline, with the model and batch as sliders.
- Sources: Williams, Waterman & Patterson, "Roofline" (CACM 2009).

## 9. Measuring Honestly

Build the benchmark harness the rest of the book depends on: a
closed- and open-loop load generator, warmup, percentiles, fixed
seeds, saturation sweeps, JSON output. The seven ways published
benchmarks mislead — averages, unstated load, cold runs, mismatched
tokenizers, cherry-picked lengths, single runs, missing provenance —
and the harness feature that prevents each.

- Build: `bench/` (Appendix A).
- Sources: Beyer et al., *SRE*, ch. 6; the GenAI-Perf and
  `vllm bench` methodologies as public reference points.

## 10. Your Model on the Bench

Two jobs before Part III starts changing things. First, check that the
hand-written engine is a real transformer: load one set of weights into
`tinyserve` and into a PyTorch implementation of the same architecture,
and compare every score. Second, record the baseline — TTFT,
inter-token latency and throughput — so that every later technique is
measured against a number rather than an impression.

- Build: the PyTorch second opinion, and the baseline results file.
- Objectives: differential-test numerical code; say why a teaching
  implementation is slower without being wrong; trust a teaching
  engine's ratios and distrust its absolute times.
- Sources: McKeeman, "Differential Testing for Software" (1998);
  Goldberg, "What Every Computer Scientist Should Know About
  Floating-Point Arithmetic" (1991).

# Part III — Build an Inference Engine from Scratch

Tier 0–1. Each chapter adds one mechanism and measures the gain.

## 11. The Naive Generate Loop

Recompute the whole sequence every token. Measure it; watch cost grow
quadratically with length; name the waste precisely.

- Numbers: FLOPs per token as a function of position.

## 12. The KV Cache

Cache keys and values; compute only the new token. The largest single
win in the book, built in forty lines. The memory formula:
2 × layers × KV heads × head dim × bytes per token.

- Objectives: implement a KV cache; compute its size for any model;
  explain what MQA/GQA change about it.
- Lab: cache fill-up.
- Sources: Shazeer (2019); Ainslie et al., "GQA" (EMNLP 2023).

## 13. Where the Memory Goes

Preallocated caches waste most of their space: reservation for unknown
lengths, internal fragmentation, and why "out of memory" arrives long
before the GPU is full. Measure the waste on realistic traffic.

- Sources: Kwon et al., "Efficient Memory Management for Large Language
  Model Serving with PagedAttention" (SOSP 2023), §3.

## 14. Paged Attention

Block tables mapping logical to physical KV blocks, exactly as an OS
page table maps memory. Build it; measure recovered capacity and the
batch sizes it unlocks. vLLM's founding idea, in the reader's code.

- Sources: Kwon et al. (SOSP 2023).

## 15. Prefix Caching

Shared system prompts computed once. Hash-based block reuse; a radix
tree of prefixes with LRU eviction; measured hit rates on the case
study's traffic.

- Lab: the prefix tree.
- Sources: Zheng et al., "SGLang: Efficient Execution of Structured
  Language Model Programs" (2024), RadixAttention.

## 16. Batching

Static batches, padding waste, ragged batches. Throughput rises,
latency rises, and Chapter 8's roofline predicts both.

## 17. Continuous Batching

Iteration-level scheduling: requests join and leave the batch at every
step. The scheduler loop, written out. The largest throughput gain
after the KV cache.

- Lab: requests streaming in and out of the batch.
- Sources: Yu et al., "Orca: A Distributed Serving System for
  Transformer-Based Generative Models" (OSDI 2022).

## 18. Chunked Prefill and Scheduling Policies

One long prompt stalls every other user's decode. Split prefill into
chunks and interleave. Priorities, preemption (recompute versus swap),
fairness, and the SLO-aware scheduler.

- Sources: Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM
  Inference with Sarathi-Serve" (OSDI 2024).

## 19. Disaggregated Prefill and Decode

Run the two phases on different GPUs and transfer the KV cache between
them. When it wins, what the transfer costs, and how Dynamo and
Mooncake orchestrate it. Tier 3.

- Sources: Zhong et al., "DistServe" (OSDI 2024); Patel et al.,
  "Splitwise" (ISCA 2024); Qin et al., "Mooncake" (2024).

*Design decision record I: the case study's scheduler and cache
policy, chosen from measurements.*

# Part IV — Faster Kernels (understand, don't write)

Tier 2. The goal is literacy: reading kernel code and profiler output.

## 20. Attention Kernels

FlashAttention: tiling, IO-awareness, why it is exact rather than
approximate, and why it matters more for prefill than decode. Swap it
into `tinyserve` and measure.

- Sources: Dao et al., "FlashAttention" (NeurIPS 2022); Dao,
  "FlashAttention-2" (2023); Shah et al., "FlashAttention-3" (2024).

## 21. Fused Operations and CUDA Graphs

Kernel launch overhead dominates small-batch decode. Fusion,
`torch.compile`, CUDA graph capture. Measure the latency floor drop.

- Sources: NVIDIA CUDA Programming Guide (graphs); PyTorch 2 paper
  (Ansel et al., ASPLOS 2024).

## 22. Precision and Tensor Cores

FP32/BF16/FP16/FP8, what tensor cores accelerate, and the shapes that
make a matmul fast or slow.

- Sources: Micikevicius et al., "FP8 Formats for Deep Learning" (2022).

## 23. Reading a Real Kernel

Walk one production attention kernel (FlashInfer) end to end, mapping
each part to the reader's Python. Profile `tinyserve` with Nsight
Systems and find the hot spot.

- Sources: Ye et al., "FlashInfer" (MLSys 2025).

*Design decision record II: which kernel path the case study uses, and
what it bought.*

# Part V — Smaller Models

## 24. Quantization from the Ground Up

Symmetric and asymmetric int8, scales and zero points, per-tensor
versus per-channel, built by hand on one layer. Error analysis before
any library. Tier 0.

- Sources: Jacob et al., "Quantization and Training of Neural Networks
  for Efficient Integer-Arithmetic-Only Inference" (CVPR 2018).

## 25. Weight-Only Quantization

GPTQ, AWQ, INT4, calibration data, group size. Run each on the 8B
model; measure memory, speed, and quality. Why weight-only helps
decode (bandwidth) and not prefill (compute).

- Sources: Frantar et al., "GPTQ" (ICLR 2023); Lin et al., "AWQ"
  (MLSys 2024); Dettmers et al., "LLM.int8()" (NeurIPS 2022).

## 26. FP8 and Activation Quantization

The H100/B200 default. Activation scaling, outlier channels, KV-cache
quantization, the near-free 2×, and where it breaks.

- Sources: Xiao et al., "SmoothQuant" (ICML 2023); Micikevicius et al.
  (2022).

## 27. Distillation and Pruning

When a smaller model beats a quantized larger one. Distilling to a
student, structured pruning, and the serving economics of each.

- Sources: Hinton et al., "Distilling the Knowledge in a Neural
  Network" (2015); Muralidharan et al., "Minitron" (2024).

## 28. Measuring What You Lost

Evaluate before and after every compression step. The accuracy /
latency / cost frontier drawn from the reader's own numbers, and the
go/no-go decision a team has to make.

*Design decision record III: the case study's model and precision.*

# Part VI — Faster Decoding

## 29. Speculative Decoding

A small draft model proposes; the target model verifies several tokens
in one pass; sampling stays exact. Build it on `tinyserve`. Acceptance
rate math, expected speedup, when it helps and when it hurts.

- Lab: draft-and-verify.
- Sources: Leviathan et al., "Fast Inference from Transformers via
  Speculative Decoding" (ICML 2023); Chen et al., "Accelerating Large
  Language Model Decoding with Speculative Sampling" (2023).

## 30. Self-Speculation and Draft Heads

Medusa and EAGLE-style heads, n-gram and prompt-lookup drafting: speed
without a second model.

- Sources: Cai et al., "Medusa" (ICML 2024); Li et al., "EAGLE" (ICML
  2024).

## 31. Constrained Decoding

JSON and grammar-guided generation, structured outputs, and their real
cost per token. How engines compile grammars to token masks.

- Sources: Willard & Louf, "Efficient Guided Generation for Large
  Language Models" (2023); Dong et al., "XGrammar" (2024).

## 32. Caching Above the Model

Exact-match and semantic caches, measured hit rates on agent and FAQ
traffic, and the failure mode of serving a wrong cached answer.

## 33. Long Context, Reasoning, and MoE

Serving 128K+ prompts; thinking tokens and the new cost curve of
reasoning models; mixture-of-experts serving and expert parallelism.
Where the roofline moves for each.

- Sources: Su et al., "RoFormer" (2021); Liu et al., "Ring Attention"
  (2023); DeepSeek-AI, "DeepSeek-V3" (2024) and "DeepSeek-R1" (2025).

*Design decision record IV: decoding accelerations adopted by the case
study, with the acceptance rates that justified them.*

# Part VII — The Production Stack

Tier 2–3. The reader opens the real engines and recognizes every part.

## 34. vLLM

Architecture tour with `tinyserve` as the map. Every flag traced to a
chapter. Deploy the 8B model and reproduce the baseline numbers.

- Sources: Kwon et al. (SOSP 2023); vLLM documentation at the pinned
  version.

## 35. SGLang

RadixAttention, the frontend language, structured generation, and the
workloads where it beats vLLM.

- Sources: Zheng et al. (2024).

## 36. TensorRT-LLM, and Orchestration with Dynamo

The NVIDIA path: ahead-of-time engine builds and compile-time
optimization. Then the tier above the engine — Dynamo schedules
whichever engine you chose (TensorRT-LLM, vLLM or SGLang) across many
accelerators, and handles disaggregated serving at rack scale. It
replaces none of them.

## 37. One Benchmark, Three Engines

The same model, traffic, and hardware through vLLM, SGLang and
TensorRT-LLM, from the harness. An honest, reproducible comparison and
the decision framework for choosing.

## 38. Multi-GPU Serving

Tensor, pipeline, and expert parallelism; NCCL; when to shard and when
to replicate. Serve a 70B model across one node. Tier 3.

- Sources: Shoeybi et al., "Megatron-LM" (2019); Pope et al. (MLSys
  2023).

## 39. Many Models, Many Tenants

Multi-LoRA serving, adapter routing, model routers, isolation and fair
sharing between tenants.

- Sources: Sheng et al., "S-LoRA" (MLSys 2024); Chen et al., "Punica"
  (MLSys 2024).

## 40. The Serving Layer

The gateway in front of the engine: OpenAI-compatible APIs, streaming,
retries, backpressure, rate limits, timeouts; deployment on Kubernetes;
autoscaling on the right signal (KV-cache pressure, not CPU).

*Design decision record V: the case study's engine, topology, and
gateway.*

# Part VIII — Running Inference as a Business

## 41. Capacity Planning

Model the traffic; size the fleet from the SLO; just enough queueing
theory (Little's law, M/M/c intuition, the tail) to predict p99 before
buying hardware. The case study is sized here.

- Sources: Harchol-Balter, *Performance Modeling and Design of
  Computer Systems* (2013); Beyer et al., *SRE*, ch. 22.

## 42. GPU FinOps

Cost per million tokens from first principles: utilization, reserved
versus spot, the build-versus-API decision with real numbers, and the
break-even curve. The case study is priced here.

## 43. Observability

The metrics that matter (queue depth, KV utilization, TTFT/ITL
percentiles, preemptions), tracing one request end to end,
dashboards, and catching a regression before users do.

- Sources: Beyer et al., *SRE*, ch. 6 and 10.

## 44. Reliability and Incidents

OOMs, stragglers, hot shards, bad rollouts, capacity cliffs. A runbook
per failure and the drills that make them routine. Written as
postmortems.

- Sources: Beyer et al., *SRE*, ch. 14–15; Dean & Barroso (2013).

## 45. Security and Compliance at the Serving Layer

Prompt and PII handling, tenant isolation, data residency, logging
policy, and what an auditor will ask.

## 46. The Inference Engineer

The portfolio (the harness plus three write-ups), the interview
questions mapped chapter by chapter, compensation bands and
negotiation, and the first ninety days on the job.

*Design decision record VI: the case study's final architecture and
its cost per million tokens — the number the capstone must beat.*

# Appendices

- **A. The Benchmark Harness** — every metric defined, reproduction
  instructions, JSON schema.
- **B. Answers** to ★ and ★★ exercises.
- **C. Hardware Cheat Sheet** — A100, H100, H200, B200, B300, MI300X, L40S,
  consumer cards: memory, bandwidth, FLOP/s by precision, rental price
  at revision date.
- **D. Engine Flag Crosswalk** — the same concept in vLLM, SGLang, and
  TensorRT-LLM.
- **E. The Cost Model** — the spreadsheet and formulas behind Part VIII.
- **F. Setting Up** — renting GPUs, Colab, local, containers,
  `environment.lock`.
- **G. For Instructors** — course-fit table, calendars, rubrics for
  every ★★★ exercise.
- **H. Capstone** — serve a given model under a given SLO at minimum
  cost per million tokens; graded by the harness against Chapter 46's
  number.
- **I. Glossary.**

---

# Writing order

The first three chapters were written out of order, on purpose, to
retire the biggest risk first: whether the engine spine works as a
teaching device and whether the measurement pipeline can hold the
standards. It does, so that reason has expired.

**Done:**

1. **The harness and figure pipeline** (Chapter 9's and Appendix A's
   subject matter) — built and in use by every chapter below.
2. **Chapters 12–13** — the spine's first stretch, written first: the
   KV cache and what it costs in memory.
3. **Chapters 1–3** — the opening, written once there were real numbers
   to open with.
4. **Chapters 4–11, 14–18** — front to back from there, closing Part I
   and Part II and carrying Part III through the scheduler: batching,
   continuous batching, and the policies that make it keep a promise.

Drafted so far: **1–18**, every one of them passing `make check`.
Next: **19, Disaggregated Prefill and Decode** — the last chapter of
Part III, and the first that needs more than one accelerator.

**From here, front to back.** The remaining risk is pedagogical rather
than mechanical, and it lives in Part I and the early chapters of
Part III: whether a reader new to the field is carried or lost. That
cannot be judged out of order, because it depends on what the previous
chapter left them holding.

Two exceptions, where a chapter needs a measurement a later chapter
produces: quote it from the results file that will own it, and let
`make check` fail until that chapter's experiment exists. Never write
a placeholder number.

Each Part gets a design review against STANDARDS.md before writing
begins (§8.2).

# Companion assets

- `llm-inference-book` — manuscript, own repository per the series
  pattern, with the review log.
- `FACTS.md` — the facts register, re-verified before each chapter is
  drafted and at each revision (STANDARDS.md §11).
- `llm-inference-code` — `tinyserve/`, `bench/`, `environment.lock`,
  one notebook per code chapter, CI per STANDARDS.md §3.3.
- `books.wazeem.com/llm-inference/` — labs, "what changed in the
  engines since print", errata-as-postmortems.
