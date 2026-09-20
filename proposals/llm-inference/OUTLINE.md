# LLM Inference from the Ground Up

**Subtitle:** Serve Models Fast and Cheap — The Path to Inference Engineering

**Series:** the third "from the Ground Up" book. Book 1 built the model;
Book 2 wielded the agent; this book serves the model at production speed
and cost. It picks up exactly where *Large Language Models from the
Ground Up* ends (Chapter 30, "Sampling & beyond").

## Who it is for

- Readers of Book 1 who have a working GPT and want the next job.
- Backend and ML engineers who call an API today and want to own the
  serving layer tomorrow.
- Anyone targeting an inference / GPU performance / ML platform role.

Prerequisites: Python, basic PyTorch, and either Book 1 or an equivalent
understanding of attention and autoregressive decoding. No CUDA required;
Part IV explains kernels, it does not ask the reader to write them.

## The spine: build your own engine, then read the real ones

As Book 1 went micrograd → GPT, this book goes naive loop → engine.
Parts II–VI grow one small Python engine (working name **`tinyserve`**)
from a recompute-everything generate loop into one with a paged KV cache,
continuous batching, chunked prefill, quantized weights, and speculative
decoding. Every technique is built, measured, and only then named. In
Part VII the reader opens vLLM and SGLang and recognizes every component.

Two running models: Book 1's own 825K-parameter GPT (fits anywhere, every
mechanism visible) and a Qwen/Llama-class open model in the 1B–8B range
(real numbers). Every measurement in the book comes from the companion
**benchmark harness** (Appendix A), which a reader reproduces on one
rented GPU in under an hour. The harness is the book's proof and moat.

## The compute ladder

| Tier | Runs on | Parts |
|------|---------|-------|
| 0 | CPU / free Colab | I, II, III (825K GPT) |
| 1 | Colab T4 or one consumer GPU | III (1B model), V, VI |
| 2 | One rented H100/H200 (~$3/hr) | IV, VI (FP8), VII, VIII |
| 3 | 2–8 GPU node, a few hours total | Ch 18, 37, 38 only |

Tiers 0–1 cover roughly two thirds of the book. Total GPU budget to
write it: a few thousand dollars; to read it: under $100.

Conventions carried over from Book 1: ★/★★/★★★ exercises with ★ and ★★
answers in Appendix B, "Watch it move" boxes pointing at browser labs,
one Colab notebook per code chapter, and a course-fit appendix.

---

## Part I — Why Inference (no code)

The reader leaves Part I able to explain, in plain words, why serving
is expensive, why the two phases of a request behave differently, and
what the job is.

### 1. The Cost of a Token
A model is trained once and served forever; inference is most of its
lifetime cost. Follow one request from HTTP to bill. Cost per million
tokens as the number the whole book optimizes. What an inference
engineer does in a week, and what the role pays.

### 2. Prefill and Decode
The prompt is read all at once; the answer is written one token at a
time. Why these are effectively two different programs with different
bottlenecks, and why every optimization in the book targets one or the
other. *Lab: watch one request move through both phases.*

### 3. The Memory Wall
Weights live in HBM; every decode step re-reads all of them to produce
one token. Bandwidth, not FLOPs, is the limit. Why a $30,000 GPU sits
mostly idle serving one user, and why batching exists.

### 4. Latency, Throughput, and the SLO
Time to first token, inter-token latency, tokens per second, goodput.
Why you cannot maximize all of them, p50 vs p99, and how a service level
objective turns engineering into a constrained optimization.

### 5. The Serving Landscape
API vs self-hosting, the four engines (vLLM, SGLang, TensorRT-LLM,
Dynamo), the hardware tiers, and who runs what. A map of the rest of
the book.

## Part II — Foundations

### 6. Reading the GPU
SMs, HBM, the memory hierarchy, FLOPs versus bytes. `nvidia-smi` and
what its numbers mean. How to rent an H100 for an hour without being
surprised by the bill. *Build: a script that reports what a GPU can do.*

### 7. Arithmetic Intensity and the Roofline
Compute the intensity of prefill and decode by hand for a real model
and place both on a roofline. The one graph that explains the whole
book: decode is bandwidth-bound, prefill is compute-bound, and batching
moves decode to the right. *Lab: the interactive roofline.*

### 8. Measuring Honestly
Build the benchmark harness: load generator, warmup, percentiles,
fixed seeds, saturation curves. The seven ways published benchmarks
lie (and how to avoid each). This harness is used in every later
chapter. *Build: `bench/`.*

### 9. Your Model on the Bench
Load the 825K GPT and a 1B open model through plain PyTorch. First
measurements: TTFT, ITL, throughput at batch 1. These baselines are
the "before" for every number that follows.

## Part III — Build an Inference Engine from Scratch

Tier 0–1. Each chapter adds one mechanism to `tinyserve` and measures
the gain against the harness.

### 10. The Naive Generate Loop
Recompute the whole sequence every token. Measure it. Watch the cost
grow with length. Name the waste.

### 11. The KV Cache
Cache keys and values; compute only the new token. The single largest
win in the book, built in forty lines. Memory math: bytes per token per
layer, and what that means at 8K context. *Lab: KV cache fill-up.*

### 12. Where the Memory Goes
Preallocated caches waste most of their space: internal fragmentation,
reservation for unknown lengths, and why "out of memory" arrives long
before the GPU is full. Measure the waste.

### 13. Paged Attention
Virtual-to-physical block tables for the KV cache, exactly like an OS
page table. Build it; measure recovered capacity and the batch sizes it
unlocks. This is vLLM's founding idea, in the reader's own code.

### 14. Prefix Caching
Shared system prompts computed once. Hashing blocks, a radix tree of
prefixes (SGLang's RadixAttention), eviction policy, measured hit
rates on realistic traffic. *Lab: the prefix tree.*

### 15. Batching
Static batches, padding waste, ragged batches. Throughput rises,
latency rises, and the roofline from Chapter 7 explains exactly why.

### 16. Continuous Batching
Iteration-level scheduling: requests join and leave the batch every
step. The scheduler loop, written out. The largest throughput jump
after the KV cache. *Lab: watch requests stream in and out.*

### 17. Chunked Prefill and Scheduling Policies
A long prompt stalls every other user's decode. Split prefill into
chunks and interleave. Priorities, preemption, swapping, fairness, and
the SLO-aware scheduler.

### 18. Disaggregated Prefill and Decode
Run the two phases on different GPUs and ship the KV cache between
them. When it wins, what it costs, and how Dynamo does it. (Tier 3.)

## Part IV — Faster Kernels (understand, don't write)

### 19. Attention Kernels
FlashAttention: tiling, IO awareness, why it is exact rather than
approximate, and why it matters more for prefill than decode. Swap it
into `tinyserve` and measure.

### 20. Fused Operations and CUDA Graphs
Kernel launch overhead dominates small-batch decode. Fusion,
`torch.compile`, CUDA graph capture. Measure the latency floor drop.

### 21. Precision and Tensor Cores
FP32/FP16/BF16, what tensor cores actually accelerate, and the shapes
that make a matmul fast or slow.

### 22. Reading a Real Kernel
Walk through one production attention kernel (FlashInfer) and map each
part to the reader's Python. The goal is literacy: reading kernel code
and profiler output, not writing CUDA. *Build: profile `tinyserve` with
Nsight and find the hot spot.*

## Part V — Smaller Models

### 23. Quantization from the Ground Up
Symmetric and asymmetric int8, scales and zero points, per-tensor vs
per-channel, built by hand on a single layer. Error analysis before any
library.

### 24. Weight-Only Quantization
GPTQ, AWQ, INT4, calibration data, group size. Run each on the 8B
model; measure memory, speed, and quality. Why weight-only helps
decode (bandwidth) and not prefill.

### 25. FP8 and Activation Quantization
The H100/B200 default. Activation scaling, KV-cache quantization, and
the near-free 2x. Where it breaks.

### 26. Distillation and Pruning
When a smaller model beats a quantized big one. Distilling onto a
student, structured pruning, and the serving economics of each.

### 27. Measuring What You Lost
Evaluate before and after every compression step. The accuracy /
latency / cost frontier drawn from the reader's own numbers, and the
go/no-go decision a team actually has to make.

## Part VI — Faster Decoding

### 28. Speculative Decoding
A small draft model guesses; the big model verifies several tokens in
one pass. Build it on `tinyserve`. Acceptance rate math, when it helps
and when it hurts. *Lab: draft-and-verify.*

### 29. Self-Speculation and Draft Heads
Medusa/EAGLE-style heads, n-gram and prompt-lookup drafting: speed
without a second model.

### 30. Constrained Decoding
JSON and grammar-guided generation, structured outputs, and their real
cost per token. How engines compile grammars to token masks.

### 31. Caching Above the Model
Exact-match and semantic caches, measured hit rates on agent and FAQ
traffic, and the failure mode of serving a wrong cached answer.

### 32. Long Context, Reasoning, and MoE
Serving 128K+ prompts, thinking tokens and the new cost curve of
reasoning models, and the basics of mixture-of-experts serving. Where
the roofline moves for each.

## Part VII — The Production Stack

Tier 2. The reader now opens the real engines and recognizes every part.

### 33. vLLM
Architecture tour with `tinyserve` as the map. Every flag traced back
to a chapter. Deploy the 8B model and hit the baseline numbers.

### 34. SGLang
RadixAttention, the frontend language, structured generation, and the
workloads where it beats vLLM.

### 35. TensorRT-LLM and Dynamo
The NVIDIA path: compile-time optimization, engine builds, and
disaggregated serving at rack scale.

### 36. One Benchmark, Four Engines
The same model, traffic, and hardware through all four engines. An
honest, reproducible comparison table, and the decision framework for
choosing one.

### 37. Multi-GPU Serving
Tensor, pipeline, and expert parallelism; NCCL; when to shard and when
to replicate. Serve a 70B model across a node. (Tier 3.)

### 38. Many Models, Many Tenants
Multi-LoRA serving, adapter routing, model routers, isolation and fair
sharing between tenants.

### 39. The Serving Layer
The gateway in front of the engine: OpenAI-compatible APIs, streaming,
retries, backpressure, rate limits, timeouts, deployment on Kubernetes,
autoscaling on the right signal.

## Part VIII — Running Inference as a Business

### 40. Capacity Planning
Model the traffic, size the fleet from the SLO, and just enough
queueing theory to predict p99 before buying hardware.

### 41. GPU FinOps
Cost per million tokens from first principles: utilization, reserved
vs spot, the build-vs-API decision with real numbers, and the
break-even curve.

### 42. Observability
The metrics that matter, tracing one request end to end, dashboards,
and catching a regression before users do.

### 43. Reliability and Incidents
OOMs, stragglers, hot shards, bad rollouts, capacity cliffs. A runbook
per failure, and the drills that make it routine.

### 44. Security and Compliance at the Serving Layer
Prompt and PII handling, tenant isolation, data residency, logging
policy, and what an auditor will ask.

### 45. Becoming an Inference Engineer
The portfolio (the harness plus three write-ups), the interview
questions mapped chapter by chapter, salary bands and how to negotiate,
and the first ninety days on the job.

## Appendices

- **A. The Benchmark Harness** — reference for `bench/`, every metric
  defined, reproduction instructions.
- **B. Answers** to ★ and ★★ exercises.
- **C. Hardware Cheat Sheet** — H100, H200, B200, MI300X, L40S, consumer
  cards: memory, bandwidth, FLOPs by precision, price per hour.
- **D. Engine Flag Crosswalk** — the same concept in vLLM, SGLang, and
  TensorRT-LLM.
- **E. The Cost Model** — the spreadsheet and formulas behind Part VIII.
- **F. Setting Up** — renting GPUs, Colab, local, and containers.
- **G. For Instructors** — course-fit table, calendars, rubrics.
- **H. Capstone** — serve a given model under a given SLO at minimum
  cost per million tokens; graded by the harness.
- **I. Glossary.**

---

## Suggested writing order

1. **Chapter 8 and Appendix A first.** The harness is what every other
   chapter measures with; write it once and every number in the book
   comes from it.
2. **Part III** (the spine). If the engine works and the gains are
   measurable, the book works.
3. **Parts I–II**, now that there is something concrete to point at.
4. **Parts V–VI**, which extend the engine.
5. **Part VII**, then **IV**, then **VIII** and Chapter 45.

## Companion assets

- `llm-inference-book` (manuscript, own repo, per the series pattern)
- `llm-inference-code`: `tinyserve/`, `bench/`, one notebook per code
  chapter
- Companion page at `books.wazeem.com/llm-inference/` with the labs, a
  "what changed in the engines since print" page, and errata
