# Facts Register

Every time-sensitive claim used in the manuscript, per STANDARDS.md
§11. Verify each entry against its primary source before drafting a
chapter that uses it; update the date. `UNVERIFIED` entries may not
appear in the text.

Format: claim · value · source · last verified.

## Hardware

| Claim | Value | Source | Verified |
|---|---|---|---|
| H100 SXM HBM3 | 80 GB, 3.35 TB/s | NVIDIA H100 datasheet | 2026-09 |
| H100 BF16 / FP8 dense peak | ~990 / ~1,980 TFLOP/s | NVIDIA H100 datasheet | 2026-09 |
| H200 HBM3e | 141 GB, 4.8 TB/s | NVIDIA H200 datasheet | 2026-09 |
| B200 HBM3e | 180–192 GB (SKU-dependent), ~8 TB/s; ~4,500 TFLOP/s dense FP8 | NVIDIA Blackwell datasheet; provider spec pages | 2026-09 |
| B300 (Blackwell Ultra) HBM3e | 288 GB, ~8 TB/s; ~15 PFLOP/s dense FP4 | provider spec pages; SemiAnalysis InferenceX | 2026-09 |
| NVLink (H100) per GPU | 900 GB/s | NVIDIA H100 datasheet | 2026-09 |
| PCIe 5.0 x16 | ~64 GB/s per direction | PCI-SIG | stable |

## Prices (on-demand, per GPU-hour)

| Claim | Value | Source | Verified |
|---|---|---|---|
| H100 on-demand median | $3.25; range $1.49 (marketplace) – $6.98 (hyperscaler) | GPU price indexes (aimultiple, gpuperhour, IntuitionLabs) | 2026-09 |
| B200 on-demand median | $6.52; specialist providers $4–6 | same | 2026-09 |
| B300 on-demand median | $7.87 | same | 2026-09 |

## Engines

| Claim | Value | Source | Verified |
|---|---|---|---|
| vLLM stable release | v0.28.0 (2026-08-26); Rust frontend available | github.com/vllm-project/vllm/releases | 2026-09 |
| SGLang release model | stable tags lag; new model support lands in nightlies (0.5.21.dev, 2026-09-20) | github.com/sgl-project/sglang, sgl-project/whl | 2026-09 |
| TensorRT-LLM | last stable 2026-04-20; pre-release 1.3.0rc27 (2026-09-17). Since 1.0 the PyTorch architecture is the default and the LLM API is stable | NVIDIA release notes; PyPI | 2026-09 |
| NVIDIA Dynamo | **not a serving engine.** An orchestration layer that schedules other backends (TensorRT-LLM, vLLM, SGLang, PyTorch) across many accelerators and handles disaggregated serving | NVIDIA product pages and newsroom | 2026-09 |

## Models

| Claim | Value | Source | Verified |
|---|---|---|---|
| Current Qwen generation | Qwen3.8; the open-weight dense release is 27B (Apache 2.0, 2026-08-14). No 4B/8B/14B in this generation; small sizes are in Qwen3.5/3.6 | Hugging Face model cards; Qwen release notes | 2026-09 |
| Current small open-weight families | Gemma 4 (sub-16 GB class), GLM-5.3-Flash, Qwen3.6-27B; six open-weight releases 2026-08-10 to 08-29 | Hugging Face blog roundups | 2026-09 |
| **Running model (8B class)** | UNVERIFIED — decide at Chapter 10 per STANDARDS §11.5: dense, ~8B, permissive license, current generation, ~1B sibling. Candidates to evaluate: Gemma 4 small sizes, Qwen3.6 small sizes, GLM-5.3-Flash | model cards | — |
| Reference architecture for "numbers to know" | Llama-3-style 8B: 32 layers, 8 KV heads, head dim 128 | Llama 3 model card | stable |

## Derived numbers (recomputed by the harness; inputs above)

| Claim | Value | Depends on |
|---|---|---|
| 8B BF16 weights | 16 GB | parameter count |
| Decode floor, 8B, batch 1, H100 | 16 GB / 3.35 TB/s ≈ 4.8 ms/token ≈ 210 tok/s | H100 bandwidth |
| KV cache per token, reference 8B, BF16 | 2 × 32 × 8 × 128 × 2 B = 128 KiB | architecture |
| KV cache per 8K sequence | 1 GiB | above |
| H100 BF16 ridge point | ~990e12 / 3.35e12 ≈ 300 FLOP/byte | H100 specs |
| Tokens in flight to reach the ridge, 8B | ≈ 300 / 2 ≈ 150 | ridge, 2 FLOP/param/token |
| Prefill, 8B, 1,000 tokens, ideal | 2 × 8e9 × 1e3 = 16 TFLOP ≈ 16 ms at peak | H100 BF16 peak |

## API prices (per million tokens, cheapest tracked provider)

| Claim | Value | Source | Verified |
|---|---|---|---|
| Hosted Llama-3.1-8B | $0.02 input / $0.05 output | published price-comparison indexes | 2026-09 |
| Open-weight floor, direct-stack hosts | ~$0.02-$0.20 per 1M input | same | 2026-09 |
| Same open weights across providers | differ severalfold on hardware, batching, quantization and margin | same | 2026-09 |
| Output vs input price | output typically 2-5x input | same | 2026-09 |

## Compensation

| Claim | Value | Source | Verified |
|---|---|---|---|
| Inference / GPU specialists | $300K-$500K+ total comp | 2026 compensation surveys and salary guides | 2026-09 |
| LLM fine-tuning and inference, broader | $220K-$350K total comp | same | 2026-09 |
| CUDA / GPU optimization | $300K-$500K+ | same | 2026-09 |

## Monitoring

| Claim | Value | Source | Verified |
|---|---|---|---|
| `nvidia-smi` GPU-Util | fraction of *time* one or more kernels ran, not fraction of the machine used | NVIDIA docs and analyses | 2026-09 |
| Consequence | a kernel on 1 SM of an H100 reports 100% while SM efficiency is 1/132 = 0.7% | same | 2026-09 |
| H100 streaming multiprocessors | 132 | NVIDIA H100 whitepaper | 2026-09 |
| Better metrics | `DCGM_FI_PROF_SM_ACTIVE` (cycles with a warp resident), `DCGM_FI_PROF_SM_OCCUPANCY` | NVIDIA DCGM docs | 2026-09 |

## Batching

| Claim | Value | Source | Verified |
|---|---|---|---|
| vLLM `--max-num-seqs` | maximum sequences processed in a single iteration | vLLM docs, `configuration/engine_args` | 2026-09 |
| vLLM `--max-num-batched-tokens` | maximum total tokens, summed across sequences, in one iteration; the batch is measured in tokens per iteration, not sequences | same | 2026-09 |
| SGLang `--max-running-requests` | "The maximum number of running requests"; default `None` (derived from the memory pool) | SGLang docs, `advanced_features/server_arguments` | 2026-09 |
| SGLang `--max-prefill-tokens` | "The maximum number of tokens in a prefill batch"; default 16,384 | same | 2026-09 |
| SGLang `--chunked-prefill-size` | chunk size for chunked prefill; -1 disables it; default `None` | same | 2026-09 |

## Prefix caching

| Claim | Value | Source | Verified |
|---|---|---|---|
| vLLM automatic prefix caching | on by default on the V1 engine; `--no-enable-prefix-caching` turns it off | vLLM docs, `features/automatic_prefix_caching` and `usage/v1_guide` | 2026-09 |
| vLLM block hashing | "we hash each kv-cache block by the tokens in the block and the tokens in the prefix before the block"; the parent block's hash is a component; "We only cache full blocks" | vLLM design doc, `design/prefix_caching` | 2026-09 |
| vLLM eviction | LRU over a free queue; a finished request's blocks are appended "in the *reverse* order", because "the last block of a request must hash more tokens and is less likely to be reused by other requests. As a result, it should be evicted first" | same | 2026-09 |
| vLLM block hash algorithm | "As of v0.11, the default hashing algorithm is `sha256`"; `--prefix-caching-hash-algo` also takes `sha256_cbor`, `xxhash`, `xxhash_cbor` | vLLM design doc, `design/prefix_caching` | 2026-09 |
| Why the hash matters | a non-cryptographic hash "theoretically increases the risk of hash collisions, which can cause undefined behavior or even leak private information in multi-tenant environments" | same | 2026-09 |
| SGLang radix cache | on unless `--disable-radix-cache` is passed; `--radix-eviction-policy` defaults to `lru` and also accepts `lfu`, `slru`, `priority` | SGLang docs, `advanced_features/server_arguments` | 2026-09 |
| vLLM prefix-cache metrics | `vllm:prefix_cache_queries` and `vllm:prefix_cache_hits`, both counted in **tokens**: queries rises by the prompt's token count, hits by the tokens found cached | vLLM docs, `design/metrics` | 2026-09 |
| Anthropic prompt caching price | cache write 1.25x base input (5-minute TTL) or 2x (1-hour); cache read 0.1x base input | platform.claude.com, `build-with-claude/prompt-caching` | 2026-09 |
| Anthropic minimum cacheable prompt | 512-4,096 tokens depending on the model; shorter prompts are silently not cached | same | 2026-09 |
| OpenAI prompt caching | "enabled by default for supported OpenAI models"; applies from 1,024 tokens; cached input at 0.1x the uncached rate on GPT-5.6 and later; entries live about 30 minutes after last use | OpenAI docs, `guides/prompt-caching` | 2026-09 |

## Build versus buy

| Claim | Value | Source | Verified |
|---|---|---|---|
| Utilization needed for self-hosting to pay | ~60%+; at 10% utilization every self-hosted figure is ~10x worse | 2026 cost-analysis write-ups | 2026-09 |
| All-in cost versus raw GPU bill | engineering and ops run 3-5x the hardware cost | same | 2026-09 |
| Break-even vs frontier closed APIs | roughly 2-5M tokens/day on reserved GPUs | same | 2026-09 |
| Break-even vs cheap open-model APIs | often 50M+ tokens/day | same | 2026-09 |

## Books and competing titles

| Claim | Value | Verified |
|---|---|---|
| Inference-specific books in print | *vLLM and the Engineering of Fast LLM Inference* (Leanpub, 2026-09); *The SGLang Production Handbook* (Leanpub, 2026-09); *Hands-On LLM Serving and Optimization* | 2026-09 |
