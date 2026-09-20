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
