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
| InfiniBand NDR (ConnectX-7) | 400 Gb/s per port = 50 GB/s | NVIDIA ConnectX-7 datasheet; Quantum-2 switch page | 2026-09 |
| Quantum-2 switch | "64 400Gb/s ports or 128 200Gb/s ports", 51.2 Tb/s bidirectional aggregate | NVIDIA Quantum-2 platform page | 2026-09 |

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

## Continuous batching and scheduling

| Claim | Value | Source | Verified |
|---|---|---|---|
| Orca's headline result | "36.9x throughput improvement at the same level of latency" over NVIDIA FasterTransformer, on GPT-3 175B; the paper's two contributions are *iteration-level scheduling* and *selective batching* | Yu, Jeong, Kim, Kim and Chun, OSDI 2022, pp. 521-538 (usenix.org/conference/osdi22/presentation/yu) | 2026-09 |
| vLLM `--scheduling-policy` | "The scheduling policy to use: - 'fcfs' means first come first served, i.e. requests are handled in order of arrival. - 'priority' means requests are handled based on given priority"; default `fcfs` | vLLM docs, `configuration/engine_args` | 2026-09 |
| vLLM `--watermark` | "Fraction of total KV cache blocks to keep free when admitting waiting or preempted requests into running queue"; default `0.0` | same | 2026-09 |
| vLLM `--gpu-memory-utilization` | "The fraction of GPU memory to be used for the model executor, ranging from 0 to 1"; default `0.92` | same | 2026-09 |
| vLLM preemption mode | "In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture." Swap copied blocks to CPU memory; `vllm:num_requests_swapped` is "a legacy metric now deprecated, as CPU swapping is no longer used in V1" | vLLM docs, `configuration/optimization` and `design/metrics` | 2026-09 |
| TensorRT-LLM's name for it | "TensorRT-LLM relies on a component, called the Batch Manager, to support in-flight batching of requests (also known in the community as continuous batching or iteration-level batching)" | NVIDIA/TensorRT-LLM, `docs/source/batch_manager.md` | 2026-09 |
| vLLM preemption warning | "Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode because there is not enough KV cache space." | vLLM docs, `configuration/optimization` | 2026-09 |
| vLLM remedies for frequent preemption | increase `gpu_memory_utilization`; decrease `max_num_seqs` or `max_num_batched_tokens`; increase `tensor_parallel_size` or `pipeline_parallel_size` | same | 2026-09 |
| What a preempted request loses | "It will be re-scheduled in future and re-start its prefill phase"; the request "has been put back in the waiting queue in order to make room for other requests to complete" | vLLM docs, `design/metrics` | 2026-09 |
| vLLM scheduler metrics | `vllm:num_requests_running` ("Number of requests currently running") and `vllm:num_requests_waiting`, both gauges; `vllm:time_to_first_token_seconds` and `vllm:inter_token_latency_seconds`, both histograms. No `vllm:num_preemptions_total` appears in the metrics design doc. | vLLM docs, `design/metrics` | 2026-09 |
| SGLang `--schedule-policy` | "The scheduling policy of the requests"; default `fcfs`; also `lpm`, `random`, `dfs-weight`, `lof`, `priority`, `routing-key` | SGLang docs, `advanced_features/server_arguments` | 2026-09 |
| SGLang `--schedule-conservativeness` | "How conservative the schedule policy is. A larger value means more conservative scheduling. Use a larger value if you see requests being retracted frequently."; default `1.0` | same | 2026-09 |
| SGLang `--retraction-policy` | chooses which requests are removed when the KV cache fills: `length` prefers retracting requests with shorter outputs, `priority` retracts lower-priority ones first | same | 2026-09 |

## Chunked prefill and scheduling policies

| Claim | Value | Source | Verified |
|---|---|---|---|
| Sarathi-Serve's mechanism | "Sarathi-Serve introduces chunked-prefills which splits a prefill request into near equal sized chunks" and "creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes" | Agrawal, Kedia, Panwar, Mohan, Kwatra, Gulavani, Tumanov and Ramjee, OSDI 2024, pp. 117-134 (usenix.org/conference/osdi24/presentation/agrawal) | 2026-09 |
| Sarathi-Serve's headline result | "For Mistral-7B on single A100 GPUs, we achieve 2.6x higher serving capacity and up to 3.7x higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM"; "up to 5.6x gain in the end-to-end serving capacity" with pipeline parallelism on Falcon-180B | same | 2026-09 |
| vLLM chunked prefill default | "In V1, chunked prefill is enabled by default whenever possible." | vLLM docs, `configuration/optimization` | 2026-09 |
| vLLM's scheduling order | the policy "prioritizes decode requests": it "batches all pending decode requests before scheduling any prefill operations", then fills the remaining token budget with prefills, chunking those that do not fit | same | 2026-09 |
| The token-budget trade-off | "Smaller values (e.g., 2048) achieve better ITL because there are fewer prefills slowing down decodes"; "Higher values achieve better time to first token (TTFT) as you can process more prefill tokens in a batch"; "For optimal throughput, we recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs" | same | 2026-09 |

## Disaggregated prefill and decode

| Claim | Value | Source | Verified |
|---|---|---|---|
| DistServe | Zhong, Liu, Chen, Hu, Zhu, Liu, Jin and Zhang, "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving", OSDI 2024, pp. 193-210. Serves "7.4x more requests or 12.6x tighter SLO" than state-of-the-art systems while meeting latency constraints for over 90% of requests; colocating the phases "leads to strong prefill-decoding interferences"; it "places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation" | usenix.org/conference/osdi24/presentation/zhong-yinmin | 2026-09 |
| Splitwise | Patel, Choukse, Zhang, Shah, Goiri, Maleki and Bianchini, "Splitwise: Efficient Generative LLM Inference Using Phase Splitting", ISCA 2024, doi:10.1109/ISCA59077.2024.00019. Splits the two phases onto separate machines so each runs on hardware suited to it; reports 1.4x higher throughput at 20% lower cost, or 2.35x better throughput at the same cost and power | Microsoft Research publication page; ACM DL | 2026-09 |
| Mooncake | Qin, Li, He, Cui, Ren, Zhang, Wu, Zheng and Xu, "Mooncake: Trading More Storage for Less Computation - A KVCache-centric Architecture for Serving LLM Chatbot", FAST 2025, pp. 155-170 (Best Paper). "Increases the effective request capacity by 59%~498% when compared to baseline methods, all while complying with SLOs"; in production Kimi handles "115% and 107% more requests on NVIDIA A800 and H800 clusters". The arXiv version (2407.00079) reports "up to a 525% increase in throughput in certain simulated scenarios" and "75% more requests" under real workloads | usenix.org/conference/fast25/presentation/qin; arxiv.org/abs/2407.00079 | 2026-09 |
| vLLM disaggregated prefilling | "This feature is experimental and subject to change." Configured with `--kv-transfer-config`; connectors include NixlConnector, LMCacheConnectorV1 and MooncakeConnector. Benefits given: "Tuning time-to-first-token (TTFT) and inter-token-latency (ITL) separately" and "Controlling tail ITL" by keeping prefill from interrupting decode. **"Disaggregated prefill DOES NOT improve throughput."** | vLLM docs, `features/disagg_prefill` | 2026-09 |
| SGLang PD disaggregation | `--disaggregation-mode` takes `prefill` or `decode`; `--disaggregation-transfer-backend` takes `mooncake` (default), `nixl` or `ascend`; `--disaggregation-ib-device` names the InfiniBand device. The router takes `--pd-disaggregation`, `--prefill [url]`, `--decode [url]`. Motivated by "Prefill Interruption" and "DP Attention Imbalance" | SGLang docs, `advanced_features/pd_disaggregation` | 2026-09 |
| NVIDIA Dynamo | "Dynamo leverages NIXL to transfer KV cache directly from the VRAM of the prefill engine to the VRAM of the decode engine"; "The KV transfer is non-blocking, allowing GPU forward passes to continue serving other requests during the transfer". Suggests "a larger TP for the memory-bound decoding phase while a smaller TP for the computation-bound prefill phase" | NVIDIA Dynamo docs, `design-docs/disaggregated-serving` | 2026-09 |

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

## Attention kernels

| Fact | Value | Source | Checked |
|---|---|---|---|
| H100 shared memory per SM | 228 KB (A100: 164 KB) | NVIDIA Hopper Tuning Guide, release 13.3, §1 | 2026-09-20 |
| H100 max shared memory per thread block | 227 KB; "CUDA reserves 1 KB of shared memory per thread block" | NVIDIA Hopper Tuning Guide 13.3 | 2026-09-20 |
| H100 shared memory carveouts | "0, 8, 16, 32, 64, 100, 132, 164, 196 and 228 KB per SM" | NVIDIA Hopper Tuning Guide 13.3 | 2026-09-20 |
| Static shared memory allocation limit | 48 KB, with explicit opt-in required above it | NVIDIA Hopper Tuning Guide 13.3 | 2026-09-20 |
| FlashAttention is exact | "an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM" | Dao, Fu, Ermon, Rudra, Ré, arXiv:2205.14135, abstract | 2026-09-20 |
| FlashAttention-1 IO complexity | "requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes" | arXiv:2205.14135, abstract | 2026-09-20 |
| FlashAttention-1 speedups | 15% end-to-end on BERT-large (seq 512) over the MLPerf 1.1 record; 3x on GPT-2 (seq 1K) | arXiv:2205.14135, abstract | 2026-09-20 |
| FlashAttention-3 speedup | "speedup on H100 GPUs by 1.5-2.0x with FP16" | Shah, Bikshandi, Zhang, Thakkar, Ramani, Dao, arXiv:2407.08608, abstract | 2026-09-20 |
| FlashAttention-3 FP16 throughput | "reaching up to 740 TFLOPs/s (75% utilization)" | arXiv:2407.08608, abstract | 2026-09-20 |
| FlashAttention-3 FP8 throughput | "close to 1.2 PFLOPs/s"; "2.6x lower numerical error than a baseline FP8 attention" | arXiv:2407.08608, abstract | 2026-09-20 |
| FlashAttention-2 utilization on H100 | "only 35% utilization on the H100 GPU" | arXiv:2407.08608, abstract | 2026-09-20 |
| vLLM backend selection | "vLLM iterates through backends in priority order"; "The first compatible backend is selected" | vLLM docs, Attention Backend Feature Support | 2026-09-20 |
| vLLM backend override flags | `--attention-backend`, or `--attention-config.backend` / `-ac.backend` (mutually exclusive) | vLLM docs, Attention Backend Feature Support | 2026-09-20 |
| vLLM FlashAttention version default | "Default is FA4 on SM100+ (Blackwell), FA3 on SM90 (Hopper), FA2 otherwise"; override `--attention-config.flash_attn_version` | vLLM docs, Attention Backend Feature Support | 2026-09-20 |
| A100 memory hierarchy | "HBM: 1.5 TB/s (40 GB)" against "SRAM: 19 TB/s (20 MB)" -- about 13x the bandwidth, aggregate across all multiprocessors | Dao et al., arXiv:2205.14135, Figure 1 (left) | 2026-09-20 |
| Online softmax | The running-maximum rescaling FlashAttention's one pass depends on | Milakov and Gimelshein, "Online normalizer calculation for softmax", arXiv:1805.02867 | 2026-09-20 |
