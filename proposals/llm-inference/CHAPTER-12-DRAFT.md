# 12. The KV Cache

*Sample chapter, written to [STANDARDS.md](STANDARDS.md). This file is
generated: every number in it is resolved from
`code/results/ch12.json`, and every table is inserted from
`code/tables/`. Edit `chapters/ch12.md`, then run `make ch12` in
`code/` to re-measure and re-render. `make check` fails if this file
has drifted from the measurements.*

**Depends on:** Chapter 2 (what the model does), Chapter 9 (the
harness), Chapter 11 (the naive loop).
**Tier 0** — runs on a laptop CPU, no GPU, free.

## Objectives

By the end of this chapter you can:

1. Explain what the model recomputes when it generates without a cache,
   and why that cost grows with every token.
2. Implement a KV cache, and show that it changes speed without
   changing a single output token.
3. Compute the memory a KV cache needs for any model, from its
   architecture.
4. Explain why the measured speedup is smaller than the arithmetic
   predicts — the first appearance of the idea the rest of the book
   turns on.

## Why it matters

In Chapter 11 you built a generate loop that works, and it was slow in
a way that got worse as it went. Generating 256 tokens from a
128-token prompt took **11.2 seconds**, and the last
token took about **2.94 times longer** than the first.
Nothing about the model changed during that run. The model was doing
the same work over and over.

This chapter removes that repeated work. On the same machine, the same
run drops to **0.16 seconds**. No output token changes.

That is the largest single speedup in this book, and it comes from one
observation about attention that you can see by hand.

> **If you're new here: what "keys" and "values" are**
>
> Chapter 2 introduced attention as each token *looking at* the tokens
> before it. Mechanically, every token produces three vectors: a
> **query** (what am I looking for?), a **key** (what do I offer to
> anyone looking?), and a **value** (what do I hand over when someone
> looks at me?). A token attends to an earlier token by comparing its
> query against that token's key, and collecting that token's value in
> proportion to the match.
>
> The only thing you need from that for this chapter: to compute the
> next token, the model needs the **keys and values of every token
> before it** — but it does *not* need their queries. That asymmetry is
> the whole idea.

## The observation

Watch what happens across two consecutive steps of the loop from
Chapter 11.

You have a 4-token prompt and the model has just produced a fifth
token. To produce the sixth, Chapter 11's loop runs the model over all
five tokens. But tokens 1 through 4 have not changed. Their keys and
values are bit-for-bit what they were a moment ago, because each
token's key and value depend only on that token and the ones before
it — never on anything that comes after.

```
step 5:  [t1 t2 t3 t4]                 -> compute K,V for t1..t4   (4 tokens)
step 6:  [t1 t2 t3 t4 t5]              -> compute K,V for t1..t5   (5 tokens)
                                            ^^^^^^^^^^^
                                            4 of these 5 are identical
                                            to what you computed last step
step 7:  [t1 t2 t3 t4 t5 t6]           -> compute K,V for t1..t6   (6 tokens)
                                            5 of 6 already computed
```

By step 256, the model recomputes hundreds of tokens' worth of
keys and values to produce one new token. That is the shape of the
curve you measured in Chapter 11.

So: compute each token's key and value **once**, keep them, and reuse
them. That store is the **KV cache**, and the rest of this chapter is
that idea, built and measured.

## Build

The cache is two arrays per layer — one for keys, one for values —
plus a count of how many positions are filled.

<!-- abridged: tinyserve/model.py -->

```python
class KVCache:
    def __init__(self, cfg: Config, max_seq: int | None = None) -> None:
        self.cfg = cfg
        self.max_seq = max_seq or cfg.max_seq
        shape = (cfg.n_kv_heads, self.max_seq, cfg.head_dim)
        self.k = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self.v = [np.zeros(shape, dtype=DType) for _ in range(cfg.n_layers)]
        self.length = 0
```

The cache stores what it is given and hands back everything attention
should read:

<!-- listing: tinyserve/model.py KVCache.append no-docstring -->

```python
def append(self, layer: int, k: np.ndarray, v: np.ndarray,
           start: int) -> tuple[np.ndarray, np.ndarray]:
    total = start + k.shape[1]
    self.k[layer][:, start:total] = k
    self.v[layer][:, start:total] = v
    return self.k[layer][:, :total], self.v[layer][:, :total]
```

The model's forward pass then needs three changes. It takes only the
**new** tokens rather than the whole sequence; it asks the cache where
they go; and it attends over everything the cache returns.

<!-- abridged: tinyserve/model.py -->

```python
    start = cache.length if cache is not None else 0
    ...
        if cache is not None:
            k, v = cache.append(i, k, v, start)
```

That the model calls a method rather than writing into arrays is not
tidiness. It is the seam that lets Chapter 14 replace the
storage entirely without touching a line of the model.

Two details are worth pausing on, because they are the reason this
works rather than incidental plumbing.

**Positions come from the cache, not from the input.** A token's
position in the sequence used to be its index in the array you passed
in. Now you pass one token and it is at position `start`. Get this
wrong and the model still runs, still produces fluent-looking text, and
is quietly wrong — the failure mode you will meet again in Chapter 14.

**One function serves both phases.** With many new tokens and an empty
cache, `forward` is a prefill. With one new token and a full cache, it
is a decode step. This is not a convenience; it is what prefill and
decode actually are, and Chapter 19 will split them across different
machines on exactly this seam.

Generating now has two phases instead of one loop:

<!-- abridged: tinyserve/generate.py -->

```python
def cached(model: Model, prompt: list[int], n_new: int) -> Run:
    cache = KVCache(model.cfg, max_seq=len(prompt) + n_new)
    ...
    logits = forward(model, np.array(prompt), cache)  # prefill
    nxt = int(logits[-1].argmax())
    ...
    for _ in range(n_new - 1):
        ...
        logits = forward(model, np.array([nxt]), cache)  # decode: one token
        nxt = int(logits[-1].argmax())
```

### First, check that nothing changed

Before measuring a speedup, prove you did not buy it with accuracy.
Both paths decode greedily, so for a given prompt they must return
*identical* tokens:

<!-- listing: tinyserve/generate.py check_equivalence -->

```python
def check_equivalence(model: Model, prompt: list[int], n_new: int) -> bool:
    return naive(model, prompt, n_new).tokens == cached(model, prompt, n_new).tokens
```

It returns `True`. Keep this test; every optimization in Parts III and
IV should either pass an equivalence check like it or come with an
explicit statement of what it changes and by how much. Quantization in
Part V is the first technique in this book that fails such a test on
purpose, and Chapter 28 is about measuring exactly what it costs.

## Measure

Same model, same prompt, same machine: a 128-token prompt,
128 tokens generated, 7 repeats after 2 warmup
runs, with the two paths run alternately so that drift on a shared
machine lands on both equally.

<!-- include: tables/ch12-head-to-head.md -->
| Measurement | Without a cache | With a KV cache | Ratio |
|---|---|---|---|
| Time to first token (prefill) | 22.24 ms | 20.27 ms | **1.10x** |
| Decode step, p50 | 27.83 ms | 0.51 ms | **55x** |
| Decode step, p99 | 45.05 ms | 0.70 ms | **65x** |
| Decode throughput | 36 tok/s | 1,937 tok/s | **54x** |
| Total, 128 prompt + 128 generated | 3.580 s | 0.086 s | **42x** |

Read the first row before the others. **Time to first token barely
moves** — 22.2 ms against 20.3 ms. Producing the
first token means processing the prompt, and there is nothing cached
yet to help; both paths do the same work. The cache does nothing for
prefill and everything for decode. That split runs through the whole
book: TTFT and inter-token latency are governed by different limits,
which is why Chapter 5 asked you to write them into an SLO as separate
numbers.

![Time per generated token](code/figures/ch12-per-step.svg)

**Figure 12.1** — Without a cache, every token costs more than the
last. Time per generated token against position, log scale. Without a
cache the cost climbs from 25 ms to 74 ms; with
a cache it stays flat near 0.54 ms. Both spend the same time
on the first token, which is prefill. *Provenance in
`code/figures/ch12-per-step.caption.txt`.*

The uncached line climbs because each step processes a sequence one
token longer than the last. Between the first and last generated token
the sequence grows from 129 to 383 tokens, a factor
of 2.97, and the step cost grows by a factor of
2.94. The two agree closely, and that is the finding: **at
this size, the cost of a step is proportional to the length of the
sequence it reprocesses.**

You might expect worse than proportional, since attention compares
every token against every other and so grows with the *square* of the
length. It does — but on a model this small the quadratic term is a
minor part of the total, which is dominated by the per-token
projections and feed-forward layers. Exercise 12.5 asks you to find the
sequence length where the quadratic term takes over.

The cached line is flat, and that flatness is the point. Each decode
step does the same work no matter how long the conversation is: one
token's projections, and one row of attention against a cache that is
cheap to read and expensive only to store. Storage is Chapter 13's
problem.

<!-- include: tables/ch12-scaling.md -->
| Tokens generated | Without a cache | With a cache | Measured speedup | Speedup predicted by FLOPs |
|---|---|---|---|---|
| 16 | 0.384 s | 0.032 s | **12x** | 15x |
| 32 | 0.819 s | 0.041 s | **20x** | 29x |
| 64 | 1.745 s \* | 0.057 s | **30x** | 55x |
| 128 | 4.190 s \* | 0.090 s | **46x** | 100x |
| 256 | 11.242 s \* | 0.164 s | **69x** | 181x |

\* run-to-run spread exceeded 5%; see the note on the measuring machine.

![Total time against tokens generated](code/figures/ch12-scaling.svg)

**Figure 12.2** — The gap widens with every token generated. Total time
against the number of tokens generated, both axes logarithmic.
*Provenance in `code/figures/ch12-scaling.caption.txt`.*

The speedup is not a constant. It grows with every token you generate,
because the work you are avoiding grows with every token you generate.
Fitted over this range, uncached total time grows as roughly
*n*<sup>1.21</sup> and cached as *n*<sup>0.59</sup>.

Neither exponent is a clean 1 or 2, and it is worth saying why rather
than rounding to the nearest textbook shape. The prompt is
128 tokens, so at the short end of the sweep most of the
work is prefill — a fixed cost both paths pay, which flattens both
curves. As generation grows past the prompt, the uncached exponent
climbs toward 2. Do not quote an asymptotic exponent from a range where
a constant dominates.

## Where it breaks

**It costs memory, and the bill scales with every concurrent user.**

For our model: 2 arrays (keys and values) × 4 layers ×
4 KV heads × 32 dimensions × 4 bytes per element
= **4,096 bytes per token**. In general:

> KV bytes per token = 2 × layers × KV heads × head dimension × bytes
> per element

<!-- include: tables/ch12-memory.md -->
| | `tinyserve` (this chapter) | Llama-3-style 8B |
|---|---|---|
| Layers | 4 | 32 |
| KV heads | 4 | 8 |
| Head dimension | 32 | 128 |
| Bytes per element | 4 (fp32) | 2 (bf16) |
| **KV cache per token** | **4,096 B** | **128 KiB** |
| Per 8,192-token sequence | 32 MiB | 1.0 GiB |

**128 KiB per token. 1.0 GiB per 8,192-token sequence.** An
H100 holds 80 GB; an 8B model in bf16 takes 16 GB of
that, leaving about 64 GB. That is room for roughly
59 sequences of 8K tokens — before any **fragmentation**,
memory that is held but cannot be used, and
<!-- defines: fragmentation -->
before anything else needs memory. Your serving capacity is a memory
question, not a compute question. Chapter 13 measures how much of that
64 GB a naive allocator actually wastes, and Chapter 14 gets it
back.

This is also why production models use **grouped-query attention**.
Every KV head you remove is a proportional cut to this table. A model
with 32 query heads and 8 KV heads stores a quarter of the cache of one
with 32 of each, which is why the reference model above has 8.

**The speedup is smaller than the arithmetic says it should be.**

Look again at the last two columns of the scaling table. At
256 tokens the cached path does **181x**
fewer multiply-adds, and runs **69x** faster. A factor
of about 2.6 went missing.

It is not measurement error, and it is the most important sentence in
this chapter. Without a cache, each step multiplies a
383×128 matrix by a 128×128 one — big enough for the linear
algebra library to use the machine well. With a cache, each step
multiplies a **1**×128 matrix by a 128×128 one. The arithmetic is
trivially small, but the model's weights must still be read out of
memory in full to do it. You stopped paying for arithmetic and started
paying for memory traffic, and the hardware is far worse at the second.

You have just met the central fact of this book, on a laptop CPU, three
chapters into building an engine: **decoding is limited by how fast you
can read weights, not by how fast you can multiply.** Chapter 8's
roofline predicted it. Chapter 16 will turn it from a disappointment
into a strategy: if reading the weights costs the same whether you
serve one user or a hundred, serve a hundred.

## In production

Every serving engine does what you just built; the differences are in
what they do about the memory bill.

- **vLLM** and **SGLang** allocate the cache in fixed-size blocks
  rather than one array per sequence, the subject of Chapter 14. In
  vLLM the share of GPU memory set aside for it is
  `--gpu-memory-utilization`, the block size is `--block-size`, and
  `--max-model-len` caps how long a single sequence's cache may grow.
- **Cache precision** is a flag, not a rewrite: serving the cache in
  fp8 halves the table above at some cost in quality. Chapter 26
  measures that cost.
- **Prefix sharing.** Two requests with the same system prompt compute
  and store identical keys and values. Chapter 15's prefix cache stores
  them once.
- **Eviction and preemption.** <!-- defines: eviction, preemption -->
  Your cache assumes a sequence runs to completion with its memory
  reserved. A real server runs out, and must **evict** somebody —
  taking memory back from a running sequence and recomputing or
  restoring it later, which is **preemption**. It must decide whose
  cache to drop and whether to recompute it or swap it out. That is
  Chapter 18's scheduler.

## Numbers to remember

| Quantity | Value |
|---|---|
| KV cache per token, Llama-3-style 8B, bf16 | 128 KiB |
| KV cache per 8K-token sequence, same model | 1.0 GiB |
| KV bytes per token, general | 2 × layers × KV heads × head dim × bytes |
| Speedup from a KV cache | grows with output length; 69x at 256 tokens here |
| Effect on time to first token | none — prefill has nothing to reuse |

## Sources

- Vaswani et al., "Attention Is All You Need", NeurIPS 2017 — attention,
  and the causal structure that makes caching valid.
- Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need",
  arXiv:1911.02150, 2019 — states the memory-bandwidth problem of
  decoding and introduces multi-query attention.
- Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints", EMNLP 2023 — grouped-query
  attention, the middle ground production models use.
- Pope et al., "Efficiently Scaling Transformer Inference", MLSys 2023
  — the arithmetic of the memory wall at serving scale.
- Kwon et al., "Efficient Memory Management for Large Language Model
  Serving with PagedAttention", SOSP 2023 — §3 measures the waste this
  chapter's allocator creates, and Chapter 14 fixes.

## A note on the measuring machine

These numbers come from a shared 4-vCPU cloud instance
(Intel(R) Xeon(R) Processor @ 2.10GHz), where 5 of the 10 sweep
measurements had a run-to-run spread above 5%, marked in the table. The
headline speedup itself varies by about 17% across
7 paired repeats, so treat it as "about 42x", not
as a precise figure.

What is stable is the *shape*: the ratio grows with output length, time
to first token does not move, and the decode step goes from tens of
milliseconds to well under one. The memory figures carry no
uncertainty at all — they are arithmetic on the model's architecture,
not measurements.

Absolute times here are not portable and are not meant to be. Run
`make ch12` on your own machine and expect different absolute numbers
and the same shape.

## Exercises

**★ 12.1** Compute the KV cache size per token for a model with 48
layers, 8 KV heads, head dimension 128, served in bf16. How many
16K-token sequences fit in 40 GB?

**★ 12.2** Time to first token barely changed between the two paths.
Explain why in two sentences, without using the word "cache".

**★★ 12.3** `forward` takes the new tokens' positions from
`cache.length`. Break it: change `start` to always be 0, and run
`check_equivalence`. Does it fail? Now run it with a 4-token prompt and
inspect the output tokens. What does this tell you about tests that
only check that nothing crashed?

**★★ 12.4** Set `n_kv_heads=1` in `Config` (multi-query attention) and
rerun `python3 -m bench.run_ch12`. Predict the change in cache size and
in decode latency *before* you run it, then explain any gap between
your prediction and the measurement.

**★★★ 12.5** This chapter claims the quadratic cost of attention is
minor at this model size, and that it takes over at longer sequences.
Find where. Using the harness, measure the uncached step cost at
sequence lengths 128, 512, 2,048 and 8,192, fit the growth exponent at
each end, and report the length at which attention passes the
projections and feed-forward layers in cost. Predict it from the model
config first. Report with percentiles and provenance, per the
harness's rules.
