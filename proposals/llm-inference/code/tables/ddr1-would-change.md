| Decision | What would change it |
|---|---|
| Page the KV cache in blocks of 16 tokens | a kernel that needs longer contiguous runs than a block, or replies short enough that a block's worth of waste stops being small beside them |
| Prefix caching on, with a prefix tree and least-recently-used eviction of leaves | traffic with no shared prefixes, where the index costs something and returns nothing. Nothing measured here makes the feature worth turning off when prefixes are shared at all |
| Continuous batching: decide the batch every iteration | nothing in this book. Static batching lost on every measure at every rate tried |
| Chunked prefill, with a 512-token per-iteration budget | a looser between-token promise, which would buy a larger budget and a better first token; or a tighter one, which the smaller budgets serve at a first token this service could not sell |
| First come, first served, with no priority tiers | running the pool near full. Squeezed, shortest first is 22.9x better on the median and 7.5x better on the worst slowdown, and this decision flips |
| Preempt by recomputing, not by swapping the cache out to host memory | a swap-in that restores a cache gradually rather than all at once. What lost here was the shape of the re-entry, not the cost of the copy: even over NVLink, which is fast enough on the arithmetic, swapping still lost |
| One fleet where every machine does both phases | hardware that differs by phase, or a promise tight enough that one fleet has to over-provision to keep it. Neither is true here, and this is the decision in this record most likely to be wrong for a service that is not this one |

A decision without a condition attached to it is a habit. These are the conditions -- the things that, if they became true of a service, would make the row above the wrong answer for it.
