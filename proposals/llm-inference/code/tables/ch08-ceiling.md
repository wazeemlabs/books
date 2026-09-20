| Change | Ceiling on arithmetic per byte | Reaches break-even? |
|---|---|---|
| as served here | 81 | **no** — 3.6x short |
| KV cache in one byte instead of two | 163 | **no** — 1.8x short |
| half the context length | 163 | **no** — 1.8x short |
| both | 326 | **yes**, just clears it |

However large the batch, decode's arithmetic per byte cannot pass these values, against a break-even point of 296.
