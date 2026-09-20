| Architecture checked | Largest difference in any score | Same tokens chosen |
|---|---|---|
| reference shape (4L, d=128, 4Q/4KV heads) | 2.4e-06 | **yes** |
| grouped-query attention (4L, d=256, 8Q/2KV heads) | 3.8e-06 | **yes** |
| deeper (8L, d=192, 6Q/6KV heads) | 2.6e-06 | **yes** |
| single key-value head (4L, d=128, 4Q/1KV heads) | 2.5e-06 | **yes** |

Same weights, 24 tokens, against PyTorch 2.14.0+cpu. Differences of this size are float32 rounding: the two implementations do the same arithmetic in a different order.
