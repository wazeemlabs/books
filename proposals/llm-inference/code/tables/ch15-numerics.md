| Step of the forward pass | Contracts over | Same answer when the sequence is 96 tokens longer? |
|---|---|---|
| A projection (Q, K, V, feed-forward) | the model dimension, which does not change | **yes** |
| The softmax in attention | the sequence, which does | **no** — differs by 1.5e-08 |

And what that does to the keys the cache hands back, layer by layer:

| Layer | Difference from the same key computed afresh |
|---|---|
| 0 | exactly 0 |
| 1 | 1.9e-06 |
| 2 | 2.1e-06 |
| 3 | 2.6e-06 |

Layer 0's keys come from the embedding and a projection alone, so they are identical. Every layer after it has been through an attention softmax.
