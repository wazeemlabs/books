| Regime | Best split | Disaggregated | Colocated | Ratio |
|---|---|---|---|---|
| 100-token prompts at 200 req/s | 2P + 10D | 59,940 tok/s | 59,945 tok/s | **1.00x** |
| 300-token prompts at 200 req/s | 2P + 10D | 59,933 tok/s | 59,939 tok/s | **1.00x** |
| 1,200-token prompts at 200 req/s | 5P + 7D | 59,491 tok/s | 59,853 tok/s | **0.99x** |
| 4,000-token prompts at 200 req/s | 6P + 6D | 24,972 tok/s | 38,469 tok/s | **0.65x** |
| 4,000-token prompts at 60 req/s | 5P + 7D | 18,067 tok/s | 18,102 tok/s | **1.00x** |
| 2 accelerators at 33 req/s | 1P + 1D | 10,056 tok/s | 10,088 tok/s | **1.00x** |
| 4 accelerators at 67 req/s | 2P + 2D | 20,024 tok/s | 20,098 tok/s | **1.00x** |
| 8 accelerators at 133 req/s | 3P + 5D | 39,869 tok/s | 39,952 tok/s | **1.00x** |
| 12 accelerators at 200 req/s | 5P + 7D | 59,491 tok/s | 59,853 tok/s | **0.99x** |

Every disaggregated row re-searches the split, so none of them is a straw man. The colocated arm is one accelerator per replica running Chapter 18's scheduler. The 4,000-token row holds the prompt tokens arriving per second at the case study's, so the fleet is not simply saturated; the rows above it do not, which is why both designs fall behind at 200 req/s with long prompts.
