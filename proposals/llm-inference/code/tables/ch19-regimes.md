| Regime | Best split | Disaggregated | Colocated | Ratio |
|---|---|---|---|---|
| 100-token prompts at 200 req/s | 2P + 10D | 58,600 tok/s | 58,593 tok/s | **1.00x** |
| 300-token prompts at 200 req/s | 2P + 10D | 58,514 tok/s | 58,514 tok/s | **1.00x** |
| 1,200-token prompts at 200 req/s | 4P + 8D | 56,066 tok/s | 57,861 tok/s | **0.97x** |
| 4,000-token prompts at 200 req/s | 7P + 5D | 23,301 tok/s | 38,891 tok/s | **0.60x** |
| 4,000-token prompts at 60 req/s | 6P + 6D | 17,301 tok/s | 17,398 tok/s | **0.99x** |
| 2 accelerators at 33 req/s | 1P + 1D | 9,655 tok/s | 9,779 tok/s | **0.99x** |
| 4 accelerators at 67 req/s | 2P + 2D | 19,215 tok/s | 19,296 tok/s | **1.00x** |
| 8 accelerators at 133 req/s | 3P + 5D | 37,996 tok/s | 38,578 tok/s | **0.98x** |
| 12 accelerators at 200 req/s | 4P + 8D | 56,066 tok/s | 57,861 tok/s | **0.97x** |

Every disaggregated row re-searches the split, so none of them is a straw man. The colocated arm is one accelerator per replica running Chapter 18's scheduler. The 4,000-token row holds the prompt tokens arriving per second at the case study's, so the fleet is not simply saturated; the rows above it do not, which is why both designs fall behind at 200 req/s with long prompts.
