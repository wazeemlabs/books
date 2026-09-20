| Regime | Best split | Disaggregated | Colocated | Ratio |
|---|---|---|---|---|
| 100-token prompts at 200 req/s | 2P + 10D | 55,166 tok/s | 55,250 tok/s | **1.00x** |
| 300-token prompts at 200 req/s | 2P + 10D | 54,793 tok/s | 54,871 tok/s | **1.00x** |
| 1,200-token prompts at 200 req/s | 4P + 8D | 50,969 tok/s | 52,664 tok/s | **0.97x** |
| 4,000-token prompts at 200 req/s | 7P + 5D | 20,704 tok/s | 36,007 tok/s | **0.57x** |
| 4,000-token prompts at 60 req/s | 6P + 6D | 17,085 tok/s | 17,367 tok/s | **0.98x** |
| 2 accelerators at 33 req/s | 1P + 1D | 9,633 tok/s | 9,793 tok/s | **0.98x** |
| 4 accelerators at 67 req/s | 2P + 2D | 18,729 tok/s | 19,211 tok/s | **0.97x** |
| 8 accelerators at 133 req/s | 3P + 5D | 35,774 tok/s | 36,754 tok/s | **0.97x** |
| 12 accelerators at 200 req/s | 4P + 8D | 50,969 tok/s | 52,664 tok/s | **0.97x** |

Every disaggregated row re-searches the split, so none of them is a straw man. The colocated arm is one accelerator per replica running Chapter 18's scheduler. The 4,000-token row holds the prompt tokens arriving per second at the case study's, so the fleet is not simply saturated; the rows above it do not, which is why both designs fall behind at 200 req/s with long prompts.
