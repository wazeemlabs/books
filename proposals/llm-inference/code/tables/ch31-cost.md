| Per token | Amount | As a share of a decode step |
|---|---|---|
| Additions to the logits | 128,256 | 0.000801% of its arithmetic |
| Mask read from the table | 16,032 bytes | 0.00010% of its bytes |
| A decode step, for comparison | 16.2 GB read, 16.0 GFLOP | 100% |

At the reference model's 128,256-token vocabulary, one bit of mask per token packed. Counted, not timed: the mask is one lookup and one addition per vocabulary entry, and a decode step reads every weight in the model, so both sides are exact. A real implementation can be slower than this arithmetic -- the lookup has to find the right row, and with a real tokenizer whose pieces cut across the grammar, building the table is the hard part -- but the floor is five orders of magnitude below the step it rides on.
