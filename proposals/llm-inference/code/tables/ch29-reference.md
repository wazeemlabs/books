| Draft | Guesses | Tokens a round | Times faster | Between tokens |
|---|---|---|---|---|
| int8 copy of the target | 2 | 2.44 | **1.22x** | 3.96 ms |
| int4 copy, groups of 32 | 3 | 2.95 | **1.52x** | 3.17 ms |
| a model a tenth the size | 6 | 3.95 | **2.47x** | 1.96 ms |
| a model a fortieth the size | 10 | 4.57 | **3.66x** | 1.32 ms |

The book's 8B model, whose decode step is 4.84 ms between tokens without any of this (Chapter 16's cost model, one sequence, 1,500 tokens of context). Every row assumes 80% of guesses are accepted, which is a stated assumption and not a measurement -- the acceptance rate depends on the draft, the target and the traffic, and the only honest way to get it is to measure the pair you actually have.
