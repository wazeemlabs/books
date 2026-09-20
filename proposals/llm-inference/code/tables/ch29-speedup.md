| Draft (cost of one draft step) | 30% accepted | 50% accepted | 70% accepted | 80% accepted | 90% accepted | 95% accepted |
|---|---|---|---|---|---|---|
| int8 copy of the target (0.5 of a target step) | 1.00x (worse than not bothering) | 1.00x (worse than not bothering) | **1.13x** at k=1 | **1.22x** at k=2 | **1.38x** at k=3 | **1.51x** at k=5 |
| int4 copy, groups of 32 (0.312 of a target step) | 1.00x (worse than not bothering) | **1.14x** at k=1 | **1.35x** at k=2 | **1.52x** at k=3 | **1.83x** at k=5 | **2.11x** at k=8 |
| a model a tenth the size (0.1 of a target step) | **1.18x** at k=1 | **1.46x** at k=2 | **1.98x** at k=4 | **2.47x** at k=6 | **3.43x** at k=10 | **4.48x** at k=15 |
| a model a fortieth the size (0.025 of a target step) | **1.32x** at k=2 | **1.76x** at k=4 | **2.67x** at k=7 | **3.66x** at k=10 | **5.96x** at k=18 | **9.11x** at k=28 |

Each cell is the best number of guesses per round and what it is worth. A round costs k draft steps and one target step and yields (1 - a^(k+1)) / (1 - a) tokens, so the best k rises with the acceptance rate and with how cheap the draft is. Nothing here is measured on hardware: the acceptance rate is a parameter and the costs are ratios of decode steps, priced by Chapter 16.
