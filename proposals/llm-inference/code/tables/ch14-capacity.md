| | Reserving the full context | Paged, 16-token blocks |
|---|---|---|
| Sequences that fit in 64 GB | 59 | **327** |
| Memory held that is in use | 16% (Chapter 13) | **99.5%** |
| Wasted per sequence | up to the whole context | **7.7 tokens** |
| Cost per decode step | 1.00x | **1.14x** |

Capacity computed for the reference model over 4,000 sampled requests; the step cost is measured on tinyserve. Chapter 13 predicted these capacities from arithmetic before this allocator existed.
