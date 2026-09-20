| Of capacity | What this server does | What a classical queue would do | Over-predicted by |
|---|---|---|---|
| 6% | 1.05x | 1.07x | **1.0x** |
| 13% | 1.10x | 1.15x | **1.0x** |
| 26% | 1.21x | 1.35x | **1.1x** |
| 39% | 1.33x | 1.63x | **1.2x** |
| 51% | 1.45x | 2.06x | **1.4x** |
| 64% | 1.58x | 2.81x | **1.8x** |
| 77% | 1.70x | 4.39x | **2.6x** |
| 84% | 1.80x | 6.12x | **3.4x** |
| 90% | 1.92x | 10.11x | **5.3x** |
| 97% | 2.10x | 28.91x | **13.8x** |
| 103% | 2.65x | over capacity | -- |
| 116% | 3.89x | over capacity | -- |
| 129% | 5.24x | over capacity | -- |

Slowdown is time in the system divided by 1.47 s, which is what one request takes with the machine to itself. The classical column is 1 / (1 - utilization), the standard result for a single-server queue and the arithmetic behind every rule of thumb about not running servers hot. It does not describe this server, and the last column is how much hardware believing it would buy.
