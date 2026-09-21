| Of capacity | What this server does | What a classical queue would do | Over-predicted by |
|---|---|---|---|
| 7% | 1.04x | 1.07x | **1.0x** |
| 13% | 1.09x | 1.15x | **1.1x** |
| 26% | 1.20x | 1.35x | **1.1x** |
| 39% | 1.32x | 1.65x | **1.2x** |
| 52% | 1.45x | 2.10x | **1.5x** |
| 65% | 1.57x | 2.90x | **1.8x** |
| 79% | 1.71x | 4.67x | **2.7x** |
| 85% | 1.81x | 6.72x | **3.7x** |
| 92% | 1.99x | 12.00x | **6.0x** |
| 98% | 2.94x | 56.04x | **19.1x** |
| 105% | 8.36x | over capacity | -- |
| 118% | 24.36x | over capacity | -- |
| 131% | 37.57x | over capacity | -- |

Slowdown is time in the system divided by 1.47 s, which is what one request takes with the machine to itself. The classical column is 1 / (1 - utilization), the standard result for a single-server queue and the arithmetic behind every rule of thumb about not running servers hot. It does not describe this server, and the last column is how much hardware believing it would buy.
