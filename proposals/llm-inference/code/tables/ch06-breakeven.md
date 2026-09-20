| Precision | GPU pricing | Cost per 1M tokens at full tilt | Utilization needed to beat the API |
|---|---|---|---|
| bf16 (two bytes) | on-demand median | $0.066 | **never** — would need 133% |
| bf16 (two bytes) | cheapest marketplace | $0.030 | **61%** |
| fp8 (one byte) | on-demand median | $0.033 | **66%** |
| fp8 (one byte) | cheapest marketplace | $0.015 | **30%** |

Against a published API price of $0.05 per million output tokens (cheapest tracked hosted 8B, September 2026). Hardware only: engineering and operations typically add another 3-5x on top.
