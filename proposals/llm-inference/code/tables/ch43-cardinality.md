| Labels | Values on the last one | Gauge series | Series for one histogram |
|---|---|---|---|
| `model` | 3 | 3 | 66 |
| `model`, `tenant` | 40 | 120 | 2,640 |
| `model`, `tenant`, `endpoint` | 4 | 480 | 10,560 |
| `model`, `tenant`, `endpoint`, `instance` | 8 | 3,840 | 84,480 |

A histogram is not one time series. It is one per bucket plus a sum and a count, which for vLLM's between-tokens histogram is 19 + 3 = 22 before a single label is attached. Multiply by every combination of label values and a metrics bill stops being about volume and starts being about cardinality.
