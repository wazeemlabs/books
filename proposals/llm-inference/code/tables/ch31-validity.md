| The model's own grasp of JSON | With the mask: valid | finished | Without it: valid | finished |
|---|---|---|---|---|
| none at all | **100.0%** | 99.7% | 1.3% | 100.0% |
| slight | **100.0%** | 100.0% | 1.3% | 100.0% |
| some | **100.0%** | 100.0% | 1.7% | 100.0% |
| good | **100.0%** | 100.0% | 0.2% | 97.3% |
| strong | **100.0%** | 99.8% | 0.0% | 11.0% |

600 documents at each level, cut off at 400 tokens. "Valid" is the share of the documents that *finished* which parse; a walk cut off at the token limit is incomplete, which is a different failure and is counted in the column beside it. The model here is a stand-in whose preference for legal tokens can be turned up, which is the only property of a real model this measurement depends on.
