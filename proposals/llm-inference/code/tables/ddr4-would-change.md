| Decision | What would change it |
|---|---|
| Speculative decoding behind a measurement, with the tree sized to the batch rather than to a paper | the measurement that has not been made: the share of this service's replies that quote their prompt. Chapter 30 measured that number spanning 7% to 92% across four shapes of reply, which is the difference between worthwhile and pointless |
| If it is on, draft from the prompt | traffic that is not grounded in its prompt: the same drafter returns 1.11x there, and a trained head is the only option left |
| Constrain the structured endpoint, compiling each schema once at startup | accepting arbitrary JSON Schema per request, which moves the compile onto the request path and into the first-token latency |
| An exact-match response cache, keyed on every field the answer depends on | traffic with no repeated questions, where the cache is dead weight -- and agent traffic, where it already is |
| No thinking budget on the default endpoint | a second endpoint for work that is worth minutes and is asked for asynchronously, which is what the provider documentation recommends above a 32k budget |
| A dense model, and a context ceiling well short of 128K | demand large enough to keep a mixture's fleet busy, where its cost per token is several times better -- this is a decision about scale, not about architecture |

A decision without a condition attached to it is a habit. Two of these conditions are measurements nobody has taken yet, which is the honest state of a decision record written while the service is still being built.
