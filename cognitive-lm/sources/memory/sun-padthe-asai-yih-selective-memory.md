https://openreview.net/forum?id=5a2H3HEN61

# Semi-parametric language model with selective memory
Authors: Alicia Yi Sun, Karthik Padthe, Akari Asai, Wen-tau Yih (2025, OpenReview)

STATUS: full text NOT fetched. OpenReview returned a browser-verification challenge to curl, the web fetcher, WebFetch and Playwright (2026-09-26). Not on arXiv (arXiv API title+author search returned nothing).

Abstract content as surfaced by WebSearch (snippet of the OpenReview page, not verified on page):
- "identifying atomic facts that are not present in a pretrained LLM's parametric memory. These facts are then stored in an external, non-parametric memory."
- "Subsequently, the model undergoes continual pretraining, enabling it to learn when to consult this external memory at inference time."
- "uses a compact external memory that selectively stores only the facts not clearly present in the LLM's parametric memory, resulting in minimal additional inference-time costs in terms of both time and space."

Independent confirmation, Co-LMLM (arXiv 2607.07707) related work, verified on fetched PDF: "More recently, SPLM [Sun et al., 2025] stores atomic facts absent from a pretrained model's parametric memory in a compact external memory, and continue pretrains the model to retrieve from this external, non-parametric memory at inference"

Retry 2026-09-26 (archive build): https://openreview.net/pdf?id=5a2H3HEN61 via the web fetcher redirected to the OpenReview browser-verification challenge page. Full text still not fetched.
