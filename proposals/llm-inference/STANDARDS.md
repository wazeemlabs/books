# Editorial and Engineering Standards

*LLM Inference from the Ground Up* is written to the standard of a
first-party engineering text: the reader must be able to trust every
number, reproduce every figure, and trace every technique to the work
that introduced it. These rules are the bar. A chapter that does not
meet them is not finished.

## 1. Evidence

1.1 **No performance claim without a measurement.** Every throughput,
latency, memory, or cost figure in the text comes from a run of the
companion harness (`bench/`). "Roughly 2x faster" is not permitted
unless the run that produced it is cited.

1.2 **Every figure carries its provenance.** The caption names the
hardware (GPU SKU, count, interconnect), the software (driver, CUDA,
PyTorch, engine version), the model and revision, the traffic profile,
and the harness commit hash. `make figures` regenerates every figure
in the book from the harness output; a figure that cannot be
regenerated is removed.

1.3 **Percentiles, not averages.** Latency is reported as p50 and p99
(p99.9 where it matters), with the offered load stated. Throughput is
reported at the load that meets the chapter's SLO, and at saturation,
and the reader is told which one they are looking at.

1.4 **Warm, repeated, and seeded.** Each reported number is the median
of at least three runs after warmup, with fixed seeds and fixed prompt
sets. The variance is reported when it exceeds 5%.

1.5 **Say what was not measured.** When a result is on one GPU class,
the text says so and does not extrapolate to another.

## 2. Sources

2.1 **Primary sources for core claims.** Each technique is cited to
the paper, system, or documentation that introduced it, in the
chapter's *Sources* section. Blog posts and vendor marketing may be
cited for context, never as the only support for a technical claim.

2.2 **Public and stable.** Cite arXiv identifiers, conference venues,
or versioned documentation. A URL alone is not a citation.

2.3 **Attribute ideas.** PagedAttention is vLLM's idea; RadixAttention
is SGLang's; continuous batching is Orca's. The book says so.

## 3. Reproducibility

3.1 **Pinned environment.** `environment.lock` in the companion repo
pins driver, CUDA, PyTorch, every engine, and every model revision
used in the book. The lock is updated only at a numbered revision of
the book, and the companion page records what changed.

3.2 **Tiered compute, declared per chapter.** Each chapter header
states the tier it runs on (0: CPU/Colab, 1: T4 or one consumer GPU,
2: one H100/H200, 3: multi-GPU node) and an estimated cost to
reproduce it at published rental prices.

3.3 **Continuous verification.** CI runs every Tier 0 notebook on each
commit and every Tier 1–2 chapter on each release candidate. A chapter
whose code does not run does not ship.

3.4 **Cold reproduction before release.** For each chapter, a reviewer
who did not write it reproduces every number from a clean machine
using only the text and the repo. Discrepancies above the stated
variance are defects.

## 4. Prose

4.1 Follow the Google developer documentation style guide: second
person, present tense, active voice, short sentences, one idea per
paragraph, terms defined on first use and used consistently after.

4.2 **No marketing language.** "Blazing", "massive", "game-changing",
and their relatives do not appear. A speedup is a number with a
condition.

4.3 **Units, always.** Bytes are bytes (GiB when binary), bandwidth is
GB/s, time is ms or µs, cost is USD per million tokens with the token
type (input, output, or blended) stated.

4.4 **Precision honesty.** Two significant figures unless a third is
meaningful. "4.8 ms" not "4.776 ms" from a noisy run.

4.5 **The reader is a peer.** Explain from first principles; never
condescend; never skip a step because it is "obvious".

## 5. Chapter structure

Every chapter uses the same skeleton, in this order:

1. **Objectives** — two to four things the reader can do afterward,
   each testable.
2. **Why it matters** — the production problem, in one page, with a
   number.
3. **Build** — the mechanism, constructed in the reader's engine.
4. **Measure** — the harness run, the figure, the provenance.
5. **Where it breaks** — the conditions under which the technique
   hurts, with a measurement showing it.
6. **In production** — how the real engines implement it, and the
   flags that control it.
7. **Numbers to remember** — three to five figures with units.
8. **Sources** — primary citations.
9. **Exercises** — ★ (answer in Appendix B), ★★ (answer in Appendix B),
   ★★★ (rubric in the instructor materials).

Each Part ends with a **Design decision record** for the running case
study (§7): the decision made, the options considered, the measured
tradeoffs, and what would change the decision.

## 6. Code

6.1 `tinyserve` is a real package: typed, tested, linted, under 3,000
lines, readable in one sitting. Its tests are its specification.

6.2 Every code listing in the text is extracted from the repo at a
tagged commit, never typed into the manuscript by hand.

6.3 The harness reports machine-readable results (JSON) that the
figure pipeline consumes; no number is transcribed by hand.

## 7. The running case study

A single production service is threaded through the book so that
every technique is judged against a stated objective, not admired in
isolation:

- **Service:** a customer-support assistant behind a chat UI.
- **Model:** an 8B-parameter open model (Part V explores smaller).
- **Traffic:** 200 requests/s at peak, 1,200 input tokens and 300
  output tokens on average, heavy shared system prompt, diurnal.
- **SLO:** p99 TTFT ≤ 1,000 ms, p99 ITL ≤ 50 ms, 99.9% availability.
- **Objective:** minimum cost per million output tokens that meets the
  SLO.

Chapter 4 states it, Chapter 40 sizes it, Chapter 41 prices it, and
the capstone (Appendix H) asks the reader to beat the book's number.

## 8. Review and release

8.1 **Two reviewers per chapter:** one technical reviewer with
production inference experience, one cold-reproduction reviewer
(§3.4). Both sign off in the review log before a chapter is marked
done.

8.2 **A design review per Part** before writing begins, against this
document and the outline.

8.3 **Errata as postmortems.** Every correction is logged on the
companion page with the error, the root cause (wrong measurement,
wrong source, wrong reasoning), and the fix. The log is public.

8.4 **Numbered revisions.** The book is re-issued on a schedule tied to
engine releases; the companion "what changed since print" page tracks
the delta between revisions, per chapter.

## 9. Figures and accessibility

9.1 Every figure has alt text that states the finding, not the
chart type. Colors are distinguishable in grayscale and to readers
with color-vision deficiency. Axes are labeled with units. Log scales
are marked.

9.2 One message per figure. A figure that needs a paragraph to explain
is two figures.
