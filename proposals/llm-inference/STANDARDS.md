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

### 5.1 Part I and walkthrough chapters

Part I carries no code, and a walkthrough chapter (such as Chapter 2)
follows one example through a mechanism rather than building and then
measuring one thing. Both may replace **Build** with a section that
derives the result by hand (*Work it out*) or narrates the path
(*Step 1..n*), and both may distribute their evidence through the
narrative instead of concentrating it under **Measure**.

What does not change: the chapter still opens with **Objectives** and
**Why it matters**, still carries a limitations section, **Numbers to
remember**, **Sources** and **Exercises**, and every quantitative claim
in it is still generated from a results file with provenance (§1). A
chapter with no generated figure or table is not evidence-backed,
whatever its headings say, and `make audit` enforces that rather than
enforcing a heading word.

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

Chapter 5 states it, Chapter 41 sizes it, Chapter 42 prices it, and
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

## 10. Pedagogy: from the ground up

The book assumes a reader who is new to the field and refuses to lose
them. These rules govern how every concept is introduced.

10.1 **Intuition first, then the mechanism, then the name.** Each
concept opens with a concrete situation the reader can picture (one
request, one GPU, one user waiting), then the mechanism is built in
the reader's own code, and only then is the standard term introduced
and the paper cited. The reader meets "KV cache" after they have
already built one.

10.2 **One new idea per section.** A section that introduces two
concepts is two sections. Advanced material is reached by a sequence
of small steps, never by a jump.

10.3 **A prerequisite ladder, enforced.** Each chapter lists the
chapters it depends on. No chapter uses a term, formula, or tool that
an earlier chapter has not introduced. The manuscript tooling checks
first uses of glossary terms against the ladder.

10.4 **A figure per mechanism.** Every mechanism gets a diagram of
what is happening in memory or on the GPU, before and after. Figures
show the thing, not a chart about the thing; charts come in the
*Measure* section.

10.5 **Worked examples with real numbers.** Every formula is applied
to the running model with actual values before it is stated in
general form. "128 KiB per token" comes before
"2 × L × H_kv × d × bytes".

10.6 **Analogies with their limits stated.** Paged attention is
introduced through OS paging; the text then says exactly where the
analogy stops holding.

10.7 **"If you're new here" boxes.** Where a chapter touches
territory a newcomer may lack (what a tensor is, what a process is,
what a percentile is), a short box explains it in place. Nobody is
sent away to read something else first.

10.7.1 **Every term is explained before it is used — including the
intermediate ones.** The reader is assumed to know nothing. Not only
the hard words: *parameter*, *bandwidth*, *utilization*, *allocator*,
*percentile*, *kernel*, *driver* and their like are explained on first
use too, because the reader who needs them is exactly the reader who
will not ask.

Every such term lives in `GLOSSARY.md` with the chapter that
introduces it, and that chapter claims it with a marker beside the
explanation:

    <!-- defines: arithmetic intensity, memory-bound -->

`make check` fails if a term is used in an earlier chapter than the one
that claims it, if a glossary term is never claimed, or if a chapter
claims something the glossary does not list. Terms whose bare form is
ordinary English ("a block of numbers", a column headed "Value") are
still defined but exempt from the ordering check, and are listed
explicitly in `bench/terms.py`.

10.7.2 **Explain, then name.** The plain explanation comes first and
the standard term second — "the raw scores have a name: **logits**" —
so the reader understands the thing before being asked to carry a
label for it. A name introduced without its explanation is a term the
reader will skip.

10.8 **Book 1 is a deeper path, not a prerequisite.** Chapter 2 gives
everything about the model that this book needs. Pointers to *Large
Language Models from the Ground Up* are offered for depth and never
required.

10.9 **Plain words before symbols.** Every equation is preceded by a
sentence saying what it means and followed by one saying what it
implies. No equation stands alone.

10.10 **Re-derive, don't refer back.** When a later chapter needs an
earlier result, it restates it in one line rather than sending the
reader back forty pages.

## 11. Currency: the field moves faster than the manuscript

The authors' knowledge of the field has a date; the field does not.
Model names, sizes, prices, engine versions, hardware specifications,
and benchmark results change monthly. These rules keep the book
current and honest about when each fact was true.

11.1 **The facts register.** Every time-sensitive claim in the book is
an entry in `FACTS.md`: the claim, the value, the primary source, and
the date it was last verified. A claim not in the register may not
appear in the manuscript.

11.2 **Verify at writing time, not from memory.** Before a chapter is
drafted, every register entry it uses is re-checked against a current
primary source (release notes, datasheets, price sheets, model cards)
by web search. A value that cannot be verified is marked
`UNVERIFIED` and does not ship.

11.3 **Date every number in the text.** Prices, versions, and rankings
appear with their verification month: "the H100 on-demand median was
$3.25/hr (September 2026)". A number with no date is a defect.

11.4 **Physics is stable; products are not.** Chapters are structured
so that mechanisms (bandwidth-bound decode, paging, batching) live in
the body and products (engine versions, flags, model names, prices)
live in tables, appendices, and the companion page. The body should
survive two years; the tables are re-verified at every revision.

11.5 **Model choices are decisions with criteria, not names.** The
running model is defined by requirements (dense, ~8B parameters,
permissive license, current generation, a ~1B sibling for Tier 1) and
the register records which model met them at each revision.

11.6 **Revision cadence.** The register is fully re-verified at each
numbered revision of the book, and the companion page publishes the
diff.
