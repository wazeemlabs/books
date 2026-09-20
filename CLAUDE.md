# CLAUDE.md

## Never skip

Before calling anything done, in `proposals/llm-inference/code`:

    make check    # cross-references, code tests, manuscript audit, drift

All four gates must pass. Then re-read the chapter against `OUTLINE.md`
and `STANDARDS.md`, and check its numbers against the chapters around
it. Only then start the next one.

## Never guess

- **Numbers** — generated from a results file, never typed or recalled.
- **Ageing facts** (models, prices, versions, specs, rival books) —
  re-check by web search at writing time; record value, source and date
  in `FACTS.md`. My training has a cutoff; the field does not.
- **Citations** — confirm the authors, the venue, and that it says what
  I claim.
- **Code** — run it. Every listing comes from a file that executes.
- **Figures** — open the rendered PNG and read it. Labels on curves,
  labels on each other, legends over data and bars drawn off the top
  of a panel are all invisible in the code that produced them.
- **Terms** — the reader knows nothing. Explain every term on first
  use, intermediate ones included, then give its name. Add it to
  `GLOSSARY.md` and claim it with `<!-- defines: term -->`.

Cannot verify it? Mark it `UNVERIFIED` and say so.

## On finding a mistake

Fix the cause, not the symptom: a number that could drift becomes
generated, a reference that could go stale becomes symbolic, a fact
living in four files gets one home.

## Reporting

Say what you verified and what you changed. Never report work as done
when a check has not passed.

Full rules: `proposals/llm-inference/STANDARDS.md`.
