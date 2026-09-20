# CLAUDE.md

## Re-verify before claiming anything is done

This applies to every chapter, outline, table, number, quote and
citation — anything written here.

- **Numbers** come from a results file, never from memory or an earlier
  draft. Re-run the check (`make check` in
  `proposals/llm-inference/code`) and quote what it prints.
- **Facts that age** — model names and sizes, prices, versions, hardware
  specs, which books already exist — are re-checked by web search at
  writing time, never recalled. Record value, source and date in
  `FACTS.md`. My training has a cutoff; the field does not.
- **Citations** — confirm the paper, authors and venue exist and say
  what I claim they say. Never cite from memory alone.
- **Prose** — re-read what I just wrote against the source it came
  from, including cross-references and chapter numbers, before
  committing.
- **Code** — run it. A listing in a chapter must come from a file that
  executes.

If something cannot be verified, mark it `UNVERIFIED` and say so.
Never ship a confident-sounding guess.

## Finish a chapter before starting the next

A chapter is not done when the prose is written. Before moving on:

- Run `make check` in `proposals/llm-inference/code`. It runs the
  cross-reference resolver, the code assertions, the whole-manuscript
  audit and the drift check. All four must pass.
- Re-read the chapter against the outline and the standards: is a
  promised section missing, a figure generated but never shown, a
  number quoted that nothing computes, a claim left unsourced?
- Check the chapter's numbers against the chapters around it. The same
  quantity must not have two answers in two places.
- Say what you verified and what you changed. Only then start the next
  chapter.

Finding a mistake here is the system working. Fix the cause, not just
the symptom: if a number could drift, make it generated; if a reference
could go stale, make it symbolic; if a fact lived in four files, give
it one home.

## Reporting

State what was verified and how. If a check was skipped or failed, say
so plainly. Never report work as done when its check has not passed.

Full rules for the inference book: `proposals/llm-inference/STANDARDS.md`.
