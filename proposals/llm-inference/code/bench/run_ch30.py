"""Chapter 30: drafting without a second model.

Five measurements:

1. What a draft head weighs, exactly, from the architecture its
   authors published -- and what that does to every decode step,
   including the ones where the draft is wrong.
2. How often a draft built from nothing but the prompt is right, on
   four shapes of task, measured over real prose.
3. What that buys, once the guesses are arranged as a tree instead of
   a chain and verified in one pass.
4. Where the tree stops paying, which is not where the acceptance rate
   runs out.
5. What a head has to achieve to earn back its own weight.

    python3 -m bench.run_ch30
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tinyserve.cost import flops_forward
from tinyserve.drafting import (best_shape, chain_shape, coverage, eagle_cost,
                                expected_accepted, medusa_cost, ngram_cost,
                                nodes, reach, step_cost_ratio, tokens,
                                zipf_cover)
from tinyserve.reference import (CONTEXT_TOKENS, HBM_BYTES_PER_S,
                                 KV_BYTES_PER_TOKEN, MODEL, PEAK_BF16_FLOPS,
                                 PROMPT_TOKENS, WEIGHT_BYTES)
from tinyserve.serving import decode_step

from .harness import write

SEED = 0
CHAPTERS = Path("../chapters")
# The document the tasks are grounded in, and the one used for the
# control. Three chapters make a prompt of a realistic length for an
# input-grounded task; a fourth, never shown to the drafter, is what
# "unrelated" means.
DOCUMENT = ["ch13", "ch14", "ch15"]
ELSEWHERE = "ch22"
OUTPUT_TOKENS = 1_200
SUMMARY_SENTENCES = 45
EDIT_RATE = 0.10                 # words changed in the edited extract
BREADTHS = [1, 2, 4, 8]
DEPTHS = 8
MIN_N, MAX_N = 2, 8
BUDGETS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]
BATCHES = [1, 8, 32, 128]
# EAGLE publishes a parameter count per target model rather than an
# architecture; this is the one for the smallest model in its table,
# which is the closest to the reference model's size (FACTS.md).
EAGLE_PARAMETERS = 240_000_000
EAGLE_FOR = "Vicuna-7B"

CODE = re.compile(r"```.*?```", re.S)
INLINE = re.compile(r"`[^`]*`")
MARKER = re.compile(r"<!--.*?-->", re.S)
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
LINK = re.compile(r"\]\([^)]*\)")
TABLE = re.compile(r"^\|.*$", re.M)
HEADING = re.compile(r"^#+\s*", re.M)


def prose(path: Path) -> str:
    """A chapter's words, with everything that is not prose removed.

    Code, tables, markers and placeholders are not what a model writes
    and not what a reader reads; leaving them in would measure the
    repetitiveness of markdown rather than of English.
    """
    text = path.read_text()
    for pattern in (CODE, INLINE, MARKER, PLACEHOLDER, TABLE):
        text = pattern.sub(" ", text)
    text = LINK.sub("] ", text)
    return HEADING.sub("", text)


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]


def tasks(rng) -> dict[str, dict]:
    """Four shapes of request, each a prompt and what the reply turns
    out to be.

    The first two are what prompt lookup decoding is sold for: a reply
    grounded in the prompt. The third is a reply that continues the
    prompt without quoting it. The fourth is the control, where the
    reply has nothing to do with the prompt at all.
    """
    doc = " ".join(prose(CHAPTERS / f"{c}.md") for c in DOCUMENT)
    other = prose(CHAPTERS / f"{ELSEWHERE}.md")
    doc_tokens, other_tokens = tokens(doc), tokens(other)
    sents = sentences(doc)
    vocabulary = sorted(set(doc_tokens))

    picked = sorted(rng.choice(len(sents), SUMMARY_SENTENCES, replace=False))
    extract = tokens(" ".join(sents[i] for i in picked))[:OUTPUT_TOKENS]

    # The same extract with one word in ten swapped for another word
    # from the document: what an edit or a paraphrase does to the
    # overlap, without pretending to be a model rewriting it.
    edited = list(extract)
    for i in range(len(edited)):
        if edited[i].isalpha() and rng.random() < EDIT_RATE:
            edited[i] = str(rng.choice(vocabulary))

    cut = int(len(doc_tokens) * 0.6)
    return {
        "an extract from the prompt":
            {"prompt": doc_tokens, "truth": extract},
        "an extract, lightly edited":
            {"prompt": doc_tokens, "truth": edited},
        "prose continuing the prompt":
            {"prompt": doc_tokens[:cut],
             "truth": doc_tokens[cut:cut + OUTPUT_TOKENS]},
        "text unrelated to the prompt":
            {"prompt": doc_tokens, "truth": other_tokens[:OUTPUT_TOKENS]},
    }


# --- 1. what a head weighs -----------------------------------------------


def heads() -> dict:
    costs = [medusa_cost(heads=n) for n in (1, 2, 3, 5)]
    costs.append(eagle_cost(EAGLE_PARAMETERS))
    costs.append(ngram_cost())
    rows = [{"name": c.name, "parameters": c.parameters, "bytes": c.bytes,
             "share_of_model": c.share_of_model, "note": c.note,
             "step_ratio": step_cost_ratio(c),
             "speedup_to_break_even": step_cost_ratio(c)}
            for c in costs]
    return {"rows": rows, "weight_bytes": WEIGHT_BYTES,
            "vocab": MODEL.vocab_size, "d_model": MODEL.d_model,
            "eagle_for": EAGLE_FOR}


# --- 2. how often a prompt-built draft is right --------------------------


def drafting(task_set: dict) -> list[dict]:
    rows = []
    for name, task in task_set.items():
        out = coverage(task["prompt"], task["truth"], DEPTHS, BREADTHS,
                       min_n=MIN_N, max_n=MAX_N)
        cover = out["cover"]
        rows.append({"task": name, "prompt_tokens": len(task["prompt"]),
                     **{k: v for k, v in out.items() if k != "cover"},
                     "cover": {str(b): v for b, v in cover.items()},
                     # The same rates read the way a reader should read
                     # them: the chance of getting at least d tokens,
                     # rather than the chance of getting the d-th given
                     # the first d-1.
                     "reach": {str(b): reach(chain_shape(DEPTHS), cover)
                               for b in cover}})
    return rows


# --- 3 and 4. chains, trees, and what they are worth ---------------------


def round_seconds(batch: int, n_nodes: int, context: int) -> float:
    """One verification pass over `n_nodes` tokens for each of `batch`
    sequences, under the book's roofline.

    The weights are fetched once for the whole batch, as in any decode
    step. Each sequence's own cache is its own. The arithmetic is a
    forward pass over `n_nodes` positions per sequence, attending back
    over its context -- sequences do not attend to each other, so it is
    counted per sequence and summed.
    """
    bytes_read = WEIGHT_BYTES + batch * context * KV_BYTES_PER_TOKEN
    flops = batch * flops_forward(MODEL, n_nodes, context + n_nodes)
    return max(bytes_read / HBM_BYTES_PER_S, flops / PEAK_BF16_FLOPS)


def payoff(cover: dict[int, list[float]], batch: int, context: int,
           draft_ratio: float = 1.0) -> list[dict]:
    """What each node budget is worth, as a chain and as the best tree.

    `draft_ratio` is what the drafter does to a decode step: 1.0 for an
    n-gram index, which costs nothing a GPU can see, and more than that
    for a head whose weights have to be read.
    """
    base = decode_step(batch, context).seconds * draft_ratio
    rows = []
    for budget in BUDGETS:
        k = max(1, budget)
        chain = chain_shape(min(k, DEPTHS))
        while nodes(chain) > budget and chain:
            chain = chain[:-1]
        tree, tree_value, tree_nodes_used = best_shape(budget, cover, DEPTHS)
        chain_value = expected_accepted(chain, cover) if chain else 1.0
        rows.append({
            "budget": budget,
            "chain": list(chain), "chain_nodes": nodes(chain),
            "chain_tokens": chain_value,
            "chain_speedup": chain_value * base
                             / round_seconds(batch, max(nodes(chain), 1), context),
            "tree": list(tree), "tree_nodes": tree_nodes_used,
            "tree_tokens": tree_value,
            "tree_speedup": tree_value * base
                            / round_seconds(batch, max(tree_nodes_used, 1), context),
            "pass_seconds": round_seconds(batch, budget, context),
            "decode_seconds": base,
        })
    return rows


def where_it_stops(batch: int, context: int) -> dict:
    """The node count at which a verification pass stops being free.

    A pass over a handful of extra tokens costs nothing beyond reading
    the weights, which it was going to read anyway. Past some width it
    is doing enough arithmetic to be compute-bound, and every further
    node is paid for in full. That point, not the acceptance rate, is
    what caps a tree.
    """
    bytes_read = WEIGHT_BYTES + batch * context * KV_BYTES_PER_TOKEN
    memory_s = bytes_read / HBM_BYTES_PER_S
    lo, hi = 1, 1 << 20
    while lo < hi:
        mid = (lo + hi) // 2
        flops = batch * flops_forward(MODEL, mid, context + mid)
        if flops / PEAK_BF16_FLOPS >= memory_s:
            hi = mid
        else:
            lo = mid + 1
    return {"batch": batch, "context": context, "free_nodes": lo,
            "memory_seconds": memory_s}


# --- 5. what a tree is worth to a drafter that ranks properly -----------

ALPHAS = [0.3, 0.5, 0.7, 0.85]
TREE_BREADTHS = [1, 2, 4, 8, 16]
TREE_BUDGETS = [8, 16, 32, 64, 128]


def trees() -> dict:
    """Chain against best tree, for a drafter whose ranking means
    something.

    The n-gram drafter measured above gets almost nothing from a second
    candidate, so it cannot answer this question. A trained head can:
    its second-most-likely token is a real second guess. `zipf_cover`
    models that, pinned to one number -- the head's top-1 acceptance --
    and the rest of this is exact arithmetic over the model.
    """
    rows = []
    for alpha in ALPHAS:
        cover = zipf_cover(alpha, TREE_BREADTHS, DEPTHS)
        for budget in TREE_BUDGETS:
            chain = chain_shape(min(budget, DEPTHS))
            shape, value, used = best_shape(budget, cover, DEPTHS)
            base = expected_accepted(chain, cover)
            rows.append({
                "alpha": alpha, "budget": budget,
                "chain_tokens": base,
                "tree": list(shape), "tree_nodes": used,
                "tree_tokens": value,
                "gain": value / base - 1,
                "top_b": {str(b): cover[b][0] for b in TREE_BREADTHS},
            })
    return {"rows": rows, "alphas": ALPHAS, "breadths": TREE_BREADTHS,
            "budgets": TREE_BUDGETS, "model_not_measurement": True}


def best_budget(context: int, alpha: float = 0.7) -> list[dict]:
    """The tree worth verifying at each batch size.

    The papers report at batch 1, where a verification pass is free up
    to a few hundred tokens and a wide tree costs nothing. A server
    runs a batch. This searches the same budgets at each batch and
    reports the one that actually comes out fastest, and what the
    paper's setting would have done there instead.
    """
    cover = zipf_cover(alpha, TREE_BREADTHS, DEPTHS)
    rows = []
    for batch in BATCHES:
        base = decode_step(batch, context).seconds
        best, at = None, None
        for budget in TREE_BUDGETS + [1, 2, 3, 4, 6]:
            shape, value, used = best_shape(budget, cover, DEPTHS)
            if not shape:
                continue
            speed = value * base / round_seconds(batch, used, context)
            if best is None or speed > best:
                best, at = speed, (budget, shape, value, used)
        wide, wide_value, wide_used = best_shape(max(TREE_BUDGETS), cover,
                                                 DEPTHS)
        rows.append({
            "batch": batch, "alpha": alpha,
            "best_budget": at[0], "best_tree": list(at[1]),
            "best_nodes": at[3], "best_tokens": at[2], "best_speedup": best,
            "widest_nodes": wide_used, "widest_tokens": wide_value,
            "widest_speedup": wide_value * base
                              / round_seconds(batch, wide_used, context),
        })
    return rows


# --- 6. what a head has to achieve --------------------------------------


def demands(context: int) -> list[dict]:
    """The accepted length a drafter needs before it has paid for itself.

    A round costs the drafter's overhead on the step plus a
    verification pass over the whole tree, and yields whatever the tree
    accepted. Below the number here the whole apparatus is slower than
    decoding one token at a time, which at a production batch size it
    very easily is.
    """
    rows = []
    for cost in (medusa_cost(heads=3), eagle_cost(EAGLE_PARAMETERS),
                 ngram_cost()):
        ratio = step_cost_ratio(cost)
        for batch in BATCHES:
            base = decode_step(batch, context).seconds
            for budget in (3, 64):
                need = ratio * round_seconds(batch, budget, context) / base
                rows.append({
                    "name": cost.name, "batch": batch, "budget": budget,
                    "step_ratio": ratio,
                    "tokens_needed": need,
                    "of_the_budget": need / (budget + 1),
                })
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)
    task_set = tasks(rng)
    head_rows = heads()
    draft_rows = drafting(task_set)
    by_task = {r["task"]: {int(k): v for k, v in r["cover"].items()}
               for r in draft_rows}
    grounded = "an extract from the prompt"
    payoffs = {name: payoff(by_task[name], batch=1, context=CONTEXT_TOKENS)
               for name in by_task}
    batches = {str(b): payoff(by_task[grounded], batch=b,
                              context=CONTEXT_TOKENS)
               for b in BATCHES}
    free = [where_it_stops(b, CONTEXT_TOKENS) for b in BATCHES]

    payload = {
        "heads": head_rows,
        "drafting": draft_rows,
        "payoff": payoffs,
        "batches": batches,
        "free_nodes": free,
        "trees": trees(),
        "best_budget": best_budget(CONTEXT_TOKENS),
        "demands": demands(CONTEXT_TOKENS),
        "assumptions": {
            "seed": SEED, "document": DOCUMENT, "elsewhere": ELSEWHERE,
            "output_tokens": OUTPUT_TOKENS,
            "summary_sentences": SUMMARY_SENTENCES, "edit_rate": EDIT_RATE,
            "breadths": BREADTHS, "depths": DEPTHS,
            "min_n": MIN_N, "max_n": MAX_N, "budgets": BUDGETS,
            "batches": BATCHES, "context": CONTEXT_TOKENS,
            "prompt_tokens": PROMPT_TOKENS,
            "eagle_parameters": EAGLE_PARAMETERS, "eagle_for": EAGLE_FOR,
            "grounded_task": grounded,
            "alphas": ALPHAS, "tree_breadths": TREE_BREADTHS,
            "tree_budgets": TREE_BUDGETS,
            "model_not_measurement": True,
        },
        "model_not_measurement": True,
    }
    path = write("results/ch30.json", payload)
    print(f"wrote {path}")

    print("  what a draft head weighs:")
    for r in head_rows["rows"]:
        print(f"    {r['name']:<22}{r['parameters'] / 1e9:5.2f}B"
              f"{r['bytes'] / 1e9:7.2f} GB"
              f"{r['share_of_model'] * 100:7.1f}% of the model"
              f"   every step x{r['step_ratio']:.3f}")
    print(f"  drafting from the prompt alone, {DEPTHS} deep:")
    for r in draft_rows:
        c = {int(k): v for k, v in r["cover"].items()}
        print(f"    {r['task']}: a guess at all {r['had_a_guess'] * 100:.0f}% "
              f"of positions")
        for b in BREADTHS:
            line = "  ".join(f"{v * 100:4.0f}" for v in c[b][:5])
            print(f"      top-{b}: {line}   (depth 1..5, % right)")
    print("  what it is worth, on the grounded task, batch 1:")
    for r in payoffs[grounded]:
        print(f"    budget {r['budget']:>4}: chain {r['chain_tokens']:5.2f} tok "
              f"x{r['chain_speedup']:5.2f}   tree {str(tuple(r['tree'])):<20}"
              f"{r['tree_tokens']:5.2f} tok x{r['tree_speedup']:5.2f}")
    print("  where a verification pass stops being free:")
    for f in free:
        print(f"    batch {f['batch']:>4}: {f['free_nodes']:>6,} nodes")
    print("  what a tree is worth to a drafter that ranks (a model):")
    for r in payload["trees"]["rows"]:
        if r["budget"] in (8, 64):
            print(f"    top-1 {r['alpha']:.2f}, {r['budget']:>3} nodes: "
                  f"chain {r['chain_tokens']:5.2f} -> tree "
                  f"{str(tuple(r['tree'])):<24}{r['tree_tokens']:5.2f} "
                  f"({r['gain'] * 100:+.0f}%)")
    print("  the tree worth verifying at each batch (modelled, top-1 0.70):")
    for r in payload["best_budget"]:
        print(f"    batch {r['batch']:>4}: best {str(tuple(r['best_tree'])):<24}"
              f"{r['best_nodes']:>4} nodes  x{r['best_speedup']:5.2f}   "
              f"the widest tree ({r['widest_nodes']} nodes) would give "
              f"x{r['widest_speedup']:5.2f}")
    print("  accepted tokens a round must yield to break even:")
    for r in payload["demands"]:
        if r["budget"] == 64:
            print(f"    {r['name']:<22} batch {r['batch']:>4}, 64 nodes: "
                  f"{r['tokens_needed']:6.2f} tokens "
                  f"({r['of_the_budget'] * 100:.0f}% of what the tree offers)")


if __name__ == "__main__":
    main()
