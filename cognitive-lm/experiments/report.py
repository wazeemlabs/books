"""Aggregate results/seed*.json into figures and a markdown table.

    python -m experiments.report
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = [json.load(open(p)) for p in sorted(glob.glob("results/seed*.json"))]
os.makedirs("figures", exist_ok=True)
SIZES = sorted({int(k.split("_d")[1]) for k in R[0]["param"]})


def ms(vals):
    v = np.array(vals, dtype=float)
    return v.mean(), v.std()


def get(path):
    out = []
    for r in R:
        x = r
        for p in path:
            x = x[p]
        out.append(x)
    return out


def pm(path, pct=True):
    m, s = ms(get(path))
    return f"{100 * m:.1f} ± {100 * s:.1f}" if pct else f"{m:.3f} ± {s:.3f}"


lines = [f"# Results ({len(R)} seeds, mean ± std)\n"]
w = R[0]["world"]
lines.append(f"World: {w['facts']} facts, {w['seen']} seen, {w['singletons']} singletons, "
             f"singleton rate N1/M = {w['singleton_rate']:.3f}, missing mass = {w['missing_mass']:.3f}, "
             f"unseen share of test queries = {w['test_unseen']:.3f}.\n")

# ---- 1. membership ----------------------------------------------------------
lines.append("## 1. Can the weights tell 'seen once' from 'never seen'? (AUROC, 0.5 = coin flip)\n")
lines.append("| model | params | seen 1x vs unseen | seen 2-3x | seen 4-7x | seen 8x+ |")
lines.append("|---|---|---|---|---|---|")
for kind in ["plain", "idk"]:
    for d in SIZES:
        k = f"{kind}_d{d}"
        mem = ["param", k, "membership"]
        lines.append(f"| {'IDK-trained' if kind == 'idk' else 'plain'} LM d={d} | {R[0]['param'][k]['params']:,} | "
                     + " | ".join(pm(mem + [f"auroc_seen{b}_vs_unseen"], pct=False) for b in ["1", "2-3", "4-7", "8+"]) + " |")
lines.append("| Bloom filter, 10 bits/key | 0 | ≈ 0.995 (1 − FPR/2) | same | same | same |\n")

# ---- 2. main comparison -----------------------------------------------------
lines.append("## 2. Accuracy, hallucination and memory on the natural query stream\n")
lines.append("Accuracy = correct answers / all queries. Hallucination = wrong answers / all queries "
             "(any answer to a never-seen fact counts as a hallucination). "
             "Acc@hal≤1% is the best accuracy the model reaches at some confidence threshold "
             "while keeping hallucination at or under 1%.\n")
lines.append("| system | acc (default) | halluc (default) | acc @ hal≤1% | acc @ hal≤5% | ghost-person halluc | total bits |")
lines.append("|---|---|---|---|---|---|---|")
for kind in ["plain", "idk"]:
    for d in SIZES:
        k = f"{kind}_d{d}"
        lines.append(f"| {'IDK-trained' if kind == 'idk' else 'plain'} LM d={d} | {pm(['param', k, 'default', 'accuracy'])} | "
                     f"{pm(['param', k, 'default', 'hallucination'])} | {pm(['param', k, 'acc_at_hal_1pct'])} | "
                     f"{pm(['param', k, 'acc_at_hal_5pct'])} | {pm(['param', k, 'default', 'ghost_hallucination'])} | "
                     f"{R[0]['param'][k]['bits'] / 1e6:.2f} M |")
for k in R[0]["cmlm"]:
    b = np.mean(get(["cmlm", k, "extra", "total_bits"]))
    lines.append(f"| {k} | {pm(['cmlm', k, 'accuracy'])} | {pm(['cmlm', k, 'hallucination'])} | "
                 f"{pm(['cmlm', k, 'accuracy'])}* | {pm(['cmlm', k, 'accuracy'])}* | "
                 f"{pm(['cmlm', k, 'ghost_hallucination'])} | {b / 1e6:.2f} M |")
lines.append("\n\\* the complementary systems are run at a single operating point; their hallucination "
             "is already under 1%, so their default accuracy is their acc@hal≤1%.\n")

# ---- 3. consolidation sweep ----------------------------------------------------
lines.append("## 3. Consolidation threshold k (facts seen ≥ k times are consolidated into the core)\n")
lines.append("| system | store entries | answered from store | answered from core | core recall on facts seen ≥10x | consolidation failures | total bits |")
lines.append("|---|---|---|---|---|---|---|")
for k in R[0]["cmlm"]:
    if not k.startswith("cmlm_"):
        continue
    lines.append(f"| {k} | {np.mean(get(['cmlm', k, 'extra', 'store_entries'])):.0f} | "
                 f"{pm(['cmlm', k, 'extra', 'store_answered'])} | {pm(['cmlm', k, 'extra', 'core_answered'])} | "
                 f"{pm(['cmlm', k, 'core_recall_count10plus'])} | "
                 f"{np.mean(get(['cmlm', k, 'failed_consolidation'])):.0f} | "
                 f"{np.mean(get(['cmlm', k, 'extra', 'total_bits'])) / 1e6:.2f} M |")

# Good-Turing prediction of store load, as a share of queries about SEEN facts
from cogllm.world import WorldConfig, build_world  # noqa: E402

lines.append("\nGood-Turing check (theory note, P4). Share of seen-fact queries that still need the store, "
             "predicted from the training count histogram alone vs measured:\n")
lines.append("| system | predicted Σ_{n<k} θ_n / (1 − N1/M) | measured |")
lines.append("|---|---|---|")
for key in R[0]["cmlm"]:
    if not key.startswith("cmlm_"):
        continue
    k = int(key.split("_k")[1])
    pred, meas = [], []
    for r in R:
        c = r["config"]
        wd = build_world(WorldConfig(n_first=160, n_last=160, n_entities=c["n_entities"], zipf_s=c["zipf"],
                                     n_mentions=c["n_mentions"], seed=c["seed"]))
        N = np.bincount(wd.counts.ravel())
        M = len(wd.stream)
        theta = [(n + 1) * (N[n + 1] if n + 1 < len(N) else 0) / M for n in range(len(N))]
        pred.append(sum(theta[1:k]) / (1 - N[1] / M))
        meas.append(r["cmlm"][key]["extra"]["store_answered"] / (1 - r["world"]["test_unseen"]))
    lines.append(f"| {key} | {100 * np.mean(pred):.1f}% | {100 * np.mean(meas):.1f}% ± {100 * np.std(meas):.1f} |")

# ---- 4. IDK tax + graded abstention ----------------------------------------------
lines.append("\n## 4. Does 'I don't know' training cost the weights recall? (forced answer on seen facts)\n")
lines.append("| size | plain LM | IDK-trained LM |")
lines.append("|---|---|---|")
for d in SIZES:
    lines.append(f"| d={d} | {pm(['param', f'plain_d{d}', 'forced_recall_seen'])} | {pm(['param', f'idk_d{d}', 'forced_recall_seen'])} |")
lines.append("\nIDK-trained LM: how often it answers, by exposure count (default threshold)\n")
lines.append("| size | never seen | seen 1x | 2-3x | 4-7x | 8x+ | ghost person |")
lines.append("|---|---|---|---|---|---|---|")
for d in SIZES:
    k = f"idk_d{d}"
    lines.append(f"| d={d} | " + " | ".join(pm(['param', k, 'answer_rate_by_count', b]) for b in ["0", "1", "2-3", "4-7", "8+"])
                 + f" | {pm(['param', k, 'answer_rate_ghost'])} |")
lines.append("\n## 5. Graded abstention (complementary systems)\n")
k0 = next(k for k in R[0]["cmlm"] if k.startswith("cmlm_"))
lines.append(f"On never-seen facts, the right kind of 'I don't know' (unknown person vs known person, unknown fact): "
             f"{pm(['cmlm', k0, 'graded_abstain_acc_on_unseen'])}. "
             f"Ghost people labelled 'never heard of them': {pm(['cmlm', k0, 'ghost_says_unknown_person'])}.\n")
open("RESULTS.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figures ------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
buckets = ["1", "2-3", "4-7", "8+"]
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for kind, ls in [("plain", "--"), ("idk", "-")]:
    for d in SIZES:
        k = f"{kind}_d{d}"
        y = [np.mean(get(["param", k, "membership", f"auroc_seen{b}_vs_unseen"])) for b in buckets]
        ax.plot(buckets, y, ls, marker="o", label=f"{'IDK-trained' if kind == 'idk' else 'plain'} d={d}")
ax.axhline(0.995, color="k", lw=1.5, label="Bloom filter (10 bits/key)")
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.set_xlabel("times the fact was seen in training")
ax.set_ylabel("AUROC: seen vs never seen")
ax.set_title("Weights are a poor membership test for rare facts")
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig("figures/membership.png")

fig, ax = plt.subplots(figsize=(5.2, 3.6))
for kind, ls in [("plain", "--"), ("idk", "-")]:
    for d in SIZES:
        k = f"{kind}_d{d}"
        pts = R[0]["param"][k]["frontier"]
        hal = np.mean([[p["hallucination"] for p in r["param"][k]["frontier"]] for r in R], 0)
        acc = np.mean([[p["accuracy"] for p in r["param"][k]["frontier"]] for r in R], 0)
        ax.plot(hal, acc, ls, marker=".", label=f"{'IDK-trained' if kind == 'idk' else 'plain'} d={d}")
for k, mk in [("store_only", "s"), (next(k for k in R[0]["cmlm"] if k.startswith("cmlm_d32")), "*"),
              (next(k for k in R[0]["cmlm"] if k.startswith("cmlm_d64")), "P")]:
    ax.plot(np.mean(get(["cmlm", k, "hallucination"])), np.mean(get(["cmlm", k, "accuracy"])), mk, ms=10, label=k)
ax.set_xlabel("hallucination rate (wrong answers / all queries)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs hallucination (confidence-threshold sweep)")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/frontier.png")

fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
for d in sorted({int(k.split("_d")[1].split("_")[0]) for k in R[0]["cmlm"] if k.startswith("cmlm_")}):
    ks = [k for k in R[0]["cmlm"] if k.startswith(f"cmlm_d{d}_")]
    kv = [int(k.split("_k")[1]) for k in ks]
    axs[0].plot(kv, [np.mean(get(["cmlm", k, "extra", "store_entries"])) for k in ks], "o-", label=f"core d={d}")
    axs[1].plot(kv, [np.mean(get(["cmlm", k, "extra", "store_answered"])) for k in ks], "o-", label=f"core d={d}")
    axs[2].plot(kv, [np.mean(get(["cmlm", k, "core_recall_count10plus"])) for k in ks], "o-", label=f"core d={d}")
axs[0].set_title("episodic store size")
axs[1].set_title("share of queries needing the store")
axs[2].set_title("core recall on facts seen ≥10x")
for a in axs:
    a.set_xscale("log")
    a.set_xlabel("consolidation threshold k")
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/consolidation.png")
print("figures written")
