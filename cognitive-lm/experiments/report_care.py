"""Aggregate results/care_seed*.json into RESULTS.md and figures/.

    python -m experiments.report_care
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = [json.load(open(p)) for p in sorted(glob.glob("results/care_seed*.json"))]
assert R, "no results/care_seed*.json"
os.makedirs("figures", exist_ok=True)
SIZES = list(R[0]["sizes"].keys())


def get(path, r):
    for p in path:
        r = r[p]
    return r


def vals(path):
    return np.array([get(path, r) for r in R], dtype=float)


def pm(path, pct=True, dec=1):
    v = vals(path)
    f = 100 if pct else 1
    return f"{f * v.mean():.{dec}f} ± {f * v.std():.{dec}f}"


def params(d):
    return R[0]["sizes"][d]["params"]


L = [f"# Results: checksum-aided recall ({len(R)} seeds, mean ± std)\n"]
w = R[0]["world"]
f = R[0]["filters"]
L.append(f"Hard world: 10,000 people × 4 relations, answers are 2-token strings from 1,024 values. "
         f"{w['seen_facts']:,} distinct facts seen in training, singleton rate N1/M = {w['singleton_rate']:.3f}, "
         f"never-seen share of test queries = {vals(['world', 'test_unseen']).mean():.3f}. "
         f"Measured filter false-positive rates: triple checksum {f['eps_triple']:.5f} (14 bits/fact), "
         f"key filter {f['eps_key']:.4f}, entity filter {f['eps_entity']:.4f}.\n")
L.append("Accuracy = correct answers / all queries. Hallucination = wrong answers / all queries (any answer "
         "to a never-seen fact counts). Weights-only systems are shown at their best confidence threshold "
         "under each hallucination budget.\n")

# ---- main table -----------------------------------------------------------------------
L.append("## 1. Main result\n")
L.append("| weights | params | weights only: greedy acc / halluc | weights only: acc @ hal ≤ 1% | + IDK training: acc @ hal ≤ 1% "
         "| **CAR N=1** acc / halluc | **CAR+prefix N=50** acc / halluc | **CARE** acc / halluc | CARE store (% of facts) |")
L.append("|---|---|---|---|---|---|---|---|---|")
for d in SIZES:
    s = ["sizes", d]
    L.append(f"| d={d} | {params(d):,} | {pm(s + ['weights', 'default', 'accuracy'])} / {pm(s + ['weights', 'default', 'hallucination'])} "
             f"| {pm(s + ['weights', 'acc_at_hal_1pct'])} | {pm(s + ['weights_idk', 'acc_at_hal_1pct'])} "
             f"| {pm(s + ['CAR_N1', 'accuracy'])} / {pm(s + ['CAR_N1', 'hallucination'], dec=2)} "
             f"| {pm(s + ['CAR_prefix_N50', 'accuracy'])} / {pm(s + ['CAR_prefix_N50', 'hallucination'], dec=2)} "
             f"| {pm(s + ['CARE_N20', 'accuracy'])} / {pm(s + ['CARE_N20', 'hallucination'], dec=2)} "
             f"| {pm(s + ['CARE_N20', 'store_share_of_seen_facts'], dec=0)} |")
L.append(f"| store only (no weights) | 0 | | | | | | {pm(['store_only', 'accuracy'])} / {pm(['store_only', 'hallucination'], dec=2)} | 100 |")
L.append("")

# ---- memory -------------------------------------------------------------------------------
L.append("## 2. Memory outside the weights (bits per seen fact)\n")
L.append("| system | non-weight bits / seen fact | accuracy | hallucination |")
L.append("|---|---|---|---|")
d0 = SIZES[-2] if len(SIZES) > 1 else SIZES[0]
rows = [("store only (exact key + value + count, plus filters)", ["store_only"]),
        ("no weights: full scan of all 1,024 values, 14-bit checksum", ["no_weights", "full_scan_N1024"]),
        ("no weights: full scan, 20-bit checksum", ["no_weights", "full_scan_N1024_b20"]),
        ("no weights: full scan, 28-bit checksum", ["no_weights", "full_scan_N1024_b28"]),
        ("no weights: prefix checksum, random order, N=50", ["no_weights", "prefix_random_order_N50"]),
        (f"CAR N=1, weights d={d0}", ["sizes", d0, "CAR_N1"]),
        (f"CAR N=20, weights d={d0}", ["sizes", d0, "CAR_N20"]),
        (f"CAR+prefix N=50, weights d={d0}", ["sizes", d0, "CAR_prefix_N50"]),
        (f"CARE N=20, weights d={d0}", ["sizes", d0, "CARE_N20"])]
for lab, p in rows:
    L.append(f"| {lab} | {vals(p + ['non_weight_bits_per_seen_fact']).mean():.1f} | {pm(p + ['accuracy'])} | {pm(p + ['hallucination'], dec=2)} |")
L.append("")

# ---- list size ------------------------------------------------------------------------------
L.append("## 3. List size N, and the list-decoding formula (docs/theory.md §5)\n")
L.append("| weights | N | accuracy (measured) | accuracy (predicted) | hallucination (measured) | hallucination (predicted) | tip of the tongue |")
L.append("|---|---|---|---|---|---|---|")
for d in SIZES:
    for N in [1, 5, 20]:
        p = ["sizes", d, f"CAR_N{N}"]
        L.append(f"| d={d} | {N} | {pm(p + ['accuracy'])} | {pm(p + ['predicted', 'accuracy'])} | {pm(p + ['hallucination'], dec=2)} "
                 f"| {pm(p + ['predicted', 'hallucination'], dec=2)} | {pm(p + ['levels', 'tip_of_tongue'])} |")
L.append("")

# ---- graded ---------------------------------------------------------------------------------
L.append("## 4. Graded 'I don't know' (share of all queries, CARE, largest weights)\n")
dl = SIZES[-1]
p = ["sizes", dl, "CARE_N20", "levels"]
L.append(f"- never heard of this person: {pm(p + ['unknown_entity'])}%")
L.append(f"- know the person, never read this fact: {pm(p + ['unknown_fact'])}%")
L.append(f"- tip of the tongue (read it, can't bring it back): {pm(p + ['tip_of_tongue'], dec=2)}%")
L.append(f"- ghost people (never existed) answered: {pm(['sizes', dl, 'CARE_N20', 'ghost_hallucination'], dec=2)}% "
         f"(weights only, greedy: {pm(['sizes', dl, 'weights', 'default', 'ghost_hallucination'])}%; "
         f"IDK-trained: {pm(['sizes', dl, 'weights_idk', 'default', 'ghost_hallucination'])}%)\n")

# ---- noise -----------------------------------------------------------------------------------
L.append("## 5. Noisy fact extraction (mentions filed under the wrong person), weights d=32\n")
L.append("| mislinks | weights greedy halluc | store only acc / halluc | CAR N=1 | CAR N=5 | CAR N=20 | CARE | CARE strict |")
L.append("|---|---|---|---|---|---|---|---|")
for i, row in enumerate(R[0]["noise"]):
    def cell(k):
        a = np.array([r["noise"][i][k]["accuracy"] for r in R])
        h = np.array([r["noise"][i][k]["hallucination"] for r in R])
        return f"{100 * a.mean():.1f} / {100 * h.mean():.2f}"
    g = np.mean([r["noise"][i]["weights_greedy"] for r in R])
    L.append(f"| {int(100 * row['mislink'])}% | {100 * g:.1f} | {cell('store_only')} | {cell('CAR_N1')} | {cell('CAR_N5')} "
             f"| {cell('CAR_N20')} | {cell('CARE_N20')} | {cell('CARE_N20_strict')} |")
L.append("\nCARE strict answers a fact read only once and not recalled by the weights with \"I read that once, "
         "but can't confirm it\" instead of asserting it.\n")

# ---- ablations -------------------------------------------------------------------------------
dk = next((d for d in SIZES if "k2_CAR_N1" in R[0]["sizes"][d]), None)
if dk:
    L.append(f"## 6. Ablation: weights trained only on facts seen ≥ 2 times (d={dk})\n")
    L.append("| | all facts | facts seen ≥ 2× |")
    L.append("|---|---|---|")
    L.append(f"| weights greedy accuracy | {pm(['sizes', dk, 'weights', 'default', 'accuracy'])} | {pm(['sizes', dk, 'k2_weights_greedy'])} |")
    for k, lab in [("CAR_N1", "CAR N=1"), ("CAR_N5", "CAR N=5"), ("CAR_prefix_N50", "CAR+prefix N=50")]:
        L.append(f"| {lab} accuracy / halluc | {pm(['sizes', dk, k, 'accuracy'])} / {pm(['sizes', dk, k, 'hallucination'], dec=2)} "
                 f"| {pm(['sizes', dk, 'k2_' + k, 'accuracy'])} / {pm(['sizes', dk, 'k2_' + k, 'hallucination'], dec=2)} |")
    L.append("")

open("RESULTS.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---- figures ---------------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 140, "font.size": 9})
cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for i, d in enumerate(SIZES):
    fr = np.array([[[p["hallucination"], p["accuracy"]] for p in r["sizes"][d]["weights"]["frontier"]] for r in R]).mean(0)
    ax.plot(np.maximum(fr[:, 0], 1e-5), fr[:, 1], "--", color=cols[i], lw=1, label=f"weights only d={d}")
    for k, mk in [("CAR_N1", "o"), ("CAR_prefix_N50", "s"), ("CARE_N20", "*")]:
        h = max(vals(["sizes", d, k, "hallucination"]).mean(), 1e-5)
        ax.plot(h, vals(["sizes", d, k, "accuracy"]).mean(), mk, color=cols[i], ms=8 if mk == "*" else 5)
ax.plot([], [], "ko", ms=5, label="+ checksum, N=1")
ax.plot([], [], "ks", ms=5, label="+ prefix checksum, N=50")
ax.plot([], [], "k*", ms=8, label="CARE")
ax.set_xscale("log")
ax.set_xlabel("hallucination rate (log; 1e-5 = zero)")
ax.set_ylabel("accuracy")
ax.set_title("Same weights, with and without a knowledge checksum")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig("figures/frontier.png")

fig, ax = plt.subplots(figsize=(6.0, 3.8))
xs = [params(d) for d in SIZES]
for key, lab, ls in [(["weights", "acc_at_hal_1pct"], "weights only (best at ≤1% halluc)", "--"),
                     (["weights_idk", "acc_at_hal_1pct"], "weights + IDK training (≤1%)", ":"),
                     (["CAR_N1", "accuracy"], "+ checksum N=1", "-"),
                     (["CAR_prefix_N50", "accuracy"], "+ prefix checksum N=50", "-"),
                     (["CARE_N20", "accuracy"], "CARE", "-")]:
    m = [vals(["sizes", d] + key).mean() for d in SIZES]
    s = [vals(["sizes", d] + key).std() for d in SIZES]
    ax.errorbar(xs, m, yerr=s, fmt=ls + "o", ms=4, capsize=2, label=lab)
ax.axhline(vals(["store_only", "accuracy"]).mean(), color="grey", lw=0.8, label="ceiling (every seen fact)")
ax.set_xscale("log")
ax.set_xlabel("parameters in the weights")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy with hallucination held under 1%")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig("figures/scaling.png")

fig, ax = plt.subplots(figsize=(4.4, 4.0))
for d in SIZES:
    for N in [1, 5, 20]:
        p = ["sizes", d, f"CAR_N{N}"]
        ax.plot(vals(p + ["predicted", "hallucination"]).mean(), vals(p + ["hallucination"]).mean(), "o", ms=4)
lim = ax.get_xlim()[1]
ax.plot([0, lim], [0, lim], "k:", lw=0.8)
ax.set_xlabel("predicted hallucination (formula)")
ax.set_ylabel("measured hallucination")
ax.set_title("List-decoding formula vs measurement")
fig.tight_layout()
fig.savefig("figures/theory_check.png")
print("figures written")
