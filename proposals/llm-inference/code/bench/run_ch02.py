"""Chapter 2: trace one prompt through the model, and record everything.

The chapter shows what a model does when it answers, so every figure in
it is drawn from a real forward pass rather than sketched. This script
runs that pass and records the tensors.

The model here is deliberately tiny -- a 48-word vocabulary, so every
score fits on a page and every token has a name. Chapter 10 puts a
realistic model on the bench.

    python3 -m bench.run_ch02
"""

from __future__ import annotations

import numpy as np

from tinyserve.model import Config, KVCache, build, forward, softmax

from .harness import write

# A toy vocabulary. Real tokenizers split into sub-word pieces; whole
# words keep this chapter readable, and Chapter 10 uses a real one.
WORDS = [
    "<pad>", "<end>", "the", "cat", "sat", "on", "mat", "dog", "ran", "to",
    "park", "a", "big", "small", "and", "it", "was", "sunny", "day", "she",
    "he", "they", "walked", "home", "then", "slept", "under", "tree", "near",
    "river", "birds", "sang", "all", "morning", "wind", "blew", "leaves",
    "fell", "slowly", "down", "children", "played", "outside", "until",
    "evening", "came", "quietly", ".",
]
INDEX = {w: i for i, w in enumerate(WORDS)}
PROMPT = ["the", "cat", "sat", "on", "the"]
TOP_K = 10

# For the "what this costs" comparison at the end of the chapter.
REF_PARAMS, REF_WEIGHT_BYTES = 8e9, 16e9
REF_KV_BYTES_PER_TOKEN = 131_072
REF_HBM_BYTES_PER_S = 3.35e12  # H100 SXM; FACTS.md


def _parameter_split(cfg: Config) -> dict:
    """Where a layer's parameters actually sit: attention, or feed-forward.

    The chapter claims most of them are in the feed-forward tables. That
    is checkable from the shapes, so it is checked rather than asserted.
    """
    d, hd = cfg.d_model, cfg.head_dim
    attention = (d * cfg.n_heads * hd          # queries
                 + 2 * d * cfg.n_kv_heads * hd  # keys and values
                 + cfg.n_heads * hd * d)        # output projection
    feed_forward = 2 * d * cfg.d_ff
    total = attention + feed_forward
    return {"attention_per_layer": attention, "feed_forward_per_layer": feed_forward,
            "feed_forward_share": feed_forward / total}


def main() -> None:
    cfg = Config(vocab_size=len(WORDS), d_model=128, n_layers=4,
                 n_heads=4, n_kv_heads=4, d_ff=512, max_seq=64)
    model = build(cfg, seed=0)
    tokens = np.array([INDEX[w] for w in PROMPT])

    trace: dict = {}
    logits = forward(model, tokens, trace=trace)

    # The next token is chosen from the scores at the LAST position.
    last = logits[-1]
    probs = softmax(last)
    order = np.argsort(last)[::-1][:TOP_K]

    layers = [
        {
            "layer": i + 1,
            # One head's weights, printed in full in the chapter.
            "attention_head_1": l["attention_weights"][0].tolist(),
            "n_heads": int(l["attention_weights"].shape[0]),
            # How much each stage moves the running representation.
            "norm_after_attention": float(np.linalg.norm(l["after_attention"][-1])),
            "norm_after_feed_forward": float(np.linalg.norm(l["after_feed_forward"][-1])),
        }
        for i, l in enumerate(trace["layers"])
    ]

    cache = KVCache(cfg, max_seq=len(tokens) + 1)
    bytes_per_token_kv = cache.bytes_per_token()

    payload = {
        "vocabulary": {"size": len(WORDS), "words": WORDS},
        "prompt": {"words": PROMPT, "tokens": tokens.tolist()},
        "model": {
            "params": model.n_params,
            "config": {k: getattr(cfg, k) for k in
                       ("vocab_size", "d_model", "n_layers", "n_heads",
                        "n_kv_heads", "d_ff")},
            "head_dim": cfg.head_dim,
            "weight_bytes": model.n_params * 4,  # fp32
        },
        "shapes": {
            "tokens": list(tokens.shape),
            "embedding": list(trace["embedding"].shape),
            "attention_weights_per_layer": list(trace["layers"][0]["attention_weights"].shape),
            "keys_per_layer": list(trace["layers"][0]["keys"].shape),
            "final": list(trace["final"].shape),
            "logits": list(trace["logits"].shape),
        },
        "parameter_split": _parameter_split(cfg),
        "layers": layers,
        "next_token": {
            "top_k": [{"word": WORDS[int(i)], "token": int(i),
                       "score": float(last[i]), "probability": float(probs[i])}
                      for i in order],
            "chosen": WORDS[int(order[0])],
            "vocab_scores": int(last.size),
        },
        "causal_check": {
            "rows_sum_to_one": bool(np.allclose(
                trace["layers"][0]["attention_weights"].sum(-1), 1.0)),
            "upper_triangle_is_zero": bool(np.all(
                np.triu(trace["layers"][0]["attention_weights"], 1) == 0)),
            "note": "architectural, not learned: a token cannot attend forward",
        },
        "cost": {
            "tiny": {
                "weight_bytes_read_per_token": model.n_params * 4,
                "kv_bytes_per_token": bytes_per_token_kv,
            },
            "reference_8b": {
                "params": REF_PARAMS,
                "weight_bytes_read_per_token": REF_WEIGHT_BYTES,
                "kv_bytes_per_token": REF_KV_BYTES_PER_TOKEN,
                "flops_per_token": 2 * REF_PARAMS,
                "hbm_bytes_per_s": REF_HBM_BYTES_PER_S,
                "decode_floor_ms": REF_WEIGHT_BYTES / REF_HBM_BYTES_PER_S * 1e3,
            },
        },
    }

    path = write("results/ch02.json", payload)
    print(f"wrote {path}")
    print(f"  prompt            {' '.join(PROMPT)}  ->  {tokens.tolist()}")
    print(f"  params            {model.n_params:,} over a {len(WORDS)}-word vocabulary")
    print(f"  causal structure  rows sum to 1: "
          f"{payload['causal_check']['rows_sum_to_one']}, "
          f"no forward attention: {payload['causal_check']['upper_triangle_is_zero']}")
    print(f"  next token        {payload['next_token']['chosen']!r} "
          f"(p={payload['next_token']['top_k'][0]['probability']:.3f} "
          f"of {len(WORDS)} choices)")


if __name__ == "__main__":
    main()
