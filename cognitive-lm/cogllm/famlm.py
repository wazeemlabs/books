"""FamLM: a transformer with a familiarity sense.

A hashed count table (count-min sketch, Engram-style multi-order n-gram
hashing) records, for every n-gram the model reads:
    ctx counts   how often this context has been read        (familiarity)
    next counts  how often each next token followed it       (recognition)
The counts are bucketed and fed back into the network:
    - a familiarity embedding added to the residual stream at each position
    - a learned logit bias per candidate token, from (context bucket, next bucket)

Training is PREQUENTIAL: at every step the model sees the counts as they were
BEFORE the current batch was read, and only then are the counts updated.
So during pretraining the model experiences reading things for the first
time, the second time, the tenth time, and it learns from that alone what
"never seen" means for its own predictions. No IDK labels, no extractor.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block

_C1 = 0x7FB5D329728EA185
_C2 = 0x81DADEF4BC2DD44D - (1 << 64)  # as signed int64
N_BUCKETS = 9  # 0,1,2,3-4,5-8,9-16,17-64,65+, n/a
_EDGES = torch.tensor([1, 2, 3, 5, 9, 17, 65])


def _mix(h):
    h = h ^ (h >> 31)
    h = h * _C1
    h = h ^ (h >> 27)
    h = h * _C2
    return h ^ (h >> 33)


def bucket(c):
    return torch.bucketize(c, _EDGES, right=True)


class CountSketch:
    """Multi-order hashed n-gram counters. `orders` are n-gram lengths."""

    def __init__(self, orders=(2, 3, 4, 5, 6), log2_width=21, depth=2, counter_bits=8):
        self.orders, self.W, self.depth = list(orders), 1 << log2_width, depth
        self.cap = (1 << counter_bits) - 1
        self.counter_bits = counter_bits
        O = len(self.orders)
        self.ctx = torch.zeros(O, depth, self.W, dtype=torch.int32)
        self.nxt = torch.zeros(O, depth, self.W, dtype=torch.int32)
        self.salts = torch.tensor([[(o * 131 + d * 7 + 1) * 0x9E3779B97F4A7C15 % (1 << 62) for d in range(depth)]
                                   for o in range(O)], dtype=torch.int64)

    def n_bits(self):
        return 2 * self.ctx.numel() * self.counter_bits

    def _ngram_hash(self, x):
        """x (B, T) -> (O, B, T) hash of the n-gram ENDING at each position; -1 where too short."""
        B, T = x.shape
        out = torch.full((len(self.orders), B, T), -1, dtype=torch.int64)
        x = x.to(torch.int64)
        for oi, n in enumerate(self.orders):
            if n > T:
                continue
            h = torch.zeros(B, T - n + 1, dtype=torch.int64)
            for i in range(n):
                h = _mix(h * 1000003 + x[:, i:T - n + 1 + i] + 1)
            out[oi, :, n - 1:] = h
        return out

    def _idx(self, h, oi):
        """h (...) int64 -> (depth, ...) table indices for order oi."""
        return torch.stack([_mix(h ^ self.salts[oi, d]) & (self.W - 1) for d in range(self.depth)])

    def _lookup(self, table, h, oi):
        idx = self._idx(h, oi)
        vals = torch.stack([table[oi, d][idx[d]] for d in range(self.depth)])
        return vals.min(0).values

    def features(self, x):
        """ctx buckets (B, T, O) and the n-gram hashes (O, B, T) for later next-token lookups."""
        H = self._ngram_hash(x)
        O = len(self.orders)
        ctxb = torch.full((x.shape[0], x.shape[1], O), N_BUCKETS - 1, dtype=torch.long)
        for oi in range(O):
            ok = H[oi] >= 0
            if ok.any():
                ctxb[..., oi][ok] = bucket(self._lookup(self.ctx, H[oi][ok], oi))
        return ctxb, H

    def next_buckets(self, H, cand):
        """cand (B, T, K) token ids -> (B, T, O, K) buckets of count(n-gram, candidate)."""
        B, T, K = cand.shape
        O = len(self.orders)
        out = torch.full((B, T, O, K), N_BUCKETS - 1, dtype=torch.long)
        for oi in range(O):
            ok = H[oi] >= 0
            if not ok.any():
                continue
            hc = _mix(H[oi][ok][:, None] * 1000033 + cand[ok].to(torch.int64) + 7)
            out[:, :, oi, :][ok] = bucket(self._lookup(self.nxt, hc, oi))
        return out

    def add(self, x):
        """Count every n-gram in x, and every (n-gram, next token) pair."""
        H = self._ngram_hash(x)
        for oi in range(len(self.orders)):
            ok = H[oi] >= 0
            hv = H[oi][ok]
            for d, ix in enumerate(self._idx(hv, oi)):
                self.ctx[oi, d].index_add_(0, ix, torch.ones_like(ix, dtype=torch.int32))
            nxt_ok = ok.clone()
            nxt_ok[:, -1] = False
            hn = H[oi][nxt_ok]
            tok = x[:, 1:].to(torch.int64)[nxt_ok[:, :-1]]
            hc = _mix(hn * 1000033 + tok + 7)
            for d, ix in enumerate(self._idx(hc, oi)):
                self.nxt[oi, d].index_add_(0, ix, torch.ones_like(ix, dtype=torch.int32))
        self.ctx.clamp_(max=self.cap)
        self.nxt.clamp_(max=self.cap)


class FamGPT(nn.Module):
    """GPT + familiarity embedding + recognition logit bias."""

    def __init__(self, vocab, n_orders, d=64, n_layer=2, n_head=4, max_len=8, use_ctx=True, use_next=True):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.use_ctx, self.use_next = use_ctx, use_next
        self.fam = nn.Embedding(n_orders * N_BUCKETS, d)
        # recognition bias: order x context bucket x next bucket, plus a gate from the hidden state
        self.rec = nn.Parameter(torch.zeros(n_orders, N_BUCKETS, N_BUCKETS))
        self.rec_gate = nn.Linear(d, n_orders)
        self.register_buffer("cand", torch.arange(vocab))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, sketch=None, cand=None, target=None, K=16):
        """cand: explicit (V',) candidate ids to score with the sketch (evaluation);
        otherwise the top-K of the base logits (+ the target token when training)."""
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T))
        use = sketch is not None and (self.use_ctx or self.use_next)
        if use:
            ctxb, H = sketch.features(idx)
            if self.use_ctx:
                off = torch.arange(ctxb.shape[-1]) * N_BUCKETS
                x = x + self.fam(ctxb + off).sum(-2)
        for b in self.blocks:
            x = b(x)
        h = self.ln(x)
        logits = self.head(h)
        if use and self.use_next:
            B = idx.shape[0]
            if cand is not None:
                c = cand[None, None, :].expand(B, T, len(cand))
                dup = torch.zeros_like(c, dtype=torch.bool)
            else:
                c = logits.detach().topk(K - 1, dim=-1).indices
                t = target[..., None] if target is not None else logits.detach().topk(K, dim=-1).indices[..., -1:]
                dup = (c == t).any(-1, keepdim=True)
                c = torch.cat([c, t], -1)
                dup = torch.cat([torch.zeros_like(c[..., :-1], dtype=torch.bool), dup], -1)
            nb = sketch.next_buckets(H, c)                            # (B, T, O, K)
            O = nb.shape[2]
            ob = torch.arange(O)[None, None, :, None].expand_as(nb)
            cb = ctxb[..., None].expand_as(nb)
            gate = torch.sigmoid(self.rec_gate(h))[..., None]          # (B, T, O, 1)
            bias = (gate * self.rec[ob, cb, nb]).sum(2).masked_fill(dup, 0.0)
            logits = logits.scatter_add(-1, c, bias)
        return logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()


def train_prequential(model, sketch, seqs, epochs=1, batch=256, lr=3e-3, seed=0, update=True, shuffle_after_first=True,
                      extra=None, log=None):
    """Prequential training: features from the sketch BEFORE each batch is counted.

    extra: optional (seqs, mask) of IDK examples mixed in (never counted in the sketch)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    n = len(seqs)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    for ep in range(epochs):
        order = torch.randperm(n, generator=g) if (ep > 0 and shuffle_after_first) else torch.arange(n)
        tot = 0.0
        for i in range(0, n, batch):
            x = seqs[order[i:i + batch]]
            mask = torch.ones(x.shape[0], x.shape[1] - 1, dtype=torch.bool)
            if extra is not None:
                j = torch.randint(0, len(extra[0]), (max(1, x.shape[0] // 4),), generator=g)
                x = torch.cat([x, extra[0][j]])
                mask = torch.cat([mask, extra[1][j]])
            inp = x[:, :-1]
            logits = model(inp, sketch, target=x[:, 1:])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
            loss = (loss * mask.reshape(-1)).sum() / mask.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if update and sketch is not None:
                sketch.add(seqs[order[i:i + batch]])  # IDK examples are never counted
            tot += loss.item() * len(x)
        if log:
            log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.3f}")
    model.eval()
    return model
