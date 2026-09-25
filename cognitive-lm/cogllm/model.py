"""A minimal GPT, small enough to train on a laptop CPU."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.n_head, D // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, d=128, n_layer=2, n_head=4, max_len=8):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

    def n_params(self):
        # tied head counted once, position table excluded (it is tiny and fixed)
        return sum(p.numel() for p in self.parameters()) - self.pos.weight.numel()


def train_lm(model, seqs, loss_mask, epochs, batch=256, lr=3e-3, wd=0.0, seed=0, log=None):
    """Next-token training. seqs: LongTensor (N, T); loss_mask: (N, T-1) bool
    marking which target positions count (lets IDK examples train only the
    answer slot, so we never teach the model a fake name)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    steps = epochs * math.ceil(len(seqs) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(seqs), generator=g)
        tot = 0.0
        for i in range(0, len(seqs), batch):
            j = perm[i:i + batch]
            x, m = seqs[j], loss_mask[j]
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
            loss = (loss * m.reshape(-1)).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(seqs):.3f}")
    model.eval()
    return model
