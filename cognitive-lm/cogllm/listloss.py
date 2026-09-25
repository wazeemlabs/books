"""List-decoding objective: only ask the weights to get the answer into the top N.

With a checksum doing the final pick, the weights don't need the right value
at rank 1, only somewhere in the first N. In bits, that is log2(V/N) instead
of log2(V) per fact. This loss stops pushing a fact once its true value
beats the N-th best wrong value by a margin, leaving capacity for others.
"""

import math

import torch
import torch.nn.functional as F

from .world import N_SPECIAL, World


def train_lm_list(model, seqs, world: World, N=5, epochs=30, batch=256, lr=3e-3, margin=1.0, tau=0.5,
                  ce_weight=0.1, seed=0, log=None):
    V, R = world.cfg.n_values, world.cfg.n_relations
    rel_base = N_SPECIAL + world.cfg.n_first + world.cfg.n_last
    val_base = rel_base + R
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    steps = epochs * math.ceil(len(seqs) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    ar = torch.arange(V)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(seqs), generator=g)
        tot = 0.0
        for i in range(0, len(seqs), batch):
            x = seqs[perm[i:i + batch]]
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            # ordinary next-token loss everywhere except the answer slot (index 3)
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
            other = torch.cat([ce[:, :3], ce[:, 4:]], 1).mean()
            r = x[:, 3] - rel_base
            cand = val_base + r[:, None] * V + ar[None, :]
            s = logits[:, 3].gather(1, cand)
            true = tgt[:, 3] - val_base - r * V
            s_true = s.gather(1, true[:, None]).squeeze(1)
            wrong = s.scatter(1, true[:, None], float("-inf"))
            s_n = wrong.topk(N, dim=1).values[:, -1]
            hinge = tau * F.softplus((s_n + margin - s_true) / tau)
            loss = other + hinge.mean() + ce_weight * F.cross_entropy(s, true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(seqs):.3f}")
    model.eval()
    return model


def train_lm_gated(model, seqs, world, slots, Ns, epochs=30, batch=256, lr=3e-3, floor=0.1, seed=0, log=None):
    """Rank-gated cross-entropy: ordinary CE, but an answer token that is
    already inside the top N gets its loss scaled down to `floor`. Easy,
    frequent facts stop soaking up capacity once the checksum can already
    pick them out of the list; that capacity goes to the rest."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    steps = epochs * math.ceil(len(seqs) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(seqs), generator=g)
        tot = 0.0
        for i in range(0, len(seqs), batch):
            x = seqs[perm[i:i + batch]]
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
            wts = torch.ones_like(ce)
            for j, N in zip(slots, Ns):
                with torch.no_grad():
                    lj = logits[:, j]
                    rank = (lj > lj.gather(1, tgt[:, j:j + 1])).sum(1)
                    wts[:, j] = torch.where(rank < N, torch.full_like(ce[:, j], floor), torch.ones_like(ce[:, j]))
            loss = (ce * wts).sum() / wts.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(seqs):.3f}")
    model.eval()
    return model


def train_lm_list_mt(model, seqs, world, Ns=(3, 2), epochs=30, batch=256, lr=3e-3, margin=1.0, tau=0.5,
                     ce_weight=0.1, seed=0, log=None):
    """Multi-token answers: a top-N_i hinge on each answer sub-token (teacher
    forced), so the joint top-(N_1 x N_2) list tends to hold the answer."""
    P = world.cfg.value_pool
    lo = world.sub_tok(0)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.98))
    steps = epochs * math.ceil(len(seqs) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    slots = [3 + i for i in range(len(Ns))]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(seqs), generator=g)
        tot = 0.0
        for i in range(0, len(seqs), batch):
            x = seqs[perm[i:i + batch]]
            logits = model(x[:, :-1])
            tgt = x[:, 1:]
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none").view(tgt.shape)
            keep = [j for j in range(tgt.shape[1]) if j not in slots]
            loss = ce[:, keep].mean()
            for j, N in zip(slots, Ns):
                s = logits[:, j, lo:lo + P]
                true = tgt[:, j] - lo
                s_true = s.gather(1, true[:, None]).squeeze(1)
                s_n = s.scatter(1, true[:, None], float("-inf")).topk(N, dim=1).values[:, -1]
                loss = loss + (tau * F.softplus((s_n + margin - s_true) / tau)).mean() + ce_weight * F.cross_entropy(s, true)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(x)
        if log and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            log(f"  epoch {ep + 1}/{epochs} loss {tot / len(seqs):.3f}")
    model.eval()
    return model
