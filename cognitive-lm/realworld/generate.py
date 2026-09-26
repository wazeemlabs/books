"""The weights as a list decoder: OLMo-2 answers each question greedily and
with a beam of N candidates, few-shot, stopping at the end of the line.

Results are cached per question (one JSON line each) so generation is never
repeated.
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "allenai/OLMo-2-0425-1B"


def few_shot_prefix(exemplars):
    return "".join(f"Q: {q}\nA: {a}\n\n" for q, a in exemplars)


class Generator:
    def __init__(self, model=MODEL, device=None, dtype=torch.bfloat16):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        # SDPA attention on MPS returns NaN logits for every left-padded prompt in a
        # batch (torch 2.8, transformers 4.57, any dtype); eager attention is correct
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, attn_implementation="eager").to(self.device).eval()
        # an answer ends at the end of its line; every token containing a newline
        # stops generation, so beams are ranked on the answer alone
        self.stop_ids = sorted({self.tok.eos_token_id} | {i for t, i in self.tok.get_vocab().items()
                                                         if "\n" in self.tok.convert_tokens_to_string([t])})

    def _answer(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True).split("\n")[0].strip()

    @torch.no_grad()
    def run(self, prompts, n_beams=10, max_new_tokens=16):
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.device)
        L = enc["input_ids"].shape[1]
        g = self.model.generate(**enc, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                                eos_token_id=self.stop_ids, pad_token_id=self.tok.pad_token_id,
                                output_scores=True, return_dict_in_generate=True)
        # per-token log-probabilities of the greedy answer (0 after it stopped)
        lp = torch.stack([torch.log_softmax(s.float(), -1) for s in g.scores], 1)
        if torch.isnan(lp).any():
            raise RuntimeError("NaN logits in generation; the attention kernel is mishandling padding")
        new = g.sequences[:, L:]
        tok_lp = lp.gather(2, new[:, :lp.shape[1], None]).squeeze(2)
        alive = torch.ones_like(new, dtype=torch.bool)
        for t in range(1, new.shape[1]):
            alive[:, t] = alive[:, t - 1] & ~torch.isin(new[:, t - 1], torch.tensor(self.stop_ids, device=new.device))
        b = self.model.generate(**enc, do_sample=False, num_beams=n_beams, num_return_sequences=n_beams,
                                max_new_tokens=max_new_tokens, eos_token_id=self.stop_ids,
                                pad_token_id=self.tok.pad_token_id, length_penalty=0.0,
                                output_scores=True, return_dict_in_generate=True)
        out = []
        for i in range(len(prompts)):
            keep = alive[i] & ~torch.isin(new[i], torch.tensor(self.stop_ids, device=new.device))
            beams, seen = [], set()
            for j in range(n_beams):
                text = self._answer(b.sequences[i * n_beams + j, L:])
                if text and text not in seen:  # beams that differ only after the answer collapse
                    seen.add(text)
                    beams.append([text, float(b.sequences_scores[i * n_beams + j])])
            out.append({"greedy": self._answer(new[i]),
                        "greedy_logprob": float(tok_lp[i][keep].sum()),
                        "greedy_first_prob": float(tok_lp[i][0].exp()),
                        "beams": beams})
        return out


def generate_cached(gen, items, path, prefix, batch=8, **kw):
    """items: (id, question). Returns {id: result}, generating only what is not cached."""
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                done[r["id"]] = r
    todo = [(i, q) for i, q in items if i not in done]
    for s in range(0, len(todo), batch):
        chunk = todo[s:s + batch]
        res = gen.run([f"{prefix}Q: {q}\nA:" for _, q in chunk], **kw)
        with open(path, "a") as f:
            for (i, _), r in zip(chunk, res):
                r["id"] = i
                done[i] = r
                f.write(json.dumps(r) + "\n")
        print(f"generated {min(s + batch, len(todo))}/{len(todo)}", flush=True)
    return done
